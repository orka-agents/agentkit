package render

import (
	"encoding/json"
	"fmt"
	"math/big"
	"strconv"
	"strings"

	"github.com/goccy/go-yaml/ast"
	"github.com/goccy/go-yaml/lexer"
	"github.com/goccy/go-yaml/parser"
	"github.com/goccy/go-yaml/token"
)

// maxFloatExponent caps float literal exponents; an integral value then
// expands to at most a few thousand digits.
const maxFloatExponent = 4096

// decodeJSONDocuments decodes every non-null document in a YAML or JSON stream
// into JSON-shaped values: map[string]any, []any, string, bool, nil, and
// json.Number. Numbers keep their source text so a high-precision value is
// never silently rounded before it is validated, digested, or re-rendered.
func decodeJSONDocuments(data []byte) ([]any, error) {
	var docs []any
	for _, chunk := range splitYAMLDocuments(data) {
		file, err := parser.ParseBytes(chunk, 0)
		if err != nil {
			return nil, err
		}
		for _, doc := range file.Docs {
			if doc == nil || doc.Body == nil {
				continue
			}
			value, err := (&jsonDecoder{anchors: map[string]any{}}).value(doc.Body)
			if err != nil {
				return nil, err
			}
			if value != nil {
				docs = append(docs, value)
			}
		}
	}
	return docs, nil
}

// splitYAMLDocuments cuts a stream at each `---` marker line. goccy's parser
// drops every document after an empty one (`---\n---\n`, common in rendered
// Helm output), so each document is parsed on its own.
func splitYAMLDocuments(data []byte) [][]byte {
	lineStarts := []int{0}
	for i, b := range data {
		if b == '\n' {
			lineStarts = append(lineStarts, i+1)
		}
	}
	var chunks [][]byte
	start := 0
	for _, tk := range lexer.Tokenize(string(data)) {
		if tk.Type != token.DocumentHeaderType || tk.Position.Line < 1 || tk.Position.Line > len(lineStarts) {
			continue
		}
		if offset := lineStarts[tk.Position.Line-1]; offset > start {
			chunks = append(chunks, data[start:offset])
			start = offset
		}
	}
	return append(chunks, data[start:])
}

type jsonDecoder struct {
	anchors map[string]any
}

func (d *jsonDecoder) value(node ast.Node) (any, error) {
	switch n := node.(type) {
	case *ast.NullNode, *ast.CommentGroupNode:
		return nil, nil
	case *ast.StringNode:
		// goccy leaves integers beyond uint64 as plain strings; keep them numbers.
		if n.Token != nil && n.Token.Type == token.StringType && isDecimalInteger(n.Value) {
			if number := strings.TrimPrefix(n.Value, "+"); json.Valid([]byte(number)) {
				return json.Number(number), nil
			}
		}
		return n.Value, nil
	case *ast.LiteralNode:
		return n.Value.Value, nil
	case *ast.BoolNode:
		return n.Value, nil
	case *ast.IntegerNode:
		return json.Number(fmt.Sprint(n.Value)), nil
	case *ast.FloatNode:
		return floatNumber(n.Token.Value)
	case *ast.InfinityNode, *ast.NanNode:
		return nil, fmt.Errorf("YAML number %q must be finite", node.GetToken().Value)
	case *ast.TagNode:
		if token.ReservedTagKeyword(n.Start.Value) != token.StringTag {
			return nil, fmt.Errorf("YAML tag %s is not supported", n.Start.Value)
		}
		if s, ok := n.Value.(*ast.StringNode); ok {
			return s.Value, nil
		}
		if _, ok := n.Value.(ast.ScalarNode); !ok {
			return nil, fmt.Errorf("YAML tag %s requires a scalar value", n.Start.Value)
		}
		return n.Value.GetToken().Value, nil
	case *ast.AnchorNode:
		value, err := d.value(n.Value)
		if err != nil {
			return nil, err
		}
		d.anchors[n.Name.GetToken().Value] = value
		return value, nil
	case *ast.AliasNode:
		name := n.Value.GetToken().Value
		value, ok := d.anchors[name]
		if !ok {
			return nil, fmt.Errorf("YAML alias %q has no anchor", name)
		}
		return value, nil
	case *ast.MappingNode:
		return d.mapping(n.Values)
	case *ast.MappingValueNode:
		return d.mapping([]*ast.MappingValueNode{n})
	case *ast.SequenceNode:
		out := make([]any, 0, len(n.Values))
		for _, item := range n.Values {
			value, err := d.value(item)
			if err != nil {
				return nil, err
			}
			out = append(out, value)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported YAML node %s", node.Type())
	}
}

func (d *jsonDecoder) mapping(entries []*ast.MappingValueNode) (map[string]any, error) {
	out := make(map[string]any, len(entries))
	var merged []map[string]any
	for _, entry := range entries {
		if entry.Key.IsMergeKey() {
			sources, err := d.mergeSources(entry.Value)
			if err != nil {
				return nil, err
			}
			merged = append(merged, sources...)
			continue
		}
		key, err := d.key(entry.Key)
		if err != nil {
			return nil, err
		}
		if _, exists := out[key]; exists {
			return nil, fmt.Errorf("YAML mapping key %q is duplicated", key)
		}
		value, err := d.value(entry.Value)
		if err != nil {
			return nil, err
		}
		out[key] = value
	}
	// Explicit keys win over merged ones, and earlier merge sources win over later ones.
	for _, source := range merged {
		for key, value := range source {
			if _, exists := out[key]; !exists {
				out[key] = value
			}
		}
	}
	return out, nil
}

func (d *jsonDecoder) mergeSources(node ast.Node) ([]map[string]any, error) {
	value, err := d.value(node)
	if err != nil {
		return nil, err
	}
	switch typed := value.(type) {
	case map[string]any:
		return []map[string]any{typed}, nil
	case []any:
		out := make([]map[string]any, 0, len(typed))
		for _, item := range typed {
			source, ok := item.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("YAML merge key value must be a mapping or a list of mappings")
			}
			out = append(out, source)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("YAML merge key value must be a mapping or a list of mappings")
	}
}

func (d *jsonDecoder) key(node ast.MapKeyNode) (string, error) {
	var keyNode ast.Node = node
	if explicit, ok := node.(*ast.MappingKeyNode); ok {
		keyNode = explicit.Value
	}
	value, err := d.value(keyNode)
	if err != nil {
		return "", err
	}
	switch typed := value.(type) {
	case string:
		return typed, nil
	case json.Number:
		return typed.String(), nil
	case bool:
		return strconv.FormatBool(typed), nil
	case nil:
		return "null", nil
	default:
		return "", fmt.Errorf("YAML mapping keys must be scalars")
	}
}

func isDecimalInteger(value string) bool {
	digits := strings.TrimLeft(value, "+-")
	if len(value)-len(digits) > 1 || digits == "" {
		return false
	}
	for _, r := range digits {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// floatNumber rewrites a YAML float literal (`.5`, `1.`, `+1_000.0`) as the
// equivalent JSON number text without going through float64. Integral values
// become exact integers, so `9007199254740993.0` stays 9007199254740993.
func floatNumber(raw string) (json.Number, error) {
	text := strings.ReplaceAll(raw, "_", "")
	sign := ""
	if strings.HasPrefix(text, "+") || strings.HasPrefix(text, "-") {
		if text[0] == '-' {
			sign = "-"
		}
		text = text[1:]
	}
	mantissa, exponent, hasExponent := strings.Cut(strings.ToLower(text), "e")
	if strings.HasPrefix(mantissa, ".") {
		mantissa = "0" + mantissa
	}
	if strings.HasSuffix(mantissa, ".") {
		mantissa += "0"
	}
	text = sign + mantissa
	if hasExponent {
		// Bound the exponent before exact arithmetic expands it into digits.
		if value, err := strconv.Atoi(exponent); err != nil || value > maxFloatExponent || value < -maxFloatExponent {
			return "", fmt.Errorf("YAML number %q is out of range", raw)
		}
		text += "e" + exponent
	}
	if !json.Valid([]byte(text)) {
		return "", fmt.Errorf("YAML number %q is not a valid JSON number", raw)
	}
	if number, ok := new(big.Rat).SetString(text); ok && number.IsInt() && number.Sign() != 0 {
		return json.Number(number.Num().String()), nil
	}
	return json.Number(text), nil
}
