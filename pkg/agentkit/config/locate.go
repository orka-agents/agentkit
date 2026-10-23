package config

import (
	"errors"
	"strconv"
	"strings"

	"github.com/goccy/go-yaml/ast"
	"github.com/goccy/go-yaml/parser"
)

// Problem is one finding reported by Validate. Path is a YAML path such as
// `$.tools[1].urlEnv` naming the place in the Agentkitfile the problem is
// about; it is empty when no single place applies. Error returns Message
// alone, so callers without the source file see the same text as before.
type Problem struct {
	Path    string
	Message string
}

func (p *Problem) Error() string { return p.Message }

// Span is where a problem is in an Agentkitfile: 1-based lines and columns,
// with EndLine covering the whole located node (for example a full tool
// entry).
type Span struct {
	StartLine   int
	StartColumn int
	EndLine     int
	EndColumn   int
}

// LocatedProblem is a Problem together with its Span in the source file.
type LocatedProblem struct {
	Problem
	Span
}

// LocateProblems resolves every problem in err, as returned by Validate, to
// its place in data. A path that is not in the file, such as a required field
// that is missing, resolves to its nearest parent that is; a path with no
// part in the file resolves to line 1. Errors that are not a *Problem are
// returned with an empty Path and also point at line 1.
func LocateProblems(err error, data []byte) []LocatedProblem {
	leaves := flattenJoined(err)
	if len(leaves) == 0 {
		return nil
	}
	file, parseErr := parser.ParseBytes(data, 0)
	located := make([]LocatedProblem, 0, len(leaves))
	for _, leaf := range leaves {
		var p *Problem
		if !errors.As(leaf, &p) {
			p = &Problem{Message: leaf.Error()}
		}
		span := firstLine
		if parseErr == nil {
			span = locatePath(file, p.Path)
		}
		located = append(located, LocatedProblem{Problem: *p, Span: span})
	}
	return located
}

var firstLine = Span{StartLine: 1, StartColumn: 1, EndLine: 1, EndColumn: 1}

// flattenJoined returns the individual errors of an errors.Join tree.
func flattenJoined(err error) []error {
	if err == nil {
		return nil
	}
	joined, ok := err.(interface{ Unwrap() []error })
	if !ok {
		return []error{err}
	}
	var out []error
	for _, e := range joined.Unwrap() {
		out = append(out, flattenJoined(e)...)
	}
	return out
}

// pathSegment is one step of a YAML path: a mapping key or a sequence index.
type pathSegment struct {
	key     string
	index   int
	isIndex bool
}

// parseYAMLPath splits a path of the form `$.a.b[2].'quoted.key'` into
// segments. It accepts the subset of YAML path syntax Validate produces.
func parseYAMLPath(path string) ([]pathSegment, bool) {
	if !strings.HasPrefix(path, "$") {
		return nil, false
	}
	rest := path[1:]
	var segs []pathSegment
	for rest != "" {
		switch rest[0] {
		case '.':
			rest = rest[1:]
			if rest != "" && (rest[0] == '\'' || rest[0] == '"') {
				quote := rest[0]
				end := 1
				for end < len(rest) && rest[end] != quote {
					if quote == '"' && rest[end] == '\\' {
						end++
					}
					end++
				}
				if end >= len(rest) {
					return nil, false
				}
				key := rest[1:end]
				if quote == '"' {
					var err error
					key, err = strconv.Unquote(rest[:end+1])
					if err != nil {
						return nil, false
					}
				}
				segs = append(segs, pathSegment{key: key})
				rest = rest[end+1:]
				continue
			}
			end := strings.IndexAny(rest, ".[")
			if end < 0 {
				end = len(rest)
			}
			if end == 0 {
				return nil, false
			}
			segs = append(segs, pathSegment{key: rest[:end]})
			rest = rest[end:]
		case '[':
			end := strings.IndexByte(rest, ']')
			if end < 0 {
				return nil, false
			}
			index, err := strconv.Atoi(rest[1:end])
			if err != nil || index < 0 {
				return nil, false
			}
			segs = append(segs, pathSegment{index: index, isIndex: true})
			rest = rest[end+1:]
		default:
			return nil, false
		}
	}
	return segs, true
}

// yamlPathKey appends a mapping key to a YAML path, quoting keys that
// contain path syntax (for example label keys such as "example.com/team").
func yamlPathKey(base, key string) string {
	if key != "" && !strings.ContainsAny(key, ".[]'\" ") {
		return base + "." + key
	}
	return base + "." + strconv.Quote(key)
}

// locatePath walks file along path and returns the span of the deepest node
// it reaches. A matched mapping key is located at the key, so a missing
// child field points at its parent's line (for example `model:`).
func locatePath(file *ast.File, path string) Span {
	if file == nil || len(file.Docs) == 0 || file.Docs[0].Body == nil {
		return firstLine
	}
	segs, ok := parseYAMLPath(path)
	if !ok || len(segs) == 0 {
		return firstLine
	}

	current := unwrapNode(file.Docs[0].Body)
	var anchor ast.Node
	var anchorValue ast.Node
	for _, seg := range segs {
		next, keyNode := childNode(current, seg)
		if next == nil {
			break
		}
		anchor, anchorValue = keyNode, next
		current = unwrapNode(next)
	}
	if anchor == nil {
		return firstLine
	}
	return spanOf(anchor, anchorValue)
}

// childNode returns the value of seg within node, and the node to report the
// position of: the key for a mapping entry, the entry itself for a sequence
// item.
func childNode(node ast.Node, seg pathSegment) (value ast.Node, at ast.Node) {
	switch n := node.(type) {
	case *ast.MappingNode:
		if seg.isIndex {
			return nil, nil
		}
		for _, mv := range n.Values {
			if mappingKey(mv) == seg.key {
				return mv.Value, mv.Key
			}
		}
	case *ast.MappingValueNode:
		if !seg.isIndex && mappingKey(n) == seg.key {
			return n.Value, n.Key
		}
	case *ast.SequenceNode:
		if seg.isIndex && seg.index < len(n.Values) {
			item := n.Values[seg.index]
			return item, item
		}
	}
	return nil, nil
}

func mappingKey(mv *ast.MappingValueNode) string {
	if mv == nil || mv.Key == nil || mv.Key.GetToken() == nil {
		return ""
	}
	return mv.Key.GetToken().Value
}

// unwrapNode looks through anchors and tags to the node they decorate.
func unwrapNode(node ast.Node) ast.Node {
	for {
		switch n := node.(type) {
		case *ast.AnchorNode:
			node = n.Value
		case *ast.TagNode:
			node = n.Value
		default:
			return node
		}
	}
}

// spanOf starts at the reported node and extends to the last line of the
// value it names, so a tool entry covers all of its lines.
func spanOf(at ast.Node, value ast.Node) Span {
	tok := at.GetToken()
	if tok == nil || tok.Position == nil {
		return firstLine
	}
	span := Span{
		StartLine:   tok.Position.Line,
		StartColumn: tok.Position.Column,
		EndLine:     tok.Position.Line,
		EndColumn:   tok.Position.Column + len(tok.Value),
	}
	for _, n := range []ast.Node{at, value} {
		if n == nil {
			continue
		}
		ast.Walk(&endVisitor{span: &span}, n)
	}
	return span
}

type endVisitor struct{ span *Span }

func (v *endVisitor) Visit(n ast.Node) ast.Visitor {
	if n == nil || n.Type() == ast.CommentType {
		return v
	}
	tok := n.GetToken()
	if tok == nil || tok.Position == nil {
		return v
	}
	line := tok.Position.Line
	endColumn := tok.Position.Column + len(tok.Value)
	if line > v.span.EndLine || (line == v.span.EndLine && endColumn > v.span.EndColumn) {
		v.span.EndLine = line
		v.span.EndColumn = endColumn
	}
	return v
}
