// Package abi renders the frozen agent.yaml writer contract shared by the Go
// frontend and Python runtime readers.
package abi

import (
	"bytes"
	"encoding/json"
	"math"
	"math/big"
	"reflect"
	"strconv"
	"strings"

	"github.com/goccy/go-yaml"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
	"github.com/orka-agents/agentkit/pkg/agentkit/effective"
)

// Version is the schema version of the baked agent.yaml (docs/agent-abi.md).
// It MUST equal agentkit-serve's config.ABI_VERSION ("v0").
const Version = "v0"

// Path is where the rendered agent.yaml is baked into the agent image.
const Path = "/agent/agent.yaml"

const yamlNegativeZero = "-0.0"

// The following types are the WRITER half of the frozen agent.yaml ABI
// (docs/agent-abi.md). agentkit-serve reads this file with pydantic
// extra="forbid", so these structs MUST emit EXACTLY the keys documented there.
// Field order here is the emit order.

type yamlNumber string

func (n yamlNumber) MarshalYAML() ([]byte, error) {
	return []byte(n), nil
}

// yamlString centralizes scalar rendering for every string in the ABI. Always
// emitting a quoted scalar prevents YAML syntax and implicit type resolution
// from changing string data across the Go and Python readers.
type yamlString string

func (s yamlString) MarshalYAML() ([]byte, error) {
	return []byte(strconv.Quote(string(s))), nil
}

func yamlStrings(values []string) []yamlString {
	if len(values) == 0 {
		return nil
	}
	out := make([]yamlString, len(values))
	for i, value := range values {
		out[i] = yamlString(value)
	}
	return out
}

func yamlFloat(value float64, bitSize int) yamlNumber {
	rendered := strconv.FormatFloat(value, 'f', -1, bitSize)
	if value == 0 && math.Signbit(value) {
		rendered = yamlNegativeZero
	}
	return yamlNumber(rendered)
}

type abiMetadata struct {
	Name yamlString `yaml:"name"`
}

type abiModel struct {
	Provider  yamlString `yaml:"provider"`
	BaseURL   yamlString `yaml:"baseURL"`
	Name      yamlString `yaml:"name"`
	APIKeyEnv yamlString `yaml:"apiKeyEnv,omitempty"`
	Auth      *abiAuth   `yaml:"auth,omitempty"`
}

type abiToolHeader struct {
	Name     yamlString `yaml:"name"`
	Value    yamlString `yaml:"value,omitempty"`
	ValueEnv yamlString `yaml:"valueEnv,omitempty"`
}

type abiAuth struct {
	Type     yamlString `yaml:"type"`
	TokenEnv yamlString `yaml:"tokenEnv,omitempty"`
	Audience yamlString `yaml:"audience,omitempty"`
}

type abiTool struct {
	Name      yamlString      `yaml:"name"`
	Type      yamlString      `yaml:"type,omitempty"`
	Transport yamlString      `yaml:"transport,omitempty"`
	Command   []yamlString    `yaml:"command,omitempty"`
	URLEnv    yamlString      `yaml:"urlEnv,omitempty"`
	Headers   []abiToolHeader `yaml:"headers,omitempty"`
	Auth      *abiAuth        `yaml:"auth,omitempty"`
	Approval  yamlString      `yaml:"approval,omitempty"`
	Env       []yamlString    `yaml:"env,omitempty"`
}

type abiBrokeredTool struct {
	Name          yamlString `yaml:"name"`
	Description   yamlString `yaml:"description"`
	BrokeredClass yamlString `yaml:"brokeredClass"`
	Parameters    any        `yaml:"parameters"`
	SchemaDigest  yamlString `yaml:"schemaDigest,omitempty"`
}

type abiEnvVar struct {
	Name     yamlString `yaml:"name"`
	Required bool       `yaml:"required,omitempty"`
}

type abiContextProvider struct {
	Name         yamlString `yaml:"name,omitempty"`
	Type         yamlString `yaml:"type"`
	Source       yamlString `yaml:"source,omitempty"`
	Path         yamlString `yaml:"path,omitempty"`
	ToolRef      yamlString `yaml:"toolRef,omitempty"`
	Index        yamlString `yaml:"index,omitempty"`
	EndpointEnv  yamlString `yaml:"endpointEnv,omitempty"`
	IndexEnv     yamlString `yaml:"indexEnv,omitempty"`
	StoreNameEnv yamlString `yaml:"storeNameEnv,omitempty"`
	Auth         *abiAuth   `yaml:"auth,omitempty"`
}

type abiContext struct {
	Providers []abiContextProvider `yaml:"providers,omitempty"`
}

type abiObservability struct {
	OTel struct {
		EndpointEnv yamlString `yaml:"endpointEnv,omitempty"`
	} `yaml:"otel,omitempty"`
	Logs struct {
		LevelEnv yamlString `yaml:"levelEnv,omitempty"`
	} `yaml:"logs,omitempty"`
}

type abiExpose struct {
	OpenAI bool `yaml:"openai"`
	Port   int  `yaml:"port"`
}

type abiAgent struct {
	ABIVersion    yamlString        `yaml:"abiVersion"`
	Metadata      abiMetadata       `yaml:"metadata"`
	Model         abiModel          `yaml:"model"`
	Instructions  yamlString        `yaml:"instructions"`
	Tools         []abiTool         `yaml:"tools"`
	BrokeredTools []abiBrokeredTool `yaml:"brokeredTools,omitempty"`
	Env           []abiEnvVar       `yaml:"env,omitempty"`
	Context       *abiContext       `yaml:"context,omitempty"`
	Observability *abiObservability `yaml:"observability,omitempty"`
	Expose        abiExpose         `yaml:"expose"`
}

func expandJSONNumber(value string) string {
	lower := strings.ToLower(value)
	if json.Valid([]byte(lower)) {
		if number, ok := new(big.Rat).SetString(lower); ok && number.IsInt() {
			if number.Sign() == 0 && strings.HasPrefix(lower, "-") {
				return yamlNegativeZero
			}
			return number.Num().String()
		}
	}
	parts := strings.Split(lower, "e")
	if len(parts) != 2 {
		return value
	}
	exponent, err := strconv.Atoi(parts[1])
	if err != nil {
		return value
	}
	mantissa := parts[0]
	sign := ""
	if strings.HasPrefix(mantissa, "-") || strings.HasPrefix(mantissa, "+") {
		if mantissa[0] == '-' {
			sign = "-"
		}
		mantissa = mantissa[1:]
	}
	fracLen := 0
	if dot := strings.IndexByte(mantissa, '.'); dot >= 0 {
		fracLen = len(mantissa) - dot - 1
		mantissa = mantissa[:dot] + mantissa[dot+1:]
	}
	mantissa = strings.TrimLeft(mantissa, "0")
	if mantissa == "" {
		if sign == "-" {
			return yamlNegativeZero
		}
		return "0"
	}
	decimalPos := len(mantissa) - fracLen + exponent
	var out string
	switch {
	case decimalPos <= 0:
		out = "0." + strings.Repeat("0", -decimalPos) + mantissa
	case decimalPos >= len(mantissa):
		out = mantissa + strings.Repeat("0", decimalPos-len(mantissa))
	default:
		out = mantissa[:decimalPos] + "." + mantissa[decimalPos:]
	}
	if strings.Contains(out, ".") {
		out = strings.TrimRight(out, "0")
		out = strings.TrimRight(out, ".")
	}
	if out == "" || out == "-" {
		return "0"
	}
	return sign + out
}

func isNegativeJSONZero(value string) bool {
	if !strings.HasPrefix(value, "-") {
		return false
	}
	coefficient := value[1:]
	if exponent := strings.IndexAny(coefficient, "eE"); exponent >= 0 {
		coefficient = coefficient[:exponent]
	}
	sawZero := false
	for _, char := range coefficient {
		switch char {
		case '0':
			sawZero = true
		case '.':
		default:
			return false
		}
	}
	return sawZero
}

func yamlJSONNumber(value json.Number) yamlNumber {
	raw := value.String()
	if isNegativeJSONZero(raw) {
		return yamlNumber("-0.0")
	}
	return yamlNumber(expandJSONNumber(raw))
}

func copyMap(in map[string]any) map[any]any {
	if in == nil {
		return nil
	}
	out := make(map[any]any, len(in))
	for key, value := range in {
		out[yamlString(key)] = copyAny(value)
	}
	return out
}

// normalizeJSONValue routes brokered schemas through encoding/json, matching the
// representation used by Go validation and digesting (including []byte base64).
func normalizeJSONValue(v any) (any, error) {
	encoded, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	var normalized any
	if err := decoder.Decode(&normalized); err != nil {
		return nil, err
	}
	return copyAny(normalized), nil
}

func copyAny(v any) any {
	switch typed := v.(type) {
	case string:
		return yamlString(typed)
	case map[string]any:
		if typed == nil {
			return nil
		}
		return copyMap(typed)
	case []any:
		if typed == nil {
			return nil
		}
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = copyAny(item)
		}
		return out
	case json.Number:
		return yamlJSONNumber(typed)
	default:
		return copyReflectedJSON(typed)
	}
}

func copyReflectedJSON(value any) any {
	if value == nil {
		return nil
	}
	rv := reflect.ValueOf(value)
	for rv.Kind() == reflect.Interface || rv.Kind() == reflect.Pointer {
		if rv.IsNil() {
			return nil
		}
		rv = rv.Elem()
	}
	if number, ok := rv.Interface().(json.Number); ok {
		return yamlNumber(expandJSONNumber(number.String()))
	}
	switch rv.Kind() {
	case reflect.Map:
		if rv.IsNil() {
			return nil
		}
		if rv.Type().Key().Kind() != reflect.String {
			return value
		}
		out := make(map[any]any, rv.Len())
		iter := rv.MapRange()
		for iter.Next() {
			out[yamlString(iter.Key().String())] = copyAny(iter.Value().Interface())
		}
		return out
	case reflect.Slice:
		if rv.IsNil() {
			return nil
		}
		if rv.Type().Elem().Kind() == reflect.Uint8 {
			return value
		}
		out := make([]any, rv.Len())
		for i := 0; i < rv.Len(); i++ {
			out[i] = copyAny(rv.Index(i).Interface())
		}
		return out
	case reflect.Array:
		out := make([]any, rv.Len())
		for i := 0; i < rv.Len(); i++ {
			out[i] = copyAny(rv.Index(i).Interface())
		}
		return out
	case reflect.Float32:
		return yamlFloat(rv.Float(), 32)
	case reflect.Float64:
		return yamlFloat(rv.Float(), 64)
	default:
		return rv.Interface()
	}
}

// Render produces the baked /agent/agent.yaml from an effective Agent. The
// output is byte-compatible with agentkit-serve's strict (extra=forbid) reader.
func Render(agent effective.Agent) ([]byte, error) {
	out := abiAgent{
		ABIVersion: yamlString(Version),
		Metadata:   abiMetadata{Name: yamlString(agent.Metadata.Name)},
		Model: abiModel{
			Provider:  yamlString(agent.Model.Provider),
			BaseURL:   yamlString(agent.Model.BaseURL),
			Name:      yamlString(agent.Model.Name),
			APIKeyEnv: yamlString(agent.Model.APIKeyEnv),
		},
		Instructions: yamlString(agent.Instructions),
		Tools:        make([]abiTool, 0, len(agent.Tools)),
		Env:          make([]abiEnvVar, 0, len(agent.Env)),
		Expose:       abiExpose{OpenAI: agent.Expose.OpenAI, Port: agent.Expose.Port},
	}
	if agent.Model.Auth != nil {
		out.Model.Auth = &abiAuth{Type: yamlString(agent.Model.Auth.Type), TokenEnv: yamlString(agent.Model.Auth.TokenEnv), Audience: yamlString(agent.Model.Auth.Audience)}
	}
	for _, t := range agent.Tools {
		tool := abiTool{
			Name:      yamlString(t.Name),
			Type:      yamlString(t.Type),
			Transport: yamlString(t.Transport),
			Command:   yamlStrings(t.Command),
			URLEnv:    yamlString(t.URLEnv),
			Headers:   make([]abiToolHeader, 0, len(t.Headers)),
			Approval:  yamlString(t.Approval),
			Env:       yamlStrings(t.Env),
		}
		for _, h := range t.Headers {
			tool.Headers = append(tool.Headers, abiToolHeader{Name: yamlString(h.Name), Value: yamlString(h.Value), ValueEnv: yamlString(h.ValueEnv)})
		}
		if len(tool.Headers) == 0 {
			tool.Headers = nil
		}
		if t.Auth != nil {
			tool.Auth = &abiAuth{Type: yamlString(t.Auth.Type), TokenEnv: yamlString(t.Auth.TokenEnv), Audience: yamlString(t.Auth.Audience)}
		}
		out.Tools = append(out.Tools, tool)
	}
	brokeredTools, err := renderBrokeredTools(agent.BrokeredTools)
	if err != nil {
		return nil, err
	}
	out.BrokeredTools = brokeredTools
	for _, e := range agent.Env {
		out.Env = append(out.Env, abiEnvVar{Name: yamlString(e.Name), Required: e.Required})
	}
	if len(out.Env) == 0 {
		out.Env = nil
	}
	if len(agent.Context.Providers) > 0 {
		ctx := &abiContext{Providers: make([]abiContextProvider, 0, len(agent.Context.Providers))}
		for _, provider := range agent.Context.Providers {
			p := abiContextProvider{
				Name:         yamlString(provider.Name),
				Type:         yamlString(provider.Type),
				Source:       yamlString(provider.Source),
				Path:         yamlString(provider.Path),
				ToolRef:      yamlString(provider.ToolRef),
				Index:        yamlString(provider.Index),
				EndpointEnv:  yamlString(provider.EndpointEnv),
				IndexEnv:     yamlString(provider.IndexEnv),
				StoreNameEnv: yamlString(provider.StoreNameEnv),
			}
			if provider.Auth != nil {
				p.Auth = &abiAuth{Type: yamlString(provider.Auth.Type), TokenEnv: yamlString(provider.Auth.TokenEnv), Audience: yamlString(provider.Auth.Audience)}
			}
			ctx.Providers = append(ctx.Providers, p)
		}
		out.Context = ctx
	}
	if agent.Observability.OTel.EndpointEnv != "" || agent.Observability.Logs.LevelEnv != "" {
		obs := &abiObservability{}
		obs.OTel.EndpointEnv = yamlString(agent.Observability.OTel.EndpointEnv)
		obs.Logs.LevelEnv = yamlString(agent.Observability.Logs.LevelEnv)
		out.Observability = obs
	}

	return yaml.Marshal(out)
}

// RenderBrokeredTools renders brokeredTools entries exactly as Render bakes
// them into agent.yaml, as a fragment under a top-level brokeredTools key or,
// when bare is set, as the list alone.
func RenderBrokeredTools(tools []config.BrokeredTool, bare bool) ([]byte, error) {
	rendered, err := renderBrokeredTools(tools)
	if err != nil {
		return nil, err
	}
	if bare {
		return yaml.Marshal(rendered)
	}
	return yaml.Marshal(struct {
		BrokeredTools []abiBrokeredTool `yaml:"brokeredTools"`
	}{BrokeredTools: rendered})
}

func renderBrokeredTools(tools []config.BrokeredTool) ([]abiBrokeredTool, error) {
	var out []abiBrokeredTool
	for _, t := range tools {
		parameters, err := normalizeJSONValue(t.Parameters)
		if err != nil {
			return nil, err
		}
		out = append(out, abiBrokeredTool{
			Name:          yamlString(t.Name),
			Description:   yamlString(t.Description),
			BrokeredClass: yamlString(t.BrokeredClass),
			Parameters:    parameters,
			SchemaDigest:  yamlString(t.SchemaDigest),
		})
	}
	return out, nil
}
