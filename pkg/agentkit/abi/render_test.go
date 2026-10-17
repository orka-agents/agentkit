package abi

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/goccy/go-yaml"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
	"github.com/orka-agents/agentkit/pkg/agentkit/effective"
)

// testAPIKeyEnvName is the NAME of an env var (not a secret). Hoisted to a const
// so gosec's G101 string-literal credential heuristic does not false-positive on
// the struct literal below.
const (
	testAPIKeyEnvName        = "OPENAI_API_KEY" //nolint:gosec // G101: env var NAME, not a credential
	testInstructions         = "Be helpful and cite sources."
	testHelloBase64          = "SGVsbG8="
	testYAMLPositiveInfinity = ".inf"
)

const (
	jsonSchemaTypeKey        = "type"
	jsonSchemaDescriptionKey = "description"
	jsonSchemaDefaultKey     = "default"
	jsonSchemaEnumKey        = "enum"
	jsonSchemaObject         = "object"
	jsonSchemaString         = "string"
	jsonSchemaNumber         = "number"
	jsonSchemaPropertiesKey  = "properties"
	jsonSchemaMinimumKey     = "minimum"
)

func sampleConfig() *config.AgentConfig {
	return &config.AgentConfig{
		APIVersion: "v1alpha1",
		Kind:       "Agent",
		Metadata:   config.Metadata{Name: "acme-support"},
		Runtime:    "pydantic-ai",
		Model: config.Model{
			Provider:  "openai-compatible",
			BaseURL:   "https://api.openai.com/v1",
			Name:      "gpt-4o-mini",
			APIKeyEnv: testAPIKeyEnvName,
		},
		Tools: []config.Tool{
			{Name: "fetch", Command: []string{"uvx", "mcp-server-fetch"}, Env: []string{"FETCH_TIMEOUT"}},
		},
		Expose: config.Expose{OpenAI: true, Port: 8080},
	}
}

func sampleAgent() effective.Agent {
	return effective.FromConfig(sampleConfig(), testInstructions)
}

func TestVersionAndPath(t *testing.T) {
	if Version != "v0" {
		t.Fatalf("Version = %q, want v0", Version)
	}
	if Path != "/agent/agent.yaml" {
		t.Fatalf("Path = %q, want /agent/agent.yaml", Path)
	}
}

// TestRenderOmitsExtensionFields proves x-/X- fields accepted at load time
// (AllowFieldPrefixes) never reach the baked agent.yaml: they are dropped at
// decode because no AgentConfig field exists to hold them, so there is
// nothing for effective.FromConfig or Render to carry forward.
func TestRenderOmitsExtensionFields(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
x-owner: platform-team
metadata:
  name: extension-fields-agent
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
  x-model-note: internal
instructions: hi
tools:
  - name: fetch
    command: ["mcp-server-fetch"]
    x-tool-note: internal
expose:
  openai: true
`)
	cfg, err := config.NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	if strings.Contains(string(out), "x-") {
		t.Fatalf("expected no x- fields in rendered agent.yaml, got:\n%s", out)
	}
}

func TestRenderAgentYAMLMatchesGolden(t *testing.T) {
	out, err := Render(sampleAgent())
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	want, err := os.ReadFile("testdata/agent.yaml")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if string(out) != string(want) {
		t.Fatalf("rendered agent.yaml drifted from golden\n--- got ---\n%s\n--- want ---\n%s", out, want)
	}
}

func TestRenderAgentYAMLShape(t *testing.T) {
	out, err := Render(sampleAgent())
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	s := string(out)

	// The reader is extra=forbid: assert the required top-level keys are present
	// and no foreign key leaks in. Optional env is covered below.
	mustContain := []string{"abiVersion:", "metadata:", "model:", "instructions:", "tools:", "expose:", "baseURL:", "apiKeyEnv:"}
	for _, k := range mustContain {
		if !strings.Contains(s, k) {
			t.Errorf("rendered agent.yaml missing %q\n---\n%s", k, s)
		}
	}
	// Keys the writer must NEVER emit (would trip extra=forbid or leak schema).
	mustNotContain := []string{"apiVersion:", "kind:", "provider: \n", "inline:", "file:"}
	for _, k := range mustNotContain {
		if strings.Contains(s, k) {
			t.Errorf("rendered agent.yaml unexpectedly contains %q\n---\n%s", k, s)
		}
	}
	if !strings.Contains(s, "abiVersion: \"v0\"") {
		t.Errorf("expected quoted abiVersion: v0\n---\n%s", s)
	}
}

// TestRenderAgentYAMLRoundTrips renders the agent.yaml and parses it back with a
// strict (DisallowUnknownField) decoder to guard against the writer emitting any
// key the extra=forbid Python reader would reject. The cross-language proof —
// loading this package's golden with agentkit_serve_common.config.load — lives
// in runtimes/common/tests/test_abi_contract.py.
func TestRenderAgentYAMLRoundTrips(t *testing.T) {
	out, err := Render(sampleAgent())
	if err != nil {
		t.Fatalf("render error: %v", err)
	}

	// A mirror of the ABI shape; strict decode fails on any unexpected key.
	var got struct {
		ABIVersion string `yaml:"abiVersion"`
		Metadata   struct {
			Name string `yaml:"name"`
		} `yaml:"metadata"`
		Model struct {
			Provider  string `yaml:"provider"`
			BaseURL   string `yaml:"baseURL"`
			Name      string `yaml:"name"`
			APIKeyEnv string `yaml:"apiKeyEnv"`
		} `yaml:"model"`
		Instructions string `yaml:"instructions"`
		Tools        []struct {
			Name    string   `yaml:"name"`
			Command []string `yaml:"command"`
			Env     []string `yaml:"env"`
		} `yaml:"tools"`
		Env []struct {
			Name     string `yaml:"name"`
			Required bool   `yaml:"required"`
		} `yaml:"env"`
		Expose struct {
			OpenAI bool `yaml:"openai"`
			Port   int  `yaml:"port"`
		} `yaml:"expose"`
	}
	if err := yaml.UnmarshalWithOptions(out, &got, yaml.Strict()); err != nil {
		t.Fatalf("rendered agent.yaml has a key the strict reader rejects: %v\n---\n%s", err, out)
	}
	if got.ABIVersion != "v0" {
		t.Errorf("abiVersion = %q, want v0", got.ABIVersion)
	}
	if got.Model.BaseURL != "https://api.openai.com/v1" || got.Model.APIKeyEnv != "OPENAI_API_KEY" {
		t.Errorf("model aliases not emitted correctly: %+v", got.Model)
	}
	if len(got.Tools) != 1 || got.Tools[0].Name != "fetch" {
		t.Errorf("tools not round-tripped: %+v", got.Tools)
	}
}

func TestRenderAgentYAMLIncludesEnvRequirements(t *testing.T) {
	cfg := sampleConfig()
	cfg.Env = []config.EnvVar{
		{Name: "REQUIRED_FOO", Required: true},
		{Name: "OPTIONAL_BAR"},
	}
	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}

	s := string(out)
	for _, want := range []string{"env:", "name: \"REQUIRED_FOO\"", "required: true", "name: \"OPTIONAL_BAR\""} {
		if !strings.Contains(s, want) {
			t.Fatalf("rendered agent.yaml missing %q\n---\n%s", want, s)
		}
	}
	if strings.Contains(s, "required: false") {
		t.Fatalf("optional env vars should omit required: false\n---\n%s", s)
	}
}

func TestRenderAgentYAMLIncludesRemoteMCPTool(t *testing.T) {
	cfg := sampleConfig()
	cfg.Tools = []config.Tool{{
		Name:      "toolbox",
		Type:      config.ToolTypeMCP,
		Transport: config.ToolTransportStreamableHTTP,
		URLEnv:    "TOOLBOX_ENDPOINT",
		Headers: []config.ToolHeader{{
			Name:  "Foundry-Features",
			Value: "Toolboxes=V1Preview",
		}},
		Auth: &config.Auth{Type: config.AuthTypeBearer, TokenEnv: "TOOLBOX_TOKEN"},
	}}
	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}

	s := string(out)
	for _, want := range []string{
		"type: \"mcp\"",
		"transport: \"streamable-http\"",
		"urlEnv: \"TOOLBOX_ENDPOINT\"",
		"name: \"Foundry-Features\"",
		"value: \"Toolboxes=V1Preview\"",
		"auth:",
		"type: \"bearer\"",
		"tokenEnv: \"TOOLBOX_TOKEN\"",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("rendered agent.yaml missing %q\n---\n%s", want, s)
		}
	}
}

func TestRenderAgentYAMLIncludesContextAndObservability(t *testing.T) {
	cfg := sampleConfig()
	cfg.Context.Providers = []config.ContextProvider{{
		Name:        "knowledge",
		Type:        config.ContextTypeSearch,
		EndpointEnv: "SEARCH_ENDPOINT",
		IndexEnv:    "SEARCH_INDEX",
	}}
	cfg.Observability.OTel.EndpointEnv = "OTEL_EXPORTER_OTLP_ENDPOINT"
	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}

	s := string(out)
	for _, want := range []string{
		"context:",
		"providers:",
		"name: \"knowledge\"",
		"type: \"search\"",
		"endpointEnv: \"SEARCH_ENDPOINT\"",
		"indexEnv: \"SEARCH_INDEX\"",
		"observability:",
		"endpointEnv: \"OTEL_EXPORTER_OTLP_ENDPOINT\"",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("rendered agent.yaml missing %q\n---\n%s", want, s)
		}
	}
}

func TestRenderAgentYAMLIncludesModelWorkloadIdentityAuth(t *testing.T) {
	cfg := sampleConfig()
	cfg.Model.Auth = &config.Auth{Type: config.AuthTypeWorkloadIdentity, Audience: "https://ai.azure.com/.default"}
	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	s := string(out)
	for _, want := range []string{"auth:", "type: \"workload-identity-token\"", "audience: \"https://ai.azure.com/.default\""} {
		if !strings.Contains(s, want) {
			t.Fatalf("rendered agent.yaml missing %q\n---\n%s", want, s)
		}
	}
}

func TestRenderAgentYAMLIncludesBrokeredTools(t *testing.T) {
	cfg := sampleConfig()
	cfg.Tools = nil
	cfg.BrokeredTools = []config.BrokeredTool{{
		Name:          "check-network-telemetry",
		Description:   "Read telemetry.",
		BrokeredClass: config.BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaObject,
			jsonSchemaPropertiesKey: map[string]any{
				"site":        map[string]any{jsonSchemaTypeKey: jsonSchemaString, jsonSchemaMinimumKey: 0.000001},
				"typedFloats": map[string]float64{jsonSchemaMinimumKey: 0.000001},
				"empty":       map[string]any{},
			},
		},
	}}
	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	s := string(out)
	for _, want := range []string{"brokeredTools:", "name: \"check-network-telemetry\"", "description: \"Read telemetry.\"", "brokeredClass: \"read\"", "\"properties\":", "\"site\":", "\"minimum\": 0.000001", "\"typedFloats\":", "\"empty\": {}"} {
		if !strings.Contains(s, want) {
			t.Fatalf("rendered agent.yaml missing %q\n---\n%s", want, s)
		}
	}
	for _, never := range []string{"\"url\":", "\"secretRef\":", "\"headers\":", "\"auth\":", "1e-06"} {
		if strings.Contains(s, never) {
			t.Fatalf("rendered brokered agent.yaml leaked %q\n---\n%s", never, s)
		}
	}
}

func TestRenderAgentYAMLFormatsJSONNumberBrokeredSchemaValuesAsNumbers(t *testing.T) {
	cfg := sampleConfig()
	cfg.Tools = nil
	cfg.BrokeredTools = []config.BrokeredTool{{
		Name:          "numeric-tool",
		Description:   "Read numeric data.",
		BrokeredClass: config.BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaObject,
			jsonSchemaPropertiesKey: map[string]any{
				"small": map[string]any{jsonSchemaTypeKey: jsonSchemaNumber, jsonSchemaMinimumKey: json.Number("1e-7")},
			},
		},
	}}

	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	if !strings.Contains(string(out), "\"minimum\": 0.0000001") || strings.Contains(string(out), "1e-7") {
		t.Fatalf("rendered agent.yaml did not preserve json.Number as fixed YAML number\n---\n%s", out)
	}
}

func TestRenderAgentYAMLRejectsYAMLBinarySchemaValues(t *testing.T) {
	const binaryAgentkitfile = `apiVersion: v1alpha1
kind: Agent
metadata:
  name: binary-schema
runtime: pydantic-ai
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: Read binary fixtures.
brokeredTools:
- name: inspect-binary
  description: Inspect binary metadata.
  brokeredClass: read
  parameters:
    type: object
    properties:
      payload:
        type: string
        default: !!binary SGVsbG8=
expose:
  openai: true
  port: 8080
`

	cfg, err := config.NewFromBytes([]byte(binaryAgentkitfile))
	if err != nil {
		t.Fatalf("load binary agentkitfile: %v", err)
	}
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "unsupported JSON value []uint8") {
		t.Fatalf("binary schema must be rejected rather than silently coerced, got: %v", err)
	}
}

func TestRenderAgentYAMLNormalizesNamedBrokeredSchemaTypes(t *testing.T) {
	type schemaString string
	type schemaBytes []byte
	type schemaStrings []schemaString
	type schemaMap map[schemaString]any

	const (
		propertyName = schemaString("line:\u2028break")
		enumValue    = schemaString("value\twith-tab")
	)
	tool := config.BrokeredTool{
		Name:          "typed-schema-tool",
		Description:   "Read typed schema data.",
		BrokeredClass: config.BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaObject,
			jsonSchemaPropertiesKey: schemaMap{
				propertyName: map[string]any{
					jsonSchemaTypeKey:    schemaString(jsonSchemaString),
					jsonSchemaDefaultKey: schemaBytes("Hello"),
					jsonSchemaEnumKey:    schemaStrings{enumValue},
				},
			},
		},
	}
	binaryDigest, err := config.BrokeredToolSchemaDigest(tool)
	if err != nil {
		t.Fatalf("digest named schema types: %v", err)
	}
	equivalent := tool
	equivalent.Parameters = map[string]any{
		jsonSchemaTypeKey: jsonSchemaObject,
		jsonSchemaPropertiesKey: map[string]any{
			string(propertyName): map[string]any{
				jsonSchemaTypeKey:    jsonSchemaString,
				jsonSchemaDefaultKey: testHelloBase64,
				jsonSchemaEnumKey:    []string{string(enumValue)},
			},
		},
	}
	stringDigest, err := config.BrokeredToolSchemaDigest(equivalent)
	if err != nil {
		t.Fatalf("digest equivalent schema types: %v", err)
	}
	if binaryDigest != stringDigest {
		t.Fatalf("named schema digest = %q, equivalent digest = %q", binaryDigest, stringDigest)
	}
	tool.SchemaDigest = binaryDigest

	agent := sampleAgent()
	agent.Tools = nil
	agent.BrokeredTools = []config.BrokeredTool{tool}
	out, err := Render(agent)
	if err != nil {
		t.Fatalf("render named schema types: %v", err)
	}
	var got struct {
		BrokeredTools []struct {
			Parameters struct {
				Properties map[string]struct {
					Type    string   `yaml:"type"`
					Default any      `yaml:"default"`
					Enum    []string `yaml:"enum"`
				} `yaml:"properties"`
			} `yaml:"parameters"`
		} `yaml:"brokeredTools"`
	}
	if err := yaml.Unmarshal(out, &got); err != nil {
		t.Fatalf("parse rendered named schema types: %v\n---\n%s", err, out)
	}
	property, ok := got.BrokeredTools[0].Parameters.Properties[string(propertyName)]
	if !ok {
		t.Fatalf("rendered schema lost named property key %q: %#v", propertyName, got.BrokeredTools[0].Parameters.Properties)
	}
	if property.Type != jsonSchemaString {
		t.Errorf("type = %q, want %q", property.Type, jsonSchemaString)
	}
	if property.Default != testHelloBase64 {
		t.Errorf("default = %#v (%T), want base64 string", property.Default, property.Default)
	}
	if len(property.Enum) != 1 || property.Enum[0] != string(enumValue) {
		t.Errorf("enum = %#v, want %q", property.Enum, enumValue)
	}
}

func TestRenderAgentYAMLRoundTripsYAMLSensitiveStrings(t *testing.T) {
	tests := []struct {
		name  string
		value string
	}{
		{name: "document start", value: "---"},
		{name: "document end", value: "..."},
		{name: "document end prefix", value: "... value"},
		{name: "explicit key prefix", value: "? ask"},
		{name: "merge key", value: "<<"},
		{name: "value key", value: "="},
		{name: "positive infinity", value: testYAMLPositiveInfinity},
		{name: "negative infinity", value: "-.Inf"},
		{name: "not a number", value: ".NaN"},
		{name: "explicit key tab prefix", value: "?\task"},
		{name: "leading tab", value: "\tvalue"},
		{name: "embedded tab", value: "before\tafter"},
		{name: "trailing tab", value: "value\t"},
		{name: "control", value: "before\x01after"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			value := test.value
			agent := sampleAgent()
			agent.Instructions = value
			agent.Tools = nil
			agent.BrokeredTools = []config.BrokeredTool{{
				Name:          "indicator-tool",
				Description:   "Read indicator values.",
				BrokeredClass: config.BrokeredClassRead,
				Parameters: map[string]any{
					jsonSchemaTypeKey: jsonSchemaObject,
					jsonSchemaPropertiesKey: map[string]any{
						value: map[string]any{
							jsonSchemaTypeKey:    jsonSchemaString,
							jsonSchemaDefaultKey: value,
						},
					},
				},
			}}

			out, err := Render(agent)
			if err != nil {
				t.Fatalf("render YAML-sensitive string %q: %v", value, err)
			}
			var got struct {
				Instructions  string `yaml:"instructions"`
				BrokeredTools []struct {
					Parameters struct {
						Properties map[string]struct {
							Default string `yaml:"default"`
						} `yaml:"properties"`
					} `yaml:"parameters"`
				} `yaml:"brokeredTools"`
			}
			if err := yaml.Unmarshal(out, &got); err != nil {
				t.Fatalf("parse rendered YAML-sensitive string %q: %v\n---\n%s", value, err, out)
			}
			if got.Instructions != value {
				t.Errorf("instructions = %q, want %q", got.Instructions, value)
			}
			property, ok := got.BrokeredTools[0].Parameters.Properties[value]
			if !ok {
				t.Fatalf("rendered schema lost property key %q: %#v", value, got.BrokeredTools[0].Parameters.Properties)
			}
			if property.Default != value {
				t.Errorf("schema default = %q, want %q", property.Default, value)
			}
		})
	}
}

func TestRenderAgentYAMLRoundTripsOrdinaryMultilineStringsInNestedFields(t *testing.T) {
	const multiline = "first line\nsecond line\rthird line"
	agent := sampleAgent()
	agent.Tools = nil
	agent.BrokeredTools = []config.BrokeredTool{{
		Name:          "multiline-tool",
		Description:   multiline,
		BrokeredClass: config.BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey:        jsonSchemaObject,
			jsonSchemaDescriptionKey: multiline,
			jsonSchemaPropertiesKey: map[string]any{
				multiline: map[string]any{
					jsonSchemaTypeKey:        jsonSchemaString,
					jsonSchemaDescriptionKey: multiline,
				},
			},
		},
	}}

	out, err := Render(agent)
	if err != nil {
		t.Fatalf("render multiline strings: %v", err)
	}
	var got struct {
		BrokeredTools []struct {
			Description string `yaml:"description"`
			Parameters  struct {
				Description string `yaml:"description"`
				Properties  map[string]struct {
					Description string `yaml:"description"`
				} `yaml:"properties"`
			} `yaml:"parameters"`
		} `yaml:"brokeredTools"`
	}
	if err := yaml.Unmarshal(out, &got); err != nil {
		t.Fatalf("rendered multiline agent.yaml did not parse: %v\n---\n%s", err, out)
	}
	if len(got.BrokeredTools) != 1 {
		t.Fatalf("brokeredTools = %#v", got.BrokeredTools)
	}
	tool := got.BrokeredTools[0]
	for field, value := range map[string]string{
		jsonSchemaDescriptionKey:            tool.Description,
		"parameters.description":            tool.Parameters.Description,
		"parameters.properties.description": tool.Parameters.Properties[multiline].Description,
	} {
		if value != multiline {
			t.Errorf("%s = %q, want %q", field, value, multiline)
		}
	}
}

func TestRenderAgentYAMLEscapesYAMLLineBreaksInEveryStringField(t *testing.T) {
	const lineBreakText = "NEL:\u0085LS:\u2028PS:\u2029end"
	expected := map[string]bool{}
	marked := func(field string) string {
		value := field + " " + lineBreakText
		expected[value] = true
		return value
	}

	agent := effective.Agent{
		Metadata: config.Metadata{Name: marked("metadata.name")},
		Model: config.Model{
			Provider:  marked("model.provider"),
			BaseURL:   marked("model.baseURL"),
			Name:      marked("model.name"),
			APIKeyEnv: marked("model.apiKeyEnv"),
			Auth: &config.Auth{
				Type:     marked("model.auth.type"),
				TokenEnv: marked("model.auth.tokenEnv"),
				Audience: marked("model.auth.audience"),
			},
		},
		Instructions: marked("instructions"),
		Tools: []config.Tool{{
			Name:      marked("tools.name"),
			Type:      marked("tools.type"),
			Transport: marked("tools.transport"),
			Command:   []string{marked("tools.command[0]"), marked("tools.command[1]")},
			URLEnv:    marked("tools.urlEnv"),
			Headers: []config.ToolHeader{{
				Name:     marked("tools.headers.name"),
				Value:    marked("tools.headers.value"),
				ValueEnv: marked("tools.headers.valueEnv"),
			}},
			Auth: &config.Auth{
				Type:     marked("tools.auth.type"),
				TokenEnv: marked("tools.auth.tokenEnv"),
				Audience: marked("tools.auth.audience"),
			},
			Approval: marked("tools.approval"),
			Env:      []string{marked("tools.env[0]"), marked("tools.env[1]")},
		}},
		BrokeredTools: []config.BrokeredTool{{
			Name:          marked("brokeredTools.name"),
			Description:   marked("brokeredTools.description"),
			BrokeredClass: marked("brokeredTools.brokeredClass"),
			Parameters: map[string]any{
				marked("brokeredTools.parameters.key"): marked("brokeredTools.parameters.value"),
				"slice":                                []string{marked("brokeredTools.parameters.slice")},
			},
			SchemaDigest: marked("brokeredTools.schemaDigest"),
		}},
		Env: []config.EnvVar{{Name: marked("env.name")}},
		Context: config.Context{Providers: []config.ContextProvider{{
			Name:         marked("context.providers.name"),
			Type:         marked("context.providers.type"),
			Source:       marked("context.providers.source"),
			Path:         marked("context.providers.path"),
			ToolRef:      marked("context.providers.toolRef"),
			Index:        marked("context.providers.index"),
			EndpointEnv:  marked("context.providers.endpointEnv"),
			IndexEnv:     marked("context.providers.indexEnv"),
			StoreNameEnv: marked("context.providers.storeNameEnv"),
			Auth: &config.Auth{
				Type:     marked("context.providers.auth.type"),
				TokenEnv: marked("context.providers.auth.tokenEnv"),
				Audience: marked("context.providers.auth.audience"),
			},
		}}},
		Observability: config.Observability{
			OTel: config.ObservabilityOTel{EndpointEnv: marked("observability.otel.endpointEnv")},
			Logs: config.ObservabilityLogs{LevelEnv: marked("observability.logs.levelEnv")},
		},
		Expose: config.Expose{OpenAI: true, Port: 8080},
	}

	out, err := Render(agent)
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	if strings.ContainsAny(string(out), "\u0085\u2028\u2029") {
		t.Fatalf("rendered agent.yaml contains an unescaped YAML line-break code point\n---\n%s", out)
	}

	var decoded any
	if err := yaml.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("rendered agent.yaml did not parse: %v\n---\n%s", err, out)
	}
	seen := map[string]bool{}
	var collectStrings func(any)
	collectStrings = func(value any) {
		switch typed := value.(type) {
		case string:
			seen[typed] = true
		case map[string]any:
			for key, item := range typed {
				seen[key] = true
				collectStrings(item)
			}
		case map[any]any:
			for key, item := range typed {
				collectStrings(key)
				collectStrings(item)
			}
		case []any:
			for _, item := range typed {
				collectStrings(item)
			}
		}
	}
	collectStrings(decoded)
	for value := range expected {
		if !seen[value] {
			t.Errorf("round-tripped agent.yaml lost %q\n---\n%s", value, out)
		}
	}
}

func TestRenderAgentYAMLEdgeCasesMatchCrossLanguageGolden(t *testing.T) {
	const (
		lineBreakText = "NEL:\u0085LS:\u2028PS:\u2029end"
		propertyName  = "line:\u2028break"
	)

	tool := config.BrokeredTool{
		Name:          "unicode-numeric-tool",
		Description:   "description " + lineBreakText,
		BrokeredClass: config.BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey:        jsonSchemaObject,
			jsonSchemaDescriptionKey: "schema " + lineBreakText,
			jsonSchemaPropertiesKey: map[string]any{
				"12:34:56": map[string]any{
					jsonSchemaTypeKey:    jsonSchemaString,
					jsonSchemaDefaultKey: "2001-12-14 21:59:43.10 -5",
				},
				testYAMLPositiveInfinity: map[string]any{
					jsonSchemaTypeKey:    jsonSchemaString,
					jsonSchemaDefaultKey: testYAMLPositiveInfinity,
				},
				"<<": map[string]any{
					jsonSchemaTypeKey:    jsonSchemaString,
					jsonSchemaDefaultKey: "<<",
				},
				"=": map[string]any{
					jsonSchemaTypeKey:    jsonSchemaString,
					jsonSchemaDefaultKey: "=",
				},
				"? ask": map[string]any{
					jsonSchemaTypeKey:    jsonSchemaString,
					jsonSchemaDefaultKey: "before\tafter",
				},
				"binary": map[string]any{
					jsonSchemaTypeKey:    jsonSchemaString,
					jsonSchemaDefaultKey: testHelloBase64,
				},
				propertyName: map[string]any{
					jsonSchemaTypeKey:        "number",
					jsonSchemaDescriptionKey: "property " + lineBreakText,
					jsonSchemaMinimumKey:     math.Copysign(0, -1),
				},
			},
			"required": []any{propertyName},
		},
	}
	digest, err := config.BrokeredToolSchemaDigest(tool)
	if err != nil {
		t.Fatalf("schema digest: %v", err)
	}
	tool.SchemaDigest = digest

	cfg := sampleConfig()
	cfg.Tools = nil
	cfg.Model.APIKeyEnv = ""
	cfg.Instructions = config.Source{Inline: "instructions " + lineBreakText}
	cfg.BrokeredTools = []config.BrokeredTool{tool}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("edge-case config should pass Go validation: %v", err)
	}

	out, err := Render(effective.FromConfig(cfg, cfg.Instructions.Inline))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	want, err := os.ReadFile("testdata/edge-cases.yaml")
	if err != nil {
		t.Fatalf("read edge-case golden: %v", err)
	}
	if string(out) != string(want) {
		t.Fatalf("rendered edge-case agent.yaml drifted from cross-language golden\n--- got ---\n%s\n--- want ---\n%s", out, want)
	}
}

func TestRenderAgentYAMLRejectsUnderflowingJSONNumbers(t *testing.T) {
	cfg := sampleConfig()
	cfg.Tools = nil
	cfg.Instructions = config.Source{Inline: testInstructions}
	cfg.BrokeredTools = []config.BrokeredTool{{
		Name:          "tiny-number-tool",
		Description:   "Read tiny numeric data.",
		BrokeredClass: config.BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaObject,
			jsonSchemaPropertiesKey: map[string]any{
				"tiny": map[string]any{jsonSchemaTypeKey: jsonSchemaNumber, jsonSchemaMinimumKey: json.Number("-1e-350")},
			},
		},
	}}
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "cannot be represented exactly") {
		t.Fatalf("underflowing json.Number must fail exact representation validation, got: %v", err)
	}
}

func TestRenderAgentYAMLPreservesNegativeZeroBrokeredSchemaFloats(t *testing.T) {
	cfg := sampleConfig()
	cfg.Tools = nil
	tool := config.BrokeredTool{
		Name:          "negative-zero-tool",
		Description:   "Preserve negative zero.",
		BrokeredClass: config.BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaObject,
			jsonSchemaPropertiesKey: map[string]any{
				"offset":        map[string]any{jsonSchemaTypeKey: jsonSchemaNumber, jsonSchemaDefaultKey: math.Copysign(0, -1)},
				"jsonOffset":    map[string]any{jsonSchemaTypeKey: jsonSchemaNumber, jsonSchemaDefaultKey: json.Number("-0e0")},
				"largeInteger":  map[string]any{jsonSchemaTypeKey: "integer", jsonSchemaDefaultKey: json.Number("9007199254740995.0")},
				"typedNested":   map[string][]float64{"enum": {math.Copysign(0, -1)}},
				"typedNil":      map[string]any{jsonSchemaDefaultKey: map[string]string(nil)},
				"typedNilSlice": map[string]any{jsonSchemaDefaultKey: []string(nil)},
			},
		},
	}
	digest, err := config.BrokeredToolSchemaDigest(tool)
	if err != nil {
		t.Fatalf("digest error: %v", err)
	}
	tool.SchemaDigest = digest
	cfg.BrokeredTools = []config.BrokeredTool{tool}

	out, err := Render(effective.FromConfig(cfg, testInstructions))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	if strings.Count(string(out), `"default": -0.0`) != 2 {
		t.Fatalf("rendered agent.yaml did not preserve negative zero as a float\n---\n%s", out)
	}
	if !strings.Contains(string(out), `"default": 9007199254740995`) || strings.Contains(string(out), "9007199254740995.0") {
		t.Fatalf("rendered agent.yaml did not preserve a large integral decimal as an integer\n---\n%s", out)
	}
	if strings.Count(string(out), yamlNegativeZero) != 3 {
		t.Fatalf("rendered agent.yaml did not normalize negative zero in typed nested containers\n---\n%s", out)
	}
	if strings.Count(string(out), `"default": null`) != 2 {
		t.Fatalf("rendered agent.yaml did not preserve typed nil containers as null\n---\n%s", out)
	}
}
