package config

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/orka-agents/agentkit/pkg/agentkit/runtimes"
)

const (
	safeLookupToolName      = "safe_lookup"
	jsonSchemaPropertiesKey = "properties"
	brokeredSiteField       = "site"
	brokeredSafeDescription = "safe schema"
	benignNearMissValue     = "near miss"
)

// TestKindProbeRejectsKindlessFile is the regression guard for the AIKit
// silent-misparse bug (plan §5.1/§16.1): a file without `kind: Agent` must be a
// loud load-time error, never a silently-empty config.
func TestKindProbeRejectsKindlessFile(t *testing.T) {
	// An AIKit-style inference file: no kind, has backends/models.
	in := []byte("apiVersion: v1alpha1\nbackends:\n  - llama-cpp\n")
	_, err := NewFromBytes(in)
	if err == nil {
		t.Fatal("expected error for kind-less file, got nil (the silent-misparse bug)")
	}
	if !strings.Contains(err.Error(), "kind") {
		t.Fatalf("expected a kind-related error, got: %v", err)
	}
}

func TestKindProbeRejectsWrongKind(t *testing.T) {
	in := []byte("apiVersion: v1alpha1\nkind: Crew\nmetadata:\n  name: x\n")
	_, err := NewFromBytes(in)
	if err == nil || !strings.Contains(err.Error(), "Crew") {
		t.Fatalf("expected unsupported-kind error mentioning Crew, got: %v", err)
	}
}

func TestStrictParseRejectsUnknownField(t *testing.T) {
	// `runtimes:` (typo of `runtime:`) must be a load-time error under strict mode.
	in := []byte("apiVersion: v1alpha1\nkind: Agent\nmetadata:\n  name: x\nruntimes: pydantic-ai\n")
	_, err := NewFromBytes(in)
	if err == nil {
		t.Fatal("expected strict-mode error for unknown field `runtimes`, got nil")
	}
}

func TestInstructionsAcceptsBareString(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: hello
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: |
  Be helpful.
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	if got := strings.TrimSpace(cfg.Instructions.Inline); got != "Be helpful." {
		t.Fatalf("instructions inline = %q, want %q", got, "Be helpful.")
	}
	if cfg.Instructions.File != "" {
		t.Fatalf("expected empty File, got %q", cfg.Instructions.File)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("valid four-keys config failed validation: %v", err)
	}
}

func TestInstructionsAcceptsFileMapping(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: hello
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions:
  file: ./prompt.md
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	if cfg.Instructions.File != "./prompt.md" {
		t.Fatalf("instructions file = %q, want ./prompt.md", cfg.Instructions.File)
	}
}

func TestValidateReportsMultipleErrors(t *testing.T) {
	// Missing model.name, instructions, and expose.openai=false → at least 3 errors,
	// all reported at once (errors.Join, plan §16.2 #3).
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: broken
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
expose:
  openai: false
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected validation errors, got nil")
	}
	msg := verr.Error()
	for _, want := range []string{"model.name", "instructions", "expose.openai"} {
		if !strings.Contains(msg, want) {
			t.Errorf("validation error missing %q; full: %s", want, msg)
		}
	}
}

func TestValidateRejectsControlPlaneMetadataLabels(t *testing.T) {
	reserved := []string{
		nativeImageLabelNamespace,
		ImageLabelNativeRuntime,
		ImageLabelNativeName,
		ImageLabelNativeABI,
		nativeImageLabelNamespace + ".future-control",
		legacyNativeImageLabelNamespace,
		legacyNativeImageLabelNamespace + ".runtime",
		portableImageLabelNamespace,
		ImageLabelPortableABI,
		ImageLabelPortableRuntime,
		ImageLabelPortableProtocols,
		ImageLabelPortableCapabilities,
		portableImageLabelNamespace + ".future-control",
		orkaImageLabelNamespace,
		ImageLabelOrkaHarnessVersion,
		orkaImageLabelNamespace + ".future-control",
		ImageLabelOCITitle,
	}

	for _, label := range reserved {
		t.Run(label, func(t *testing.T) {
			cfg, err := NewFromBytes(agentBaseYAML(""))
			if err != nil {
				t.Fatalf("parse error: %v", err)
			}
			cfg.Metadata.Labels = map[string]string{label: "spoofed"}

			verr := cfg.Validate()
			if verr == nil {
				t.Fatalf("expected reserved metadata label %q to be rejected", label)
			}
			for _, want := range []string{"metadata.labels", label, "reserved"} {
				if !strings.Contains(verr.Error(), want) {
					t.Errorf("validation error missing %q; full: %s", want, verr)
				}
			}
		})
	}
}

func TestValidateAllowsUnrelatedMetadataLabels(t *testing.T) {
	cfg, err := NewFromBytes(agentBaseYAML(""))
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	cfg.Metadata.Labels = map[string]string{
		"com.example/team":                           "agentkit",
		"org.opencontainers.image.description":       "helpful agent",
		nativeImageLabelNamespace + "-runtime":       benignNearMissValue,
		legacyNativeImageLabelNamespace + "-runtime": benignNearMissValue,
		portableImageLabelNamespace + "-custom":      benignNearMissValue,
		orkaImageLabelNamespace + "-tools":           benignNearMissValue,
		"ai.example.agentkit.runtime":                "unrelated namespace",
	}

	if verr := cfg.Validate(); verr != nil {
		t.Fatalf("unrelated metadata labels should be allowed: %v", verr)
	}
}

func TestValidateRejectsSecretLiteralInApiKeyEnv(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: leaky
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: sk-secret-value-here
instructions: hi
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if verr := cfg.Validate(); verr == nil || !strings.Contains(verr.Error(), "apiKeyEnv") {
		t.Fatalf("expected secret-literal rejection on apiKeyEnv, got: %v", verr)
	}
}

func TestValidateRejectsEmptyToolCommandEntry(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: emptycmd
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
tools:
  - name: fetch
    command: [""]
    env: [FETCH_TOKEN]
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil || !strings.Contains(verr.Error(), "command[0]") {
		t.Fatalf("expected empty command entry rejection, got: %v", verr)
	}
	located := LocateProblems(verr, in)
	if len(located) != 1 {
		t.Fatalf("expected one command problem, got: %+v", located)
	}
	if lp := located[0]; lp.Path != "$.tools[0].command[0]" || lp.StartLine != 13 || lp.EndLine != 13 {
		t.Fatalf("command problem = %+v, want $.tools[0].command[0] on line 13", lp)
	}
}

func TestValidateRejectsEmptyToolEnvName(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: emptyenv
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
tools:
  - name: fetch
    command: ["uvx", "mcp-server-fetch"]
    env: [""]
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if verr := cfg.Validate(); verr == nil || !strings.Contains(verr.Error(), "env entry is empty") {
		t.Fatalf("expected empty env name rejection, got: %v", verr)
	}
}

func TestValidateRejectsImageTool(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: imgtool
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
tools:
  - name: fetch
    image: ghcr.io/example/server-fetch:latest
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if verr := cfg.Validate(); verr == nil || !strings.Contains(verr.Error(), "image-based") {
		t.Fatalf("expected v0 rejection of image-based tool, got: %v", verr)
	}
}

// agentBaseYAML returns a minimal valid four-keys agentkitfile with the given
// `runtime:` line spliced in (empty string omits it).
func agentBaseYAML(runtimeLine string) []byte {
	rt := ""
	if runtimeLine != "" {
		rt = "runtime: " + runtimeLine + "\n"
	}
	return []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: rt-agent
` + rt + `model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
expose:
  openai: true
`)
}

// TestValidateAcceptsRegisteredRuntimes proves the widened runtime gate (plan §8):
// the canonical MAF name, its "maf" alias, LangGraph, the default runtime, and
// an omitted runtime all validate.
func TestValidateAcceptsRegisteredRuntimes(t *testing.T) {
	for _, rt := range []string{"", "pydantic-ai", "microsoft-agent-framework", "maf", "langgraph"} {
		cfg, err := NewFromBytes(agentBaseYAML(rt))
		if err != nil {
			t.Fatalf("runtime %q: parse error: %v", rt, err)
		}
		if verr := cfg.Validate(); verr != nil {
			t.Errorf("runtime %q: expected valid, got: %v", rt, verr)
		}
	}
}

// TestValidateRejectsUnknownRuntime keeps the deterministic gate: an unregistered
// runtime is a clear error that lists the supported set.
func TestValidateRejectsUnknownRuntime(t *testing.T) {
	cfg, err := NewFromBytes(agentBaseYAML("langchain"))
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil || !strings.Contains(verr.Error(), "runtime") {
		t.Fatalf("expected unknown-runtime rejection, got: %v", verr)
	}
	for _, want := range []string{"microsoft-agent-framework", "langgraph"} {
		if !strings.Contains(verr.Error(), want) {
			t.Errorf("error should list supported runtime %q; got: %v", want, verr)
		}
	}
}

func TestEnvDeclarationsValidateAndParse(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: env-agent
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
env:
  - name: REQUIRED_FOO
    required: true
  - name: OPTIONAL_BAR
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(cfg.Env) != 2 {
		t.Fatalf("Env len = %d, want 2", len(cfg.Env))
	}
	if cfg.Env[0].Name != "REQUIRED_FOO" || !cfg.Env[0].Required {
		t.Fatalf("first env = %+v, want required REQUIRED_FOO", cfg.Env[0])
	}
	if cfg.Env[1].Name != "OPTIONAL_BAR" || cfg.Env[1].Required {
		t.Fatalf("second env = %+v, want optional OPTIONAL_BAR", cfg.Env[1])
	}
	if verr := cfg.Validate(); verr != nil {
		t.Fatalf("valid env declarations failed validation: %v", verr)
	}
}

func TestEnvDeclarationsRejectInvalidNamesDuplicatesAndValues(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: env-agent
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
env:
  - name: required-foo
  - name: REQUIRED_FOO
  - name: REQUIRED_FOO
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected env validation errors, got nil")
	}
	msg := verr.Error()
	for _, want := range []string{"env[0].name", "[A-Z0-9_]+", "duplicate env var name"} {
		if !strings.Contains(msg, want) {
			t.Errorf("validation error missing %q; full: %s", want, msg)
		}
	}

	withValue := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: env-agent
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
env:
  - name: REQUIRED_FOO
    value: do-not-allow-literals
expose:
  openai: true
`)
	if _, err := NewFromBytes(withValue); err == nil || !strings.Contains(err.Error(), "value") {
		t.Fatalf("expected strict parse rejection for env.value literal, got: %v", err)
	}
}

func TestValidateRejectsStdioToolWhenRuntimeLacksCapability(t *testing.T) {
	old := append([]runtimes.RuntimeSpec(nil), runtimes.Runtimes...)
	t.Cleanup(func() { runtimes.Runtimes = old })
	for i := range runtimes.Runtimes {
		if runtimes.Runtimes[i].Name == runtimes.PydanticAI {
			runtimes.Runtimes[i].Capabilities = nil
		}
	}

	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: caps-agent
runtime: pydantic-ai
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
tools:
  - name: fetch
    command: ["uvx", "mcp-server-fetch"]
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected runtime capability validation error, got nil")
	}
	msg := verr.Error()
	for _, want := range []string{"runtime \"pydantic-ai\"", "stdio-mcp"} {
		if !strings.Contains(msg, want) {
			t.Errorf("validation error missing %q; full: %s", want, msg)
		}
	}
}

func TestValidateAcceptsRemoteMCPToolWithHeadersAndAuth(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: remote-mcp-agent
runtime: maf
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
tools:
  - name: toolbox
    type: mcp
    transport: streamable-http
    urlEnv: TOOLBOX_ENDPOINT
    headers:
      - name: Foundry-Features
        value: Toolboxes=V1Preview
      - name: X-Trace
        valueEnv: TOOLBOX_TRACE_HEADER
    auth:
      type: bearer
      tokenEnv: TOOLBOX_TOKEN
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if verr := cfg.Validate(); verr != nil {
		t.Fatalf("valid remote MCP tool failed validation: %v", verr)
	}
}

func TestValidateRejectsUnsupportedBrokeredSchemaCompositionKeywords(t *testing.T) {
	for _, keyword := range []string{"allOf", "anyOf", "oneOf", "$ref", "if", "then", "else", "contains", "propertyNames", "dependentSchemas", "patternProperties", "unevaluatedProperties", "uniqueItems"} {
		cfg := validMinimalConfig()
		cfg.BrokeredTools = []BrokeredTool{{
			Name:          safeLookupToolName,
			Description:   brokeredSafeDescription,
			BrokeredClass: BrokeredClassRead,
			Parameters: map[string]any{
				jsonSchemaTypeKey: jsonSchemaTypeObject,
				keyword:           []any{},
			},
		}}
		if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "not supported") {
			t.Fatalf("keyword %q: expected unsupported schema rejection, got: %v", keyword, err)
		}
	}
}

func TestValidateAcceptsBrokeredPropertyNamedType(t *testing.T) {
	cfg := validMinimalConfig()
	cfg.BrokeredTools = []BrokeredTool{{
		Name:          safeLookupToolName,
		Description:   brokeredSafeDescription,
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				jsonSchemaTypeKey: map[string]any{jsonSchemaTypeKey: jsonSchemaTypeString},
			},
		},
	}}

	if err := cfg.Validate(); err != nil {
		t.Fatalf("property named type should validate: %v", err)
	}
}

func TestValidateRejectsInvalidBrokeredSchemaTypeValuesAndDefaults(t *testing.T) {
	cases := []map[string]any{
		{jsonSchemaTypeKey: jsonSchemaTypeObject, jsonSchemaPropertiesKey: map[string]any{brokeredSiteField: map[string]any{jsonSchemaTypeKey: nil}}},
		{jsonSchemaTypeKey: jsonSchemaTypeObject, jsonSchemaPropertiesKey: map[string]any{"n": map[string]any{jsonSchemaTypeKey: jsonSchemaTypeInteger, jsonSchemaDefaultKey: "1"}}},
		{jsonSchemaTypeKey: jsonSchemaTypeObject, jsonSchemaPropertiesKey: map[string]any{"n": map[string]any{jsonSchemaTypeKey: jsonSchemaTypeInteger, jsonSchemaDefaultKey: 9007199254740993.0}}},
		{jsonSchemaTypeKey: jsonSchemaTypeObject, jsonSchemaPropertiesKey: map[string]any{brokeredSiteField: map[string]any{jsonSchemaTypeKey: jsonSchemaTypeString, jsonSchemaEnumKey: []any{0, "ok"}}}},
	}
	for _, schema := range cases {
		cfg := validMinimalConfig()
		cfg.BrokeredTools = []BrokeredTool{{
			Name:          safeLookupToolName,
			Description:   brokeredSafeDescription,
			BrokeredClass: BrokeredClassRead,
			Parameters:    schema,
		}}
		if err := cfg.Validate(); err == nil {
			t.Fatalf("expected invalid schema rejection for %#v", schema)
		}
	}
}

func TestValidateRejectsEmptyBrokeredSchemaEnums(t *testing.T) {
	cases := []map[string]any{
		{jsonSchemaTypeKey: jsonSchemaTypeObject, jsonSchemaEnumKey: []any{}},
		{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				brokeredSiteField: map[string]any{jsonSchemaTypeKey: jsonSchemaTypeString, jsonSchemaEnumKey: []any{}},
			},
		},
	}
	for _, schema := range cases {
		cfg := validMinimalConfig()
		cfg.BrokeredTools = []BrokeredTool{{
			Name:          safeLookupToolName,
			Description:   brokeredSafeDescription,
			BrokeredClass: BrokeredClassRead,
			Parameters:    schema,
		}}
		if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "enum must contain at least one value") {
			t.Fatalf("expected empty enum rejection for %#v, got: %v", schema, err)
		}
	}
}

func TestValidateAcceptsLargeBrokeredIntegerConstraintsWithoutInt64Narrowing(t *testing.T) {
	cfg := validMinimalConfig()
	cfg.BrokeredTools = []BrokeredTool{{
		Name:          safeLookupToolName,
		Description:   brokeredSafeDescription,
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			"minProperties":   1e19,
		},
	}}

	if err := cfg.Validate(); err != nil {
		t.Fatalf("large integral JSON Schema constraint should validate: %v", err)
	}

	cfg.BrokeredTools[0].Parameters["minProperties"] = 1.5
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "must be a non-negative integer") {
		t.Fatalf("fractional JSON Schema constraint should fail validation, got: %v", err)
	}
}

func TestValidatePreservesLargeIntegerSchemaValuesBeforeTypeChecking(t *testing.T) {
	cfg := validMinimalConfig()
	cfg.BrokeredTools = []BrokeredTool{{
		Name:          safeLookupToolName,
		Description:   brokeredSafeDescription,
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				"n": map[string]any{
					jsonSchemaTypeKey:    jsonSchemaTypeInteger,
					jsonSchemaDefaultKey: int64(9007199254740993),
					jsonSchemaEnumKey:    []any{int64(9007199254740993)},
				},
			},
		},
	}}

	if err := cfg.Validate(); err != nil {
		t.Fatalf("large integer schema values should validate without float normalization: %v", err)
	}
}

func TestValidateNormalizesJSONContainersWithoutChangingScalarMeaning(t *testing.T) {
	type count int64
	pointerNumber := json.Number("2.0")
	namedSchema := map[string]any{jsonSchemaTypeKey: jsonSchemaTypeInteger, jsonSchemaDefaultKey: count(1)}
	numberSchema := map[string]any{jsonSchemaTypeKey: jsonSchemaTypeInteger, jsonSchemaDefaultKey: json.Number("1.0")}
	fractionalSchema := map[string]any{jsonSchemaTypeKey: jsonSchemaTypeNumber, jsonSchemaDefaultKey: json.Number("0.1")}
	properties := map[string]any{
		"fractional": fractionalSchema,
		"named":      namedSchema,
		"number":     numberSchema,
		"pointer":    map[string]any{jsonSchemaTypeKey: jsonSchemaTypeInteger, jsonSchemaDefaultKey: &pointerNumber},
	}
	cfg := validMinimalConfig()
	cfg.BrokeredTools = []BrokeredTool{{
		Name:          safeLookupToolName,
		Description:   brokeredSafeDescription,
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey:       jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: properties,
		},
	}}

	if err := cfg.Validate(); err != nil {
		t.Fatalf("JSON-serializable named and json.Number integer values should validate: %v", err)
	}

	numberSchema[jsonSchemaDefaultKey] = json.Number("1.00000000000000001")
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "cannot be represented exactly") {
		t.Fatalf("lossy non-integral json.Number must fail exact representation validation, got: %v", err)
	}
	numberSchema[jsonSchemaDefaultKey] = json.Number("100000000000000000001.0")
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "must match the declared JSON Schema type") {
		t.Fatalf("large float-form json.Number must fail integer validation, got: %v", err)
	}
	numberSchema[jsonSchemaDefaultKey] = json.Number("100000000000000000001")
	if err := cfg.Validate(); err != nil {
		t.Fatalf("large lexical integer json.Number should validate exactly: %v", err)
	}
	numberSchema[jsonSchemaDefaultKey] = json.Number("1/1")
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "must be JSON serializable") {
		t.Fatalf("non-JSON json.Number syntax must fail serialization validation, got: %v", err)
	}
	numberSchema[jsonSchemaDefaultKey] = json.Number("1.0")
	fractionalSchema[jsonSchemaDefaultKey] = json.Number("1e-400")
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "cannot be represented exactly") {
		t.Fatalf("underflowing json.Number must fail exact representation validation, got: %v", err)
	}
	fractionalSchema[jsonSchemaDefaultKey] = json.Number("0.1")

	namedSchema[jsonSchemaDefaultKey] = map[string]any(nil)
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "must match the declared JSON Schema type") {
		t.Fatalf("typed nil map must retain JSON null semantics, got: %v", err)
	}
}

func TestValidateRejectsLossyTypelessJSONNumberSchemaValues(t *testing.T) {
	cfg := validMinimalConfig()
	cfg.BrokeredTools = []BrokeredTool{{
		Name:          safeLookupToolName,
		Description:   brokeredSafeDescription,
		BrokeredClass: BrokeredClassWrite,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				"ratio": map[string]any{jsonSchemaDefaultKey: json.Number("0.100000000000000005")},
			},
		},
	}}

	err := cfg.Validate()
	if err == nil || !strings.Contains(err.Error(), "cannot be represented exactly") {
		t.Fatalf("lossy typeless json.Number must fail validation, got: %v", err)
	}
}

func TestValidateMeasuresBrokeredSchemaSizeWithCanonicalJSON(t *testing.T) {
	cfg := validMinimalConfig()
	cfg.BrokeredTools = []BrokeredTool{{
		Name:          safeLookupToolName,
		Description:   brokeredSafeDescription,
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey:            jsonSchemaTypeObject,
			brokeredDigestDescriptionKey: strings.Repeat("<", 11_000),
		},
	}}

	if err := cfg.Validate(); err != nil {
		t.Fatalf("canonical schema below the byte limit should validate despite json.Marshal HTML escaping: %v", err)
	}

	cfg.BrokeredTools[0].Parameters[brokeredDigestDescriptionKey] = strings.Repeat("a", 64*1024)
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "schema is too large") {
		t.Fatalf("canonical schema above the byte limit should fail validation, got: %v", err)
	}
}

func TestValidateRejectsInvalidRemoteMCPToolShapes(t *testing.T) {
	cases := map[string]string{ //nolint:gosec // test YAML uses credential-looking field names/invalid examples, not real secrets
		"missing type": `tools:
  - name: remote
    transport: streamable-http
    urlEnv: REMOTE_MCP_URL
`,
		"bad url env": `tools:
  - name: remote
    type: mcp
    transport: streamable-http
    urlEnv: https://example.com/mcp
`,
		"static authorization": `tools:
  - name: remote
    type: mcp
    transport: streamable-http
    urlEnv: REMOTE_MCP_URL
    headers:
      - name: Authorization
        value: static-auth-header
`,
		"static api key header": `tools:
  - name: remote
    type: mcp
    transport: streamable-http
    urlEnv: REMOTE_MCP_URL
    headers:
      - name: X-API-Key
        value: not-secret-looking
`,
		"static cookie header": `tools:
  - name: remote
    type: mcp
    transport: streamable-http
    urlEnv: REMOTE_MCP_URL
    headers:
      - name: Cookie
        value: cookie-static-value
`,
		"authorization value env plus auth": `tools:
  - name: remote
    type: mcp
    transport: streamable-http
    urlEnv: REMOTE_MCP_URL
    headers:
      - name: Authorization
        valueEnv: AUTH_HEADER
    auth:
      type: bearer
      tokenEnv: REMOTE_TOKEN
`,
		"missing token": `tools:
  - name: remote
    type: mcp
    transport: streamable-http
    urlEnv: REMOTE_MCP_URL
    auth:
      type: bearer
`,
	}
	for name, toolYAML := range cases {
		t.Run(name, func(t *testing.T) {
			in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: remote-mcp-agent
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
` + toolYAML + `expose:
  openai: true
`)
			cfg, err := NewFromBytes(in)
			if err != nil {
				t.Fatalf("parse error: %v", err)
			}
			if verr := cfg.Validate(); verr == nil {
				t.Fatal("expected validation error, got nil")
			}
		})
	}
}

func TestValidateRejectsUnsupportedContextProviderCapabilities(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: context-agent
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
context:
  providers:
    - name: knowledge
      type: search
      endpointEnv: SEARCH_ENDPOINT
      indexEnv: SEARCH_INDEX
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected unsupported context capability error, got nil")
	}
	if !strings.Contains(verr.Error(), runtimes.CapabilityContextProviderSearch) {
		t.Fatalf("expected search capability error, got: %v", verr)
	}
}

func TestValidateRejectsUnsupportedModelWorkloadIdentityAndApproval(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: capability-agent
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  auth:
    type: workload-identity-token
    audience: https://example.com/.default
instructions: hi
tools:
  - name: remote
    type: mcp
    transport: streamable-http
    urlEnv: REMOTE_MCP_URL
    approval: always
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected unsupported capability errors, got nil")
	}
	msg := verr.Error()
	for _, want := range []string{runtimes.CapabilityModelWorkloadIdentityAuth, runtimes.CapabilityToolApproval} {
		if !strings.Contains(msg, want) {
			t.Fatalf("expected %s in validation error, got: %v", want, verr)
		}
	}
}

func TestValidateMAFSupportsSearchAndMemoryContextProviders(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: context-agent
runtime: microsoft-agent-framework
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
context:
  providers:
    - name: knowledge
      type: search
      endpointEnv: SEARCH_ENDPOINT
      indexEnv: SEARCH_INDEX
      auth:
        type: workload-identity-token
        audience: https://search.azure.com/.default
    - name: memory
      type: memory
      endpointEnv: MEMORY_ENDPOINT
      storeNameEnv: MEMORY_STORE_NAME
      auth:
        type: workload-identity-token
        audience: https://ai.azure.com/.default
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if verr := cfg.Validate(); verr != nil {
		t.Fatalf("MAF search+memory context providers should validate: %v", verr)
	}
}

func TestValidateRejectsInvalidMCPSkillsToolRefs(t *testing.T) {
	cases := map[string]string{
		"unknown": `tools:
  - name: toolbox
    type: mcp
    transport: streamable-http
    urlEnv: TOOLBOX_ENDPOINT
context:
  providers:
    - type: skills
      source: mcp
      toolRef: missing
`,
		"stdio": `tools:
  - name: local
    command: ["uvx", "mcp-server-fetch"]
context:
  providers:
    - type: skills
      source: mcp
      toolRef: local
`,
	}
	for name, yaml := range cases {
		t.Run(name, func(t *testing.T) {
			in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: skills-agent
runtime: microsoft-agent-framework
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
` + yaml + `expose:
  openai: true
`)
			cfg, err := NewFromBytes(in)
			if err != nil {
				t.Fatalf("parse error: %v", err)
			}
			if verr := cfg.Validate(); verr == nil {
				t.Fatal("expected invalid MCP skills toolRef validation error, got nil")
			}
		})
	}
}

func TestValidateAllowsMCPSkillsWithoutIndexWhenToolRefIsRemoteMCP(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: skills-agent
runtime: microsoft-agent-framework
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
tools:
  - name: toolbox
    type: mcp
    transport: streamable-http
    urlEnv: TOOLBOX_ENDPOINT
context:
  providers:
    - type: skills
      source: mcp
      toolRef: toolbox
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if verr := cfg.Validate(); verr != nil {
		t.Fatalf("valid MCP skills toolRef should pass without index: %v", verr)
	}
}

func TestValidateRejectsBearerAuthForContextProviders(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: context-agent
runtime: microsoft-agent-framework
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
context:
  providers:
    - name: knowledge
      type: search
      endpointEnv: SEARCH_ENDPOINT
      indexEnv: SEARCH_INDEX
      auth:
        type: bearer
        tokenEnv: SEARCH_TOKEN
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected bearer context auth validation error, got nil")
	}
	if !strings.Contains(verr.Error(), "not supported for context providers") {
		t.Fatalf("expected context auth error, got: %v", verr)
	}
}

func TestValidateRejectsRelativeFilesystemSkillsPath(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: skills-agent
runtime: microsoft-agent-framework
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
context:
  providers:
    - type: skills
      source: filesystem
      path: ./skills
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected relative filesystem skills path validation error, got nil")
	}
	if !strings.Contains(verr.Error(), "/agent/skills") {
		t.Fatalf("expected /agent/skills guidance, got: %v", verr)
	}
}

func TestValidateRejectsBearerAuthForSkillsContextProvider(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: skills-agent
runtime: microsoft-agent-framework
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
context:
  providers:
    - type: skills
      source: filesystem
      path: /agent/skills
      auth:
        type: bearer
        tokenEnv: SKILLS_TOKEN
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected bearer auth validation error for skills context provider, got nil")
	}
	if !strings.Contains(verr.Error(), "must not be set for skills context providers") {
		t.Fatalf("expected context auth error, got: %v", verr)
	}
}

func TestFilesystemSkillsPathUsesPOSIXSemantics(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: skills-agent
runtime: microsoft-agent-framework
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
context:
  providers:
    - type: skills
      source: filesystem
      path: /agent/skills/support-style
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if verr := cfg.Validate(); verr != nil {
		t.Fatalf("POSIX /agent/skills subpath should validate: %v", verr)
	}
}

func TestValidateRejectsAuthForSkillsContextProvider(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: skills-agent
runtime: microsoft-agent-framework
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
context:
  providers:
    - type: skills
      source: filesystem
      path: /agent/skills
      auth:
        type: workload-identity-token
        audience: https://ai.azure.com/.default
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected skills auth validation error, got nil")
	}
	if !strings.Contains(verr.Error(), "auth must not be set for skills context providers") {
		t.Fatalf("expected skills auth error, got: %v", verr)
	}
}

func TestValidateRejectsUnsupportedLogObservability(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: logs-agent
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
observability:
  logs:
    levelEnv: LOG_LEVEL
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected unsupported log observability validation error, got nil")
	}
	if !strings.Contains(verr.Error(), "observability.logs.levelEnv is not supported") {
		t.Fatalf("expected logs support error, got: %v", verr)
	}
}

func TestBrokeredToolSchemaDigestMatchesPythonCanonicalJSON(t *testing.T) {
	tool := BrokeredTool{
		Name:          "check-network-telemetry",
		Description:   "São & R&D <safe>",
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				brokeredSiteField: map[string]any{jsonSchemaTypeKey: jsonSchemaTypeString, brokeredDigestDescriptionKey: "São & R&D <safe>"},
			},
			jsonSchemaRequiredKey: []any{brokeredSiteField},
		},
	}

	digest, err := BrokeredToolSchemaDigest(tool)
	if err != nil {
		t.Fatalf("digest: %v", err)
	}
	if digest != "sha256:7066de4e62dd1a6550701772aad901e1efcf5eb81a4f639252f82e6c7be8d4c1" {
		t.Fatalf("digest = %q", digest)
	}
}

func TestBrokeredToolSchemaDigestPreservesIntegralJSONNumberDecimals(t *testing.T) {
	decimalTool := BrokeredTool{
		Name:          "numeric-tool",
		Description:   brokeredSafeDescription,
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				"id": map[string]any{jsonSchemaDefaultKey: json.Number("9007199254740995.0")},
			},
		},
	}
	integerTool := decimalTool
	integerTool.Parameters = map[string]any{
		jsonSchemaTypeKey: jsonSchemaTypeObject,
		jsonSchemaPropertiesKey: map[string]any{
			"id": map[string]any{jsonSchemaDefaultKey: int64(9007199254740995)},
		},
	}

	decimalDigest, err := BrokeredToolSchemaDigest(decimalTool)
	if err != nil {
		t.Fatalf("decimal digest error: %v", err)
	}
	integerDigest, err := BrokeredToolSchemaDigest(integerTool)
	if err != nil {
		t.Fatalf("integer digest error: %v", err)
	}
	if decimalDigest != integerDigest {
		t.Fatalf("integral decimal digest = %s, integer digest = %s", decimalDigest, integerDigest)
	}
}

func TestCanonicalJSONStringMatchesPythonForUnicodeLineSeparators(t *testing.T) {
	value := "line\u2028paragraph\u2029literal\\u2028"

	encoded, err := canonicalJSONString(value)
	if err != nil {
		t.Fatalf("canonicalJSONString error: %v", err)
	}
	want := "\"line\u2028paragraph\u2029literal\\\\u2028\""
	if string(encoded) != want {
		t.Fatalf("canonicalJSONString = %q, want %q", encoded, want)
	}
}

func TestBrokeredToolSchemaDigestNormalizesIntegerValuedFloatConstraints(t *testing.T) {
	tool := BrokeredTool{
		Name:          "numeric-tool",
		Description:   "Numeric constraints.",
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				"retries": map[string]any{jsonSchemaTypeKey: jsonSchemaTypeNumber, jsonSchemaMinimumKey: 1.0},
			},
		},
	}

	digest, err := BrokeredToolSchemaDigest(tool)
	if err != nil {
		t.Fatalf("digest: %v", err)
	}
	if digest != "sha256:7ad9d43791e157981bcd65fd8452c9e71a64064875cc1330ced42d4956bf7d75" {
		t.Fatalf("digest = %q", digest)
	}
}

func TestBrokeredToolSchemaDigestCanonicalizesNumericConstraints(t *testing.T) {
	tool := BrokeredTool{
		Name:          "num-tool",
		Description:   "Numeric schema",
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				"n": map[string]any{jsonSchemaTypeKey: jsonSchemaTypeNumber, jsonSchemaMinimumKey: 1e-6},
			},
			jsonSchemaRequiredKey: []any{"n"},
		},
	}

	digest, err := BrokeredToolSchemaDigest(tool)
	if err != nil {
		t.Fatalf("digest: %v", err)
	}
	if digest != "sha256:83bf12180154a21f8ba19049687e24acee9ef430966af67a83721a10bf7eee50" {
		t.Fatalf("digest = %q", digest)
	}
}

func TestBrokeredToolSchemaDigestCanonicalizesPositiveExponentNumbers(t *testing.T) {
	tool := BrokeredTool{
		Name:          "large-num-tool",
		Description:   "Large numeric schema",
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				"n": map[string]any{jsonSchemaTypeKey: jsonSchemaTypeNumber, jsonSchemaMaximumKey: 1e20},
			},
		},
	}

	digest, err := BrokeredToolSchemaDigest(tool)
	if err != nil {
		t.Fatalf("digest: %v", err)
	}
	if digest != "sha256:51ebe9cdbb967453ba3ae9fb737566028814968d8121ba0d32825f7c8ffb5639" {
		t.Fatalf("digest = %q", digest)
	}
}

func TestValidateAcceptsDigestForTypedNestedBrokeredSchemaMaps(t *testing.T) {
	tool := BrokeredTool{
		Name:          safeLookupToolName,
		Description:   brokeredSafeDescription,
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				brokeredSiteField: map[string]string{jsonSchemaTypeKey: jsonSchemaTypeString},
			},
		},
	}
	digest, err := BrokeredToolSchemaDigest(tool)
	if err != nil {
		t.Fatalf("digest: %v", err)
	}
	tool.SchemaDigest = digest
	cfg := validMinimalConfig()
	cfg.BrokeredTools = []BrokeredTool{tool}

	if err := cfg.Validate(); err != nil {
		t.Fatalf("valid typed nested schema with digest failed validation: %v", err)
	}
}

func TestBrokeredToolSchemaDigestPreservesLargeIntegers(t *testing.T) {
	tool := BrokeredTool{
		Name:          "large-int-tool",
		Description:   "Large integer schema",
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				"n": map[string]any{jsonSchemaTypeKey: jsonSchemaTypeInteger, "const": int64(9007199254740993)},
			},
		},
	}

	digest, err := BrokeredToolSchemaDigest(tool)
	if err != nil {
		t.Fatalf("digest: %v", err)
	}
	if digest != "sha256:d0e703a2d84f12caa3275fbf8632ec8626ecc84fb27e187380ed61d54cef0e23" {
		t.Fatalf("digest = %q", digest)
	}
}

func TestBrokeredToolSchemaDigestCanonicalizesJSONNumberExponentSpellings(t *testing.T) {
	tool := BrokeredTool{
		Name:          "json-number-tool",
		Description:   "JSON number schema",
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				"n": map[string]any{jsonSchemaTypeKey: jsonSchemaTypeNumber, jsonSchemaMinimumKey: json.Number("1e-7")},
			},
		},
	}

	digest, err := BrokeredToolSchemaDigest(tool)
	if err != nil {
		t.Fatalf("digest: %v", err)
	}
	if digest != "sha256:f38729b2c519d7a92f3ddcbb7139271a0a6e25f515e180fa5a16737e5a0efcb4" {
		t.Fatalf("digest = %q", digest)
	}
}

func TestBrokeredToolSchemaDigestCanonicalizesExponentNumberSpellings(t *testing.T) {
	tool := BrokeredTool{
		Name:          "exponent-num-tool",
		Description:   "Exponent numeric schema",
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				"small": map[string]any{jsonSchemaTypeKey: jsonSchemaTypeNumber, "minimum": 1e-7},
				"large": map[string]any{jsonSchemaTypeKey: jsonSchemaTypeNumber, jsonSchemaMaximumKey: 1e21},
			},
		},
	}

	digest, err := BrokeredToolSchemaDigest(tool)
	if err != nil {
		t.Fatalf("digest: %v", err)
	}
	if digest != "sha256:bb4fac58c3f65a33ed3c4ebfa10e5da9c3dc5a9250a1316f64d25e40e0e645e0" {
		t.Fatalf("digest = %q", digest)
	}
}

func TestValidateAcceptsBrokeredToolsWithMatchingDigest(t *testing.T) {
	tool := BrokeredTool{
		Name:          "check-network-telemetry",
		Description:   "Read sanitized optical telemetry.",
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				brokeredSiteField: map[string]any{"type": "string"},
			},
			jsonSchemaRequiredKey: []any{brokeredSiteField},
		},
	}
	digest, err := BrokeredToolSchemaDigest(tool)
	if err != nil {
		t.Fatalf("digest: %v", err)
	}
	tool.SchemaDigest = digest
	cfg := validMinimalConfig()
	cfg.BrokeredTools = []BrokeredTool{tool}

	if err := cfg.Validate(); err != nil {
		t.Fatalf("valid brokered tool failed validation: %v", err)
	}
}

func TestValidateAcceptsHarmlessBrokeredDescriptions(t *testing.T) {
	for _, description := range []string{"Read basic telemetry", "Count model tokens", "Count model tokens 100", "Count model tokens: 1,000."} {
		cfg := validMinimalConfig()
		cfg.BrokeredTools = []BrokeredTool{{
			Name:          safeLookupToolName,
			Description:   description,
			BrokeredClass: BrokeredClassRead,
			Parameters:    map[string]any{jsonSchemaTypeKey: jsonSchemaTypeObject},
		}}

		if err := cfg.Validate(); err != nil {
			t.Errorf("description %q should be accepted: %v", description, err)
		}
	}
}

func TestValidateRejectsUnsafeBrokeredDescriptions(t *testing.T) {
	for _, description := range []string{"contains sk-secret", "execution at https://tool.default", "Bearer token required", "Basic Zm9vOmJhcg==", "Basic=dXNlcjpwYXNz", "Basic user:pass", "Basic alice: pass1", "alice:pass is the Basic credential", "Use Basic credential abc123", "Use Basic `dXNlcjpwYXNz`", "Use HTTP Basic value dXNlcjpwYXNz", "Use HTTP Basic value=dXNlcjpwYXNz", "Use HTTP-Basic value dXNlcjpwYXNz", "token=abc123", `{"token":"abc123"}`, `'access_token'=abc123`, "token.value=abc123", "token[value]=abc123", "Use --token abc123", "access token abc123", "token abc123", "token 123", "token abc", "token ABCDEF", `"token" "abc123"`, "model token abc123", `input token "abc123"`, `token "abcdef"`, `"token" "ABCDEF"`, `token "abc"`, `input token '123'`, "token.value abc123", "token[value] abc123", "request.token abc123", "request/token abc123", "request-token abc123", "requesttoken abc123", `token used: 'abc'`, "Use abc123 as model [token]", "Count requests. Use model token 123", "Finished counting. Model token 123", `Finished "counting." Model token 123`, "Count model tokens 123456789012345678901234567890", "Read {auth}", "Read (header)", "Read [REDACTED_AUTH_HEADER]", "execution at tool.default.svc.cluster.local"} {
		cfg := validMinimalConfig()
		cfg.BrokeredTools = []BrokeredTool{{
			Name:          safeLookupToolName,
			Description:   description,
			BrokeredClass: BrokeredClassRead,
			Parameters:    map[string]any{jsonSchemaTypeKey: jsonSchemaTypeObject},
		}}

		if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "description") {
			t.Fatalf("description %q: expected unsafe description rejection, got: %v", description, err)
		}
	}
}

func TestValidateRejectsTypedNestedUnsafeBrokeredSchemaValues(t *testing.T) {
	cfg := validMinimalConfig()
	cfg.BrokeredTools = []BrokeredTool{{
		Name:          safeLookupToolName,
		Description:   brokeredSafeDescription,
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				brokeredSiteField: map[string]string{jsonSchemaTypeKey: jsonSchemaTypeString, jsonSchemaDefaultKey: "sk-not-a-real-secret"},
			},
		},
	}}

	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "secret-like") {
		t.Fatalf("expected typed nested unsafe value rejection, got: %v", err)
	}
}

func TestValidateRejectsPrivateKeyBrokeredNamesAndStrings(t *testing.T) {
	for _, params := range []map[string]any{
		{jsonSchemaTypeKey: jsonSchemaTypeObject, jsonSchemaPropertiesKey: map[string]any{"privateKey": map[string]any{jsonSchemaTypeKey: jsonSchemaTypeString}}},
		{jsonSchemaTypeKey: jsonSchemaTypeObject, jsonSchemaPropertiesKey: map[string]any{brokeredSiteField: map[string]any{jsonSchemaTypeKey: jsonSchemaTypeString, jsonSchemaDefaultKey: "BEGIN PRIVATE KEY"}}},
	} {
		cfg := validMinimalConfig()
		cfg.BrokeredTools = []BrokeredTool{{
			Name:          safeLookupToolName,
			Description:   brokeredSafeDescription,
			BrokeredClass: BrokeredClassRead,
			Parameters:    params,
		}}
		if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "safe") && !strings.Contains(err.Error(), "secret-like") {
			t.Fatalf("expected private key rejection, got: %v", err)
		}
	}
}

func TestValidateRejectsUnsafeBrokeredSchemaStringValues(t *testing.T) {
	for _, value := range []string{"see https://internal-tool", "Bearer abc", "Bearer: abc", "Bearer=abc", "authorization header", "token=abc123", "tool.default.svc.cluster.local", "example ghp_not_real", "AWS key AKIAEXAMPLE"} {
		cfg := validMinimalConfig()
		cfg.BrokeredTools = []BrokeredTool{{
			Name:          safeLookupToolName,
			Description:   brokeredSafeDescription,
			BrokeredClass: BrokeredClassRead,
			Parameters: map[string]any{
				jsonSchemaTypeKey: jsonSchemaTypeObject,
				jsonSchemaPropertiesKey: map[string]any{
					brokeredSiteField: map[string]any{jsonSchemaTypeKey: jsonSchemaTypeString, brokeredDigestDescriptionKey: value},
				},
			},
		}}

		if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "secret-like") && !strings.Contains(err.Error(), "URL") {
			t.Fatalf("value %q: expected unsafe schema string rejection, got: %v", value, err)
		}
	}
}

func TestValidateRejectsCommonCredentialBrokeredParameterNames(t *testing.T) {
	for _, field := range []string{brokeredAuthenticationWord, "authConfig", "clientSecret", "dbPassword", "passphrase", "pwd", "apiKey", credentialHeaderAPIKey, "baseUrl", "callbackURL", "apiEndpoint", "sessionCookie", "cookies"} {
		cfg := validMinimalConfig()
		cfg.BrokeredTools = []BrokeredTool{{
			Name:          safeLookupToolName,
			Description:   brokeredSafeDescription,
			BrokeredClass: BrokeredClassRead,
			Parameters: map[string]any{
				jsonSchemaTypeKey: jsonSchemaTypeObject,
				jsonSchemaPropertiesKey: map[string]any{
					field: map[string]any{jsonSchemaTypeKey: jsonSchemaTypeString},
				},
			},
		}}

		if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "not safe") {
			t.Fatalf("field %q: expected unsafe brokered schema rejection, got: %v", field, err)
		}
	}
}

func TestValidateRejectsCommonCredentialBrokeredRequiredNames(t *testing.T) {
	for _, field := range []string{brokeredAuthenticationWord, "authConfig", "clientSecret", "dbPassword", "passphrase", "pwd", "apiKey", credentialHeaderAPIKey, "baseUrl", "callbackURL", "apiEndpoint", "sessionCookie", "cookies"} {
		cfg := validMinimalConfig()
		cfg.BrokeredTools = []BrokeredTool{{
			Name:          safeLookupToolName,
			Description:   brokeredSafeDescription,
			BrokeredClass: BrokeredClassRead,
			Parameters: map[string]any{
				jsonSchemaTypeKey:     jsonSchemaTypeObject,
				jsonSchemaRequiredKey: []any{field},
			},
		}}

		if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "not safe") {
			t.Fatalf("field %q: expected unsafe brokered required rejection, got: %v", field, err)
		}
	}
}

func TestValidateAcceptsHarmlessBrokeredAuthorField(t *testing.T) {
	cfg := validMinimalConfig()
	cfg.BrokeredTools = []BrokeredTool{{
		Name:          safeLookupToolName,
		Description:   brokeredSafeDescription,
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				"author": map[string]any{jsonSchemaTypeKey: jsonSchemaTypeString},
			},
			jsonSchemaRequiredKey: []any{"author"},
		},
	}}

	if err := cfg.Validate(); err != nil {
		t.Fatalf("safe author field should validate: %v", err)
	}
}

func TestValidateRejectsUnsafeBrokeredToolSchema(t *testing.T) {
	for _, unsafeName := range []string{brokeredTokenWord, "access_token", "authHeader", "authorizationHeader", "httpHeaders", "accessKey", "clientSecretValue", "tokenValue", "apiSecretKey", brokeredUnsafeCookieKey, "subscriptionKey", "xFunctionsKey"} {
		cfg := validMinimalConfig()
		cfg.BrokeredTools = []BrokeredTool{{
			Name:          safeLookupToolName,
			Description:   brokeredSafeDescription,
			BrokeredClass: BrokeredClassRead,
			Parameters: map[string]any{
				jsonSchemaTypeKey: jsonSchemaTypeObject,
				jsonSchemaPropertiesKey: map[string]any{
					unsafeName: map[string]any{jsonSchemaTypeKey: jsonSchemaTypeString},
				},
			},
		}}

		if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "not safe") {
			t.Fatalf("expected unsafe brokered schema rejection for %s, got: %v", unsafeName, err)
		}
	}
}

func TestHasUnsafeBrokeredTextMatchesAuthSchemesAsWords(t *testing.T) {
	for _, safe := range []string{"This tool basically reads telemetry", "Read crossbearer telemetry labels"} {
		if hasUnsafeBrokeredText(safe) {
			t.Fatalf("expected %q to be safe brokered text", safe)
		}
	}
	for _, unsafe := range []string{"Use bearer auth", "Basic authentication required"} {
		if !hasUnsafeBrokeredText(unsafe) {
			t.Fatalf("expected %q to be unsafe brokered text", unsafe)
		}
	}
}

func TestValidateRejectsSecretLiteralBrokeredSchemaStringValues(t *testing.T) {
	cfg := validMinimalConfig()
	cfg.BrokeredTools = []BrokeredTool{{
		Name:          safeLookupToolName,
		Description:   brokeredSafeDescription,
		BrokeredClass: BrokeredClassRead,
		Parameters: map[string]any{
			jsonSchemaTypeKey: jsonSchemaTypeObject,
			jsonSchemaPropertiesKey: map[string]any{
				brokeredSiteField: map[string]any{jsonSchemaTypeKey: jsonSchemaTypeString, jsonSchemaDefaultKey: "sk-not-a-real-secret"},
			},
		},
	}}

	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "secret-like material") {
		t.Fatalf("expected secret literal brokered schema rejection, got: %v", err)
	}
}

func TestValidateRejectsBrokeredToolDigestMismatchAndNameOverlap(t *testing.T) {
	cfg := validMinimalConfig()
	cfg.Tools = []Tool{{Name: safeLookupToolName, Command: []string{"echo", "ok"}}}
	cfg.BrokeredTools = []BrokeredTool{{
		Name:          safeLookupToolName,
		Description:   brokeredSafeDescription,
		BrokeredClass: BrokeredClassRead,
		Parameters:    map[string]any{jsonSchemaTypeKey: jsonSchemaTypeObject},
		SchemaDigest:  "sha256:" + strings.Repeat("0", 64),
	}}

	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected validation errors, got nil")
	}
	msg := err.Error()
	for _, want := range []string{"tools and brokeredTools cannot be mixed", "schemaDigest does not match"} {
		if !strings.Contains(msg, want) {
			t.Fatalf("expected %q in error, got: %v", want, err)
		}
	}
}

func TestStrictParseRejectsUnsafeBrokeredTopLevelField(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: brokered
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
instructions: hi
brokeredTools:
  - name: safe_lookup
    description: safe schema
    brokeredClass: read
    parameters:
      type: object
    url: http://tool.default.svc.cluster.local
expose:
  openai: true
`)
	_, err := NewFromBytes(in)
	if err == nil || !strings.Contains(err.Error(), "url") {
		t.Fatalf("expected strict parse rejection for unsafe field, got: %v", err)
	}
}

func validMinimalConfig() *AgentConfig {
	cfg, err := NewFromBytes(agentBaseYAML(""))
	if err != nil {
		panic(err)
	}
	return cfg
}
