package render

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/goccy/go-yaml"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
)

const (
	testBrokeredToolsTarget = "brokered-tools"
	testDispatchToolCRD     = `apiVersion: core.orka.ai/v1alpha1
kind: Tool
metadata:
  name: dispatch-work-order
spec:
  description: Dispatch a field tech.
  brokeredToolClass: write
  parameters:
    type: object
    properties:
      incident:
        type: string
    required: [incident]
  http:
    url: http://tool.default.svc.cluster.local
    method: POST
    authSecretRef:
      name: do-not-export
      key: token
`
	testTelemetryToolCRD = `apiVersion: core.orka.ai/v1alpha1
kind: Tool
metadata:
  name: check-network-telemetry
spec:
  description: Read telemetry.
  brokeredToolClass: read
  parameters:
    type: object
    properties:
      site:
        type: string
  http:
    url: http://tool.default.svc.cluster.local
    method: POST
    headers:
      Authorization: do-not-export
`
)

func writeTestFile(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func toolDocs(t *testing.T, content string) []any {
	t.Helper()
	docs, err := decodeJSONDocuments([]byte(content))
	if err != nil {
		t.Fatalf("decodeJSONDocuments() error = %v", err)
	}
	return docs
}

func toolCRD(name, class, extraSpec string) string {
	doc := "apiVersion: core.orka.ai/v1alpha1\nkind: Tool\nmetadata:\n  name: " + name + "\nspec:\n  description: Tool " + name + ".\n"
	if class != "" {
		doc += "  brokeredToolClass: " + class + "\n"
	}
	return doc + "  parameters:\n    type: object\n" + extraSpec
}

func TestOrkaBrokeredToolsUsesCanonicalClassAndSortsByName(t *testing.T) {
	docs := toolDocs(t, toolCRD("read-telemetry", "read", "")+"---\n"+toolCRD("dispatch-work-order", "write", "")+"---\n"+toolCRD("coordinate-response", "coordination", ""))

	tools, err := OrkaBrokeredTools(docs, false)
	if err != nil {
		t.Fatalf("OrkaBrokeredTools() error = %v", err)
	}
	var got []string
	for _, tool := range tools {
		got = append(got, tool.Name+"="+tool.BrokeredClass)
	}
	if want := "coordinate-response=coordination,dispatch-work-order=write,read-telemetry=read"; strings.Join(got, ",") != want {
		t.Fatalf("tools = %v, want %s", got, want)
	}
}

func TestOrkaBrokeredToolsRejectsInvalidResources(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content string
		want    string
	}{
		{"api version", strings.Replace(toolCRD("read-telemetry", "read", ""), "v1alpha1", "v1", 1), "unsupported Orka Tool GVK in document 0: apiVersion=core.orka.ai/v1, kind=Tool"},
		{"kind", strings.Replace(toolCRD("runtime", "read", ""), "kind: Tool", "kind: AgentRuntime", 1), "kind=AgentRuntime; expected apiVersion=core.orka.ai/v1alpha1, kind=Tool"},
		{"non-object document", "---\nnot-a-resource\n---\n" + toolCRD("read-telemetry", "read", ""), "document 0: Orka Tool CRD must be an object"},
		{"unknown class", toolCRD("admin-tool", "admin", ""), `unsupported spec.brokeredToolClass "admin"; expected read, write, coordination`},
		{"no brokered tools", toolCRD("local-only-tool", "", "  brokeredClass: read\n"), "no brokered Orka Tool CRDs found; set spec.brokeredToolClass"},
		{"non-object parameters", strings.Replace(toolCRD("read-telemetry", "read", ""), "  parameters:\n    type: object\n", "  parameters: [type]\n", 1), "parameters schema must be an object"},
		{"unsafe schema", toolCRD("bad", "read", "    properties:\n      tokenValue:\n        type: string\n"), `tool CRD "bad": brokeredTool.parameters.properties.tokenValue is not safe`},
		{"duplicate names", toolCRD("read-telemetry", "read", "") + "---\n" + toolCRD("read-telemetry", "read", ""), "duplicate brokered tool names: read-telemetry"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := OrkaBrokeredTools(toolDocs(t, tc.content), true)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("OrkaBrokeredTools() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestOrkaBrokeredToolsSkipsUnclassifiedTools(t *testing.T) {
	tools, err := OrkaBrokeredTools(toolDocs(t, toolCRD("local-only-tool", "", "")+"---\n"+toolCRD("read-telemetry", "read", "")), false)
	if err != nil {
		t.Fatalf("OrkaBrokeredTools() error = %v", err)
	}
	if len(tools) != 1 || tools[0].Name != "read-telemetry" {
		t.Fatalf("tools = %+v", tools)
	}
}

func TestOrkaBrokeredToolsNeverLetsLegacyAliasesOverrideCanonicalFields(t *testing.T) {
	docs := toolDocs(t, `apiVersion: core.orka.ai/v1alpha1
kind: Tool
metadata:
  name: canonical-tool
spec:
  name: legacy-tool
  description: Canonical description.
  summary: Legacy description.
  brokeredToolClass: write
  brokeredClass: read
  parameters: {type: object, properties: {canonical: {type: string}}}
  inputSchema: {type: object, properties: {legacy: {type: string}}}
`)

	tools, err := OrkaBrokeredTools(docs, false)
	if err != nil {
		t.Fatalf("OrkaBrokeredTools() error = %v", err)
	}
	properties, _ := tools[0].Parameters["properties"].(map[string]any)
	if tools[0].Name != "canonical-tool" || tools[0].Description != "Canonical description." || tools[0].BrokeredClass != "write" || properties["canonical"] == nil || properties["legacy"] != nil {
		t.Fatalf("tool = %+v", tools[0])
	}
}

func TestOrkaBrokeredToolsDefaultsParametersToObject(t *testing.T) {
	docs := toolDocs(t, "apiVersion: core.orka.ai/v1alpha1\nkind: Tool\nmetadata: {name: ping}\nspec: {description: Ping., brokeredToolClass: read}\n")

	tools, err := OrkaBrokeredTools(docs, false)
	if err != nil {
		t.Fatalf("OrkaBrokeredTools() error = %v", err)
	}
	if tools[0].Parameters[schemaTypeKey] != schemaTypeObject || len(tools[0].Parameters) != 1 {
		t.Fatalf("parameters = %#v", tools[0].Parameters)
	}
}

func TestBrokeredToolsFromFilesRendersSchemaOnlyFragmentWithDigests(t *testing.T) {
	path := writeTestFile(t, "tools.yaml", "---\n"+testDispatchToolCRD+"---\n"+testTelemetryToolCRD)

	out, err := BrokeredToolsFromFiles([]string{path}, true, false)
	if err != nil {
		t.Fatalf("BrokeredToolsFromFiles() error = %v", err)
	}
	for _, leaked := range []string{"url", "http", "secretRef", "Authorization", "do-not-export"} {
		if strings.Contains(string(out), leaked) {
			t.Fatalf("fragment leaked %q:\n%s", leaked, out)
		}
	}
	var got struct {
		BrokeredTools []config.BrokeredTool `yaml:"brokeredTools"`
	}
	if err := yaml.UnmarshalWithOptions(out, &got, yaml.Strict()); err != nil {
		t.Fatalf("fragment is not a brokeredTools mapping: %v\n%s", err, out)
	}
	if len(got.BrokeredTools) != 2 || got.BrokeredTools[0].Name != "check-network-telemetry" || got.BrokeredTools[1].Name != "dispatch-work-order" {
		t.Fatalf("fragment tools = %+v", got.BrokeredTools)
	}
	want, err := config.BrokeredToolSchemaDigest(config.BrokeredTool{
		Name:          "check-network-telemetry",
		Description:   "Read telemetry.",
		BrokeredClass: config.BrokeredClassRead,
		Parameters:    map[string]any{schemaTypeKey: schemaTypeObject, "properties": map[string]any{"site": map[string]any{schemaTypeKey: schemaTypeString}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.BrokeredTools[0].SchemaDigest != want {
		t.Fatalf("schemaDigest = %q, want %q", got.BrokeredTools[0].SchemaDigest, want)
	}
	if err := config.ValidateBrokeredTool(got.BrokeredTools[1]); err != nil {
		t.Fatalf("rendered entry does not validate with its digest: %v", err)
	}
}

func TestBrokeredToolsFromFilesCanOmitDigestAndKey(t *testing.T) {
	path := writeTestFile(t, "tools.yaml", testDispatchToolCRD)

	out, err := BrokeredToolsFromFiles([]string{path}, false, true)
	if err != nil {
		t.Fatalf("BrokeredToolsFromFiles() error = %v", err)
	}
	if strings.Contains(string(out), "schemaDigest") || strings.Contains(string(out), "brokeredTools") || !strings.HasPrefix(string(out), "- name: \"dispatch-work-order\"") {
		t.Fatalf("bare fragment =\n%s", out)
	}
}

func TestBrokeredToolsFromFilesKeepsNumbersExact(t *testing.T) {
	precise := writeTestFile(t, "precise.yaml", toolCRD("precise-number", "read", "    properties:\n      identifier:\n        type: integer\n        default: 9007199254740993.0\n"))
	out, err := BrokeredToolsFromFiles([]string{precise}, true, false)
	if err != nil {
		t.Fatalf("BrokeredToolsFromFiles() error = %v", err)
	}
	if !strings.Contains(string(out), `"default": 9007199254740993`+"\n") {
		t.Fatalf("high-precision integral float was not kept exact:\n%s", out)
	}

	lossy := writeTestFile(t, "lossy.yaml", toolCRD("lossy-number", "write", "    properties:\n      ratio:\n        default: 0.100000000000000005\n"))
	if _, err := BrokeredToolsFromFiles([]string{lossy}, false, false); err == nil || !strings.Contains(err.Error(), "cannot be represented exactly") {
		t.Fatalf("BrokeredToolsFromFiles() error = %v, want lossy number rejection", err)
	}
}

func TestBrokeredToolsFromFilesRejectsDuplicateNamesAcrossFiles(t *testing.T) {
	first := writeTestFile(t, "first.yaml", testDispatchToolCRD)
	second := writeTestFile(t, "second.yaml", testDispatchToolCRD)

	if _, err := BrokeredToolsFromFiles([]string{first, second}, true, false); err == nil || !strings.Contains(err.Error(), "duplicate brokered tool names: dispatch-work-order") {
		t.Fatalf("BrokeredToolsFromFiles() error = %v", err)
	}
}

func TestRunCLIBrokeredToolsWritesOutputFile(t *testing.T) {
	src := writeTestFile(t, "tools.yaml", testDispatchToolCRD+"---\n"+testTelemetryToolCRD)
	out := filepath.Join(t.TempDir(), "brokered-tools.yaml")
	var stdout, stderr bytes.Buffer

	// Flags may follow the file arguments, as with the former Python CLI.
	code := RunCLI([]string{testTargetFlag, testBrokeredToolsTarget, src, "--no-digest", "-o", out}, &stdout, &stderr)

	if code != 0 || stdout.Len() != 0 {
		t.Fatalf("RunCLI() code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	rendered, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(rendered), "schemaDigest") || !strings.Contains(string(rendered), "check-network-telemetry") {
		t.Fatalf("output file =\n%s", rendered)
	}
}

func TestRunCLIBrokeredToolsReportsErrorsWithoutWritingOutput(t *testing.T) {
	src := writeTestFile(t, "tools.yaml", toolCRD("local-only-tool", "", ""))
	out := filepath.Join(t.TempDir(), "brokered-tools.yaml")
	var stdout, stderr bytes.Buffer

	code := RunCLI([]string{testTargetFlag, testBrokeredToolsTarget, "--output", out, src}, &stdout, &stderr)

	if code != 2 || stdout.Len() != 0 || !strings.Contains(stderr.String(), "agentkit render: no brokered Orka Tool CRDs") {
		t.Fatalf("RunCLI() code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if _, err := os.Stat(out); !os.IsNotExist(err) {
		t.Fatalf("output file exists after failure: %v", err)
	}
}

func TestRunCLIRejectsFlagsAndArgumentsForOtherTargets(t *testing.T) {
	src := writeTestFile(t, "tools.yaml", testDispatchToolCRD)
	for _, tc := range []struct {
		args []string
		want string
	}{
		{[]string{testTargetFlag, testBrokeredToolsTarget, testNameFlag, testRuntimeName, src}, "does not accept --name"},
		{[]string{testTargetFlag, TargetOrkaAgentRuntime, "--bare", testNameFlag, testRuntimeName, testExternalEndpointFlag, testExampleEndpoint}, "does not accept --bare"},
		{[]string{testTargetFlag, TargetOrkaAgentRuntime, testNameFlag, testRuntimeName, testExternalEndpointFlag, testExampleEndpoint, src}, "takes no file arguments"},
		{[]string{testTargetFlag, testBrokeredToolsTarget}, "at least one Orka Tool CRD file is required"},
		{[]string{testTargetFlag, TargetFoundryHostedAgent, src, src}, "takes exactly one profile file"},
	} {
		var stdout, stderr bytes.Buffer
		if code := RunCLI(tc.args, &stdout, &stderr); code != 2 || stdout.Len() != 0 || !strings.Contains(stderr.String(), tc.want) {
			t.Fatalf("RunCLI(%v) code=%d stdout=%q stderr=%q, want %q", tc.args, code, stdout.String(), stderr.String(), tc.want)
		}
	}
}
