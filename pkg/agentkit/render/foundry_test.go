package render

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/goccy/go-yaml"
)

const testFoundryProfile = `target: foundry-hosted-agent
name: agentkit-agent
image: docker.io/example/agentkit-foundry-wrapper:latest
foundryWrapper: true
`

func TestFoundryHostedAgentRendersExampleProfile(t *testing.T) {
	got, err := FoundryHostedAgentFromFile(filepath.Join("..", "..", "..", "deploy", "foundry", "agentkit.foundry.yaml.example"))
	if err != nil {
		t.Fatalf("FoundryHostedAgentFromFile() error = %v", err)
	}
	want := `kind: hosted
name: agentkit-agent
image: docker.io/example/agentkit-foundry-wrapper:latest
protocols:
- protocol: invocations
  version: 1.0.0
- protocol: responses
  version: 1.0.0
resources:
  cpu: "0.25"
  memory: 0.5Gi
environment_variables:
- name: TOOLBOX_ENDPOINT
  value: ${TOOLBOX_ENDPOINT}
`
	if string(got) != want {
		t.Fatalf("hosted agent mismatch\n got:\n%s\nwant:\n%s", got, want)
	}
}

func TestFoundryHostedAgentMapsProfileShapes(t *testing.T) {
	got, err := FoundryHostedAgent([]byte(`agentName: from-agent-name
image: example.invalid/agent:1
imageKind: foundry-wrapper
protocols:
  - responses
  - protocol: invocations
    version: 2.0.0
resources:
  memory: 1Gi
  cpu: 1
environment:
  - name: TOOLBOX_ENDPOINT
  - name: FEATURE_COUNT
    value: "123"
`))
	if err != nil {
		t.Fatalf("FoundryHostedAgent() error = %v", err)
	}
	var agent struct {
		Name      string `yaml:"name"`
		Protocols []struct {
			Protocol string `yaml:"protocol"`
			Version  string `yaml:"version"`
		} `yaml:"protocols"`
		Resources            yaml.MapSlice   `yaml:"resources"`
		EnvironmentVariables []foundryEnvVar `yaml:"environment_variables"`
	}
	if err := yaml.Unmarshal(got, &agent); err != nil {
		t.Fatalf("rendered YAML is invalid: %v\n%s", err, got)
	}
	if agent.Name != "from-agent-name" || len(agent.Protocols) != 2 || agent.Protocols[0].Version != foundryProtocolVersion || agent.Protocols[1].Protocol != "invocations" || agent.Protocols[1].Version != "2.0.0" {
		t.Fatalf("hosted agent =\n%s", got)
	}
	if len(agent.Resources) != 2 || agent.Resources[0].Key != "memory" {
		t.Fatalf("resources lost profile order:\n%s", got)
	}
	wantEnv := []foundryEnvVar{{Name: "TOOLBOX_ENDPOINT", Value: "${TOOLBOX_ENDPOINT}"}, {Name: "FEATURE_COUNT", Value: "123"}}
	if len(agent.EnvironmentVariables) != 2 || agent.EnvironmentVariables[0] != wantEnv[0] || agent.EnvironmentVariables[1] != wantEnv[1] {
		t.Fatalf("environment_variables = %+v, want %+v", agent.EnvironmentVariables, wantEnv)
	}
	if strings.Contains(string(got), "environmentVariables") {
		t.Fatalf("rendered camelCase environmentVariables:\n%s", got)
	}
}

func TestFoundryHostedAgentValidation(t *testing.T) {
	for _, tc := range []struct {
		name    string
		profile string
		want    string
	}{
		{"target", "target: orka-agentruntime\n" + strings.TrimPrefix(testFoundryProfile, "target: foundry-hosted-agent\n"), "profile target must be foundry-hosted-agent"},
		{"name", strings.Replace(testFoundryProfile, "name: agentkit-agent\n", "", 1), "profile must set name"},
		{"image", strings.Replace(testFoundryProfile, "image: docker.io/example/agentkit-foundry-wrapper:latest\n", "", 1), "profile must set image"},
		{"wrapper", strings.Replace(testFoundryProfile, "foundryWrapper: true\n", "", 1), "must be a Foundry protocol wrapper image"},
		{"env name", testFoundryProfile + "env:\n  lower_case: x\n", `env name "lower_case" must match [A-Z0-9_]+`},
		{"env list", testFoundryProfile + "env:\n  - VALUE_ONLY\n", "env list entries must be mappings with name/value"},
		{"env shape", testFoundryProfile + "env: TOOLBOX_ENDPOINT\n", "env must be a mapping or list"},
		{"protocol", testFoundryProfile + "protocols:\n  - version: 1.0.0\n", "invalid protocol entry"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := FoundryHostedAgent([]byte(tc.profile)); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("FoundryHostedAgent() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestRunCLIRendersFoundryHostedAgent(t *testing.T) {
	profile := filepath.Join(t.TempDir(), "profile.yaml")
	if err := os.WriteFile(profile, []byte(testFoundryProfile), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer

	code := RunCLI([]string{testTargetFlag, TargetFoundryHostedAgent, profile}, &stdout, &stderr)

	if code != 0 || !strings.HasPrefix(stdout.String(), "kind: hosted\nname: agentkit-agent\n") {
		t.Fatalf("RunCLI() code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
}
