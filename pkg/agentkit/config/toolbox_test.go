package config

import (
	"strings"
	"testing"

	"github.com/containerd/platforms"
)

// Fixture lines that TestToolboxValidateRejects rewrites.
const (
	fixtureName     = "name: yq-jq"
	fixtureARM64Key = "linux/arm64: {url"
	fixtureYQURL    = "url: https://example.com/yq_linux_amd64"
	fixtureExtract  = "extract: linux-amd64/helm"
)

const (
	toolboxTestSHA256A = "1111111111111111111111111111111111111111111111111111111111111111"
	toolboxTestSHA256B = "2222222222222222222222222222222222222222222222222222222222222222"
	toolboxTestDigest  = "sha256:3333333333333333333333333333333333333333333333333333333333333333"
)

func validToolboxYAML() string {
	return `#syntax=ghcr.io/orka-agents/agentkit/agentkit:test
apiVersion: v1alpha1
kind: Toolbox
metadata:
  name: yq-jq
  labels:
    com.example.team: platform
spec:
  tools:
    - name: yq
      description: YAML processor
      source:
        download:
          linux/amd64: {url: https://example.com/yq_linux_amd64, sha256: "` + toolboxTestSHA256A + `"}
          linux/arm64: {url: https://example.com/yq_linux_arm64, sha256: "` + toolboxTestSHA256B + `"}
    - name: helm
      source:
        download:
          linux/amd64: {url: https://example.com/helm-linux-amd64.tar.gz, sha256: "` + toolboxTestSHA256A + `", extract: linux-amd64/helm}
    - name: kubectl
      source:
        image: registry.k8s.io/kubectl@` + toolboxTestDigest + `
        path: /bin/kubectl
  verify:
    runtimeImage: ghcr.io/orka-agents/orka/acp-codex-runtime@` + toolboxTestDigest + `
    commands:
      - yq --version
      - helm version --short
`
}

func loadToolbox(t *testing.T, in string) *ToolboxConfig {
	t.Helper()
	doc, err := NewDocumentFromBytes([]byte(in))
	if err != nil {
		t.Fatalf("NewDocumentFromBytes: %v", err)
	}
	if doc.Toolbox == nil || doc.Agent != nil {
		t.Fatalf("document = %+v, want only a toolbox", doc)
	}
	return doc.Toolbox
}

func TestToolboxParsesEveryField(t *testing.T) {
	cfg := loadToolbox(t, validToolboxYAML())
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}

	if cfg.MountPath() != "/opt/yq-jq" {
		t.Fatalf("MountPath = %q", cfg.MountPath())
	}
	if cfg.Metadata.Labels["com.example.team"] != "platform" {
		t.Fatalf("labels = %v", cfg.Metadata.Labels)
	}
	if len(cfg.Spec.Tools) != 3 {
		t.Fatalf("tools = %+v", cfg.Spec.Tools)
	}
	yq, helm, kubectl := cfg.Spec.Tools[0], cfg.Spec.Tools[1], cfg.Spec.Tools[2]
	if yq.Name != "yq" || yq.Description != "YAML processor" {
		t.Fatalf("yq = %+v", yq)
	}
	arm, ok := yq.Source.DownloadFor(platforms.MustParse("linux/aarch64"))
	if !ok || arm.URL != "https://example.com/yq_linux_arm64" || arm.SHA256 != toolboxTestSHA256B || arm.Extract != "" {
		t.Fatalf("yq arm64 download = %+v, %v", arm, ok)
	}
	if got := arm.Digest().String(); got != "sha256:"+toolboxTestSHA256B {
		t.Fatalf("Digest = %q", got)
	}
	if got := strings.Join(yq.Source.DownloadPlatforms(), ","); got != "linux/amd64,linux/arm64" {
		t.Fatalf("DownloadPlatforms = %q", got)
	}
	if dl, ok := helm.Source.DownloadFor(platforms.MustParse("linux/amd64")); !ok || dl.Extract != "linux-amd64/helm" {
		t.Fatalf("helm download = %+v, %v", dl, ok)
	}
	if _, ok := helm.Source.DownloadFor(platforms.MustParse("linux/arm64")); ok {
		t.Fatal("helm has no arm64 download")
	}
	if kubectl.Source.Image != "registry.k8s.io/kubectl@"+toolboxTestDigest || kubectl.Source.Path != "/bin/kubectl" {
		t.Fatalf("kubectl = %+v", kubectl)
	}
	if v := cfg.Spec.Verify; v == nil || !strings.HasSuffix(v.RuntimeImage, toolboxTestDigest) || len(v.Commands) != 2 {
		t.Fatalf("verify = %+v", v)
	}
}

func TestToolboxVerifyIsOptional(t *testing.T) {
	in, _, _ := strings.Cut(validToolboxYAML(), "  verify:")
	if err := loadToolbox(t, in).Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}

func TestToolboxStrictParseRejectsUnknownFields(t *testing.T) {
	for name, replace := range map[string][2]string{
		"spec":     {"  tools:", "  tool:"},
		"tool":     {"      description:", "      summary:"},
		"source":   {"        path:", "        file:"},
		"download": {`, extract: linux-amd64/helm}`, `, unpack: linux-amd64/helm}`},
		"verify":   {"    commands:", "    command:"},
	} {
		t.Run(name, func(t *testing.T) {
			in := strings.Replace(validToolboxYAML(), replace[0], replace[1], 1)
			if _, err := NewDocumentFromBytes([]byte(in)); err == nil {
				t.Fatal("expected strict-mode error")
			}
		})
	}
}

func TestToolboxValidateRejects(t *testing.T) {
	tests := []struct {
		name    string
		old     string
		new     string
		wantErr string
	}{
		{"wrong apiVersion", "apiVersion: v1alpha1", "apiVersion: v1", `apiVersion "v1" is not supported`},
		{"missing name", "  name: yq-jq\n", "", "metadata.name is required"},
		{"uppercase name", fixtureName, "name: YQ", `metadata.name "YQ" must be`},
		{"name with dot", fixtureName, "name: yq.jq", `metadata.name "yq.jq" must be`},
		{"name ends with dash", fixtureName, "name: yq-", `metadata.name "yq-" must be`},
		{"name too long", fixtureName, "name: " + strings.Repeat("a", 64), "must be 1-63"},
		{"reserved name", fixtureName, "name: codex", "reserved: Orka's runtime images already use /opt/codex"},
		{"reserved label", "com.example.team: platform", "ai.orka.toolbox: forged", `uses reserved AgentKit/Orka control-plane label namespace "ai.orka"`},
		{"bad tool name", "- name: yq\n", "- name: Y_Q\n", `spec.tools[0].name "Y_Q" must be`},
		{"duplicate tool name", "- name: helm\n", "- name: yq\n", `spec.tools[1].name "yq" duplicates spec.tools[0].name`},
		{"missing tool name", "- name: helm\n      source:", "- source:", "spec.tools[1].name is required"},
		{"bare os platform", fixtureARM64Key, "linux: {url", `spec.tools[0].source.download["linux"]: platform must be <os>/<arch>`},
		{"non-linux platform", fixtureARM64Key, "windows/arm64: {url", `download["windows/arm64"]: platform must be one of linux/amd64, linux/arm64`},
		{"unsupported architecture", fixtureARM64Key, "linux/riscv64: {url", `download["linux/riscv64"]: platform must be one of linux/amd64, linux/arm64`},
		{"duplicate platform", fixtureARM64Key, "linux/x86_64: {url", `download["linux/x86_64"] names the same platform as "linux/amd64"`},
		{"http url", fixtureYQURL, "url: http://example.com/yq_linux_amd64", ".url must be an absolute https URL"},
		{"relative url", fixtureYQURL, "url: /yq_linux_amd64", ".url must be an absolute https URL"},
		{"credential url", fixtureYQURL, "url: https://user:pass@example.com/yq", ".url must not embed credentials"},
		{"missing sha256", `, sha256: "` + toolboxTestSHA256A + `"}`, "}", ".sha256 must be 64 lowercase hex characters"},
		{"uppercase sha256", `sha256: "` + toolboxTestSHA256B + `"`, `sha256: "` + strings.ToUpper(strings.Repeat("ab", 32)) + `"`, ".sha256 must be 64 lowercase hex characters"},
		{"absolute extract", fixtureExtract, "extract: /linux-amd64/helm", `.extract "/linux-amd64/helm" must be a clean relative file path`},
		{"dotdot extract", fixtureExtract, "extract: ../helm", `.extract "../helm" must be a clean relative file path`},
		{"unclean extract", fixtureExtract, "extract: ./linux-amd64/helm", `.extract "./linux-amd64/helm" must be a clean relative file path`},
		{"dot extract", fixtureExtract, `extract: "."`, `.extract "." must be a clean relative file path`},
		{"tagged image", "registry.k8s.io/kubectl@" + toolboxTestDigest, "registry.k8s.io/kubectl:v1.34.0", `spec.tools[2].source.image "registry.k8s.io/kubectl:v1.34.0" must be pinned by digest`},
		{"invalid image", "registry.k8s.io/kubectl@" + toolboxTestDigest, "Registry/Kubectl@" + toolboxTestDigest, "is not a valid image reference"},
		{"image without path", "        path: /bin/kubectl\n", "", "spec.tools[2].source.path is required with spec.tools[2].source.image"},
		{"relative image path", "path: /bin/kubectl", "path: bin/kubectl", `source.path "bin/kubectl" must be a clean absolute file path`},
		{"root image path", "path: /bin/kubectl", "path: /", `source.path "/" must be a clean absolute file path`},
		{"both sources", "        path: /bin/kubectl\n", "        path: /bin/kubectl\n        download:\n          linux/amd64: {url: https://example.com/k, sha256: \"" + toolboxTestSHA256A + "\"}\n", "spec.tools[2].source must set exactly one of download or image"},
		{"no source", "        image: registry.k8s.io/kubectl@" + toolboxTestDigest + "\n        path: /bin/kubectl\n", "        download: {}\n", "spec.tools[2].source must set download or image"},
		{"path without image", "linux-amd64/helm}\n", "linux-amd64/helm}\n        path: /bin/helm\n", "spec.tools[1].source.path is only valid with spec.tools[1].source.image"},
		{"tagged runtime image", "acp-codex-runtime@" + toolboxTestDigest, "acp-codex-runtime:latest", `spec.verify.runtimeImage "ghcr.io/orka-agents/orka/acp-codex-runtime:latest" must be pinned by digest`},
		{"missing runtime image", "    runtimeImage: ghcr.io/orka-agents/orka/acp-codex-runtime@" + toolboxTestDigest + "\n", "", "spec.verify.runtimeImage is required"},
		{"no verify commands", "    commands:\n      - yq --version\n      - helm version --short\n", "    commands: []\n", "spec.verify.commands must list at least one command"},
		{"blank verify command", "      - yq --version", `      - "  "`, "spec.verify.commands[0] must not be empty"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := validToolboxYAML()
			if !strings.Contains(in, tt.old) {
				t.Fatalf("fixture does not contain %q", tt.old)
			}
			cfg := loadToolbox(t, strings.Replace(in, tt.old, tt.new, 1))
			err := cfg.Validate()
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Validate error = %v, want it to contain %q", err, tt.wantErr)
			}
		})
	}
}

func TestToolboxValidateRequiresTools(t *testing.T) {
	cfg := loadToolbox(t, "apiVersion: v1alpha1\nkind: Toolbox\nmetadata:\n  name: empty\nspec:\n  tools: []\n")
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "spec.tools must list at least one tool") {
		t.Fatalf("Validate error = %v", err)
	}
}

func TestDocumentDispatchesOnKind(t *testing.T) {
	doc, err := NewDocumentFromBytes([]byte("apiVersion: v1alpha1\nkind: Agent\nmetadata:\n  name: hello\n"))
	if err != nil || doc.Agent == nil || doc.Toolbox != nil {
		t.Fatalf("agent document = %+v, %v", doc, err)
	}

	for in, want := range map[string]string{
		"apiVersion: v1alpha1\n":             "kind is not defined; expected `kind: Agent` or `kind: Toolbox`",
		"apiVersion: v1alpha1\nkind: Crew\n": `kind "Crew" is not supported; expected ` + "`kind: Agent` or `kind: Toolbox`",
	} {
		if _, err := NewDocumentFromBytes([]byte(in)); err == nil || err.Error() != want {
			t.Fatalf("NewDocumentFromBytes(%q) error = %v, want %q", in, err, want)
		}
	}
}

func TestAgentLoaderRejectsToolbox(t *testing.T) {
	_, err := NewFromBytes([]byte(validToolboxYAML()))
	if err == nil || !strings.Contains(err.Error(), `kind "Toolbox" is not supported; expected `+"`kind: Agent`") {
		t.Fatalf("NewFromBytes error = %v", err)
	}
}

func TestToolboxAcceptsExtensionFields(t *testing.T) {
	in := strings.Replace(validToolboxYAML(), "spec:\n", `x-yq-sha256: &yq-sha "`+toolboxTestSHA256A+`"
spec:
  x-note: shared checksums live in x- fields
`, 1)
	in = strings.Replace(in, `sha256: "`+toolboxTestSHA256A+`"}`+"\n          linux/arm64", `sha256: *yq-sha}`+"\n          linux/arm64", 1)
	cfg := loadToolbox(t, in)
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if got := cfg.Spec.Tools[0].Source.Download["linux/amd64"].SHA256; got != toolboxTestSHA256A {
		t.Fatalf("anchored sha256 = %q", got)
	}
}
