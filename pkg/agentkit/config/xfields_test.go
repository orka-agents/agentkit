package config

import "testing"

// baseAgentWithXFields is a minimal valid agentkitfile with an x- field at
// every level the issue calls out: top level, model, tools[0], expose, and
// context.providers[0]. It also exercises the mapping form of `instructions`
// (itself strict-decoded by Source.UnmarshalYAML) with its own x- field, and
// an anchor defined inside a top-level x- field and reused in tools[].env.
const baseAgentWithXFields = `apiVersion: v1alpha1
kind: Agent
x-owner: platform-team
x-proxy-env: &proxy-env [HTTP_PROXY, HTTPS_PROXY, NO_PROXY]
metadata:
  name: x-fields-agent
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
  x-model-note: internal
instructions:
  inline: hi
  x-reviewed-by: platform-team
tools:
  - name: fetch
    command: ["mcp-server-fetch"]
    env: *proxy-env
    x-tool-note: shared across tools
context:
  providers:
    - type: search
      x-provider-note: internal
expose:
  openai: true
  x-expose-note: internal
`

func TestXFieldsAcceptedAtEveryLevel(t *testing.T) {
	cfg, err := NewFromBytes([]byte(baseAgentWithXFields))
	if err != nil {
		t.Fatalf("expected x- fields to be accepted and ignored, got: %v", err)
	}
	if cfg.Metadata.Name != "x-fields-agent" {
		t.Fatalf("expected the rest of the config to decode normally, got metadata.name=%q", cfg.Metadata.Name)
	}
}

func TestXFieldsUpperCasePrefixAccepted(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
X-Upper: anything
metadata:
  name: x
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
expose:
  openai: true
`)
	if _, err := NewFromBytes(in); err != nil {
		t.Fatalf("expected X-Upper to be accepted, got: %v", err)
	}
}

// TestXFieldAnchorReusedInToolsEnv proves goccy resolves an anchor defined
// inside an x- field (itself skipped by AllowFieldPrefixes) and reused
// elsewhere in the file, rather than the anchor silently vanishing along with
// the field that defines it.
func TestXFieldAnchorReusedInToolsEnv(t *testing.T) {
	cfg, err := NewFromBytes([]byte(baseAgentWithXFields))
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(cfg.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(cfg.Tools))
	}
	want := []string{"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"}
	got := cfg.Tools[0].Env
	if len(got) != len(want) {
		t.Fatalf("expected env %v (anchor resolved), got %v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expected env %v (anchor resolved), got %v", want, got)
		}
	}
}

func TestXFieldsStillRejectsMisspelledTopLevelField(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: x
toolz:
  - name: fetch
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
expose:
  openai: true
`)
	if _, err := NewFromBytes(in); err == nil {
		t.Fatal("expected strict-mode error for misspelled `toolz`, got nil")
	}
}

func TestXFieldsStillRejectsMisspelledToolField(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: x
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
tools:
  - name: fetch
    comand: ["mcp-server-fetch"]
expose:
  openai: true
`)
	if _, err := NewFromBytes(in); err == nil {
		t.Fatal("expected strict-mode error for misspelled `comand`, got nil")
	}
}

func TestXFieldsRejectsNameWithoutDash(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
xfoo: anything
metadata:
  name: x
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
expose:
  openai: true
`)
	if _, err := NewFromBytes(in); err == nil {
		t.Fatal("expected strict-mode error for `xfoo` (no dash, not an extension field), got nil")
	}
}

func TestXFieldsRejectsDashBeforeX(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
-x: anything
metadata:
  name: x
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
expose:
  openai: true
`)
	if _, err := NewFromBytes(in); err == nil {
		t.Fatal("expected strict-mode error for `-x` (dash first, not an extension field), got nil")
	}
}

// TestXFieldInMetadataLabelsIsKeptAsOrdinaryKey proves the extension-prefix
// exception only applies to real Agentkitfile fields, not to keys inside a
// free-form map that already accepts any key.
func TestXFieldInMetadataLabelsIsKeptAsOrdinaryKey(t *testing.T) {
	in := []byte(`apiVersion: v1alpha1
kind: Agent
metadata:
  name: x
  labels:
    x-team: platform
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: hi
expose:
  openai: true
`)
	cfg, err := NewFromBytes(in)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if got, want := cfg.Metadata.Labels["x-team"], "platform"; got != want {
		t.Fatalf("expected metadata.labels[\"x-team\"]=%q (kept as an ordinary key), got %q", want, got)
	}
}
