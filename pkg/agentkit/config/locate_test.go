package config

import (
	"strings"
	"testing"

	"github.com/goccy/go-yaml/parser"
)

// locateFixture has a known line for every path the tests look up.
const locateFixture = `apiVersion: v1alpha1
kind: Agent
metadata:
  name: hello
  labels:
    example.com/team: core
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
instructions: {file: ./prompt.md}
tools:
  - name: search
    command: [search-mcp]
  - name: fetch
    command: [fetch-mcp]
    urlEnv: FETCH_URL
    headers:
      - name: X-Key
        valueEnv: KEY
expose:
  openai: true
`

func locateIn(t *testing.T, src, path string) Span {
	t.Helper()
	file, err := parser.ParseBytes([]byte(src), 0)
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return locatePath(file, path)
}

func TestLocatePath(t *testing.T) {
	cases := []struct {
		name string
		path string
		line int
	}{
		{"top-level key", "$.kind", 2},
		{"nested mapping key", "$.model.baseURL", 9},
		{"sequence item", "$.tools[1]", 14},
		{"field in a sequence item", "$.tools[1].urlEnv", 16},
		{"nested lists", "$.tools[1].headers[0].valueEnv", 19},
		{"flow mapping", "$.instructions.file", 10},
		{"quoted key with path syntax", "$.metadata.labels.'example.com/team'", 6},
		{"missing key falls back to parent", "$.model.name", 7},
		{"missing nested key falls back to nearest parent", "$.model.auth.type", 7},
		{"index out of range falls back to the list", "$.tools[5].name", 11},
		{"path not in the file falls back to line 1", "$.observability.otel", 1},
		{"empty path is line 1", "", 1},
		{"malformed path is line 1", "tools[", 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := locateIn(t, locateFixture, tc.path).StartLine; got != tc.line {
				t.Fatalf("locate %q: line %d, want %d", tc.path, got, tc.line)
			}
		})
	}
}

func TestLocatePathSpanCoversWholeEntry(t *testing.T) {
	span := locateIn(t, locateFixture, "$.tools[1]")
	if span.StartLine != 14 || span.EndLine != 19 {
		t.Fatalf("tools[1] span = lines %d-%d, want 14-19", span.StartLine, span.EndLine)
	}
	if span := locateIn(t, locateFixture, "$.tools[1].urlEnv"); span.EndLine != 16 {
		t.Fatalf("scalar field span ends on line %d, want 16", span.EndLine)
	}
}

func TestLocatePathInstructionsForms(t *testing.T) {
	plain := "apiVersion: v1alpha1\nkind: Agent\ninstructions: |\n  Be helpful.\n"
	if got := locateIn(t, plain, "$.instructions").StartLine; got != 3 {
		t.Fatalf("plain-string instructions: line %d, want 3", got)
	}
	// A child path on a plain string falls back to the string's key.
	if got := locateIn(t, plain, "$.instructions.file").StartLine; got != 3 {
		t.Fatalf("instructions.file on a plain string: line %d, want 3", got)
	}

	block := "apiVersion: v1alpha1\nkind: Agent\ninstructions:\n  file: ./prompt.md\n"
	if got := locateIn(t, block, "$.instructions.file").StartLine; got != 4 {
		t.Fatalf("block-mapping instructions.file: line %d, want 4", got)
	}
}

func TestLocatePathThroughAnchorsAndAliases(t *testing.T) {
	src := `apiVersion: v1alpha1
kind: Agent
x-env: &shared-env [API_TOKEN]
model: &model
  name: gpt
tools:
  - name: search
    env: *shared-env
`
	if got := locateIn(t, src, "$.model.name").StartLine; got != 5 {
		t.Fatalf("field under an anchored mapping: line %d, want 5", got)
	}
	if got := locateIn(t, src, "$.tools[0].env").StartLine; got != 8 {
		t.Fatalf("aliased value: line %d, want 8", got)
	}
}

func TestParseYAMLPath(t *testing.T) {
	segs, ok := parseYAMLPath("$.tools[1].headers[0].'x.y'")
	if !ok || len(segs) != 5 {
		t.Fatalf("parse: ok=%v segs=%v", ok, segs)
	}
	if segs[1].index != 1 || !segs[1].isIndex || segs[4].key != "x.y" {
		t.Fatalf("unexpected segments %+v", segs)
	}
	for _, bad := range []string{"tools", "$..a", "$.a[x]", "$.a[-1]", "$.'open", `$."open\"`, `$."bad\q"`} {
		if _, ok := parseYAMLPath(bad); ok {
			t.Fatalf("parseYAMLPath(%q) accepted a malformed path", bad)
		}
	}
}

func TestYAMLPathKeyRoundTrips(t *testing.T) {
	for _, key := range []string{"team", "example.com/team", "has space", "", "ai.orka.owner's", `a."b\c`, "line\nbreak"} {
		segs, ok := parseYAMLPath(yamlPathKey("$.metadata.labels", key))
		if !ok || len(segs) != 3 || segs[2].key != key {
			t.Fatalf("key %q did not round-trip: ok=%v segs=%+v", key, ok, segs)
		}
	}
}

func TestLocateProblemsPreservesLabelKeys(t *testing.T) {
	src := `apiVersion: v1alpha1
kind: Agent
metadata:
  name: hello
  labels:
    ai.orka.owners: other
    ai.orka.owner's: original
`
	cfg, err := NewFromBytes([]byte(src))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	for _, lp := range LocateProblems(cfg.Validate(), []byte(src)) {
		if strings.HasPrefix(lp.Message, `metadata.labels["ai.orka.owner's"]`) {
			if lp.StartLine != 7 {
				t.Fatalf("apostrophe label: line %d (path %q), want 7", lp.StartLine, lp.Path)
			}
			return
		}
	}
	t.Fatal("no problem reported for reserved label containing an apostrophe")
}

func TestLocateProblemsHelperFields(t *testing.T) {
	type location struct {
		path string
		line int
	}
	cases := []struct {
		name string
		src  string
		want map[string]location
	}{
		{
			name: "tool fields and items",
			src: `tools:
  - name: search
    command: [search-mcp]
  - name: fetch
    type: grpc
    transport: grpc
    urlEnv: https://example.com/mcp
    env:
      - API_KEY
      - bad-key
    headers:
      - name: X-Key
        valueEnv: API_KEY
      - name: bad header
        value: sk-example
        valueEnv: bad-key
    auth:
      type: bearer
      tokenEnv: bad-key
      audience: invalid
`,
			want: map[string]location{
				"tools[1] (fetch): remote MCP tools must set type":      {"$.tools[1].type", 5},
				"tools[1] (fetch): remote MCP tools must set transport": {"$.tools[1].transport", 6},
				"tools[1] (fetch): urlEnv":                              {"$.tools[1].urlEnv", 7},
				"tools[1] (fetch): remote MCP tools must use headers":   {"$.tools[1].env", 8},
				"tools[1] (fetch): env entry":                           {"$.tools[1].env[1]", 10},
				"tools[1] (fetch).headers[1].name":                      {"$.tools[1].headers[1].name", 14},
				"tools[1] (fetch).headers[1] must set exactly one":      {"$.tools[1].headers[1]", 14},
				"tools[1] (fetch).headers[1].value looks":               {"$.tools[1].headers[1].value", 15},
				"tools[1] (fetch).headers[1].valueEnv":                  {"$.tools[1].headers[1].valueEnv", 16},
				"tools[1] (fetch).auth.tokenEnv":                        {"$.tools[1].auth.tokenEnv", 19},
				"tools[1] (fetch).auth.audience":                        {"$.tools[1].auth.audience", 20},
			},
		},
		{
			name: "model auth fields",
			src: `model:
  auth:
    type: workload-identity-token
    tokenEnv: API_KEY
    audience: ""
`,
			want: map[string]location{
				"model.auth.tokenEnv": {"$.model.auth.tokenEnv", 4},
				"model.auth.audience": {"$.model.auth.audience", 5},
			},
		},
		{
			name: "context auth fields",
			src: `context:
  providers:
    - name: docs
      type: search
      auth:
        type: bearer
        tokenEnv: bad-key
`,
			want: map[string]location{
				"context.providers[0].auth.type":     {"$.context.providers[0].auth.type", 6},
				"context.providers[0].auth.tokenEnv": {"$.context.providers[0].auth.tokenEnv", 7},
			},
		},
		{
			name: "brokered tool fields",
			src: `brokeredTools:
  - name: first
    description: Search documents
    brokeredClass: read
    parameters: {type: object}
  - name: bad!
    description: https://example.com
    brokeredClass: unknown
    parameters: null
    schemaDigest: invalid
`,
			want: map[string]location{
				"brokeredTools[1].name":          {"$.brokeredTools[1].name", 6},
				"brokeredTools[1].description":   {"$.brokeredTools[1].description", 7},
				"brokeredTools[1].brokeredClass": {"$.brokeredTools[1].brokeredClass", 8},
				"brokeredTools[1].parameters":    {"$.brokeredTools[1].parameters", 9},
				"brokeredTools[1].schemaDigest":  {"$.brokeredTools[1].schemaDigest", 10},
			},
		},
		{
			name: "observability fields",
			src: `observability:
  otel:
    endpointEnv: bad-key
  logs:
    levelEnv: bad-key
`,
			want: map[string]location{
				"observability.otel.endpointEnv": {"$.observability.otel.endpointEnv", 3},
				"observability.logs.levelEnv":    {"$.observability.logs.levelEnv", 5},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := "apiVersion: v1alpha1\nkind: Agent\n" + tc.src
			cfg, err := NewFromBytes([]byte(src))
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			verr := cfg.Validate()
			found := map[string]bool{}
			for _, lp := range LocateProblems(verr, []byte(src)) {
				for prefix, want := range tc.want {
					if strings.HasPrefix(lp.Message, prefix) {
						found[prefix] = true
						if lp.Path != want.path || lp.StartLine != want.line+2 {
							t.Errorf("%q: path %q line %d, want path %q line %d", lp.Message, lp.Path, lp.StartLine, want.path, want.line+2)
						}
					}
				}
			}
			for prefix := range tc.want {
				if !found[prefix] {
					t.Errorf("no problem starting with %q in:\n%v", prefix, verr)
				}
			}
		})
	}
}

// brokenAgentkitfile has one mistake per line listed in the test below.
const brokenAgentkitfile = `apiVersion: v1alpha1
kind: Agent
metadata:
  name: hello
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
instructions: Be helpful.
tools:
  - name: search
    type: grpc
    command: [search-mcp]
env:
  - name: OPENAI_API_KEY
  - name: my-key
context:
  providers:
    - name: docs
      type: search
      endpointEnv: SEARCH_ENDPOINT
expose:
  openai: true
  port: 70000
`

func TestLocateProblemsReportsTheLineOfEachMistake(t *testing.T) {
	cfg, err := NewFromBytes([]byte(brokenAgentkitfile))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	verr := cfg.Validate()
	if verr == nil {
		t.Fatal("expected validation errors")
	}
	want := map[string]int{
		// Missing fields point at their parent.
		"model.name is required":        5,
		"context.providers[0].indexEnv": 18,
		// Wrong values point at the field.
		`tools[0] (search): type "grpc" is not supported`: 11,
		`env[1].name "my-key" must be an env var NAME`:    15,
		"expose.port 70000 is out of range":               23,
	}
	found := map[string]bool{}
	for _, lp := range LocateProblems(verr, []byte(brokenAgentkitfile)) {
		for prefix, line := range want {
			if strings.HasPrefix(lp.Message, prefix) {
				found[prefix] = true
				if lp.StartLine != line {
					t.Errorf("%q: line %d (path %q), want %d", lp.Message, lp.StartLine, lp.Path, line)
				}
			}
		}
	}
	for prefix := range want {
		if !found[prefix] {
			t.Errorf("no problem starting with %q in:\n%v", prefix, verr)
		}
	}
}

func TestProblemMessagesAreUnchanged(t *testing.T) {
	cfg, err := NewFromBytes([]byte(brokenAgentkitfile))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	verr := cfg.Validate()
	for _, lp := range LocateProblems(verr, []byte(brokenAgentkitfile)) {
		if strings.Contains(lp.Message, "$.") {
			t.Fatalf("YAML path leaked into message: %q", lp.Message)
		}
	}
	if strings.Contains(verr.Error(), "$.") {
		t.Fatalf("YAML path leaked into Validate output: %v", verr)
	}
}

func TestLocateProblemsNonProblemErrorsPointAtLineOne(t *testing.T) {
	located := LocateProblems(errString("plain"), []byte(locateFixture))
	if len(located) != 1 || located[0].StartLine != 1 || located[0].Message != "plain" {
		t.Fatalf("got %+v", located)
	}
	if LocateProblems(nil, []byte(locateFixture)) != nil {
		t.Fatal("nil error should locate nothing")
	}
}

type errString string

func (e errString) Error() string { return string(e) }
