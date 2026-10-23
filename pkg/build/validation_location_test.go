package build

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/moby/buildkit/solver/errdefs"
)

// brokenLocatedAgentkitfile is missing model.name (reported at `model:`, line
// 5) and has an out-of-range expose.port (line 11).
const brokenLocatedAgentkitfile = `apiVersion: v1alpha1
kind: Agent
metadata:
  name: hello
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
instructions: Be helpful.
expose:
  openai: true
  port: 70000
`

func TestBuildReportsValidationProblemsWithFileAndLine(t *testing.T) {
	const filename = "agents/broken.yaml"
	configRef := &memoryReference{files: map[string][]byte{filename: []byte(brokenLocatedAgentkitfile)}}
	c := &fakeBuildClient{
		opts: client.BuildOpts{Opts: map[string]string{keyFilename: filename}},
		expected: []expectedSolve{
			{sourcePrefix: localDockerfileSourcePrefix, ref: configRef},
		},
	}

	_, err := Build(context.Background(), c)
	if err == nil {
		t.Fatal("Build() error = nil, want validation errors")
	}
	for _, want := range []string{
		"agents/broken.yaml:5: model.name is required",
		"agents/broken.yaml:11: expose.port 70000 is out of range",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("Build() error = %q, want substring %q", err, want)
		}
	}

	sources := errdefs.Sources(err)
	if len(sources) == 0 {
		t.Fatal("Build() error carries no source location")
	}
	src := sources[0]
	if got := src.GetInfo().GetFilename(); got != filename {
		t.Fatalf("source filename = %q, want %q", got, filename)
	}
	if got := src.GetInfo().GetLanguage(); got != "YAML" {
		t.Fatalf("source language = %q, want YAML", got)
	}
	if !bytes.Equal(src.GetInfo().GetData(), []byte(brokenLocatedAgentkitfile)) {
		t.Fatal("source data is not the Agentkitfile contents")
	}
	if ranges := src.GetRanges(); len(ranges) != 1 || ranges[0].GetStart().GetLine() != 5 {
		t.Fatalf("source ranges = %v, want one range starting on line 5", ranges)
	}

	var printed bytes.Buffer
	if err := src.Print(&printed); err != nil {
		t.Fatalf("Source.Print() error = %v", err)
	}
	if !strings.Contains(printed.String(), "model:") {
		t.Fatalf("printed snippet = %q, want it to show the model: line", printed.String())
	}
}

func TestLocateValidationErrorsAttachesASourceToEveryProblem(t *testing.T) {
	data := []byte(brokenLocatedAgentkitfile)
	cfgErr := locateValidationErrors(joinedProblems(t, data), defaultAgentkitfileName, data)
	joined, ok := cfgErr.(interface{ Unwrap() []error })
	if !ok {
		t.Fatalf("locateValidationErrors() = %T, want a joined error", cfgErr)
	}
	for _, e := range joined.Unwrap() {
		if len(errdefs.Sources(e)) != 1 {
			t.Fatalf("problem %q carries %d sources, want 1", e, len(errdefs.Sources(e)))
		}
		if !strings.HasPrefix(e.Error(), defaultAgentkitfileName+":") {
			t.Fatalf("problem %q is missing the file:line prefix", e)
		}
	}
}

func joinedProblems(t *testing.T, data []byte) error {
	t.Helper()
	loaded, err := parseLoadedAgentkitfile(defaultAgentkitfileName, data, map[string]string{}, nil)
	if err != nil {
		t.Fatalf("parseLoadedAgentkitfile() error = %v", err)
	}
	verr := loaded.config.Validate()
	if verr == nil {
		t.Fatal("expected validation errors")
	}
	return verr
}
