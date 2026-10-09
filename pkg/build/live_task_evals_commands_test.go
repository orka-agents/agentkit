package build

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/orka-agents/agentkit/pkg/agentkit/config"
)

const (
	liveEvalModel       = "qwen-3.5-2b"
	liveEvalModelEnv    = "MODEL_API_KEY"
	liveEvalUnsupported = "unsupported"
	liveEvalTrialsEnv   = "EVAL_TRIALS"
)

func TestLiveTaskEvalFixturesMatchAdapters(t *testing.T) {
	for _, adapter := range []string{runtimePydca, runtimeMAFName, runtimeLangGraph} {
		t.Run(adapter, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("..", "..", "test", "evals", "agentkitfile-"+adapter+".yaml"))
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := config.NewFromBytes(data)
			if err != nil {
				t.Fatal(err)
			}
			if err := cfg.Validate(); err != nil {
				t.Fatal(err)
			}
			if cfg.Runtime != adapter || cfg.Model.Name != liveEvalModel || cfg.Model.BaseURL != "http://eval-fixture:8090/v1" {
				t.Fatalf("eval runtime/model mismatch: %+v", cfg)
			}
			if cfg.Model.APIKeyEnv != liveEvalModelEnv || len(cfg.Tools) != 1 || cfg.Tools[0].Name != "evals" || cfg.Tools[0].URLEnv != "EVAL_MCP_URL" {
				t.Fatal("eval fixture must use declared environment names and controlled MCP tools")
			}
		})
	}
}

func TestLiveTaskEvalsRejectInvalidSelectionBeforeDocker(t *testing.T) {
	for _, tc := range []struct {
		name  string
		args  []string
		trial string
		want  string
	}{
		{"unsupported-adapter", []string{liveEvalUnsupported}, "3", "invalid adapter"},
		{"too-many-arguments", []string{runtimePydca, "extra"}, "3", "usage:"},
		{"zero-trials", []string{runtimePydca}, "0", liveEvalTrialsEnv},
		{"too-many-trials", []string{runtimePydca}, "11", liveEvalTrialsEnv},
		{"invalid-trials", []string{runtimePydca}, "3.5", liveEvalTrialsEnv},
	} {
		t.Run(tc.name, func(t *testing.T) {
			//nolint:gosec // Fixed invalid selections, rejected before container creation.
			cmd := exec.Command("bash", append([]string{"scripts/live-task-evals.sh"}, tc.args...)...)
			cmd.Dir = filepath.Join("..", "..")
			cmd.Env = replaceEnvironment(os.Environ(), map[string]string{liveEvalTrialsEnv: tc.trial})
			out, err := cmd.CombinedOutput()
			if err == nil || !strings.Contains(string(out), tc.want) {
				t.Fatalf("invalid selection not rejected: %v: %s", err, out)
			}
		})
	}
}
