package build

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/orka-agents/agentkit/pkg/agentkit/config"
)

const harnessLiveMode = "live"

func TestHarnessE2EAdapterSelection(t *testing.T) {
	if _, err := exec.LookPath("jq"); err != nil {
		t.Skip("jq is required for the harness artifact writer")
	}
	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	all := []string{runtimePydca, runtimeMAFName, runtimeLangGraph}
	for _, mode := range []string{"offline", harnessLiveMode} {
		for _, tc := range []struct {
			name, adapter string
			want          []string
		}{
			{"all", "", all},
			{"pydantic", runtimePydca, all[:1]},
			{runtimeMAFAls, runtimeMAFName, all[1:2]},
			{"maf-alias", runtimeMAFAls, all[1:2]},
			{runtimeLangGraph, runtimeLangGraph, all[2:]},
		} {
			t.Run(mode+"/"+tc.name, func(t *testing.T) {
				tempDir := t.TempDir()
				binDir := filepath.Join(tempDir, "bin")
				if err := os.Mkdir(binDir, 0o755); err != nil {
					t.Fatal(err)
				}
				writeCommandStub(t, binDir, "docker", `
case "$1" in
  info) printf 'linux/amd64\n' ;;
  buildx) printf 'Driver: docker\n' ;;
esac
`)
				// Stop before downloads, builds, or container creation. The script's
				// run.json records the real argument parsing and adapter selection.
				writeCommandStub(t, binDir, "git", `
case "$*" in
  *fetch*) exit 42 ;;
  *rev-parse*) printf '0000000000000000000000000000000000000000\n' ;;
esac
`)
				for _, command := range []string{"curl", "go", "make"} {
					writeCommandStub(t, binDir, command, "")
				}
				args := []string{"scripts/orka-harness-v2-e2e.sh", mode}
				if tc.adapter != "" {
					args = append(args, tc.adapter)
				}
				cmd := exec.Command("bash", args...)
				cmd.Dir = repoRoot
				artifactRoot := filepath.Join(tempDir, "artifacts")
				cmd.Env = replaceEnvironment(os.Environ(), map[string]string{
					commandPathEnv: binDir + string(os.PathListSeparator) + os.Getenv(commandPathEnv),
					"RUNNER_TEMP":  tempDir, "ARTIFACT_DIR": artifactRoot,
					commandPlatformEnv: "", commandBuilderEnv: "",
				})
				out, err := cmd.CombinedOutput()
				if err == nil {
					t.Fatalf("expected the controlled fetch stop: %s", out)
				}
				var exitErr *exec.ExitError
				if !errors.As(err, &exitErr) || exitErr.ExitCode() != 42 {
					t.Fatalf("runner failed before the controlled stop: %v: %s", err, out)
				}
				paths, err := filepath.Glob(filepath.Join(artifactRoot, "*", "run.json"))
				if err != nil || len(paths) != 1 {
					t.Fatalf("run artifact paths = %v, error = %v: %s", paths, err, out)
				}
				data, err := os.ReadFile(paths[0])
				if err != nil {
					t.Fatal(err)
				}
				var result struct {
					Mode     string   `json:"mode"`
					Adapters []string `json:"adapters"`
				}
				if err := json.Unmarshal(data, &result); err != nil {
					t.Fatal(err)
				}
				if result.Mode != mode || !reflect.DeepEqual(result.Adapters, tc.want) {
					t.Fatalf("run selection = %s/%v, want %s/%v", result.Mode, result.Adapters, mode, tc.want)
				}
			})
		}
	}
}

func TestHarnessE2ERejectsUnsupportedSelection(t *testing.T) {
	for _, args := range [][]string{{harnessLiveMode, "unsupported"}, {"offline", "unsupported"}, {"unknown"}, {harnessLiveMode, runtimePydca, "extra"}} {
		t.Run(strings.Join(args, "/"), func(t *testing.T) {
			//nolint:gosec // Fixed invalid argument cases; no external input or container startup.
			cmd := exec.Command("bash", append([]string{"scripts/orka-harness-v2-e2e.sh"}, args...)...)
			cmd.Dir = filepath.Join("..", "..")
			out, err := cmd.CombinedOutput()
			if err == nil || (!strings.Contains(string(out), "unsupported") && !strings.Contains(string(out), "expected a mode")) {
				t.Fatalf("selection was not rejected before startup: %v: %s", err, out)
			}
		})
	}
}

func TestHarnessLiveFixturesMatchAdapters(t *testing.T) {
	for _, adapter := range []string{runtimePydca, runtimeMAFName, runtimeLangGraph} {
		t.Run(adapter, func(t *testing.T) {
			path := filepath.Join("..", "..", "test", "orka-harness-v2", "agentkitfile-"+adapter+"-live.yaml")
			data, err := os.ReadFile(path)
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
			if cfg.Runtime != adapter || cfg.Model.Name != "qwen-3.5-2b" || cfg.Model.BaseURL != "http://provider.invalid/v1" {
				t.Fatalf("live fixture runtime/model mismatch: %+v", cfg)
			}
		})
	}
}
