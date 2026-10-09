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
	commandPathEnv                   = "PATH"
	commandLogEnv                    = "COMMAND_LOG"
	commandPlatformEnv               = "PLATFORM"
	commandBuilderEnv                = "BUILDER"
	commandOpenAIKeyEnv              = "OPENAI_API_KEY" //nolint:gosec // Environment variable name, not a credential.
	commandBuildServeTarget          = "build-serve"
	commandBuildServeMAFTarget       = "build-serve-maf"
	commandBuildServeLangGraphTarget = "build-serve-langgraph"
)

func TestRuntimeAdapterBuildTargetsHonorPlatform(t *testing.T) {
	tests := []struct {
		target     string
		dockerfile string
		image      string
	}{
		{target: commandBuildServeTarget, dockerfile: "runtimes/pydantic-ai/Dockerfile", image: "agentkit-serve:platform-test"},
		{target: commandBuildServeMAFTarget, dockerfile: "runtimes/microsoft-agent-framework/Dockerfile", image: "agentkit-serve-maf:platform-test"},
		{target: commandBuildServeLangGraphTarget, dockerfile: "runtimes/langgraph/Dockerfile", image: "agentkit-serve-langgraph:platform-test"},
	}

	for _, tt := range tests {
		t.Run(tt.target, func(t *testing.T) {
			cmd := makeAdapterDryRunCommand(tt.target)
			cmd.Dir = filepath.Join("..", "..")
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("make dry run failed: %v\n%s", err, out)
			}
			command := string(out)
			for _, want := range []string{
				"docker buildx build",
				"-f " + tt.dockerfile,
				"-t " + tt.image,
				"--platform linux/arm64",
				"--load",
			} {
				if !strings.Contains(command, want) {
					t.Fatalf("%s command = %q, want substring %q", tt.target, command, want)
				}
			}
		})
	}
}

func makeAdapterDryRunCommand(target string) *exec.Cmd {
	switch target {
	case commandBuildServeTarget:
		return exec.Command("make", "--no-print-directory", "-n", commandBuildServeTarget, "PLATFORM=linux/arm64", "TAG=platform-test")
	case commandBuildServeMAFTarget:
		return exec.Command("make", "--no-print-directory", "-n", commandBuildServeMAFTarget, "PLATFORM=linux/arm64", "TAG=platform-test")
	case commandBuildServeLangGraphTarget:
		return exec.Command("make", "--no-print-directory", "-n", commandBuildServeLangGraphTarget, "PLATFORM=linux/arm64", "TAG=platform-test")
	default:
		panic("unsupported adapter build target: " + target)
	}
}

func TestRunTestAgentIsHostReachableAndCapturesCurlToken(t *testing.T) {
	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("resolve repository root: %v", err)
	}
	tempDir := t.TempDir()
	binDir := filepath.Join(tempDir, "bin")
	if err := os.Mkdir(binDir, 0o755); err != nil {
		t.Fatalf("create fake bin directory: %v", err)
	}
	commandLog := filepath.Join(tempDir, "commands.log")

	writeCommandStub(t, binDir, "docker", `
{
  printf 'docker'
  for arg in "$@"; do printf '\t%s' "$arg"; done
  printf '\n'
} >>"${COMMAND_LOG}"
`)

	const modelKey = "model-key-must-not-appear"
	const localToken = "command-capture-token"
	cmd := exec.Command(
		"make",
		"--no-print-directory",
		"run-test-agent",
		"PLATFORM=linux/arm64",
		"TAG=command-capture",
		"LOCAL_AUTH_TOKEN="+localToken,
	)
	cmd.Dir = repoRoot
	cmd.Env = replaceEnvironment(os.Environ(), map[string]string{
		commandLogEnv:       commandLog,
		"MAKEFLAGS":         "",
		commandOpenAIKeyEnv: modelKey,
		commandPathEnv:      binDir + string(os.PathListSeparator) + os.Getenv(commandPathEnv),
	})
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run-test-agent command capture failed: %v\n%s", err, out)
	}
	logBytes, err := os.ReadFile(commandLog)
	if err != nil {
		t.Fatalf("read command log: %v", err)
	}
	command := string(logBytes)
	for _, want := range []string{
		"\t-p\t127.0.0.1:8080:8080\t",
		"\t-e\tAGENTKIT_BIND=0.0.0.0\t",
		"\t-e\tAGENTKIT_AUTH_TOKEN=" + localToken + "\t",
		"\t-e\tOPENAI_API_KEY\t",
		"\thello-agent:command-capture\n",
	} {
		if !strings.Contains(command, want) {
			t.Fatalf("captured docker command = %q, want substring %q", command, want)
		}
	}
	combined := string(out) + command
	if strings.Contains(combined, modelKey) {
		t.Fatalf("run-test-agent output leaked model key: %q", combined)
	}
	for _, want := range []string{
		"Authorization: Bearer " + localToken,
		"http://127.0.0.1:8080/v1/models",
	} {
		if !strings.Contains(string(out), want) {
			t.Fatalf("run-test-agent output = %q, want substring %q", out, want)
		}
	}
}

type liveAIKitAdapter struct {
	runtime string
	slug    string
	target  string
	image   string
}

var liveAIKitAdapters = []liveAIKitAdapter{
	{runtime: runtimePydca, slug: "pydantic", target: commandBuildServeTarget, image: "agentkit-serve"},
	{runtime: runtimeMAFName, slug: runtimeMAFAls, target: commandBuildServeMAFTarget, image: "agentkit-serve-maf"},
	{runtime: runtimeLangGraph, slug: runtimeLangGraph, target: commandBuildServeLangGraphTarget, image: "agentkit-serve-langgraph"},
}

const (
	liveAIKitTestToken                = "live-command-capture-token" //nolint:gosec // Test-only local bearer token.
	liveAIKitExternalProviderSentinel = "external-provider-key-must-not-appear"
)

func TestLiveAIKitScriptForwardsDetectedPlatformToAdapterBuild(t *testing.T) {
	tests := []struct {
		name     string
		args     []string
		adapters []liveAIKitAdapter
		platform string
		builder  string
	}{
		{name: runtimePydca, args: []string{runtimePydca}, adapters: liveAIKitAdapters[:1]},
		{name: runtimeMAFName, args: []string{runtimeMAFName}, adapters: liveAIKitAdapters[1:2]},
		{name: runtimeLangGraph, args: []string{runtimeLangGraph}, adapters: liveAIKitAdapters[2:]},
		{name: "maf-alias", args: []string{runtimeMAFAls}, adapters: liveAIKitAdapters[1:2]},
		{name: "default-all", adapters: liveAIKitAdapters},
		{name: "explicit-platform-and-builder", args: []string{runtimeLangGraph}, adapters: liveAIKitAdapters[2:], platform: "linux/amd64", builder: "live-builder"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			commands, out, err := captureLiveAIKitCommands(t, tt.args, map[string]string{
				commandPlatformEnv: tt.platform,
				commandBuilderEnv:  tt.builder,
			})
			if err != nil {
				t.Fatalf("live script command capture failed: %v\n%s", err, out)
			}
			platform := tt.platform
			if platform == "" {
				platform = "linux/arm64"
			}
			assertLiveAIKitSharedCommands(t, commands, platform)
			builds := capturedCommandLines(commands, "make\tbuild-serve")
			if len(builds) != len(tt.adapters) {
				t.Fatalf("adapter builds = %q, want %d", builds, len(tt.adapters))
			}
			previousRelease := -1
			for i, adapter := range tt.adapters {
				want := "make\t" + adapter.target + "\tTAG=command-capture\tPLATFORM=" + platform
				if builds[i] != want {
					t.Fatalf("adapter build %d = %q, want %q", i, builds[i], want)
				}
				buildIndex, releaseIndex := assertLiveAIKitAdapterCommands(t, commands, adapter, platform, tt.builder)
				if strings.Index(commands, builds[i]) <= previousRelease || buildIndex <= previousRelease {
					t.Errorf("%s built before the previous agent was released", adapter.runtime)
				}
				previousRelease = releaseIndex
			}
			requests := capturedCommandLines(commands, "curl\t-fsS\t--connect-timeout\t5\t--max-time\t120\t")
			if len(requests) != len(tt.adapters) {
				t.Fatalf("agent requests = %q, want %d", requests, len(tt.adapters))
			}
			for _, request := range requests {
				for _, want := range []string{
					"\t-H\tAuthorization: Bearer " + liveAIKitTestToken,
					"\t-H\tContent-Type: application/json\t--data\t",
					`"model":"qwen-3.5-2b","stream":false`,
					"DONE42",
					"\thttp://127.0.0.1:18080/v1/chat/completions",
				} {
					if !strings.Contains(request, want) {
						t.Errorf("agent request = %q, want %q", request, want)
					}
				}
			}
			for _, forbidden := range []string{
				"vekil", "COPILOT_GITHUB_TOKEN", "TOKEN_DIR", commandOpenAIKeyEnv, "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY",
				liveAIKitExternalProviderSentinel,
			} {
				if strings.Contains(commands+out, forbidden) {
					t.Errorf("live smoke retains external provider dependency %q", forbidden)
				}
			}
		})
	}
}

func assertLiveAIKitSharedCommands(t *testing.T, commands, platform string) {
	t.Helper()
	for _, prefix := range []string{
		"docker\tnetwork\tcreate\t", "docker\tnetwork\trm\t", "make\tbuild-agentkit\t",
		"curl\t-fsS\t--max-time\t15\t", "curl\t-fsS\t--connect-timeout\t5\t--max-time\t300\t",
	} {
		if got := capturedCommandLines(commands, prefix); len(got) != 1 {
			t.Errorf("shared commands with prefix %q = %q, want exactly one", prefix, got)
		}
	}
	aikitRuns := capturedCommandLines(commands, "docker\trun\t")
	var aikitRun string
	for _, run := range aikitRuns {
		if strings.Contains(run, "\t--network-alias\taikit\t") {
			if aikitRun != "" {
				t.Fatal("AIKit started more than once")
			}
			aikitRun = run
		}
	}
	for _, want := range []string{
		"ghcr.io/kaito-project/aikit/qwen3.5:2b@sha256:",
		"--cpus\t4", "--platform\t" + platform, "--network-alias\taikit",
		"LOCALAI_FORCE_META_BACKEND_CAPABILITY=cpu", "--config-file=/config.yaml", "test/aikit-e2e/model.yaml,dst=/config.yaml,readonly",
	} {
		if !strings.Contains(aikitRun, want) {
			t.Errorf("AIKit run omits %q: %s", want, aikitRun)
		}
	}
	networks := capturedCommandLines(commands, "docker\tnetwork\tcreate\t")
	if len(networks) != 1 {
		t.Fatal("missing single owned network")
	}
	network := strings.TrimPrefix(networks[0], "docker\tnetwork\tcreate\t")
	for _, run := range aikitRuns {
		if !strings.Contains(run, "\t--network\t"+network+"\t") {
			t.Errorf("container does not share AIKit's network: %s", run)
		}
	}
	if !strings.Contains(commands, "docker\tnetwork\trm\t"+network+"\n") {
		t.Errorf("owned network was not removed: %s", commands)
	}
	if !strings.Contains(commands, "docker\trm\t-fv\t"+strings.TrimSuffix(network, "-network")+"-aikit\n") {
		t.Errorf("owned AIKit container was not removed: %s", commands)
	}
	warmups := capturedCommandLines(commands, "curl\t-fsS\t--connect-timeout\t5\t--max-time\t300\t")
	if len(warmups) == 1 && strings.Index(commands, warmups[0]) > strings.Index(commands, "make\tbuild-serve") {
		t.Error("model warmup ran after an adapter build")
	}
}

func assertLiveAIKitAdapterCommands(t *testing.T, commands string, adapter liveAIKitAdapter, platform, builder string) (int, int) {
	t.Helper()
	image := adapter.slug + "-live-agent:command-capture"
	var build, run string
	for _, line := range capturedCommandLines(commands, "docker\t") {
		if strings.HasPrefix(line, "docker\tbuildx\tbuild\t") && strings.Contains(line, "\t-t\t"+image+"\t") {
			if build != "" {
				t.Fatalf("duplicate %s agent build", adapter.runtime)
			}
			build = line
		}
		if strings.HasPrefix(line, "docker\trun\t") && strings.HasSuffix(line, "\t"+image) {
			if run != "" {
				t.Fatalf("duplicate %s agent run", adapter.runtime)
			}
			run = line
		}
	}
	for _, want := range []string{
		"\t-f\ttest/agentkitfile-" + adapter.slug + "-live.yaml\t",
		"\t--build-arg\tBUILDKIT_SYNTAX=agentkit:command-capture\t",
		"\t--build-arg\tadapter=" + adapter.image + ":command-capture\t",
		"\t--platform\t" + platform + "\t", "\t--load\t--provenance=false",
	} {
		if !strings.Contains(build, want) {
			t.Errorf("%s agent build = %q, want %q", adapter.runtime, build, want)
		}
	}
	if builder != "" && !strings.Contains(build, "\t--builder\t"+builder+"\t") {
		t.Errorf("%s agent build ignores builder %q: %s", adapter.runtime, builder, build)
	}
	for _, want := range []string{
		"\t--platform\t" + platform + "\t", "\t-p\t127.0.0.1:18080:8080\t",
		"\t-e\tAGENTKIT_BIND=0.0.0.0\t", "\t-e\tAGENTKIT_AUTH_TOKEN=" + liveAIKitTestToken + "\t",
		"\t-e\tMODEL_API_KEY=not-needed\t",
	} {
		if !strings.Contains(run, want) {
			t.Errorf("%s agent run = %q, want %q", adapter.runtime, run, want)
		}
	}
	fields := strings.Split(run, "\t")
	var name string
	for i, field := range fields {
		if field == "--name" && i+1 < len(fields) {
			name = fields[i+1]
		}
	}
	if name == "" || !strings.HasSuffix(name, "-"+adapter.slug+"-agent") {
		t.Fatalf("%s agent container name = %q", adapter.runtime, name)
	}
	release := "docker\trm\t-fv\t" + name + "\n"
	if strings.Count(commands, release) != 1 {
		t.Fatalf("%s agent not released exactly once: %s", adapter.runtime, commands)
	}
	buildIndex, runIndex, releaseIndex := strings.Index(commands, build), strings.Index(commands, run), strings.Index(commands, release)
	requestIndex := strings.Index(commands[runIndex:], "\thttp://127.0.0.1:18080/v1/chat/completions\n")
	if runIndex <= buildIndex || requestIndex < 0 || releaseIndex <= runIndex+requestIndex {
		t.Errorf("%s build/run/request/release are out of order: %s", adapter.runtime, commands)
	}
	return buildIndex, releaseIndex
}

func TestLiveAIKitScriptRejectsInvalidAdapter(t *testing.T) {
	for _, args := range [][]string{{"unknown"}, {""}, {runtimeMAFAls, runtimeLangGraph}} {
		t.Run(strings.Join(args, "/"), func(t *testing.T) {
			commands, out, err := captureLiveAIKitCommands(t, args, nil)
			if err == nil || !strings.Contains(out, "usage:") {
				t.Fatalf("invalid selection %q accepted or missing usage: %v\n%s", args, err, out)
			}
			if commands != "" {
				t.Fatalf("invalid selection invoked CLI commands: %s", commands)
			}
		})
	}
}

func TestLiveAIKitScriptRejectsFailedAgentInferenceAndCleansUp(t *testing.T) {
	for _, tc := range []struct {
		name     string
		response string
		curlExit string
	}{
		{name: "missing-sentinel", response: `{"choices":[{"message":{"content":"Not the sentinel."}}]}`, curlExit: "0"},
		{name: "non-string-content", response: `{"choices":[{"message":{"content":42}}]}`, curlExit: "0"},
		{name: "http-error", response: `{"choices":[{"message":{"content":"DONE42"}}]}`, curlExit: "22"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			commands, out, err := captureLiveAIKitCommands(t, nil, map[string]string{
				"AGENT_RESPONSE_JSON": tc.response, "AGENT_CURL_EXIT": tc.curlExit,
			})
			if err == nil {
				t.Fatalf("failed inference accepted: %s", out)
			}
			if builds := capturedCommandLines(commands, "make\tbuild-serve"); len(builds) != 1 || !strings.HasPrefix(builds[0], "make\tbuild-serve\t") {
				t.Errorf("continued to another adapter after failure: %q", builds)
			}
			if removals := capturedCommandLines(commands, "docker\trm\t-fv\t"); len(removals) != 2 {
				t.Errorf("failure cleanup did not remove AIKit and agent: %q", removals)
			}
			if removals := capturedCommandLines(commands, "docker\tnetwork\trm\t"); len(removals) != 1 {
				t.Errorf("failure cleanup did not remove owned network: %q", removals)
			}
			if !strings.Contains(out, "Live AIKit-backed AgentKit E2E failed") || strings.Contains(out, liveAIKitTestToken) {
				t.Errorf("failure diagnostics missing or leaked token: %s", out)
			}
		})
	}
}

func TestLiveAIKitFixturesUseLocalModel(t *testing.T) {
	for _, adapter := range liveAIKitAdapters {
		t.Run(adapter.runtime, func(t *testing.T) {
			path := filepath.Join("..", "..", "test", "agentkitfile-"+adapter.slug+"-live.yaml")
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := config.NewFromBytes(data)
			if err != nil {
				t.Fatalf("parse %s: %v", path, err)
			}
			if err := cfg.Validate(); err != nil {
				t.Fatalf("validate %s: %v", path, err)
			}
			if cfg.Runtime != adapter.runtime || cfg.Metadata.Name != adapter.slug+"-live-e2e-agent" {
				t.Errorf("fixture targets wrong adapter: %+v", cfg)
			}
			if cfg.Model.Provider != "openai-compatible" || cfg.Model.BaseURL != "http://aikit:8080/v1" ||
				cfg.Model.Name != "qwen-3.5-2b" || cfg.Model.APIKeyEnv != "MODEL_API_KEY" || cfg.Model.Auth != nil {
				t.Errorf("fixture does not use unauthenticated local AIKit model: %+v", cfg.Model)
			}
			if !cfg.Expose.OpenAI || !strings.Contains(cfg.Instructions.Inline, "DONE42") {
				t.Error("fixture omits OpenAI endpoint or sentinel instruction")
			}
		})
	}
}

func captureLiveAIKitCommands(t *testing.T, args []string, overrides map[string]string) (string, string, error) {
	t.Helper()
	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("resolve repository root: %v", err)
	}
	jqPath, err := exec.LookPath("jq")
	if err != nil {
		t.Skip("jq is required for the live E2E response validator")
	}
	tempDir := t.TempDir()
	binDir := filepath.Join(tempDir, "bin")
	if err := os.Mkdir(binDir, 0o755); err != nil {
		t.Fatalf("create fake bin directory: %v", err)
	}
	commandLog := filepath.Join(tempDir, "commands.log")
	if err := os.WriteFile(commandLog, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	record := `
{
  printf '%s' "${0##*/}"
  for arg in "$@"; do printf '\t%s' "$arg"; done
  printf '\n'
} >>"${COMMAND_LOG}"
`
	writeCommandStub(t, binDir, "docker", record+`
case "${1:-}" in
  info)
    case "$*" in
      *NCPU*) printf '8\n' ;;
      *) printf 'arm64\n' ;;
    esac
    ;;
  inspect) printf 'true\n' ;;
  logs) printf 'Authorization: Bearer %s\n' "$AGENTKIT_AUTH_TOKEN" ;;
esac
`)
	writeCommandStub(t, binDir, "make", record)
	writeCommandStub(t, binDir, "go", record)
	writeCommandStub(t, binDir, "curl", record+`
case "$*" in
  */v1/models*) printf '{"data":[{"id":"qwen-3.5-2b"}]}' ;;
  *:18089/v1/chat/completions*)
    printf '{"model":"qwen-3.5-2b","created":123,"choices":[{"message":{"content":"OK"}}]}'
    ;;
  *:18080/v1/chat/completions*) printf '%s' "$AGENT_RESPONSE_JSON"; exit "$AGENT_CURL_EXIT" ;;
esac
`)
	writeCommandStub(t, binDir, "jq", `exec "$REAL_JQ" "$@"`)
	//nolint:gosec // Fixed repository script and test-owned adapter arguments; Docker, make, curl, and go are stubbed.
	cmd := exec.Command("bash", append([]string{"scripts/live-aikit-agent-e2e.sh"}, args...)...)
	cmd.Dir = repoRoot
	env := map[string]string{
		commandLogEnv: commandLog, commandPathEnv: binDir + string(os.PathListSeparator) + os.Getenv(commandPathEnv),
		"AIKIT_IMAGE": "", "AIKIT_HOST_PORT": "18089", "AGENTKIT_LIVE_HOST_PORT": "18080", "AGENTKIT_AUTH_TOKEN": liveAIKitTestToken,
		commandPlatformEnv: "", commandBuilderEnv: "", "BUILDX_BUILDER": "", "RUNNER_TEMP": tempDir, "TAG": "command-capture", "REAL_JQ": jqPath,
		"AGENT_RESPONSE_JSON": `{"model":"qwen-3.5-2b","created":123,"choices":[{"message":{"content":"DONE42"}}]}`, "AGENT_CURL_EXIT": "0",
		"MODEL_API_KEY":     liveAIKitExternalProviderSentinel,
		commandOpenAIKeyEnv: liveAIKitExternalProviderSentinel, "ANTHROPIC_API_KEY": liveAIKitExternalProviderSentinel,
		"AZURE_OPENAI_API_KEY": liveAIKitExternalProviderSentinel, "COPILOT_GITHUB_TOKEN": liveAIKitExternalProviderSentinel,
		"TOKEN_DIR": filepath.Join(tempDir, "provider-tokens-must-not-be-used"),
	}
	for key, value := range overrides {
		env[key] = value
	}
	cmd.Env = replaceEnvironment(os.Environ(), env)
	out, runErr := cmd.CombinedOutput()
	logBytes, err := os.ReadFile(commandLog)
	if err != nil {
		t.Fatalf("read command log: %v", err)
	}
	workDirs, err := filepath.Glob(filepath.Join(tempDir, "agentkit-live-aikit.*"))
	if err != nil || len(workDirs) != 0 {
		t.Errorf("live script left temporary work directories: %q: %v", workDirs, err)
	}
	return string(logBytes), string(out), runErr
}

func capturedCommandLines(commands, prefix string) []string {
	var lines []string
	for _, line := range strings.Split(commands, "\n") {
		if strings.HasPrefix(line, prefix) {
			lines = append(lines, line)
		}
	}
	return lines
}

func TestAIKitCPUQuotaUsesAvailableCPUs(t *testing.T) {
	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ available, limit string }{
		{"1", "1"}, {"2", "2"}, {"4", "4"}, {"8", "4"}, {"0", ""}, {"invalid", ""},
	} {
		t.Run(tc.available, func(t *testing.T) {
			tempDir := t.TempDir()
			binDir := filepath.Join(tempDir, "bin")
			if err := os.Mkdir(binDir, 0o755); err != nil {
				t.Fatal(err)
			}
			logPath := filepath.Join(tempDir, "docker.log")
			writeCommandStub(t, binDir, "docker", `
if [ "$1" = info ]; then printf '%s\n' "$DAEMON_CPUS"; exit 0; fi
printf '%s\n' "$@" >"$COMMAND_LOG"
`)
			//nolint:gosec // Fixed shell source and repository/test-owned paths, not external input.
			cmd := exec.Command("bash", "-c", `source "$1"; start_aikit "$2" -d --name quota-test`,
				"quota", filepath.Join(repoRoot, "scripts", "aikit-e2e-common.sh"), filepath.Join(repoRoot, "test", "aikit-e2e", "model.yaml"))
			cmd.Env = replaceEnvironment(os.Environ(), map[string]string{
				commandPathEnv: binDir + string(os.PathListSeparator) + os.Getenv(commandPathEnv),
				"DAEMON_CPUS":  tc.available, commandLogEnv: logPath,
			})
			out, err := cmd.CombinedOutput()
			if tc.limit == "" {
				if err == nil {
					t.Fatalf("invalid CPU count accepted: %s", out)
				}
				if _, err := os.Stat(logPath); !os.IsNotExist(err) {
					t.Fatal("invalid CPU count started a container")
				}
				return
			}
			if err != nil {
				t.Fatalf("container startup failed: %v: %s", err, out)
			}
			commands, err := os.ReadFile(logPath)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(commands), "--cpus\n"+tc.limit+"\n") {
				t.Fatalf("CPU quota = %s, want %s", commands, tc.limit)
			}
		})
	}
}

func TestAIKitWarmupRejectsInferenceFailures(t *testing.T) {
	if _, err := exec.LookPath("jq"); err != nil {
		t.Skip("jq is required for the live E2E warmup validator")
	}
	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name     string
		response string
		curlExit string
		wantOK   bool
	}{
		{"valid", `{"model":"qwen-3.5-2b","created":123,"choices":[{"message":{"content":"OK"}}]}`, "0", true},
		{"http-error", `{"error":"model unavailable"}`, "22", false},
		{"http-error-with-valid-json", `{"model":"qwen-3.5-2b","created":123,"choices":[{"message":{"content":"OK"}}]}`, "22", false},
		{"wrong-model", `{"model":"another-model","created":123,"choices":[{"message":{"content":"OK"}}]}`, "0", false},
		{"missing-created", `{"model":"qwen-3.5-2b","choices":[{"message":{"content":"OK"}}]}`, "0", false},
		{"empty-content", `{"model":"qwen-3.5-2b","created":123,"choices":[{"message":{"content":""}}]}`, "0", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tempDir := t.TempDir()
			binDir := filepath.Join(tempDir, "bin")
			if err := os.Mkdir(binDir, 0o755); err != nil {
				t.Fatal(err)
			}
			writeCommandStub(t, binDir, "curl", `printf '%s' "$WARMUP_JSON"; exit "$CURL_EXIT"`)
			// The conditional disables Bash errexit inside the function, so a failed
			// curl must be propagated explicitly rather than hidden by valid JSON.
			//nolint:gosec // Fixed shell source and repository/test-owned paths, not external input.
			cmd := exec.Command("bash", "-c", `source "$1"; if warm_aikit "$2" "$3"; then exit 0; else exit 1; fi`,
				"warmup", filepath.Join(repoRoot, "scripts", "aikit-e2e-common.sh"), "http://aikit:8080", filepath.Join(tempDir, "response.json"))
			cmd.Env = replaceEnvironment(os.Environ(), map[string]string{
				commandPathEnv: binDir + string(os.PathListSeparator) + os.Getenv(commandPathEnv),
				"WARMUP_JSON":  tc.response, "CURL_EXIT": tc.curlExit,
			})
			out, err := cmd.CombinedOutput()
			if (err == nil) != tc.wantOK {
				t.Fatalf("warmup success = %v, want %v: %s", err == nil, tc.wantOK, out)
			}
		})
	}
}

func writeCommandStub(t *testing.T, dir, name, body string) {
	t.Helper()
	path := filepath.Join(dir, name)
	contents := "#!/bin/sh\nset -eu\n" + body
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("write %s stub: %v", name, err)
	}
	if err := os.Chmod(path, 0o700); err != nil {
		t.Fatalf("make %s stub executable: %v", name, err)
	}
}

func replaceEnvironment(base []string, replacements map[string]string) []string {
	out := make([]string, 0, len(base)+len(replacements))
	for _, entry := range base {
		key, _, ok := strings.Cut(entry, "=")
		if ok {
			if _, replace := replacements[key]; replace {
				continue
			}
		}
		out = append(out, entry)
	}
	for key, value := range replacements {
		out = append(out, key+"="+value)
	}
	return out
}
