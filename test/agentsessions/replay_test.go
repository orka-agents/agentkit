package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/aramase/agentsessions/api"
	v1 "github.com/aramase/agentsessions/api/genpb"
	"github.com/aramase/agentsessions/controller"
	"github.com/aramase/agentsessions/harnesswire"
	"github.com/aramase/agentsessions/sqlitelog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/proto"
)

const (
	hostPin      = "b212d498ba52615b5087b2579bdd482809065642"
	harnessToken = "agentsessions-offline-host-token"
	proofLabel   = "io.github.orka-agents.agentkit.agentsessions-e2e"
)

func docker(t *testing.T, args ...string) []byte {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	output, err := exec.CommandContext(ctx, "docker", args...).CombinedOutput()
	if err != nil {
		t.Fatalf("docker %s: %v\n%s", args[0], err, output)
	}
	return bytes.TrimSpace(output)
}

// observedClient records the actual Start frames sent to the remote Python service.
// It changes no request fields; the controller must recover Config from the journal.
type observedClient struct {
	v1.HarnessClient
	starts []*v1.Start
}

type observedStream struct {
	grpc.BidiStreamingClient[v1.ControllerFrame, v1.Event]
	client *observedClient
}

func (c *observedClient) Connect(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[v1.ControllerFrame, v1.Event], error) {
	stream, err := c.HarnessClient.Connect(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return &observedStream{BidiStreamingClient: stream, client: c}, nil
}

func (s *observedStream) Send(frame *v1.ControllerFrame) error {
	if start := frame.GetStart(); start != nil {
		s.client.starts = append(s.client.starts, proto.Clone(start).(*v1.Start))
	}
	return s.BidiStreamingClient.Send(frame)
}

func connectHarness(t *testing.T, ctx context.Context, address string) (*observedClient, api.Harness) {
	t.Helper()
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	client := &observedClient{HarnessClient: v1.NewHarnessClient(conn)}
	har := harnesswire.NewClientHarness(client)
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		probe, cancel := context.WithTimeout(ctx, time.Second)
		_, err := har.Describe(probe)
		cancel()
		if err == nil {
			return client, har
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatal("container Harness.Describe did not become ready")
	return nil, nil
}

func startContainer(t *testing.T, image, network, configDigest, implementationDigest, runID string) (string, string) {
	t.Helper()
	id := string(docker(t, "run", "-d", "--rm", "--network", network,
		"--label", proofLabel+"="+runID,
		"-e", "AGENTKIT_PROTOCOL=agentsessions",
		"-e", "AGENTKIT_BIND=0.0.0.0",
		"-e", "AGENTKIT_PORT=8080",
		"-e", "AGENTKIT_AUTH_TOKEN="+harnessToken,
		"-e", "AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST="+configDigest,
		"-e", "AGENTKIT_AGENTSESSIONS_IMPLEMENTATION_DIGEST="+implementationDigest, image))
	t.Cleanup(func() { _ = exec.Command("docker", "rm", "-f", id).Run() })
	if actual := string(docker(t, "inspect", "--format", "{{.Image}}", id)); actual != image {
		t.Fatal("actor does not use the resolved immutable image ID")
	}
	// Internal Docker networks have no published ports. This Linux-host proof
	// dials the test actor directly while it has no default egress gateway.
	ip := string(docker(t, "inspect", "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", id))
	if net.ParseIP(ip) == nil {
		t.Fatal("test actor has no internal-network IP")
	}
	return id, net.JoinHostPort(ip, "8080")
}

func assertNoProviderCredentials(t *testing.T, container string) {
	t.Helper()
	// Only inspect this test-owned container. Never dump host env or secret values.
	probe := `import os,pathlib
names={'OPENAI_API_KEY','ANTHROPIC_API_KEY','AZURE_OPENAI_API_KEY','GOOGLE_API_KEY','GEMINI_API_KEY','AGENTKIT_WORKLOAD_IDENTITY_TOKEN','AGENTKIT_ACP_PROVIDER_TOKEN'}
assert not names.intersection(os.environ)
for p in pathlib.Path('/proc').iterdir():
 if p.name.isdigit():
  try: raw=(p/'environ').read_bytes()
  except (FileNotFoundError,ProcessLookupError): continue
  actual={x.split(b'=',1)[0].decode() for x in raw.split(b'\x00') if x}
  assert not names.intersection(actual)
print('provider credentials absent')`
	if got := string(docker(t, "exec", container, "/usr/local/bin/python", "-c", probe)); got != "provider credentials absent" {
		t.Fatalf("credential probe failed: %q", got)
	}
	var info []struct {
		Config struct{ Env []string }
	}
	if err := json.Unmarshal(docker(t, "inspect", container), &info); err != nil {
		t.Fatal(err)
	}
	for _, value := range info[0].Config.Env {
		if strings.HasPrefix(value, "OPENAI_API_KEY=") {
			t.Fatal("baked model key env must not be injected")
		}
	}
}

// changedInputs deliberately diverges one reconstructed model request; replay must
// fail its model-input fingerprint check without invoking a live provider.
type changedInputs struct{ api.Harness }

func (h changedInputs) Run(ctx context.Context, start *api.Start, sink api.EventSink) error {
	copyStart := *start
	copyStart.Inputs = append([]api.Message(nil), start.Inputs...)
	copyStart.Inputs[len(copyStart.Inputs)-1] = *api.TextMessage("user", "changed replay input")
	return h.Harness.Run(ctx, &copyStart, sink)
}

func TestRunnerSignalCleanup(t *testing.T) {
	image := os.Getenv("AGENTKIT_AGENTSESSIONS_IMAGE")
	if image == "" {
		t.Skip("built image required for proof-runner signal cleanup")
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	script := filepath.Join(cwd, "../../scripts/agentsessions-e2e.sh")
	for _, tc := range []struct {
		name   string
		signal syscall.Signal
		exit   int
	}{{"interrupt", syscall.SIGINT, 130}, {"terminate", syscall.SIGTERM, 143}} {
		t.Run(tc.name, func(t *testing.T) {
			runID := fmt.Sprintf("signal-%d-%d", os.Getpid(), time.Now().UnixNano())
			filter := "label=" + proofLabel + "=" + runID
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, script, "--skip-build")
			cmd.Env = append(os.Environ(), "AGENTKIT_AGENTSESSIONS_IMAGE="+image, "AGENTKIT_AGENTSESSIONS_RUN_ID="+runID)
			var output bytes.Buffer
			cmd.Stdout, cmd.Stderr = &output, &output
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			finished := make(chan error, 1)
			go func() { finished <- cmd.Wait() }()
			ready := false
			for i := 0; i < 200; i++ {
				if len(docker(t, "network", "ls", "-q", "--filter", filter)) > 0 {
					ready = true
					break
				}
				select {
				case err := <-finished:
					t.Fatalf("runner ended before interruption: %v\n%s", err, output.Bytes())
				case <-time.After(25 * time.Millisecond):
				}
			}
			if !ready {
				_ = cmd.Process.Signal(syscall.SIGTERM)
				<-finished
				t.Fatal("runner did not create a labelled test network")
			}
			if err := cmd.Process.Signal(tc.signal); err != nil {
				t.Fatal(err)
			}
			err := <-finished
			var exited *exec.ExitError
			if !errors.As(err, &exited) || exited.ExitCode() != tc.exit {
				t.Fatalf("signal exit: got %v, want %d\n%s", err, tc.exit, output.Bytes())
			}
			if len(docker(t, "ps", "-aq", "--filter", filter)) != 0 || len(docker(t, "network", "ls", "-q", "--filter", filter)) != 0 {
				t.Fatal("interrupted proof left labelled Docker resources")
			}
		})
	}
}

func fixtureDeclaresAPIKeyEnv(abi []byte) bool {
	return regexp.MustCompile(`(?m)^  apiKeyEnv: "OPENAI_API_KEY"$`).Match(abi)
}

func TestFixtureAPIKeyEnv(t *testing.T) {
	for _, tc := range []struct {
		name string
		abi  string
		want bool
	}{
		{
			name: "declared",
			abi:  "model:\n  apiKeyEnv: \"OPENAI_API_KEY\"\n",
			want: true,
		},
		{
			name: "missing",
			abi:  "model:\n  name: \"host-model\"\n",
		},
		{
			name: "wrong_name",
			abi:  "model:\n  apiKeyEnv: \"OTHER_API_KEY\"\n",
		},
		{
			name: "wrong_field",
			abi:  "model:\n  otherKeyEnv: \"OPENAI_API_KEY\"\n",
		},
		{
			name: "missing_with_unrelated_name",
			abi:  "model:\n  name: \"host-model\"\ninstructions: \"OPENAI_API_KEY\"\n",
		},
		{
			name: "wrong_name_with_unrelated_name",
			abi:  "model:\n  apiKeyEnv: \"OTHER_API_KEY\"\ninstructions: \"OPENAI_API_KEY\"\n",
		},
		{
			name: "wrong_name_prefix",
			abi:  "model:\n  apiKeyEnv: \"OPENAI_API_KEY_OTHER\"\n",
		},
		{
			name: "pair_in_instructions",
			abi:  "model:\n  name: \"host-model\"\ninstructions: \"apiKeyEnv: \\\"OPENAI_API_KEY\\\"\"\n",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := fixtureDeclaresAPIKeyEnv([]byte(tc.abi)); got != tc.want {
				t.Fatalf("fixtureDeclaresAPIKeyEnv() = %t, want %t", got, tc.want)
			}
		})
	}
}

func TestContainerStatelessReplay(t *testing.T) {
	image := os.Getenv("AGENTKIT_AGENTSESSIONS_IMAGE")
	if image == "" {
		t.Skip("run scripts/agentsessions-e2e.sh for the built-image replay proof")
	}
	runID := os.Getenv("AGENTKIT_AGENTSESSIONS_RUN_ID")
	if runID == "" {
		runID = fmt.Sprintf("%d-%d", os.Getpid(), time.Now().UnixNano())
	}
	if valid, err := regexp.MatchString(`^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$`, runID); err != nil || !valid {
		t.Fatal("invalid test resource run ID")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	ctx = metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer "+harnessToken)
	imageDigest := string(docker(t, "image", "inspect", "--format", "{{.Id}}", image))
	// Inspect only this public fixture ABI in a container with no network.
	abi := docker(t, "run", "--rm", "--label", proofLabel+"="+runID, "--network", "none", "--entrypoint", "/usr/local/bin/python", imageDigest,
		"-c", "import pathlib,sys; sys.stdout.buffer.write(pathlib.Path('/agent/agent.yaml').read_bytes())")
	// Hash in-container so CLI whitespace trimming cannot change the exact-byte binding.
	configDigest := string(docker(t, "run", "--rm", "--label", proofLabel+"="+runID, "--network", "none", "--entrypoint", "/usr/local/bin/python", imageDigest,
		"-c", "import hashlib,pathlib; print('sha256:'+hashlib.sha256(pathlib.Path('/agent/agent.yaml').read_bytes()).hexdigest())"))
	if !fixtureDeclaresAPIKeyEnv(abi) {
		t.Fatal("fixture must declare only the absent OPENAI_API_KEY name")
	}
	if !bytes.Contains(abi, []byte("provider-must-not-be-used.invalid")) {
		t.Fatal("original model endpoint must be an unusable trap")
	}

	network := string(docker(t, "network", "create", "--internal", "--label", proofLabel+"="+runID, "agentkit-agentsessions-"+runID))
	t.Cleanup(func() { _ = exec.Command("docker", "network", "rm", network).Run() })
	if got := string(docker(t, "network", "inspect", "--format", "{{.Internal}}", network)); got != "true" {
		t.Fatal("model container must be on an internal, no-egress Docker network")
	}
	container, address := startContainer(t, imageDigest, network, configDigest, imageDigest, runID)
	liveClient, har := connectHarness(t, ctx, address)
	desc, err := har.Describe(ctx)
	if err != nil || desc.ID != "agentkit:"+configDigest+":"+imageDigest || !reflect.DeepEqual(desc.Models, []string{"host-model"}) || desc.Capabilities.Resumability != api.ResumabilityStatelessReplay {
		t.Fatalf("unexpected remote descriptor: %v", err)
	}
	assertNoProviderCredentials(t, container)

	path := filepath.Join(t.TempDir(), "sessions.db")
	store, err := sqlitelog.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	journal := store.Session("agentkit-text-session")
	configs := [][]byte{[]byte("\x00first-config\xff"), []byte("\x00second-config\xfe")}
	liveCalls := 0
	model := func(_ context.Context, req api.ModelRequest) (api.ModelResponse, error) {
		liveCalls++
		if req.Model != "host-model" {
			return api.ModelResponse{}, errors.New("wrong host model requested")
		}
		var got []string
		for _, message := range req.Messages {
			if message.Role == "system" {
				continue
			}
			got = append(got, message.Role+":"+message.Text())
		}
		want := []string{"user:same", "user:question"}
		if liveCalls == 2 {
			want = []string{"user:same", "user:question", "assistant:offline:question", "user:same", "user:question"}
		}
		if !reflect.DeepEqual(got, want) {
			return api.ModelResponse{}, fmt.Errorf("unexpected model conversation: got %q, want %q", got, want)
		}
		return api.ModelResponse{Message: *api.TextMessage("assistant", "offline:question")}, nil
	}
	for _, config := range configs {
		live, err := controller.New(journal, model, controller.WithStart(config, 17))
		if err != nil {
			t.Fatal(err)
		}
		records, err := journal.Read(1)
		if err != nil {
			t.Fatal(err)
		}
		if err := live.Exec(ctx, har, []api.Message{*api.TextMessage("user", "same"), *api.TextMessage("user", "question")}, int64(len(records))); err != nil {
			t.Fatalf("live execution: %v", err)
		}
	}
	before, err := journal.Read(1)
	if err != nil {
		t.Fatal(err)
	}
	var modelCalls, outputCount, startCount, endCount int
	var liveOutputs []string
	for _, record := range before {
		switch record.Event.Kind {
		case api.EventExecutionStart:
			if !bytes.Equal(record.Event.ExecutionStart.Config, configs[startCount]) || record.Event.ExecutionStart.ResumeFromSeq != 17 {
				t.Fatal("execution config/cursor was not journaled exactly")
			}
			startCount++
		case api.EventModelCall:
			modelCalls++
		case api.EventOutput:
			outputCount++
			liveOutputs = append(liveOutputs, record.Event.Message.Text())
		case api.EventEnd:
			endCount++
		}
	}
	if liveCalls != 2 || modelCalls != 2 || outputCount != 2 || startCount != 2 || endCount != 2 || len(before) != 12 {
		t.Fatalf("unexpected journal: calls=%d model=%d output=%d start=%d end=%d records=%d", liveCalls, modelCalls, outputCount, startCount, endCount, len(before))
	}
	if len(liveClient.starts) != 2 {
		t.Fatal("expected two live Start frames")
	}
	for i, start := range liveClient.starts {
		if !bytes.Equal(start.Config, configs[i]) || start.ResumeFromSeq != 17 {
			t.Fatal("live Config/cursor did not reach the remote wire")
		}
	}
	assertNoProviderCredentials(t, container)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	docker(t, "rm", "-f", container)
	container, address = startContainer(t, imageDigest, network, configDigest, imageDigest, runID)
	replayClient, replayHarness := connectHarness(t, ctx, address)
	assertNoProviderCredentials(t, container)
	store, err = sqlitelog.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	journal = store.Session("agentkit-text-session")
	replayCalls := 0
	replay, err := controller.New(journal, func(context.Context, api.ModelRequest) (api.ModelResponse, error) {
		replayCalls++
		return api.ModelResponse{}, errors.New("replay must not invoke a live model")
	}, controller.WithStart([]byte("wrong current config"), 999))
	if err != nil {
		t.Fatal(err)
	}
	outputs, err := replay.Replay(ctx, replayHarness)
	if err != nil {
		t.Fatalf("reconstruction after container/journal restart: %v", err)
	}
	if replayCalls != 0 || replay.ModelInvocations() != 0 || !reflect.DeepEqual(outputs, liveOutputs) {
		t.Fatalf("replay mismatch or model invocation: outputs=%q calls=%d", outputs, replayCalls)
	}
	if len(replayClient.starts) != 2 {
		t.Fatal("replay must actually execute the remote harness for both recorded turns")
	}
	for i, start := range replayClient.starts {
		if !bytes.Equal(start.Config, configs[i]) || start.ResumeFromSeq != 17 {
			t.Fatal("replay used current instead of journal-restored Config/cursor")
		}
	}
	if _, err := replay.Replay(ctx, changedInputs{Harness: replayHarness}); err == nil || !strings.Contains(err.Error(), "model input hash mismatch") {
		t.Fatalf("altered model request must fail replay fingerprint check: %v", err)
	}
	if replayCalls != 0 || replay.ModelInvocations() != 0 {
		t.Fatal("divergent replay called a live model")
	}
	after, err := journal.Read(1)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("replay changed journal: %v", err)
	}
	if err := journal.Verify(); err != nil {
		t.Fatal(err)
	}
	assertNoProviderCredentials(t, container)
	proof := map[string]any{
		"agentsessions_commit": hostPin, "image_digest": imageDigest, "harness_id": desc.ID,
		"live_model_calls": liveCalls, "journal_model_calls": modelCalls, "journal_outputs": outputCount,
		"replay_model_calls": replayCalls, "outputs_equal": true, "journal_unchanged": true,
		"execution_configs_recorded": true, "execution_configs_restored": true,
		"container_restarted": true, "journal_reopened": true, "model_input_mismatch_refused": true,
		"provider_credentials_absent": true, "egress_isolated": true,
	}
	raw, err := json.Marshal(proof)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("AGENTSESSIONS_REPLAY_PROOF=%s\n", raw)
	// Do not print the raw journal or any environment values.
}
