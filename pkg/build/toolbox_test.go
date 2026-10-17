package build

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/exporter/containerimage/exptypes"
	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/moby/buildkit/solver/pb"
	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
	"github.com/orka-agents/agentkit/pkg/agentkit/toolbox"
)

const (
	toolboxTestDigest = "sha256:3333333333333333333333333333333333333333333333333333333333333333"
	toolboxLabel      = `{"version":1,"mountPath":"/opt/yq-jq","pathEntries":["bin"]}`
	toolboxAMD64      = "linux/amd64"
	toolboxARM64      = "linux/arm64"
)

func toolboxAgentkitfile(verify bool) []byte {
	in := `apiVersion: v1alpha1
kind: Toolbox
metadata:
  name: yq-jq
spec:
  tools:
    - name: yq
      source:
        download:
          linux/amd64: {url: https://example.com/yq_linux_amd64, sha256: "1111111111111111111111111111111111111111111111111111111111111111"}
          linux/arm64: {url: https://example.com/yq_linux_arm64, sha256: "2222222222222222222222222222222222222222222222222222222222222222"}
    - name: helm
      source:
        download:
          linux/amd64: {url: https://example.com/helm-linux-amd64.tar.gz, sha256: "1111111111111111111111111111111111111111111111111111111111111111", extract: linux-amd64/helm}
          linux/arm64: {url: https://example.com/helm-linux-arm64.tar.gz, sha256: "2222222222222222222222222222222222222222222222222222222222222222", extract: linux-arm64/helm}
    - name: kubectl
      source:
        image: registry.k8s.io/kubectl@` + toolboxTestDigest + `
        path: /bin/kubectl
`
	if verify {
		in += `  verify:
    runtimeImage: debian@` + toolboxTestDigest + `
    commands: [yq --version]
`
	}
	return []byte(in)
}

// toolboxClient is a gateway client that loads one agentkitfile, answers
// install solves with a result.json, and records every solve.
type toolboxClient struct {
	client.Client

	opts         client.BuildOpts
	agentkitfile []byte
	frontend     *llb.State
	failTool     string

	mu     sync.Mutex
	solves []client.SolveRequest
}

func newToolboxClient(agentkitfile []byte, opts map[string]string) *toolboxClient {
	frontend := llb.Image("example.com/agentkit:test")
	return &toolboxClient{
		opts:         client.BuildOpts{Opts: opts},
		agentkitfile: agentkitfile,
		frontend:     &frontend,
	}
}

func (c *toolboxClient) BuildOpts() client.BuildOpts {
	return c.opts
}

func (c *toolboxClient) CurrentFrontend() (*llb.State, error) {
	return c.frontend, nil
}

func (c *toolboxClient) Solve(_ context.Context, req client.SolveRequest) (*client.Result, error) {
	c.mu.Lock()
	c.solves = append(c.solves, req)
	c.mu.Unlock()

	sources, err := definitionSources(req.Definition)
	if err != nil {
		return nil, err
	}
	ref := &memoryReference{files: map[string][]byte{}}
	switch args := terminalExecArgs(req.Definition); {
	case hasSourcePrefix(sources, localDockerfileSourcePrefix):
		ref.files[defaultAgentkitfileName] = c.agentkitfile
	case strings.Contains(args, toolbox.InstallCommand):
		if c.failTool != "" && strings.Contains(args, "/src/"+c.failTool+" ") {
			return nil, errors.New("process did not complete successfully: exit code: 1")
		}
		sum := sha256.Sum256([]byte(args))
		dt, err := json.Marshal(toolbox.Result{SHA256: hex.EncodeToString(sum[:])})
		if err != nil {
			return nil, err
		}
		ref.files[toolbox.ResultFile] = dt
	}
	result := client.NewResult()
	result.SetRef(ref)
	return result, nil
}

// solvesWith returns the solves whose result is an exec running command.
func (c *toolboxClient) solvesWith(command string) []client.SolveRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []client.SolveRequest
	for _, req := range c.solves {
		if strings.Contains(terminalExecArgs(req.Definition), command) {
			out = append(out, req)
		}
	}
	return out
}

func (c *toolboxClient) solveCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.solves)
}

// terminalExecArgs returns the args of the exec a definition resolves to, or
// "" when its result is not an exec.
func terminalExecArgs(def *pb.Definition) string {
	if def == nil || len(def.Def) == 0 {
		return ""
	}
	ops := make(map[digest.Digest]*pb.Op, len(def.Def))
	for _, dt := range def.Def {
		var op pb.Op
		if err := op.Unmarshal(dt); err != nil {
			return ""
		}
		ops[digest.FromBytes(dt)] = &op
	}
	var last pb.Op
	if err := last.Unmarshal(def.Def[len(def.Def)-1]); err != nil || len(last.Inputs) != 1 {
		return ""
	}
	if exec := ops[digest.Digest(last.Inputs[0].Digest)].GetExec(); exec != nil {
		return strings.Join(exec.Meta.Args, " ")
	}
	return ""
}

func imageConfigFromMeta(t *testing.T, res *client.Result, key string) specs.Image {
	t.Helper()
	dt, ok := res.Metadata[key]
	if !ok {
		t.Fatalf("result metadata has no %s: %v", key, res.Metadata)
	}
	var image specs.Image
	if err := json.Unmarshal(dt, &image); err != nil {
		t.Fatal(err)
	}
	return image
}

func TestBuildToolboxMultiPlatform(t *testing.T) {
	c := newToolboxClient(toolboxAgentkitfile(true), map[string]string{keyTargetPlatform: toolboxAMD64 + "," + toolboxARM64})
	res, err := Build(context.Background(), c)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	var exported exptypes.Platforms
	if err := json.Unmarshal(res.Metadata[exptypes.ExporterPlatformsKey], &exported); err != nil {
		t.Fatal(err)
	}
	for _, platform := range []string{toolboxAMD64, toolboxARM64} {
		if res.Refs[platform] == nil {
			t.Fatalf("no ref for %s: %v", platform, res.Refs)
		}
		image := imageConfigFromMeta(t, res, exptypes.ExporterImageConfigKey+"/"+platform)
		if got := image.Config.Labels[config.ImageLabelOrkaToolbox]; got != toolboxLabel {
			t.Fatalf("%s label = %q", platform, got)
		}
		if image.OS+"/"+image.Architecture != platform {
			t.Fatalf("image platform = %s/%s, want %s", image.OS, image.Architecture, platform)
		}
	}
	if len(exported.Platforms) != 2 {
		t.Fatalf("exported platforms = %+v", exported.Platforms)
	}

	installs := c.solvesWith(toolbox.InstallCommand)
	if len(installs) != 6 {
		t.Fatalf("install solves = %d, want 3 tools x 2 platforms", len(installs))
	}
	if verifies := c.solvesWith(toolbox.VerifyCommand); len(verifies) != 0 {
		t.Fatalf("default target ran %d verify solves", len(verifies))
	}
	for _, want := range []string{"--arch amd64 --dest /out --member linux-amd64/helm", "--arch arm64 --dest /out --member linux-arm64/helm", "--src /src/kubectl --arch arm64"} {
		var found bool
		for _, req := range installs {
			found = found || strings.Contains(terminalExecArgs(req.Definition), want)
		}
		if !found {
			t.Fatalf("no install solve with %q", want)
		}
	}
}

func TestBuildToolboxVerifyTargetEvaluatesAndKeepsImage(t *testing.T) {
	build := func(target string) (*toolboxClient, *client.Result) {
		t.Helper()
		c := newToolboxClient(toolboxAgentkitfile(true), map[string]string{keyTargetPlatform: toolboxAMD64, keyTarget: target})
		res, err := Build(context.Background(), c)
		if err != nil {
			t.Fatalf("Build(target=%q): %v", target, err)
		}
		return c, res
	}
	defaultClient, defaultResult := build("")
	verifyClient, verifyResult := build(toolboxTargetVerify)

	verifies := verifyClient.solvesWith(toolbox.VerifyCommand)
	if len(verifies) != 1 || !verifies[0].Evaluate {
		t.Fatalf("verify solves = %+v, want one evaluated solve", verifies)
	}

	// The last solve is the image: it must be identical with and without verify.
	defaultImage := defaultClient.solves[len(defaultClient.solves)-1].Definition
	verifyImage := verifyClient.solves[len(verifyClient.solves)-1].Definition
	if strings.Contains(terminalExecArgs(verifyImage), toolbox.VerifyCommand) {
		t.Fatal("the image definition resolves to the verify stage")
	}
	defaultDef, _ := json.Marshal(defaultImage.Def)
	verifyDef, _ := json.Marshal(verifyImage.Def)
	if string(defaultDef) != string(verifyDef) {
		t.Fatal("the verify target changed the image definition")
	}
	if string(defaultResult.Metadata[exptypes.ExporterImageConfigKey]) != string(verifyResult.Metadata[exptypes.ExporterImageConfigKey]) {
		t.Fatal("the verify target changed the image config")
	}
}

func TestBuildToolboxRejectsBeforeFetching(t *testing.T) {
	tests := []struct {
		name         string
		agentkitfile []byte
		opts         map[string]string
		noFrontend   bool
		want         string
	}{
		{
			name: "missing platform download",
			agentkitfile: []byte(strings.Replace(string(toolboxAgentkitfile(false)),
				"          linux/arm64: {url: https://example.com/helm-linux-arm64.tar.gz", "          # no arm64 helm: {url: https://example.com/helm-linux-arm64.tar.gz", 1)),
			opts: map[string]string{keyTargetPlatform: toolboxAMD64 + "," + toolboxARM64},
			want: `tool "helm" has no download for linux/arm64 (it has linux/amd64)`,
		},
		{
			name:         "unsupported architecture",
			agentkitfile: toolboxAgentkitfile(false),
			opts:         map[string]string{keyTargetPlatform: toolboxAMD64 + ",linux/riscv64"},
			want:         "platform linux/riscv64 is not supported; toolboxes are built for linux/amd64, linux/arm64",
		},
		{
			name:         "non-linux platform",
			agentkitfile: toolboxAgentkitfile(false),
			opts:         map[string]string{keyTargetPlatform: "windows/amd64"},
			want:         "platform windows/amd64 is not supported; toolboxes are built for linux/amd64, linux/arm64",
		},
		{
			name:         "verify without spec.verify",
			agentkitfile: toolboxAgentkitfile(false),
			opts:         map[string]string{keyTarget: toolboxTargetVerify},
			want:         `target "verify" requires spec.verify`,
		},
		{
			name:         "agent route target",
			agentkitfile: toolboxAgentkitfile(false),
			opts:         map[string]string{keyTarget: "pydantic-ai/image"},
			want:         `no route for target "pydantic-ai/image": kind Toolbox supports the default target and "verify"`,
		},
		{
			name:         "invalid toolbox",
			agentkitfile: []byte(strings.Replace(string(toolboxAgentkitfile(false)), "name: yq-jq", "name: YQ", 1)),
			want:         `validating agentkitfile: metadata.name "YQ" must be`,
		},
		{
			name:         "frontend image unavailable",
			agentkitfile: toolboxAgentkitfile(false),
			opts:         map[string]string{keyTargetPlatform: toolboxAMD64},
			noFrontend:   true,
			want:         "kind Toolbox must be built through the AgentKit frontend image",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := newToolboxClient(tt.agentkitfile, tt.opts)
			if tt.noFrontend {
				c.frontend = nil
			}
			_, err := Build(context.Background(), c)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Build error = %v, want it to contain %q", err, tt.want)
			}
			if got := c.solveCount(); got != 1 {
				t.Fatalf("solves = %d, want only the agentkitfile load", got)
			}
		})
	}
}

func TestBuildToolboxInstallErrorNamesToolAndPlatform(t *testing.T) {
	c := newToolboxClient(toolboxAgentkitfile(false), map[string]string{keyTargetPlatform: toolboxARM64})
	c.failTool = "helm"
	_, err := Build(context.Background(), c)
	if err == nil || !strings.Contains(err.Error(), `installing tool "helm" for linux/arm64`) {
		t.Fatalf("Build error = %v", err)
	}
}

func TestBuildToolboxDefaultsToWorkerPlatform(t *testing.T) {
	c := newToolboxClient(toolboxAgentkitfile(false), map[string]string{})
	c.opts.Workers = []client.WorkerInfo{{Platforms: []specs.Platform{{OS: "linux", Architecture: "arm64"}}}}
	res, err := Build(context.Background(), c)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if image := imageConfigFromMeta(t, res, exptypes.ExporterImageConfigKey); image.Architecture != "arm64" {
		t.Fatalf("image architecture = %q, want the worker's arm64", image.Architecture)
	}
	if n := len(c.solvesWith("--arch arm64")); n != 3 {
		t.Fatalf("arm64 install solves = %d, want 3", n)
	}
	if res.Ref == nil {
		t.Fatalf("single-platform result has no ref: %+v", res)
	}
}
