package toolbox

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/containerd/platforms"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/solver/pb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
	"github.com/orka-agents/agentkit/pkg/agentkit/toolbox"
)

const (
	testSHA256A    = "1111111111111111111111111111111111111111111111111111111111111111"
	testSHA256B    = "2222222222222222222222222222222222222222222222222222222222222222"
	testDigest     = "sha256:3333333333333333333333333333333333333333333333333333333333333333"
	testKubectl    = "registry.k8s.io/kubectl@" + testDigest
	testRuntime    = "ghcr.io/orka-agents/orka/acp-codex-runtime@" + testDigest
	testFrontendID = "docker-image://example.com/agentkit:test"
)

var (
	linuxAMD64 = platforms.MustParse("linux/amd64")
	linuxARM64 = platforms.Normalize(platforms.MustParse("linux/arm64"))
)

func testFrontend() Frontend {
	return Frontend{
		State:    llb.Image("example.com/agentkit:test"),
		Platform: linuxAMD64,
	}
}

func testToolbox(t *testing.T) *config.ToolboxConfig {
	t.Helper()
	doc, err := config.NewDocumentFromBytes([]byte(`apiVersion: v1alpha1
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
          linux/amd64: {url: https://example.com/yq_linux_amd64, sha256: "` + testSHA256A + `"}
          linux/arm64: {url: https://example.com/yq_linux_arm64, sha256: "` + testSHA256B + `"}
    - name: helm
      source:
        download:
          linux/amd64: {url: https://example.com/helm-linux-amd64.tar.gz, sha256: "` + testSHA256A + `", extract: linux-amd64/helm}
          linux/arm64: {url: https://example.com/helm-linux-arm64.tar.gz, sha256: "` + testSHA256B + `", extract: linux-arm64/helm}
    - name: kubectl
      source:
        image: ` + testKubectl + `
        path: /bin/kubectl
  verify:
    runtimeImage: ` + testRuntime + `
    commands:
      - yq --version
      - helm version --short
`))
	if err != nil {
		t.Fatal(err)
	}
	if err := doc.Toolbox.Validate(); err != nil {
		t.Fatal(err)
	}
	return doc.Toolbox
}

type definitionOps struct {
	execs         []*pb.ExecOp
	execPlatforms []*pb.Platform
	sources       []*pb.Op
	files         []*pb.FileOp
}

func marshalOps(t *testing.T, st llb.State) definitionOps {
	t.Helper()
	def, err := st.Marshal(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var out definitionOps
	for _, dt := range def.ToPB().Def {
		var op pb.Op
		if err := op.Unmarshal(dt); err != nil {
			t.Fatal(err)
		}
		switch {
		case op.GetExec() != nil:
			out.execs = append(out.execs, op.GetExec())
			out.execPlatforms = append(out.execPlatforms, op.Platform)
		case op.GetSource() != nil:
			out.sources = append(out.sources, &op)
		case op.GetFile() != nil:
			out.files = append(out.files, op.GetFile())
		}
	}
	return out
}

func (d definitionOps) source(t *testing.T, identifier string) *pb.Op {
	t.Helper()
	for _, op := range d.sources {
		if op.GetSource().Identifier == identifier {
			return op
		}
	}
	t.Fatalf("no source %q in %v", identifier, d.sources)
	return nil
}

func onlyExec(t *testing.T, d definitionOps) *pb.ExecOp {
	t.Helper()
	if len(d.execs) != 1 {
		t.Fatalf("exec ops = %d, want 1", len(d.execs))
	}
	return d.execs[0]
}

func mountAt(t *testing.T, exec *pb.ExecOp, dest string) *pb.Mount {
	t.Helper()
	for _, m := range exec.Mounts {
		if m.Dest == dest {
			return m
		}
	}
	t.Fatalf("no mount at %s", dest)
	return nil
}

func assertOffline(t *testing.T, exec *pb.ExecOp) {
	t.Helper()
	if exec.Network != pb.NetMode_NONE {
		t.Fatalf("network = %v, want none", exec.Network)
	}
	if root := mountAt(t, exec, "/"); !root.Readonly || root.Output != int64(pb.SkipOutput) {
		t.Fatalf("root mount = %+v, want read-only without output", root)
	}
}

func TestInstallStateDownloadsWithChecksumAndInstallsOffline(t *testing.T) {
	cfg := testToolbox(t)
	st, err := InstallState(testFrontend(), cfg.Spec.Tools[1], linuxARM64)
	if err != nil {
		t.Fatal(err)
	}
	ops := marshalOps(t, st)

	download := ops.source(t, "https://example.com/helm-linux-arm64.tar.gz").GetSource()
	if got := download.Attrs[pb.AttrHTTPChecksum]; got != "sha256:"+testSHA256B {
		t.Fatalf("checksum = %q", got)
	}
	if got := download.Attrs[pb.AttrHTTPFilename]; got != "helm" {
		t.Fatalf("filename = %q", got)
	}
	ops.source(t, testFrontendID)

	exec := onlyExec(t, ops)
	if p := ops.execPlatforms[0]; p.Architecture != "amd64" {
		t.Fatalf("install exec platform = %+v, want the worker's amd64", p)
	}
	// The install runs natively, so it uses the amd64 helper for an arm64 tool.
	wantArgs := []string{"/usr/libexec/agentkit/agentkit-toolbox-linux-amd64", toolbox.InstallCommand, "--src", "/src/helm", "--arch", linuxARM64.Architecture, "--dest", "/out", "--member", "linux-arm64/helm"}
	if !reflect.DeepEqual(exec.Meta.Args, wantArgs) {
		t.Fatalf("args = %q, want %q", exec.Meta.Args, wantArgs)
	}
	assertOffline(t, exec)
	if src := mountAt(t, exec, "/src"); !src.Readonly {
		t.Fatalf("/src mount = %+v, want read-only", src)
	}
	if out := mountAt(t, exec, "/out"); out.Readonly || out.Output == int64(pb.SkipOutput) {
		t.Fatalf("/out mount = %+v, want a writable output", out)
	}
}

func TestInstallStatePlainDownloadHasNoMember(t *testing.T) {
	cfg := testToolbox(t)
	st, err := InstallState(testFrontend(), cfg.Spec.Tools[0], linuxAMD64)
	if err != nil {
		t.Fatal(err)
	}
	exec := onlyExec(t, marshalOps(t, st))
	if strings.Contains(strings.Join(exec.Meta.Args, " "), "--member") {
		t.Fatalf("args = %q, want no --member", exec.Meta.Args)
	}
}

func TestInstallStateCopiesImageFileWithinImage(t *testing.T) {
	cfg := testToolbox(t)
	st, err := InstallState(testFrontend(), cfg.Spec.Tools[2], linuxARM64)
	if err != nil {
		t.Fatal(err)
	}
	ops := marshalOps(t, st)

	image := ops.source(t, "docker-image://"+testKubectl)
	if p := image.Platform; p == nil || p.Architecture != linuxARM64.Architecture || p.OS != "linux" {
		t.Fatalf("image platform = %+v, want linux/arm64", p)
	}
	var copied bool
	for _, file := range ops.files {
		for _, action := range file.Actions {
			if c := action.GetCopy(); c != nil {
				copied = c.Src == "/bin/kubectl" && c.Dest == "/kubectl" && c.FollowSymlink
			}
		}
	}
	if !copied {
		t.Fatalf("file ops = %v, want /bin/kubectl copied to /kubectl following symlinks", ops.files)
	}
	exec := onlyExec(t, ops)
	if got := strings.Join(exec.Meta.Args, " "); !strings.Contains(got, "--src /src/kubectl --arch arm64") {
		t.Fatalf("args = %q", got)
	}
	assertOffline(t, exec)
}

func TestInstallStateRequiresPlatformDownload(t *testing.T) {
	cfg := testToolbox(t)
	_, err := InstallState(testFrontend(), cfg.Spec.Tools[0], platforms.MustParse("linux/riscv64"))
	if err == nil || !strings.Contains(err.Error(), `tool "yq" has no download for linux/riscv64`) {
		t.Fatalf("InstallState error = %v", err)
	}
}

func installedTools(cfg *config.ToolboxConfig) []Tool {
	tools := make([]Tool, len(cfg.Spec.Tools))
	for i, tool := range cfg.Spec.Tools {
		tools[i] = Tool{
			ToolboxTool: tool,
			State:       llb.Image("example.com/installed-" + tool.Name),
			Result:      toolbox.Result{SHA256: fmt.Sprintf("%064d", i)},
		}
	}
	return tools
}

func TestAssembleLaysOutToolsInOneLayer(t *testing.T) {
	cfg := testToolbox(t)
	tools := installedTools(cfg)
	st, image, err := Assemble(cfg, tools, linuxAMD64)
	if err != nil {
		t.Fatal(err)
	}
	ops := marshalOps(t, st)
	if len(ops.files) != 1 {
		t.Fatalf("file ops = %d, want 1 (a single layer)", len(ops.files))
	}

	var got []string
	for _, action := range ops.files[0].Actions {
		switch {
		case action.GetMkdir() != nil:
			m := action.GetMkdir()
			got = append(got, fmt.Sprintf("mkdir %s %o parents=%v owner=%d", m.Path, m.Mode, m.MakeParents, m.Owner.User.GetByID()))
		case action.GetCopy() != nil:
			c := action.GetCopy()
			got = append(got, fmt.Sprintf("copy %s %s %o owner=%d:%d", c.Src, c.Dest, c.Mode, c.Owner.User.GetByID(), c.Owner.Group.GetByID()))
		case action.GetMkfile() != nil:
			f := action.GetMkfile()
			want, err := Inventory(cfg, tools, linuxAMD64)
			if err != nil || string(f.Data) != string(want) {
				t.Fatalf("inventory = %s, want %s (%v)", f.Data, want, err)
			}
			got = append(got, fmt.Sprintf("mkfile %s %o owner=%d", f.Path, f.Mode, f.Owner.User.GetByID()))
		}
	}
	want := []string{
		"mkdir /opt/yq-jq/bin 755 parents=true owner=0",
		"copy /tool /opt/yq-jq/bin/yq 555 owner=0:0",
		"copy /tool /opt/yq-jq/bin/helm 555 owner=0:0",
		"copy /tool /opt/yq-jq/bin/kubectl 555 owner=0:0",
		"mkfile /opt/yq-jq/.toolbox.json 444 owner=0",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("actions =\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
	if image.Architecture != "amd64" || image.OS != "linux" {
		t.Fatalf("image platform = %+v", image.Platform)
	}
}

func TestNewImageConfigHasOnlyGeneratedAndUserLabels(t *testing.T) {
	cfg := testToolbox(t)
	// Validation rejects ai.orka.* labels; the generated label still wins if
	// a config bypasses it.
	cfg.Metadata.Labels[config.ImageLabelOrkaToolbox] = "forged"
	image, err := NewImageConfig(cfg, linuxARM64)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		config.ImageLabelOrkaToolbox: `{"version":1,"mountPath":"/opt/yq-jq","pathEntries":["bin"]}`,
		config.ImageLabelOCITitle:    "yq-jq",
		"com.example.team":           "platform",
	}
	if !reflect.DeepEqual(image.Config.Labels, want) {
		t.Fatalf("labels = %v, want %v", image.Config.Labels, want)
	}
	image.Config.Labels = nil
	if !reflect.DeepEqual(image.Config, specs.ImageConfig{}) {
		t.Fatalf("config = %+v, want no entrypoint, user, env, or ports", image.Config)
	}
	if image.RootFS.Type != "layers" || image.Architecture != linuxARM64.Architecture {
		t.Fatalf("image = %+v", image)
	}
}

func TestInventoryListsEachTool(t *testing.T) {
	cfg := testToolbox(t)
	dt, err := Inventory(cfg, installedTools(cfg), linuxARM64)
	if err != nil {
		t.Fatal(err)
	}
	want := `{
  "version": 1,
  "name": "yq-jq",
  "platform": "linux/arm64",
  "tools": [
    {
      "name": "yq",
      "description": "YAML processor",
      "path": "bin/yq",
      "sha256": "` + fmt.Sprintf("%064d", 0) + `",
      "source": {
        "download": {
          "url": "https://example.com/yq_linux_arm64",
          "sha256": "` + testSHA256B + `"
        }
      }
    },
    {
      "name": "helm",
      "path": "bin/helm",
      "sha256": "` + fmt.Sprintf("%064d", 1) + `",
      "source": {
        "download": {
          "url": "https://example.com/helm-linux-arm64.tar.gz",
          "sha256": "` + testSHA256B + `",
          "extract": "linux-arm64/helm"
        }
      }
    },
    {
      "name": "kubectl",
      "path": "bin/kubectl",
      "sha256": "` + fmt.Sprintf("%064d", 2) + `",
      "source": {
        "image": "` + testKubectl + `",
        "path": "/bin/kubectl"
      }
    }
  ]
}
`
	if string(dt) != want {
		t.Fatalf("inventory =\n%s\nwant\n%s", dt, want)
	}
	if !json.Valid(dt) {
		t.Fatal("inventory is not valid JSON")
	}
}

func TestInventoryRequiresInstalledDigest(t *testing.T) {
	cfg := testToolbox(t)
	tools := installedTools(cfg)
	tools[1].Result.SHA256 = ""
	if _, err := Inventory(cfg, tools, linuxAMD64); err == nil || !strings.Contains(err.Error(), `tool "helm" has no installed SHA-256`) {
		t.Fatalf("Inventory error = %v", err)
	}
}

func TestVerifyStateRunsCommandsSandboxed(t *testing.T) {
	cfg := testToolbox(t)
	toolboxState, _, err := Assemble(cfg, installedTools(cfg), linuxARM64)
	if err != nil {
		t.Fatal(err)
	}
	// An amd64 worker verifying the arm64 toolbox.
	ops := marshalOps(t, VerifyState(testFrontend(), cfg, toolboxState, linuxARM64))

	// Solving the returned state must run the exec. A read-only root has no
	// output, so returning the exec's root would only pull the runtime image.
	exec := onlyExec(t, ops)
	runtime := ops.source(t, "docker-image://"+testRuntime)
	if runtime.Platform == nil || runtime.Platform.Architecture != linuxARM64.Architecture {
		t.Fatalf("runtime platform = %+v", runtime.Platform)
	}
	if p := ops.execPlatforms[0]; p.Architecture != linuxARM64.Architecture {
		t.Fatalf("verify exec platform = %+v, want arm64", p)
	}

	// The helper must be built for the exec's platform, not the worker's.
	wantArgs := []string{"/run/agentkit/agentkit-toolbox-linux-arm64", toolbox.VerifyCommand, "--timeout", "1m0s", "--", "yq --version", "helm version --short"}
	if !reflect.DeepEqual(exec.Meta.Args, wantArgs) {
		t.Fatalf("args = %q, want %q", exec.Meta.Args, wantArgs)
	}
	if exec.Meta.User != "65534:65534" || exec.Meta.Cwd != "/tmp" {
		t.Fatalf("user = %q, cwd = %q", exec.Meta.User, exec.Meta.Cwd)
	}
	for _, want := range []string{"PATH=/usr/local/bin:/usr/bin:/bin:/opt/yq-jq/bin", "HOME=/tmp"} {
		if !containsString(exec.Meta.Env, want) {
			t.Fatalf("env = %q, want %q", exec.Meta.Env, want)
		}
	}
	assertOffline(t, exec)
	if tmp := mountAt(t, exec, "/tmp"); tmp.MountType != pb.MountType_TMPFS {
		t.Fatalf("/tmp mount = %+v, want tmpfs", tmp)
	}
	if m := mountAt(t, exec, "/opt/yq-jq"); !m.Readonly || m.Selector != "/opt/yq-jq" {
		t.Fatalf("toolbox mount = %+v, want read-only /opt/yq-jq", m)
	}
	if m := mountAt(t, exec, "/run/agentkit"); !m.Readonly || m.Selector != toolbox.HelperDir {
		t.Fatalf("helper mount = %+v, want read-only %s of the frontend", m, toolbox.HelperDir)
	}
	if m := mountAt(t, exec, "/run/agentkit-verify"); m.Output == int64(pb.SkipOutput) {
		t.Fatalf("result mount = %+v, want an output", m)
	}
}

func containsString(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}

func TestInventoryOmitsURLQueryButFetchKeepsIt(t *testing.T) {
	cfg := testToolbox(t)
	const signed = "https://downloads.example.com/yq?X-Amz-Signature=secret-signature&token=secret-token#frag"
	tool := cfg.Spec.Tools[0]
	download := tool.Source.Download["linux/amd64"]
	download.URL = signed
	tool.Source.Download = map[string]config.ToolboxDownload{"linux/amd64": download}

	dt, err := Inventory(cfg, []Tool{{ToolboxTool: tool, Result: toolbox.Result{SHA256: testSHA256A}}}, linuxAMD64)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(dt), "secret") || strings.Contains(string(dt), "frag") {
		t.Fatalf("inventory leaks the URL query or fragment:\n%s", dt)
	}
	if !strings.Contains(string(dt), `"url": "https://downloads.example.com/yq"`) {
		t.Fatalf("inventory = %s, want the URL without its query", dt)
	}

	st, err := InstallState(testFrontend(), tool, linuxAMD64)
	if err != nil {
		t.Fatal(err)
	}
	marshalOps(t, st).source(t, signed)
}
