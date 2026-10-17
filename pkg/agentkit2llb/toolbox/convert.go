// Package toolbox converts a kind: Toolbox config into BuildKit LLB. Each tool
// is fetched with a checksum or a digest, then installed by the AgentKit
// frontend binary running from its own image with no network, so exactly one
// checked regular file per tool reaches the FROM scratch output.
package toolbox

import (
	"fmt"
	pathpkg "path"

	"github.com/containerd/platforms"
	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
	"github.com/orka-agents/agentkit/pkg/agentkit/toolbox"
)

const (
	// verifyUser owns nothing in the runtime image, like an Orka agent process.
	verifyUser = "65534:65534"
	// verifySystemPath mirrors the system PATH Orka gives agent processes;
	// toolbox folders come after it so they never replace system commands.
	verifySystemPath = "/usr/local/bin:/usr/bin:/bin"
	// verifyHelperDir is where the frontend binary is mounted for verify.
	verifyHelperDir = "/run/agentkit"
	// verifyResultDir is an unused writable mount. A read-only root has no
	// output, so solving this mount is what makes BuildKit run the commands.
	verifyResultDir = "/run/agentkit-verify"

	installSourceDir = "/src"
	installDestDir   = "/out"
)

// Frontend is the AgentKit frontend image. Its static binary runs the install
// and verify helpers inside the build.
type Frontend struct {
	State llb.State
	// Binary is the absolute path of the agentkit binary inside State.
	Binary string
	// Platform is the platform State's binary runs on.
	Platform specs.Platform
}

// Tool is one tool installed for one platform.
type Tool struct {
	config.ToolboxTool
	// State holds toolbox.ToolFile and toolbox.ResultFile at its root.
	State  llb.State
	Result toolbox.Result
}

// InstallState fetches tool for platform and installs it with the frontend
// binary. The returned state holds toolbox.ToolFile and toolbox.ResultFile.
func InstallState(fe Frontend, tool config.ToolboxTool, platform specs.Platform) (llb.State, error) {
	pf := platforms.Format(platform)
	sourceFile := pathpkg.Join(installSourceDir, tool.Name)
	args := []string{
		fe.Binary, toolbox.InstallCommand,
		"--src", sourceFile,
		"--arch", platform.Architecture,
		"--dest", installDestDir,
	}

	var src llb.State
	if tool.Source.Image != "" {
		image := llb.Image(tool.Source.Image, llb.Platform(platform), llb.WithCustomNamef("[%s] pull %s", pf, tool.Source.Image))
		// BuildKit resolves path, symlinks included, inside the image root, so
		// a symlink can never reach outside it. Install then requires a file.
		src = llb.Scratch().File(
			llb.Copy(image, tool.Source.Path, "/"+tool.Name, &llb.CopyInfo{FollowSymlinks: true}),
			llb.WithCustomNamef("[%s] copy %s from %s", pf, tool.Source.Path, tool.Source.Image),
		)
	} else {
		download, ok := tool.Source.DownloadFor(platform)
		if !ok {
			return llb.State{}, fmt.Errorf("tool %q has no download for %s", tool.Name, pf)
		}
		src = llb.HTTP(download.URL,
			llb.Checksum(download.Digest()),
			llb.Filename(tool.Name),
			llb.WithCustomNamef("[%s] download %s", pf, tool.Name),
		)
		if download.Extract != "" {
			args = append(args, "--member", download.Extract)
		}
	}

	run := fe.State.Run(
		llb.Args(args),
		llb.Platform(fe.Platform),
		llb.Network(llb.NetModeNone),
		llb.ReadonlyRootFS(),
		llb.AddMount(installSourceDir, src, llb.Readonly),
		llb.WithCustomNamef("[%s] install %s", pf, tool.Name),
	)
	return run.AddMount(installDestDir, llb.Scratch()), nil
}

// Assemble lays the installed tools out under the toolbox mount path in one
// layer and returns it with the toolbox image config.
func Assemble(cfg *config.ToolboxConfig, tools []Tool, platform specs.Platform) (llb.State, *specs.Image, error) {
	inventory, err := Inventory(cfg, tools, platform)
	if err != nil {
		return llb.State{}, nil, err
	}
	root := llb.WithUIDGID(0, 0)
	rootOwner := &llb.ChownOpt{User: &llb.UserOpt{UID: 0}, Group: &llb.UserOpt{UID: 0}}
	binDir := pathpkg.Join(cfg.MountPath(), config.ToolboxPathEntry)

	files := llb.Mkdir(binDir, 0o755, llb.WithParents(true), root)
	for _, tool := range tools {
		files = files.Copy(tool.State, "/"+toolbox.ToolFile, pathpkg.Join(binDir, tool.Name), &llb.CopyInfo{
			Mode:     &llb.ChmodOpt{Mode: 0o555},
			ChownOpt: rootOwner,
		})
	}
	files = files.Mkfile(pathpkg.Join(cfg.MountPath(), InventoryFile), 0o444, inventory, root)

	state := llb.Scratch().File(files, llb.WithCustomNamef("[%s] assemble %s", platforms.Format(platform), cfg.MountPath()))
	image, err := NewImageConfig(cfg, platform)
	if err != nil {
		return llb.State{}, nil, err
	}
	return state, image, nil
}

// VerifyState runs the verify commands inside the runtime image as a user that
// owns nothing, with no network, a read-only root, a per-command timeout, and
// the toolbox mounted read-only where Orka mounts it. Solving the returned
// state runs the commands; it never becomes part of an image.
func VerifyState(fe Frontend, cfg *config.ToolboxConfig, toolboxState llb.State, platform specs.Platform) llb.State {
	verify := cfg.Spec.Verify
	mountPath := cfg.MountPath()
	args := append([]string{
		pathpkg.Join(verifyHelperDir, pathpkg.Base(fe.Binary)), toolbox.VerifyCommand,
		"--timeout", toolbox.DefaultVerifyTimeout.String(),
		"--",
	}, verify.Commands...)

	runtime := llb.Image(verify.RuntimeImage, llb.Platform(platform),
		llb.WithCustomNamef("[%s] pull %s", platforms.Format(platform), verify.RuntimeImage))
	run := runtime.Run(
		llb.Args(args),
		llb.User(verifyUser),
		llb.Network(llb.NetModeNone),
		llb.ReadonlyRootFS(),
		llb.Dir("/tmp"),
		llb.AddEnv("HOME", "/tmp"),
		llb.AddEnv("PATH", verifySystemPath+":"+pathpkg.Join(mountPath, config.ToolboxPathEntry)),
		llb.AddMount("/tmp", llb.Scratch(), llb.Tmpfs()),
		llb.AddMount(mountPath, toolboxState, llb.SourcePath(mountPath), llb.Readonly),
		llb.AddMount(verifyHelperDir, fe.State, llb.SourcePath(pathpkg.Dir(fe.Binary)), llb.Readonly),
		llb.WithCustomNamef("[%s] verify %s in %s", platforms.Format(platform), mountPath, verify.RuntimeImage),
	)
	return run.AddMount(verifyResultDir, llb.Scratch())
}
