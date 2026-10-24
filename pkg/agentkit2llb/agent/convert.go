// Package agent converts an effective Agent into a BuildKit LLB graph plus
// an OCI image config.
package agent

import (
	"os"

	"github.com/moby/buildkit/client/llb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/orka-agents/agentkit/pkg/agentkit/abi"
	"github.com/orka-agents/agentkit/pkg/agentkit/effective"
	"github.com/orka-agents/agentkit/pkg/utils"
)

// Composition describes resolved, platform-specific image inputs for the
// experimental base-image plus harness-image build path.
type Composition struct {
	BaseRef       string
	HarnessRef    string
	ArtifactPath  string
	User          string
	UID           int
	GID           int
	BaseImage     *specs.Image
	BaseName      string
	BaseDigest    string
	HarnessName   string
	HarnessDigest string
}

// Agentkit2LLB converts an effective Agent to an LLB state and image config. The
// adapter image (agentkit-serve) is the base; the only delta merged on top is
// the baked agent.yaml.
func Agentkit2LLB(agent effective.Agent, adapterRef string, platform *specs.Platform) (llb.State, *specs.Image, error) {
	if platform == nil {
		return llb.State{}, nil, os.ErrInvalid
	}

	agentYAML, err := abi.Render(agent)
	if err != nil {
		return llb.State{}, nil, err
	}

	state := llb.Image(adapterRef, llb.Platform(*platform)).File(
		llb.Mkdir("/agent", 0o755, llb.WithParents(true)).
			Mkfile(abi.Path, 0o644, agentYAML),
	)

	return state, NewImageConfig(agent, platform), nil
}

// ComposeAgentkit2LLB builds an agent rootfs from a user-selected base image and
// a label-described harness artifact. AgentKit controls the final artifact
// destination, configuration path, ownership, and OCI execution metadata.
func ComposeAgentkit2LLB(agent effective.Agent, composition Composition, platform *specs.Platform) (llb.State, *specs.Image, error) {
	if platform == nil || composition.BaseImage == nil {
		return llb.State{}, nil, os.ErrInvalid
	}

	agentYAML, err := abi.Render(agent)
	if err != nil {
		return llb.State{}, nil, err
	}

	harness := llb.Image(composition.HarnessRef, llb.Platform(*platform))
	state := llb.Image(composition.BaseRef, llb.Platform(*platform)).File(
		llb.Copy(
			harness,
			composition.ArtifactPath,
			utils.AgentKitRoot,
			&llb.CopyInfo{
				CopyDirContentsOnly: true,
				CreateDestPath:      true,
			},
			llb.WithUIDGID(composition.UID, composition.GID),
		),
	).File(
		llb.Mkdir(
			"/agent",
			0o755,
			llb.WithParents(true),
			llb.WithUIDGID(composition.UID, composition.GID),
		).Mkfile(
			abi.Path,
			0o644,
			agentYAML,
			llb.WithUIDGID(composition.UID, composition.GID),
		),
	)

	return state, NewComposedImageConfig(agent, platform, composition), nil
}
