package build

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/containerd/platforms"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/frontend/gateway/client"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
	"github.com/orka-agents/agentkit/pkg/agentkit/toolbox"
	toolboxllb "github.com/orka-agents/agentkit/pkg/agentkit2llb/toolbox"
	"github.com/orka-agents/agentkit/pkg/utils"
	"github.com/pkg/errors"
	"golang.org/x/sync/errgroup"
)

const (
	// toolboxTargetVerify builds the toolbox, runs spec.verify against it, and
	// outputs the same image the default target does.
	toolboxTargetVerify = "verify"
)

// currentFrontendClient is implemented by BuildKit's gateway client, which
// can return the definition of the frontend image it is running from.
type currentFrontendClient interface {
	CurrentFrontend() (*llb.State, error)
}

// handleToolbox builds a kind: Toolbox image for every target platform.
func handleToolbox(ctx context.Context, c client.Client, cfg *config.ToolboxConfig) (*client.Result, error) {
	opts := c.BuildOpts().Opts
	target := opts[keyTarget]
	if target != "" && target != toolboxTargetVerify {
		return nil, errors.Errorf("no route for target %q: kind Toolbox supports the default target and %q", target, toolboxTargetVerify)
	}
	verify := target == toolboxTargetVerify
	if verify && cfg.Spec.Verify == nil {
		return nil, errors.Errorf("target %q requires spec.verify", toolboxTargetVerify)
	}

	cacheImports, err := parseCacheOptions(opts)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse cache import options")
	}
	targetPlatforms, err := requestedPlatforms(c)
	if err != nil {
		return nil, err
	}
	if err := checkToolboxPlatforms(cfg, targetPlatforms); err != nil {
		return nil, err
	}
	fe, err := currentFrontend(c)
	if err != nil {
		return nil, err
	}

	return buildPlatforms(ctx, targetPlatforms, func(ctx context.Context, platform *specs.Platform, multiPlatform bool) (*buildResult, error) {
		return buildToolboxImage(ctx, c, cfg, fe, platform, multiPlatform, verify, cacheImports)
	})
}

// checkToolboxPlatforms reports every tool without a download for a target
// platform before anything is fetched.
func checkToolboxPlatforms(cfg *config.ToolboxConfig, targetPlatforms []*specs.Platform) error {
	var missing []string
	for _, platform := range targetPlatforms {
		if !config.IsToolboxPlatform(*platform) {
			missing = append(missing, fmt.Sprintf("platform %s is not supported; toolboxes are built for %s",
				platforms.Format(*platform), strings.Join(config.ToolboxPlatforms, ", ")))
			continue
		}
		for _, tool := range cfg.Spec.Tools {
			if tool.Source.Image != "" {
				continue
			}
			if _, ok := tool.Source.DownloadFor(*platform); !ok {
				missing = append(missing, fmt.Sprintf("tool %q has no download for %s (it has %s)",
					tool.Name, platforms.Format(*platform), strings.Join(tool.Source.DownloadPlatforms(), ", ")))
			}
		}
	}
	if len(missing) > 0 {
		return errors.Errorf("toolbox cannot be built for the requested platforms:\n  %s", strings.Join(missing, "\n  "))
	}
	return nil
}

func currentFrontend(c client.Client) (toolboxllb.Frontend, error) {
	cf, ok := c.(currentFrontendClient)
	if !ok {
		return toolboxllb.Frontend{}, errors.New("kind Toolbox needs a BuildKit gateway client that exposes the frontend image")
	}
	state, err := cf.CurrentFrontend()
	if err != nil {
		return toolboxllb.Frontend{}, errors.Wrap(err, "loading the AgentKit frontend image")
	}
	if state == nil {
		return toolboxllb.Frontend{}, errors.New("kind Toolbox must be built through the AgentKit frontend image (#syntax=...); BuildKit did not provide it")
	}
	// The frontend runs in a Linux container on the worker, so its own
	// architecture is the worker's native one.
	native := platforms.DefaultSpec()
	native.OS = utils.PlatformLinux
	native = platforms.Normalize(native)
	if !config.IsToolboxPlatform(native) {
		return toolboxllb.Frontend{}, errors.Errorf("kind Toolbox needs a BuildKit worker running on %s, not %s",
			strings.Join(config.ToolboxPlatforms, " or "), platforms.Format(native))
	}
	return toolboxllb.Frontend{State: *state, Platform: native}, nil
}

// buildToolboxImage installs every tool for one platform, records each
// installed file's digest, and assembles the image. The verify target also
// runs spec.verify against that exact image before returning it unchanged.
func buildToolboxImage(ctx context.Context, c client.Client, cfg *config.ToolboxConfig, fe toolboxllb.Frontend, platform *specs.Platform, multiPlatform, verify bool, cacheImports []client.CacheOptionsEntry) (*buildResult, error) {
	pf := platforms.Format(*platform)
	tools := make([]toolboxllb.Tool, len(cfg.Spec.Tools))
	eg, egCtx := errgroup.WithContext(ctx)
	for i, tool := range cfg.Spec.Tools {
		eg.Go(func() error {
			installed, err := installTool(egCtx, c, fe, tool, *platform, cacheImports)
			if err != nil {
				return errors.Wrapf(err, "installing tool %q for %s", tool.Name, pf)
			}
			tools[i] = installed
			return nil
		})
	}
	if err := eg.Wait(); err != nil {
		return nil, err
	}

	state, image, err := toolboxllb.Assemble(cfg, tools, *platform)
	if err != nil {
		return nil, err
	}
	if verify {
		verifyState := toolboxllb.VerifyState(fe, cfg, state, *platform)
		if _, err := solveState(ctx, c, verifyState, cacheImports, true); err != nil {
			return nil, errors.Wrapf(err, "verifying toolbox for %s in %s", pf, cfg.Spec.Verify.RuntimeImage)
		}
	}
	return solveImage(ctx, c, state, image, platform, multiPlatform, cacheImports)
}

func installTool(ctx context.Context, c client.Client, fe toolboxllb.Frontend, tool config.ToolboxTool, platform specs.Platform, cacheImports []client.CacheOptionsEntry) (toolboxllb.Tool, error) {
	state, err := toolboxllb.InstallState(fe, tool, platform)
	if err != nil {
		return toolboxllb.Tool{}, err
	}
	ref, err := solveState(ctx, c, state, cacheImports, false)
	if err != nil {
		return toolboxllb.Tool{}, err
	}
	dt, err := ref.ReadFile(ctx, client.ReadRequest{Filename: toolbox.ResultFile})
	if err != nil {
		return toolboxllb.Tool{}, errors.Wrap(err, "reading install result")
	}
	installed := toolboxllb.Tool{ToolboxTool: tool, State: state}
	if err := json.Unmarshal(dt, &installed.Result); err != nil {
		return toolboxllb.Tool{}, errors.Wrap(err, "decoding install result")
	}
	return installed, nil
}

func solveState(ctx context.Context, c client.Client, state llb.State, cacheImports []client.CacheOptionsEntry, evaluate bool) (client.Reference, error) {
	def, err := state.Marshal(ctx)
	if err != nil {
		return nil, errors.Wrap(err, "failed to marshal definition")
	}
	res, err := c.Solve(ctx, client.SolveRequest{
		Definition:   def.ToPB(),
		CacheImports: cacheImports,
		Evaluate:     evaluate,
	})
	if err != nil {
		return nil, err
	}
	return res.SingleRef()
}
