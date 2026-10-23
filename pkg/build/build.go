package build

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"math"
	"strings"

	"github.com/containerd/platforms"
	controlapi "github.com/moby/buildkit/api/services/control"
	"github.com/moby/buildkit/exporter/containerimage/exptypes"
	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/moby/buildkit/solver/errdefs"
	"github.com/moby/buildkit/solver/pb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
	"github.com/orka-agents/agentkit/pkg/agentkit/effective"
	agentllb "github.com/orka-agents/agentkit/pkg/agentkit2llb/agent"
	"github.com/pkg/errors"
	"golang.org/x/sync/errgroup"
)

const (
	localNameContext        = "context"
	localNameDockerfile     = "dockerfile"
	defaultAgentkitfileName = "agentkitfile.yaml"

	keyFilename       = "filename"
	keyTarget         = "target"
	keyTargetPlatform = "platform"
	keyCacheImports   = "cache-imports"
)

// Build is the AgentKit frontend entrypoint registered with BuildKit. It loads
// and validates the agentkitfile, resolves the <runtime>/<outputkind> route via
// the flat router, and dispatches to the route handler.
func Build(ctx context.Context, c client.Client) (*client.Result, error) {
	opts := c.BuildOpts().Opts

	loaded, err := loadAgentkitfile(ctx, c)
	if err != nil {
		return nil, errors.Wrap(err, "getting agentkitfile")
	}
	cfg := loaded.config

	if err := validateAgentConfig(loaded); err != nil {
		return nil, errors.Wrap(err, "validating agentkitfile")
	}

	target := opts[keyTarget]
	effectiveRuntime := canonicalEffectiveRuntime(cfg.Runtime)
	matched, route, rc, ok := lookupRoute(target, effectiveRuntime)
	if !ok {
		if targetRuntime, namesRuntime := targetRuntimeSegment(target); namesRuntime && targetRuntime != effectiveRuntime {
			return nil, errors.Errorf(
				"no route for target %q: target runtime %q does not match effective runtime %q",
				target,
				targetRuntime,
				effectiveRuntime,
			)
		}
		return nil, errors.Errorf("no route for target %q with effective runtime %q", target, effectiveRuntime)
	}
	if handler := contextRouteHandlers[matched]; handler != nil {
		return handler(ctx, c, cfg, rc, loaded.instructions)
	}
	return route.Handler(ctx, c, cfg, rc)
}

// validateAgentConfig runs the report-all config validation (plan §16.2 #3).
// Each problem is reported as "<file>:<line>: <message>" and carries its
// location, so BuildKit clients can show the surrounding lines of the file.
func validateAgentConfig(loaded *loadedAgentkitfile) error {
	err := loaded.config.Validate()
	if err == nil || loaded.source == nil {
		return err
	}
	return locateValidationErrors(err, loaded.filename, loaded.source)
}

func locateValidationErrors(err error, filename string, source []byte) error {
	info := &pb.SourceInfo{Filename: filename, Data: source, Language: "YAML"}
	located := config.LocateProblems(err, source)
	errs := make([]error, 0, len(located))
	for _, lp := range located {
		e := fmt.Errorf("%s:%d: %s", filename, lp.StartLine, lp.Message)
		errs = append(errs, errdefs.WithSource(e, &errdefs.Source{
			Info: info,
			Ranges: []*pb.Range{{
				Start: &pb.Position{Line: clampInt32(lp.StartLine), Character: clampInt32(lp.StartColumn)},
				End:   &pb.Position{Line: clampInt32(lp.EndLine), Character: clampInt32(lp.EndColumn)},
			}},
		}))
	}
	return stderrors.Join(errs...)
}

// clampInt32 converts a line or column number for BuildKit's int32 positions.
func clampInt32(n int) int32 {
	if n > math.MaxInt32 {
		return math.MaxInt32
	}
	if n < 0 {
		return 0
	}
	return int32(n)
}

// HandleAgent is the image-output route handler. It resolves instructions,
// then solves the agent image for every target platform in parallel — mirroring
// AIKit's buildInference multi-platform errgroup.
func HandleAgent(ctx context.Context, c client.Client, cfg *config.AgentConfig, rc *RuntimeConfig) (*client.Result, error) {
	return handleAgent(ctx, c, cfg, rc, localContextReader{client: c})
}

func handleAgent(ctx context.Context, c client.Client, cfg *config.AgentConfig, rc *RuntimeConfig, reader contextFileReader) (*client.Result, error) {
	opts := c.BuildOpts().Opts

	cacheImports, err := parseCacheOptions(opts)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse cache import options")
	}

	// Resolve instructions (inline → as-is; file → read from build context) BEFORE
	// converting, so the baked agent.yaml carries a fully-resolved scalar (ABI).
	instructions, err := resolveInstructionSource(ctx, reader, cfg.Instructions)
	if err != nil {
		return nil, errors.Wrap(err, "resolving instructions")
	}

	agentSpec := effective.FromConfig(cfg, instructions)

	adapterRef := rc.AdapterRef(opts)

	// Default the build platform to the buildkit host's os/arch, preferring the
	// first worker's platform (AIKit parity).
	defaultBuildPlatform := platforms.DefaultSpec()
	if workers := c.BuildOpts().Workers; len(workers) > 0 && len(workers[0].Platforms) > 0 {
		defaultBuildPlatform = workers[0].Platforms[0]
	}

	targetPlatforms := []*specs.Platform{&defaultBuildPlatform}
	if platform, exists := opts[keyTargetPlatform]; exists && platform != "" {
		targetPlatforms, err = parsePlatforms(platform)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to parse target platforms %s", platform)
		}
	}

	isMultiPlatform := len(targetPlatforms) > 1
	exportPlatforms := &exptypes.Platforms{
		Platforms: make([]exptypes.Platform, len(targetPlatforms)),
	}
	finalResult := client.NewResult()

	eg, ctx := errgroup.WithContext(ctx)
	for i, tp := range targetPlatforms {
		func(i int, platform *specs.Platform) {
			eg.Go(func() (err error) {
				result, err := buildImage(ctx, c, agentSpec, adapterRef, platform, isMultiPlatform, cacheImports)
				if err != nil {
					return errors.Wrap(err, "failed to build image")
				}
				result.AddToClientResult(finalResult)
				exportPlatforms.Platforms[i] = result.ExportPlatform
				return nil
			})
		}(i, tp)
	}
	if err := eg.Wait(); err != nil {
		return nil, err
	}

	if isMultiPlatform {
		dt, err := json.Marshal(exportPlatforms)
		if err != nil {
			return nil, err
		}
		finalResult.AddMeta(exptypes.ExporterPlatformsKey, dt)
	}

	return finalResult, nil
}

// buildResult is the result of a single-platform image build (AIKit parity).
type buildResult struct {
	Reference      client.Reference
	ImageConfig    []byte
	Platform       *specs.Platform
	MultiPlatform  bool
	ExportPlatform exptypes.Platform
}

// AddToClientResult wires the build result into a client result (AIKit parity).
func (br *buildResult) AddToClientResult(cr *client.Result) {
	if br.MultiPlatform {
		cr.AddMeta(exptypes.ExporterImageConfigKey+"/"+br.ExportPlatform.ID, br.ImageConfig)
		cr.AddRef(br.ExportPlatform.ID, br.Reference)
	} else {
		cr.AddMeta(exptypes.ExporterImageConfigKey, br.ImageConfig)
		cr.SetRef(br.Reference)
	}
}

// buildImage converts the effective Agent to LLB for one platform and solves it.
func buildImage(ctx context.Context, c client.Client, agentSpec effective.Agent, adapterRef string, platform *specs.Platform, multiPlatform bool, cacheImports []client.CacheOptionsEntry) (*buildResult, error) {
	result := buildResult{Platform: platform, MultiPlatform: multiPlatform}

	state, image, err := agentllb.Agentkit2LLB(agentSpec, adapterRef, platform)
	if err != nil {
		return nil, err
	}

	result.ImageConfig, err = json.Marshal(image)
	if err != nil {
		return nil, errors.Wrap(err, "failed to marshal image config")
	}

	def, err := state.Marshal(ctx)
	if err != nil {
		return nil, errors.Wrap(err, "failed to marshal definition")
	}

	res, err := c.Solve(ctx, client.SolveRequest{
		Definition:   def.ToPB(),
		CacheImports: cacheImports,
	})
	if err != nil {
		return nil, errors.Wrap(err, "failed to solve")
	}

	result.Reference, err = res.SingleRef()
	if err != nil {
		return nil, err
	}

	result.ExportPlatform = exptypes.Platform{Platform: platforms.DefaultSpec()}
	if result.Platform != nil {
		result.ExportPlatform.Platform = *result.Platform
	}
	result.ExportPlatform.ID = platforms.Format(result.ExportPlatform.Platform)

	return &result, nil
}

// getBuildArg returns the value of build-arg:<k>, or "".
func getBuildArg(opts map[string]string, k string) string {
	if opts != nil {
		if v, ok := opts["build-arg:"+k]; ok {
			return v
		}
	}
	return ""
}

// parseBuildArgs applies build-arg overrides to the config. v0 supports only
// `runtime` (agents do not take a model-from-arg like AIKit).
func parseBuildArgs(opts map[string]string, cfg *config.AgentConfig) error {
	if cfg == nil {
		return nil
	}
	if rt := getBuildArg(opts, "runtime"); rt != "" {
		cfg.Runtime = rt
	}
	return nil
}

// parsePlatforms parses a comma-separated list of platforms (AIKit parity).
func parsePlatforms(v string) ([]*specs.Platform, error) {
	var pp []*specs.Platform
	for _, p := range strings.Split(v, ",") {
		parsed, err := platforms.Parse(p)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to parse target platform %s", p)
		}
		parsed = platforms.Normalize(parsed)
		pp = append(pp, &parsed)
	}
	return pp, nil
}

// parseCacheOptions handles given cache imports (AIKit parity).
func parseCacheOptions(opts map[string]string) ([]client.CacheOptionsEntry, error) {
	var cacheImports []client.CacheOptionsEntry
	if cacheImportsStr := opts[keyCacheImports]; cacheImportsStr != "" {
		var cacheImportsUM []*controlapi.CacheOptionsEntry
		if err := json.Unmarshal([]byte(cacheImportsStr), &cacheImportsUM); err != nil {
			return nil, errors.Wrapf(err, "failed to unmarshal %s", keyCacheImports)
		}
		for i, um := range cacheImportsUM {
			if um == nil {
				return nil, errors.Errorf("%s entry %d is null", keyCacheImports, i)
			}
			cacheImports = append(cacheImports, client.CacheOptionsEntry{Type: um.Type, Attrs: um.Attrs})
		}
	}
	return cacheImports, nil
}
