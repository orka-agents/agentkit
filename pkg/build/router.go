package build

import (
	"context"
	"sort"
	"strings"

	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
	"github.com/orka-agents/agentkit/pkg/agentkit/runtimes"
	"github.com/orka-agents/agentkit/pkg/utils"
)

// RouteHandler builds a result for a resolved <runtime>/<outputkind> route.
type RouteHandler func(ctx context.Context, c client.Client, cfg *config.AgentConfig, rc *RuntimeConfig) (*client.Result, error)

type contextRouteHandler func(ctx context.Context, c client.Client, cfg *config.AgentConfig, rc *RuntimeConfig, reader contextFileReader) (*client.Result, error)

// Route is one entry in the flat router (plan §7.1). Keep this as the original
// single-field public layout: downstream packages may use unkeyed Route literals.
type Route struct {
	Handler RouteHandler
}

// RuntimeConfig is the per-runtime adapter configuration.
type RuntimeConfig struct {
	Name              string
	defaultAdapterRef string
}

// AdapterRef returns the adapter image ref to use as the LLB base, honoring the
// `--build-arg adapter=` override (the local dev-loop seam, plan §15.6) and
// falling back to the runtime's default.
func (rc *RuntimeConfig) AdapterRef(opts map[string]string) string {
	if ref := getBuildArg(opts, "adapter"); ref != "" {
		return ref
	}
	return rc.defaultAdapterRef
}

// routes maps "<runtime>/<outputkind>" to a handler. runtimeConfigs maps a
// runtime name to its config. Both are DERIVED in init() from runtimes.Runtimes
// (the single source of truth) — dispatch as data.
var (
	routes = map[string]Route{}
	// Keep request-scoped context dispatch outside the exported Route value so
	// its historical one-field unkeyed literals remain source-compatible.
	contextRouteHandlers = map[string]contextRouteHandler{}
	runtimeConfigs       = map[string]*RuntimeConfig{}
)

// registerRuntime wires a runtime adapter and its image output route.
func registerRuntime(rc *RuntimeConfig) {
	runtimeConfigs[rc.Name] = rc
	key := rc.Name + "/" + utils.OutputKindImage
	routes[key] = Route{Handler: HandleAgent}
	contextRouteHandlers[key] = handleAgent
}

func init() {
	// Derive the wiring from the single canonical declaration in pkg/agentkit/runtimes.
	// Adding a runtime is one RuntimeSpec literal there — nothing changes here.
	// (No lockstep guard is needed: there is exactly one source of truth, not two
	// registries to reconcile.)
	for _, rt := range runtimes.Runtimes {
		registerRuntime(&RuntimeConfig{
			Name:              rt.Name,
			defaultAdapterRef: rt.DefaultAdapterRef,
		})
	}

	// Register the debug/agent-yaml target. It is not runtime-specific.
	const debugAgentYAML = "debug/agent-yaml"
	routes[debugAgentYAML] = Route{Handler: nil} // context handler only
	contextRouteHandlers[debugAgentYAML] = handleDebugAgentYAML
}

// IsRegisteredRuntime reports whether name (after alias resolution) has a wired
// runtime adapter. The wiring is derived from runtimes.Runtimes, so this agrees with
// runtimes.IsKnownRuntime by construction. It is a build-package predicate over the
// adapter set; the config validator uses runtimes.IsKnownRuntime instead (config
// importing build would form a config→build import cycle, since build imports
// config).
func IsRegisteredRuntime(name string) bool {
	_, ok := runtimeConfigs[runtimes.CanonicalRuntime(name)]
	return ok
}

// defaultRuntime returns the runtime to use when the config does not name one.
func defaultRuntime() string {
	return runtimes.DefaultRuntime()
}

func canonicalEffectiveRuntime(runtime string) string {
	if runtime == "" {
		runtime = defaultRuntime()
	}
	return runtimes.CanonicalRuntime(runtime)
}

func targetRuntimeSegment(target string) (string, bool) {
	if target == "" {
		return "", false
	}
	segment, _, _ := strings.Cut(target, "/")
	runtime := runtimes.CanonicalRuntime(segment)
	_, registered := runtimeConfigs[runtime]
	return runtime, registered
}

// lookupRoute resolves a build target plus the effective runtime to a route and
// its RuntimeConfig.
//
// Target semantics (Dalec lookupTarget, plan §15.5.4 — empty target is NOT an
// error here; unlike Dalec we have a single sensible default output): an empty
// target resolves to "<runtime>/image". A bare runtime ("pydantic-ai") resolves
// to "<runtime>/image". Otherwise exact match, then longest-prefix match.
func lookupRoute(target, runtime string) (matched string, route Route, rc *RuntimeConfig, ok bool) {
	// Resolve an omitted or aliased runtime to the canonical adapter identity so
	// the registry lookup and the "<runtime>/image" route key always agree.
	runtime = canonicalEffectiveRuntime(runtime)
	rc, rcOK := runtimeConfigs[runtime]
	if !rcOK {
		return "", Route{}, nil, false
	}

	// The target may ALSO name the runtime by alias — either bare ("maf") or as
	// the leading route segment ("maf/image"). Route keys are canonical, so
	// canonicalize the target's runtime segment before any comparison/lookup, or
	// an alias target would miss every branch and fail to route.
	target = canonicalizeTargetRuntime(target)

	// A target that names a registered runtime must agree with the effective
	// runtime. Otherwise route and RuntimeConfig would describe different
	// adapters (for example, langgraph/image with pydantic-ai config).
	targetRuntime, namesRuntime := targetRuntimeSegment(target)
	if namesRuntime && targetRuntime != runtime {
		return "", Route{}, nil, false
	}

	// Empty or bare-runtime target → the runtime's image route.
	if target == "" || target == runtime {
		r, rOK := routes[runtime+"/"+utils.OutputKindImage]
		return runtime + "/" + utils.OutputKindImage, r, rc, rOK
	}

	// Exact match.
	if r, rOK := routes[target]; rOK {
		return target, r, rc, true
	}

	// Longest-prefix match on "<key>/...".
	var candidates []string
	for k := range routes {
		if strings.HasPrefix(target, k+"/") {
			candidates = append(candidates, k)
		}
	}
	if len(candidates) > 0 {
		sort.Strings(candidates)
		k := candidates[len(candidates)-1]
		return k, routes[k], rc, true
	}

	return "", Route{}, nil, false
}

// canonicalizeTargetRuntime rewrites the runtime portion of a build target to its
// canonical name, so an alias target ("maf" or "maf/image") matches the
// canonically-keyed routes. The first "/"-separated segment is the runtime; any
// remainder (the output kind) is preserved verbatim. An empty target is returned
// unchanged.
func canonicalizeTargetRuntime(target string) string {
	if target == "" {
		return ""
	}
	seg, rest, hasRest := strings.Cut(target, "/")
	canon := runtimes.CanonicalRuntime(seg)
	if hasRest {
		return canon + "/" + rest
	}
	return canon
}
