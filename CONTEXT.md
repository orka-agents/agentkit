# AgentKit context

AgentKit is a BuildKit gateway frontend plus a set of Python runtime adapter
images. The frontend turns a strict `kind: Agent` YAML file into an OCI image;
the runtime adapter reads `/agent/agent.yaml` and serves an OpenAI-compatible
non-streaming Chat Completions façade. Optional protocol wrappers can expose other
container contracts, such as Foundry Hosted Agents, without changing the core
Agent YAML ABI.

## Domain glossary

### Agent

A target-neutral description of a model, instructions, tools, context sources,
and serving surface. Users author an Agent in an Agentkitfile. The build converts
it into an effective Agent and then into a runtime adapter image plus a baked
Agent YAML ABI file.

### Agentkitfile

The user-authored YAML input consumed by the Go BuildKit frontend. It is strict,
kind-discriminated input (`apiVersion: v1alpha1`, `kind: Agent` or `kind: Toolbox`)
and can contain authored sources such as inline instructions or an instructions
file path.

### Effective Agent

The build-ready Agent value produced after validation and build-time source
resolution. It carries canonical runtime identity, effective serve port,
fully-resolved instructions, copied labels, copied tool definitions, env
requirements, context providers, and observability settings so ABI and image
writers do not reinterpret raw authoring defaults.

### Toolbox

A `kind: Toolbox` Agentkitfile and the `FROM scratch` image built from it: tools
installed read-only under `/opt/<name>/bin`, labeled `ai.orka.toolbox` so Orka
can mount them into the pods of its built-in coding agents. It is not an Agent
and has no runtime adapter.

### Instruction Source

The authored source of an Agent's system prompt. Supported sources are inline
instructions and file-backed instructions in the BuildKit context. The build
resolves an Instruction Source into the Effective Agent's instruction scalar
before rendering the ABI.

### Runtime Catalog

The Go catalog in `pkg/agentkit/runtimes` that defines supported runtime names,
aliases, default adapter image refs, and provider-neutral capabilities. Config
validation and build routing both read this catalog so they agree on runtime
identity and feature support.

### Runtime Capability

A provider-neutral feature flag declared by a Runtime Adapter, for example
`stdio-mcp` or `streamable-http-mcp`. The Agentkitfile validator checks requested
features against the runtime catalog before build so unsupported features fail
clearly.

### Build Route

The `<runtime>/image` route selected by `pkg/build/router.go`. Empty target and
bare runtime targets resolve to the selected runtime's image route. Runtime
aliases such as `maf` are canonicalized before route lookup.

### Agent YAML ABI

The `/agent/agent.yaml` contract between the Go frontend writer and Python
runtime reader. The writer renders it from an Effective Agent; every runtime
adapter reads the same shape. The ABI stores resolved prompts, model connection
metadata, tool declarations, env requirements, context provider declarations,
observability env names, and expose settings.

### Runtime Adapter

A concrete in-image implementation that reads the Agent YAML ABI and serves the
OpenAI-compatible `/v1` façade using a specific agent framework. Current adapters
are pydantic-ai, Microsoft Agent Framework, and LangGraph.

### Shared Runtime Core

The `agentkit-serve-common` Python package. It owns the ABI reader, CLI/network
posture, FastAPI façade, Foundry protocol wrapper, conversation normalization,
shared adapter support, and the neutral runtime interfaces. It imports no agent
framework.

### Runtime Factory

The adapter module interface consumed by the shared server and protocol wrappers.
Each adapter exposes `build_runtime(spec)`, returning a Runtime Session that owns
framework-specific agent lifecycle.

### Runtime Session

The live adapter handle entered once for the FastAPI lifespan. It starts and
keeps warm framework resources such as MCP tool sessions, and handles each
normalized Run Request.

### Conversation

The ordered OpenAI message list supplied to `/v1/chat/completions`. The shared
runtime core normalizes it before any adapter sees it.

### Run Request

The framework-neutral request for one non-streaming agent run. It contains prior
system/user/assistant history plus the final user prompt. Client-owned tool turns
and unsupported roles are removed because the built agent owns its tools.

### Tool Env Allowlist

The list of env var names on each stdio tool spec. Runtime adapters pass only
these present env vars into the tool subprocess and reject undeclared `${VAR}`
interpolation, preventing model keys and other process secrets from leaking into
unrelated tools.

### Runtime Env Requirement

A top-level `env:` declaration in an Agentkitfile/Agent YAML ABI. It records only
an environment variable name and whether it is required; values stay in the
runtime environment and are never baked into the image.

### Remote MCP Tool

A `tools[]` entry with `type: mcp`, `transport: streamable-http`, and `urlEnv`.
AgentKit core stores only env var names and non-secret static headers; runtime
adapters resolve the URL, headers, and generic auth at startup/request time.

### Context Provider

A provider-neutral `context.providers[]` entry for search, skills, or memory. The
schema is present so provider-specific deployment tooling can map resources to
generic env names, but runtime behavior is capability-gated.
