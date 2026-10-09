# Architecture

AgentKit has two halves:

1. a Go BuildKit gateway frontend that turns an Agentkitfile into an OCI image,
2. Python runtime adapter images that read the baked agent contract and serve the
   agent.

The stable seam between the halves is `/agent/agent.yaml`, documented in
[`agent-abi.md`](agent-abi.md).

## Build-time flow

```text
Docker/BuildKit
  -> cmd/frontend
  -> pkg/build.Build
  -> pkg/agentkit/config.NewFromBytes + Validate
  -> pkg/build.resolveInstructions
  -> pkg/agentkit/effective.FromConfig
  -> pkg/agentkit/abi.Render
  -> pkg/agentkit2llb/agent.Agentkit2LLB
  -> final OCI image
```

### `cmd/frontend`

`cmd/frontend/main.go` is the BuildKit gateway entrypoint. It configures logging
and delegates every build to `build.Build` via `grpcclient.RunFromEnvironment`.

### `pkg/agentkit/config`

This package owns the authored Agentkitfile schema and validation rules:

- `load.go` probes `{apiVersion, kind}` before strict decoding.
- `specs.go` declares `AgentConfig`, `Metadata`, `Model`, and `Expose`.
- `source.go` models instruction sources as a tagged union: inline text or a
  file path in the build context.
- `tool.go` models tools as a source union. The accepted tool source is a stdio
  `command`; an `image` field is present in the Go shape but validation rejects
  it.
- `validate.go` enforces supported runtime names, OpenAI-compatible models,
  one instruction source, unique tool names, secret-name hygiene, and
  `expose.openai: true`.

The loader is intentionally strict: kind-less files, unsupported kinds, unknown
fields, invalid sources, and likely secret literals all fail before LLB work
starts.

### `pkg/agentkit/runtimes`

`catalog.go` is the single source of truth for shipped runtime identities:

- `pydantic-ai` (default),
- `microsoft-agent-framework` with alias `maf`,
- `langgraph`.

It also carries each runtime's default adapter image ref. The config validator
and the build router both consult this package, avoiding duplicate runtime
registries.

The YAML files in `runtimes/catalog/` mirror this Go catalog and are covered by
`catalog_file_test.go`.

### `pkg/build`

`build.go` is the frontend orchestrator:

- loads the Agentkitfile from local, Git, or HTTP BuildKit context,
- applies build-arg overrides such as `runtime`,
- validates the config,
- resolves the route for `target` + runtime,
- resolves file-backed instructions from the build context,
- canonicalizes the config into an effective Agent,
- solves one LLB graph per target platform, and
- wires image config metadata into the BuildKit result.

`router.go` derives `<runtime>/image` routes from the runtime catalog. Empty
`target`, a bare runtime target, and exact `<runtime>/image` all route to the
same image handler. Runtime aliases are canonicalized before lookup.

`instructions.go` is the seam between authored instruction sources and the
BuildKit context.

### `pkg/agentkit/effective`

`effective.FromConfig` turns a validated authored config plus resolved
instructions into a build-ready Agent. It applies defaults and canonicalization:

- empty runtime becomes `pydantic-ai`,
- runtime aliases become canonical names,
- empty port becomes `8080`,
- mutable maps and slices are copied before downstream rendering.

Downstream packages consume this effective shape so they do not reinterpret raw
authoring defaults.

### `pkg/agentkit/abi`

`abi.Render` renders the effective Agent into the exact YAML shape consumed by
the Python reader. It writes `abiVersion`, metadata, model, resolved
instructions, tool commands/env allowlists, and expose information. `abi.Path` is
`/agent/agent.yaml`.

### `pkg/agentkit2llb/agent`

`Agentkit2LLB` starts from the selected adapter image and overlays one file:
`/agent/agent.yaml`. `NewImageConfig` then sets the final image metadata:

- user `1000:1000`,
- workdir `/`,
- entrypoint `/opt/agentkit/bin/agentkit-serve`,
- command `--config /agent/agent.yaml`,
- `PATH`, `AGENTKIT_BIND=127.0.0.1`, and `PYTHONUNBUFFERED=1`,
- exposed serve port plus the Foundry default `8088` when the ABI uses the generic default port,
- AgentKit and OCI labels.

No tool root filesystem or secret value is merged into the image.

## Runtime flow

```text
agentkit-serve --config /agent/agent.yaml --protocol openai|foundry|orka
  -> runtimes/common.config.load
  -> runtimes/common.cli.run
  -> selected protocol skin
       openai  -> server.create_app
       foundry -> foundry.create_foundry_app
       orka    -> orka.create_orka_app
  -> adapter agent_factory.build_runtime
  -> RuntimeSession.run(RunRequest)
  -> protocol-shaped response or SSE frame
```

### Protocol skins over one runtime seam

AgentKit owns protocol portability by keeping `/agent/agent.yaml` and
`RuntimeSession.run(RunRequest)` as the single runtime seam, then layering thin
HTTP protocol skins over that seam. The same image can be started with
`AGENTKIT_PROTOCOL=openai` (default), `AGENTKIT_PROTOCOL=foundry`, or
`AGENTKIT_PROTOCOL=orka` without rebuilding the agent.

- `openai` exposes the standalone OpenAI-compatible `/v1` surface:
  `/healthz`, `/v1/models`, and non-streaming `/v1/chat/completions`.
- `foundry` exposes `/readiness`, `/invocations`, and the current
  non-streaming, synchronous `/responses` compatibility surface. Its capability
  is named `foundry-responses-minimal` until full Responses lifecycle parity is
  implemented or delegated to official protocol libraries.
- `orka` exposes observed-mode `orka.harness.v1`: `/v1/health`,
  `/v1/capabilities`, `/v1/turns`, `/v1/turns/{turnID}/events` over SSE, and
  `/v1/turns/{turnID}/cancel`. One Orka turn maps to one runtime `RunRequest`.

AgentKit does not implement Orka governance. It reports capabilities, lifecycle
frames, result text, image labels, and renderable registration manifests. Orka
remains responsible for trust tiers, approval decisions, Tool CRD enforcement,
idempotency, and side-effect policy.

### `runtimes/common`

`agentkit_serve_common` is framework-neutral:

- `config.py` strictly loads the baked ABI and checks the ABI version.
- `cli.py` loads config, chooses protocol/bind/port, and refuses non-loopback
  binds unless `AGENTKIT_AUTH_TOKEN` is set. Orka mode also requires the token
  for turn/event/cancel endpoints.
- `server.py` serves the OpenAI-compatible `/healthz`, `/v1/models`, and
  `/v1/chat/completions` protocol skin.
- `foundry.py` serves the Foundry `/readiness`, `/invocations`, and minimal
  `/responses` protocol skin.
- `orka.py` serves the observed-mode `orka.harness.v1` HTTP+SSE protocol skin.
- `conversation.py` converts protocol request bodies into a `RunRequest` whose
  prompt/history are provider-neutral and whose optional Orka fields carry
  `turn_id`, `correlation_id`, deadline, metadata, and per-run env.
- `runtime.py` defines `RunRequest` consumers: `RuntimeFactory`,
  `RuntimeSession`, `RunResult`, and `AgentRunError`.
- `adapter_support.py` provides shared helpers for API-key resolution, tool env
  allowlists, MCP timeout parsing, and framework exception normalization.
- `model_errors.py` defines the runtime-owned model error codes every protocol
  skin reports instead of framework or upstream text.
- `conformance.py` defines adapter-neutral HTTP behavior tests reused by each
  adapter package.
- `parity.py` defines the wire-level parity suite each adapter package imports.
  It runs the adapter's real runtime against a scripted loopback model and a
  stdio MCP fixture through the OpenAI, Foundry, and Orka skins.

The shared server imports no agent framework.

### Runtime adapters

Each adapter package contributes a thin CLI binding plus `agent_factory.py`:

- `runtimes/pydantic-ai` builds a pydantic-ai `Agent`, supports both pydantic-ai
  MCP APIs used across 1.x and 2.x, prefixes tool names, and maps pydantic-ai
  run results into `RunResult`.
- `runtimes/microsoft-agent-framework` builds a Microsoft Agent Framework chat
  client and agent, enforces guardrails that keep Azure/Foundry/Copilot packages
  out of the generic adapter, and maps MAF results into `RunResult`.
- `runtimes/langgraph` builds a LangGraph ReAct-style graph with LangChain
  OpenAI chat models and persistent MCP sessions, aggregates token usage across
  model calls, and maps the final AI message into `RunResult`.

Adapters remain separate images with separate framework dependencies. The shared
core is installed into each adapter image first, then the adapter package adds its
framework-specific dependencies.

## Security and network posture

- Agent images run as non-root (`1000:1000`).
- The default bind is loopback (`127.0.0.1`).
- Setting `AGENTKIT_BIND` to a non-loopback host requires
  `AGENTKIT_AUTH_TOKEN`; protected endpoints then require
  `Authorization: Bearer <token>`.
- OpenAI `/healthz` and Orka `/v1/health` and `/v1/capabilities` remain
  unauthenticated for liveness/discovery checks.
- `model.apiKeyEnv` and tool `env` values are names, not secret values.
- Tool subprocesses receive only their declared env names and never inherit the
  full container environment.
