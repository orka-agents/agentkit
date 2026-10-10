# agentsessions Harness

Select `--protocol agentsessions` or `AGENTKIT_PROTOCOL=agentsessions` on the
adapter's existing `agentkit-serve` entrypoint. All three runtimes — `pydantic-ai`,
`microsoft-agent-framework` (build selector alias `maf`) and `langgraph` — support
**host-mediated, text-only Chat model execution**. There is no fallback to the
normal provider runtime. This is native protobuf gRPC: unary
`agentsessions.v1.Harness.Describe` and bidirectional `Harness.Connect`, not an
HTTP facade or ACP.

The wire sources and controller proof are pinned to
[`aramase/agentsessions@b212d498ba52615b5087b2579bdd482809065642`](https://github.com/aramase/agentsessions/tree/b212d498ba52615b5087b2579bdd482809065642),
Go module `v0.1.3-0.20261006182201-b212d498ba52`. Compatibility with other host
revisions is not established by this proof.

## Supported subset

| Surface | Supported behavior / limit |
|---|---|
| Model execution | Baked model name and instructions; ordered `system`/`user`/`assistant` text at the model bridge. The host invokes the model or supplies a journaled result. |
| Conversation | History INPUT/OUTPUT becomes ordered user/assistant turns, preserving empty text. Earlier Inputs and the final Input are appended exactly once; zero Inputs do not invent a user turn. |
| Host metadata | Known journal metadata is validated but is not prompt content. Resume cursor and identity are not framework state. |
| `Start.Config` | Opaque bytes forwarded unchanged in `RunRequest.config`, excluded from its repr, and ignored by the current text policy. The pinned host journals/restores them through `EVENT_EXECUTION_START`; Config is not an env, credential, option or prompt channel. |
| Recovery | Describe advertises `STATELESS_REPLAY` and `fork_safe=true` for this restricted profile. Every Start creates fresh framework resources; there is no cross-turn framework cache. |
| Capabilities | No tools advertised; `requires_gpu=false`, `streaming=false`, `reasoning_replay=false`. |
| Unsupported content | Tool schemas/calls/results, approvals, multimodal/file/data/reasoning parts, developer roles, unknown history and model options fail closed. No private wire encodings are accepted. |

The common exchange emits `EVENT_MODEL_CALL` with the baked model, ordered text
messages, deterministic per-execution IDs (`model-1`, `model-2`, …), empty params
and no input hash. **The host computes the fingerprint, journals model effects,
and owns OUTPUT and usage.** All three adapter hooks return `None`, so they do
not add duplicate OUTPUT or invent usage.

## Immutable startup binding

Set both deployment-owned values as lowercase `sha256:<64 hex>`:

- `AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST`: SHA-256 of the **exact
  bytes** of `/agent/agent.yaml` (or `--config`). Compute it outside the runtime;
  YAML-equivalent rewrites fail verification.
- `AGENTKIT_AGENTSESSIONS_IMPLEMENTATION_DIGEST`: immutable adapter image digest
  covering the implementation and installed dependency closure, not a moving
  tag. Retain that implementation and the exact baked configuration for replay.

The descriptor ID is `agentkit:<configuration digest>:<implementation digest>`.
These values bind the ABI and implementation identity; they are **not
authentication** or attestation of an independently measured image.

Startup rejects baked direct tools, `brokeredTools` and every context provider.
A nonempty value in the baked model's `apiKeyEnv` is rejected without printing
it. Model URL userinfo, queries and fragments are also rejected without echoing
the URL. Required-env declarations and absent provider credentials are not
resolved; the original provider URL and workload-identity auth are never used.
Keep provider keys out of the image, process, Config, logs and argv. The local
bridge token is generated per execution and is not a provider credential.

See the [keyless launch example](../README.md#agentsessions-keyless-text-profile).

## Transport and authentication

`AGENTKIT_BIND` defaults to `127.0.0.1`; `AGENTKIT_PORT` overrides the baked
expose port (normally 8080). Nonloopback binding requires `AGENTKIT_AUTH_TOKEN`.
When configured, **both RPCs** require exactly one gRPC metadata entry
`authorization: Bearer <token>`, even on loopback. Start identity is not
authentication. Clients must inject metadata explicitly; the upstream reference
harnesswire client does not do so by itself.

Transport is h2c, with no built-in TLS or HTTP health endpoint. Isolate a
nonloopback listener on a private network or behind a trusted TLS/auth proxy;
a bearer token does not make public plaintext transport safe. The container is
a trust boundary, not a network sandbox: deployment must prevent unexpected
outbound traffic and exporters. The adapter settings below do not establish
universal telemetry or network enforcement.

## Execution lifecycle and failures

Connect admits one execution at a time. Start must be first, with a nonempty
execution ID of at most 1024 UTF-8 bytes. Later frames must match **both** its
execution ID and session string exactly; empty sessions are allowed (the
reference client omits session). Invalid first frames abort the RPC with
`INVALID_ARGUMENT`. A busy admission receives `FAILED/RESOURCE_EXHAUSTED`
without disturbing the active execution.

Keep the request stream open until END so the controller can service effects.
The reader remains live independently of the runner. Completion, early failure
and cancellation clean up owned resources and release admission **before** END,
so a new turn may start at END without draining EOF. Cancel, repeated
cancellation, EOF/half-close and disconnect settle owned cleanup; exactly one
safe END is emitted where writable. No delivery/availability claim is made for
an unwritable transport. A detected cleanup failure produces a safe
`FAILED/INTERNAL` END where writable, never a successful terminal status.

On a main-thread POSIX event loop with signal support, the CLI owns SIGINT and
SIGTERM while serving. The first signal closes admission and stops the gRPC
transport, then waits for execution-owned teardown before the event loop exits.
SIGTERM returns normally; SIGINT raises `KeyboardInterrupt` after cleanup.
Further SIGINT/SIGTERM signals do not interrupt that drain, and prior handlers
are restored afterward. Embedded async `serve()` does not install signal
handlers; off-thread or unsupported-loop callers retain their prior signal
behavior. Cleanup has no forced deadline: deployment grace periods and
uncatchable SIGKILL can still terminate the process before teardown finishes.

The returned server preserves the gRPC server API. `stop()` waits for execution
cleanup even when its caller is canceled. `wait_for_termination()` includes that
cleanup and retains its timeout result without canceling the shutdown. Stopping
before the first start is a no-op, as in native gRPC. Public TCP port binding
also enforces the nonloopback authentication gate; Unix sockets are local-only.

ModelResult must correlate to the single pending call. Unsolicited/duplicate or
mismatched replies, missing message payloads, non-assistant/nontext content,
oversized results, negative usage or a different usage model fail with
`INVALID_ARGUMENT`. Unsupported Start content and tool/approval replies fail
with `UNIMPLEMENTED`; repeated Start, unknown control frames and mismatched
identities fail with `INVALID_ARGUMENT`. Internal framework or reader failures
use safe `INTERNAL` descriptions, never raw exceptions, request content, Config
or credentials.

Execution is serialized: one pending model effect and a one-event queue.
Protobuf send/receive messages are bounded at 4 MiB including framing fields;
bridge HTTP bodies, ModelResult and encoded bridge responses each have a 1 MiB
limit. Concurrent HTTP requests are refused and listener concurrency is bounded.

## Per-Start framework resources

Each hook owns a fresh ephemeral `127.0.0.1` HTTP bridge with a fresh bearer token,
SDK client and framework agent/graph. Clients and listener requests close before
admission is released. The shared core remains framework-neutral; adapters opt
in through `async run_agentsessions(binding, request, exchange) -> RunResult |
None`. Missing hooks fail closed rather than calling `build_runtime`; there is
no callback-in-metadata or environment-selected model bridge.

| Runtime | Restricted execution path |
|---|---|
| Pydantic AI | Fresh `OpenAIProvider`/`OpenAIChatModel`/`Agent`; documented graph iteration stops after the SDK model request, before tool or output-validation retries, preserving empty host completions. |
| Microsoft Agent Framework | Fresh public `RawOpenAIChatCompletionClient` and `RawAgent`, with buffered `Agent.run`. These omit MAF's two default-enabled telemetry layers without changing global providers. No cached session, context provider, tool or MCP state participates. |
| LangGraph | Fresh Chat Completions model and compiled graph invoked with `ainvoke`; no checkpointer, store, global cache, LangSmith tracing or Responses route. Canonical text roles are preserved even for o-series models; `temperature=None` avoids injecting sampling options. |

All use an explicitly injected AsyncOpenAI client bound to the local bridge:
proxy/environment discovery and redirects disabled, retries zero, explicit
local Authorization, connect timeout 5 seconds and no wall-clock read timeout.
The controller owns model completion, reply deadlines and cancellation; ambient
provider credentials and the normal provider builders are not used.

The bridge accepts only `/v1/chat/completions` with the exact baked `model`,
ordered text `messages` and optional boolean `stream`. All other options,
including tool schemas and `stream_options`, are rejected before a model effect.
`stream=true` is **buffered synthetic terminal SSE only**: after the complete
validated ModelResult, one final chunk with `finish_reason="stop"`, then
`[DONE]`. It is not true streaming and does not change advertised capabilities.

## Checks and recovery proof

### Built-image controller replay

Run on the **Linux Docker daemon host**, with Docker/Buildx, Go matching
`test/agentsessions/go.mod` and `setsid`. The test connects to a private container
IP on an internal-only network; Docker Desktop/remote-daemon clients cannot use
this path without running on the daemon host. Builds need registry/dependency
access; runtime model effects need no external provider or credentials.

```sh
scripts/agentsessions-e2e.sh --build pydantic-ai
scripts/agentsessions-e2e.sh --build maf
scripts/agentsessions-e2e.sh --build langgraph
```

Syntax: `scripts/agentsessions-e2e.sh [--build|--skip-build]
[pydantic-ai|maf|microsoft-agent-framework|langgraph]`. The runtime defaults to
`pydantic-ai`, or `AGENTKIT_AGENTSESSIONS_RUNTIME` when no selector is supplied.
There is no `--help` mode; invalid modes or extra arguments print usage and
exit 2. Unsupported runtime selectors also exit 2.

`--build` builds the real BuildKit frontend, selected adapter and AgentKit-derived
fixture image under `test/agentsessions/`. `--skip-build` uses an already-built
image (`AGENTKIT_AGENTSESSIONS_IMAGE` can select it); rebuild after implementation
changes before treating the proof as fresh. Plain nested `go test` **skips**
`TestContainerStatelessReplay` without that image variable. CI covers all three
runtimes plus runtime-selection and SIGINT/SIGTERM cleanup tests. The runner
owns a unique Docker label and process group for quiescent cleanup.

All three fixtures use the same logical text agent, a `.invalid` provider URL
and absent `OPENAI_API_KEY`, exercising the actual framework/SDK image, not a
fake SDK or Pydantic TestModel. An offline host model serves two live turns,
producing **2 live model calls, 2 journaled model calls and 2 outputs**. The proof
closes/reopens SQLite, replaces the container at its immutable image ID, then
calls **`Controller.Replay`** to re-execute both turns. It requires:

- **0 replay model calls**, equal outputs and an unchanged, verified journal;
- exact opaque Config/cursor restoration rather than current constructor values;
- refusal of an intentional model-input fingerprint mismatch without a model call;
- absent provider credentials and an internal-only, no-egress fixture network.

`AGENTSESSIONS_REPLAY_PROOF` reports the actual runtime, immutable image digest,
host pin and assertions. `Sessions.Replay` read-only redelivery and Go/Python
wire interoperability are **not reconstruction proofs**. This establishes only
the container/controller text profile, not Session service, placement, Canvas,
kind deployment or live-provider integration.

### Wire regeneration and common checks

`runtimes/common/agentkit_serve_common/agentsessions/` contains exact proto inputs,
the upstream Apache-2.0 license, provenance/source hashes and generated Python
stubs. Runtime compatibility bounds are separate from the exact compiler pins
in `scripts/agentsessions-generator-requirements.txt`.

```sh
scripts/generate-agentsessions-stubs.sh --fetch  # fetch pinned inputs, verify, generate
scripts/generate-agentsessions-stubs.sh          # regenerate verified vendored inputs
scripts/generate-agentsessions-stubs.sh --check  # temporary regeneration; no writes
uv run --directory runtimes/common --extra dev pytest -q
```

The generator uses `uv`; `UV=/absolute/path/to/uv` overrides its executable.
Common tests run a native Go client against Python gRPC using the isolated
`test/agentsessions/` module, without expanding root Go dependencies. See the
[development guide](development.md) for general package checks.
