# Runtime adapters

Runtime adapters execute the same baked `/agent/agent.yaml` contract through
different Python agent frameworks. They share one framework-neutral server core
and differ only at the `agent_factory.py` boundary.

## Shared runtime core

The `agentkit-serve-common` package under `runtimes/common/` owns behavior that
must be identical across adapters:

| Module | Responsibility |
|---|---|
| `config.py` | Strict `/agent/agent.yaml` reader and ABI version check. |
| `cli.py` | `agentkit-serve --config ... --protocol openai\|foundry\|orka\|acp\|agentsessions`, bind/port handling, auth startup gates. |
| `server.py` | FastAPI app and OpenAI-compatible response/error envelopes. |
| `foundry.py` | Foundry `/readiness`, `/invocations`, and minimal `/responses` skin. |
| `orka.py` | Observed-mode `orka.harness.v1` HTTP+SSE skin. |
| `acp.py` | Strict ACP stdio child for an Orka `orka.harness.v2` supervisor. |
| `agentsessions/` | Native protobuf gRPC Harness, immutable startup binding, validated text history, bounded model exchange and per-execution loopback bridge. |
| `conversation.py` | Protocol request normalization into `RunRequest`. |
| `runtime.py` | `RuntimeFactory`, `RuntimeSession`, `RunResult`, `AgentRunError`. |
| `adapter_support.py` | API-key lookup, tool env projection, timeout parsing, error normalization. |
| `model_errors.py` | Runtime-owned model error codes and messages shared by every protocol skin. |
| `conformance.py` | Shared HTTP behavior tests adapter packages import. |
| `parity.py` | Shared wire-level suite that runs each adapter's real runtime against a scripted model and MCP tool. |

HTTP app factories receive an adapter module satisfying `RuntimeFactory` and
call `factory.build_runtime(spec)` and `RuntimeSession.run(request)`. ACP uses
its supervisor-bound runtime contract. agentsessions instead uses the optional
explicit `async run_agentsessions(binding, request, exchange) -> RunResult |
None` adapter hook, awaited once per Start. Missing hooks fail closed with
`UNIMPLEMENTED`; there is no normal provider-runtime fallback. The shared core
never imports pydantic-ai, Microsoft Agent Framework, LangChain, OpenAI SDK types,
Azure, Foundry SDKs, or Orka controllers.

## Protocol surfaces

All adapters can serve the same selected protocol surface. `openai` is the
default. Select with `--protocol` or `AGENTKIT_PROTOCOL`.

| Profile | Model/tool/env behavior |
|---|---|
| Normal HTTP (`openai`, `foundry`, `orka`) | Uses configured provider clients and runtime-supported baked MCP/context capabilities; startup/per-run env rules apply. |
| ACP stdio | Supervisor-bound provider proxy and prompt-scoped MCP broker; rejects baked direct/brokered tools, with the MAF packaged-skill exception described below. |
| agentsessions native gRPC | All three adapters support host-mediated text only. Requires exact configuration and immutable implementation digests; rejects baked tools, brokered tools and every context provider. Does not resolve required/provider env, use original provider URLs/auth or build the normal runtime. Fresh state per Start; Config stays opaque. |

The normal HTTP rules below do **not** widen the
[agentsessions supported subset](agentsessions.md). Its `Describe`/bidirectional
`Connect` RPCs use h2c, advertise stateless replay/fork safety only for that
subset, and add no HTTP health endpoint. Tools, approvals, multimodal/reasoning
parts, developer roles and model options fail closed.

OpenAI mode exposes:

- `GET /healthz` returns `{"status":"ok"}` and is always open. It returns 503
  `{"status":"unhealthy"}` after a run reports a fatal runtime failure, such as
  a stdio MCP tool subprocess exiting.
- `GET /v1/models` returns the one configured model name.
- `POST /v1/chat/completions` runs the agent once and returns one
  `chat.completion` object with a single assistant message.

Foundry mode exposes `/readiness`, `/invocations`, and synchronous
`/responses`. It defaults to port `8088` when the ABI kept the generic default
port; generated images expose both `8080` and `8088` in OCI metadata for that
case. Orka mode exposes `orka.harness.v1` health, capabilities, turn
acceptance, SSE replay, and cancel endpoints.

ACP mode opens no listener. It speaks newline-delimited ACP JSON-RPC on stdin
and stdout. The child verifies the configured model and SHA-256 digest of the
exact `/agent/agent.yaml` bytes before accepting a session. It rejects baked
direct tools and `brokeredTools`. The Microsoft Agent Framework adapter can
load [packaged skill instructions](instruction-skills.md); other context
providers remain prohibited. At session creation it
accepts at most one loopback HTTP MCP server with bearer authentication, which
is the prompt-scoped broker created by the Orka supervisor.

OpenAI-compatible HTTP request behavior is intentionally narrow:

- `stream: true` returns HTTP 400 with code `stream_unsupported`.
- non-empty `tools` returns HTTP 400 with code `tools_unsupported`.
- `tool_choice` values other than missing, empty, `none`, or `auto` return HTTP
  400 with code `tool_choice_unsupported`.
- the final message must have role `user`.
- prior `system`, `user`, and `assistant` messages become history.
- prior `tool` and unknown roles are ignored because the built agent owns its
  tools.
- the model receives the baked instructions first, then the client's history in
  order.
- `X-AgentKit-Session-Id`, when present, is forwarded through the neutral
  `RunRequest` for runtime/session correlation. It never replaces the history the
  client sent. Orka mode additionally forwards `turn_id`, `correlation_id`,
  `deadline`, `metadata`, and per-run `env` fields.

The protocol layer owns the conversation: OpenAI and Foundry clients send it, and
Orka mode keeps each runtime session's completed user/assistant turns, as the
ACP child does. Runtime adapters do not keep their own transcript.

Run failures use an OpenAI-shaped error envelope with `type: agent_error` and
runtime-owned messages. Framework and model SDK text can carry upstream bodies,
URLs, and echoed credentials, so it never reaches the client:

| Failure | Status | `code` |
|---|---|---|
| Model returned 401 or 403 | 503 | `ModelAuthRejected` |
| Model returned 429 or 5xx after SDK retries | 503 | `ModelUnavailable` |
| Model returned another 4xx, or the request failed in transport | 502 | `ModelUpstreamError` |
| MCP transport or protocol failure | 502 | `MCPToolProtocolError` |
| Orka runtime session failed to start | 503 | `RuntimeStartFailed` |
| LangGraph result has no messages list or assistant message | 502 | `LangGraphResultError` |
| LangGraph graph is not initialized | 500 | `AgentNotInitialized` |
| Other unclassified framework or SDK exception | 502 | `AgentRunFailed` |

Foundry mode reports the same model codes with its `upstream_status` field; it
reports other run failures as `RuntimeFailure` with a fixed message. Orka mode
reports every code in `TurnFailed` frames. Operators get a warning log for
each failure: the HTTP status for model HTTP errors, whose bodies may echo
credentials; the exception types for MCP failures and Orka runtime startup
failures; and a traceback otherwise.

## Model endpoint compatibility

In normal HTTP modes, adapters use the baked `model.baseURL` and `model.name` to
construct their OpenAI-compatible chat client. They do not special-case a
provider: the endpoint can be OpenAI, another hosted provider, a local gateway,
an in-cluster service,
or a prebuilt or custom [AIKit](https://github.com/kaito-project/aikit) model
image. AIKit is just an example of an OpenAI-compatible endpoint. For no-auth
endpoints, omit `model.apiKeyEnv` unless you place an auth proxy in front of the
endpoint, and make sure the generated AgentKit container can resolve the
configured `baseURL` at runtime.

## Network posture

The generated image defaults to `AGENTKIT_BIND=127.0.0.1`. In HTTP modes:

- loopback binds need no token except in Orka mode,
- non-loopback binds such as `0.0.0.0` require `AGENTKIT_AUTH_TOKEN`, and
- when a token is set, protected endpoints require
  `Authorization: Bearer <token>`.

OpenAI `/healthz` and Orka `/v1/health` and `/v1/capabilities` are intentionally
unauthenticated so container platforms and orchestrators can probe/discover the
service. Orka turn, event, cancel, and output endpoints always require a token.

ACP mode ignores bind and port settings because it uses stdio. The Orka
supervisor injects only `AGENTKIT_ACP_PROVIDER_BASE_URL`,
`AGENTKIT_ACP_PROVIDER_TOKEN`, `AGENTKIT_ACP_MODEL`, and
`AGENTKIT_ACP_AGENT_CONFIGURATION_DIGEST` into the child.

agentsessions also requires `AGENTKIT_AUTH_TOKEN` for nonloopback binds. When
configured, both RPCs require exactly one `authorization: Bearer <token>` gRPC
metadata entry, including on loopback; the reference host client must be given
this metadata explicitly. Deploy h2c privately or behind a trusted TLS/auth
proxy, and enforce outbound isolation at the deployment boundary. Per-hook
telemetry settings do not constitute a universal network sandbox.

## Tool lifecycle and env projection

In normal HTTP runtimes, tools are MCP servers declared in the ABI. Stdio tools
use `name`, `command`, and an `env` allowlist; remote tools use `type: mcp`,
`transport: streamable-http`, `urlEnv`, optional headers, and generic auth. Adapter
factories are responsible for turning each tool spec into their framework's MCP
integration.

Shared invariants:

Per-run env supplied by Orka is forwarded in `RunRequest.env` and helper functions
can resolve credentials from that mapping before falling back to process env.
Startup-scoped model clients and long-lived MCP sessions still resolve their own
startup credentials at runtime initialization; they are not rebuilt for every turn.

- a missing or empty command fails before serving,
- `AGENTKIT_MCP_TIMEOUT` controls MCP initialization timeout; MAF also uses it
  for tool requests, including [Orka approval waits](orka-human-approval.md),
- each tool subprocess receives only env vars declared in that tool's `env`,
- undeclared `${VAR}` interpolation inside a declared env value is rejected, and
- tool sessions are entered once for the app lifespan and reused across requests,
- an admitted MCP tool error returns to the model as a fixed failure result,
  never the tool's own error text,
- a tool session that is gone for good is fatal, whether a stdio subprocess
  exited or a remote server no longer knows the session: OpenAI `/healthz` and
  Foundry `/readiness` start failing so the platform replaces the container, and
  Orka mode rebuilds that runtime session for its next turn. A server's error
  for one call is not fatal,
- remote MCP clients inject headers only for the configured origin and do not
  follow redirects with credentials.

## Adapter packages

### pydantic-ai

Path: `runtimes/pydantic-ai/`

- Console script package name: `agentkit-serve`.
- Adapter image target built by `make build-serve`.
- Uses `OpenAIChatModel` and `OpenAIProvider`.
- Supports both older `MCPServerStdio` and newer `MCPToolset` /
  `StdioTransport` APIs.
- Maps pydantic-ai message history and usage objects into the neutral contract.
- agentsessions creates a fresh local-bridge SDK client and Agent per Start,
  stopping graph iteration after the model request and before tool/validation
  retries. The host owns output/usage, including empty completions.

### Microsoft Agent Framework

Path: `runtimes/microsoft-agent-framework/`

- Console script package name: `agentkit-serve-maf`.
- Adapter image target built by `make build-serve-maf`.
- Runtime selector: `microsoft-agent-framework` or alias `maf`.
- Depends on the bounded MAF core/OpenAI packages, the MCP SDK, and provider
  adapters needed by generic AgentKit capabilities such as workload-identity
  model auth, Azure AI Search context, and external memory.
- Guardrail tests prevent unrelated cloud packages such as CopilotStudio/Purview
  from crossing the adapter boundary.
- Normal runtime supports session-aware runs, remote MCP, filesystem/MCP skills,
  search context, and memory context through generic ABI fields. Sessions keep
  context-provider state only; each run's conversation comes from the request.
- agentsessions uses fresh public `RawAgent` and
  `RawOpenAIChatCompletionClient` for buffered execution, omitting both default
  MAF telemetry layers without mutating globals. No normal session cache,
  context/tool/MCP state or provider builder participates.

### LangGraph

Path: `runtimes/langgraph/`

- Console script package name: `agentkit-serve`.
- Adapter image target built by `make build-serve-langgraph`.
- Runtime selector: `langgraph`.
- Uses LangChain OpenAI chat models, LangGraph, `langchain-mcp-adapters`, and
  persistent MCP sessions.
- Aggregates token usage from every AI message in a tool-using graph run.
- Guardrail tests keep Azure and Foundry packages out of the generic adapter.
- agentsessions creates a fresh local-bridge chat model and compiled graph:
  no checkpointer/store/cache, LangSmith tracing or Responses API. Explicit
  `temperature=None` and canonical text messages avoid injected model options
  and o-series developer-role rewrites.

## Adding an adapter

To add a single-agent runtime:

1. create a new adapter package with `agentkit_serve/__main__.py` that calls
   `agentkit_serve_common.cli.run(agent_factory)`,
2. implement `agent_factory.build_runtime(spec) -> RuntimeSession`,
3. add an adapter Dockerfile that installs `runtimes/common` before the adapter,
4. add a `RuntimeSpec` in `pkg/agentkit/runtimes/catalog.go`,
5. add the matching `runtimes/catalog/*.yaml` entry and tests/fixtures, and
6. import the shared conformance tests in the adapter's test suite.

For agentsessions, additionally implement the optional per-Start hook with fresh
cancellable resources and the shared restricted model exchange. Do not reuse
normal provider/tool builders or return duplicate host-owned output/usage; see
[the profile and proof guide](agentsessions.md).

No shared server changes should be necessary when the adapter can satisfy the
neutral `RuntimeSession` contract.
