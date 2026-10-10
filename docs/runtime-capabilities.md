# Runtime capabilities

AgentKit keeps runtime feature support explicit. Runtime identity and capabilities
are declared in `pkg/agentkit/runtimes` and mirrored in `runtimes/catalog/*.yaml`;
the catalog consistency test ensures they stay in sync.

Capabilities are feature/protocol flags used by validation, image metadata, and
orchestrator registration before an image build. Unsupported requested features
should fail clearly instead of silently producing a degraded runtime image.

## Capability names

Current and reserved names:

- `stdio-mcp` — stdio MCP servers declared with `tools[].command`.
- `streamable-http-mcp` — remote MCP over Streamable HTTP.
- `foundry-invocations-protocol` — Foundry hosted-agent `/readiness` +
  `/invocations` wrapper.
- `foundry-responses-minimal` — current Foundry `/responses` wrapper: synchronous
  and non-streaming. It supports a deterministic schema-only brokered function-call
  loop when `agent.yaml` contains `brokeredTools`, using hosted-compatible
  response IDs and a memory or optional file-backed continuation store. Do not
  treat this as full Responses parity for background, streaming, polling, cancel,
  or multi-replica platform-managed production state.
- `orka-harness-v1` — observed-mode native Orka `orka.harness.v1` wire protocol over HTTP+SSE (`HealthResponse`, flat `CapabilitiesResponse`, `StartTurnRequest`, `StartTurnResponse`, and `HarnessEventFrame`).
- `orka-observed-tools` — AgentKit-owned tools/MCP execute inside the runtime;
  Orka observes lifecycle/output frames and governs externally.
- `orka-brokered-tools` — reserved for a future mode where Orka brokers tool
  execution and approval frames. Not advertised by default.
- `filesystem-skills` — local filesystem skill sources.
- `mcp-skills` — MCP-backed skill sources.
- `context-provider-search` — external search/RAG context provider.
- `context-provider-memory` — external memory context provider.
- `context-provider-skills` — skills represented through a context-provider ABI.
- `workload-identity-token-auth` — workload identity tokens for tools/resources.
- `model-workload-identity-auth` — workload identity tokens for model calls.
- `otel-export` — OpenTelemetry export support.
- `tool-approval` — tool approval / human-in-the-loop policy support.

Avoid provider-specific resource names such as `foundry-toolbox`; those should map
to generic capabilities such as `streamable-http-mcp` plus deployment-profile env
and auth wiring. Orka-specific names here describe the protocol contract AgentKit
exposes; Orka remains responsible for policy, approval, idempotency, and
side-effect governance.

Orka v2's `supportsBrokeredToolApprovals` registration capability covers
controller-managed approvals over the ACP child's MCP connection. It is separate
from AgentKit's unsupported local `tool-approval` capability and the legacy v1
brokered hooks below. See [Human approval for Orka tools](orka-human-approval.md)
for the qualified direct MAF and hosted Foundry paths.

## Current support

All three adapters serve both `POST /v1/chat/completions` and
`POST /v1/responses` in `AGENTKIT_PROTOCOL=openai`. They share one listener,
runtime session, baked tools, auth, and lifecycle; no new protocol or capability
flag is required. Generic Responses is synchronous, stateless, and text-only,
with client-supplied message history and completed assistant text and usage.
It rejects unsupported features before execution, as defined in
[the HTTP contract](agent-abi.md#served-http-contract). This does not change
`foundry-responses-minimal` or Foundry/brokered continuation support.

| Runtime | Capabilities |
|---|---|
| `pydantic-ai` | `stdio-mcp`, `streamable-http-mcp`, `foundry-invocations-protocol`, `foundry-responses-minimal`, `orka-harness-v1`, `orka-observed-tools` |
| `microsoft-agent-framework` / `maf` | `stdio-mcp`, `streamable-http-mcp`, `foundry-invocations-protocol`, `foundry-responses-minimal`, `orka-harness-v1`, `orka-observed-tools`, `workload-identity-token-auth`, `model-workload-identity-auth`, `context-provider-skills`, `filesystem-skills`, `mcp-skills`, `context-provider-search`, `context-provider-memory` |
| `langgraph` | `stdio-mcp`, `streamable-http-mcp`, `foundry-invocations-protocol`, `foundry-responses-minimal`, `orka-harness-v1`, `orka-observed-tools` |

Context-provider schemas are capability-gated per runtime; the MAF adapter
currently declares skills, search, and memory support. OTel export, local tool
approval enforcement, log-level observability, and native real-model Orka
brokered-tool adapters remain gated until a runtime and protocol contract declare
support.

The shared runtime package now defines the neutral brokered-tool Interface types
(`BrokeredToolDefinition`, `BrokeredToolCall`, `BrokeredToolResult`,
`ToolBroker`, and `BrokeredRuntimeSession`) so framework adapters have a deep
seam to implement brokered tools. The Orka HTTP skin wires brokered
read/write/coordination and `/continue` behind
`AGENTKIT_ORKA_ENABLE_BROKERED_READ=1`,
`AGENTKIT_ORKA_ENABLE_BROKERED_WRITE=1`, and
`AGENTKIT_ORKA_ENABLE_BROKERED_COORDINATION=1`; default capabilities still
advertise observed mode only. Foundry hosted `/responses` can also exercise a
deterministic brokered function-call loop from static `brokeredTools`. For
A4/A5 fallback validation, `AGENTKIT_FOUNDRY_BROKERED_MODEL_LOOP=1` enables a
lower-level OpenAI-compatible model loop that exposes static safe
brokered schemas as function tools, emits hosted Responses `function_call`
items, and resumes the model with Orka-provided `function_call_output`. Orka
remains responsible for coordination policy,
quotas, child-task lineage, and namespace/agent authorization. Native framework
adapter brokered hooks are still intentionally gated: today the brokered profiles
are validated through the offline echo/conformance runtime, while real model
adapters should only enable those gates after their native pause/resume/tool-output
hooks have matching conformance coverage.

`AGENTKIT_MODEL_API` is a shared startup selector for the upstream model API.
All three runtime adapters and the Foundry brokered model loop support
`chat_completions`, the default, explicit `responses`, and opt-in `auto`.
This choice is independent of local/container deployment, inbound HTTP/ACP
protocol, and MAF model auth mode.
MAF supports both APIs with API keys, token hooks, and its Azure project-credential
fallback. LangGraph explicitly sets its SDK transport so ambient LangChain
settings cannot override this selector.

Explicit Responses requires a backend that implements `/responses`; explicit
Chat requires `/chat/completions`. Neither explicit selector falls back. Use
`AGENTKIT_MODEL_API=auto` to send the first real request to Responses without a
separate probe. Only a recognized initial endpoint/API rejection allows one
Chat retry. The concrete choice is cached per runtime/backend/model lifetime;
acceptance locks Responses even if a stream or output later fails. Unknown
404s, missing models, auth/rate-limit errors, timeouts, and generic 5xx failures
never trigger fallback. History, tools, and upstream Responses `store: false`
are preserved, and rejected Responses resources close before Chat starts.
See [the exact rejection rules](runtime-adapters.md#model-endpoint-compatibility).

Foundry brokered auto persists the concrete API for continuation and restores
it on resume. Explicit mismatch protections remain in place. Both inbound
routes remain available with every selector. Invalid selector values fail
before client/auth initialization. The separate `agentsessions` host-mediated
model contract remains Chat-only. Native Anthropic
Messages is not supported. This selector does not change the agent.yaml model
ABI or the endpoints AgentKit exposes.

## Brokered runtime feasibility decisions

| Runtime adapter | Brokered status | Feasibility decision |
|---|---|---|
| `pydantic-ai` | Conformance/demo only | The current gate swaps to `OfflineEchoRuntime` for Orka brokered conformance. Native pydantic-ai brokered support should only be advertised after a real function-tool pause/resume path can submit Orka `ToolCallResult` values back into the running agent without direct-tool bypass. |
| `microsoft-agent-framework` / `maf` | Conformance/demo only | The current gate swaps to `OfflineEchoRuntime`. Native MAF brokered support needs a framework hook for externally brokered tool calls and long approval waits before production advertisement. |
| `langgraph` | Conformance/demo only | The current gate swaps to `OfflineEchoRuntime`. Native LangGraph brokered support is feasible only with explicit graph/tool-output resume state and direct-tool bypass controls. |

These decisions keep checked-in/runtime-rendered Orka facades truthful: observed mode
is the default, while brokered read/write/coordination are conformance-gated and
must not be enabled for real model adapters until the corresponding native hooks
and Orka conformance evidence exist.
