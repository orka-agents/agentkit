# The `agent.yaml` ABI (v0)

`/agent/agent.yaml` is the built contract between the Go frontend writer and the
Python runtime reader.

- The **writer** renders it from an Effective Agent and bakes it into the image.
- The **reader** (`agentkit-serve-common`) loads it at startup with strict schema
  validation.

The ABI is intentionally target-neutral. Provider-specific deployment profiles or
resource provisioners should map external resources to the generic fields below
instead of adding provider-specific top-level keys.

## Location

`/agent/agent.yaml` (constant `abi.Path`).

## Example

```yaml
abiVersion: v0

metadata:
  name: url-summarizer

model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: "OPENAI_API_KEY"
  # Optional future generic model auth. Capability-gated; apiKeyEnv remains the
  # normal v0 model-auth path.
  # auth:
  #   type: workload-identity-token
  #   audience: https://example.com/.default

instructions: |
  Summarize any URL the user gives you in three bullet points.

tools:
  # Stdio MCP server.
  - name: fetch
    command: ["npx", "-y", "@modelcontextprotocol/server-fetch"]
    env: ["FETCH_TIMEOUT"]

  # Remote MCP server over Streamable HTTP.
  - name: toolbox
    type: mcp
    transport: streamable-http
    urlEnv: TOOLBOX_ENDPOINT
    headers:
      - name: Foundry-Features
        value: Toolboxes=V1Preview
      - name: X-Trace
        valueEnv: TOOLBOX_TRACE_HEADER
    auth:
      type: bearer
      tokenEnv: TOOLBOX_TOKEN

# Alternative to `tools` above for Foundry hosted Orka-brokered mode. v0 does
# not allow owned `tools` and `brokeredTools` together, so remove/comment the
# `tools` block before enabling this schema-only block.
# brokeredTools:
#   - name: check-network-telemetry
#     description: Read sanitized optical telemetry.
#     brokeredClass: read
#     parameters:
#       type: object
#       properties:
#         site:
#           type: string
#       required: [site]
#     schemaDigest: sha256:<optional deploy-time digest>

env:
  - name: REQUIRED_FOO
    required: true
  - name: OPTIONAL_BAR

# Provider-neutral context schema. Runtime behavior is capability-gated.
context:
  providers:
    - name: knowledge
      type: search
      endpointEnv: SEARCH_ENDPOINT
      indexEnv: SEARCH_INDEX
    - name: support-style
      type: skills
      source: filesystem
      path: /agent/skills
    - name: user-memory
      type: memory
      endpointEnv: MEMORY_ENDPOINT
      storeNameEnv: MEMORY_STORE_NAME

observability:
  otel:
    endpointEnv: OTEL_EXPORTER_OTLP_ENDPOINT
  # logs.levelEnv is reserved but rejected until a runtime wires log-level support.

expose:
  openai: true
  port: 8080
```

## Top-level fields

| Field | Required | Description |
|---|---:|---|
| `abiVersion` | yes | ABI version understood by the runtime reader. Current value: `v0`. |
| `metadata.name` | yes | Agent image/name label. |
| `model` | yes | Hosted OpenAI-compatible model connection metadata. |
| `instructions` | yes | Fully-resolved system prompt scalar. |
| `tools` | no | Owned MCP tools, either stdio or Streamable HTTP. |
| `brokeredTools` | no | Static safe Orka-brokered tool schemas for Foundry hosted Responses mode. |
| `env` | no | Runtime env var requirements by name only. |
| `context` | no | Provider-neutral context providers; runtime capability-gated. Filesystem skills paths must be pre-staged under `/agent/skills`; arbitrary build-context directories are not copied into the image. |
| `observability` | no | Provider-neutral observability env names; runtime capability-gated. |
| `expose` | yes | Serving surface and port. |

## Tool forms

### Stdio MCP

```yaml
tools:
  - name: fetch
    command: ["uvx", "mcp-server-fetch"]
    env: ["FETCH_TIMEOUT"]
```

Runtime adapters spawn the command for the app lifespan and pass **only** the env
var names listed in `env` when those variables are present in the container
environment.

### Streamable HTTP MCP

```yaml
tools:
  - name: toolbox
    type: mcp
    transport: streamable-http
    urlEnv: TOOLBOX_ENDPOINT
    headers:
      - name: Foundry-Features
        value: Toolboxes=V1Preview
      - name: X-API-Key
        valueEnv: TOOLBOX_API_KEY
    auth:
      type: bearer
      tokenEnv: TOOLBOX_TOKEN
```

`urlEnv`, `valueEnv`, and `tokenEnv` are env var names. Values are resolved at
runtime and never baked into the image. Static credential headers such as
`Authorization`, `Cookie`, `X-API-Key`, and provider subscription-key headers are
rejected; use `valueEnv` or `auth` instead. Runtime HTTP clients inject headers
same-origin only and avoid replaying credentials across redirects.

Supported auth types:

- `bearer` with `tokenEnv`.
- `workload-identity-token` with opaque `audience`, only for runtimes that declare
  `workload-identity-token-auth`.

## Brokered tool schemas

`brokeredTools` is used by Foundry hosted brokered Responses mode. It is
intentionally schema-only: the reader rejects execution URLs, auth/header/token
fields, Secret refs, unsafe parameter names, unknown brokered classes, malformed
JSON Schema, duplicate names, and owned-tool/brokered-tool name overlap.

When `schemaDigest` is present, it must match the deterministic digest of the
safe model-facing schema fields. Generate it from Orka Tool CRDs during
deployment so stale or hand-edited schemas fail before a live run. Orka remains
the execution and policy authority even when AgentKit's static schema is valid.

See `docs/foundry-hosted-brokered.md` for the hosted Responses continuation
lifecycle and state/scaling limits.

## Reader contract

The shared Python runtime core must:

1. load and validate this file, exiting non-zero on missing, invalid, or
   unsupported files;
2. validate `env[]` requirements and report missing env var names without values;
3. construct the selected adapter runtime from the validated `AgentSpec`;
4. resolve `model.apiKeyEnv` from the process environment or use the no-auth
   placeholder when no key env is declared;
5. start MCP tool sessions for the application lifespan:
   - stdio tools receive only their declared env allowlist;
   - remote tools resolve URL/header/auth material from env names and generic auth;
6. serve the OpenAI-compatible façade consistently across adapters.

## Writer contract

The Go frontend must:

1. validate the authored Agentkitfile before rendering,
2. resolve instruction sources into a scalar string,
3. canonicalize runtime aliases and default the port before ABI rendering,
4. emit exactly the keys the reader expects,
5. preserve tool, env, context, and observability declarations after validation,
   and
6. never write secret values.

## Served HTTP contract

With `AGENTKIT_PROTOCOL=openai`, the runtime serves:

- `GET /healthz` — liveness.
- `GET /v1/models` — one-model listing containing `model.name`.
- `POST /v1/chat/completions` — non-streaming run that returns one
  `chat.completion` object.
- `POST /v1/responses` runs synchronously with stateless, text-only input and
  returns one completed assistant message with `output_text`, `input_tokens`,
  `output_tokens`, and `total_tokens` usage, and `store: false`.

Both POST routes share the listener, runtime session, baked MCP tools, auth,
health state, and application lifespan. Optional `X-AgentKit-Session-Id` is
forwarded to runtime adapters for provider-neutral session/memory correlation;
it does not retain or replace the client-supplied transcript.

`AGENTKIT_MODEL_API` independently selects the upstream model transport:
`chat_completions` by default, explicit `responses`, or opt-in `auto`. Neither
client route forces an upstream API, and no new protocol or ABI field is needed.
Both inbound endpoints remain available. Explicit selectors never fall back.

For example, pass `-e AGENTKIT_MODEL_API=auto` to the container. Auto makes the
first real Responses request without a separate probe and caches the concrete
API for that runtime/backend/model lifetime. Only a recognized initial
unsupported endpoint/API rejection permits one Chat retry. Unknown 404s,
model-missing errors, 401/403/429, timeouts, and generic 5xx failures do not.
Once any Responses HTTP request is accepted, including streams and
malformed/incomplete output, auto never switches APIs. A permitted retry closes
Responses runtime/tool resources before Chat starts and preserves request
history and tools; upstream Responses calls keep `store: false`.
See [model endpoint compatibility](runtime-adapters.md#model-endpoint-compatibility).

Foundry brokered auto persists and restores the concrete API for continuations.
Explicit selector/continuation mismatch protections are unchanged.

Runtime protocol selection happens outside the ABI with `agentkit-serve
--protocol` or `AGENTKIT_PROTOCOL`; the same `/agent/agent.yaml` file is reused by
every protocol skin.

The reusable Foundry wrapper in `agentkit_serve_common.foundry` reuses the same
`RuntimeFactory` / `RuntimeSession` seam and exposes `/readiness`, `/invocations`,
and a minimal non-streaming `/responses` endpoint. It forwards Foundry session
IDs from query/header data to the runtime and tolerates client-supplied
`stream: true` by returning a normal completed non-streaming response.

The reusable Orka wrapper in `agentkit_serve_common.orka` exposes observed-mode
`orka.harness.v1` over `/v1/health`, `/v1/capabilities`, `/v1/turns`, SSE events,
and cancel. Orka turn metadata is carried in optional `RunRequest` fields and
does not change the ABI file shape.

`POST /v1/chat/completions` rejects:

- `stream: true`,
- request-supplied `tools`,
- request-supplied `tool_choice` values other than missing, empty, `none`, or
  `auto`,
- an empty `messages` array, and
- requests whose final message is not a `user` message.

The Chat response collapses any intermediate framework/tool loop into one
assistant message with `finish_reason: "stop"`.

`POST /v1/responses` accepts `input` as a string or a non-empty message array
whose final message is `user`. Array items must be messages with `system`,
`developer`, `user`, or `assistant` roles. Content is a string or a list of
`input_text`/`output_text` parts with string `text`. Developer messages become
system history; assistant `phase`, when supplied as `commentary` or
`final_answer`, is preserved in normalized history. Top-level `instructions`
becomes client system history before the input history and after the baked
instructions. Each request supplies its own conversation. Usage reports aggregate
counts. Cache and reasoning detail counters are zero because the runtime seam
does not report those breakdowns.

Before runtime or tool execution, Responses rejects:

- `stream: true`, non-empty `tools`, or specific `tool_choice` values. Only
  missing, empty, `none`, and `auto` tool choices are accepted;
- `previous_response_id` or `conversation` state references;
- `background: true` or `store: true`;
- non-message input items, including function calls and function outputs, or
  multimodal content, including images, audio, and files; and
- empty message arrays, unsupported roles/content/phase, or a final message
  whose role is not `user`.

Responses runs baked tools internally and returns their final text, not client
function-call continuations. These restrictions apply to generic
`/v1/responses`, not Foundry `/responses` or its brokered continuation contract.

## Network and process contract

Generated images run `agentkit-serve --config /agent/agent.yaml` as user
`1000:1000`. They bind `127.0.0.1` by default. If `AGENTKIT_BIND` is set to a
non-loopback host such as `0.0.0.0`, startup requires `AGENTKIT_AUTH_TOKEN`; then
`/v1/*` requests must include `Authorization: Bearer <token>`. `/healthz` remains
open.
