# Foundry hosted brokered Responses mode

AgentKit can expose a Foundry hosted `/responses` surface that pauses on an
Orka-brokered function call and resumes only after Orka returns a
`function_call_output`. This mode is for hosted agents that must let Orka remain
the sole policy, approval, credential, idempotency, and execution authority.

## Security invariant

Hosted AgentKit receives only safe brokered schemas:

- tool name;
- description;
- brokered class: `read`, `write`, or `coordination`;
- JSON parameters schema; and
- optional `schemaDigest` drift metadata.

Hosted AgentKit must never receive Orka Tool execution URLs, Kubernetes Secret
refs, auth headers, bearer tokens, downstream credentials, approval-bypass
metadata, or durable Orka control-plane credentials. Direct AgentKit-owned tools
are not allowed in the same v0 `agent.yaml` as `brokeredTools`; mixed owned-tool
and brokered-tool mode is intentionally deferred. `/invocations` is disabled when
`brokeredTools` are configured so it cannot bypass Orka. The container emits a
Responses `function_call` item and waits for Orka to execute and resume it.

## Static schema configuration

Foundry hosted-agent endpoint-scoped `/responses` calls do not accept dynamic
request-level `tools` in this mode. Instead, bake safe schemas into `agent.yaml`:

```yaml
brokeredTools:
  - name: check-network-telemetry
    description: Read sanitized optical telemetry.
    brokeredClass: read
    parameters:
      type: object
      properties:
        site:
          type: string
      required: [site]
    schemaDigest: sha256:<optional digest generated from the safe schema>
```

The ABI loader rejects unsafe fields such as `url`, `headers`, `secretRef`,
`auth`, and `token`, rejects unknown brokered classes, requires a top-level JSON
Schema `type: object`, and validates `schemaDigest` when present.

## Drift workflow

Orka Tool CRDs remain the source of truth. Deployment tooling should export only
the safe model-facing subset into AgentKit `brokeredTools`. The shared helper
`agentkit_serve_common.brokered.generate_brokered_tools_from_orka_tool_crds` and
its CLI wrapper produce deterministic entries and compute `schemaDigest`; AgentKit
startup fails if a configured digest no longer matches the safe schema in
`agent.yaml`. Orka still validates every live call against current Tool CRDs at
execution time.

The exporter accepts canonical `core.orka.ai/v1alpha1` `Tool` resources and
reads the governed class from `spec.brokeredToolClass`. Tools without that field
are not brokered and are omitted; exporting an input set with no brokered tools
fails instead of silently downgrading a tool to `read`.

Example export command:

```sh
agentkit-brokered-tools ./orka-tools/*.yaml -o brokered-tools.generated.yaml
```

The output is an `agent.yaml` fragment shaped as:

```yaml
brokeredTools:
  - name: check-network-telemetry
    description: Read telemetry.
    brokeredClass: read
    parameters:
      type: object
    schemaDigest: sha256:...
```

Use `--no-digest` only for ad-hoc demos where drift failure is not desired, and
`--bare` when another deployment templater owns the top-level `brokeredTools` key.

## Responses lifecycle

Initial request:

```json
{"input":"please read telemetry"}
```

Deterministic local brokered mode requires the prompt to name exactly one
configured tool whenever multiple tools are configured, and for any non-conformance
single tool. Only the special `conformance_read` smoke-test schema may be selected
without an explicit tool-name mention. If no configured tool name is present when
explicit selection is required, the request is rejected with
`brokered_tool_selection_required` instead of silently choosing the wrong tool. A
real model-adapter implementation must replace this deterministic selection with
model-driven tool choice.

Brokered response:

```json
{
  "status": "completed",
  "output": [
    {
      "type": "function_call",
      "call_id": "call_<response-id>_1",
      "name": "conformance_read",
      "arguments": "{\"probe\":true}",
      "status": "completed"
    }
  ]
}
```

Continuation request:

```json
{
  "previous_response_id": "<response-id>",
  "agent_session_id": "<platform-session-id>",
  "brokered_continuation_proof": "<configured-proof>",
  "input": [
    {
      "type": "function_call_output",
      "call_id": "call_<response-id>_1",
      "output": "{\"approved\":true,\"output\":{\"success\":true}}",
      "status": "completed"
    }
  ]
}
```

Canonical approved payload exposed back to the model:

```json
{"approved":true,"output":{"success":true}}
```

Canonical denied/error payload:

```json
{
  "approved": false,
  "error": {
    "code": "approval_declined",
    "message": "Human declined dispatch-work-order"
  }
}
```

Both `approved` values describe final results. Neither value represents a
pending human review. Orka holds its original tool call while the review is
pending, then sends exactly one final result for that call. The hosted adapter
uses fixed messages for Orka's `approval_declined`, `approval_expired`,
`approval_cancelled`, `approval_stale`, `tool_execution_failed`, and
`tool_outcome_unknown` codes. A tool failure remains distinct from a declined
review. An unknown outcome finishes the continuation without another model
request, preventing an automatic retry of an uncertain action.

`function_call_output` is privileged continuation input. It is rejected unless a
known `previous_response_id` has a pending matching `call_id` **and** the request
uses the Orka-only continuation path. Configure
`AGENTKIT_FOUNDRY_BROKERED_CONTINUATION_PROOF` and have the Orka hosted-Responses
adapter send it either as `X-AgentKit-Brokered-Continuation-Proof` or as the
top-level JSON field `brokered_continuation_proof`. AgentKit accepts either
candidate so a wrong or stripped header does not override a matching body value.
The proof is never added to model input, persisted response state, or response
payloads. Ordinary client requests must not be able to submit tool results.
Unknown response IDs, unknown call IDs, orphan tool outputs, duplicate conflicts,
malformed outputs, missing or wrong continuation auth, and multiple tool outputs
all return deterministic error envelopes. An identical duplicate continuation
returns its original next response, even if later tool rounds have completed.

The body field is an AgentKit/Orka compatibility extension rather than a standard
OpenAI Responses field. Prove that the target Foundry gateway accepts and forwards
it before relying on this carrier. A static proof in request bodies is suitable
only for bounded demos because platform/request logging can expose replayable
body values; production designs should use a short-lived proof bound to the
response, call, output digest, and expiry.

For Foundry hosted sessions, `agent_session_id` is a gateway routing field. Orka
must capture the platform-returned value from the initial public response and
send it in the continuation body. Inside the container, AgentKit treats
`FOUNDRY_AGENT_SESSION_ID` as the authoritative sandbox identity and uses body,
query, or compatibility-header values only when that hosted identity is absent.
Continuation state with a stored session ID requires the same effective session;
a missing or different identity is rejected with `response_session_mismatch`.
AgentKit does not synthesize or echo a public
`agent_session_id`; Foundry owns that response field.

## Response IDs and state

The adapter uses the Azure hosted Responses SDK ID generator when available, so
new response IDs use the hosted-compatible `caresp_...` form instead of the old
hand-rolled `resp_<uuid>` form.

`/readiness` fails with HTTP 503 when continuation auth is missing and reports
`continuationAuth: missing`; the adapter also refuses to start brokered
function-call responses until the proof is configured.

By default the state backend is in-memory and suitable for deterministic local
smoke and single-replica demos only. Set
`AGENTKIT_FOUNDRY_RESPONSE_STATE_FILE=/path/to/state.json` to persist pending and
completed brokered response state as an atomically rewritten JSON file. For a
Foundry hosted-session smoke, use an already-expanded path under the persisted
session home, such as `$HOME/.agentkit/foundry-response-state.json`, and run one
app process/Uvicorn worker. A literal `$HOME` in an environment value may not be
expanded by the deployment system, so prefer launcher-side `Path.home()` path
construction. The public continuation must still carry the same
`agent_session_id` so Foundry routes it to that sandbox.

An immediate initial/continuation pair proves routing affinity, not file recovery.
To prove persistence, restart the AgentKit process between the two requests in
the same hosted session and show that it reloads the pending state. Preserve the
state file and keep the test inside the configured state TTL and Orka task/session
limits. This does not establish recovery of a lost Foundry runtime session.
Deployments without a session-persisted
file, shared file, or platform-managed store must pin one replica or use sticky
routing; otherwise a continuation that lands on a different/restarted container
fails safely with `unknown_previous_response_id`. Pending state expires after
`AGENTKIT_FOUNDRY_RESPONSE_STATE_TTL_SECONDS` (default: 900 seconds); expired
continuations fail with `response_state_expired`. The store is bounded by
`AGENTKIT_FOUNDRY_RESPONSE_STATE_MAX_PENDING` (default: 128), and generated
brokered arguments are bounded by `AGENTKIT_FOUNDRY_BROKERED_MAX_ARGUMENT_BYTES`
(default: 8192) before state is accepted. Brokered continuation outputs are
bounded by `AGENTKIT_FOUNDRY_BROKERED_MAX_OUTPUT_BYTES` (default: 65536) before
they are persisted, replayed, embedded in deterministic responses, or sent back
through the model loop. A platform-managed state backend is still required
before treating multi-replica production as fully supported.

For Orka-managed human approval, configure
`AGENTKIT_FOUNDRY_RESPONSE_STATE_TTL_SECONDS=1800` and a hosted-session idle
timeout of at least 1800 seconds. The review can consume 600 seconds and tool
execution another 240 seconds, followed by result delivery and model continuation.
The default 900-second state lifetime leaves too little margin at that limit.
Task deadlines, active leases, and platform session limits still apply. See
[Human approval for Orka tools](orka-human-approval.md) for the full wait budget.

The file is sensitive runtime state, not harmless metadata. In model-loop mode
it includes model messages such as system instructions, conversation history,
and user prompts, in addition to brokered arguments/outputs and cached final
payloads. Store it on access-controlled storage and apply an appropriate
retention/deletion policy. Logical TTL is not secure deletion: expired JSON can
remain in a stopped session's file until the store is loaded and rewritten, and
the file may remain visible through Foundry session-file APIs. Use synthetic data
and document the deployment as single-principal/single-tenant unless explicit
Foundry isolation keys are configured consistently for all requests and session
operations.

## Streaming

`/responses` honors `stream: true` with server-sent events, both with and without
`brokeredTools`.
AgentKit sends `response.created` before starting model work, then sends
`response.completed` or `response.failed` with the same response ID. These events
include the effective `agent_session_id` when one is present. This lets Orka
record which hosted response it accepted before waiting for the model.

The stream contains acknowledgement and completion events, without token deltas.
Validation errors before acknowledgement keep their normal HTTP error status.
Disconnecting the stream cancels the active runtime or model request and releases
any in-progress brokered state. Orka still uses authenticated session stop and idle
checks to confirm hosted cleanup. Requests without `stream: true` keep the JSON
response format. Streaming does not add persistence or replay behavior to runtimes
without `brokeredTools`.

## Troubleshooting

- `invocations_disabled_in_brokered_mode`: call `/responses`; brokered mode disables `/invocations` to avoid direct tool bypass.
- `tools_unsupported`: remove request-level `tools`; use static `brokeredTools`.
- `tool_choice_unsupported`: brokered Foundry mode owns tool selection.
- `brokered_continuation_auth_required`: set `AGENTKIT_FOUNDRY_BROKERED_CONTINUATION_PROOF` before accepting brokered continuations.
- `brokered_continuation_forbidden`: only Orka should send the configured proof in `X-AgentKit-Brokered-Continuation-Proof` or top-level `brokered_continuation_proof`.
- `missing_previous_response_id`: a `function_call_output` cannot start a new run.
- `unknown_previous_response_id`: state is missing, expired/purged, or routed to a
  different replica.
- `response_session_mismatch`: the continuation reached a different effective Foundry session than the one that created the pending response.
- `response_pending_function_call_output`: finish the pending brokered tool call before sending a normal follow-up turn against that response.
- `brokered_tool_selection_required`: name exactly one configured brokered tool in deterministic mode when multiple schemas or any non-conformance single schema are configured.
- `brokered_response_state_full`: too many uncontinued brokered responses are pending. Completed entries are evicted before this error is returned.
- `brokered_arguments_too_large`: generated brokered call arguments exceeded the configured pending-state byte budget.
- `brokered_output_too_large`: a brokered `function_call_output` exceeded the configured output byte budget.
- `unknown_call_id`: the output did not match the pending function call.
- `conflicting_duplicate_continuation`: the same `call_id` was already completed
  with different output.

## SDK conformance spike app

`agentkit_serve_common.foundry_conformance.create_foundry_conformance_app()` is a
small, production-independent hosted Responses app for Phase A0 smokes. It is
also packaged as the `agentkit-foundry-conformance` console script so the same
app can be used as a hosted container entrypoint. It uses
`azure-ai-agentserver-responses` for request parsing, SDK-assigned `caresp_...`
response IDs, response envelopes, event sequencing, and in-memory response state.
It returns the deterministic `conformance_read` function call on the first
`/responses` request and completes when resumed with the matching
`function_call_output`.

Local runnable entrypoint check:

```sh
uv run --directory runtimes/common --extra dev --extra foundry-conformance agentkit-foundry-conformance --dry-run
```

Container entrypoint example for the A0 spike image:

```Dockerfile
ENTRYPOINT ["agentkit-foundry-conformance", "--host", "0.0.0.0", "--port", "8088"]
```

A minimal deployable container fixture lives in `test/foundry-brokered-conformance/`:

```sh
docker buildx build --builder desktop-linux . \
  -f test/foundry-brokered-conformance/Dockerfile \
  --platform linux/amd64 \
  -t agentkit-foundry-brokered-conformance:test --load --provenance=false
```

Use `test/foundry-brokered-conformance/foundry.agent.yaml.example` as the hosted
agent manifest template; it advertises `responses` protocol version `2.0.0` for
this A0 spike. You can also run the transcript helper against a local container
by setting `AGENT_RESPONSES_ENDPOINT=http://127.0.0.1:18088/responses` and
`AGENT_RESPONSES_BEARER_TOKEN=local-dummy-token`.

For a one-command local pre-deploy smoke of the SDK conformance image, run:

```sh
deploy/foundry/scripts/local_brokered_conformance_container.sh \
  --fixture sdk \
  --platform linux/amd64 \
  --transcript-dir ./foundry-brokered-local-transcript
```

Local proof:

```sh
uv run --directory runtimes/common --extra dev --extra foundry-conformance pytest -q tests/test_foundry_conformance.py
```

Live direct-endpoint proof after deployment:

```sh
export AGENT_RESPONSES_ENDPOINT="https://<hosted-agent>/responses"
export AZURE_SUBSCRIPTION_ID="<subscription>"
deploy/foundry/doctor.sh --brokered-conformance
deploy/foundry/scripts/foundry_brokered_conformance.sh conformance_read ./foundry-brokered-transcript
```

The script performs the initial `function_call` request, posts the matching
`function_call_output` continuation with `previous_response_id`, asserts SDK-style
`caresp_...` IDs, and saves request/response JSON plus `summary.json` as a
sanitized transcript. Re-run `python3 deploy/foundry/scripts/verify_brokered_transcript.py <transcript-dir> --expected-final-text '<known expected assistant result>'` to verify archived transcript evidence later. This live transcript is still required before claiming A0
completion; the local test only proves the SDK-hosted contract before deployment.

## Production brokered-only fixture

`test/foundry-brokered-agentkit/` packages the production AgentKit Foundry
brokered path, not the standalone SDK conformance app. It bakes a minimal
`agent.yaml` with static `brokeredTools` and runs:

```sh
agentkit-foundry-brokered --config /agent/agent.yaml --host 0.0.0.0 --port 8088
```

Local proof for this production adapter path:

```sh
docker build . -f test/foundry-brokered-agentkit/Dockerfile -t agentkit-foundry-brokered:local
docker run --rm \
  -e AGENTKIT_AUTH_TOKEN=local-dummy-token \
  -e AGENTKIT_FOUNDRY_BROKERED_CONTINUATION_PROOF=local-dev-proof \
  -p 127.0.0.1:18092:8088 \
  agentkit-foundry-brokered:local

AGENT_RESPONSES_ENDPOINT=http://127.0.0.1:18092/responses \
AGENT_RESPONSES_BEARER_TOKEN=local-dummy-token \
AGENTKIT_CONTINUATION_PROOF=local-dev-proof \
AGENTKIT_EXPECTED_CALL_ID=auto \
AGENTKIT_EXPECTED_CALL_ID_PREFIX=call_ \
deploy/foundry/scripts/foundry_brokered_conformance.sh \
  conformance_read ./foundry-brokered-agentkit-transcript
```

Use this fixture when you want to validate AgentKit's generated `call_<response-id>_1`
call IDs and continuation-proof enforcement before pushing a real brokered
AgentKit image. The all-in-one local helper supports this fixture too:

```sh
deploy/foundry/scripts/local_brokered_conformance_container.sh \
  --fixture agentkit \
  --platform linux/amd64 \
  --transcript-dir ./foundry-brokered-agentkit-transcript
```

## Model-driven tool workflows

Set `AGENTKIT_FOUNDRY_BROKERED_MODEL_LOOP=1` to let the model choose tools and
work through a task. `AGENTKIT_MODEL_API` selects the upstream model API:

- `chat_completions` is the default, preserving existing Chat-only backends.
- `responses` explicitly selects the Responses API for models that need it.
- Other values fail startup validation. No model-name inference, automatic API
  fallback, or automatic disabling of reasoning is performed.

For a reasoning-model deployment that requires Responses for function tools:

```sh
AGENTKIT_FOUNDRY_BROKERED_MODEL_LOOP=1
AGENTKIT_MODEL_API=responses
```

This setting controls the upstream model connection, not AgentKit's hosted
`/responses` endpoint. Both APIs are also available through the direct Pydantic
AI, Microsoft Agent Framework, and LangGraph adapters, including standalone
Docker runs. MAF supports both APIs with API-key, token-hook, or Foundry
project-credential auth. The model backend must support the selected API;
AgentKit does not silently switch to the other one. Invalid selections fail
before SDK/auth initialization. In this brokered loop, Chat mode sends `messages` to
`<model.baseURL>/chat/completions`; Responses mode sends `input` items to
`<model.baseURL>/responses`. Both expose the same safe static `brokeredTools`
and preserve Orka's sequential approval and continuation contract. A base URL
ending in either API endpoint is normalized to the selected API's sibling;
the startup setting, not the URL suffix, determines the API.

Each request resends the retained transcript. Responses requests use
`store: false`, retain replayable encrypted reasoning in its original position,
and preserve assistant message boundaries, phase, and refusal text. Responses
results must have `status: "completed"`; Chat results with a truncation or other
non-completion or missing finish reason are rejected. Model output and tool arguments
remain bounded and validated in either mode.

The selected API is recorded with new pending model-loop state. Resuming that
state with a different API returns `brokered_model_api_mismatch` before accepting
the tool result or making another model request. Unmarked legacy model-loop
state is treated as Chat Completions. Finish pending turns from Responses-only
development builds before upgrading, because those builds did not record the API.
`AGENTKIT_MODEL_API` is startup configuration, not a per-turn `input.env` override.

Each operational tool call returns a `function_call` for Orka to execute. After
Orka sends the matching `function_call_output`, AgentKit resumes the model. It
can request another tool or return an answer. Every round gets a fresh response
ID and call ID; the next result must match both. AgentKit validates each new
call's name, arguments, and schema before returning it. Retrying the same result
returns the cached next response without another model request. File-backed
state preserves these completed rounds across restarts.

The model can make up to 16 sequential tool calls per user turn. At the limit,
AgentKit asks for a final answer with `tool_choice: "none"` and rejects any
further tool call. The model endpoint must support `parallel_tool_calls: false`, which asks
for one tool call at a time. AgentKit also rejects parallel tool batches if a
model ignores that setting. AgentKit-owned MCP and direct operational tools
remain disabled.

An HTTP 429 from the model service is retried up to twice within the same
hosted response. AgentKit honors `retry-after-ms` or `Retry-After` delays of
up to 60 seconds each. A longer server delay ends the response with
`ModelUnavailable` and `upstream_status: 429`. Missing or malformed delay
headers use short exponential backoff with jitter. Disconnecting the hosted
stream cancels the wait. Other HTTP errors, transport failures, and invalid
model responses are not retried. These model retries do not repeat Orka tool
operations or submit a new hosted response.

Agents can also use [bundled instruction skills](instruction-skills.md). With
filesystem skills configured under `/agent/skills`, AgentKit advertises the
local `load_skill` tool alongside the brokered schemas. It returns the packaged
instructions from an immutable startup snapshot. Skill loads count toward the
same 16-call budget; they do not invoke Orka or execute scripts. All operational
calls described by those instructions still go through Orka.

For Orka harness v2, use the companion
[agent-runtime-foundry broker](https://github.com/orka-agents/agent-runtime-foundry/blob/main/docs/harness-v2.md).
Give its `ORKA_FOUNDRY_BROKER_AGENTKIT_CONTINUATION_PROOF` the same value as the
hosted agent's `AGENTKIT_FOUNDRY_BROKERED_CONTINUATION_PROOF`, using a secret of
at least 32 bytes without whitespace. The broker adds this value only after
checking ownership, the active lease, and the expected response and call. It
also translates MCP results into AgentKit's approved/error envelope. Keep the
proof out of the ACP child configuration. The gateway must forward the proof
field; local tests cannot establish that a public Foundry deployment does so.
See the shared-proof limitations above. Human tool approvals require a qualified
Orka controller, supervisor, Foundry broker, and AgentKit image that support
`supportsBrokeredToolApprovals`. Older combinations remain unsupported. Review
policy, reviewer permissions, cancellation, and execution decisions stay in Orka.

## Hosted follow-up questions

In model-loop mode, a follow-up user message can reference the final response ID
from the previous turn. It must use the same effective hosted session identity.
AgentKit retains recent user and assistant messages so the model can understand
questions such as "Which incident was that?" Raw tool messages and system
instructions are excluded from this retained dialogue. An assistant answer can
still contain information derived from a tool result.

History keeps complete recent exchanges within half the model-message byte
limit or one eighth of the response-state byte limit, whichever is smaller.
It shares the response store's TTL, capacity, and optional file persistence.
Missing, expired, evicted, or mismatched session history is rejected. Requests
without a session identity remain stateless. Pending tool calls must finish
before a user can continue their response.

## Implementation status and evidence

This section records the current AgentKit-side evidence against the implementation
plan. It is intentionally explicit about what is locally proven versus what still
requires deployed Foundry/Orka/Fibey state.

| Plan area | Current AgentKit status | Evidence | Remaining gate |
|---|---|---|---|
| Golden hosted Responses fixtures | Implemented locally | `runtimes/common/tests/fixtures/foundry_brokered/*`, `tests/test_foundry_brokered_protocol.py` | Orka repo must consume/verify the same wire shapes. |
| A0 SDK hosted Responses spike | Local SDK app and container fixture implemented | `agentkit_serve_common.foundry_conformance`, `agentkit-foundry-conformance`, `test/foundry-brokered-conformance/`, `tests/test_foundry_conformance.py` | Deploy to Foundry and archive a verified live transcript with `deploy/foundry/scripts/foundry_brokered_conformance.sh`. |
| A1 hosted Responses lifecycle/state | Implemented for deterministic/local brokered mode; file-backed state available for single-writer/sticky deployments | `agentkit_serve_common.foundry.create_foundry_app`, `tests/test_foundry_brokered_protocol.py` | Live Foundry must accept generated response IDs and state/routing constraints must be chosen for deployment. |
| A2 static schemas and drift control | Implemented in Go writer/validator and Python runtime; export CLI added | `brokeredTools` ABI, `agentkit-brokered-tools`, `tests/test_config_validation.py`, `tests/test_brokered_schema.py`, Go config/ABI tests | Orka Tool CRDs must be exported during deployment and current digests verified before live runs. |
| A3 deterministic brokered runtime | Implemented for local/fake hosted protocol integration | deterministic `/responses` brokered path and tests | Live Orka deterministic read/write smoke still required. |
| A4 framework pause/resume decision | Lower-level OpenAI-compatible fallback implemented; high-level framework native hooks remain gated | `agentkit_serve_common.foundry_model_loop`, `AGENTKIT_FOUNDRY_BROKERED_MODEL_LOOP=1`, model-loop tests | Live model smoke for brokered read/write prompts. |
| A5 first real model adapter brokered mode | Model loop supports sequential brokered calls, local instruction skills, and bounded hosted dialogue | `tests/test_foundry_brokered_protocol.py`, `tests/test_foundry_tool_workflows.py` | Deployed real model read and write prompts, including policy/error outcomes. |
| A6 live Orka integration | Not proven in this repo state | Local AgentKit/Foundry side helpers exist | Deploy AgentKit and Orka hosted-Responses adapter; run brokered read/write approval smoke. |
| A7 Fibey | Runtime supports packaged skills and sequential tool workflows | Catalog/ACP/MAF skill tests and hosted workflow tests | Package the Fibey skills and schemas, then validate the deployed scenario. |
| A8 hardening/docs/review | Local docs/tests/autoreview complete for current patch | This doc, `docs/agent-abi.md`, `docs/runtime-capabilities.md`; full tests/lint; `$autoreview` clean | Record live transcript and Orka/Fibey validation evidence before final completion. |

Local verification commands used for the current AgentKit patch:

```sh
uv run --directory runtimes/common --extra dev pytest -q
go test ./...
make lint
git diff --check
.agents/skills/autoreview/scripts/autoreview
```

Completion must still be judged against live evidence: this local status matrix is
not a substitute for Foundry accepting `previous_response_id`, Orka brokering the
actual read/write calls, or Fibey completing the end-to-end scenario.
