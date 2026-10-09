# Use an AgentKit image with Orka

AgentKit supports two distinct Orka integrations. New BYO deployments should
use `orka.harness.v2`: Orka's hardened supervisor runs AgentKit as an ACP stdio
child. The older `orka.harness.v1` mode remains an observed HTTP+SSE adapter.

## Harness v2 BYO runtime

Build the AgentKit agent image first and address it by digest. In an Orka
checkout, layer the v2 supervisor onto that immutable image:

```sh
make docker-build-acp-agentkit-runtime \
  AGENTKIT_RUNTIME_IMAGE=ghcr.io/acme/fibey@sha256:<agentkit-image-digest> \
  AGENTKIT_ADAPTER_DIGEST=sha256:<agentkit-image-digest> \
  ACP_AGENTKIT_RUNTIME_IMG=ghcr.io/acme/fibey-orka-v2:dev
```

The composed image runs `orka-acp-runtime`. For each RuntimeSession, the
supervisor starts this child under a private UID/GID and session tree:

```sh
/opt/agentkit/bin/agentkit-serve \
  --config /agent/agent.yaml \
  --protocol acp
```

The v2 path is strict:

- `/agent/agent.yaml` must not contain direct `tools` or `brokeredTools`;
- the Microsoft Agent Framework runtime also supports bundled, instruction-only
  filesystem skills; other context providers remain prohibited. See
  [Bundled skills in governed mode](instruction-skills.md);
- the registered model must equal `model.name` in the baked config;
- `agentConfigurationDigest` is `sha256:` plus the SHA-256 of the exact
  `/agent/agent.yaml` bytes;
- the runtime advertises only `agentkit-serve-acp`, with the digest-pinned
  AgentKit source image identity as its adapter digest;
- Orka sends `AgentConfiguration: null`; the image-bound config is authoritative;
- provider calls use the supervisor's loopback proxy, and tools use its one
  prompt-scoped loopback HTTP MCP server;
- the child retains successful user/assistant history for Session continuation.

Before reusing a session after a successful prompt, the caller must use Orka's
canonical `CreateWorkspaceDelta` operation to validate the workspace, including
when no files changed. Failure, cancellation, and lease expiry retire the v2
session. The ACP child's separate history rollback behavior does not make a
retired supervisor session reusable.

Deploy the composed image as an operator-owned v2 supervisor service, configure
the standard `ORKA_ACP_*` profile, fence, token-file, and runtime identity
settings, then register it with Orka's strict-governed `AgentRuntime` sample.
The profile model and `agentConfigurationDigest` must match the baked AgentKit
config. The registration's `adapterName` must be `agentkit-serve-acp`. Its
`adapterDigest` and the composition build's `AGENTKIT_ADAPTER_DIGEST` must both
equal the `sha256:` digest from `AGENTKIT_RUNTIME_IMAGE`. Set the profile's
`providerKind` to `agentkit` and advertise
`supportsAgentSessionConfiguration: false` and `supportsPermissions: false`.
Orka-managed MCP tool approval uses the existing waiting call, without a local
ACP permission callback. A qualified Microsoft Agent Framework composition can
advertise `supportsBrokeredToolApprovals: true` and configure Orka's
`approvalRequiredTools`. The controller, supervisor, and AgentKit image must all
support that contract. Older images and unqualified adapters remain unsupported.
If the registration allows brokered tools, the Task must submit that exact
`allowedTools` list.

See [Human approval for Orka tools](orka-human-approval.md) for wait limits,
result handling, and acceptance checks through Orka's approval API.

Set `ORKA_ACP_CONTROLLER_EPOCH` from Orka's current `ControllerEpoch` record.
Select by `spec.name` because the resource name is hashed. This lookup requires
exactly one matching record with a positive integer epoch:

```sh
kubectl -n <orka-controller-namespace> get cepoch -o json |
  jq -er '
    [.items[] | select(.spec.name == "orka-controller") | .status.epoch] |
    if length == 1 then
      .[0] | select(type == "number") | select(. > 0 and . == floor)
    else
      error("expected exactly one ControllerEpoch for orka-controller")
    end'
```

The supervisor reads the epoch only during startup. The operator that owns this
service must watch the record and restart or replace the supervisor whenever it
changes. Preserve `ORKA_ACP_RUNTIME_INSTANCE_ID` across that restart and issue a
new `ORKA_ACP_SUPERVISOR_BOOT_ID`. Orka keeps a stale-epoch registration not
ready and refuses new Task bindings until authenticated status reports the
current value. AgentKit itself is the ACP child and does not manage this fence.

Orka freezes the AgentRuntime UID, generation, endpoint, profile, authentication
Secret versions, and observed instance into each Task binding. It revalidates
them before dispatch and recovery mutations. `Task.spec.execution.workspace`
is not supported for external runtimes; repository input still uses
`Task.spec.workspace`.

See Orka's `website/docs/guides/bring-your-own-agent-runtime.md` and
`config/samples/core_v1alpha1_agentruntime.yaml` for the registration and
authentication contract.

## Test the composed v2 runtime

Run the shared offline/live entrypoint from the AgentKit checkout:

```sh
# Offline defaults to pydantic-ai, microsoft-agent-framework, and langgraph.
scripts/orka-harness-v2-e2e.sh offline
scripts/orka-harness-v2-e2e.sh offline microsoft-agent-framework

# Live defaults to all three adapters against a bundled local AIKit model.
scripts/orka-harness-v2-e2e.sh live
scripts/orka-harness-v2-e2e.sh live pydantic-ai
scripts/orka-harness-v2-e2e.sh live langgraph
```

Run in a Linux shell on the Docker daemon's host so the runner and daemon share
the temporary registry's loopback address. The runner needs Git, make, curl, jq,
Go 1.27, and Docker Buildx with a daemon-backed builder. Set `BUILDER` to select that builder. `PLATFORM`
defaults to `linux/amd64` or `linux/arm64` according to the Docker daemon's
architecture. Image, registry, and dependency downloads require network access.

Each run fetches Orka at
[`55cb3d5232b4a9b697e72471e346c0a6493d4c21`](https://github.com/orka-agents/orka/tree/55cb3d5232b4a9b697e72471e346c0a6493d4c21),
whose `go.mod` declares Go 1.27. A sibling Orka checkout is unnecessary. The runner
builds AgentKit from the current checkout through the Makefile flow, resolves each
source agent image through a run-owned registry to an immutable digest, and uses
that same digest for `AGENTKIT_RUNTIME_IMAGE` and `AGENTKIT_ADAPTER_DIGEST`. It
composes the image with Orka's official AgentKit Dockerfile. The registered model
and configuration digest match the baked model and exact `/agent/agent.yaml`
bytes.

The test driver uses Orka's native v2 client from within the pinned module. Every
mode runs the production supervisor, real ACP child, framework clients, provider
proxy, and prompt-scoped MCP proxy. Offline mode supplies deterministic local
provider and controller/broker fixtures and needs no external model credentials.
A passing offline run requires these observable results:

- startup, health/capabilities, adapter identity, and the child's private process
  identity match the registered profile;
- a provider response carries a unique marker, an allowed MCP tool reaches the
  broker with the expected arguments and correlation, and its receipt returns
  to the provider and affects the answer;
- two successful prompts share a session, with prior user/assistant history
  appearing exactly once at the provider;
- injected failures settle as failures, and a confirmed blocked broker call
  settles on cancellation or lease expiry without a conflicting success;
- wrong authentication, fences, model, or config identity start no unauthorized
  provider/tool work, and session cleanup removes the child and private paths.

Live mode runs `ghcr.io/kaito-project/aikit/qwen3.5:2b`, pinned by digest in
`scripts/aikit-e2e-common.sh`, directly on the run-owned Docker network.
It requires no provider credentials and makes no external inference calls.
`test/aikit-e2e/model.yaml` configures bounded CPU inference, greedy sampling,
and native tool templates. The runner warms the model before the timed turns.
Both modes accept an optional adapter argument; `maf` aliases
`microsoft-agent-framework`. Each live adapter must produce a real model response,
return an exact MCP tool receipt, continue the same session with a second
successful prompt, and cancel during a blocking tool.
The live provider bridge removes the OpenAI `strict` tool-schema hint because
LocalAI 4.10 treats it as forced tool-only generation even with automatic tool
choice. Tool definitions and parameters, provider history, and v2 assertions
remain unchanged. Offline scenarios remain the deterministic failure and
deadline checks.
Readiness and inference errors fail the live run; CI does not skip live coverage
based on repository secrets.

Set `ARTIFACT_DIR` to retain sanitized JSON results, for example:

```sh
ARTIFACT_DIR=/tmp/agentkit-v2-results scripts/orka-harness-v2-e2e.sh offline
```

Read results by adapter and scenario. They record source/image digests, asserted
outcomes, and safe diagnostics. Raw transcripts, bearer tokens, and session
credentials are excluded. The runner removes its containers, networks, registry,
and temporary resources on success, failure, or interruption. Existing harness
v1 and OpenAI HTTP smoke checks run separately.

## Harness v1 observed mode

AgentKit images can expose observed-mode `orka.harness.v1` without rebuilding the
agent. Start the same image with Orka mode enabled:

```sh
docker run --rm \
  -e AGENTKIT_PROTOCOL=orka \
  -e AGENTKIT_AUTH_TOKEN=dev-token \
  -e AGENTKIT_BIND=0.0.0.0 \
  -p 127.0.0.1:8080:8080 \
  ghcr.io/acme/fibey@sha256:...
```

Open endpoints:

- `GET /v1/health`
- `GET /v1/capabilities`

Bearer-authenticated endpoints:

- `POST /v1/turns`
- `GET /v1/turns/{turnID}/events?afterSeq=...`
- `POST /v1/turns/{turnID}/continue` (brokered gates only)
- `POST /v1/turns/{turnID}/cancel`
- `GET /v1/turns/{turnID}/output?ref=...` (reserved; returns 404 unless a future
  adapter stores large outputs by reference)

In observed mode, AgentKit maps one Orka turn to one `RuntimeSession.run(RunRequest)` call. It
emits Orka-native `HarnessEventFrame` SSE frames and exactly one terminal frame
(`TurnCompleted`, `TurnFailed`, or `TurnCancelled`). AgentKit-owned tools and MCP
servers continue to execute inside the runtime; Orka observes the run and remains
responsible for policy, approvals, trust tiers, Tool CRDs, idempotency, and
side-effect governance.

Orka sends only the new prompt for each turn. AgentKit keeps the completed
user/assistant turns of each `runtimeSessionID` and passes them to the runtime as
history, so every runtime adapter continues a session the same way. Failed and
cancelled turns are not added. If a run reports a fatal runtime failure, such as
an MCP tool session closing, the next turn in that runtime session builds a
fresh runtime with the same history. History is kept apart from runtimes, so a
session also keeps it when its runtime is evicted for capacity
(`AGENTKIT_ORKA_MAX_RUNTIME_SESSIONS`, default 64) or fails to restart. AgentKit
keeps the history of up to `AGENTKIT_ORKA_MAX_SESSION_HISTORIES` sessions
(default 256, never fewer than the runtime session limit) and drops the least
recently used session without a live runtime first. Each session's history is
capped at `AGENTKIT_ORKA_MAX_SESSION_HISTORY_BYTES` of text (default 1 MiB);
past that, the oldest completed turns are dropped.

Runtimes start lazily on a session's first turn. If one fails to start, for
example because a remote MCP tool rejects its credential, the turn fails with
`RuntimeStartFailed`. Tool clients put credential-bearing URLs and upstream
bodies in startup errors, so the AgentKit log records only the exception types.
Configuration errors such as a missing env var keep their own message.

Current AgentKit Serve Orka support is **observed mode by default**. The default
capability response intentionally omits `brokeredToolClasses` and
`supportsContinuation`. Brokered read, write, and coordination are implemented
behind `AGENTKIT_ORKA_ENABLE_BROKERED_READ=1`,
`AGENTKIT_ORKA_ENABLE_BROKERED_WRITE=1`, and
`AGENTKIT_ORKA_ENABLE_BROKERED_COORDINATION=1` conformance gates for runtimes that
implement the internal brokered-tool Interface.

This repository's `agentkit-serve` implementation is separate from OpenAI's
public AgentKit/Agents SDK product surface. A future adapter may target that
product explicitly, but this Orka mode documents only the local AgentKit Serve
runtime described in this repo.

## Offline Orka conformance/demo mode

Set `AGENTKIT_ORKA_OFFLINE_ECHO=1` only for conformance tests or local demos
that must not call a model provider. In Orka mode this replaces the adapter's
framework runtime with a no-provider echo runtime while preserving the same
`orka.harness.v1` HTTP skin, auth behavior, event stream, cancellation, and
capability response. This is useful for kind demos that need
`AgentRuntime` readiness to pass without live model credentials; do not use it
for production agent deployments.

AgentKit still enforces its per-turn environment allowlist in Orka mode. When an
AgentKit image is used behind Orka `Agent.spec.runtime.runtimeRef`, declare any
Orka-injected env names that the controller may send in the AgentKitfile ABI. For
the offline/kind demos this typically includes:

```yaml
env:
  - name: ORKA_CONTROLLER_URL
  - name: ORKA_RESULT_ENDPOINT
  - name: ORKA_PARENT_TASK
  - name: ORKA_PRIOR_TASK
  - name: ORKA_PRIOR_TASK_NAMESPACE
  - name: ORKA_COORDINATION_DEPTH
```

For brokered read/write/coordination conformance, also set the matching
`AGENTKIT_ORKA_ENABLE_BROKERED_*` gate. Those gated modes advertise
`toolExecutionModes: [observed, brokered]`, the enabled `brokeredToolClasses`,
and `supportsContinuation: true`, emit `ToolCallRequested`, accept
`/v1/turns/{turnID}/continue`, emit `ToolResultReceived`, and then complete the
turn. Coordination tool policy, quotas, child-task lineage, and agent/namespace
rules remain Orka-owned; AgentKit receives only safe brokered tool schemas and
results. The offline demo coordinator can target a worker Agent with
`AGENTKIT_ORKA_OFFLINE_DELEGATE_AGENT`; production adapters should enable
brokered gates only after their native tool pause/resume path passes conformance.

## Orka wire contract

`GET /v1/health` returns Orka `HealthResponse`:

```json
{
  "version": "orka.harness.v1",
  "status": "ok",
  "ready": true,
  "checkedAt": "2026-06-27T00:00:00Z",
  "metadata": {"agentName": "fibey-agentkit"}
}
```

`GET /v1/capabilities` returns flat Orka `CapabilitiesResponse` fields:

```json
{
  "version": "orka.harness.v1",
  "protocolVersion": "orka.harness.v1",
  "transport": "http+sse",
  "runtimeName": "agentkit-serve",
  "runtimeVersion": "0.0.0",
  "providerKind": "kubernetes-service",
  "toolExecutionModes": ["observed"],
  "supportsCancel": true,
  "supportsRuntimeSessions": true,
  "supportsSuspend": false,
  "supportsWorkspaceSnapshot": false,
  "maxConcurrentTurns": 1,
  "metadata": {
    "agentName": "fibey-agentkit",
    "model": "gpt-4o-mini",
    "agentkitProvider": "openai-compatible"
  }
}
```

Start a turn with Orka `StartTurnRequest`:

```json
{
  "version": "orka.harness.v1",
  "namespace": "default",
  "taskName": "fibey-task",
  "sessionName": "fibey-session",
  "runtimeSessionID": "runtime-session-1",
  "turnID": "turn-1",
  "correlationID": "corr-1",
  "deadline": "2026-06-27T00:05:00Z",
  "authIdentity": {"subject": "system:serviceaccount:default:orka"},
  "input": {
    "prompt": "Investigate alert A-123",
    "contextRefs": [],
    "env": [
      {"name": "FOO", "value": "BAR"}
    ]
  },
  "toolExecutionMode": "observed",
  "metadata": {}
}
```

AgentKit responds with Orka `StartTurnResponse`:

```json
{
  "version": "orka.harness.v1",
  "accepted": true,
  "runtimeSessionID": "runtime-session-1",
  "turnID": "turn-1",
  "correlationID": "corr-1",
  "eventStreamPath": "/v1/turns/turn-1/events"
}
```

`input.contextRefs` are accepted as safe Orka references. AgentKit does not fetch
Orka-owned context objects in observed mode yet; instead it forwards the reference
list to runtime adapters only in `RunRequest.metadata["contextRefs"]`. It does not
promote request-controlled references into model prompt or system history.

SSE `data:` payloads are Orka `HarnessEventFrame` objects. A successful observed
turn normally emits `TurnStarted`, `RuntimeOutput`, then `TurnCompleted`:

```json
{
  "version": "orka.harness.v1",
  "type": "TurnCompleted",
  "runtimeSessionID": "runtime-session-1",
  "turnID": "turn-1",
  "correlationID": "corr-1",
  "seq": 3,
  "createdAt": "2026-06-27T00:00:02Z",
  "severity": "info",
  "summary": "turn completed",
  "content": {},
  "contentText": "",
  "completed": {
    "result": "assistant response text",
    "finalEventSeq": 3
  },
  "failed": null,
  "error": null,
  "metadata": {}
}
```

Cancel with Orka `CancelTurnRequest`:

```json
{
  "version": "orka.harness.v1",
  "namespace": "default",
  "taskName": "fibey-task",
  "sessionName": "fibey-session",
  "runtimeSessionID": "runtime-session-1",
  "turnID": "turn-1",
  "correlationID": "corr-1",
  "reason": "user requested cancel"
}
```

AgentKit responds with Orka `CancelTurnResponse`:

```json
{
  "version": "orka.harness.v1",
  "accepted": true,
  "runtimeSessionID": "runtime-session-1",
  "turnID": "turn-1",
  "correlationID": "corr-1",
  "message": "cancel accepted"
}
```

Rejected requests use Orka's native error body, `{"error": "<message>"}`.
Starting a turn whose `turnID` is still running returns 409
`turn already exists`; a retained terminal turn returns 409
`turn already completed`; and a start while another turn runs returns 409
`maximum concurrent turns reached`. Orka's client matches these exact messages.

CI runs Orka's own AgentKit conformance suite
(`internal/harness/conformance`, `TestExternalAgentKitServe*`) at the pinned
`test/orka-harness-v2/orka-revision` against this checkout's
`create_orka_app`.

## Offline smoke coverage

The GitHub Actions container smoke starts one built test-agent image with:

```sh
AGENTKIT_PROTOCOL=orka
AGENTKIT_BIND=0.0.0.0
AGENTKIT_AUTH_TOKEN=ci-smoke-token
```

It verifies native Orka health/capabilities, confirms unauthenticated turn start
is rejected, starts an authenticated turn with an already-expired deadline, and
checks the SSE stream returns a `TurnStarted` frame followed by a terminal
`TurnFailed` frame with Orka identity fields and no legacy `payload` field. The
expired deadline keeps this smoke offline: it proves the harness protocol without
calling OpenAI or another model endpoint.

For deeper local validation, run the common Python Orka protocol tests:

```sh
uv run --directory runtimes/common --extra dev pytest -q tests/test_orka_protocol.py
```

## Render a harness v1 AgentRuntime manifest

The AgentKit renderer currently emits the harness v1 registration shape. Deploy
the AgentKit image yourself, for example as a Kubernetes Deployment/Service,
with:

- `AGENTKIT_PROTOCOL=orka`
- `AGENTKIT_BIND=0.0.0.0` so the Kubernetes Service can reach the harness outside
  the container network namespace
- `AGENTKIT_AUTH_TOKEN` sourced from the same Secret referenced by
  `spec.clientAuth.bearerTokenSecretRef`

Then render the AgentRuntime registration for that endpoint:

```sh
agentkit render --target orka-agentruntime \
  --external-endpoint http://fibey-agentkit.default.svc.cluster.local:8080 \
  --name fibey-agentkit
```

Output shape:

```yaml
apiVersion: core.orka.ai/v1alpha1
kind: AgentRuntime
metadata:
  name: fibey-agentkit
spec:
  contractVersion: orka.harness.v1
  deployment:
    mode: external-endpoint
    endpoint: http://fibey-agentkit.default.svc.cluster.local:8080
  clientAuth:
    bearerTokenSecretRef:
      name: fibey-agentkit-harness-token
      key: token
  capabilities:
    toolExecutionModes:
      - observed
    supportsCancel: true
    supportsRuntimeSessions: true
```

Use `--auth-secret-name` and `--auth-secret-key` when your client-auth Secret
uses a different name or key. The `--image` flag is reserved for a future Orka
managed-image CRD mode and fails clearly against the current external-endpoint
schema.
