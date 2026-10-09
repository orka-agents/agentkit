# agentsessions Harness

Select `--protocol agentsessions` or `AGENTKIT_PROTOCOL=agentsessions` on an
adapter's existing `agentkit-serve` entrypoint. This is native protobuf gRPC
`agentsessions.v1.Harness.Describe` and bidirectional `Harness.Connect`, not an
HTTP/ACP facade. **Pydantic AI supports host-mediated, text-only Chat model
execution.** Other adapters fail closed with `FAILED/UNIMPLEMENTED` (12); there
is no fallback to their normal provider runtime. Tools remain unsupported.

The wire sources are pinned to
[`aramase/agentsessions@b212d498ba52615b5087b2579bdd482809065642`](https://github.com/aramase/agentsessions/tree/b212d498ba52615b5087b2579bdd482809065642),
Go module `v0.1.3-0.20261006182201-b212d498ba52`. That host journals and restores
opaque `Start.Config` via `EVENT_EXECUTION_START`.

## Immutable startup binding

Set both deployment-owned values (lowercase `sha256:<64 hex>`):

- `AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST`: SHA-256 of the **exact
  bytes** of `/agent/agent.yaml` (or `--config`). YAML-equivalent rewrites fail.
- `AGENTKIT_AGENTSESSIONS_IMPLEMENTATION_DIGEST`: immutable adapter image digest
  covering the adapter implementation and installed dependency closure. Do not
  use a moving tag. Deployment must retain that implementation for replay.

The descriptor ID is `agentkit:<configuration digest>:<implementation digest>`.
Describe advertises the baked model name, stateless replay and fork safety for
this restricted profile; no tools, GPU, streaming or reasoning replay.
Deployment-supplied digests bind identity; they are **not authentication** or an
attestation of an independently measured image.

Startup rejects baked direct tools, brokered tools and all context providers.
A nonempty credential in the baked model's `apiKeyEnv` is rejected without
printing its value. Credential-bearing model URL userinfo, queries and fragments
are also rejected without echoing the URL. Absent provider credentials and required-env declarations
are not resolved. Original model URLs and workload identity are never used.
Do not inject provider secrets into the process, image, Config or argv.

## Network and controls

`AGENTKIT_BIND` defaults to `127.0.0.1`; `AGENTKIT_PORT` overrides the baked
expose port (normally 8080). Nonloopback binding requires
`AGENTKIT_AUTH_TOKEN`. When configured, **both** RPCs require exactly one gRPC
metadata entry `authorization: Bearer <token>`, including on loopback. Start
identity is not used as authentication. Clients must inject metadata explicitly;
the upstream reference harnesswire client does not do this by itself.

Transport is h2c (no built-in TLS). A nonloopback listener must be isolated on a
private network, or behind a trusted TLS/auth proxy; possession of a bearer
token does not make public plaintext transport safe. No HTTP readiness endpoint
is added by this skeleton.

Connect admits one execution at a time. Start must be first and carry a nonempty
execution ID (at most 1024 UTF-8 bytes, leaving room for bounded terminal frames).
Later frames must match **both** the execution ID and the session
string exactly; empty session strings are allowed (the reference client omits
session). Invalid first frames terminate the RPC with `INVALID_ARGUMENT`.
Once admitted, completion/failure/cancellation ends with exactly one END while
writable, using the original execution ID on every event. Other admissions get
`FAILED/RESOURCE_EXHAUSTED` without disturbing the active execution.

The reader stays live independently of the execution task. Cancel, EOF/half-close
and transport disconnect cancel execution and release admission after cleanup.
Keep the request stream open until END, as the host reference client does.
Correlated ModelResult replies are validated by the independent reader. Missing
or duplicate results, mismatched call IDs, non-assistant/nontext content, oversized
results, negative usage or a different usage model fail with `INVALID_ARGUMENT`.
Unsolicited ModelResult is also invalid. Tool/approval replies are refused with
`UNIMPLEMENTED`; repeated Start and unknown frames fail with `INVALID_ARGUMENT`.
Failures use fixed descriptions,
never exception text, request content, Config or credentials. Protobuf send and
receive messages are bounded at 4 MiB, including all framing fields.

## Explicit per-execution exchange

The common package exports:

```python
load_verified_agentsessions_binding(path: str | Path) -> VerifiedAgentsessionsBinding
# binding.spec, configuration_digest, implementation_digest, descriptor_id

ExecutionRunner = Callable[
    [VerifiedAgentsessionsBinding, RunRequest, ExecutionExchange],
    Awaitable[RunResult | None],
]
# exchange.input_count distinguishes zero Inputs from an empty text prompt.
# await exchange.call(messages: Sequence[common_pb2.Message]) -> common_pb2.Message
create_server(binding, *, runner=None, auth_token=None) -> grpc.aio.Server
async def serve(binding, *, bind="127.0.0.1", port=8080, auth_token=None, runner=None): ...
def run(binding, *, bind="127.0.0.1", port=8080, auth_token=None, runner=None): ...
```

The returned server preserves the gRPC server API. `stop()` waits for execution
cleanup even when its caller is canceled. `wait_for_termination()` includes that
cleanup and retains its timeout result without canceling the shutdown. Stopping
before the first start is a no-op, as in native gRPC. Public TCP port binding
also enforces the nonloopback authentication gate; Unix sockets are local-only.

CLI recognizes only the optional adapter `async run_agentsessions(binding,
request, exchange) -> RunResult | None` hook. This extends the internal PR1
hook; no adapter shipped its earlier two-argument form. There is **no fallback
to `build_runtime`**, callback-in-metadata or environment-selected model bridge.
The hook is awaited once per Start and owns fresh cancellable resources.

`RunRequest.config: bytes = b""` receives Start.Config unchanged and is excluded
from the dataclass repr. The pinned host journals/replays these bytes. **Config
is opaque and ignored by the current text policy**: it is never parsed into
credentials, environment variables, model options or prompts. Do not treat
nonempty Config as a request to change model behavior.

History INPUT/OUTPUT messages become ordered user/assistant text turns, including
empty text. Earlier Inputs are appended once and the final Input becomes
`prompt`, also including empty text. The exchange retains the original input
count so an inputless invocation does not fabricate a user message. Nontext,
tool roles/events, unknown history, model options and mismatched message bodies
fail closed before execution. Known host metadata is not prompt content; cursor
and identity do not become hidden framework state. No cross-turn cache exists.

The exchange emits `EVENT_MODEL_CALL` with the baked model, ordered text
messages, deterministic per-execution `model-1`, `model-2`, ... IDs, empty params
and no input_hash. **The host computes the hash, records the effect, invokes the
model (or serves the journaled result on replay), and journals OUTPUT.** There is
one pending effect and a one-event queue; overlapping calls are refused. The
pydantic hook returns `None`, avoiding an additional OUTPUT or fabricated USAGE.
A neutral test hook can still return RunResult to emit an independent finalized
assistant OUTPUT. Exceptions produce sanitized FAILED/INTERNAL END.

## Pydantic AI loopback text policy

Each Start creates an ephemeral `127.0.0.1` HTTP listener with a fresh local
bearer token (not a provider key). A fresh SDK AsyncOpenAI client is explicitly
injected into OpenAIProvider, then a fresh OpenAIChatModel and Agent are created.
The original provider URL, apiKeyEnv and workload identity are not resolved.
HTTP client proxy/environment discovery and redirects are disabled; SDK retries
are zero, and no direct-provider fallback exists. Explicit local Authorization
also prevents ambient `OPENAI_CUSTOM_HEADERS` from replacing the per-run token.
Client, Agent, listener and pending request resources close before admission is
released.

The bridge accepts only `/v1/chat/completions` requests with the exact baked model,
ordered `system`/`user`/`assistant` text messages and an optional boolean `stream`.
Tool schemas/calls/roles, developer roles, file/data/reasoning content and all
other options (including stream_options/usage requests) are rejected before a
model effect. Requests and encoded responses are bounded to 1 MiB; ModelResult
is also bounded to 1 MiB, within the gRPC 4 MiB transport limit. Concurrent HTTP
bodies are refused and transport concurrency is bounded.

`stream=true` supports **buffered synthetic terminal SSE only**, after the entire
validated ModelResult arrives: one final chunk with `finish_reason="stop"`, then
`[DONE]`. This is not real streaming and no usage is invented or advertised.
Describe continues to advertise streaming=false and reasoning_replay=false.
The pydantic hook uses documented Agent graph iteration to finish after the SDK
model request, before framework tool/output-validation retries. This preserves
valid empty host completions as well as empty conversation turns.

## Regeneration and checks

`runtimes/common/agentkit_serve_common/agentsessions/` contains the exact proto
inputs, upstream Apache-2.0 license, source hash/provenance manifest and generated
Python stubs. Runtime compatibility bounds are separate from exact compiler pins
in `scripts/agentsessions-generator-requirements.txt`.

```sh
scripts/generate-agentsessions-stubs.sh --fetch  # fetch exact pinned inputs, verify hashes, generate
scripts/generate-agentsessions-stubs.sh          # regenerate from vendored, verified inputs
scripts/generate-agentsessions-stubs.sh --check  # regenerate in temp space; fail on drift, no writes
uv run --directory runtimes/common --extra dev pytest -q
```

The script uses `uv`; `UV=/absolute/path/to/uv` overrides its executable. CI
checks regeneration and common tests run a native Go client against the Python
gRPC server using an isolated module under `test/agentsessions/` (no root Go
dependency expansion). That client check proves wire compatibility, **not host
replay**.

### Built-image controller replay proof

Run on the **Linux Docker daemon host** (the test connects directly to a private
container IP on an internal-only network):

```sh
scripts/agentsessions-e2e.sh --build
```

The script builds the BuildKit frontend, Pydantic AI adapter and AgentKit-derived
image from `test/agentsessions/Agentkitfile.yaml`, then sets
`AGENTKIT_AGENTSESSIONS_IMAGE` and runs `TestContainerStatelessReplay` in the
nested Go module. A plain nested `go test` **skips** this container proof when
that image variable is absent. `--skip-build` uses an already-built image; rebuild
after implementation changes before treating its result as fresh evidence.

The fixture uses the exact pinned agentsessions controller, native harnesswire
client and SQLite journal against the real Python/SDK container. An offline host
model serves two live turns; there is no Pydantic TestModel override or provider
credential. It verifies ordered model context and exact opaque Config/cursor
bytes in wire Starts and execution-start journal records. It then closes and
reopens SQLite, replaces the container and calls **`Controller.Replay`**, actually
re-executing both turns against recorded model effects. A provider that fails if
invoked must remain unused. Replay must restore journaled Config/cursors rather
than current constructor values, preserve outputs and the journal, and reject a
changed model-input fingerprint without invoking the provider.

The recorded full-image run observed **2 live model calls, 2 journaled model
calls, and 0 replay model calls**, with equal outputs, an unchanged verified
journal, restored Config/cursors and successful container/database restart.
The fixture checks absence of provider credentials and uses an internal-only,
no-egress Docker network. `AGENTSESSIONS_REPLAY_PROOF` reports the immutable image
digest, host pin and these assertions. `Sessions.Replay` journal redelivery alone
is not re-execution proof. This is a container/controller text-profile proof,
**not service or placement integration**.
