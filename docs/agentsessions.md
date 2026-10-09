# agentsessions Harness (protocol skeleton)

Select `--protocol agentsessions` or `AGENTKIT_PROTOCOL=agentsessions` on any
adapter's existing `agentkit-serve` entrypoint. This is native protobuf gRPC
`agentsessions.v1.Harness.Describe` and bidirectional `Harness.Connect`, not an
HTTP/ACP facade. **Model and tool execution are not implemented yet.** A valid
Start ends with `FAILED`, gRPC code `UNIMPLEMENTED` (12); no provider is called.

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
Model/tool/approval replies are refused with `UNIMPLEMENTED`; repeated Start and
unknown frames fail with `INVALID_ARGUMENT`. Failures use fixed descriptions,
never exception text, request content, Config or credentials. Protobuf send and
receive messages are bounded at 4 MiB, including all framing fields.

## Explicit execution seam (for the next implementation)

The common package exports:

```python
load_verified_agentsessions_binding(path: str | Path) -> VerifiedAgentsessionsBinding
# binding.spec, configuration_digest, implementation_digest, descriptor_id

ExecutionRunner = Callable[
    [VerifiedAgentsessionsBinding, RunRequest],
    Awaitable[RunResult | None],
]
create_server(binding, *, runner=None, auth_token=None) -> grpc.aio.Server
async def serve(binding, *, bind="127.0.0.1", port=8080, auth_token=None, runner=None): ...
def run(binding, *, bind="127.0.0.1", port=8080, auth_token=None, runner=None): ...
```

The returned server preserves the gRPC server API. `stop()` waits for execution
cleanup even when its caller is canceled. `wait_for_termination()` includes that
cleanup and retains its timeout result without canceling the shutdown. Stopping
before the first start is a no-op, as in native gRPC. Public TCP port binding
also enforces the nonloopback authentication gate; Unix sockets are local-only.

CLI recognizes only an optional adapter `async run_agentsessions(binding,
request) -> RunResult | None` hook. There is **no fallback to `build_runtime`**.
The hook is awaited once per Start and must own a fresh execution and close its
resources under cancellation (`async with`/`finally`). No shipped adapter
implements it in this PR. The injection seam exercises neutral protocol tests;
it is not permission to call a provider directly.

`RunRequest.config: bytes = b""` receives Start.Config unchanged and is excluded
from the dataclass repr. Config is never parsed or injected into prompts/env.
History INPUT/OUTPUT messages become ordered user/assistant text turns; prior
Inputs are appended once and the final Input becomes `prompt`. An inputless
Start uses an empty prompt without fabricating an input. Nontext, tool roles,
tool events, unknown history and mismatched message bodies fail closed. Known
host metadata is not prompt content; cursor and identity do not become hidden
framework state. There is no cross-turn framework/session cache.

A returned RunResult emits one finalized assistant OUTPUT (even empty text),
then COMPLETED END; `None` emits only COMPLETED END. Usage is not invented or
forwarded by the skeleton. Exceptions produce FAILED/INTERNAL with a sanitized
message. The next PR must add the serialized, correlated model-effect exchange
and fresh host-bound adapter execution before enabling a real model runner; it
must avoid duplicating host-journaled model completions as OUTPUT. No model
forwarding, HTTP bridge, model retry or tool mediation is present here.

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
dependency expansion). This proves wire compatibility, **not host replay**;
controller/journal/container replay evidence belongs to the next PR.
