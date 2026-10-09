# agentkit-serve-common

`agentkit-serve-common` is the framework-neutral Python core shared by all
AgentKit runtime adapters. It contains the runtime behavior that must be
identical whether the selected adapter is pydantic-ai, Microsoft Agent Framework,
or LangGraph.

## Modules

- `config.py` — strict `/agent/agent.yaml` ABI loader, version check, env
  requirement validation, and provider-neutral schema validation.
- `cli.py` — `agentkit-serve --config ... --protocol acp|openai|foundry|orka`,
  bind/port handling, and startup auth gates for non-loopback binds and Orka
  protected endpoints.
- `acp.py` — ACP protocol v1 over newline-delimited JSON-RPC on stdio for the
  Orka harness v2 supervisor.
- `server.py` — FastAPI app for `/healthz`, `/v1/models`,
  `/v1/chat/completions`, and `/v1/responses` on one listener.
- `responses.py` normalizes text-only Responses input and shares
  response/usage encoding.
- `foundry.py` — reusable Foundry Hosted Agent protocol wrapper for
  `/readiness`, `/invocations`, and minimal non-streaming `/responses`.
- `orka.py` — observed-mode `orka.harness.v1` wrapper for `/v1/health`,
  `/v1/capabilities`, `/v1/turns`, SSE replay, and cancel.
- `conversation.py` — protocol request normalization into a framework-neutral
  `RunRequest`, including optional per-turn env/deadline/metadata fields.
- `runtime.py` — `RuntimeFactory`, `RuntimeSession`, `RunResult`, and
  `AgentRunError`.
- `adapter_support.py` — API-key resolution, declared-only tool env projection,
  remote MCP URL/header/auth resolution, MCP HTTP client factories, MCP timeout
  parsing, and framework exception normalization.
- `conformance.py` — shared HTTP behavior tests imported by adapter test suites.

## Adapter seam

`server.create_app(spec, factory, auth_token)`,
`foundry.create_foundry_app(spec, factory)`, and
`orka.create_orka_app(spec, factory, auth_token)` receive an adapter module that
satisfies `RuntimeFactory`. The shared core calls only `factory.build_runtime(spec)`
and `RuntimeSession.run(request) -> RunResult`. It never imports framework
packages or touches raw framework agent lifecycle.

This keeps framework dependency lock-in inside each adapter's `agent_factory.py`.

## Generic client APIs

`AGENTKIT_PROTOCOL=openai` serves Chat Completions and Responses together. Both
routes use the same runtime session, baked tools, auth, health state, and app
lifespan. `AGENTKIT_MODEL_API` selects only the upstream model API:
`chat_completions` by default or explicit `responses`, with no `auto` or fallback.
It neither selects nor disables either client route.

`POST /v1/responses` accepts text `input` or a message array ending in a user
message. It supports system/developer/user/assistant roles, `input_text` and
`output_text` parts, and assistant `phase` in history. Top-level `instructions`
becomes client system history after the baked instructions. Both routes forward
`X-AgentKit-Session-Id` for correlation without retaining a transcript.
Responses returns completed assistant `output_text`, usage, and `store: false`.
Streaming, request tools or specific tool choices, stored conversations,
background execution, and non-message/multimodal input are rejected before a
run. Foundry `/responses` and brokered continuations keep their separate
contracts. See [the HTTP contract](../../docs/agent-abi.md#served-http-contract).

## Orka ACP child mode

Orka harness v2 starts the adapter as one child process per RuntimeSession:

```sh
agentkit-serve --config /agent/agent.yaml --protocol acp
```

ACP mode requires `AGENTKIT_ACP_AGENT_CONFIGURATION_DIGEST` to equal the
`sha256:` digest of the exact config file bytes and `AGENTKIT_ACP_MODEL` to
equal `model.name`. It replaces the baked model endpoint and credential with
`AGENTKIT_ACP_PROVIDER_BASE_URL` and `AGENTKIT_ACP_PROVIDER_TOKEN`.

The child accepts one ACP session, text and resource-link prompt blocks,
cancellation, and at most one loopback HTTP MCP server carrying a bearer
Authorization header. Resource links are added to the model prompt as labeled
text and are never fetched by the child. The runtime keeps successful user and
assistant turns for later prompts. It rejects baked `tools` and `brokeredTools`.
The Microsoft Agent Framework adapter accepts
[packaged skill instructions](../../docs/instruction-skills.md); other context
providers remain prohibited. Orka owns process and workspace isolation, prompt-scoped
MCP authority, provider proxying, and cleanup proof.

## Adding a runtime adapter

A new single-agent adapter should provide:

1. `agent_factory.py` implementing `build_runtime(spec) -> RuntimeSession`,
2. a thin `__main__.py` that calls `agentkit_serve_common.cli.run(agent_factory)`,
3. adapter tests that import the shared conformance checks, and
4. an adapter image that installs this common package before the adapter package.

Adapters remain separate images with separate framework dependencies, while this
package is installed into each image as the shared façade/runtime core.


## Brokered tool schema export

`agentkit-serve-common` includes a small deployment helper for Foundry hosted
Orka-brokered mode:

```sh
agentkit-brokered-tools ./orka-tools/*.yaml -o brokered-tools.generated.yaml
```

It reads Orka Tool CRD YAML/JSON documents and writes a safe `brokeredTools:`
`agent.yaml` fragment containing only name, description, brokered class, JSON
parameters schema, and optional schema digest. Execution URLs, auth headers,
Secret refs, tokens, and other credential-shaped schema fields are rejected or
omitted before the fragment is model-visible.

Inputs must use the canonical `core.orka.ai/v1alpha1` `Tool` shape. The exporter
reads `spec.brokeredToolClass`; unclassified tools are not brokered and are
skipped, and an input set with no classified tools fails rather than defaulting
their class to `read`.


## Foundry brokered conformance app

The common package also installs `agentkit-foundry-conformance`, a tiny
Azure Responses SDK app for Phase A0 hosted brokered smokes. It serves
`/readiness` and `/responses`, emits a deterministic `conformance_read`
`function_call`, and completes after a matching `function_call_output`
continuation.

Install the optional `foundry-conformance` extra when using this SDK-backed
entrypoint; normal runtime adapter images install the common package without it.

```sh
uv run --extra foundry-conformance agentkit-foundry-conformance --host 0.0.0.0 --port 8088
```
