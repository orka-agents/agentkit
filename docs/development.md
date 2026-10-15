# Development guide

This repository contains a Go BuildKit frontend, Python runtime adapter packages,
Docker adapter images, and integration fixtures. The Makefile is the source of
truth for the local Docker loop.

## Local prerequisites

- Go matching `go.mod`.
- Python 3.12 for runtime package work.
- Docker Buildx.
- A daemon-backed Buildx builder for `--load` workflows. The Makefile defaults
  `BUILDER=desktop-linux` for test-agent builds because docker-container builders
  cannot see local images unless they are pushed to a registry.

## Core loop

```sh
make build-agentkit      # frontend gateway image -> agentkit:test
make build-serve         # pydantic-ai adapter    -> agentkit-serve:test
make build-test-agent    # fixture agent image    -> hello-agent:test
make run-test-agent      # run the image; needs OPENAI_API_KEY
```

`build-test-agent` connects the local images with build args:

- `BUILDKIT_SYNTAX=agentkit:test` makes the fixture use the local frontend.
- `adapter=agentkit-serve:test` makes the Go converter use the local adapter
  image as the LLB base.

To exercise the experimental production composition path instead:

```sh
make build-agentkit TAG=spike
make build-portable-harness-spike TAG=spike PLATFORM=linux/arm64
make build-composed-test-agent-spike TAG=spike PLATFORM=linux/arm64
```

This passes paired `baseImage` and `harnessImage` inputs through the real
frontend. The target defaults to `ubuntu:24.04` and can be overridden with
`BASE_IMAGE`, `HARNESS_IMAGE`, and `COMPOSED_AGENT_IMAGE`.

## Runtime-specific loops

Microsoft Agent Framework:

```sh
make build-serve-maf
make build-test-agent RUNTIME=maf
```

LangGraph:

```sh
make build-serve-langgraph
make build-test-agent RUNTIME=langgraph
```

`RUNTIME` selects the adapter image, fixture file, and output image tag:

| `RUNTIME` | Adapter image | Fixture | Output image |
|---|---|---|---|
| `pydantic-ai` | `agentkit-serve:test` | `test/agentkitfile-hello.yaml` | `hello-agent:test` |
| `maf` / `microsoft-agent-framework` | `agentkit-serve-maf:test` | `test/agentkitfile-maf-hello.yaml` | `maf-agent:test` |
| `langgraph` | `agentkit-serve-langgraph:test` | `test/agentkitfile-langgraph-hello.yaml` | `langgraph-agent:test` |

## Go checks

```sh
golangci-lint run ./... --timeout 5m
golangci-lint fmt --diff
go vet ./...
go test ./... -race
go build -o /tmp/agentkit-frontend ./cmd/frontend
```

Go tests cover:

- strict Agentkitfile loading and validation,
- instruction source resolution,
- runtime aliasing and route lookup,
- effective Agent defaults and copy semantics,
- ABI rendering and golden round trips,
- OCI image config generation, and
- runtime catalog file parity.

## Python checks

For one adapter package:

```sh
cd runtimes/langgraph
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ../common -e '.[dev]' build
python -m compileall agentkit_serve ../common/agentkit_serve_common
python -m pytest -q
python -m build --wheel
```

For `runtimes/common`, omit `-e ../common` and compile/test
`agentkit_serve_common` directly.

Python tests cover:

- ABI reader validation,
- OpenAI façade conformance shared by every adapter,
- conversation normalization,
- runtime lifecycle startup/shutdown,
- tool env allowlist behavior,
- MCP timeout parsing,
- framework import guardrails, and
- adapter-specific result/usage mapping.

## Docker and smoke checks

The CI Docker job builds:

1. the frontend image,
2. all three adapter images,
3. a fixture agent image for each runtime,
4. each generated agent enough to pass `/healthz`, and
5. one generated agent in `AGENTKIT_PROTOCOL=orka` mode far enough to prove the
   native harness health/capabilities, bearer auth, turn acceptance, and SSE
   terminal-frame shape.

The smoke containers bind `0.0.0.0` inside the container and set
`AGENTKIT_AUTH_TOKEN`, proving the startup auth gate is satisfied while probe
endpoints remain unauthenticated. The Orka smoke uses an already-expired turn
`deadline` so it can verify native Orka failure frames offline without calling a
live model provider.

## Harness v2 end-to-end checks

The composed v2 checks build the current AgentKit frontend and agent images,
layer Orka's production supervisor onto each immutable image, and exercise the
real ACP, provider, and MCP paths. The normal PR/push offline matrix covers
`pydantic-ai`, `microsoft-agent-framework`, and `langgraph` with deterministic
local fixtures and no external model credentials. A separate live MAF lane uses
Copilot through the digest-pinned Vekil image and requires real model, tool, and
session-continuation results.

```sh
scripts/orka-harness-v2-e2e.sh offline
scripts/orka-harness-v2-e2e.sh offline langgraph
VEKIL_CACHE_DIR="$HOME/.config/vekil" scripts/orka-harness-v2-e2e.sh live
```

Run these commands in a Linux shell on the Docker daemon's host, with a
daemon-backed Buildx builder and Go matching the pinned Orka module, currently Go 1.27 at commit
`55cb3d5232b4a9b697e72471e346c0a6493d4c21`. The runner fetches that exact revision;
no pre-existing Orka checkout is needed. `BUILDER` selects the builder, and
`PLATFORM` defaults to the Docker daemon's Linux amd64/arm64 architecture. Allow
network access for registry images and build dependencies. For live mode,
`COPILOT_GITHUB_TOKEN` can replace the local Vekil auth cache.

Set `ARTIFACT_DIR` to keep safe JSON results with the adapter, scenario, source
and image digests, and failure diagnostics. Configured live authentication or
inference failures fail the check; unavailable CI secret access produces an
explicit skip. A skip does not establish live coverage. See
[the v2 test guide](orka.md#test-the-composed-v2-runtime) for scenario assertions,
session retirement rules, credentials, and cleanup behavior. The existing v1
container and OpenAI HTTP smoke jobs remain independent checks.

Validate the runner syntax and focused ACP input/output behavior locally:

```sh
bash -n scripts/orka-harness-v2-e2e.sh
shellcheck scripts/orka-harness-v2-e2e.sh
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 .github/workflows/*.yml
uv run --directory runtimes/common --extra dev pytest -q tests/test_acp_protocol.py tests/test_cli_protocol.py
```

## Live Copilot/Vekil E2E

The optional live job runs `scripts/live-copilot-agent-e2e.sh` when
`COPILOT_GITHUB_TOKEN` is available and the run is allowed to access repository
secrets. It uses `ghcr.io/sozercan/vekil:v0.14.10`, pinned by digest, to provide an
OpenAI-compatible endpoint and validates a real built AgentKit container through
`/v1/chat/completions`.

For local workstation validation, the same script can also use an existing Vekil
auth cache instead of a token. Leave `COPILOT_GITHUB_TOKEN` unset and set
`VEKIL_CACHE_DIR` if your cache is not in `~/.config/vekil`:

```sh
VEKIL_HOST_PORT=31339 \
AGENTKIT_LIVE_HOST_PORT=18086 \
TAG=e2e-script \
scripts/live-copilot-agent-e2e.sh
```

When `PLATFORM` is unset, the script picks `linux/arm64` on arm64 Docker hosts
and `linux/amd64` on amd64 hosts so locally built adapter images match the live
agent image.

Fork PRs, Dependabot runs without secrets, and repos without the token/cache skip
or fail before live provider calls while normal offline checks still run.

## Foundry Hosted Agents fixture

`AGENTKIT_PROTOCOL=foundry` now exposes Foundry Hosted Agents' `/readiness`,
`/invocations`, and minimal `/responses` surfaces directly from every adapter
image. `test/foundry-hosted-agent/` remains as a compatibility smoke fixture for
older wrapping flows; AgentKit still owns `/agent/agent.yaml` and the runtime
lifecycle in both modes.
