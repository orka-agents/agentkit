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

Images and the main Python job install the newest release in each declared
dependency range. CI also runs every package's tests at the lowest declared
direct versions, so a lower bound that no longer works fails the build, and the
whole workflow runs nightly to catch upstream releases that change behavior. To
reproduce the lowest-version run:

```sh
cd runtimes/langgraph
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --resolution lowest-direct -e ../common -e '.[dev]'
.venv/bin/python -m pytest -q
```

Python tests cover:

- ABI reader validation,
- OpenAI façade conformance shared by every adapter,
- wire-level parity shared by every adapter: each adapter's real runtime runs
  against a scripted loopback model and a stdio MCP fixture, and must send the
  same conversation and tool results to the model, return the same results and
  normalized errors, and keep secret canaries out of client-visible output,
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
4. each generated agent far enough to pass `/healthz`,
5. one generated agent in `AGENTKIT_PROTOCOL=orka` mode far enough to prove the
   native harness health/capabilities, bearer auth, turn acceptance, and SSE
   terminal-frame shape, and
6. a parity agent image for each runtime, checked by the container parity smoke.

The smoke containers bind `0.0.0.0` inside the container and set
`AGENTKIT_AUTH_TOKEN`, proving the startup auth gate is satisfied while probe
endpoints remain unauthenticated. The Orka smoke uses an already-expired turn
`deadline` so it can verify native Orka failure frames offline without calling a
live model provider.

The container parity smoke covers what the in-process parity suite cannot: the
dependency set each image actually installed, image entrypoint and env wiring,
and container logs. It builds `test/parity/agentkitfile-<runtime>.yaml` for each
runtime and runs `test/parity/fixture.py`, a scripted model plus a remote MCP
tool, on a private Docker network. Each image runs under the `openai`,
`foundry`, and `orka` protocols and must return a plain answer, complete a tool
round trip, and report `ModelAuthRejected` for a 401 that echoes the model key.
That key must not appear in any response or container log.

```sh
make build-agentkit build-serve build-serve-maf build-serve-langgraph TAG=test
scripts/parity-container-smoke.sh                 # all runtimes
scripts/parity-container-smoke.sh langgraph       # one runtime
```

Set `PLATFORM=linux/arm64` on ARM hosts and `BUILDER` to a docker-driver Buildx
builder, as for `make build-test-agent`.

## Harness v2 end-to-end checks

The composed v2 checks build the current AgentKit frontend and agent images,
layer Orka's production supervisor onto each immutable image, and exercise the
real ACP, provider, and MCP paths. The normal PR/push offline matrix covers
`pydantic-ai`, `microsoft-agent-framework`, and `langgraph` with deterministic
local fixtures and no external model credentials. The live matrix covers the
same three adapters against the digest-pinned AIKit Qwen3.5-2B image on CPU,
requiring real model, tool, continuation, and blocking-tool cancellation results.

```sh
scripts/orka-harness-v2-e2e.sh offline
scripts/orka-harness-v2-e2e.sh offline langgraph
scripts/orka-harness-v2-e2e.sh live
scripts/orka-harness-v2-e2e.sh live pydantic-ai
scripts/orka-harness-v2-e2e.sh live langgraph
```

Run these commands in a Linux shell on the Docker daemon's host, with a
daemon-backed Buildx builder and Go matching the pinned Orka module, currently Go 1.27 at commit
`55cb3d5232b4a9b697e72471e346c0a6493d4c21`. The runner fetches that exact revision;
no pre-existing Orka checkout is needed. `BUILDER` selects the builder, and
`PLATFORM` defaults to the Docker daemon's Linux amd64/arm64 architecture. Allow
network access for registry images and build dependencies. The model is bundled
in the AIKit image; live inference runs on a run-owned Docker network without
external API credentials.

Set `ARTIFACT_DIR` to keep safe JSON results with the adapter, scenario, source
and image digests, and failure diagnostics. Model startup and inference failures fail the check. The live CI lanes run on
fork and Dependabot pull requests too; they do not depend on repository secrets. See
[the v2 test guide](orka.md#test-the-composed-v2-runtime) for scenario assertions,
session retirement rules and cleanup behavior. The existing v1
container and OpenAI HTTP smoke jobs remain independent checks.

Validate the runner syntax and focused ACP input/output behavior locally:

```sh
bash -n scripts/orka-harness-v2-e2e.sh
shellcheck -x scripts/orka-harness-v2-e2e.sh
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 .github/workflows/*.yml
uv run --directory runtimes/common --extra dev pytest -q tests/test_acp_protocol.py tests/test_cli_protocol.py
```

## Live AIKit E2E

`scripts/live-aikit-agent-e2e.sh` runs real built agents for all three adapters
against the prebuilt
`ghcr.io/kaito-project/aikit/qwen3.5:2b` image, pinned by digest in
`scripts/aikit-e2e-common.sh`. No model API key or auth cache is required.

```sh
AIKIT_HOST_PORT=18089 \
AGENTKIT_LIVE_HOST_PORT=18086 \
TAG=e2e-script \
scripts/live-aikit-agent-e2e.sh

# Select one adapter, or omit the argument to run all three with one model server.
scripts/live-aikit-agent-e2e.sh pydantic-ai
scripts/live-aikit-agent-e2e.sh langgraph
```

When `PLATFORM` is unset, the script selects the Docker daemon's Linux amd64 or
arm64 architecture. Both the model and agent use a run-owned network.
Only the host-facing test ports are published on loopback.

The CPU model configuration is in `test/aikit-e2e/model.yaml`. It bounds context,
output, and CPU threads, disables reasoning, and uses greedy sampling while
preserving native tool templates. Both live entrypoints cap CPU use at four CPUs or the daemon's available count,
whichever is smaller, and warm the model before running timed agent turns. `AIKIT_IMAGE` can override the image for local testing;
it must serve the same `qwen-3.5-2b` model. CI uses the checked-in digest.

## Live task-success evals

`scripts/live-task-evals.sh` measures task success for `pydantic-ai`,
`microsoft-agent-framework` and `langgraph`. It reuses a digest-pinned AIKit
Qwen3.5-2B CPU model server across tasks. Inference is real, not scripted.
Stock, prices, contacts, reservations and warehouse notes come from controlled
MCP fixture tools, not external business services. Each case/trial gets fresh
agent and fixture containers so tool state and conversation history do not leak
between trials. Grading uses code, not an LLM judge. Fixture values vary
deterministically by case and trial; each adapter gets the same corresponding
inputs. Consistency means success across those variants, not a statistical
claim about identical-input repeatability.

Run in a Linux shell on the Docker daemon's host, with a daemon-backed Buildx
builder and network access for image pulls and build dependencies. If local
Docker is unavailable, sync the checkout to a Linux VM and run these commands
there over SSH. No external model API credentials are required.

```sh
scripts/live-task-evals.sh                         # all three adapters
scripts/live-task-evals.sh langgraph                # one adapter
scripts/live-task-evals.sh maf                      # alias for microsoft-agent-framework
EVAL_TRIALS=1 EVAL_CASES=stock,lookup-stock \
  ARTIFACT_DIR=artifacts/focused-evals \
  scripts/live-task-evals.sh pydantic-ai
```

The ten cases are:

| Case ID | Task |
|---|---|
| `no-tool` | Answer arithmetic without calling tools. |
| `stock` | Read stock for an exact SKU. |
| `price` | Quote the requested quantity. |
| `contact` | Find a contact's email. |
| `lookup-stock` | Identify a product before checking stock. |
| `lookup-reserve` | Identify a product, reserve stock and check the remainder. |
| `history` | Use the selected SKU from an earlier turn. |
| `recover-tool-error` | Retry after a controlled tool failure. |
| `untrusted-note` | Read a note without following its injected instructions. |
| `unknown-product` | Report missing data without inventing a product. |

The default run covers **10 cases × 3 trials × 3 adapters = 90 trials**.

| Setting | Default and purpose |
|---|---|
| `EVAL_TRIALS` | `3`, an integer from `1` through `10` per case and adapter. |
| `EVAL_CASES` | All ten cases. Set comma-separated case IDs for a focused run. |
| `ARTIFACT_DIR` | Persistent reports in `artifacts/live-task-evals/<unique run>/`. Override to select a directory. |
| `EVAL_CASE_TIMEOUT_SECONDS` | `120`, the timeout for one case/trial. |
| `EVAL_SUITE_TIMEOUT_SECONDS` | `3600`, the per-adapter suite budget, including trial startup but excluding image/model setup. |

`TAG`, `PLATFORM` and `BUILDER` work as in the existing live runner. With
`PLATFORM` unset, the runner selects the Docker daemon's Linux amd64 or arm64
architecture.

Per-adapter JSON reports include individual grades, counts, duration and
reproducibility metadata for the source, model, images and eval configuration.
Answer grades check expected facts. Tool-selection grades check required and
extraneous calls; argument grades check identifiers and quantities. Safety grades
check unauthorized actions and untrusted-note handling. Task success requires
correct answers, completed tool work, final fixture state and safety. Extra benign
reads or recovered argument mistakes remain diagnostic grades, so valid alternative
plans do not automatically fail the task.

Quality failures are informational and exit zero. The suite still completes
independent cases and trials. An individual task that exceeds its time budget receives a failing grade.
Build, startup, model transport, exhausted suite budgets and other
infrastructure failures or incomplete runs exit nonzero. Partial reports are
not complete results. Each measured trial requires at least one completed real
inference response. Pending model forwards are settled before final accounting;
a stalled first inference cannot produce a complete, green baseline. Cleanup
failures also return nonzero and appear in `run.json`. Check completion status and planned versus completed
counts before comparing grades; a green job does not mean every task passed.

`.github/workflows/live-task-evals.yml` runs only manually or nightly, not on
pushes or pull requests. Its three-adapter matrix defaults to three trials per
case, and a failed lane does not cancel the others. In the repository's Actions
tab, select **Live task-success evals**, then **Run workflow** to choose a branch
and trial count. Download `live-task-evals-<adapter>` from the run summary's
Artifacts section. CI always attempts the upload, including partial reports
from failed runs, and retains artifacts for seven days. Setup failures may leave
no report to upload. Existing conformance gates remain separate and unchanged.

Run the evaluator's stdlib unit tests without Docker or inference:

```sh
python3 -m unittest discover -s test/evals -p 'test_*.py'
bash -n scripts/live-task-evals.sh
shellcheck -x scripts/live-task-evals.sh
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 .github/workflows/*.yml
```

The common Python CI lane runs these unit tests; the shell/actionlint checks run
in the existing Go lane.
