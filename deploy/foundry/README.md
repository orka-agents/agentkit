# Foundry deployment helpers

These helpers keep provider-specific resource wiring outside the core AgentKit
agent schema. They emit generic environment variables consumed by AgentKit
features such as remote MCP tools and context providers.

## Hosted-agent profile shape

```yaml
target: foundry-hosted-agent
agent: ./agentkitfile.yaml
image: docker.io/acme/hr-agent-foundry-wrapper:v1
foundryWrapper: true
protocols:
  - invocations
  - responses
env:
  TOOLBOX_ENDPOINT: ${FOUNDRY_PROJECT_ENDPOINT}/toolboxes/${TOOLBOX_NAME}/mcp?api-version=v1
```

The reusable Python Foundry adapter in `agentkit_serve_common.foundry` is selected
with `AGENTKIT_PROTOCOL=foundry` (or `agentkit-serve --protocol foundry`) and
exposes `/readiness`, `/invocations`, and a minimal non-streaming `/responses`
endpoint without adding Foundry-specific keys to `/agent/agent.yaml`.

## Hosted-agent rendering

Use `agentkit render --target foundry-hosted-agent` to map a provider-neutral
Foundry deployment profile to the `agent.yaml` shape consumed by azd hosted
agents. The `image` must already serve Foundry's `/readiness`, `/invocations`,
and `/responses` contract (for example by wrapping an AgentKit image with
`test/foundry-hosted-agent/foundry_live.py`):

```sh
agentkit render --target foundry-hosted-agent deploy/foundry/agentkit.foundry.yaml.example -o agent.yaml
```

From a source checkout, `go run ./cmd/frontend render ...` runs the same
command.

The renderer intentionally emits Foundry's sample-compatible
`environment_variables` list form:

```yaml
environment_variables:
  - name: TOOLBOX_ENDPOINT
    value: ${TOOLBOX_ENDPOINT}
```

Do not hand-author the camelCase `environmentVariables` map form for hosted
AgentKit validation: it can validate locally while failing to inject variables
into the hosted container.

The `foundry-hosted-agent` target requires `foundryWrapper: true` (or `imageKind:
foundry-wrapper`) so a plain AgentKit `/v1` image is not accidentally advertised
as a Foundry-hosted protocol image.

## Resource setup

- `toolbox/setup.sh` references a Foundry Toolbox and writes `TOOLBOX_ENDPOINT`.
- `search/setup.sh` records Azure Search context-provider env names.
- `memory/setup.sh` records external memory context-provider env names.
- `rbac/assign-agent-identity.sh` grants the newly-created hosted agent identity
  the project/account role required for workload-identity model/tool calls (defaults to `Foundry User`).
- `scripts/invoke_responses.sh` sends the minimal portable hosted Responses
  payload (`{"input":"..."}`), avoiding gateway-specific optional fields.
- `scripts/foundry_brokered_conformance.sh` runs the Phase A0 brokered
  Responses function-call/continuation loop against a deployed `/responses`
  endpoint and writes a sanitized transcript directory for review evidence.
- `scripts/local_brokered_conformance_container.sh` builds the conformance
  container locally, runs it, and exercises the same transcript helper before an
  image is pushed to Foundry.
- `doctor.sh` checks the expected Foundry/project env and local CLI prerequisites.
  Use `doctor.sh --brokered-conformance` before running the brokered transcript
  helper to verify `AGENT_RESPONSES_ENDPOINT` and auth prerequisites.

These scripts intentionally produce local `output.env` files that should not be
committed with live subscription or endpoint values.


## Brokered conformance smoke


Before pushing the image, validate the packaged container locally:

```sh
deploy/foundry/scripts/local_brokered_conformance_container.sh \
  --platform linux/amd64 \
  --tag agentkit-foundry-brokered-conformance:amd64-test \
  --port 18090 \
  --transcript-dir ./foundry-brokered-local-transcript
```

After deploying an image that serves
`agentkit_serve_common.foundry_conformance.create_foundry_conformance_app()`, run:

```sh
export AGENT_RESPONSES_ENDPOINT="https://<hosted-agent>/responses"
# Optional: export AZURE_SUBSCRIPTION_ID="<subscription>" to select an account.
deploy/foundry/doctor.sh --brokered-conformance
deploy/foundry/scripts/foundry_brokered_conformance.sh conformance_read ./foundry-brokered-transcript
```

To validate the production AgentKit brokered path locally instead of the
standalone SDK conformance app, see `test/foundry-brokered-agentkit/`. That
fixture uses `agentkit-foundry-brokered` and expects generated call IDs, so run
the transcript helper with `AGENTKIT_EXPECTED_CALL_ID=auto` and
`AGENTKIT_EXPECTED_CALL_ID_PREFIX=call_`.

Alternatively set `AGENT_RESPONSES_BEARER_TOKEN` to use a pre-acquired token
instead of invoking `az account get-access-token`. If `AZURE_SUBSCRIPTION_ID` is
omitted, the helper uses the current `az` account. The script stores request and
response JSON files plus `summary.json`; do not include bearer tokens in the
transcript. Re-run
`python3 deploy/foundry/scripts/verify_brokered_transcript.py <transcript-dir> --expected-final-text '<known expected assistant result>'`
to verify archived transcript evidence later.
