# Agentkitfile reference

An Agentkitfile is the user-authored BuildKit frontend input for AgentKit. It is
a YAML file with `kind: Agent`, usually named `agentkitfile.yaml` and referenced
with Docker's `#syntax=` directive.

```yaml
#syntax=ghcr.io/orka-agents/agentkit/agentkit:latest
apiVersion: v1alpha1
kind: Agent
metadata:
  name: url-summarizer
runtime: pydantic-ai
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
instructions: |
  Summarize any URL in three bullet points.
tools:
  - name: fetch
    command: ["uvx", "mcp-server-fetch"]
    env: ["FETCH_TIMEOUT"]
expose:
  openai: true
  port: 8080
```

## Load and validation behavior

The Go loader in `pkg/agentkit/config` first probes `apiVersion` and `kind`, then
strictly decodes the full schema. This means:

- `kind` is required and must be `Agent`.
- `apiVersion` is required and must be `v1alpha1`.
- Unknown or misspelled fields fail the build with line/column context.
- Validation reports all detected schema problems together where possible.

## Fields

### `metadata`

```yaml
metadata:
  name: url-summarizer
  labels:
    com.example.team: platform
```

- `name` is required and becomes both AgentKit metadata and the OCI image title.
- `labels` is optional. User-supplied labels are copied into the final image
  config alongside AgentKit labels.

### `debug`

```yaml
debug: true
```

`debug` is accepted by the schema for build-time diagnostics, but the current
frontend does not branch on it when producing the image.

### `runtime`

`runtime` selects the runtime adapter image used as the final image base.
Omitting it selects `pydantic-ai`.

| Value | Meaning |
|---|---|
| `pydantic-ai` | Default adapter backed by pydantic-ai. |
| `microsoft-agent-framework` | Adapter backed by Microsoft Agent Framework. |
| `maf` | Alias for `microsoft-agent-framework`. |
| `langgraph` | Adapter backed by LangChain/LangGraph. |

The build arg `--build-arg runtime=<name>` overrides the file value. The build
arg `--build-arg adapter=<image-ref>` overrides the selected runtime's default
adapter image ref, which is how the local dev loop tests unpublished adapters.

### `model`

```yaml
model:
  provider: openai-compatible
  baseURL: https://api.openai.com/v1
  name: gpt-4o-mini
  apiKeyEnv: OPENAI_API_KEY
```

- `provider` must be `openai-compatible`.
- `baseURL` is the OpenAI-compatible `/v1` endpoint the adapter uses.
- `name` is the model name sent to that endpoint.
- `apiKeyEnv` is optional. When present, it is the name of an env var read at
  container startup. Do not put secret values in YAML.

If `apiKeyEnv` is omitted, the runtime supplies a non-secret placeholder key for
OpenAI-compatible endpoints that do not require authentication.

#### OpenAI-compatible endpoints

AgentKit does not require the model endpoint to be OpenAI-hosted. Any
OpenAI-compatible `/v1` service is valid: OpenAI, another hosted provider, a
local gateway, an in-cluster service, or a model image from
[AIKit](https://github.com/kaito-project/aikit). AIKit is only one example.

For an AIKit example, run any AIKit image that serves the OpenAI-compatible API
on a Docker network. This can be a prebuilt CPU/GPU image or a custom model image
you create with AIKit:

```sh
docker network create agentkit-local 2>/dev/null || true

docker run -d --rm \
  --name aikit-llama \
  --network agentkit-local \
  ghcr.io/kaito-project/aikit/llama3.2:1b
```

Configure the agent to use the `/v1` URL and model name served by that endpoint:

```yaml
model:
  provider: openai-compatible
  baseURL: http://aikit-llama:8080/v1
  name: llama-3.2-1b-instruct
```

Omit `apiKeyEnv` for no-auth endpoints. Run the generated AgentKit container on
the same Docker network, or replace `baseURL` with another address reachable
from the AgentKit container, such as a Kubernetes service DNS name or
`http://host.docker.internal:<port>/v1` on Docker Desktop.

Other endpoints follow the same pattern: replace `baseURL` and `model.name` with
the values for the service you are using. For another prebuilt or custom AIKit
image, also replace the image reference and container name.

### `instructions`

Inline form:

```yaml
instructions: |
  You are concise.
```

File-backed form:

```yaml
instructions:
  file: ./prompt.md
```

Exactly one source is allowed. File paths are read from the BuildKit context
during the build; the final `/agent/agent.yaml` contains the resolved prompt
text, not the file reference.

### `tools`

Stdio MCP tools use `command`:

```yaml
tools:
  - name: fetch
    command: ["uvx", "mcp-server-fetch"]
    env: ["FETCH_TIMEOUT"]
```

Remote MCP tools use Streamable HTTP plus env-derived URL/auth material:

```yaml
tools:
  - name: toolbox
    type: mcp
    transport: streamable-http
    urlEnv: TOOLBOX_ENDPOINT
    headers:
      - name: Foundry-Features
        value: Toolboxes=V1Preview
    auth:
      type: workload-identity-token
      audience: https://ai.azure.com/.default
```

Tool rules:

- `name` is required and must be unique within the agent.
- Stdio tools set `command`; remote tools set `type: mcp`,
  `transport: streamable-http`, and `urlEnv`.
- Stdio `env` lists env var names that may be passed to the subprocess.
- Remote headers may use static non-secret values or `valueEnv`; static
  credential headers are rejected.
- Remote auth supports `bearer`/`tokenEnv` and, where the selected runtime
  declares support, `workload-identity-token`/`audience`.

Runtime adapters pass only declared, present env vars into tool subprocesses. A
tool env value that references `${OTHER_VAR}` must also list `OTHER_VAR` in the
same tool's allowlist, preventing accidental secret bleed from the parent process.


### `context`

Context providers describe external knowledge, skills, or memory without naming a
cloud provider in AgentKit core. Runtime support is capability-gated.

```yaml
context:
  providers:
    - name: knowledge
      type: search
      endpointEnv: SEARCH_ENDPOINT
      indexEnv: SEARCH_INDEX
      auth:
        type: workload-identity-token
        audience: https://search.azure.com/.default
    - name: support-style
      type: skills
      source: filesystem
      path: /agent/skills
    - name: user-memory
      type: memory
      endpointEnv: MEMORY_ENDPOINT
      storeNameEnv: MEMORY_STORE_NAME
      auth:
        type: workload-identity-token
        audience: https://ai.azure.com/.default
```

- `search` requires `endpointEnv` and `indexEnv`.
- `skills` uses `source: filesystem` with an absolute `path` under
  `/agent/skills`, or `source: mcp` with a remote streamable-http MCP `toolRef`.
- `memory` requires `endpointEnv` and `storeNameEnv`.
- Env-suffixed fields name environment variables; they never contain endpoint
  secrets or token values directly.

AgentKit does not copy arbitrary local skill directories into the image. If you
use filesystem skills, stage them under `/agent/skills` in the runtime/deployment
image or prefer MCP-backed skills. Memory providers require an explicit
`AGENTKIT_MEMORY_SCOPE` runtime env var; choose a per-user/session-safe scope.

Orka harness v2 and the hosted brokered model loop support the narrower
[bundled instruction-only skills](instruction-skills.md) mode. It exposes
`load_skill` over a startup snapshot of `SKILL.md` documents. Resources, scripts,
remote skill sources, search providers, and memory providers are not available
through that mode.

### `expose`

```yaml
expose:
  openai: true
  port: 8080
```

- `openai` must be `true`.
- `port` is optional in the authored file. When omitted, the effective Agent uses
  port `8080` and the final image exposes that port.

## Build inputs and targets

AgentKit supports the normal BuildKit frontend options used by `docker buildx`:

- `-f <file>` / `filename` selects the Agentkitfile. The default is
  `agentkitfile.yaml`.
- `--platform` can contain one or more target platforms. The frontend builds each
  platform in parallel and returns a multi-platform result when requested.
- `--target` routes to `<runtime>/image`. Empty target and a bare runtime target
  both mean the runtime's image output.
- cache import options are passed through to the BuildKit solve.

The only output kind registered by the router is an OCI image.

### Experimental split-image composition

The frontend accepts `baseImage` and `harnessImage` as an experimental paired
alternative to the combined `adapter` image:

```sh
docker buildx build . -f test/agentkitfile-hello.yaml \
  --build-arg baseImage=ubuntu:24.04 \
  --build-arg harnessImage=agentkit-harness-pydantic-ai:test
```

Both inputs are required together and cannot be combined with `adapter`. AgentKit
resolves both images for the target platform, pins the resolved digests, validates
the harness labels, copies only the declared harness artifact onto the base, and
then generates `/agent/agent.yaml`. AgentKit remains authoritative over the final
entrypoint, command, exposed ports, runtime labels, and generated configuration.

The harness image must declare:

- `io.github.orka-agents.agentkit.harness.version=1`
- `io.github.orka-agents.agentkit.harness.runtime=<Agentkitfile runtime>`
- `io.github.orka-agents.agentkit.harness.artifact=<absolute artifact path>`

The base image user must be root or numeric. Root and an unset user map to
`1000:1000`; a positive numeric UID or `UID:GID` is preserved. Only a narrow
allowlist of base environment values is inherited. Harness Python, Node, package
tools, shell, and native dependencies must be relocatable within the declared
artifact. This interface remains a spike and does not yet promise arbitrary base
image compatibility, especially for native-library differences.

## What gets baked

The built image contains:

- either the selected runtime adapter filesystem or the experimentally composed
  base and harness filesystems,
- `/agent/agent.yaml`, rendered from the effective Agent,
- an entrypoint of `/opt/agentkit/bin/agentkit-serve --config /agent/agent.yaml`,
- non-root user `1000:1000`,
- `AGENTKIT_BIND=127.0.0.1`, and
- AgentKit OCI labels for runtime, agent name, and ABI version.

Secret values are never written into the image.
