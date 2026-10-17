# Agentkitfile reference

An Agentkitfile is the user-authored BuildKit frontend input for AgentKit. It is
a YAML file with `kind: Agent`, usually named `agentkitfile.yaml` and referenced
with Docker's `#syntax=` directive. The same frontend also builds Orka toolbox
images from `kind: Toolbox`; see [`kind: Toolbox`](#kind-toolbox).

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

- `kind` is required and must be `Agent` or `Toolbox`.
- `apiVersion` is required and must be `v1alpha1`.
- Unknown or misspelled fields fail the build with line/column context.
- Validation reports all detected schema problems together where possible.

## Extension fields (`x-`)

A field whose name starts with `x-` or `X-`, at any level of the file, is
accepted and ignored — the same convention Docker Compose and OpenAPI use.
Two common uses:

- **Reusing values with YAML anchors.** No field exists to hold a value you
  only want to reuse, so define it inside an `x-` field with `&name` and
  reference it elsewhere with `*name`.
- **Notes for other tooling.** An owner, a cost center, or settings another
  tool reads from the same file, without loosening the strict-decode typo
  check for every other field.

```yaml
x-owner: platform-team
x-proxy-env: &proxy-env [HTTP_PROXY, HTTPS_PROXY, NO_PROXY]
tools:
  - name: fetch
    command: ["mcp-server-fetch"]
    env: *proxy-env
  - name: search
    command: ["mcp-server-search"]
    env: *proxy-env
```

`x-` fields are never copied anywhere: not into `/agent/agent.yaml`, not into
image labels. A key starting with `x-` inside a free-form map that already
accepts any key, such as `metadata.labels`, is unaffected — it is kept as an
ordinary key, since only real Agentkitfile fields get the extension-prefix
exception. Every other unknown or misspelled field still fails the build.

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
  both mean the runtime's image output. A `kind: Toolbox` file accepts only the
  empty target and `verify`.
- cache import options are passed through to the BuildKit solve.

The only output kind registered by the router is an OCI image.

## What gets baked

The built image contains:

- the selected runtime adapter filesystem,
- `/agent/agent.yaml`, rendered from the effective Agent,
- an entrypoint of `/opt/agentkit/bin/agentkit-serve --config /agent/agent.yaml`,
- non-root user `1000:1000`,
- `AGENTKIT_BIND=127.0.0.1`, and
- AgentKit OCI labels for runtime, agent name, and ABI version.

Secret values are never written into the image.

## `kind: Toolbox`

A Toolbox is a `FROM scratch` image of command-line tools for
[Orka toolboxes](https://github.com/orka-agents/orka/blob/main/website/docs/guides/toolboxes.md):
Orka mounts the image's folder read-only into the pods of its built-in coding
agents (Codex, Claude, Copilot, and OpenCode) and adds it to their `PATH`.
AgentKit downloads each tool for each platform, checks every SHA-256, takes
exactly one file per tool, and writes the `ai.orka.toolbox` label for you.

```yaml
#syntax=ghcr.io/orka-agents/agentkit/agentkit:latest
apiVersion: v1alpha1
kind: Toolbox
metadata:
  name: yq-helm                # installs under /opt/yq-helm
  labels:
    com.example.team: platform
spec:
  tools:
    - name: yq
      description: YAML processor
      source:
        download:
          linux/amd64: {url: https://github.com/mikefarah/yq/releases/download/v4.54.1/yq_linux_amd64, sha256: 8e34fc298390875de416e6a4afcb8cabeceb25d9aa8506c1a2f9353cf702ea5f}
          linux/arm64: {url: https://github.com/mikefarah/yq/releases/download/v4.54.1/yq_linux_arm64, sha256: 189088da0c6429ec5178dfaab1a114805f6cab0b61b165ab236efedf1d57a71b}
    - name: helm
      description: Kubernetes package manager
      source:
        download:
          linux/amd64: {url: https://get.helm.sh/helm-v3.19.0-linux-amd64.tar.gz, sha256: a7f81ce08007091b86d8bd696eb4d86b8d0f2e1b9f6c714be62f82f96a594496, extract: linux-amd64/helm}
          linux/arm64: {url: https://get.helm.sh/helm-v3.19.0-linux-arm64.tar.gz, sha256: 440cf7add0aee27ebc93fada965523c1dc2e0ab340d4348da2215737fc0d76ad, extract: linux-arm64/helm}
    - name: kubectl
      source:
        image: registry.k8s.io/kubectl@sha256:<digest>   # a digest, never a tag
        path: /bin/kubectl
  verify:                      # optional; runs only with --target verify
    runtimeImage: ghcr.io/orka-agents/orka/acp-codex-runtime@sha256:<digest>
    commands:
      - yq --version
      - helm version --short
      - kubectl version --client
```

```sh
# Build and push the toolbox for both architectures.
docker buildx build -f toolbox.yaml --platform linux/amd64,linux/arm64 \
  -t registry.example.com/tools/yq-helm:1 --push .

# Run the verify commands first; this outputs the same image.
docker buildx build -f toolbox.yaml --target verify --platform linux/amd64 .
```

Then pin the pushed digest in the Orka Agent:

```yaml
runtime:
  type: codex
  toolboxes:
    - image: registry.example.com/tools/yq-helm@sha256:<digest>
      mountPath: /opt/yq-helm
      pathEntries: [bin]
```

### Toolbox fields

- `metadata.name` is required: 1-63 lowercase letters, digits, or `-`, starting
  and ending with a letter or digit. The toolbox installs under
  `/opt/<metadata.name>`, so two toolboxes with different names never overlap
  and can be attached to the same Agent. Names Orka's runtime images already use
  under `/opt` (`claude`, `claude-agent-acp`, `codex`, `codex-acp`, `copilot`,
  `opencode`, `ripgrep`) are rejected.
- `metadata.labels` is optional and copied into the image config. The
  `ai.orka.*`, `ai.agentkit.*`, and AgentKit label namespaces and
  `org.opencontainers.image.title` are reserved.
- `spec.tools` lists at least one tool. Each tool installs as
  `/opt/<metadata.name>/bin/<name>`:
  - `name` follows the same rules as `metadata.name` and must be unique.
  - `description` is optional and recorded in the inventory.
  - `source` sets exactly one of `download` or `image`.
- `source.download` maps a platform, `linux/amd64` or `linux/arm64`, to a
  download:
  - `url` is an absolute `https` URL without credentials.
  - `sha256` is the download's checksum, 64 lowercase hex characters. BuildKit
    verifies it before anything uses the file.
  - `extract` is optional: the path of one regular file inside a `tar`,
    `tar.gz`, `tar.bz2`, or `tar.zst` archive. It must be a clean relative
    path. Without it, the download itself is the tool.

  Platforms are matched after normalization (`linux/aarch64` is
  `linux/arm64`). Toolboxes are built for `linux/amd64` and `linux/arm64`, the
  platforms Orka runs its built-in agents on. Every platform you build must have
  a download for every download tool, or the build fails before fetching
  anything.
- `source.image` is an image reference pinned by digest, and `source.path` is
  the absolute path of the tool inside it. The image is pulled for each target
  platform. Symlinks in `path` resolve inside the image only, and the result
  must be one regular file.
- `spec.verify` is optional. `runtimeImage` must be pinned by digest, and
  `commands` lists at least one shell command.

### Safety checks

AgentKit installs each tool with a small static `agentkit-toolbox` helper that
ships in its frontend image, running with no network and a read-only root. It
takes exactly one regular file per tool and fails the build when:

- an archive entry has an absolute path or a `..` component;
- the `extract` member is missing, appears more than once, or is a symlink, hard
  link, device, folder, or anything else that is not a regular file;
- an archive is larger than 1 GiB, expands to more than 2 GiB, or has more than
  100,000 entries;
- an archive contains a sparse file, or one entry's headers (PAX records, long
  names, sparse maps) exceed 4 MiB;
- the tool is larger than 512 MiB (Orka's per-file limit);
- the tool is neither an ELF binary nor a `#!` script (usually an archive
  downloaded without `extract`), or is an ELF binary for another CPU
  architecture than the platform being built.

`zip` and `xz` archives are not supported. Go's zip reader loads every
central-directory entry before the entry limit could be checked, so zip support
would need its own bounded parser.

### Output image

Each platform's image has one layer and no entrypoint, command, user, or
environment:

```text
/opt/<name>/                 0755 root:root
/opt/<name>/bin/             0755 root:root
/opt/<name>/bin/<tool>       0555 root:root, one per tool
/opt/<name>/.toolbox.json    0444 root:root
```

AgentKit generates two labels:

- `ai.orka.toolbox={"version":1,"mountPath":"/opt/<name>","pathEntries":["bin"]}`
- `org.opencontainers.image.title=<name>`

`.toolbox.json` lists, for that platform, each tool's name, description, path,
source, and the SHA-256 of the installed file. Download URLs are recorded
without their query string or fragment, which can carry tokens or presigned
signatures:

```json
{
  "version": 1,
  "name": "yq-helm",
  "platform": "linux/arm64",
  "tools": [
    {
      "name": "helm",
      "description": "Kubernetes package manager",
      "path": "bin/helm",
      "sha256": "<sha256 of /opt/yq-helm/bin/helm>",
      "source": {
        "download": {
          "url": "https://get.helm.sh/helm-v3.19.0-linux-arm64.tar.gz",
          "sha256": "440cf7add0aee27ebc93fada965523c1dc2e0ab340d4348da2215737fc0d76ad",
          "extract": "linux-arm64/helm"
        }
      }
    }
  ]
}
```

### Verify stage

`--target verify` builds the toolbox, runs every `spec.verify.commands` entry
with `/bin/sh -c` inside `runtimeImage`, and then outputs exactly the image the
default target builds. The default target never runs it. The commands run the
way Orka runs an agent's commands:

- as UID/GID `65534`, a user that owns nothing in the image;
- with no network and a read-only root filesystem;
- with the toolbox mounted read-only at `/opt/<name>`;
- with `PATH=/usr/local/bin:/usr/bin:/bin:/opt/<name>/bin` and a writable
  `HOME=/tmp`;
- with a one-minute timeout per command.

The system folders come first on `PATH`, as in Orka, so a tool named like a
command the runtime image already has (for example `awk`) runs the runtime's
copy instead. Give such tools another name.

Every command runs, and the build fails if any failed, showing each failed
command and its output. When the output shows a missing `GLIBC_x.y` version or
shared library, the failure adds a hint: the most common problem is a tool
built for a newer glibc than the runtime image has.

Verify runs the tools, so a platform other than the builder's own needs QEMU
emulation, as any cross-platform `RUN` does.
