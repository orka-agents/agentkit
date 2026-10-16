# syntax=docker/dockerfile:1
#
# agentkit-serve ADAPTER image.
#
# This is the runtime adapter the AgentKit Go converter uses as the LLB BASE
# image: the converter pulls this image and merges the resolved /agent/agent.yaml
# layer on top. It is also runnable standalone (ENTRYPOINT + CMD below).
#
# It bundles BOTH a Python runtime (for agentkit-serve / pydantic-ai) AND Node.js
# + npx and uv/uvx, so stdio MCP servers distributed on npm (`npx -y ...`) or
# PyPI (`uvx ...`) can be spawned as tool subprocesses.
#
# No secret is ever baked in: agent.yaml carries env var NAMES only; values are
# injected at runtime via `docker run -e`.

FROM python:3.12-slim-bookworm

# --- system deps: Node.js + npx (for npm-distributed MCP servers) ------------
# ca-certificates for TLS to model endpoints; nodejs/npm provide `npx`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

# --- uv / uvx (for PyPI-distributed MCP servers via `uvx ...`) ---------------
# Copy the static uv + uvx binaries from the official image onto PATH.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# AgentKit layout constants (must match pkg/utils/const.go and pkg/agentkit/abi):
#   AgentKitRoot = /opt/agentkit ; ServeBinary = /opt/agentkit/bin/agentkit-serve
#   abi.Path = /agent/agent.yaml
ENV AGENTKIT_ROOT=/opt/agentkit \
    PATH=/opt/agentkit/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- install agentkit-serve into a dedicated venv at /opt/agentkit -----------
# The console script lands at /opt/agentkit/bin/agentkit-serve (== ServeBinary).
# Build context is the REPO ROOT (see Makefile `build-serve`), so paths are
# repo-root-relative. The shared framework-neutral core (agentkit_serve_common,
# brings fastapi/uvicorn/pydantic/pyyaml) is installed FIRST so the adapter's
# `agentkit-serve-common` dependency is already satisfied; then the adapter adds
# its framework dep (pydantic-ai).
WORKDIR /src
COPY runtimes/common/pyproject.toml runtimes/common/README.md ./common/
COPY runtimes/common/agentkit_serve_common ./common/agentkit_serve_common
COPY runtimes/pydantic-ai/pyproject.toml runtimes/pydantic-ai/README.md ./pydantic-ai/
COPY runtimes/pydantic-ai/agentkit_serve ./pydantic-ai/agentkit_serve

RUN python -m venv /opt/agentkit \
    && /opt/agentkit/bin/pip install --upgrade pip \
    && /opt/agentkit/bin/pip install ./common \
    && /opt/agentkit/bin/pip install ./pydantic-ai \
    && /opt/agentkit/bin/python -c 'from a2a.server.apps import A2AFastAPIApplication' \
    && { /opt/agentkit/bin/agentkit-serve --help >/dev/null 2>&1 || true; }

# --- non-root user + agent config mount point --------------------------------
# uid 1000 owns /agent so a writer/runtime can place agent.yaml there.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin agentkit \
    && mkdir -p /agent \
    && chown -R 1000:1000 /agent /opt/agentkit

# Loopback-only by default; binding 0.0.0.0 requires AGENTKIT_AUTH_TOKEN (§10).
ENV AGENTKIT_BIND=127.0.0.1
EXPOSE 8080

USER 1000

ENTRYPOINT ["/opt/agentkit/bin/agentkit-serve"]
CMD ["--config", "/agent/agent.yaml"]
