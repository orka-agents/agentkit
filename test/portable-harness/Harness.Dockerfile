# syntax=docker/dockerfile:1.7

ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.24@sha256:99ea34acedc870ba4ad11a1f540a1c04267c9f30aadc465a94406f52dfda2c36
ARG NODE_IMAGE=node:22-bookworm-slim
ARG BUSYBOX_IMAGE=busybox:1.37.0-musl
ARG PACKAGE_IMAGE=ghcr.io/orka-agents/agentkit/serve-pydantic-ai@sha256:02a0d063e117664ebb87254761d3f940c9a8135a1cb4a61b2dd80c27a5de9c6f

FROM ${UV_IMAGE} AS uv
FROM ${NODE_IMAGE} AS node
FROM ${BUSYBOX_IMAGE} AS busybox
FROM ${PACKAGE_IMAGE} AS packages

FROM debian:bookworm-slim AS builder

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_SYSTEM_CERTS=true

RUN uv python install 3.12 \
    && python_path="$(uv python find --no-project 3.12)" \
    && python_root="$(dirname "$(dirname "$python_path")")" \
    && mkdir -p /agentkit-harness/runtime \
    && cp -aL "$python_root" /agentkit-harness/runtime/python

COPY --from=packages /opt/agentkit/lib/python3.12/site-packages /agentkit-harness/packages
COPY runtimes/common/agentkit_serve_common /agentkit-harness/packages/agentkit_serve_common
COPY runtimes/pydantic-ai/agentkit_serve /agentkit-harness/packages/agentkit_serve

RUN mkdir -p /agentkit-harness/bin /agentkit-harness/node/lib \
        /agentkit-harness/lib \
    && cp /usr/local/bin/uv /usr/local/bin/uvx /agentkit-harness/bin/ \
    && libgcc_path="$(find /lib /usr/lib -name libgcc_s.so.1 -print -quit)" \
    && test -n "$libgcc_path" \
    && cp "$libgcc_path" /agentkit-harness/lib/

COPY --from=node /usr/local/bin/node /agentkit-harness/bin/node
COPY --from=node /usr/local/lib/node_modules /agentkit-harness/node/lib/node_modules
COPY --from=node /usr/lib/*-linux-gnu/libstdc++.so.6 /agentkit-harness/lib/
COPY --from=busybox /bin/busybox /agentkit-harness/bin/busybox
COPY test/portable-harness/agentkit-serve.py /agentkit-harness/bin/agentkit-serve
COPY test/portable-harness/trust-smoke.py /agentkit-harness/bin/agentkit-trust-smoke
COPY test/portable-harness/npm.js /agentkit-harness/bin/npm
COPY test/portable-harness/npx.js /agentkit-harness/bin/npx
COPY test/portable-harness/tool-fixtures/node /agentkit-harness/tool-fixtures/node
COPY test/portable-harness/build_python_tool_wheel.py /tmp/build_python_tool_wheel.py

RUN chmod 0755 \
        /agentkit-harness/bin/agentkit-serve \
        /agentkit-harness/bin/agentkit-trust-smoke \
        /agentkit-harness/bin/busybox \
        /agentkit-harness/bin/node \
        /agentkit-harness/bin/npm \
        /agentkit-harness/bin/npx \
        /agentkit-harness/bin/uv \
        /agentkit-harness/bin/uvx \
        /agentkit-harness/tool-fixtures/node/server.js \
    && ln -s busybox /agentkit-harness/bin/sh \
    && /agentkit-harness/runtime/python/bin/python3 /tmp/build_python_tool_wheel.py \
    && PYTHONPATH=/agentkit-harness/packages \
        /agentkit-harness/runtime/python/bin/python3 -c \
        "import agentkit_serve, agentkit_serve_common, pydantic_ai"

FROM scratch

COPY --from=builder /agentkit-harness /agentkit-harness

LABEL io.github.orka-agents.agentkit.harness.version="1"
LABEL io.github.orka-agents.agentkit.harness.runtime="pydantic-ai"
LABEL io.github.orka-agents.agentkit.harness.artifact="/agentkit-harness"
