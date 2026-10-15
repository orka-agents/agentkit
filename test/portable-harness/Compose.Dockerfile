# syntax=docker/dockerfile:1.7

ARG HARNESS_IMAGE
ARG BASE_IMAGE
ARG HARNESS_LIBRARY_PATH
ARG RUNTIME_UID=1000
ARG RUNTIME_GID=1000

FROM ${HARNESS_IMAGE} AS harness
FROM ${BASE_IMAGE}

ARG RUNTIME_UID
ARG RUNTIME_GID
ARG HARNESS_LIBRARY_PATH

COPY --from=harness --chown=${RUNTIME_UID}:${RUNTIME_GID} /agentkit-harness/ /opt/agentkit/
COPY --chown=${RUNTIME_UID}:${RUNTIME_GID} test/portable-harness/agent.yaml /agent/agent.yaml

ENV AGENTKIT_BIND=127.0.0.1 \
    HOME=/tmp \
    LD_LIBRARY_PATH=${HARNESS_LIBRARY_PATH} \
    npm_config_script_shell=/opt/agentkit/bin/sh \
    PATH=/opt/agentkit/bin:$PATH \
    PYTHONPATH=/opt/agentkit/packages \
    SHELL=/opt/agentkit/bin/sh \
    PYTHONUNBUFFERED=1

EXPOSE 8080

USER ${RUNTIME_UID}:${RUNTIME_GID}

ENTRYPOINT ["/opt/agentkit/bin/agentkit-serve"]
CMD ["--config", "/agent/agent.yaml"]
