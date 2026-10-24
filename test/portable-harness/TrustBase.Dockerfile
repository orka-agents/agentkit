# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY ca.crt /tmp/agentkit-portable-harness-ca.crt

USER 0:0

RUN if command -v apt-get >/dev/null 2>&1; then \
        apt-get update \
        && apt-get install -y --no-install-recommends ca-certificates \
        && cp /tmp/agentkit-portable-harness-ca.crt \
            /usr/local/share/ca-certificates/agentkit-portable-harness.crt \
        && update-ca-certificates \
        && rm -rf /var/lib/apt/lists/*; \
    elif command -v dnf >/dev/null 2>&1; then \
        dnf install -y ca-certificates \
        && cp /tmp/agentkit-portable-harness-ca.crt \
            /etc/pki/ca-trust/source/anchors/agentkit-portable-harness.crt \
        && update-ca-trust \
        && dnf clean all; \
    elif command -v microdnf >/dev/null 2>&1; then \
        microdnf install -y ca-certificates \
        && cp /tmp/agentkit-portable-harness-ca.crt \
            /etc/pki/ca-trust/source/anchors/agentkit-portable-harness.crt \
        && update-ca-trust \
        && microdnf clean all; \
    else \
        echo "unsupported base image package manager" >&2; \
        exit 1; \
    fi \
    && rm /tmp/agentkit-portable-harness-ca.crt
