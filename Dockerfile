# The AgentKit frontend image (BuildKit gateway).
#
# Referenced as the build syntax of an agentkitfile.yaml:
#     #syntax=ghcr.io/orka-agents/agentkit/agentkit:latest
# BuildKit pulls this image and runs /bin/agentkit as the gateway frontend, which
# turns a `kind: Agent` file into a standard OCI image of a tool-using agent.
#
# Mirrors AIKit's root Dockerfile: a static, CGO-free build of ./cmd/frontend
# from a golang builder, shipped FROM scratch with just CA certs + the binary.
FROM --platform=$BUILDPLATFORM golang:1.26-bookworm@sha256:5f68ec6805843bd3981a951ffada82a26a0bd2631045c8f7dba483fa868f5ec5 AS builder

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT=""
ARG LDFLAGS

COPY . /go/src/github.com/orka-agents/agentkit
WORKDIR /go/src/github.com/orka-agents/agentkit
RUN CGO_ENABLED=0 \
    GOOS=${TARGETOS} \
    GOARCH=${TARGETARCH} \
    GOARM=${TARGETVARIANT} \
    go build -o /agentkit -ldflags "${LDFLAGS} -w -s -extldflags '-static'" ./cmd/frontend
# kind: Toolbox builds run this helper inside the build. Ship it for every
# toolbox architecture, whatever this image's own, so each exec runs a helper
# built for its platform, natively or under emulation.
RUN for arch in amd64 arm64; do \
      CGO_ENABLED=0 GOOS=linux GOARCH="$arch" \
      go build -o "/toolbox/agentkit-toolbox-linux-$arch" -ldflags "${LDFLAGS} -w -s -extldflags '-static'" ./cmd/toolbox || exit 1; \
    done

FROM scratch
LABEL org.opencontainers.image.source="https://github.com/orka-agents/agentkit"
COPY --from=builder /etc/ssl/certs /etc/ssl/certs
COPY --from=builder /agentkit /bin/agentkit
COPY --from=builder /toolbox/ /usr/libexec/agentkit/
ENTRYPOINT ["/bin/agentkit"]
