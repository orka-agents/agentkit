#!/usr/bin/env bash

# Toolbox e2e: build the kind: Toolbox fixtures with the local frontend image
# (`make build-agentkit TAG=...`) and check the multi-platform output, that the
# verify target passes and leaves the image unchanged, that a tool needing a
# newer glibc fails verify clearly, and that a missing download fails early.

set -Eeuo pipefail

log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

tag="${TAG:-test}"
builder="${BUILDER-desktop-linux}"
# The verify stage runs the tools, so it uses the builder's native platform.
native_platform="${PLATFORM:-linux/amd64}"

for command in docker python3; do
  command -v "$command" >/dev/null 2>&1 || die "missing required command: $command"
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

build() {
  local fixture="$1"
  shift
  local args=(buildx build)
  [[ -n "$builder" ]] && args+=(--builder "$builder")
  docker "${args[@]}" . -f "$fixture" \
    --build-arg BUILDKIT_SYNTAX="agentkit:$tag" \
    --provenance=false --sbom=false \
    "$@"
}

log "build test/toolbox/toolbox.yaml for linux/amd64 and linux/arm64"
build test/toolbox/toolbox.yaml --platform linux/amd64,linux/arm64 \
  -o "type=oci,dest=$work/multi.tar"
python3 test/toolbox-image-smoke.py "$work/multi.tar" agentkit-e2e linux/amd64 linux/arm64

log "build the default and verify targets for $native_platform"
build test/toolbox/toolbox.yaml --platform "$native_platform" -o "type=oci,dest=$work/default.tar"
build test/toolbox/toolbox.yaml --platform "$native_platform" --target verify \
  -o "type=oci,dest=$work/verify.tar"
default_layers="$(python3 test/toolbox-image-smoke.py "$work/default.tar" agentkit-e2e "$native_platform")"
verify_layers="$(python3 test/toolbox-image-smoke.py "$work/verify.tar" agentkit-e2e "$native_platform")"
[[ "$default_layers" == "$verify_layers" ]] || die "verify target changed the image: $default_layers != $verify_layers"

log "verify must fail for a tool that needs a newer glibc than the runtime image"
if build test/toolbox/toolbox-glibc-mismatch.yaml --platform "$native_platform" --target verify \
  -o "type=oci,dest=$work/glibc.tar" >"$work/glibc.log" 2>&1; then
  cat "$work/glibc.log" >&2
  die "verify passed for a tool that needs GLIBC_2.38 on a glibc 2.36 runtime"
fi
for want in "GLIBC_2.38' not found" 'hint: the tool needs a newer glibc' "\"trixie-awk 'BEGIN { print \\\"awk ok\\\" }'\" failed with exit status 1"; do
  grep -qF "$want" "$work/glibc.log" || { cat "$work/glibc.log" >&2; die "verify failure output lacks: $want"; }
done

log "a platform without a download must fail before anything is fetched"
if build test/toolbox/toolbox.yaml --platform linux/riscv64 -o "type=oci,dest=$work/riscv.tar" >"$work/riscv.log" 2>&1; then
  die "built a toolbox for a platform without downloads"
fi
grep -qF 'tool "yq" has no download for linux/riscv64' "$work/riscv.log" || { cat "$work/riscv.log" >&2; die "missing-platform error unclear"; }

log "toolbox e2e passed"
