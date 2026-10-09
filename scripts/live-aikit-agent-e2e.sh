#!/usr/bin/env bash

set +x
set -Eeuo pipefail

log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }

usage='usage: live-aikit-agent-e2e.sh [pydantic-ai|microsoft-agent-framework|langgraph] (maf aliases microsoft-agent-framework)'
[[ "$#" -le 1 ]] || die "$usage"
adapters=(pydantic-ai microsoft-agent-framework langgraph)
if [[ "$#" -eq 1 ]]; then
  case "$1" in
    pydantic-ai|microsoft-agent-framework|langgraph) adapters=("$1") ;;
    maf) adapters=(microsoft-agent-framework) ;;
    *) die "invalid adapter: $1; $usage" ;;
  esac
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/aikit-e2e-common.sh
source "$repo_root/scripts/aikit-e2e-common.sh"
work_dir="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/agentkit-live-aikit.XXXXXX")"
run_id="${work_dir##*/}"
aikit_container_name="$run_id-aikit"
aikit_host_port="${AIKIT_HOST_PORT:-18089}"
agent_container_name=""
network_name="$run_id-network"
agent_host_port="${AGENTKIT_LIVE_HOST_PORT:-18080}"
agent_auth_token="${AGENTKIT_AUTH_TOKEN:-agentkit-live-ci-token}"
tag="${TAG:-ci-live}"
platform="${PLATFORM:-}"
builder="${BUILDER:-}"
containers=()
network_created=false

default_platform() {
  local arch
  arch="$(docker info --format '{{.Architecture}}' 2>/dev/null || uname -m)"
  case "$arch" in
    aarch64|arm64) printf 'linux/arm64' ;;
    x86_64|amd64) printf 'linux/amd64' ;;
    *) die "unsupported Docker architecture: $arch" ;;
  esac
}

redact() {
  local text
  text="$(cat)"
  text="${text//${agent_auth_token}/[REDACTED]}"
  printf '%s' "$text" | sed -E 's/(Authorization: Bearer )[[:graph:]]+/\1[REDACTED]/g'
}

cleanup() {
  local name
  for name in ${containers[@]+"${containers[@]}"}; do
    docker rm -fv "$name" >/dev/null 2>&1 || true
  done
  if [[ "$network_created" == true ]]; then
    docker network rm "$network_name" >/dev/null 2>&1 || true
  fi
  rm -rf "$work_dir"
}

on_exit() {
  local status="$?"
  trap - EXIT
  if [[ "$status" -ne 0 ]]; then
    {
      echo '=== AIKit logs ==='
      docker logs "$aikit_container_name" 2>&1 || true
      if [[ -n "$agent_container_name" ]]; then
        echo '=== agent logs ==='
        docker logs "$agent_container_name" 2>&1 || true
      fi
    } | redact >&2
    log 'Live AIKit-backed AgentKit E2E failed'
  fi
  cleanup
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_http() {
  local name="$1" url="$2" deadline=$((SECONDS + $3))
  while (( SECONDS < deadline )); do
    [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" == true ]] ||
      die "$name exited before readiness"
    if curl -fsS --connect-timeout 2 --max-time 5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  die "$name did not become ready at $url"
}

run_adapter() {
  local adapter="$1" slug target adapter_image agent_image fixture response
  case "$adapter" in
    pydantic-ai)
      slug=pydantic
      target=build-serve
      adapter_image="agentkit-serve:$tag"
      ;;
    microsoft-agent-framework)
      slug=maf
      target=build-serve-maf
      adapter_image="agentkit-serve-maf:$tag"
      ;;
    langgraph)
      slug=langgraph
      target=build-serve-langgraph
      adapter_image="agentkit-serve-langgraph:$tag"
      ;;
  esac
  fixture="test/agentkitfile-$slug-live.yaml"
  agent_image="$slug-live-agent:$tag"
  response="$work_dir/$slug-response.json"

  log "Building $adapter adapter and live agent for $platform"
  make "$target" TAG="$tag" PLATFORM="$platform"
  docker buildx build ${buildx_args[@]+"${buildx_args[@]}"} . -f "$fixture" \
    --build-arg BUILDKIT_SYNTAX="agentkit:$tag" \
    --build-arg adapter="$adapter_image" \
    --platform "$platform" -t "$agent_image" --load --provenance=false

  log "Starting live $adapter agent"
  agent_container_name="$run_id-$slug-agent"
  containers+=("$agent_container_name")
  docker run -d --name "$agent_container_name" --platform "$platform" \
    --network "$network_name" -p "127.0.0.1:$agent_host_port:8080" \
    -e AGENTKIT_BIND=0.0.0.0 -e AGENTKIT_AUTH_TOKEN="$agent_auth_token" \
    -e MODEL_API_KEY=not-needed "$agent_image" >/dev/null
  wait_for_http "$agent_container_name" "http://127.0.0.1:$agent_host_port/healthz" 180

  log "Calling live $adapter agent /v1/chat/completions"
  curl -fsS --connect-timeout 5 --max-time 120 \
    -H "Authorization: Bearer $agent_auth_token" -H 'Content-Type: application/json' \
    --data '{"model":"qwen-3.5-2b","stream":false,"messages":[{"role":"user","content":"Reply with exactly one short sentence that includes the sentinel token DONE42."}]}' \
    "http://127.0.0.1:$agent_host_port/v1/chat/completions" >"$response"
  jq '{model, content: .choices[0].message.content}' "$response" | redact >&2
  jq -e '.choices[0].message.content | type == "string" and contains("DONE42")' "$response" >/dev/null
  log "Live AIKit-backed $adapter E2E passed"

  # Release the agent and its published port before starting the next adapter.
  docker rm -fv "$agent_container_name" >/dev/null
  containers=("$aikit_container_name")
  agent_container_name=""
}

main() {
  local command adapter
  for command in curl docker go jq make; do require_cmd "$command"; done
  [[ -n "$platform" ]] || platform="$(default_platform)"
  cd "$repo_root"

  log "Creating private Docker network $network_name"
  docker network create "$network_name" >/dev/null
  network_created=true
  containers+=("$aikit_container_name")
  log "Starting AIKit ($aikit_image)"
  start_aikit "$repo_root/test/aikit-e2e/model.yaml" -d --name "$aikit_container_name" --platform "$platform" \
    --network "$network_name" --network-alias aikit \
    -p "127.0.0.1:$aikit_host_port:8080"
  wait_for_http "$aikit_container_name" "http://127.0.0.1:$aikit_host_port/readyz" 300
  curl -fsS --max-time 15 "http://127.0.0.1:$aikit_host_port/v1/models" >"$work_dir/models.json"
  jq -e --arg model "$aikit_model" '.data | any(.id == $model)' "$work_dir/models.json" >/dev/null
  log 'Warming the local model before the agent smokes'
  warm_aikit "http://127.0.0.1:$aikit_host_port" "$work_dir/warmup.json"

  log "Building AgentKit frontend for $platform"
  buildx_args=()
  if [[ -n "$builder" ]]; then
    docker buildx inspect "$builder" --bootstrap
    export BUILDX_BUILDER="$builder"
    buildx_args=(--builder "$builder")
  else
    docker buildx inspect --bootstrap
  fi
  make build-agentkit TAG="$tag"

  for adapter in "${adapters[@]}"; do
    run_adapter "$adapter"
  done
  log 'Live AIKit-backed AgentKit E2E passed'
}

main "$@"
