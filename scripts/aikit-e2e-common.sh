#!/usr/bin/env bash

# Sourced by the two real-inference E2E entrypoints. Callers own cleanup.
aikit_image="${AIKIT_IMAGE:-ghcr.io/kaito-project/aikit/qwen3.5:2b@sha256:d838c5eebf533b5b73f67cc7f4984921f87d64d28e4a52d482740b667f46b8b4}"
aikit_model=qwen-3.5-2b

start_aikit() {
  local config="$1"
  shift
  local cpus
  cpus="$(docker info --format '{{.NCPU}}')" || return
  [[ "$cpus" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid Docker CPU count: %s\n' "$cpus" >&2; return 1; }
  if (( cpus > 4 )); then cpus=4; fi
  docker run "$@" --cpus "$cpus" -e LOCALAI_FORCE_META_BACKEND_CAPABILITY=cpu \
    --mount "type=bind,src=$config,dst=/config.yaml,readonly" \
    "$aikit_image" --config-file=/config.yaml >/dev/null
}

warm_aikit() {
  local url="$1" response="$2"
  # /readyz does not prove that the model has loaded. Warm it before timed turns.
  curl -fsS --connect-timeout 5 --max-time 300 \
    -H 'Content-Type: application/json' \
    --data '{"model":"qwen-3.5-2b","stream":false,"temperature":0,"max_tokens":32,"messages":[{"role":"user","content":"Reply with OK."}]}' \
    "$url/v1/chat/completions" >"$response" || return
  jq -e --arg model "$aikit_model" \
    '.model == $model and (.created | type == "number") and
     (.choices[0].message.content | type == "string" and length > 0)' "$response" >/dev/null
}
