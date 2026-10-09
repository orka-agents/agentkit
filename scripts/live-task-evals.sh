#!/usr/bin/env bash
# Live quality baseline. Run on the Linux Docker daemon host, including over SSH.
set +x
set -Eeuo pipefail

log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
usage='usage: live-task-evals.sh [pydantic-ai|microsoft-agent-framework|langgraph] (maf alias supported)'
[[ "$#" -le 1 ]] || die "$usage"
adapters=(pydantic-ai microsoft-agent-framework langgraph)
if [[ "$#" -eq 1 ]]; then
  case "$1" in
    pydantic-ai|microsoft-agent-framework|langgraph) adapters=("$1") ;;
    maf) adapters=(microsoft-agent-framework) ;;
    *) die "invalid adapter: $1; $usage" ;;
  esac
fi
trials="${EVAL_TRIALS:-3}"
case_timeout="${EVAL_CASE_TIMEOUT_SECONDS:-120}"
suite_timeout="${EVAL_SUITE_TIMEOUT_SECONDS:-3600}"
[[ "$trials" =~ ^([1-9]|10)$ ]] || die 'EVAL_TRIALS must be an integer 1..10'
for value in "$case_timeout" "$suite_timeout"; do
  [[ "$value" =~ ^[1-9][0-9]{0,5}$ ]] || die 'eval timeouts must be positive integers up to 999999 seconds'
done
for command in docker make curl jq python3; do
  command -v "$command" >/dev/null 2>&1 || die "missing required command: $command"
done
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/aikit-e2e-common.sh
source "$repo_root/scripts/aikit-e2e-common.sh"
# Validate selections before any Docker, build or inference side effects.
python3 - "$repo_root/test/evals" "${EVAL_CASES:-}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from cases import CASE_IDS
if sys.argv[2]:
    cases = sys.argv[2].split(',')
    if len(set(cases)) != len(cases) or any(case not in CASE_IDS for case in cases):
        raise SystemExit('unknown or duplicate EVAL_CASES')
PY
work_dir="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/agentkit-task-evals.XXXXXX")"
run_id="${work_dir##*/}"
artifact_dir="${ARTIFACT_DIR:-$repo_root/artifacts/live-task-evals/$run_id}"
mkdir -p "$artifact_dir"
artifact_dir="$(cd "$artifact_dir" && pwd)"
# Refuse accidental overwrite of another baseline run.
[[ ! -e "$artifact_dir/run.json" ]] || die "artifact directory already contains run.json: $artifact_dir"
network="$run_id-network"
model_container="$run_id-model"
network_created=false
model_created=false
phase=setup
platform="${PLATFORM:-}"
builder="${BUILDER:-}"
tag="${TAG:-$run_id}"
source_revision="${SOURCE_REVISION:-$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf unknown)}"
source_dirty=true
if git -C "$repo_root" diff --quiet HEAD -- 2>/dev/null && [[ -z "$(git -C "$repo_root" ls-files --others --exclude-standard 2>/dev/null)" ]]; then
  source_dirty=false
fi

# shellcheck disable=SC2329 # Invoked indirectly by the EXIT trap.
finish() {
  local status="$?" cleanup_failed=false container owned
  trap - EXIT
  # Attempt every run-owned cleanup operation, even if reporting/storage fails.
  if [[ "$network_created" == true ]]; then
    if ! owned="$(docker ps -a --format '{{.Names}}' --filter "label=agentkit.eval-run=$network")"; then
      cleanup_failed=true
    fi
    if [[ "$model_created" == true ]] && ! docker rm -f "$model_container" >/dev/null 2>&1; then cleanup_failed=true; fi
    while IFS= read -r container; do
      if [[ "$model_created" == true && "$container" == "$model_container" ]]; then continue; fi
      if [[ -n "$container" ]] && ! docker rm -f "$container" >/dev/null 2>&1; then cleanup_failed=true; fi
    done <<< "$owned"
    if ! docker network rm "$network" >/dev/null 2>&1; then cleanup_failed=true; fi
  fi
  if ! rm -rf "$work_dir"; then cleanup_failed=true; fi
  if [[ "$cleanup_failed" == true ]]; then
    status=2
    log 'Run-owned Docker cleanup failed; inspect the run label before retrying.'
  fi
  if ! python3 - "$artifact_dir/run.json" "$status" "$phase" "$source_revision" "$cleanup_failed" <<'PY'
import datetime
import json
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(json.dumps({
    'schemaVersion': 1, 'mode': 'live', 'exitCode': int(sys.argv[2]),
    'phase': sys.argv[3], 'sourceRevision': sys.argv[4], 'cleanupFailed': sys.argv[5] == 'true',
    'finishedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
}, indent=2) + '\n')
PY
  then
    status=2
  fi
  log "Eval reports: $artifact_dir"
  exit "$status"
}
trap 'finish' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_model() {
  local deadline=$((SECONDS + 300))
  while (( SECONDS < deadline )); do
    [[ "$(docker inspect -f '{{.State.Running}}' "$model_container" 2>/dev/null)" == true ]] || die 'model exited before readiness'
    if curl -fsS --max-time 3 "$model_url/readyz" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  die 'model readiness timeout'
}

cd "$repo_root"
if [[ -z "$platform" ]]; then
  case "$(docker info --format '{{.Architecture}}')" in
    aarch64|arm64) platform=linux/arm64 ;;
    x86_64|amd64) platform=linux/amd64 ;;
    *) die 'unsupported Docker architecture' ;;
  esac
fi
if [[ -n "$builder" ]]; then export BUILDX_BUILDER="$builder"; fi
[[ "$(docker buildx inspect | awk '/^Driver:/ {print $2}')" == docker ]] || die 'evals require a daemon-backed Buildx builder'
log 'Building the frontend and common MCP fixture image'
make build-agentkit build-serve TAG="$tag" PLATFORM="$platform" BUILDER="$builder"
docker network create --label "agentkit.eval-run=$network" "$network" >/dev/null
network_created=true
log "Starting pinned AIKit model on $platform, capped at four CPUs"
# start_aikit is shared with the existing live E2E jobs.
start_aikit "$repo_root/test/aikit-e2e/model.yaml" -d --name "$model_container" \
  --platform "$platform" --network "$network" --network-alias aikit \
  --label "agentkit.eval-run=$network" -p 127.0.0.1::8080
model_created=true
binding="$(docker port "$model_container" 8080/tcp | head -n 1)"
model_url="http://$binding"
wait_model
warm_aikit "$model_url" "$work_dir/warmup.json"
phase=evaluation
status=0
for adapter in "${adapters[@]}"; do
  case "$adapter" in
    pydantic-ai) serve_image="agentkit-serve:$tag" ;;
    microsoft-agent-framework)
      make build-serve-maf TAG="$tag" PLATFORM="$platform" BUILDER="$builder"
      serve_image="agentkit-serve-maf:$tag" ;;
    langgraph)
      make build-serve-langgraph TAG="$tag" PLATFORM="$platform" BUILDER="$builder"
      serve_image="agentkit-serve-langgraph:$tag" ;;
  esac
  agent_image="task-evals-$adapter:$tag"
  make build-test-agent TAG="$tag" PLATFORM="$platform" BUILDER="$builder" RUNTIME="$adapter" \
    SERVE_IMAGE="$serve_image" FIXTURE="test/evals/agentkitfile-$adapter.yaml" AGENT_IMAGE="$agent_image"
  dirty_args=()
  if [[ "$source_dirty" == true ]]; then dirty_args+=(--source-dirty); fi
  log "Evaluating $adapter: $trials trials per selected case"
  python3 test/evals/run.py --adapter "$adapter" --agent-image "$agent_image" \
    --fixture-image "agentkit-serve:$tag" --model-image "$aikit_image" --network "$network" \
    --platform "$platform" --model "$aikit_model" --trials "$trials" --cases "${EVAL_CASES:-}" \
    --case-timeout "$case_timeout" --suite-timeout "$suite_timeout" \
    --source-revision "$source_revision" "${dirty_args[@]+"${dirty_args[@]}"}" \
    --output "$artifact_dir/$adapter.json" || status=$?
done
phase=finished
exit "$status"
