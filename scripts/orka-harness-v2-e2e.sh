#!/usr/bin/env bash

# Do not trace this script: runtime/controller credentials are generated per run.
set +x
set -Eeuo pipefail
umask 077

log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: scripts/orka-harness-v2-e2e.sh offline [adapter]
       scripts/orka-harness-v2-e2e.sh live [adapter]

Both modes default to all adapters: pydantic-ai, microsoft-agent-framework, langgraph.
Use an adapter argument to select one; maf aliases microsoft-agent-framework.
Live uses a digest-pinned local AIKit model without external model credentials.

PLATFORM defaults to the Linux Docker daemon's architecture. BUILDER may select a
docker-driver Buildx builder. ARTIFACT_DIR selects the parent for safe run results.
USAGE
}

mode="${1:-offline}"
case "$mode" in -h|--help) usage; exit 0 ;; esac
[[ $# -le 2 ]] || die 'expected a mode and at most one adapter'
case "$mode" in
  offline|live) ;;
  *) usage >&2; die "unsupported mode: $mode" ;;
esac
adapters=()
if [[ -n "${2:-}" ]]; then
  case "$2" in
    maf) adapters=(microsoft-agent-framework) ;;
    pydantic-ai|microsoft-agent-framework|langgraph) adapters=("$2") ;;
    *) die "unsupported adapter: $2" ;;
  esac
else
  adapters=(pydantic-ai microsoft-agent-framework langgraph)
fi

for command in curl docker git go jq make; do
  command -v "$command" >/dev/null 2>&1 || die "missing required command: $command"
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/aikit-e2e-common.sh
source "$repo_root/scripts/aikit-e2e-common.sh"
temp_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
work_dir="$(mktemp -d "${temp_root%/}/agentkit-orka-v2.XXXXXX")"
run_id="${work_dir##*/}"
label="io.agentkit.orka-v2.run=$run_id"
artifact_dir=''
phase=setup
active_adapter=setup
active_runtime=''
active_artifacts=''
aikit_name=''
containers=()
volumes=()
networks=()
images=()

# Only fixed supervisor fields are retained. Child stderr, request bodies,
# credentials, and errorDetail never enter the result directory or job output.
collect_runtime_diagnostics() {
  local name="$1" destination="$2"
  docker logs "$name" 2>&1 | jq -Rc '
    fromjson? | objects
    | with_entries(select(.key == "msg" or .key == "level" or .key == "stage"
        or .key == "rpcCode" or .key == "rpcErrorName" or .key == "resultOutcome"
        or .key == "resultStopReason" or .key == "outcome" or .key == "stopReason"
        or .key == "accepted")
      | select(.value | type == "string" or type == "number" or type == "boolean"))
    | select(length > 0)
  ' >"$destination/supervisor.jsonl" || true
  collect_container_state "$name" "$destination/runtime-state.json"
}

collect_container_state() {
  docker inspect --format \
    '{"running":{{.State.Running}},"exitCode":{{.State.ExitCode}},"oomKilled":{{.State.OOMKilled}}}' \
    "$1" >"$2" 2>/dev/null || true
}

cleanup() {
  local name remaining resource cleaned=true
  local list_args=()
  # Let the real supervisor drain before removing its network and broker.
  if [[ -n "$active_runtime" ]]; then
    docker stop --time 50 "$active_runtime" >/dev/null 2>&1 || true
    collect_runtime_diagnostics "$active_runtime" "$active_artifacts"
  fi
  if [[ -n "$aikit_name" && -n "$artifact_dir" ]]; then
    collect_container_state "$aikit_name" "$artifact_dir/aikit-state.json"
  fi
  for name in ${containers[@]+"${containers[@]}"}; do
    docker rm -fv "$name" >/dev/null 2>&1 || true
  done
  for name in ${volumes[@]+"${volumes[@]}"}; do
    docker volume rm "$name" >/dev/null 2>&1 || true
  done
  for name in ${networks[@]+"${networks[@]}"}; do
    docker network rm "$name" >/dev/null 2>&1 || true
  done
  for name in ${images[@]+"${images[@]}"}; do
    docker image rm --no-prune "$name" >/dev/null 2>&1 || true
  done

  for resource in container network volume; do
    list_args=(-q --filter "label=$label")
    [[ "$resource" != container ]] || list_args+=(-a)
    if ! remaining="$(docker "$resource" ls "${list_args[@]}" 2>/dev/null)"; then
      cleaned=false
    elif [[ -n "$remaining" ]]; then
      log "Cleanup left run-owned $resource resources: $remaining"
      cleaned=false
    fi
  done
  rm -rf "$work_dir" || cleaned=false
  [[ "$cleaned" == true ]]
}

on_exit() {
  local status="$?" cleaned=true
  trap - EXIT INT TERM
  set +e
  if [[ "$status" -ne 0 ]]; then
    log "FAIL adapter=$active_adapter stage=$phase exit=$status"
  fi
  cleanup || cleaned=false
  if [[ "$cleaned" != true ]]; then
    log 'Failed to remove or verify all run-owned resources'
    [[ "$status" -ne 0 ]] || status=1
  fi
  if [[ -n "$artifact_dir" ]]; then
    if ! jq -n --arg runID "$run_id" --argjson exitCode "$status" \
      --argjson resourcesRemoved "$cleaned" \
      '{runID: $runID, exitCode: $exitCode, resourcesRemoved: $resourcesRemoved}' \
      >"$artifact_dir/cleanup.json"; then
      status=1
    fi
    log "Safe results: $artifact_dir"
  fi
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

artifact_root="${ARTIFACT_DIR:-${temp_root%/}/agentkit-orka-v2-artifacts}"
mkdir -p "$artifact_root/$run_id"
artifact_dir="$(cd "$artifact_root/$run_id" && pwd)"

daemon_platform="$(docker info --format '{{.OSType}}/{{.Architecture}}')"
case "$daemon_platform" in
  linux/amd64|linux/x86_64) default_platform=linux/amd64 ;;
  linux/arm64|linux/aarch64) default_platform=linux/arm64 ;;
  *) die "a Linux amd64 or arm64 Docker daemon is required; got $daemon_platform" ;;
esac
platform="${PLATFORM:-$default_platform}"
case "$platform" in linux/amd64|linux/arm64) ;; *) die "unsupported PLATFORM: $platform" ;; esac
builder="${BUILDER:-}"
builder_args=()
if [[ -n "$builder" ]]; then
  builder_args=("$builder")
  export BUILDX_BUILDER="$builder"
fi
builder_info="$(docker buildx inspect ${builder_args[@]+"${builder_args[@]}"} --bootstrap)"
[[ "$(awk '$1 == "Driver:" {print $2}' <<<"$builder_info")" == docker ]] ||
  die 'the selected Buildx builder must use the docker driver to resolve freshly built local images'

orka_revision="$(cat "$repo_root/test/orka-harness-v2/orka-revision")"
[[ "$orka_revision" =~ ^[0-9a-f]{40}$ ]] || die 'orka-revision must contain one full lowercase commit SHA'
source_revision="$(git -C "$repo_root" rev-parse HEAD)"
source_dirty=false
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || source_dirty=true
jq -n --arg runID "$run_id" --arg mode "$mode" --arg platform "$platform" \
  --arg agentkitRevision "$source_revision" --arg orkaRevision "$orka_revision" \
  --argjson workingTreeDirty "$source_dirty" --args \
  '{runID: $runID, mode: $mode, platform: $platform, agentkitRevision: $agentkitRevision,
    workingTreeDirty: $workingTreeDirty, orkaRevision: $orkaRevision, adapters: $ARGS.positional}' \
  "${adapters[@]}" >"$artifact_dir/run.json"

phase=orka-source
log "Fetching Orka $orka_revision for $platform"
git init -q "$work_dir/orka"
GIT_TERMINAL_PROMPT=0 git -C "$work_dir/orka" \
  -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=60 fetch -q --depth=1 \
  https://github.com/orka-agents/orka.git "$orka_revision"
git -C "$work_dir/orka" checkout -q --detach FETCH_HEAD
[[ "$(git -C "$work_dir/orka" rev-parse HEAD)" == "$orka_revision" ]] || die 'Orka checkout does not match the pin'
mkdir -p "$work_dir/orka/cmd/agentkit-v2-e2e"
cp "$repo_root"/test/orka-harness-v2/probe/*.go "$work_dir/orka/cmd/agentkit-v2-e2e/"
phase=probe-build
(
  cd "$work_dir/orka"
  CGO_ENABLED=0 GOOS=linux GOARCH="${platform#linux/}" \
    go build -buildvcs=false -trimpath -o "$work_dir/probe" ./cmd/agentkit-v2-e2e
)

registry_image='docker.io/library/registry:3@sha256:1be55279f18a2fe1a74edf2664cac61c1bea305b7b4642dab412e7affdcb3e33'
registry_name="$run_id-registry"
registry_volume="$run_id-registry"
network_name="$run_id-runtime"

wait_ready() {
  local name="$1" url="$2" deadline=$((SECONDS + $3))
  while (( SECONDS < deadline )); do
    if curl -fsS --connect-timeout 2 --max-time 3 "$url" >/dev/null 2>&1; then
      return 0
    fi
    [[ "$(docker inspect --format '{{.State.Running}}' "$name" 2>/dev/null)" == true ]] ||
      die "$name exited before becoming ready"
    sleep 1
  done
  die "$name did not become ready within $3 seconds"
}

published_port() {
  local binding
  binding="$(docker port "$1" "$2/tcp")"
  [[ "$binding" =~ ^127\.0\.0\.1:([0-9]+)$ ]] || die "expected one loopback port for $1"
  printf '%s\n' "${BASH_REMATCH[1]}"
}

phase=registry
volumes+=("$registry_volume")
docker volume create --label "$label" "$registry_volume" >/dev/null
containers+=("$registry_name")
docker run -d --name "$registry_name" --label "$label" --platform "$platform" \
  -p 127.0.0.1::5000 --mount "type=volume,src=$registry_volume,dst=/var/lib/registry" \
  "$registry_image" >/dev/null
registry_port="$(published_port "$registry_name" 5000)"
wait_ready "$registry_name" "http://127.0.0.1:$registry_port/v2/" 45
networks+=("$network_name")
network_args=()
[[ "$mode" != offline ]] || network_args=(--internal)
docker network create --label "$label" ${network_args[@]+"${network_args[@]}"} "$network_name" >/dev/null

if [[ "$mode" == live ]]; then
  phase=aikit-readiness
  aikit_name="$run_id-aikit"
  containers+=("$aikit_name")
  log "Starting local AIKit ($aikit_image)"
  start_aikit "$repo_root/test/aikit-e2e/model.yaml" -d --name "$aikit_name" --label "$label" --platform "$platform" \
    --network "$network_name" --network-alias aikit -p 127.0.0.1::8080
  aikit_port="$(published_port "$aikit_name" 8080)"
  wait_ready "$aikit_name" "http://127.0.0.1:$aikit_port/readyz" 300
  curl -fsS --max-time 15 "http://127.0.0.1:$aikit_port/v1/models" >"$work_dir/models.json"
  jq -e --arg model "$aikit_model" '.data | any(.id == $model)' "$work_dir/models.json" >/dev/null ||
    die "AIKit did not advertise the required $aikit_model model"
  phase=aikit-warmup
  warm_aikit "http://127.0.0.1:$aikit_port" "$work_dir/warmup.json"
fi

phase=frontend-build
images+=("agentkit:$run_id")
make -C "$repo_root" build-agentkit TAG="$run_id"

immutable_ref() {
  local tag="$1" ref
  ref="$(docker image inspect "$tag" --format '{{json .RepoDigests}}' |
    jq -er --arg prefix "${tag%:*}@" '[.[] | select(startswith($prefix))][0]')"
  [[ "${ref##*@}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "no immutable RepoDigest for $tag"
  printf '%s\n' "$ref"
}

run_adapter() {
  local adapter="$1" target serve_image fixture model source_tag source_ref adapter_digest
  local composed_tag composed_ref adapter_work config_volume session_volume extract_name prepare_name probe_name
  local probe_deadline probe_exit timeout=600
  local probe_args=()
  active_adapter="$adapter"
  active_artifacts="$artifact_dir/$adapter"
  mkdir -p "$active_artifacts"
  case "$adapter" in
    pydantic-ai) target=build-serve; serve_image="agentkit-serve:$run_id" ;;
    microsoft-agent-framework) target=build-serve-maf; serve_image="agentkit-serve-maf:$run_id" ;;
    langgraph) target=build-serve-langgraph; serve_image="agentkit-serve-langgraph:$run_id" ;;
  esac
  fixture="test/orka-harness-v2/agentkitfile-$adapter.yaml"
  model=gpt-4o-mini
  if [[ "$mode" == live ]]; then
    fixture="test/orka-harness-v2/agentkitfile-$adapter-live.yaml"
    model="$aikit_model"
    timeout=900
    probe_args=(--upstream http://aikit:8080)
  fi
  source_tag="127.0.0.1:$registry_port/agentkit-$adapter:$run_id"
  composed_tag="127.0.0.1:$registry_port/orka-$adapter:$run_id"
  images+=("$serve_image" "$source_tag" "$composed_tag")
  phase='adapter-build'
  log "Building $adapter and its $mode fixture from the current checkout"
  make -C "$repo_root" "$target" TAG="$run_id" PLATFORM="$platform"
  make -C "$repo_root" build-test-agent TAG="$run_id" PLATFORM="$platform" BUILDER="$builder" \
    RUNTIME="$adapter" SERVE_IMAGE="$serve_image" FIXTURE="$fixture" AGENT_IMAGE="$source_tag"
  docker push "$source_tag"
  source_ref="$(immutable_ref "$source_tag")"
  adapter_digest="${source_ref##*@}"
  phase=supervisor-build
  DOCKER_DEFAULT_PLATFORM="$platform" make -C "$work_dir/orka" docker-build-acp-agentkit-runtime \
    CONTAINER_TOOL=docker ACP_AGENTKIT_RUNTIME_IMG="$composed_tag" \
    AGENTKIT_RUNTIME_IMAGE="$source_ref" AGENTKIT_ADAPTER_DIGEST="$adapter_digest"
  docker push "$composed_tag"
  composed_ref="$(immutable_ref "$composed_tag")"
  jq -n --arg adapter "$adapter" --arg sourceImage "$source_ref" \
    --arg adapterDigest "$adapter_digest" --arg composedImage "$composed_ref" \
    --arg adapterImageID "$(docker image inspect "$serve_image" --format '{{.Id}}')" \
    --arg frontendImageID "$(docker image inspect "agentkit:$run_id" --format '{{.Id}}')" \
    '{adapter: $adapter, sourceImage: $sourceImage, adapterDigest: $adapterDigest,
      composedImage: $composedImage, adapterImageID: $adapterImageID, frontendImageID: $frontendImageID}' \
    >"$active_artifacts/images.json"

  phase=prepare
  adapter_work="$work_dir/$adapter"
  mkdir -p "$adapter_work"
  cp "$work_dir/probe" "$adapter_work/probe"
  config_volume="$run_id-$adapter-config"
  session_volume="$run_id-$adapter-sessions"
  volumes+=("$config_volume" "$session_volume")
  docker volume create --label "$label" "$config_volume" >/dev/null
  docker volume create --label "$label" "$session_volume" >/dev/null
  extract_name="$run_id-$adapter-extract"
  containers+=("$extract_name")
  docker create --name "$extract_name" --label "$label" --platform "$platform" "$source_ref" >/dev/null
  docker cp "$extract_name:/agent/agent.yaml" "$adapter_work/agent.yaml"
  docker rm -v "$extract_name" >/dev/null
  prepare_name="$run_id-$adapter-prepare"
  containers+=("$prepare_name")
  docker run --name "$prepare_name" --label "$label" --platform "$platform" --network none \
    --user 0:0 --entrypoint /bin/sh \
    --mount "type=bind,src=$adapter_work,dst=/input,readonly" \
    --mount "type=volume,src=$config_volume,dst=/e2e" \
    "$source_ref" -ec 'cp /input/probe /input/agent.yaml /e2e/; chmod 0700 /e2e; exec "$@"' -- \
    /e2e/probe prepare --dir /e2e --adapter "$adapter" --adapter-digest "$adapter_digest" \
    --model "$model" --mode "$mode"
  docker cp "$prepare_name:/e2e/runtime.env" "$adapter_work/runtime.env"
  docker rm -v "$prepare_name" >/dev/null

  phase=runtime-start
  active_runtime="$run_id-$adapter-runtime"
  containers+=("$active_runtime")
  # Mirrors Orka's RuntimePool securityContext. The official image entrypoint
  # remains intact; only /sessions and the production scratch mounts are writable.
  docker run -d --name "$active_runtime" --label "$label" --platform "$platform" \
    --network "$network_name" --network-alias runtime \
    --read-only --cap-drop ALL --cap-add CHOWN --cap-add KILL --cap-add SETGID --cap-add SETUID \
    --security-opt no-new-privileges --env-file "$adapter_work/runtime.env" \
    --mount "type=volume,src=$config_volume,dst=/e2e,readonly" \
    --mount "type=volume,src=$session_volume,dst=/sessions" \
    --tmpfs /tmp:rw,nosuid,nodev,size=512m --tmpfs /home/worker:rw,nosuid,nodev,size=256m \
    "$composed_ref" >/dev/null

  phase=probe
  probe_name="$run_id-$adapter-probe"
  containers+=("$probe_name")
  log "Exercising $adapter through the production harness v2 supervisor"
  docker run -d --name "$probe_name" --label "$label" --platform "$platform" \
    --network "$network_name" --network-alias fixture --pid "container:$active_runtime" \
    --user 0:0 --entrypoint /e2e/probe \
    --mount "type=volume,src=$config_volume,dst=/e2e,readonly" \
    --mount "type=volume,src=$session_volume,dst=/sessions,readonly" \
    --mount "type=bind,src=$active_artifacts,dst=/results" \
    "$source_ref" run --dir /e2e --runtime-url http://runtime:8080 --results /results/result.json \
    ${probe_args[@]+"${probe_args[@]}"} >/dev/null
  probe_deadline=$((SECONDS + timeout))
  while [[ "$(docker inspect --format '{{.State.Running}}' "$probe_name")" == true ]]; do
    (( SECONDS < probe_deadline )) || die "$adapter probe exceeded $timeout seconds"
    sleep 1
  done
  probe_exit="$(docker inspect --format '{{.State.ExitCode}}' "$probe_name")"
  # docker cp assigns host ownership even when the probe created the safe result
  # as root. Copy outside the bind mount first, then replace its directory entry.
  docker cp "$probe_name:/results/result.json" "$adapter_work/result.json" 2>/dev/null ||
    die "$adapter probe exited with code $probe_exit without writing result.json"
  mv -f "$adapter_work/result.json" "$active_artifacts/result.json"
  jq -c . "$active_artifacts/result.json"
  [[ "$probe_exit" == 0 ]] || die "$adapter probe failed with exit code $probe_exit"
  jq -e --arg adapter "$adapter" --arg mode "$mode" --arg digest "$adapter_digest" '
    (.scenarios | map(.scenario)) as $executed
    | ["startup-and-authentication", "admission-rejections", "child-and-session-identity",
       "completion", "continuation", "tool", "successful-session-deletion",
       "cancel-session", "cancel", "session-cleanup"] as $common
    | ($common + if $mode == "offline" then
        ["provider-failure-session", "provider-failure", "tool-failure-session", "tool-failure",
         "deadline-session", "deadline"] else [] end) as $required
    | .adapter == $adapter and .mode == $mode and .adapterDigest == $digest and .passed == true
      and (.scenarios | all(.status == "passed"))
      and (($executed | sort) == ($required | sort))
  ' "$active_artifacts/result.json" >/dev/null || die "$adapter result omitted a required successful scenario"
  phase=supervisor-shutdown
  docker stop --time 60 "$active_runtime" >/dev/null
  [[ "$(docker inspect --format '{{.State.ExitCode}}' "$active_runtime")" == 0 ]] ||
    die "$adapter supervisor did not shut down cleanly"
  collect_runtime_diagnostics "$active_runtime" "$active_artifacts"
  docker rm -v "$probe_name" "$active_runtime" >/dev/null
  active_runtime=''
  log "PASS adapter=$adapter mode=$mode"
}

for adapter in "${adapters[@]}"; do
  run_adapter "$adapter"
done
phase=complete
