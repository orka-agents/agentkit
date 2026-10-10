#!/usr/bin/env bash
# Run on the Linux Docker daemon's host: the actor has an internal-only network.
# Model effects use the pinned Go host's offline provider, never external keys.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:---build}"
if [[ $# -gt 2 || ( "$mode" != --build && "$mode" != --skip-build ) ]]; then
    printf '%s\n' 'usage: agentsessions-e2e.sh [--build|--skip-build] [pydantic-ai|maf|microsoft-agent-framework|langgraph]' >&2
    exit 2
fi
runtime="${2:-${AGENTKIT_AGENTSESSIONS_RUNTIME:-pydantic-ai}}"
case "$runtime" in
    pydantic-ai) target=build-serve; fixture=Agentkitfile.yaml; image_name=agentkit-agentsessions-text ;;
    maf|microsoft-agent-framework)
        runtime=microsoft-agent-framework
        target=build-serve-maf; fixture=Agentkitfile-maf.yaml; image_name=agentkit-agentsessions-maf-text ;;
    langgraph) target=build-serve-langgraph; fixture=Agentkitfile-langgraph.yaml; image_name=agentkit-agentsessions-langgraph-text ;;
    *) printf '%s\n' 'unsupported agentsessions proof runtime' >&2; exit 2 ;;
esac
for tool in docker go setsid; do
    command -v "$tool" >/dev/null || { printf '%s is required\n' "$tool" >&2; exit 2; }
done
cd "$root"
run_id="${AGENTKIT_AGENTSESSIONS_RUN_ID:-run-$$-$RANDOM-$RANDOM}"
if [[ ! "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$ ]]; then
    printf '%s\n' 'invalid test resource run ID' >&2
    exit 2
fi
label="io.github.orka-agents.agentkit.agentsessions-e2e=$run_id"
runner=''
scratch="$(mktemp -d -t agentkit-agentsessions.XXXXXXXX)"
cleanup() {
    local ids id quiet=0 deadline=$((SECONDS + 15))
    if [[ -n "$runner" ]]; then
        # This is the dedicated process group created below, never a shared job.
        kill -TERM -- "-$runner" 2>/dev/null || true
        wait "$runner" 2>/dev/null || true
    fi
    # A Docker create request can commit after its CLI process is terminated.
    # Reap late arrivals and require a quiescent interval, not one empty snapshot.
    while (( quiet < 4 )); do
        (( SECONDS < deadline )) || return 1
        ids="$(docker ps -aq --filter "label=$label")" || return 1
        if [[ -n "$ids" ]]; then
            quiet=0
            while IFS= read -r id; do docker rm -f "$id" >/dev/null || return 1; done <<< "$ids"
        fi
        ids="$(docker network ls -q --filter "label=$label")" || return 1
        if [[ -n "$ids" ]]; then
            quiet=0
            while IFS= read -r id; do docker network rm "$id" >/dev/null || return 1; done <<< "$ids"
        fi
        quiet=$((quiet + 1))
        sleep 0.25
    done
    rm -rf "$scratch"
}
on_exit() {
    local status=$?
    trap - EXIT
    trap '' INT TERM
    if ! cleanup; then
        printf '%s\n' 'agentsessions test resource cleanup failed' >&2
        [[ "$status" -ne 0 ]] || status=1
    fi
    exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
tag="${TAG:-agentsessions-e2e}"
image="${AGENTKIT_AGENTSESSIONS_IMAGE:-$image_name:$tag}"
if [[ "$mode" == --build ]]; then
    builder="${BUILDER:-default}"
    export BUILDX_BUILDER="$builder"
    # The gateway frontend must see local frontend/adapter images, not a remote builder.
    make build-agentkit TAG="$tag"
    make "$target" TAG="$tag"
    make build-test-agent TAG="$tag" BUILDER="$builder" RUNTIME="$runtime" \
        FIXTURE="test/agentsessions/$fixture" AGENT_IMAGE="$image"
fi
docker image inspect "$image" --format '{{.Id}}'
cd test/agentsessions
# The Go test owns normal cleanup; shell traps also reclaim this run's labelled
# resources on interruption. An isolated process group includes the test binary.
TMPDIR="$scratch" AGENTKIT_AGENTSESSIONS_RUN_ID="$run_id" AGENTKIT_AGENTSESSIONS_IMAGE="$image" \
    AGENTKIT_AGENTSESSIONS_RUNTIME="$runtime" \
    setsid --wait go test -v -count=1 -run '^TestContainerStatelessReplay$' ./... &
runner=$!
wait "$runner"
runner=''
