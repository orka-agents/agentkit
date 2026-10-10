#!/usr/bin/env bash
# The caller supplies a freshly built test/agentkitfile-a2a.yaml image.
set +x
set -Eeuo pipefail
image="${1:?usage: scripts/a2a-container-smoke.sh IMAGE}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="agentkit-a2a-${RANDOM}-$$"
network="$run_id"
model="$run_id-model"
agent="$run_id-agent"
cleanup() {
  docker rm -f "$agent" "$model" >/dev/null 2>&1 || true
  docker network rm "$network" >/dev/null 2>&1 || true
}
trap cleanup EXIT
# Fixture credentials are public test-only values, never external provider keys.
docker network create --internal "$network" >/dev/null
docker run -d --name "$model" --network "$network" --network-alias model \
  -v "$repo_root/test/a2a:/smoke:ro" --entrypoint /opt/agentkit/bin/python \
  "$image" /smoke/model.py >/dev/null
docker run -d --name "$agent" --network "$network" --network-alias agent \
  -e AGENTKIT_PROTOCOL=a2a -e AGENTKIT_BIND=0.0.0.0 \
  -e AGENTKIT_A2A_URL=http://agent:8080/ -e AGENTKIT_AUTH_TOKEN=a2a-fixture-token \
  "$image" >/dev/null
docker run --rm --network "$network" \
  -e AGENTKIT_AUTH_TOKEN=a2a-fixture-token -v "$repo_root/test/a2a:/smoke:ro" \
  --entrypoint /opt/agentkit/bin/python "$image" /smoke/client.py
