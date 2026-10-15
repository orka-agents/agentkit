#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

platform="${PLATFORM:-}"
if [[ -z "$platform" ]]; then
  case "$(docker version --format '{{.Server.Arch}}')" in
    amd64 | x86_64) platform="linux/amd64" ;;
    arm64 | aarch64) platform="linux/arm64" ;;
    *)
      printf 'unsupported Docker server architecture\n' >&2
      exit 1
      ;;
  esac
fi

tag="${TAG:-spike}"
harness_image="${HARNESS_IMAGE:-agentkit-harness-pydantic-ai:${tag}}"
bases=(
  "ubuntu:24.04"
  "fedora:42"
  "registry.access.redhat.com/ubi9/ubi-minimal:latest"
  "gcr.io/distroless/base-debian12:nonroot"
)
names=(
  "ubuntu"
  "fedora"
  "ubi"
  "distroless"
)
runtime_uids=(
  "1000"
  "1000"
  "1000"
  "65532"
)
runtime_gids=(
  "1000"
  "1000"
  "1000"
  "65532"
)
harness_library_paths=(
  ""
  ""
  ""
  "/opt/agentkit/lib"
)
trust_tests=(
  "1"
  "1"
  "1"
  "0"
)

tls_dir="$(mktemp -d)"
tls_server_pid=""
active_container=""

cleanup() {
  if [[ -n "$active_container" ]]; then
    docker rm -f "$active_container" >/dev/null 2>&1 || true
  fi
  if [[ -n "$tls_server_pid" ]]; then
    kill "$tls_server_pid" >/dev/null 2>&1 || true
  fi
  rm -rf "$tls_dir"
}
trap cleanup EXIT

cat >"$tls_dir/ca.conf" <<'EOF'
[req]
distinguished_name = subject
x509_extensions = extensions
prompt = no
[subject]
CN = AgentKit portable harness test CA
[extensions]
basicConstraints = critical,CA:TRUE
keyUsage = critical,keyCertSign,cRLSign
subjectKeyIdentifier = hash
EOF

cat >"$tls_dir/server.conf" <<'EOF'
[req]
distinguished_name = subject
req_extensions = extensions
prompt = no
[subject]
CN = host.docker.internal
[extensions]
subjectAltName = DNS:host.docker.internal
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
EOF

openssl req -x509 -newkey rsa:2048 -nodes \
  -config "$tls_dir/ca.conf" \
  -days 1 \
  -keyout "$tls_dir/ca.key" \
  -out "$tls_dir/ca.crt" >/dev/null 2>&1
openssl req -newkey rsa:2048 -nodes \
  -config "$tls_dir/server.conf" \
  -keyout "$tls_dir/server.key" \
  -out "$tls_dir/server.csr" >/dev/null 2>&1
openssl x509 -req \
  -in "$tls_dir/server.csr" \
  -CA "$tls_dir/ca.crt" \
  -CAkey "$tls_dir/ca.key" \
  -CAcreateserial \
  -days 1 \
  -extfile "$tls_dir/server.conf" \
  -extensions extensions \
  -out "$tls_dir/server.crt" >/dev/null 2>&1

python3 test/portable-harness/https_server.py \
  --cert "$tls_dir/server.crt" \
  --key "$tls_dir/server.key" \
  --port-file "$tls_dir/port" &
tls_server_pid=$!
for _ in $(seq 1 30); do
  [[ -s "$tls_dir/port" ]] && break
  sleep 0.1
done
tls_port="$(cat "$tls_dir/port")"

docker buildx build . \
  -f test/portable-harness/Harness.Dockerfile \
  --platform "$platform" \
  -t "$harness_image" \
  --load

for i in "${!bases[@]}"; do
  base="${bases[$i]}"
  name="${names[$i]}"
  runtime_uid="${runtime_uids[$i]}"
  runtime_gid="${runtime_gids[$i]}"
  harness_library_path="${harness_library_paths[$i]}"
  image="agentkit-portable-${name}:${tag}"
  trust_base="agentkit-portable-trust-base-${name}:${tag}"
  trust_image="agentkit-portable-trust-${name}:${tag}"

  printf '\n==> composing %s on %s\n' "$harness_image" "$base"
  docker buildx build . \
    -f test/portable-harness/Compose.Dockerfile \
    --build-arg "BASE_IMAGE=${base}" \
    --build-arg "HARNESS_IMAGE=${harness_image}" \
    --build-arg "HARNESS_LIBRARY_PATH=${harness_library_path}" \
    --build-arg "RUNTIME_UID=${runtime_uid}" \
    --build-arg "RUNTIME_GID=${runtime_gid}" \
    --platform "$platform" \
    -t "$image" \
    --load

  docker run --rm --platform "$platform" \
    --entrypoint /opt/agentkit/runtime/python/bin/python3 \
    "$image" \
    -c "import os, agentkit_serve, agentkit_serve_common, pydantic_ai; assert os.getuid() == ${runtime_uid}"

  docker run --rm --platform "$platform" \
    --entrypoint /opt/agentkit/bin/node \
    "$image" --version
  docker run --rm --platform "$platform" \
    --entrypoint /opt/agentkit/bin/npm \
    "$image" --version
  docker run --rm --platform "$platform" \
    --entrypoint /opt/agentkit/bin/npx \
    "$image" --version
  docker run --rm --platform "$platform" \
    --entrypoint /opt/agentkit/bin/uv \
    "$image" --version
  docker run --rm --platform "$platform" \
    --entrypoint /opt/agentkit/bin/uvx \
    "$image" --version
  docker run --rm --platform "$platform" \
    --entrypoint /opt/agentkit/bin/sh \
    "$image" -c 'test "$SHELL" = /opt/agentkit/bin/sh'
  printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' |
    docker run --rm -i --platform "$platform" \
      --entrypoint /opt/agentkit/bin/sh \
      "$image" \
      -c 'npm install --offline --ignore-scripts --prefix /tmp/node-tool /opt/agentkit/tool-fixtures/node >/dev/null && exec npx --offline -- /tmp/node-tool/node_modules/.bin/agentkit-portable-node-tool' |
    grep -q '"protocolVersion":"2025-06-18"'
  printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' |
    docker run --rm -i --platform "$platform" \
      --entrypoint /opt/agentkit/bin/sh \
      "$image" \
      -c 'uv tool install --offline --python /opt/agentkit/runtime/python/bin/python3 /opt/agentkit/tool-fixtures/agentkit_portable_python_tool-0.0.0-py3-none-any.whl >/dev/null && exec uvx --offline --python /opt/agentkit/runtime/python/bin/python3 agentkit-portable-python-tool' |
    grep -q '"protocolVersion": "2025-06-18"'

  active_container="$(
    docker run -d --rm --platform "$platform" \
      -p 127.0.0.1::8080 \
      -e AGENTKIT_BIND=0.0.0.0 \
      -e AGENTKIT_AUTH_TOKEN=portable-harness-smoke \
      "$image"
  )"
  host_port="$(
    docker port "$active_container" 8080/tcp |
      awk -F: '/127[.]0[.]0[.]1:/ { print $NF; exit }'
  )"
  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${host_port}/healthz" >/dev/null; then
      break
    fi
    sleep 1
  done
  curl -fsS "http://127.0.0.1:${host_port}/healthz"
  curl -fsS \
    -H 'Authorization: Bearer portable-harness-smoke' \
    "http://127.0.0.1:${host_port}/v1/models"
  docker rm -f "$active_container" >/dev/null
  active_container=""

  active_container="$(
    docker run -d --rm --platform "$platform" \
      -p 127.0.0.1::8080 \
      -e AGENTKIT_BIND=0.0.0.0 \
      -e AGENTKIT_AUTH_TOKEN=portable-harness-smoke \
      -e AGENTKIT_ORKA_OFFLINE_ECHO=1 \
      -e AGENTKIT_PROTOCOL=orka \
      "$image"
  )"
  host_port="$(
    docker port "$active_container" 8080/tcp |
      awk -F: '/127[.]0[.]0[.]1:/ { print $NF; exit }'
  )"
  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${host_port}/v1/health" >/dev/null; then
      break
    fi
    sleep 1
  done
  curl -fsS \
    -H 'Authorization: Bearer portable-harness-smoke' \
    -H 'Content-Type: application/json' \
    -d '{
      "version":"orka.harness.v1",
      "namespace":"default",
      "taskName":"portable-harness-smoke",
      "sessionName":"portable-harness-smoke",
      "runtimeSessionID":"runtime-session-1",
      "turnID":"turn-1",
      "correlationID":"correlation-1",
      "deadline":"2099-01-01T00:00:00Z",
      "authIdentity":{"subject":"portable-harness-smoke"},
      "input":{"prompt":"hello","contextRefs":[],"env":[]},
      "toolExecutionMode":"observed",
      "metadata":{}
    }' \
    "http://127.0.0.1:${host_port}/v1/turns" >/dev/null
  events="$(
    curl -fsS --max-time 20 \
      -H 'Authorization: Bearer portable-harness-smoke' \
      "http://127.0.0.1:${host_port}/v1/turns/turn-1/events"
  )"
  grep -q '"type":"TurnCompleted"' <<<"$events"
  grep -q 'offline echo: hello' <<<"$events"
  docker rm -f "$active_container" >/dev/null
  active_container=""

  if [[ "${trust_tests[$i]}" == "1" ]]; then
    docker buildx build "$tls_dir" \
      -f "$repo_root/test/portable-harness/TrustBase.Dockerfile" \
      --build-arg "BASE_IMAGE=${base}" \
      --platform "$platform" \
      -t "$trust_base" \
      --load
    docker buildx build . \
      -f test/portable-harness/Compose.Dockerfile \
      --build-arg "BASE_IMAGE=${trust_base}" \
      --build-arg "HARNESS_IMAGE=${harness_image}" \
      --build-arg "HARNESS_LIBRARY_PATH=${harness_library_path}" \
      --build-arg "RUNTIME_UID=${runtime_uid}" \
      --build-arg "RUNTIME_GID=${runtime_gid}" \
      --platform "$platform" \
      -t "$trust_image" \
      --load
    docker run --rm --platform "$platform" \
      --add-host host.docker.internal:host-gateway \
      --entrypoint /opt/agentkit/bin/agentkit-trust-smoke \
      "$trust_image" \
      "https://host.docker.internal:${tls_port}/"
  fi

  if [[ "${PORTABLE_HARNESS_NETWORK_TOOLS:-0}" == "1" ]]; then
    docker run --rm --platform "$platform" \
      --entrypoint /opt/agentkit/bin/npx \
      "$image" -y @modelcontextprotocol/server-everything --help
    docker run --rm --platform "$platform" \
      --entrypoint /opt/agentkit/bin/uvx \
      "$image" mcp-server-fetch --help
  fi
done
