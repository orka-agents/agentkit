#!/usr/bin/env bash

# Container parity smoke: run each runtime's built parity agent image under the
# openai, foundry, and orka protocols with both upstream Chat and Responses APIs
# against a scripted model and remote MCP tool. OpenAI mode checks both client
# routes on the same container. Checks upstream API selection, history,
# a plain answer, a tool round trip, and the normalized response to a
# 401 that echoes the model key, and that the key never appears in a response or
# container log. Needs the frontend and adapter images from `make build-agentkit
# build-serve build-serve-maf build-serve-langgraph` at the same TAG.

set -Eeuo pipefail

log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

tag="${TAG:-test}"
platform="${PLATFORM:-linux/amd64}"
builder="${BUILDER-desktop-linux}"
runtimes=("$@")
[[ ${#runtimes[@]} -gt 0 ]] || runtimes=(pydantic-ai microsoft-agent-framework langgraph)

model_key="sk-parity-container-canary"
auth_token="parity-smoke-token"
# A hung model call or SSE stream fails the scenario instead of the CI job.
request_timeout=120
network="agentkit-parity-$$"
fixture="agentkit-parity-fixture-$$"
containers=()

for command in curl docker jq make; do
  command -v "$command" >/dev/null 2>&1 || die "missing required command: $command"
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cleanup() {
  local name
  for name in ${containers[@]+"${containers[@]}"}; do
    docker rm -f "$name" >/dev/null 2>&1 || true
  done
  docker network rm "$network" >/dev/null 2>&1 || true
}
trap cleanup EXIT

serve_image() {
  case "$1" in
    pydantic-ai) echo "agentkit-serve:$tag" ;;
    microsoft-agent-framework) echo "agentkit-serve-maf:$tag" ;;
    langgraph) echo "agentkit-serve-langgraph:$tag" ;;
    *) die "unsupported runtime: $1" ;;
  esac
}

wait_for() {
  local url="$1"
  for _ in $(seq 1 90); do
    curl -fsS --max-time 5 "$url" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

# The parity agent's request to the model and MCP fixture leaves the agent
# container; everything checked here is what a client or operator can read.
assert_no_canary() {
  local label="$1" text="$2"
  if grep -qF "$model_key" <<<"$text"; then
    die "$label exposed the model key"
  fi
}

check() {
  local label="$1" filter="$2" body="$3"
  assert_no_canary "$label" "$body"
  jq -e "$filter" >/dev/null <<<"$body" || die "$label: unexpected response: $body"
}

check_history() {
  local label="$1" field="$2" body="$3" prompt="$4"
  check "$label" "($field | ltrimstr(\"parity-history: \" ) | fromjson) as \$turns |
    (\$turns | length) == 5 and
    \$turns[0][0] == \"system\" and (\$turns[0][1] | contains(\"Follow the scripted parity fixture.\")) and
    \$turns[1:] == [[\"system\",\"client system note\"],[\"user\",\"u1 edited\"],[\"assistant\",\"a1 edited\"],[\"user\",\"$prompt\"]]" "$body"
}

openai_scenarios() {
  local base="$1" label="$2" model_api="$3" out status endpoint payload token
  local headers=()
  local auth=(--max-time "$request_timeout" -H "Authorization: Bearer $auth_token" -H "Content-Type: application/json")
  out="$(curl -sS "${auth[@]}" "$base/v1/chat/completions" -d '{"messages":[{"role":"user","content":"plain:P1"}]}')"
  check "$label Chat plain" '.choices[0].message.content == "parity-answer: P1"' "$out"
  out="$(curl -sS "${auth[@]}" "$base/v1/chat/completions" -d '{"messages":[{"role":"user","content":"tool:T1"}]}')"
  check "$label Chat tool" '.choices[0].message.content | contains("receipt-T1")' "$out"
  out="$(curl -sS "${auth[@]}" "$base/v1/chat/completions" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"api:$model_api\"}]}")"
  check "$label Chat upstream API" ".choices[0].message.content == \"parity-api: $model_api\"" "$out"
  out="$(curl -sS "${auth[@]}" -H 'X-AgentKit-Session-Id: parity-history' "$base/v1/chat/completions" \
    -d '{"messages":[{"role":"user","content":"old history"}]}')"
  check "$label Chat initial history" '.choices[0].message.content == "parity-answer: old history"' "$out"
  out="$(curl -sS "${auth[@]}" -H 'X-AgentKit-Session-Id: parity-history' "$base/v1/chat/completions" \
    -d '{"messages":[{"role":"system","content":"client system note"},{"role":"user","content":"u1 edited"},{"role":"assistant","content":"a1 edited"},{"role":"user","content":"history:H1"}]}')"
  check_history "$label Chat history" '.choices[0].message.content' "$out" history:H1
  out="$(curl -sS -w '\n%{http_code}' "${auth[@]}" "$base/v1/chat/completions" -d '{"messages":[{"role":"user","content":"auth-echo"}]}')"
  status="${out##*$'\n'}"
  [[ "$status" == 503 ]] || die "$label Chat auth-echo returned HTTP $status"
  check "$label Chat auth-echo" '.error.code == "ModelAuthRejected"' "${out%$'\n'*}"

  # These Responses calls reuse the Chat container and its configured upstream API.
  out="$(curl -fsS "${auth[@]}" "$base/v1/responses" -d '{"input":"plain:R1","store":false}')"
  check "$label Responses plain" '.object == "response" and .status == "completed" and .store == false and
    (.output | length) == 1 and .output[0].type == "message" and .output[0].role == "assistant" and
    .output[0].status == "completed" and .output[0].content[0].type == "output_text" and
    .output[0].content[0].text == "parity-answer: R1" and
    (.usage.input_tokens | type) == "number" and (.usage.output_tokens | type) == "number" and
    (.usage.total_tokens | type) == "number"' "$out"
  out="$(curl -fsS "${auth[@]}" "$base/v1/responses" -d '{"input":"tool:R1"}')"
  check "$label Responses baked tool" '.output[0].content[0].text | contains("receipt-R1")' "$out"
  out="$(curl -fsS "${auth[@]}" "$base/v1/responses" -d "{\"input\":\"api:$model_api\"}")"
  check "$label Responses upstream API" ".output[0].content[0].text == \"parity-api: $model_api\"" "$out"
  out="$(curl -fsS "${auth[@]}" -H 'X-AgentKit-Session-Id: parity-history' "$base/v1/responses" \
    -d '{"input":"old Responses history"}')"
  check "$label Responses initial history" '.output[0].content[0].text == "parity-answer: old Responses history"' "$out"
  out="$(curl -fsS "${auth[@]}" -H 'X-AgentKit-Session-Id: parity-history' "$base/v1/responses" \
    -d '{"instructions":"client instructions","input":[
      {"role":"system","content":"client system note"},
      {"role":"developer","content":[{"type":"input_text","text":"client developer note"}]},
      {"role":"user","content":[{"type":"input_text","text":"u1 edited"}]},
      {"type":"message","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"a1 edited"}]},
      {"role":"user","content":[{"type":"input_text","text":"history:HR1"}]}
    ]}')"
  # $turns is a jq variable, not a shell expansion. Neither route retains old turns.
  # shellcheck disable=SC2016
  check "$label Responses history" '(.output[0].content[0].text | ltrimstr("parity-history: ") | fromjson) as $turns |
    ($turns | length) == 7 and $turns[0][0] == "system" and
    ($turns[0][1] | contains("Follow the scripted parity fixture.")) and
    $turns[1:] == [["system","client instructions"],["system","client system note"],
      ["system","client developer note"],["user","u1 edited"],["assistant","a1 edited"],["user","history:HR1"]]' "$out"
  out="$(curl -sS -w '\n%{http_code}' "${auth[@]}" "$base/v1/responses" -d '{"input":"auth-echo"}')"
  status="${out##*$'\n'}"
  [[ "$status" == 503 ]] || die "$label Responses auth-echo returned HTTP $status"
  check "$label Responses auth-echo" '.error.code == "ModelAuthRejected"' "${out%$'\n'*}"

  for endpoint in chat/completions responses; do
    if [[ "$endpoint" == responses ]]; then
      payload='{"input":"plain:unauthorized"}'
    else
      payload='{"messages":[{"role":"user","content":"plain:unauthorized"}]}'
    fi
    for token in '' invalid-token; do
      headers=(-H 'Content-Type: application/json')
      [[ -z "$token" ]] || headers+=(-H "Authorization: Bearer $token")
      out="$(curl -sS --max-time "$request_timeout" -w '\n%{http_code}' "${headers[@]}" "$base/v1/$endpoint" -d "$payload")"
      status="${out##*$'\n'}"
      [[ "$status" == 401 ]] || die "$label $endpoint client auth '${token:-missing}' returned HTTP $status"
      check "$label $endpoint client auth '${token:-missing}'" '.error.type == "invalid_request_error"' "${out%$'\n'*}"
    done
  done
}

foundry_scenarios() {
  local base="$1" label="$2" model_api="$3" out status
  local auth=(--max-time "$request_timeout" -H "Authorization: Bearer $auth_token" -H "Content-Type: application/json")
  out="$(curl -sS "${auth[@]}" "$base/responses" -d '{"input":"plain:P2"}')"
  check "$label plain" '.output[0].content[0].text == "parity-answer: P2"' "$out"
  out="$(curl -sS "${auth[@]}" "$base/responses" -d '{"input":"tool:T2"}')"
  check "$label tool" '.output[0].content[0].text | contains("receipt-T2")' "$out"
  out="$(curl -sS "${auth[@]}" "$base/responses" -d "{\"input\":\"api:$model_api\"}")"
  check "$label upstream API" ".output[0].content[0].text == \"parity-api: $model_api\"" "$out"
  out="$(curl -sS "${auth[@]}" "$base/responses" \
    -d '{"input":[{"role":"system","content":"client system note"},{"role":"user","content":"u1 edited"},{"role":"assistant","content":"a1 edited"},{"role":"user","content":"history:H2"}]}')"
  check_history "$label history" '.output[0].content[0].text' "$out" history:H2
  out="$(curl -sS -w '\n%{http_code}' "${auth[@]}" "$base/responses" -d '{"input":"auth-echo"}')"
  status="${out##*$'\n'}"
  [[ "$status" == 503 ]] || die "$label auth-echo returned HTTP $status"
  check "$label auth-echo" '.error.code == "ModelAuthRejected" and .error.upstream_status == 401' "${out%$'\n'*}"
}

orka_turn() {
  local base="$1" turn="$2" prompt="$3" deadline payload
  deadline="$(date -u -d '+5 minutes' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v+5M +%Y-%m-%dT%H:%M:%SZ)"
  payload="$(jq -n --arg turn "$turn" --arg prompt "$prompt" --arg deadline "$deadline" '{
    version: "orka.harness.v1", namespace: "default", taskName: "parity", sessionName: "parity",
    runtimeSessionID: "parity-session", turnID: $turn, correlationID: ("corr-" + $turn),
    deadline: $deadline, authIdentity: {subject: "system:serviceaccount:default:orka"},
    input: {prompt: $prompt, contextRefs: [], env: []}, toolExecutionMode: "observed", metadata: {}
  }')"
  curl -fsS --max-time "$request_timeout" -H "Authorization: Bearer $auth_token" -H "Content-Type: application/json" \
    "$base/v1/turns" -d "$payload" >/dev/null
  curl -fsS --max-time "$request_timeout" -H "Authorization: Bearer $auth_token" "$base/v1/turns/$turn/events" |
    sed -n 's/^data: //p' | tail -n 1
}

orka_scenarios() {
  local base="$1" label="$2" model_api="$3" out
  check "$label plain" '.type == "TurnCompleted" and .completed.result == "parity-answer: P3"' "$(orka_turn "$base" turn-plain plain:P3)"
  out="$(orka_turn "$base" turn-history history:H3)"
  # $turns is a jq variable, not a shell expansion.
  # shellcheck disable=SC2016
  check "$label history" '.type == "TurnCompleted" and
    ((.completed.result | ltrimstr("parity-history: ") | fromjson) as $turns |
    ($turns | length) == 4 and $turns[0][0] == "system" and
    ($turns[0][1] | contains("Follow the scripted parity fixture.")) and
    $turns[1:] == [["user","plain:P3"],["assistant","parity-answer: P3"],["user","history:H3"]])' "$out"
  check "$label upstream API" ".type == \"TurnCompleted\" and .completed.result == \"parity-api: $model_api\"" "$(orka_turn "$base" turn-api "api:$model_api")"
  check "$label tool" '.type == "TurnCompleted" and (.completed.result | contains("receipt-T3"))' "$(orka_turn "$base" turn-tool tool:T3)"
  check "$label auth-echo" '.type == "TurnFailed" and .failed.reason == "ModelAuthRejected"' "$(orka_turn "$base" turn-auth auth-echo)"
}

run_protocol() {
  local runtime="$1" protocol="$2" model_api="$3" image="parity-$1:$tag"
  local name="agentkit-parity-$runtime-$protocol-$model_api-$$" health base port
  case "$protocol" in
    openai) health=/healthz ;;
    foundry) health=/readiness ;;
    orka) health=/v1/health ;;
  esac
  containers+=("$name")
  docker run -d --name "$name" --platform "$platform" --network "$network" \
    -p 127.0.0.1::8080 \
    -e AGENTKIT_PROTOCOL="$protocol" \
    -e AGENTKIT_MODEL_API="$model_api" \
    -e AGENTKIT_PORT=8080 \
    -e AGENTKIT_BIND=0.0.0.0 \
    -e AGENTKIT_AUTH_TOKEN="$auth_token" \
    -e PARITY_MODEL_KEY="$model_key" \
    -e PARITY_MCP_URL="http://parity-fixture:8090/mcp" \
    "$image" >/dev/null
  port="$(docker port "$name" 8080/tcp | head -n 1)"
  base="http://127.0.0.1:${port##*:}"
  wait_for "$base$health" || { docker logs "$name" >&2 || true; die "$runtime $protocol $model_api never became healthy"; }
  log "Checking $runtime under $protocol with upstream $model_api"
  "${protocol}_scenarios" "$base" "$runtime $protocol $model_api" "$model_api"
  assert_no_canary "$runtime $protocol $model_api container log" "$(docker logs "$name" 2>&1)"
  docker rm -f "$name" >/dev/null
}

main() {
  local runtime protocol model_api fixture_port
  docker network create "$network" >/dev/null
  log "Starting scripted model and MCP fixture"
  containers+=("$fixture")
  # The parity Agentkitfiles bake http://parity-fixture:8090 as the model URL.
  docker run -d --name "$fixture" --platform "$platform" --network "$network" \
    --network-alias parity-fixture \
    -p 127.0.0.1::8090 \
    -v "$repo_root/test/parity/fixture.py:/parity/fixture.py:ro" \
    --entrypoint /opt/agentkit/bin/python \
    "agentkit-serve:$tag" /parity/fixture.py >/dev/null
  fixture_port="$(docker port "$fixture" 8090/tcp | head -n 1)"
  wait_for_fixture "http://127.0.0.1:${fixture_port##*:}" || { docker logs "$fixture" >&2 || true; die "fixture never started"; }

  for runtime in "${runtimes[@]}"; do
    log "Building parity agent image for $runtime"
    make build-test-agent TAG="$tag" BUILDER="$builder" PLATFORM="$platform" RUNTIME="$runtime" \
      SERVE_IMAGE="$(serve_image "$runtime")" FIXTURE="test/parity/agentkitfile-$runtime.yaml" \
      AGENT_IMAGE="parity-$runtime:$tag"
    for model_api in chat_completions responses; do
      for protocol in openai foundry orka; do
        run_protocol "$runtime" "$protocol" "$model_api"
      done
    done
  done
  log "Container parity smoke passed for chat_completions and responses: ${runtimes[*]}"
}

# The fixture answers only POSTs; any HTTP response means it is listening.
wait_for_fixture() {
  local url="$1"
  for _ in $(seq 1 60); do
    curl -sS --max-time 5 -o /dev/null "$url/v1/chat/completions" 2>/dev/null && return 0
    sleep 1
  done
  return 1
}

main
