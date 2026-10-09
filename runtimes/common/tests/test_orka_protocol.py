from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any

import pytest
from fastapi.testclient import TestClient

import agentkit_serve_common.orka as orka_module
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import ConversationTurn, RunRequest
from agentkit_serve_common.orka import ORKA_HARNESS_VERSION, create_orka_app
from agentkit_serve_common.runtime import (
    AgentRunError,
    BrokeredToolCall,
    BrokeredToolDefinition,
    BrokeredToolResult,
    OfflineEchoRuntimeFactory,
    RunResult,
    RuntimeSession,
    ToolBroker,
)

AUTH = {"authorization": "Bearer test-token"}
TERMINAL_TYPES = {"TurnCompleted", "TurnFailed", "TurnCancelled"}
EXPECTED_MAX_OUTPUT_BYTES = 512 * 1024
ORKA_CLIENT_MAX_SSE_TOKEN_BYTES = 1 << 20


def _deadline() -> str:
    return (datetime.now(UTC) + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "orka-test"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "env": [{"name": "MODEL_TOKEN"}],
            "expose": {"openai": True, "port": 8080},
        }
    )


class EchoRuntime:
    def __init__(self, *, delay: float = 0, delays: dict[str, float] | None = None) -> None:
        self.requests: list[RunRequest] = []
        self.delay = delay
        self.delays = delays or {}

    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None

    async def run(self, request: RunRequest) -> RunResult:
        import asyncio

        self.requests.append(request)
        delay = self.delays.get(request.prompt, self.delay)
        if delay:
            await asyncio.sleep(delay)
        return RunResult(text=f"echo: {request.prompt}", usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})


class EchoFactory:
    def __init__(self, *, delay: float = 0, delays: dict[str, float] | None = None) -> None:
        self.runtime = EchoRuntime(delay=delay, delays=delays)

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


class StaticOutputRuntime:
    def __init__(self, text: str) -> None:
        self.text = text

    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None

    async def run(self, request: RunRequest) -> RunResult:
        return RunResult(text=self.text)


class StaticOutputFactory:
    def __init__(self, text: str) -> None:
        self.runtime = StaticOutputRuntime(text)

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


class RaisingRuntime(StaticOutputRuntime):
    # Runtime-owned AgentRunError text is the failure detail frames still carry.
    async def run(self, request: RunRequest) -> RunResult:
        raise AgentRunError(self.text)


class RaisingFactory(StaticOutputFactory):
    def __init__(self, message: str) -> None:
        self.runtime = RaisingRuntime(message)


def _start_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": ORKA_HARNESS_VERSION,
        "namespace": "default",
        "taskName": "task-1",
        "sessionName": "session-1",
        "runtimeSessionID": "runtime-session-1",
        "turnID": "turn-1",
        "correlationID": "corr-1",
        "deadline": _deadline(),
        "authIdentity": {"subject": "system:serviceaccount:default:orka"},
        "input": {"prompt": "hello", "contextRefs": [], "env": []},
        "toolExecutionMode": "observed",
        "metadata": {},
    }
    payload.update(overrides)
    return payload


def _cancel_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "version": ORKA_HARNESS_VERSION,
        "namespace": "default",
        "taskName": "task-1",
        "sessionName": "session-1",
        "runtimeSessionID": "runtime-session-1",
        "turnID": "turn-1",
        "correlationID": "corr-1",
        "reason": "test requested cancel",
    }
    payload.update(overrides)
    return payload


def _continue_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "version": ORKA_HARNESS_VERSION,
        "namespace": "default",
        "taskName": "task-1",
        "sessionName": "session-1",
        "runtimeSessionID": "runtime-session-1",
        "turnID": "turn-1",
        "correlationID": "corr-1",
        "toolResults": [
            {
                "version": ORKA_HARNESS_VERSION,
                "runtimeSessionID": "runtime-session-1",
                "turnID": "turn-1",
                "toolCallID": "tool-call-1",
                "idempotencyKey": "runtime-session-1:turn-1:tool-call-1",
                "approved": True,
                "output": {"success": True, "data": {"answer": "ok"}},
            }
        ],
    }
    payload.update(overrides)
    return payload


def _brokered_input(**overrides: Any) -> dict[str, Any]:
    value = {
        "prompt": "hello",
        "contextRefs": [],
        "env": [],
        "tools": [
            {
                "name": "conformance_read",
                "description": "Synthetic conformance read tool",
                "brokeredClass": "read",
                "parameters": {"type": "object"},
            }
        ],
    }
    value.update(overrides)
    return value


def _brokered_write_input(**overrides: Any) -> dict[str, Any]:
    return _brokered_input(
        tools=[
            {
                "name": "conformance_write",
                "description": "Synthetic conformance write tool",
                "brokeredClass": "write",
                "parameters": {"type": "object"},
            }
        ],
        **overrides,
    )


def _brokered_coordination_input(**overrides: Any) -> dict[str, Any]:
    return _brokered_input(
        tools=[
            {
                "name": "conformance_coordination",
                "description": "Synthetic conformance coordination tool",
                "brokeredClass": "coordination",
                "parameters": {"type": "object"},
            }
        ],
        **overrides,
    )


def _wait_for_event_type(client: TestClient, turn_id: str, event_type: str) -> dict[str, Any]:
    for _ in range(100):
        events = client.app.state.turns[turn_id].events
        for event in events:
            frame = event.as_frame()
            if frame["type"] == event_type:
                return frame
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {event_type}")


def _wait_for_tool_call_id(client: TestClient, turn_id: str, tool_call_id: str) -> dict[str, Any]:
    for _ in range(100):
        events = client.app.state.turns[turn_id].events
        for event in events:
            frame = event.as_frame()
            if frame["type"] == "ToolCallRequested" and frame["toolCallID"] == tool_call_id:
                return frame
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for tool call {tool_call_id}")


def _frames(resp_text: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for raw in resp_text.strip().split("\n\n"):
        if not raw:
            continue
        data_lines = [line.removeprefix("data: ") for line in raw.splitlines() if line.startswith("data: ")]
        assert len(data_lines) == 1, raw
        frames.append(json.loads(data_lines[0]))
    return frames


def _assert_sse_lines_fit_orka_client(raw: bytes) -> None:
    data_lines = [line for line in raw.splitlines() if line.startswith(b"data: ")]
    assert data_lines
    assert max(map(len, data_lines)) < ORKA_CLIENT_MAX_SSE_TOKEN_BYTES


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _json_object_with_size(size: int) -> dict[str, str]:
    empty = _compact_json_bytes({"value": ""})
    remaining = size - len(empty)
    assert remaining >= 0
    value = "é" * (remaining // len("é".encode())) + "x" * (remaining % len("é".encode()))
    output = {"value": value}
    assert len(_compact_json_bytes(output)) == size
    return output


def _create_turn(client: TestClient, **overrides: Any) -> str:
    payload = _start_payload(**overrides)
    resp = client.post("/v1/turns", json=payload, headers=AUTH)
    assert resp.status_code == 202, resp.text
    assert resp.json() == {
        "version": ORKA_HARNESS_VERSION,
        "accepted": True,
        "runtimeSessionID": payload["runtimeSessionID"],
        "turnID": payload["turnID"],
        "correlationID": payload["correlationID"],
        "eventStreamPath": f"/v1/turns/{payload['turnID']}/events",
    }
    return payload["turnID"]


def _assert_frame_identity(frame: dict[str, Any], *, seq: int, typ: str, runtime_session_id: str = "runtime-session-1", turn_id: str = "turn-1", correlation_id: str = "corr-1") -> None:
    assert frame["version"] == ORKA_HARNESS_VERSION
    assert frame["type"] == typ
    assert frame["runtimeSessionID"] == runtime_session_id
    assert frame["turnID"] == turn_id
    assert frame["correlationID"] == correlation_id
    assert frame["seq"] == seq
    assert frame["severity"] in {"info", "error"}
    assert isinstance(frame["summary"], str)
    assert isinstance(frame["metadata"], dict)
    assert "createdAt" in frame
    assert "timestamp" not in frame
    assert "payload" not in frame


def test_orka_app_factory_requires_auth_token():
    with pytest.raises(ValueError, match="requires a bearer auth token"):
        create_orka_app(_spec(), EchoFactory())


def test_orka_health_and_capabilities_are_open_and_match_contract():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        health = client.get("/v1/health")
        caps = client.get("/v1/capabilities")

    assert health.status_code == 200
    health_body = health.json()
    assert set(health_body) == {"version", "status", "ready", "checkedAt", "metadata"}
    assert health_body["version"] == ORKA_HARNESS_VERSION
    assert health_body["status"] == "ok"
    assert health_body["ready"] is True
    assert health_body["metadata"] == {"agentName": "orka-test"}
    assert datetime.fromisoformat(health_body["checkedAt"].replace("Z", "+00:00"))

    assert caps.status_code == 200
    assert caps.json() == {
        "version": ORKA_HARNESS_VERSION,
        "protocolVersion": ORKA_HARNESS_VERSION,
        "transport": "http+sse",
        "runtimeName": "agentkit-serve",
        "runtimeVersion": "0.0.0",
        "providerKind": "kubernetes-service",
        "toolExecutionModes": ["observed"],
        "supportsCancel": True,
        "supportsRuntimeSessions": True,
        "supportsSuspend": False,
        "supportsWorkspaceSnapshot": False,
        "maxConcurrentTurns": 1,
        "maxOutputBytes": EXPECTED_MAX_OUTPUT_BYTES,
        "metadata": {"agentName": "orka-test", "model": "gpt-4o-mini", "agentkitProvider": "openai-compatible"},
    }


def test_orka_turn_lifecycle_streams_contract_frames_and_one_terminal():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client)
        resp = client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)

    assert resp.status_code == 200
    frames = _frames(resp.text)
    assert [frame["type"] for frame in frames] == ["TurnStarted", "RuntimeOutput", "TurnCompleted"]
    _assert_frame_identity(frames[0], seq=1, typ="TurnStarted")
    _assert_frame_identity(frames[1], seq=2, typ="RuntimeOutput")
    assert frames[1]["contentText"] == "echo: hello"
    assert frames[1]["content"] == {
        "usage": {"completion_tokens": 2, "prompt_tokens": 1, "total_tokens": 3},
    }
    _assert_frame_identity(frames[2], seq=3, typ="TurnCompleted")
    assert frames[2]["completed"] == {"result": "echo: hello", "finalEventSeq": 3}
    assert frames[2]["failed"] is None
    assert frames[2]["error"] is None
    terminals = [frame for frame in frames if frame["type"] in TERMINAL_TYPES]
    assert len(terminals) == 1


def test_orka_events_support_after_seq_replay():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client)
        all_events = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        replay = client.get(f"/v1/turns/{turn_id}/events?afterSeq=1", headers=AUTH)

    assert [frame["seq"] for frame in all_events] == [1, 2, 3]
    assert replay.status_code == 200
    assert [frame["type"] for frame in _frames(replay.text)] == ["RuntimeOutput", "TurnCompleted"]


def test_orka_observed_output_uses_utf8_bytes_accepts_exact_limit_and_replays_safely():
    output = "é" * (EXPECTED_MAX_OUTPUT_BYTES // len("é".encode()))
    assert len(output.encode()) == EXPECTED_MAX_OUTPUT_BYTES
    app = create_orka_app(_spec(), StaticOutputFactory(output), AUTH["authorization"].removeprefix("Bearer "))

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-output-boundary")
        response = client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)
        replay = client.get(f"/v1/turns/{turn_id}/events?afterSeq=1", headers=AUTH)

    frames = _frames(response.text)
    assert [frame["type"] for frame in frames] == ["TurnStarted", "RuntimeOutput", "TurnCompleted"]
    assert frames[1]["contentText"] == output
    assert frames[1]["content"] == {"usage": {}}
    assert frames[2]["completed"]["result"] == output
    assert [frame["type"] for frame in _frames(replay.text)] == ["RuntimeOutput", "TurnCompleted"]
    _assert_sse_lines_fit_orka_client(response.content)
    _assert_sse_lines_fit_orka_client(replay.content)


def test_orka_observed_output_over_utf8_limit_fails_without_retaining_payload_and_replays_terminal():
    output = "é" * (EXPECTED_MAX_OUTPUT_BYTES // len("é".encode())) + "x"
    output_bytes = len(output.encode())
    assert output_bytes == EXPECTED_MAX_OUTPUT_BYTES + 1
    app = create_orka_app(_spec(), StaticOutputFactory(output), AUTH["authorization"].removeprefix("Bearer "))

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-output-over-limit")
        response = client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)
        replay = client.get(f"/v1/turns/{turn_id}/events?afterSeq=1", headers=AUTH)
        retained = client.app.state.turns[turn_id].events

    frames = _frames(response.text)
    assert [frame["type"] for frame in frames] == ["TurnStarted", "TurnFailed"]
    assert frames[-1]["failed"] == {
        "reason": "MaxOutputBytesExceeded",
        "message": (
            f"runtime output is {output_bytes} UTF-8 bytes; "
            f"maxOutputBytes is {EXPECTED_MAX_OUTPUT_BYTES}"
        ),
        "retryable": False,
    }
    assert [frame["type"] for frame in _frames(replay.text)] == ["TurnFailed"]
    assert all(event.content_text != output for event in retained)
    assert all(event.completed is None or event.completed.get("result") != output for event in retained)
    _assert_sse_lines_fit_orka_client(response.content)
    _assert_sse_lines_fit_orka_client(replay.content)


def test_orka_observed_output_that_expands_past_scanner_limit_fails_with_visible_terminal():
    output = "\x00" * 200_000
    assert len(output.encode()) < EXPECTED_MAX_OUTPUT_BYTES
    app = create_orka_app(_spec(), StaticOutputFactory(output), AUTH["authorization"].removeprefix("Bearer "))

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-output-json-expansion")
        response = client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)

    frames = _frames(response.text)
    assert [frame["type"] for frame in frames] == ["TurnStarted", "TurnFailed"]
    assert frames[-1]["failed"]["reason"] == "HarnessFrameTooLarge"
    assert "Orka client limit is 1048575" in frames[-1]["failed"]["message"]
    _assert_sse_lines_fit_orka_client(response.content)


def test_orka_observed_output_with_unpaired_surrogate_fails_with_visible_terminal():
    app = create_orka_app(
        _spec(),
        StaticOutputFactory("\ud800"),
        AUTH["authorization"].removeprefix("Bearer "),
    )

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-output-invalid-utf8")
        response = client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)

    frames = _frames(response.text)
    assert [frame["type"] for frame in frames] == ["TurnStarted", "TurnFailed"]
    assert frames[-1]["failed"] == {
        "reason": "InvalidOutputEncoding",
        "message": "runtime output is not valid UTF-8",
        "retryable": False,
    }
    _assert_sse_lines_fit_orka_client(response.content)


@pytest.mark.parametrize(
    "message",
    [pytest.param("\x00" * 200_000, id="oversized"), pytest.param("\ud800", id="invalid-utf8")],
)
def test_orka_runtime_failure_with_unstreamable_detail_uses_bounded_terminal_fallback(message: str):
    app = create_orka_app(
        _spec(),
        RaisingFactory(message),
        AUTH["authorization"].removeprefix("Bearer "),
    )

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-unstreamable-failure-detail")
        response = client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)

    frames = _frames(response.text)
    assert [frame["type"] for frame in frames] == ["TurnStarted", "TurnFailed"]
    assert frames[-1]["failed"] == {
        "reason": "TerminalFrameRejected",
        "message": "terminal failure details could not be emitted safely",
        "retryable": False,
    }
    _assert_sse_lines_fit_orka_client(response.content)


def test_orka_turn_state_rejects_nonterminal_events_after_terminal():
    async def exercise() -> list[str]:
        state = orka_module.TurnState(
            runtime_session_id="runtime-session-1",
            turn_id="turn-terminal-finality",
            correlation_id="corr-1",
        )
        await state.append("TurnStarted", summary="turn started")
        await state.append(
            "TurnFailed",
            summary="turn failed",
            failed={"reason": "TestFailure", "message": "failed", "retryable": False},
        )
        with pytest.raises(orka_module.AgentRunError, match="already terminal"):
            await state.append("RuntimeOutput", summary="late output", content_text="late")
        return [event.type for event in state.events]

    assert asyncio.run(exercise()) == ["TurnStarted", "TurnFailed"]


def test_orka_rejects_an_unstreamable_start_frame_without_retaining_a_poisoned_turn():
    app = create_orka_app(_spec(), EchoFactory(), AUTH["authorization"].removeprefix("Bearer "))

    with TestClient(app, raise_server_exceptions=False) as client:
        rejected = client.post(
            "/v1/turns",
            json=_start_payload(turnID="turn-unstreamable-start", metadata={"large": "\x00" * 200_000}),
            headers=AUTH,
        )
        accepted = client.post(
            "/v1/turns",
            json=_start_payload(turnID="turn-after-unstreamable-start"),
            headers=AUTH,
        )

    assert rejected.status_code == 413
    assert "TurnStarted SSE data line" in rejected.text
    assert accepted.status_code == 202
    assert "turn-unstreamable-start" not in app.state.turns


def test_orka_rejects_start_frame_text_that_is_not_valid_utf8_without_retaining_turn():
    app = create_orka_app(_spec(), EchoFactory(), AUTH["authorization"].removeprefix("Bearer "))
    payload = _start_payload(turnID="turn-invalid-utf8-start", metadata={"invalid": "\ud800"})

    with TestClient(app, raise_server_exceptions=False) as client:
        rejected = client.post(
            "/v1/turns",
            content=json.dumps(payload, ensure_ascii=True).encode(),
            headers={**AUTH, "content-type": "application/json"},
        )

    assert rejected.status_code == 400
    assert rejected.json() == {"error": "TurnStarted contains text that is not valid UTF-8"}
    assert "turn-invalid-utf8-start" not in app.state.turns


def test_orka_duplicate_turn_rejection_matches_orka_conformance_contract():
    app = create_orka_app(_spec(), EchoFactory(delay=60), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-duplicate", input={"prompt": "slow", "contextRefs": [], "env": []})
        duplicate = client.post("/v1/turns", json=_start_payload(turnID=turn_id, input={"prompt": "slow", "contextRefs": [], "env": []}), headers=AUTH)
        cancel = client.post(f"/v1/turns/{turn_id}/cancel", json=_cancel_payload(turnID=turn_id), headers=AUTH)
        _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        completed_duplicate = client.post("/v1/turns", json=_start_payload(turnID=turn_id, input={"prompt": "again", "contextRefs": [], "env": []}), headers=AUTH)

    # Orka's client matches these exact native {"error": ...} bodies.
    assert duplicate.status_code == 409
    assert duplicate.json() == {"error": "turn already exists"}
    assert cancel.status_code == 202
    assert completed_duplicate.status_code == 409
    assert completed_duplicate.json() == {"error": "turn already completed"}


def test_orka_parameter_validation_uses_native_error_body():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        negative = client.get("/v1/turns/turn-1/events?afterSeq=-1", headers=AUTH)
        malformed = client.get("/v1/turns/turn-1/events?afterSeq=private-value", headers=AUTH)
        missing = client.get("/v1/turns/turn-1/output", headers=AUTH)
        unauthenticated = client.get("/v1/turns/turn-1/events?afterSeq=-1")

    assert (negative.status_code, negative.json()) == (400, {"error": "afterSeq: Input should be greater than or equal to 0"})
    assert malformed.status_code == 400
    assert malformed.json()["error"].startswith("afterSeq: ")
    assert "private-value" not in malformed.text
    assert (missing.status_code, missing.json()) == (400, {"error": "ref: Field required"})
    assert unauthenticated.status_code == 401


def test_orka_turn_forwards_per_turn_metadata_env_and_session_fields():
    factory = EchoFactory()
    app = create_orka_app(_spec(), factory, auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-meta",
            runtimeSessionID="runtime-session-meta",
            correlationID="corr-meta",
            input={"prompt": "hello", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "per-run"}]},
            metadata={"tenant": "acme"},
        )
        client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)

    request = factory.runtime.requests[0]
    assert request.prompt == "hello"
    assert request.turn_id == "turn-meta"
    assert request.session_id == "runtime-session-meta"
    assert request.correlation_id == "corr-meta"
    assert request.env == {"MODEL_TOKEN": "per-run"}
    assert request.metadata == {"tenant": "acme"}


def test_orka_rejects_undeclared_orka_controller_env():
    app = create_orka_app(_spec(), EchoFactory(), AUTH["authorization"].removeprefix("Bearer "))

    with TestClient(app) as client:
        resp = client.post(
            "/v1/turns",
            json=_start_payload(input={"prompt": "hi", "contextRefs": [], "env": [{"name": "ORKA_CONTROLLER_URL", "value": "http://orka-api"}]}),
            headers=AUTH,
        )

    assert resp.status_code == 400
    assert "not declared" in resp.text

def test_orka_start_turn_requires_contract_fields():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        missing_namespace = _start_payload()
        missing_namespace.pop("namespace")
        missing = client.post("/v1/turns", json=missing_namespace, headers=AUTH)
        missing_prompt = _start_payload(input={"contextRefs": [], "env": []})
        prompt_resp = client.post("/v1/turns", json=missing_prompt, headers=AUTH)
        bad_env = _start_payload(input={"prompt": "hi", "env": [{"name": "1_BAD", "value": "x"}]})
        env_resp = client.post("/v1/turns", json=bad_env, headers=AUTH)
        undeclared_env = _start_payload(input={"prompt": "hi", "env": [{"name": "OTHER_TOKEN", "value": "x"}]})
        undeclared_env_resp = client.post("/v1/turns", json=undeclared_env, headers=AUTH)
        reserved_env = _start_payload(input={"prompt": "hi", "env": [{"name": "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND", "value": "echo nope"}]})
        reserved_env_resp = client.post("/v1/turns", json=reserved_env, headers=AUTH)
        bad_context_refs = _start_payload(input={"prompt": "hi", "contextRefs": [{"kind": "artifact"}], "env": []})
        context_resp = client.post("/v1/turns", json=bad_context_refs, headers=AUTH)
        brokered = _start_payload(toolExecutionMode="brokered")
        brokered_resp = client.post("/v1/turns", json=brokered, headers=AUTH)

    assert missing.status_code == 400
    assert "namespace" in missing.text
    assert prompt_resp.status_code == 400
    assert "input.prompt" in prompt_resp.text
    assert env_resp.status_code == 400
    assert "input.env" in env_resp.text
    assert undeclared_env_resp.status_code == 400
    assert "not declared" in undeclared_env_resp.text
    assert reserved_env_resp.status_code == 400
    assert "reserved" in reserved_env_resp.text
    assert context_resp.status_code == 400
    assert "contextRefs" in context_resp.text
    assert brokered_resp.status_code == 400
    assert "toolExecutionMode" in brokered_resp.text


def test_orka_context_refs_are_validated_and_forwarded_as_safe_references():
    factory = EchoFactory()
    app = create_orka_app(_spec(), factory, auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-context",
            input={
                "prompt": "hello",
                "contextRefs": [{"kind": "artifact", "name": "ctx", "seq": 7}],
                "env": [],
            },
        )
        resp = client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)

    assert resp.status_code == 200
    request = factory.runtime.requests[0]
    assert request.metadata["contextRefs"] == '[{"kind":"artifact","name":"ctx","seq":7}]'
    assert request.history == ()


def test_orka_protected_endpoints_require_bearer_token():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        create = client.post("/v1/turns", json=_start_payload())
        events = client.get("/v1/turns/missing/events")
        cancel = client.post("/v1/turns/missing/cancel", json=_cancel_payload(turnID="missing"))

    assert create.status_code == 401
    assert events.status_code == 401
    assert cancel.status_code == 401


@pytest.mark.parametrize("turn_id", ["", " ", ".", "..", " turn", "turn ", "turn/one", r"turn\one"])
def test_orka_rejects_turn_ids_that_are_not_trimmed_single_path_segments(turn_id: str):
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        response = client.post("/v1/turns", json=_start_payload(turnID=turn_id), headers=AUTH)

    assert response.status_code == 400
    assert "turnID" in response.text


@pytest.mark.parametrize(
    ("turn_id", "escaped_turn_id"),
    [
        pytest.param("turn:one", "turn:one", id="colon"),
        pytest.param("turn one", "turn%20one", id="space"),
        pytest.param("türn-雪", "t%C3%BCrn-%E9%9B%AA", id="unicode"),
        pytest.param("turn$&+=@", "turn$&+=@", id="orka-path-safe-reserved"),
        pytest.param("turn,one", "turn%2Cone", id="escaped-reserved"),
        pytest.param("turn?one", "turn%3Fone", id="query-delimiter"),
        pytest.param("\x1cturn\x1c", "%1Cturn%1C", id="go-non-space-control"),
        pytest.param("t" * 1024, "t" * 1024, id="long-segment"),
    ],
)
def test_orka_accepts_and_path_escapes_valid_turn_segments_exactly_like_orka(turn_id: str, escaped_turn_id: str):
    app = create_orka_app(_spec(), EchoFactory(), auth_token=AUTH["authorization"].removeprefix("Bearer "))

    with TestClient(app) as client:
        response = client.post("/v1/turns", json=_start_payload(turnID=turn_id), headers=AUTH)
        if response.status_code == 202:
            event_stream_path = response.json()["eventStreamPath"]
            events = client.get(event_stream_path, headers=AUTH)
        else:
            event_stream_path = ""
            events = None

    assert response.status_code == 202, response.text
    assert event_stream_path == f"/v1/turns/{escaped_turn_id}/events"
    assert events is not None
    assert events.status_code == 200
    frames = _frames(events.text)
    assert frames[0]["turnID"] == turn_id
    assert frames[-1]["type"] == "TurnCompleted"


def test_orka_cancel_accepts_contract_request_and_produces_cancelled_terminal_frame():
    app = create_orka_app(_spec(), EchoFactory(delay=60), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-cancel", input={"prompt": "slow", "contextRefs": [], "env": []})
        cancel_payload = _cancel_payload(turnID=turn_id)
        cancel = client.post(f"/v1/turns/{turn_id}/cancel", json=cancel_payload, headers=AUTH)
        for _ in range(20):
            frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
            if frames[-1]["type"] == "TurnCancelled":
                break
            time.sleep(0.01)

    assert cancel.status_code == 202
    assert cancel.json() == {
        "version": ORKA_HARNESS_VERSION,
        "accepted": True,
        "runtimeSessionID": cancel_payload["runtimeSessionID"],
        "turnID": turn_id,
        "correlationID": cancel_payload["correlationID"],
        "message": "cancel accepted",
    }
    assert [frame["type"] for frame in frames] == ["TurnStarted", "TurnCancelled"]
    _assert_frame_identity(frames[-1], seq=2, typ="TurnCancelled", turn_id=turn_id)



def test_orka_cancel_rejects_runtime_session_or_correlation_mismatch():
    app = create_orka_app(_spec(), EchoFactory(delay=60), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-cancel-identity", input={"prompt": "slow", "contextRefs": [], "env": []})
        wrong_session = client.post(
            f"/v1/turns/{turn_id}/cancel",
            json=_cancel_payload(turnID=turn_id, runtimeSessionID="other-session"),
            headers=AUTH,
        )
        wrong_correlation = client.post(
            f"/v1/turns/{turn_id}/cancel",
            json=_cancel_payload(turnID=turn_id, correlationID="other-corr"),
            headers=AUTH,
        )

    assert wrong_session.status_code == 400
    assert "runtimeSessionID" in wrong_session.text
    assert wrong_correlation.status_code == 400
    assert "correlationID" in wrong_correlation.text


@pytest.mark.parametrize(
    ("field_name", "mismatched_value"),
    [
        ("namespace", "other-namespace"),
        ("taskName", "other-task"),
        ("sessionName", "other-session-name"),
    ],
)
def test_orka_cancel_rejects_turn_owner_mismatch_without_cancelling_turn(field_name: str, mismatched_value: str):
    app = create_orka_app(_spec(), EchoFactory(delay=60), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID=f"turn-cancel-{field_name}", input={"prompt": "slow", "contextRefs": [], "env": []})
        state = client.app.state.turns[turn_id]
        mismatch = client.post(
            f"/v1/turns/{turn_id}/cancel",
            json=_cancel_payload(turnID=turn_id, **{field_name: mismatched_value}),
            headers=AUTH,
        )

        assert mismatch.status_code == 400
        assert mismatch.json() == {"error": "cancel namespace/taskName/sessionName must match turn"}
        assert state.task is not None
        assert state.task.cancelling() == 0
        assert state.terminal_event is None

        accepted = client.post(f"/v1/turns/{turn_id}/cancel", json=_cancel_payload(turnID=turn_id), headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert accepted.status_code == 202
    assert frames[-1]["type"] == "TurnCancelled"


def test_orka_cancel_rejects_body_turn_id_mismatch():
    app = create_orka_app(_spec(), EchoFactory(delay=60), auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-cancel-mismatch", input={"prompt": "slow", "contextRefs": [], "env": []})
        mismatch = client.post(f"/v1/turns/{turn_id}/cancel", json=_cancel_payload(turnID="other-turn"), headers=AUTH)

    assert mismatch.status_code == 400
    assert "match route" in mismatch.text



def test_orka_enforces_advertised_single_active_turn_limit():
    app = create_orka_app(_spec(), EchoFactory(delay=60), auth_token="test-token")

    with TestClient(app) as client:
        first_id = _create_turn(client, turnID="turn-active", input={"prompt": "slow", "contextRefs": [], "env": []})
        rejected = client.post(
            "/v1/turns",
            json=_start_payload(turnID="turn-second", input={"prompt": "second", "contextRefs": [], "env": []}),
            headers=AUTH,
        )
        cancel = client.post(f"/v1/turns/{first_id}/cancel", json=_cancel_payload(turnID=first_id), headers=AUTH)
        assert cancel.status_code == 202
        frames = _frames(client.get(f"/v1/turns/{first_id}/events", headers=AUTH).text)
        assert frames[-1]["type"] == "TurnCancelled"
        accepted_after_terminal = client.post(
            "/v1/turns",
            json=_start_payload(turnID="turn-after", input={"prompt": "after", "contextRefs": [], "env": []}),
            headers=AUTH,
        )

    assert rejected.status_code == 409
    assert rejected.json() == {"error": "maximum concurrent turns reached"}
    assert accepted_after_terminal.status_code == 202


def test_orka_terminal_turn_retention_is_bounded():
    app = create_orka_app(_spec(), EchoFactory(), auth_token="test-token", max_terminal_turns=1)

    with TestClient(app) as client:
        first_id = _create_turn(client, turnID="turn-old", input={"prompt": "old", "contextRefs": [], "env": []})
        first_events = client.get(f"/v1/turns/{first_id}/events", headers=AUTH)
        assert first_events.status_code == 200

        second_id = _create_turn(client, turnID="turn-new", input={"prompt": "new", "contextRefs": [], "env": []})
        second_events = client.get(f"/v1/turns/{second_id}/events", headers=AUTH)
        assert second_events.status_code == 200

        evicted = client.get(f"/v1/turns/{first_id}/events", headers=AUTH)
        kept = client.get(f"/v1/turns/{second_id}/events?afterSeq=1", headers=AUTH)

    assert evicted.status_code == 404
    assert kept.status_code == 200
    assert [frame["type"] for frame in _frames(kept.text)] == ["RuntimeOutput", "TurnCompleted"]


def test_orka_lifespan_awaits_cancelled_turn_and_terminal_callback_before_runtime_close(monkeypatch):
    turn_started = threading.Event()

    class ShutdownRuntime:
        def __init__(self) -> None:
            self.state: Any = None
            self.close_observations: list[tuple[bool, str | None]] = []

        async def __aenter__(self) -> RuntimeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> bool | None:
            terminal_type = self.state.terminal_event.type if self.state.terminal_event is not None else None
            self.close_observations.append((self.state.task.done(), terminal_type))
            return None

        async def run(self, request: RunRequest) -> RunResult:
            raise AssertionError("patched turn runner should own the active task")

    class ShutdownFactory:
        def __init__(self) -> None:
            self.runtime = ShutdownRuntime()

        def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
            return self.runtime

    async def uncaught_cancel_run_turn(
        get_runtime,
        turns,
        terminal_order,
        state,
        run_request,
        *,
        max_terminal_turns,
        max_history_bytes,
        brokered_tools=None,
    ) -> None:
        del turns, terminal_order, state, max_terminal_turns, max_history_bytes, brokered_tools
        await get_runtime(run_request)
        turn_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(orka_module, "_run_turn", uncaught_cancel_run_turn)
    factory = ShutdownFactory()
    app = create_orka_app(_spec(), factory, auth_token="test-token")

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-shutdown", input={"prompt": "slow", "contextRefs": [], "env": []})
        assert turn_started.wait(timeout=2)
        state = client.app.state.turns[turn_id]
        factory.runtime.state = state
        assert state.task is not None
        assert not state.task.done()

    assert factory.runtime.close_observations == [(True, "TurnCancelled")]
    assert state.task.done()
    assert state.terminal_event is not None
    assert state.terminal_event.type == "TurnCancelled"


class EnvRuntime:
    def __init__(self, token: str) -> None:
        self.token = token
        self.requests: list[RunRequest] = []
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> RuntimeSession:
        self.entered += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        self.exited += 1
        return None

    async def run(self, request: RunRequest) -> RunResult:
        import os

        self.requests.append(request)
        return RunResult(text=f"token={self.token};run={os.environ.get('MODEL_TOKEN')}")


class EnvFactory:
    def __init__(self) -> None:
        self.runtimes: list[EnvRuntime] = []

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        import os

        token = os.environ["MODEL_TOKEN"]
        runtime = EnvRuntime(token)
        self.runtimes.append(runtime)
        return runtime


class SlowCloseEnvRuntime(EnvRuntime):
    def __init__(
        self,
        token: str,
        *,
        close_delay: float,
        close_error: BaseException | None = None,
        close_release: threading.Event | None = None,
    ) -> None:
        super().__init__(token)
        self.close_delay = close_delay
        self.close_error = close_error
        self.close_release = close_release
        self.close_calls = 0
        self.close_completed = 0
        self.close_cancelled = 0
        self.close_failed = 0
        self.close_started = threading.Event()
        self.close_finished = threading.Event()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        self.close_calls += 1
        self.close_started.set()
        try:
            if self.close_release is not None:
                await asyncio.to_thread(self.close_release.wait)
            else:
                await asyncio.sleep(self.close_delay)
        except asyncio.CancelledError:
            self.close_cancelled += 1
            raise
        if self.close_error is not None:
            self.close_failed += 1
            self.close_finished.set()
            raise self.close_error
        self.close_completed += 1
        self.exited += 1
        self.close_finished.set()
        return None


class SlowFirstCloseEnvFactory:
    def __init__(
        self,
        *,
        close_delay: float = 0.25,
        close_error: BaseException | None = None,
        close_release: threading.Event | None = None,
    ) -> None:
        self.close_delay = close_delay
        self.close_error = close_error
        self.close_release = close_release
        self.runtimes: list[SlowCloseEnvRuntime] = []

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        import os

        delay = self.close_delay if not self.runtimes else 0
        close_error = self.close_error if not self.runtimes else None
        close_release = self.close_release if not self.runtimes else None
        runtime = SlowCloseEnvRuntime(
            os.environ["MODEL_TOKEN"],
            close_delay=delay,
            close_error=close_error,
            close_release=close_release,
        )
        self.runtimes.append(runtime)
        return runtime


def test_orka_runtime_build_is_deferred_until_turn_env_is_available(monkeypatch):
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    factory = EnvFactory()
    app = create_orka_app(_spec(), factory, auth_token="test-token")
    assert factory.runtimes == []

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-env-build",
            input={"prompt": "hello", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "turn-token"}]},
        )
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        assert frames[-1]["type"] == "TurnCompleted"
        assert frames[-1]["completed"]["result"] == "token=turn-token;run=turn-token"

    assert len(factory.runtimes) == 1
    assert factory.runtimes[0].entered == 1
    assert factory.runtimes[0].exited == 1
    assert "MODEL_TOKEN" not in __import__("os").environ


def test_orka_runtime_session_cache_is_bounded_and_closes_evicted_runtime(monkeypatch):
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    factory = EnvFactory()
    app = create_orka_app(_spec(), factory, auth_token="test-token", max_runtime_sessions=1)

    with TestClient(app) as client:
        first_id = _create_turn(
            client,
            turnID="turn-session-one",
            runtimeSessionID="runtime-session-one",
            correlationID="corr-one",
            input={"prompt": "one", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "one-token"}]},
        )
        first_frames = _frames(client.get(f"/v1/turns/{first_id}/events", headers=AUTH).text)
        assert first_frames[-1]["completed"]["result"] == "token=one-token;run=one-token"

        second_id = _create_turn(
            client,
            turnID="turn-session-two",
            runtimeSessionID="runtime-session-two",
            correlationID="corr-two",
            input={"prompt": "two", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "two-token"}]},
        )
        second_frames = _frames(client.get(f"/v1/turns/{second_id}/events", headers=AUTH).text)
        assert second_frames[-1]["completed"]["result"] == "token=two-token;run=two-token"
        assert len(factory.runtimes) == 2
        assert factory.runtimes[0].exited == 1
        assert factory.runtimes[1].exited == 0

    assert factory.runtimes[1].exited == 1


def test_orka_capacity_eviction_cleanup_survives_turn_deadline_and_blocks_new_runtime(monkeypatch):
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    close_release = threading.Event()
    capacity_wait_started = threading.Event()
    original_asyncio_wait = asyncio.wait

    async def observed_asyncio_wait(fs, *, timeout=None, return_when=asyncio.ALL_COMPLETED):
        waitables = tuple(fs)
        if return_when == asyncio.FIRST_COMPLETED and any(
            isinstance(item, asyncio.Task) and item.get_name() == "agentkit-orka-runtime-close" for item in waitables
        ):
            capacity_wait_started.set()
        return await original_asyncio_wait(waitables, timeout=timeout, return_when=return_when)

    monkeypatch.setattr(orka_module.asyncio, "wait", observed_asyncio_wait)
    factory = SlowFirstCloseEnvFactory(close_release=close_release)
    app = create_orka_app(_spec(), factory, auth_token=AUTH["authorization"].removeprefix("Bearer "), max_runtime_sessions=1)

    with TestClient(app) as client:
        try:
            first_id = _create_turn(
                client,
                turnID="turn-session-one",
                runtimeSessionID="runtime-session-one",
                correlationID="corr-one",
                input={"prompt": "one", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "one-token"}]},
            )
            first_frames = _frames(client.get(f"/v1/turns/{first_id}/events", headers=AUTH).text)
            assert first_frames[-1]["type"] == "TurnCompleted"

            second_deadline = (datetime.now(UTC) + timedelta(milliseconds=500)).isoformat().replace("+00:00", "Z")
            second_id = _create_turn(
                client,
                turnID="turn-session-two",
                runtimeSessionID="runtime-session-two",
                correlationID="corr-two",
                deadline=second_deadline,
                input={"prompt": "two", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "two-token"}]},
            )
            second_frames = _frames(client.get(f"/v1/turns/{second_id}/events", headers=AUTH).text)
            assert second_frames[-1]["type"] == "TurnFailed"
            assert second_frames[-1]["failed"]["reason"] == "DeadlineExceeded"

            third_deadline = (datetime.now(UTC) + timedelta(milliseconds=500)).isoformat().replace("+00:00", "Z")
            third_id = _create_turn(
                client,
                turnID="turn-session-three",
                runtimeSessionID="runtime-session-three",
                correlationID="corr-three",
                deadline=third_deadline,
                input={"prompt": "three", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "three-token"}]},
            )
            third_frames = _frames(client.get(f"/v1/turns/{third_id}/events", headers=AUTH).text)
            assert third_frames[-1]["type"] == "TurnFailed"
            assert third_frames[-1]["failed"]["reason"] == "DeadlineExceeded"
            assert len(factory.runtimes) == 1

            capacity_wait_started.clear()
            fourth_id = _create_turn(
                client,
                turnID="turn-session-four",
                runtimeSessionID="runtime-session-four",
                correlationID="corr-four",
                input={"prompt": "four", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "four-token"}]},
            )
            assert capacity_wait_started.wait(timeout=2)
            close_release.set()
            fourth_frames = _frames(client.get(f"/v1/turns/{fourth_id}/events", headers=AUTH).text)
            assert fourth_frames[-1]["type"] == "TurnCompleted"
        finally:
            close_release.set()
        assert factory.runtimes[0].close_finished.wait(timeout=2)

    assert len(factory.runtimes) == 2
    assert factory.runtimes[0].close_calls == 1
    assert factory.runtimes[0].close_cancelled == 0
    assert factory.runtimes[0].close_completed == 1
    assert factory.runtimes[1].close_calls == 1
    assert factory.runtimes[1].close_completed == 1


def test_orka_env_replacement_cleanup_survives_turn_cancellation(monkeypatch):
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    close_release = threading.Event()
    factory = SlowFirstCloseEnvFactory(close_release=close_release)
    app = create_orka_app(_spec(), factory, auth_token=AUTH["authorization"].removeprefix("Bearer "), max_runtime_sessions=2)

    with TestClient(app) as client:
        try:
            first_id = _create_turn(
                client,
                turnID="turn-env-one",
                runtimeSessionID="runtime-session-rotating",
                correlationID="corr-one",
                input={"prompt": "one", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "one-token"}]},
            )
            first_frames = _frames(client.get(f"/v1/turns/{first_id}/events", headers=AUTH).text)
            assert first_frames[-1]["type"] == "TurnCompleted"

            second_id = _create_turn(
                client,
                turnID="turn-env-two",
                runtimeSessionID="runtime-session-rotating",
                correlationID="corr-two",
                input={"prompt": "two", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "two-token"}]},
            )
            assert factory.runtimes[0].close_started.wait(timeout=2)
            cancel = client.post(
                f"/v1/turns/{second_id}/cancel",
                json=_cancel_payload(
                    turnID=second_id,
                    runtimeSessionID="runtime-session-rotating",
                    correlationID="corr-two",
                ),
                headers=AUTH,
            )
            second_frames = _frames(client.get(f"/v1/turns/{second_id}/events", headers=AUTH).text)

            assert cancel.status_code == 202
            assert second_frames[-1]["type"] == "TurnCancelled"
            assert not factory.runtimes[0].close_finished.is_set()

            third_deadline = (datetime.now(UTC) + timedelta(milliseconds=500)).isoformat().replace("+00:00", "Z")
            third_id = _create_turn(
                client,
                turnID="turn-env-three",
                runtimeSessionID="runtime-session-rotating",
                correlationID="corr-three",
                deadline=third_deadline,
                input={"prompt": "three", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "two-token"}]},
            )
            third_frames = _frames(client.get(f"/v1/turns/{third_id}/events", headers=AUTH).text)
            assert third_frames[-1]["type"] == "TurnFailed"
            assert third_frames[-1]["failed"]["reason"] == "DeadlineExceeded"
            assert len(factory.runtimes) == 1
        finally:
            close_release.set()
        assert factory.runtimes[0].close_finished.wait(timeout=2)

        fourth_id = _create_turn(
            client,
            turnID="turn-env-four",
            runtimeSessionID="runtime-session-rotating",
            correlationID="corr-four",
            input={"prompt": "four", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "two-token"}]},
        )
        fourth_frames = _frames(client.get(f"/v1/turns/{fourth_id}/events", headers=AUTH).text)
        assert fourth_frames[-1]["type"] == "TurnCompleted"

    assert len(factory.runtimes) == 2
    assert factory.runtimes[0].close_calls == 1
    assert factory.runtimes[0].close_cancelled == 0
    assert factory.runtimes[0].close_completed == 1
    assert factory.runtimes[0].close_finished.is_set()
    assert factory.runtimes[1].close_calls == 1
    assert factory.runtimes[1].close_completed == 1


def test_orka_capacity_eviction_close_failure_fails_uncancelled_turn_without_double_close(monkeypatch):
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    factory = SlowFirstCloseEnvFactory(close_delay=0, close_error=RuntimeError("runtime close failed"))
    app = create_orka_app(_spec(), factory, auth_token=AUTH["authorization"].removeprefix("Bearer "), max_runtime_sessions=1)

    with pytest.raises(AgentRunError, match="runtime cleanup failed"):
        with TestClient(app) as client:
            first_id = _create_turn(
                client,
                turnID="turn-close-failure-one",
                runtimeSessionID="runtime-session-one",
                correlationID="corr-one",
                input={"prompt": "one", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "one-token"}]},
            )
            first_frames = _frames(client.get(f"/v1/turns/{first_id}/events", headers=AUTH).text)
            assert first_frames[-1]["type"] == "TurnCompleted"

            second_id = _create_turn(
                client,
                turnID="turn-close-failure-two",
                runtimeSessionID="runtime-session-two",
                correlationID="corr-two",
                input={"prompt": "two", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "two-token"}]},
            )
            second_frames = _frames(client.get(f"/v1/turns/{second_id}/events", headers=AUTH).text)

            assert second_frames[-1]["type"] == "TurnFailed"
            assert second_frames[-1]["failed"] == {
                "reason": "RuntimeCloseFailed",
                "message": "runtime cleanup failed; restart required before opening another runtime session",
                "retryable": False,
            }

            third_id = _create_turn(
                client,
                turnID="turn-after-close-failure",
                runtimeSessionID="runtime-session-three",
                correlationID="corr-three",
                input={"prompt": "three", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "three-token"}]},
            )
            third_frames = _frames(client.get(f"/v1/turns/{third_id}/events", headers=AUTH).text)
            assert third_frames[-1]["type"] == "TurnFailed"
            assert third_frames[-1]["failed"] == {
                "reason": "RuntimeCloseFailed",
                "message": "runtime cleanup failed; restart required before opening another runtime session",
                "retryable": False,
            }
            assert len(factory.runtimes) == 1

    assert len(factory.runtimes) == 1
    assert factory.runtimes[0].close_calls == 1
    assert factory.runtimes[0].close_failed == 1
    assert factory.runtimes[0].close_completed == 0


def test_orka_shutdown_closes_active_runtime_after_orphaned_close_failure(monkeypatch):
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    factory = SlowFirstCloseEnvFactory(close_delay=1.0, close_error=RuntimeError("orphaned close failed"))
    app = create_orka_app(_spec(), factory, auth_token=AUTH["authorization"].removeprefix("Bearer "), max_runtime_sessions=2)

    with pytest.raises(AgentRunError, match="runtime cleanup failed"):
        with TestClient(app) as client:
            first_id = _create_turn(
                client,
                turnID="turn-orphan-failure-one",
                runtimeSessionID="runtime-session-one",
                correlationID="corr-one",
                input={"prompt": "one", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "one-token"}]},
            )
            first_frames = _frames(client.get(f"/v1/turns/{first_id}/events", headers=AUTH).text)
            assert first_frames[-1]["type"] == "TurnCompleted"

            second_id = _create_turn(
                client,
                turnID="turn-orphan-failure-two",
                runtimeSessionID="runtime-session-two",
                correlationID="corr-two",
                input={"prompt": "two", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "two-token"}]},
            )
            second_frames = _frames(client.get(f"/v1/turns/{second_id}/events", headers=AUTH).text)
            assert second_frames[-1]["type"] == "TurnCompleted"

            short_deadline = (datetime.now(UTC) + timedelta(milliseconds=500)).isoformat().replace("+00:00", "Z")
            third_id = _create_turn(
                client,
                turnID="turn-orphan-failure-three",
                runtimeSessionID="runtime-session-three",
                correlationID="corr-three",
                deadline=short_deadline,
                input={"prompt": "three", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "three-token"}]},
            )
            third_frames = _frames(client.get(f"/v1/turns/{third_id}/events", headers=AUTH).text)
            assert third_frames[-1]["type"] == "TurnFailed"
            assert third_frames[-1]["failed"]["reason"] == "DeadlineExceeded"
            assert factory.runtimes[0].close_finished.wait(timeout=2)

            fourth_id = _create_turn(
                client,
                turnID="turn-after-orphan-failure",
                runtimeSessionID="runtime-session-four",
                correlationID="corr-four",
                input={"prompt": "four", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "four-token"}]},
            )
            fourth_frames = _frames(client.get(f"/v1/turns/{fourth_id}/events", headers=AUTH).text)
            assert fourth_frames[-1]["type"] == "TurnFailed"
            assert fourth_frames[-1]["failed"] == {
                "reason": "RuntimeCloseFailed",
                "message": "runtime cleanup failed; restart required before opening another runtime session",
                "retryable": False,
            }
            assert len(factory.runtimes) == 2

    assert len(factory.runtimes) == 2
    assert factory.runtimes[0].close_calls == 1
    assert factory.runtimes[0].close_failed == 1
    assert factory.runtimes[1].close_calls == 1
    assert factory.runtimes[1].close_completed == 1


def test_orka_required_env_must_be_supplied_per_turn_or_process(monkeypatch):
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "orka-required-env"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "env": [{"name": "MODEL_TOKEN", "required": True}],
            "expose": {"openai": True, "port": 8080},
        }
    )
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    app = create_orka_app(spec, EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        missing = client.post(
            "/v1/turns",
            json=_start_payload(turnID="turn-missing-env", input={"prompt": "hello", "contextRefs": [], "env": []}),
            headers=AUTH,
        )
        supplied = client.post(
            "/v1/turns",
            json=_start_payload(
                turnID="turn-supplied-env",
                input={"prompt": "hello", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "turn-token"}]},
            ),
            headers=AUTH,
        )

    assert missing.status_code == 400
    assert "MODEL_TOKEN" in missing.text
    assert supplied.status_code == 202


def test_orka_required_env_rejects_empty_turn_override_even_when_process_has_value(monkeypatch):
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "orka-required-env"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "env": [{"name": "MODEL_TOKEN", "required": True}],
            "expose": {"openai": True, "port": 8080},
        }
    )
    monkeypatch.setenv("MODEL_TOKEN", "process-token")
    app = create_orka_app(spec, EchoFactory(), auth_token="test-token")

    with TestClient(app) as client:
        empty_override = client.post(
            "/v1/turns",
            json=_start_payload(
                turnID="turn-empty-env",
                input={"prompt": "hello", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": ""}]},
            ),
            headers=AUTH,
        )
        process_fallback = client.post(
            "/v1/turns",
            json=_start_payload(turnID="turn-process-env", input={"prompt": "hello", "contextRefs": [], "env": []}),
            headers=AUTH,
        )

    assert empty_override.status_code == 400
    assert "MODEL_TOKEN" in empty_override.text
    assert process_fallback.status_code == 202


def test_orka_runtime_session_rebuilds_when_turn_env_changes(monkeypatch):
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    factory = EnvFactory()
    app = create_orka_app(_spec(), factory, auth_token="test-token", max_runtime_sessions=2)

    with TestClient(app) as client:
        first_id = _create_turn(
            client,
            turnID="turn-rotated-env-1",
            runtimeSessionID="runtime-session-rotating",
            correlationID="corr-rotating-1",
            input={"prompt": "one", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "one-token"}]},
        )
        first_frames = _frames(client.get(f"/v1/turns/{first_id}/events", headers=AUTH).text)
        assert first_frames[-1]["completed"]["result"] == "token=one-token;run=one-token"

        second_id = _create_turn(
            client,
            turnID="turn-rotated-env-2",
            runtimeSessionID="runtime-session-rotating",
            correlationID="corr-rotating-2",
            input={"prompt": "two", "contextRefs": [], "env": [{"name": "MODEL_TOKEN", "value": "two-token"}]},
        )
        second_frames = _frames(client.get(f"/v1/turns/{second_id}/events", headers=AUTH).text)
        assert second_frames[-1]["completed"]["result"] == "token=two-token;run=two-token"

    assert len(factory.runtimes) == 2
    assert factory.runtimes[0].exited == 1
    assert factory.runtimes[1].exited == 1




def test_orka_continue_is_unavailable_when_brokered_is_not_enabled():
    app = create_orka_app(_spec(), EchoFactory(), AUTH["authorization"].removeprefix("Bearer "))

    with TestClient(app) as client:
        turn_id = _create_turn(client, turnID="turn-observed-no-continue")
        cont = client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert cont.status_code == 404
    assert "continuation is not enabled" in cont.text
    assert frames[-1]["type"] == "TurnCompleted"


def test_orka_brokered_read_capability_is_feature_gated():
    default_app = create_orka_app(_spec(), EchoFactory(), AUTH["authorization"].removeprefix("Bearer "))
    enabled_app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_read=True)
    write_app = create_orka_app(
        _spec(),
        OfflineEchoRuntimeFactory(),
        "test-token",
        enable_brokered_read=True,
        enable_brokered_write=True,
        enable_brokered_coordination=True,
    )

    with TestClient(default_app) as client:
        default_caps = client.get("/v1/capabilities").json()
    with TestClient(enabled_app) as client:
        enabled_caps = client.get("/v1/capabilities").json()
    with TestClient(write_app) as client:
        write_caps = client.get("/v1/capabilities").json()

    assert default_caps["toolExecutionModes"] == ["observed"]
    assert "brokeredToolClasses" not in default_caps
    assert "supportsContinuation" not in default_caps
    assert enabled_caps["toolExecutionModes"] == ["observed", "brokered"]
    assert enabled_caps["brokeredToolClasses"] == ["read"]
    assert enabled_caps["supportsContinuation"] is True
    assert write_caps["brokeredToolClasses"] == ["read", "write", "coordination"]


def test_orka_brokered_read_round_trip_emits_tool_frames_and_completes():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_read=True)

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_input())
        tool_frame = _wait_for_event_type(client, turn_id, "ToolCallRequested")
        assert tool_frame["toolName"] == "conformance_read"
        assert tool_frame["toolCallID"] == "tool-call-1"
        assert tool_frame["content"] == {"prompt": "hello"}

        cont = client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)
        duplicate = client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)
        conflicting_payload = _continue_payload(turnID=turn_id)
        conflicting_payload["toolResults"][0]["output"] = {"success": True, "data": {"answer": "different"}}
        conflicting = client.post(f"/v1/turns/{turn_id}/continue", json=conflicting_payload, headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert cont.status_code == 202, cont.text
    assert duplicate.status_code == 202, duplicate.text
    assert conflicting.status_code == 409, conflicting.text
    assert [frame["type"] for frame in frames] == ["TurnStarted", "ToolCallRequested", "ToolResultReceived", "RuntimeOutput", "TurnCompleted"]
    assert frames[2]["toolName"] == "conformance_read"
    assert frames[2]["toolCallID"] == "tool-call-1"
    assert frames[2]["content"] == {"success": True, "data": {"answer": "ok"}}
    assert "offline brokered echo" in frames[-1]["completed"]["result"]




def test_orka_brokered_write_round_trip_emits_tool_frames_and_completes():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_write=True)

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_write_input())
        tool_frame = _wait_for_event_type(client, turn_id, "ToolCallRequested")
        assert tool_frame["toolName"] == "conformance_write"
        assert tool_frame["toolCallID"] == "tool-call-1"

        cont = client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert cont.status_code == 202, cont.text
    assert [frame["type"] for frame in frames] == ["TurnStarted", "ToolCallRequested", "ToolResultReceived", "RuntimeOutput", "TurnCompleted"]
    assert frames[1]["toolName"] == "conformance_write"
    assert frames[2]["toolName"] == "conformance_write"
    assert "offline brokered echo" in frames[-1]["completed"]["result"]


def test_orka_brokered_coordination_round_trip_emits_tool_frames_and_completes():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_coordination=True)

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_coordination_input())
        tool_frame = _wait_for_event_type(client, turn_id, "ToolCallRequested")
        assert tool_frame["toolName"] == "conformance_coordination"
        assert tool_frame["toolCallID"] == "tool-call-1"

        cont = client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert cont.status_code == 202, cont.text
    assert [frame["type"] for frame in frames] == ["TurnStarted", "ToolCallRequested", "ToolResultReceived", "RuntimeOutput", "TurnCompleted"]
    assert frames[1]["toolName"] == "conformance_coordination"
    assert frames[2]["toolName"] == "conformance_coordination"
    assert "offline brokered echo" in frames[-1]["completed"]["result"]


def test_orka_brokered_continue_accepts_declined_result_without_output_or_error():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_read=True)

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_input())
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        declined = _continue_payload(turnID=turn_id)
        declined["toolResults"][0].pop("output")
        declined["toolResults"][0]["approved"] = False
        cont = client.post(f"/v1/turns/{turn_id}/continue", json=declined, headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert cont.status_code == 202, cont.text
    assert frames[2]["type"] == "ToolResultReceived"
    assert frames[2]["error"] == {"code": "ToolCallDenied", "message": "tool call was not approved", "retryable": False}
    assert frames[-1]["type"] == "TurnCompleted"
    assert "tool error" in frames[-1]["completed"]["result"]


def test_orka_brokered_continue_rejects_wrong_identity_or_unknown_tool_call():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_read=True)

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_input())
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        wrong_correlation = client.post(
            f"/v1/turns/{turn_id}/continue",
            json=_continue_payload(turnID=turn_id, correlationID="other-corr"),
            headers=AUTH,
        )
        unknown_tool = _continue_payload(turnID=turn_id)
        unknown_tool["toolResults"][0]["toolCallID"] = "other-tool"
        unknown_tool["toolResults"][0]["idempotencyKey"] = f"runtime-session-1:{turn_id}:other-tool"
        unknown = client.post(f"/v1/turns/{turn_id}/continue", json=unknown_tool, headers=AUTH)
        mixed_batch = _continue_payload(turnID=turn_id)
        mixed_batch["toolResults"].append(unknown_tool["toolResults"][0])
        mixed = client.post(f"/v1/turns/{turn_id}/continue", json=mixed_batch, headers=AUTH)
        denied_with_output = _continue_payload(turnID=turn_id)
        denied_with_output["toolResults"][0]["approved"] = False
        denied_output = client.post(f"/v1/turns/{turn_id}/continue", json=denied_with_output, headers=AUTH)
        assert all(event.type != "ToolResultReceived" for event in client.app.state.turns[turn_id].events)
        cancel = client.post(f"/v1/turns/{turn_id}/cancel", json=_cancel_payload(turnID=turn_id), headers=AUTH)

    assert wrong_correlation.status_code == 400
    assert "correlationID" in wrong_correlation.text
    assert unknown.status_code == 400
    assert "unknown toolCallID" in unknown.text
    assert mixed.status_code == 400
    assert "unknown toolCallID" in mixed.text
    assert denied_output.status_code == 400
    assert "approved is false" in denied_output.text
    assert cancel.status_code == 202


def test_orka_brokered_start_validates_safe_read_tool_schemas():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_read=True)

    with TestClient(app) as client:
        unsupported_class = client.post(
            "/v1/turns",
            json=_start_payload(
                turnID="turn-brokered-write",
                toolExecutionMode="brokered",
                input=_brokered_input(tools=[{"name": "write_tool", "brokeredClass": "write", "parameters": {"type": "object"}}]),
            ),
            headers=AUTH,
        )
        bad_parameters = client.post(
            "/v1/turns",
            json=_start_payload(
                turnID="turn-brokered-bad-schema",
                toolExecutionMode="brokered",
                input=_brokered_input(tools=[{"name": "read_tool", "brokeredClass": "read", "parameters": []}]),
            ),
            headers=AUTH,
        )

    assert unsupported_class.status_code == 400
    assert "brokered write tools are not enabled" in unsupported_class.text
    assert bad_parameters.status_code == 400
    assert "parameters" in bad_parameters.text


class CapturingBrokeredRuntime:
    def __init__(self) -> None:
        self.tools: list[BrokeredToolDefinition] = []
        self.results: list[BrokeredToolResult] = []

    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None

    async def run(self, request: RunRequest) -> RunResult:
        raise AssertionError("brokered mode must not call direct run()")

    async def run_brokered(self, request: RunRequest, tools: list[BrokeredToolDefinition], broker: ToolBroker) -> RunResult:
        self.tools = list(tools)
        result = await broker.request_tool(
            BrokeredToolCall(
                tool_call_id="tool-call-1",
                name=tools[0].name,
                arguments={"incident": "INC-1"},
                brokered_class="read",
            )
        )
        self.results.append(result)
        return RunResult(text=f"captured {result.output}")


class CapturingBrokeredFactory:
    def __init__(self) -> None:
        self.runtime = CapturingBrokeredRuntime()

    def supports_brokered_read(self) -> bool:
        return True

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


class NonAmplifyingBrokeredRuntime:
    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None

    async def run(self, request: RunRequest) -> RunResult:
        raise AssertionError("brokered mode must not call direct run()")

    async def run_brokered(self, request: RunRequest, tools: list[BrokeredToolDefinition], broker: ToolBroker) -> RunResult:
        await broker.request_tool(
            BrokeredToolCall(
                tool_call_id="tool-call-1",
                name=tools[0].name,
                arguments={"incident": "INC-1"},
                brokered_class="read",
            )
        )
        return RunResult(text="brokered output accepted")


class NonAmplifyingBrokeredFactory:
    def __init__(self) -> None:
        self.runtime = NonAmplifyingBrokeredRuntime()

    def supports_brokered_read(self) -> bool:
        return True

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


class TwoStepBrokeredRuntime(NonAmplifyingBrokeredRuntime):
    async def run_brokered(self, request: RunRequest, tools: list[BrokeredToolDefinition], broker: ToolBroker) -> RunResult:
        for tool_call_id in ("tool-call-1", "tool-call-2"):
            await broker.request_tool(
                BrokeredToolCall(
                    tool_call_id=tool_call_id,
                    name=tools[0].name,
                    arguments={"toolCallID": tool_call_id},
                    brokered_class="read",
                )
            )
        return RunResult(text="two-step brokered output accepted")


class TwoStepBrokeredFactory(NonAmplifyingBrokeredFactory):
    def __init__(self) -> None:
        self.runtime = TwoStepBrokeredRuntime()


class MutatingBrokeredRuntime(NonAmplifyingBrokeredRuntime):
    async def run_brokered(self, request: RunRequest, tools: list[BrokeredToolDefinition], broker: ToolBroker) -> RunResult:
        result = await broker.request_tool(
            BrokeredToolCall(
                tool_call_id="tool-call-1",
                name=tools[0].name,
                arguments={"probe": True},
                brokered_class="read",
            )
        )
        result.output["nested"]["items"].append("mutated")
        return RunResult(text="mutated adapter-local result")


class MutatingBrokeredFactory(NonAmplifyingBrokeredFactory):
    def __init__(self) -> None:
        self.runtime = MutatingBrokeredRuntime()


def test_orka_brokered_retained_event_is_an_immutable_json_snapshot():
    app = create_orka_app(
        _spec(),
        MutatingBrokeredFactory(),
        AUTH["authorization"].removeprefix("Bearer "),
        enable_brokered_read=True,
    )
    output = {"nested": {"items": ["original"]}}

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_input())
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        continuation = _continue_payload(turnID=turn_id)
        continuation["toolResults"][0]["output"] = output
        accepted = client.post(f"/v1/turns/{turn_id}/continue", json=continuation, headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert accepted.status_code == 202
    assert frames[2]["type"] == "ToolResultReceived"
    assert frames[2]["content"] == output


def test_orka_brokered_accepted_result_replays_while_later_tool_call_is_pending():
    app = create_orka_app(
        _spec(),
        TwoStepBrokeredFactory(),
        AUTH["authorization"].removeprefix("Bearer "),
        enable_brokered_read=True,
    )

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-brokered-inflight-replay",
            toolExecutionMode="brokered",
            input=_brokered_input(),
        )
        _wait_for_tool_call_id(client, turn_id, "tool-call-1")
        first = _continue_payload(turnID=turn_id)
        first["toolResults"][0]["turnID"] = turn_id
        first["toolResults"][0]["idempotencyKey"] = f"runtime-session-1:{turn_id}:tool-call-1"
        accepted = client.post(f"/v1/turns/{turn_id}/continue", json=first, headers=AUTH)
        _wait_for_tool_call_id(client, turn_id, "tool-call-2")
        replayed = client.post(f"/v1/turns/{turn_id}/continue", json=first, headers=AUTH)
        second = _continue_payload(turnID=turn_id)
        second_result = second["toolResults"][0]
        second_result["turnID"] = turn_id
        second_result["toolCallID"] = "tool-call-2"
        second_result["idempotencyKey"] = f"runtime-session-1:{turn_id}:tool-call-2"
        completed = client.post(f"/v1/turns/{turn_id}/continue", json=second, headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert accepted.status_code == 202
    assert replayed.status_code == 202
    assert completed.status_code == 202
    assert frames[-1]["type"] == "TurnCompleted"


def test_orka_brokered_batch_validates_conflicts_before_oversized_members():
    app = create_orka_app(
        _spec(),
        TwoStepBrokeredFactory(),
        AUTH["authorization"].removeprefix("Bearer "),
        enable_brokered_read=True,
    )

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-brokered-batch-validation",
            toolExecutionMode="brokered",
            input=_brokered_input(),
        )
        _wait_for_tool_call_id(client, turn_id, "tool-call-1")
        first = _continue_payload(turnID=turn_id)
        first_result = first["toolResults"][0]
        first_result["turnID"] = turn_id
        first_result["idempotencyKey"] = f"runtime-session-1:{turn_id}:tool-call-1"
        assert client.post(f"/v1/turns/{turn_id}/continue", json=first, headers=AUTH).status_code == 202
        _wait_for_tool_call_id(client, turn_id, "tool-call-2")

        oversized_second = dict(first_result)
        oversized_second["toolCallID"] = "tool-call-2"
        oversized_second["idempotencyKey"] = f"runtime-session-1:{turn_id}:tool-call-2"
        oversized_second["output"] = _json_object_with_size(EXPECTED_MAX_OUTPUT_BYTES + 1)
        conflicting_first = dict(first_result)
        conflicting_first["output"] = {"different": True}
        invalid_batch = _continue_payload(turnID=turn_id)
        invalid_batch["toolResults"] = [oversized_second, conflicting_first]
        invalid = client.post(f"/v1/turns/{turn_id}/continue", json=invalid_batch, headers=AUTH)

        valid_second = _continue_payload(turnID=turn_id)
        valid_second_result = valid_second["toolResults"][0]
        valid_second_result["turnID"] = turn_id
        valid_second_result["toolCallID"] = "tool-call-2"
        valid_second_result["idempotencyKey"] = f"runtime-session-1:{turn_id}:tool-call-2"
        completed = client.post(f"/v1/turns/{turn_id}/continue", json=valid_second, headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert invalid.status_code == 409
    assert "conflicting tool result" in invalid.text
    assert completed.status_code == 202
    assert frames[-1]["type"] == "TurnCompleted"


def test_orka_brokered_json_output_accepts_exact_utf8_limit_replays_and_retains_one_copy():
    output = _json_object_with_size(EXPECTED_MAX_OUTPUT_BYTES)
    app = create_orka_app(
        _spec(),
        NonAmplifyingBrokeredFactory(),
        AUTH["authorization"].removeprefix("Bearer "),
        enable_brokered_read=True,
    )

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-brokered-output-boundary",
            toolExecutionMode="brokered",
            input=_brokered_input(),
        )
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        continuation = _continue_payload(turnID=turn_id)
        continuation["toolResults"][0]["turnID"] = turn_id
        continuation["toolResults"][0]["idempotencyKey"] = f"runtime-session-1:{turn_id}:tool-call-1"
        continuation["toolResults"][0]["output"] = output
        accepted = client.post(f"/v1/turns/{turn_id}/continue", json=continuation, headers=AUTH)
        response = client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)
        replay = client.get(f"/v1/turns/{turn_id}/events?afterSeq=1", headers=AUTH)
        duplicate = client.post(f"/v1/turns/{turn_id}/continue", json=continuation, headers=AUTH)
        state = client.app.state.turns[turn_id]

    frames = _frames(response.text)
    assert accepted.status_code == 202, accepted.text
    assert duplicate.status_code == 202, duplicate.text
    assert [frame["type"] for frame in frames] == [
        "TurnStarted",
        "ToolCallRequested",
        "ToolResultReceived",
        "RuntimeOutput",
        "TurnCompleted",
    ]
    assert frames[2]["content"] == output
    assert [frame["type"] for frame in _frames(replay.text)] == [
        "ToolCallRequested",
        "ToolResultReceived",
        "RuntimeOutput",
        "TurnCompleted",
    ]
    assert state.pending_tools == {}
    assert sum(event.content == output for event in state.events) == 1
    _assert_sse_lines_fit_orka_client(response.content)
    _assert_sse_lines_fit_orka_client(replay.content)


def test_orka_brokered_result_preflight_reserves_maximum_sequence_width(monkeypatch):
    app = create_orka_app(
        _spec(),
        NonAmplifyingBrokeredFactory(),
        AUTH["authorization"].removeprefix("Bearer "),
        enable_brokered_read=True,
    )
    original_ensure = orka_module._ensure_sse_frame_fits
    tool_result_sequences: list[int] = []

    def recording_ensure(event):
        if event.type == "ToolResultReceived":
            tool_result_sequences.append(event.seq)
        return original_ensure(event)

    monkeypatch.setattr(orka_module, "_ensure_sse_frame_fits", recording_ensure)

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_input())
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        accepted = client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)
        client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)

    assert accepted.status_code == 202
    assert tool_result_sequences[0] == 9_223_372_036_854_775_807
    assert tool_result_sequences[-1] < tool_result_sequences[0]


def test_orka_brokered_json_output_over_utf8_limit_returns_413_and_visible_terminal_failure():
    output = _json_object_with_size(EXPECTED_MAX_OUTPUT_BYTES + 1)
    app = create_orka_app(
        _spec(),
        NonAmplifyingBrokeredFactory(),
        AUTH["authorization"].removeprefix("Bearer "),
        enable_brokered_read=True,
    )

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-brokered-output-over-limit",
            toolExecutionMode="brokered",
            input=_brokered_input(),
        )
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        continuation = _continue_payload(turnID=turn_id)
        continuation["toolResults"][0]["turnID"] = turn_id
        continuation["toolResults"][0]["idempotencyKey"] = f"runtime-session-1:{turn_id}:tool-call-1"
        continuation["toolResults"][0]["output"] = output
        rejected = client.post(f"/v1/turns/{turn_id}/continue", json=continuation, headers=AUTH)
        replayed_rejection = client.post(f"/v1/turns/{turn_id}/continue", json=continuation, headers=AUTH)
        conflicting_continuation = _continue_payload(turnID=turn_id)
        conflicting_continuation["toolResults"][0]["turnID"] = turn_id
        conflicting_continuation["toolResults"][0]["idempotencyKey"] = f"runtime-session-1:{turn_id}:tool-call-1"
        conflicting_continuation["toolResults"][0]["output"] = {"different": True}
        conflicting = client.post(
            f"/v1/turns/{turn_id}/continue",
            json=conflicting_continuation,
            headers=AUTH,
        )
        response = client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)
        replay = client.get(f"/v1/turns/{turn_id}/events?afterSeq=1", headers=AUTH)
        state = client.app.state.turns[turn_id]

    message = (
        f"brokered tool output is {EXPECTED_MAX_OUTPUT_BYTES + 1} UTF-8 bytes; "
        f"maxOutputBytes is {EXPECTED_MAX_OUTPUT_BYTES}"
    )
    frames = _frames(response.text)
    assert rejected.status_code == 413
    assert rejected.json() == {"error": message}
    assert replayed_rejection.status_code == 413
    assert replayed_rejection.json() == rejected.json()
    assert conflicting.status_code == 409
    assert [frame["type"] for frame in frames] == ["TurnStarted", "ToolCallRequested", "TurnFailed"]
    assert frames[-1]["failed"] == {"reason": "MaxOutputBytesExceeded", "message": message, "retryable": False}
    assert [frame["type"] for frame in _frames(replay.text)] == ["ToolCallRequested", "TurnFailed"]
    assert state.pending_tools == {}
    assert all(event.content != output for event in state.events)
    _assert_sse_lines_fit_orka_client(response.content)
    _assert_sse_lines_fit_orka_client(replay.content)


def test_orka_brokered_under_limit_output_with_unstreamable_error_uses_frame_failure_code():
    app = create_orka_app(
        _spec(),
        NonAmplifyingBrokeredFactory(),
        AUTH["authorization"].removeprefix("Bearer "),
        enable_brokered_read=True,
    )

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-brokered-error-frame-over-limit",
            toolExecutionMode="brokered",
            input=_brokered_input(),
        )
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        continuation = _continue_payload(turnID=turn_id)
        result = continuation["toolResults"][0]
        result["turnID"] = turn_id
        result["idempotencyKey"] = f"runtime-session-1:{turn_id}:tool-call-1"
        result["output"] = {"small": True}
        result["error"] = {"code": "HugeError", "message": "\x00" * 200_000, "retryable": False}
        rejected = client.post(f"/v1/turns/{turn_id}/continue", json=continuation, headers=AUTH)
        event_response = client.get(f"/v1/turns/{turn_id}/events", headers=AUTH)
        frames = _frames(event_response.text)

    assert rejected.status_code == 413
    assert frames[-1]["type"] == "TurnFailed"
    assert frames[-1]["failed"]["reason"] == "HarnessFrameTooLarge"
    _assert_sse_lines_fit_orka_client(event_response.content)


def test_orka_brokered_output_rejection_is_atomic_with_a_racing_valid_continue(monkeypatch):
    app = create_orka_app(
        _spec(),
        NonAmplifyingBrokeredFactory(),
        AUTH["authorization"].removeprefix("Bearer "),
        enable_brokered_read=True,
    )
    rejection_locked = threading.Event()
    release_rejection = threading.Event()
    original_append_failure = orka_module._append_output_failure_locked

    def blocking_append_failure(state, message, code):
        rejection_locked.set()
        assert release_rejection.wait(timeout=5)
        return original_append_failure(state, message, code)

    monkeypatch.setattr(orka_module, "_append_output_failure_locked", blocking_append_failure)

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-brokered-output-race",
            toolExecutionMode="brokered",
            input=_brokered_input(),
        )
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        oversized = _continue_payload(turnID=turn_id)
        oversized_result = oversized["toolResults"][0]
        oversized_result["turnID"] = turn_id
        oversized_result["idempotencyKey"] = f"runtime-session-1:{turn_id}:tool-call-1"
        oversized_result["output"] = _json_object_with_size(EXPECTED_MAX_OUTPUT_BYTES + 1)
        valid = _continue_payload(turnID=turn_id)
        valid_result = valid["toolResults"][0]
        valid_result["turnID"] = turn_id
        valid_result["idempotencyKey"] = f"runtime-session-1:{turn_id}:tool-call-1"

        responses: dict[str, Any] = {}
        oversized_thread = threading.Thread(
            target=lambda: responses.setdefault(
                "oversized",
                client.post(f"/v1/turns/{turn_id}/continue", json=oversized, headers=AUTH),
            )
        )
        valid_thread = threading.Thread(
            target=lambda: responses.setdefault(
                "valid",
                client.post(f"/v1/turns/{turn_id}/continue", json=valid, headers=AUTH),
            )
        )
        oversized_thread.start()
        assert rejection_locked.wait(timeout=5)
        valid_thread.start()
        time.sleep(0.05)
        assert valid_thread.is_alive()
        release_rejection.set()
        oversized_thread.join(timeout=5)
        valid_thread.join(timeout=5)
        assert not oversized_thread.is_alive()
        assert not valid_thread.is_alive()
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert responses["oversized"].status_code == 413
    assert responses["valid"].status_code == 409
    assert frames[-1]["type"] == "TurnFailed"
    assert all(frame["type"] != "ToolResultReceived" for frame in frames)


@pytest.mark.parametrize(
    "output",
    [
        pytest.param({"answer": "ok"}, id="object"),
        pytest.param([], id="array"),
        pytest.param("", id="string"),
        pytest.param(0, id="number"),
        pytest.param(False, id="boolean"),
        pytest.param(None, id="null"),
    ],
)
def test_orka_brokered_tool_result_preserves_any_json_output_value(output: Any):
    factory = CapturingBrokeredFactory()
    app = create_orka_app(
        _spec(), factory, AUTH["authorization"].removeprefix("Bearer "), enable_brokered_read=True
    )

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_input())
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        continuation = _continue_payload(turnID=turn_id)
        continuation["toolResults"][0]["output"] = output
        response = client.post(f"/v1/turns/{turn_id}/continue", json=continuation, headers=AUTH)
        if response.status_code == 202:
            frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        else:
            frames = []

    assert response.status_code == 202, response.text
    assert len(factory.runtime.results) == 1
    assert type(factory.runtime.results[0].output) is type(output)
    assert factory.runtime.results[0].output == output
    assert frames[2]["type"] == "ToolResultReceived"
    assert type(frames[2]["content"]) is type(output)
    assert frames[2]["content"] == output


@pytest.mark.parametrize(
    ("include_output", "expected_output_present"),
    [
        pytest.param(False, False, id="absent"),
        pytest.param(True, True, id="explicit-null"),
    ],
)
def test_orka_brokered_tool_result_distinguishes_absent_output_from_explicit_null(
    include_output: bool, expected_output_present: bool
):
    factory = CapturingBrokeredFactory()
    app = create_orka_app(
        _spec(), factory, AUTH["authorization"].removeprefix("Bearer "), enable_brokered_read=True
    )

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_input())
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        continuation = _continue_payload(turnID=turn_id)
        if include_output:
            continuation["toolResults"][0]["output"] = None
        else:
            continuation["toolResults"][0].pop("output")
            continuation["toolResults"][0]["error"] = {
                "code": "NoOutput",
                "message": "tool completed without output",
                "retryable": False,
            }
        response = client.post(f"/v1/turns/{turn_id}/continue", json=continuation, headers=AUTH)
        if response.status_code == 202:
            frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        else:
            frames = []

    assert response.status_code == 202, response.text
    assert len(factory.runtime.results) == 1
    result = factory.runtime.results[0]
    assert result.output is None
    assert result.output_present is expected_output_present
    assert frames[2]["type"] == "ToolResultReceived"
    if include_output:
        assert "content" in frames[2]
        assert frames[2]["content"] is None
    else:
        assert "content" not in frames[2]
        assert frames[2]["error"] == {
            "code": "NoOutput",
            "message": "tool completed without output",
            "retryable": False,
        }


def test_orka_brokered_continue_rejects_absent_output_without_error():
    app = create_orka_app(
        _spec(), OfflineEchoRuntimeFactory(), AUTH["authorization"].removeprefix("Bearer "), enable_brokered_read=True
    )

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_input())
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        continuation = _continue_payload(turnID=turn_id)
        continuation["toolResults"][0].pop("output")
        response = client.post(f"/v1/turns/{turn_id}/continue", json=continuation, headers=AUTH)

    assert response.status_code == 400
    assert "output or error is required" in response.text


def test_orka_brokered_runtime_receives_only_safe_tool_definition_fields():
    factory = CapturingBrokeredFactory()
    app = create_orka_app(_spec(), factory, "test-token", enable_brokered_read=True)

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            toolExecutionMode="brokered",
            input=_brokered_input(
                tools=[
                    {
                        "name": "safe_lookup",
                        "description": "safe schema",
                        "brokeredClass": "read",
                        "parameters": {"type": "object", "properties": {"incident": {"type": "string"}}},
                        "url": "http://tool.default.svc.cluster.local",
                        "secretRef": {"name": "should-not-cross"},
                        "headers": {"Authorization": "should-not-cross"},
                    }
                ]
            ),
        )
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)

    assert frames[-1]["type"] == "TurnCompleted"
    assert factory.runtime.tools == [
        BrokeredToolDefinition(
            name="safe_lookup",
            description="safe schema",
            brokered_class="read",
            parameters={"properties": {"incident": {"type": "string"}}, "type": "object"},
        )
    ]
    assert not hasattr(factory.runtime.tools[0], "url")
    assert not hasattr(factory.runtime.tools[0], "secretRef")
    assert not hasattr(factory.runtime.tools[0], "headers")


class BadBrokeredRuntime(CapturingBrokeredRuntime):
    def __init__(self, call: BrokeredToolCall) -> None:
        super().__init__()
        self.call = call

    async def run_brokered(self, request: RunRequest, tools: list[BrokeredToolDefinition], broker: ToolBroker) -> RunResult:
        await broker.request_tool(self.call)
        return RunResult(text="should not complete")


class BadBrokeredFactory:
    def __init__(self, call: BrokeredToolCall) -> None:
        self.runtime = BadBrokeredRuntime(call)

    def supports_brokered_read(self) -> bool:
        return True

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


def test_orka_brokered_non_json_arguments_fail_safely():
    import datetime as dt

    app = create_orka_app(
        _spec(),
        BadBrokeredFactory(BrokeredToolCall(tool_call_id="tool-call-1", name="conformance_read", arguments={"when": dt.datetime.now()}, brokered_class="read")),
        "test-token",
        enable_brokered_read=True,
    )
    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-non-json-arguments",
            toolExecutionMode="brokered",
            input=_brokered_input(),
        )
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
    assert [frame["type"] for frame in frames] == ["TurnStarted", "TurnFailed"]
    assert frames[-1]["failed"]["reason"] == "InvalidToolArguments"


def test_orka_brokered_padded_tool_call_id_fails_immediately():
    app = create_orka_app(
        _spec(),
        BadBrokeredFactory(BrokeredToolCall(tool_call_id=" tool-call-1 ", name="conformance_read", arguments={}, brokered_class="read")),
        "test-token",
        enable_brokered_read=True,
    )
    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-padded-tool-call-id",
            toolExecutionMode="brokered",
            input=_brokered_input(),
        )
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
    assert [frame["type"] for frame in frames] == ["TurnStarted", "TurnFailed"]
    assert frames[-1]["failed"]["reason"] == "InvalidToolCallID"


def test_orka_brokered_empty_tool_call_id_fails_immediately():
    app = create_orka_app(
        _spec(),
        BadBrokeredFactory(BrokeredToolCall(tool_call_id="", name="conformance_read", arguments={}, brokered_class="read")),
        "test-token",
        enable_brokered_read=True,
    )
    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-empty-tool-call-id",
            toolExecutionMode="brokered",
            input=_brokered_input(),
        )
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
    assert [frame["type"] for frame in frames] == ["TurnStarted", "TurnFailed"]
    assert frames[-1]["failed"]["reason"] == "InvalidToolCallID"


def test_orka_brokered_unknown_tool_and_invalid_arguments_fail_safely():
    cases = [
        BrokeredToolCall(tool_call_id="tool-call-1", name="unknown", arguments={}, brokered_class="read"),
        BrokeredToolCall(tool_call_id="tool-call-1", name="conformance_read", arguments="bad", brokered_class="read"),  # type: ignore[arg-type]
    ]
    for idx, call in enumerate(cases):
        app = create_orka_app(_spec(), BadBrokeredFactory(call), "test-token", enable_brokered_read=True)
        with TestClient(app) as client:
            turn_id = _create_turn(
                client,
                turnID=f"turn-bad-brokered-{idx}",
                toolExecutionMode="brokered",
                input=_brokered_input(),
            )
            frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        assert [frame["type"] for frame in frames] == ["TurnStarted", "TurnFailed"]
        assert frames[-1]["failed"]["reason"] in {"UnknownBrokeredTool", "InvalidToolArguments"}






def test_orka_brokered_pending_tool_wait_is_bounded_by_deadline():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_write=True)
    deadline = (datetime.now(UTC) + timedelta(milliseconds=150)).isoformat().replace("+00:00", "Z")

    with TestClient(app) as client:
        turn_id = _create_turn(
            client,
            turnID="turn-brokered-deadline",
            toolExecutionMode="brokered",
            input=_brokered_write_input(),
            deadline=deadline,
        )
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        accepted_after_terminal = client.post(
            "/v1/turns",
            json=_start_payload(turnID="turn-after-brokered-deadline", input={"prompt": "after", "contextRefs": [], "env": []}),
            headers=AUTH,
        )

    assert frames[-1]["type"] == "TurnFailed"
    assert frames[-1]["failed"]["reason"] == "DeadlineExceeded"
    assert accepted_after_terminal.status_code == 202


def test_orka_brokered_late_continue_after_cancel_is_rejected():
    app = create_orka_app(_spec(), OfflineEchoRuntimeFactory(), "test-token", enable_brokered_write=True)

    with TestClient(app) as client:
        turn_id = _create_turn(client, toolExecutionMode="brokered", input=_brokered_write_input())
        _wait_for_event_type(client, turn_id, "ToolCallRequested")
        cancel = client.post(f"/v1/turns/{turn_id}/cancel", json=_cancel_payload(turnID=turn_id), headers=AUTH)
        frames = _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)
        late = client.post(f"/v1/turns/{turn_id}/continue", json=_continue_payload(turnID=turn_id), headers=AUTH)

    assert cancel.status_code == 202
    assert frames[-1]["type"] == "TurnCancelled"
    assert late.status_code == 409
    assert "already terminal" in late.text


def test_orka_brokered_mode_does_not_advertise_or_fall_back_to_direct_runtime_run():
    with pytest.raises(ValueError, match="requires a runtime factory that supports brokered tools"):
        create_orka_app(_spec(), EchoFactory(), "test-token", enable_brokered_read=True)


class ScriptedRuntime:
    def __init__(self, *outcomes: str | BaseException) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[RunRequest] = []
        self.closed = False

    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        self.closed = True
        return None

    async def run(self, request: RunRequest) -> RunResult:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return RunResult(text=outcome)


class ScriptedFactory:
    def __init__(self, *runtimes: ScriptedRuntime) -> None:
        self.runtimes = list(runtimes)

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtimes.pop(0)


def _run_prompt(client: TestClient, turn_id: str, prompt: str, runtime_session_id: str = "runtime-session-1") -> dict[str, Any]:
    _create_turn(
        client,
        turnID=turn_id,
        runtimeSessionID=runtime_session_id,
        input={"prompt": prompt, "contextRefs": [], "env": []},
    )
    return _frames(client.get(f"/v1/turns/{turn_id}/events", headers=AUTH).text)[-1]


def test_orka_runtime_session_history_carries_only_completed_turns():
    session = ScriptedRuntime("a1", AgentRunError("model service is unavailable", status=503, code="ModelUnavailable"), "a3")
    other = ScriptedRuntime("b1")
    app = create_orka_app(_spec(), ScriptedFactory(session, other), auth_token="test-token")

    with TestClient(app) as client:
        assert _run_prompt(client, "turn-1", "q1")["type"] == "TurnCompleted"
        assert _run_prompt(client, "turn-2", "q2")["failed"]["reason"] == "ModelUnavailable"
        assert _run_prompt(client, "turn-3", "q3")["type"] == "TurnCompleted"
        assert _run_prompt(client, "turn-4", "o1", runtime_session_id="runtime-session-2")["type"] == "TurnCompleted"

    committed = (ConversationTurn(role="user", text="q1"), ConversationTurn(role="assistant", text="a1"))
    assert [request.history for request in session.requests] == [(), committed, committed]
    assert other.requests[0].history == ()


class FailingStartRuntime(ScriptedRuntime):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    async def __aenter__(self) -> RuntimeSession:
        raise self.error


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(RuntimeError("401 for url https://tool.example/mcp?key=private-key"), id="exception"),
        pytest.param(
            AgentRunError("tool startup failed with private-key", status=502, code="MCPToolProtocolError"),
            id="agent-run-error",
        ),
        pytest.param(asyncio.CancelledError("Cancelled via cancel scope"), id="leaked-task-group-cancel"),
    ],
)
def test_orka_runtime_start_failure_hides_startup_detail(error, caplog):
    app = create_orka_app(_spec(), ScriptedFactory(FailingStartRuntime(error)), auth_token="test-token")

    with TestClient(app) as client:
        terminal = _run_prompt(client, "turn-1", "q1")

    assert terminal["type"] == "TurnFailed"
    assert terminal["failed"] == {"reason": "RuntimeStartFailed", "message": "runtime failed to start", "retryable": False}
    assert f"runtime session failed to start: {type(error).__name__}" in caplog.text
    assert "private-key" not in caplog.text


def test_orka_unexpected_run_exception_text_is_not_streamed():
    class LeakyRuntime(ScriptedRuntime):
        async def run(self, request: RunRequest) -> RunResult:
            raise TypeError("unexpected response body with private-key")

    app = create_orka_app(_spec(), ScriptedFactory(LeakyRuntime()), auth_token="test-token")
    with TestClient(app) as client:
        terminal = _run_prompt(client, "turn-1", "q1")

    assert terminal["failed"] == {"reason": "AgentRunFailed", "message": "agent run failed", "retryable": False}


def test_orka_fatal_run_failure_rebuilds_runtime_and_keeps_history():
    broken = ScriptedRuntime("a1", AgentRunError("MCP tool protocol failed", code="MCPToolProtocolError", fatal=True))
    rebuilt = ScriptedRuntime("a3")
    app = create_orka_app(_spec(), ScriptedFactory(broken, rebuilt), auth_token="test-token")

    with TestClient(app) as client:
        assert _run_prompt(client, "turn-1", "q1")["type"] == "TurnCompleted"
        assert _run_prompt(client, "turn-2", "q2")["failed"]["reason"] == "MCPToolProtocolError"
        assert _run_prompt(client, "turn-3", "q3")["type"] == "TurnCompleted"
        health = client.get("/v1/health").json()

    assert broken.closed
    assert rebuilt.requests[0].history == (
        ConversationTurn(role="user", text="q1"),
        ConversationTurn(role="assistant", text="a1"),
    )
    assert (health["status"], health["ready"]) == ("ok", True)


def test_orka_fatal_error_inside_exception_group_rebuilds_runtime():
    fatal = AgentRunError("MCP tool protocol failed", code="MCPToolProtocolError", fatal=True)
    broken = ScriptedRuntime(ExceptionGroup("parallel tools", [fatal]))
    rebuilt = ScriptedRuntime("a2")
    app = create_orka_app(_spec(), ScriptedFactory(broken, rebuilt), auth_token="test-token")

    with TestClient(app) as client:
        assert _run_prompt(client, "turn-1", "q1")["failed"]["reason"] == "MCPToolProtocolError"
        assert _run_prompt(client, "turn-2", "q2")["type"] == "TurnCompleted"

    assert broken.closed
    assert [request.prompt for request in rebuilt.requests] == ["q2"]


def test_orka_history_survives_failed_rebuild():
    broken = ScriptedRuntime("a1", AgentRunError("MCP tool protocol failed", code="MCPToolProtocolError", fatal=True))
    rebuilt = ScriptedRuntime("a4")
    factory = ScriptedFactory(broken, FailingStartRuntime(RuntimeError("tool unreachable")), rebuilt)
    app = create_orka_app(_spec(), factory, auth_token="test-token")

    with TestClient(app) as client:
        assert _run_prompt(client, "turn-1", "q1")["type"] == "TurnCompleted"
        assert _run_prompt(client, "turn-2", "q2")["failed"]["reason"] == "MCPToolProtocolError"
        assert _run_prompt(client, "turn-3", "q3")["failed"]["reason"] == "RuntimeStartFailed"
        assert _run_prompt(client, "turn-4", "q4")["type"] == "TurnCompleted"

    assert rebuilt.requests[0].history == (
        ConversationTurn(role="user", text="q1"),
        ConversationTurn(role="assistant", text="a1"),
    )


def test_orka_failed_start_does_not_store_or_evict_history():
    reopened = ScriptedRuntime("a2")
    factory = ScriptedFactory(ScriptedRuntime("a1"), FailingStartRuntime(RuntimeError("tool unreachable")), reopened)
    app = create_orka_app(_spec(), factory, auth_token="test-token", max_runtime_sessions=1, max_session_histories=1)

    with TestClient(app) as client:
        assert _run_prompt(client, "turn-1", "q1")["type"] == "TurnCompleted"
        failed = _run_prompt(client, "turn-2", "o1", runtime_session_id="runtime-session-2")
        stored = list(client.app.state.session_histories)
        assert _run_prompt(client, "turn-3", "q2")["type"] == "TurnCompleted"

    assert failed["failed"]["reason"] == "RuntimeStartFailed"
    assert stored == ["runtime-session-1"]
    assert reopened.requests[0].history == (
        ConversationTurn(role="user", text="q1"),
        ConversationTurn(role="assistant", text="a1"),
    )


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("client setup failed for https://tool.example/mcp?key=private-key"),
        AgentRunError("tool startup failed with private-key", status=502, code="MCPToolProtocolError"),
    ],
)
def test_orka_runtime_build_failure_reports_runtime_start_failed(error, caplog):
    class FailingFactory:
        def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
            raise error

    app = create_orka_app(_spec(), FailingFactory(), auth_token="test-token")
    with TestClient(app) as client:
        terminal = _run_prompt(client, "turn-1", "q1")

    assert terminal["failed"] == {"reason": "RuntimeStartFailed", "message": "runtime failed to start", "retryable": False}
    assert "private-key" not in caplog.text


def test_orka_runtime_build_failure_preserves_configuration_guidance():
    class FailingFactory:
        def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
            raise orka_module.AgentBuildError("required model env var is missing")

    app = create_orka_app(_spec(), FailingFactory(), auth_token="test-token")
    with TestClient(app) as client:
        terminal = _run_prompt(client, "turn-1", "q1")

    assert terminal["failed"] == {
        "reason": "AgentBuildError",
        "message": "required model env var is missing",
        "retryable": False,
    }


def test_orka_history_survives_runtime_capacity_eviction():
    first = ScriptedRuntime("a1")
    other = ScriptedRuntime("b1")
    reopened = ScriptedRuntime("a2")
    app = create_orka_app(_spec(), ScriptedFactory(first, other, reopened), auth_token="test-token", max_runtime_sessions=1)

    with TestClient(app) as client:
        assert _run_prompt(client, "turn-1", "q1")["type"] == "TurnCompleted"
        assert _run_prompt(client, "turn-2", "o1", runtime_session_id="runtime-session-2")["type"] == "TurnCompleted"
        assert _run_prompt(client, "turn-3", "q2")["type"] == "TurnCompleted"

    assert first.closed
    assert reopened.requests[0].history == (
        ConversationTurn(role="user", text="q1"),
        ConversationTurn(role="assistant", text="a1"),
    )


def test_orka_session_history_retention_is_bounded():
    app = create_orka_app(
        _spec(),
        ScriptedFactory(ScriptedRuntime("a1"), ScriptedRuntime("b1"), reopened := ScriptedRuntime("a2")),
        auth_token="test-token",
        max_runtime_sessions=1,
        max_session_histories=1,
    )

    with TestClient(app) as client:
        assert _run_prompt(client, "turn-1", "q1")["type"] == "TurnCompleted"
        assert _run_prompt(client, "turn-2", "o1", runtime_session_id="runtime-session-2")["type"] == "TurnCompleted"
        assert _run_prompt(client, "turn-3", "q2")["type"] == "TurnCompleted"

    assert reopened.requests[0].history == ()


def test_orka_session_history_drops_oldest_turns_over_byte_budget():
    session = ScriptedRuntime("a1", "a2", "a3", "a4")
    app = create_orka_app(_spec(), ScriptedFactory(session), auth_token="test-token", max_session_history_bytes=10)

    with TestClient(app) as client:
        for index in (1, 2, 3, 4):
            assert _run_prompt(client, f"turn-{index}", f"q{index}")["type"] == "TurnCompleted"
        stored = tuple(client.app.state.session_histories["runtime-session-1"])

    def turns(*numbers: int) -> tuple[ConversationTurn, ...]:
        return tuple(
            turn
            for number in numbers
            for turn in (ConversationTurn(role="user", text=f"q{number}"), ConversationTurn(role="assistant", text=f"a{number}"))
        )

    # Each completed turn is 4 bytes; a third stored turn would exceed 10 bytes.
    assert [request.history for request in session.requests] == [(), turns(1), turns(1, 2), turns(2, 3)]
    assert stored == turns(3, 4)


def test_orka_session_history_limit_never_drops_live_sessions():
    runtimes = [ScriptedRuntime(text) for text in ("a1", "b1", "c1")]
    app = create_orka_app(_spec(), ScriptedFactory(*runtimes), auth_token="test-token", max_runtime_sessions=2, max_session_histories=1)

    with TestClient(app) as client:
        for index in (1, 2, 3):
            session = f"runtime-session-{index}"
            assert _run_prompt(client, f"turn-{index}", f"q{index}", runtime_session_id=session)["type"] == "TurnCompleted"
        retained = list(client.app.state.session_histories)
        live = set(client.app.state.active_runtimes)

    # The limit rises to the runtime session limit; the evicted session goes.
    assert retained == ["runtime-session-2", "runtime-session-3"]
    assert live == set(retained)
