"""Hosted continuation compatibility with Orka's final approval outcomes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import agentkit_serve_common.foundry as foundry_module
from test_foundry_brokered_protocol import (
    CONTINUATION_AUTH,
    _FakeChatTransport,
    _app,
    _call,
    _chat_response,
    _continuation,
    _message_text,
    _model_loop_app,
    _multi_tool_spec,
    _spec,
)
from test_foundry_tool_workflows import _tool


_ERRORS = {
    "approval_declined": "The tool call was declined.",
    "approval_expired": "The tool approval expired.",
    "approval_cancelled": "The tool call was cancelled.",
    "approval_stale": "The tool approval is no longer valid.",
    "tool_execution_failed": "MCP tool execution failed.",
    "tool_outcome_unknown": "The tool execution outcome is unknown; do not retry.",
}


def _result(response, output, *, session="review-session"):
    request = _continuation(response["id"], _call(response)["call_id"], output)
    request["agent_session_id"] = session
    return request


@pytest.mark.parametrize("outcome", ["approved", *_ERRORS])
def test_review_wait_and_process_restart_preserve_completed_steps_and_final_result(tmp_path, monkeypatch, outcome):
    now = [1000.0]
    monkeypatch.setattr(foundry_module, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setenv("AGENTKIT_FOUNDRY_RESPONSE_STATE_TTL_SECONDS", "1800")
    state_file = tmp_path / "responses.json"
    spec = _multi_tool_spec()
    spec.brokered_tools[1].name = "create-simulated-work-order"
    spec.brokered_tools[1].brokered_class = "write"
    first_model = _FakeChatTransport([
        _tool("check-network-telemetry", {"site": "site-a"}),
        _tool("create-simulated-work-order"),
        _tool("check-network-telemetry", {"site": "site-b"}),
        _chat_response({"role": "assistant", "content": "The independent lookup finished."}),
    ])
    with TestClient(_model_loop_app(spec, first_model, response_state_file=state_file)) as client:
        initial = client.post("/responses", json={"input": "Check site-a, then create a simulated work order.", "agent_session_id": "review-session"}).json()
        first_output = {"approved": True, "output": {"site": "site-a", "signal": "low"}}
        pending_response = client.post("/responses", json=_result(initial, first_output), headers=CONTINUATION_AUTH)
        assert pending_response.status_code == 200, pending_response.text
        pending = pending_response.json()
        assert _call(pending)["name"] == "create-simulated-work-order"
        assert all(item["type"] != "message" for item in pending["output"])
        assert len(first_model.requests) == 2
        assert client.get("/readiness").json()["foundryResponses"]["stateTtlSeconds"] == 1800

        # Waiting does not require another response or keep the app's request busy.
        now[0] += 599
        other = client.post("/responses", json={"input": "Check site-b.", "agent_session_id": "other-session"}).json()
        other_final = client.post("/responses", json=_result(other, {"approved": True, "output": {"signal": "normal"}}, session="other-session"), headers=CONTINUATION_AUTH)
        assert other_final.status_code == 200
        assert _message_text(other_final.json()) == "The independent lookup finished."
        assert len(first_model.requests) == 4

    # Restart only the AgentKit process with saved state, within the same hosted
    # session. Account for the remaining review second and 240s execution budget.
    now[0] += 241
    second_model = _FakeChatTransport([] if outcome == "tool_outcome_unknown" else [
        _chat_response({"role": "assistant", "content": f"Final outcome: {outcome}"}),
    ])
    output = (
        {"approved": True, "output": {"receipt": "simulated-work-order-1"}}
        if outcome == "approved"
        else {"approved": False, "error": {"code": outcome, "message": "UNSAFE_TOOL_DETAIL", "details": "UNSAFE_TOOL_DETAIL"}}
    )
    payload = _result(pending, output)
    with TestClient(_model_loop_app(spec, second_model, response_state_file=state_file)) as client:
        wrong_session = dict(payload, agent_session_id="other-session")
        assert client.post("/responses", json=wrong_session, headers=CONTINUATION_AUTH).status_code == 409
        assert client.post("/responses", json=payload).status_code == 403
        assert second_model.requests == []
        final = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)
        assert final.status_code == 200, final.text
        duplicate = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)
        assert duplicate.json() == final.json()
        conflict = _result(pending, {"approved": True, "output": {"receipt": "different"}})
        assert client.post("/responses", json=conflict, headers=CONTINUATION_AUTH).status_code == 409
        if outcome == "tool_outcome_unknown":
            assert second_model.requests == []
            assert _message_text(final.json()) == f"{outcome}: {_ERRORS[outcome]}"
        else:
            assert len(second_model.requests) == 1
            items = second_model.requests[0]["input"]
            assert len([item for item in items if item.get("role") == "user"]) == 1
            tool_results = [item for item in items if item["type"] == "function_call_output"]
            assert len(tool_results) == 2
            assert json.loads(tool_results[0]["output"]) == first_output
            assert tool_results[1]["call_id"] == _call(pending)["call_id"]
            expected = output if outcome == "approved" else {"approved": False, "error": {"code": outcome, "message": _ERRORS[outcome]}}
            assert json.loads(tool_results[1]["output"]) == expected
            assert "UNSAFE_TOOL_DETAIL" not in json.dumps(items)


@pytest.mark.parametrize("saved_state", [False, True])
def test_lost_or_expired_review_state_never_restarts_the_original_action(tmp_path, monkeypatch, saved_state):
    now = [1000.0]
    monkeypatch.setattr(foundry_module, "time", SimpleNamespace(time=lambda: now[0]))
    state_file = tmp_path / "responses.json" if saved_state else None
    spec = _spec(tool_name="create-simulated-work-order", brokered_class="write")
    first_model = _FakeChatTransport([_tool("create-simulated-work-order")])
    with TestClient(_model_loop_app(spec, first_model, response_state_file=state_file, state_ttl_seconds=1800)) as client:
        initial = client.post("/responses", json={"input": "Create a simulated work order.", "agent_session_id": "review-session"}).json()
    now[0] += 1801
    second_model = _FakeChatTransport([])
    with TestClient(_model_loop_app(spec, second_model, response_state_file=state_file, state_ttl_seconds=1800)) as client:
        response = client.post("/responses", json=_result(initial, {"approved": True, "output": {"receipt": "late"}}), headers=CONTINUATION_AUTH)
        assert response.status_code in {404, 410}
        assert response.json()["error"]["code"] in {"unknown_previous_response_id", "response_state_expired"}
        assert second_model.requests == []


@pytest.mark.parametrize("code", _ERRORS)
def test_deterministic_final_error_preserves_code_without_claiming_an_action_did_not_run(code):
    with TestClient(_app(_spec(tool_name="create-simulated-work-order", brokered_class="write"))) as client:
        initial = client.post("/responses", json={"input": "create-simulated-work-order", "agent_session_id": "review-session"}).json()
        response = client.post("/responses", json=_result(initial, {"approved": False, "error": {"code": code, "message": "UNSAFE_TOOL_DETAIL"}}), headers=CONTINUATION_AUTH)
        assert response.status_code == 200
        assert _message_text(response.json()) == f"Brokered tool create-simulated-work-order: {code}: {_ERRORS[code]}"
