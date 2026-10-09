"""Public Responses regressions for sequential governed tool workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import agentkit_serve_common.foundry as foundry_module
import agentkit_serve_common.skills as skills_module
from agentkit_serve_common.config import AgentSpec
from test_foundry_brokered_protocol import (
    CONTINUATION_AUTH,
    _FakeChatTransport,
    _call,
    _chat_response,
    _continuation,
    _message_text,
    _model_loop_app,
    _multi_tool_spec,
    _spec,
)


def _tool(name, arguments=None):
    return _chat_response({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "untrusted-model-id", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments or {}),
        }}],
    })


def _result(response, output):
    return _continuation(response["id"], _call(response)["call_id"], output)


@pytest.fixture
def packaged_skill(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "skills"
    directory = root / "inspection"
    directory.mkdir(parents=True)
    document = directory / "SKILL.md"
    text = (
        "---\nname: inspection\ndescription: Inspect the affected site.\n---\n"
        "Read telemetry, then retrieve active incidents using the authorized tools.\n"
    )
    document.write_text(text, encoding="utf-8")
    open_directory = skills_module._open_directory

    def open_test_directory(path: Path):
        return open_directory(root / path.relative_to("/agent/skills"))

    monkeypatch.setattr(skills_module, "_open_directory", open_test_directory)
    data = _multi_tool_spec().model_dump(by_alias=True)
    data["context"] = {"providers": [{"type": "skills", "source": "filesystem", "path": "/agent/skills"}]}
    return AgentSpec.model_validate(data), document, text


@pytest.mark.parametrize("trailing_commentary", [False, True])
def test_hosted_skills_load_locally_between_governed_tool_rounds(packaged_skill, trailing_commentary):
    spec, document, text = packaged_skill
    fake = _FakeChatTransport([
        _tool("load_skill", {"skill_name": "inspection"}),
        _tool("check-network-telemetry", {"site": "site-a"}),
        _tool("load_skill", {"skill_name": "inspection"}),
        _tool("get-active-incidents"),
        _chat_response({"role": "assistant", "content": "Incident INC-17 explains the signal loss."}),
    ])
    if trailing_commentary:
        for response in fake.responses:
            if any(item.get("name") == "load_skill" for item in response["output"]):
                response["output"].append({
                    "type": "message", "role": "assistant", "phase": "commentary",
                    "content": [{"type": "output_text", "text": "Loading inspection."}],
                })
    app = _model_loop_app(spec, fake)
    document.unlink()  # Both skill calls must use the startup snapshot.
    with TestClient(app) as client:
        first = client.post("/responses", json={"input": "Inspect site-a"})
        assert first.status_code == 200, first.text
        assert _call(first.json())["name"] == "check-network-telemetry"
        payload = _result(first.json(), {"approved": True, "output": {"signal": "low"}})
        assert client.post("/responses", json=payload).status_code == 403
        assert len(fake.requests) == 2
        second = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)
        assert second.status_code == 200, second.text
        assert _call(second.json())["name"] == "get-active-incidents"
        final = client.post("/responses", json=_result(second.json(), {"approved": True, "output": {"incident": "INC-17"}}), headers=CONTINUATION_AUTH)
        assert final.status_code == 200, final.text
        assert _message_text(final.json()) == "Incident INC-17 explains the signal loss."
        assert final.json()["usage"] == {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10}

    advertised = {tool["name"] for tool in fake.requests[0]["tools"]}
    assert advertised == {"load_skill", "check-network-telemetry", "get-active-incidents"}
    assert "inspection" in fake.requests[0]["input"][1]["content"]
    items = fake.requests[-1]["input"]
    calls = [item for item in items if item["type"] == "function_call"]
    outputs = [item for item in items if item["type"] == "function_call_output"]
    assert len(calls) == len({call["call_id"] for call in calls}) == 4
    assert [call["call_id"] for call in calls] == [output["call_id"] for output in outputs]
    assert outputs[0]["output"] == outputs[2]["output"] == text
    assert calls[1]["call_id"] == _call(first.json())["call_id"]
    assert calls[3]["call_id"] == _call(second.json())["call_id"]
    assert all(request["parallel_tool_calls"] is False for request in fake.requests)
    if trailing_commentary:
        for index, item in enumerate(items):
            if item["type"] == "function_call" and item["name"] == "load_skill":
                assert items[index + 1] == {
                    "type": "message", "role": "assistant", "phase": "commentary", "content": "Loading inspection.",
                }
                assert items[index + 2]["type"] == "function_call_output"


@pytest.mark.parametrize("arguments", [
    {"skill_name": "missing"},
    {"skill_name": "../inspection"},
    {"skill_name": "inspection", "path": "/etc/passwd"},
    {"skill_name": 42},
])
def test_hosted_skill_load_rejects_unadvertised_input(packaged_skill, arguments):
    spec, _, _ = packaged_skill
    fake = _FakeChatTransport([_tool("load_skill", arguments)])
    with TestClient(_model_loop_app(spec, fake)) as client:
        response = client.post("/responses", json={"input": "Inspect site-a"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "InvalidToolArguments"
        assert len(fake.requests) == 1


def test_hosted_local_skill_calls_share_the_tool_budget(packaged_skill):
    spec, _, _ = packaged_skill
    fake = _FakeChatTransport([_tool("load_skill", {"skill_name": "inspection"}) for _ in range(17)])
    with TestClient(_model_loop_app(spec, fake)) as client:
        response = client.post("/responses", json={"input": "Keep loading the skill"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "tool_loop_limit_exceeded"
        assert len(fake.requests) == 17
        assert fake.requests[-1]["tool_choice"] == "none"
        assert fake.requests[-1]["tools"] == fake.requests[0]["tools"]


@pytest.mark.parametrize("denied", [False, True])
def test_chained_tool_calls_keep_pairing_replay_and_results_across_restart(tmp_path, denied):
    state_file = tmp_path / "responses.json"
    spec = _multi_tool_spec()
    fake = _FakeChatTransport([
        _tool("check-network-telemetry", {"site": "site-a"}),
        _tool("get-active-incidents"),
    ])
    first_output = {"approved": False, "error": {"code": "policy_denied", "message": "Read denied"}} if denied else {"approved": True, "output": {"site": "site-a", "signal": "low"}}
    with TestClient(_model_loop_app(spec, fake, max_pending_responses=1, response_state_file=state_file)) as client:
        initial = client.post("/responses", json={"input": "Check telemetry, then incidents", "agent_session_id": "session-a"}).json()
        first_result = _result(initial, first_output)
        first_result["agent_session_id"] = "session-a"
        unauthorized = client.post("/responses", json=first_result)
        assert unauthorized.status_code == 403
        response = client.post("/responses", json=first_result, headers=CONTINUATION_AUTH)
        assert response.status_code == 200, response.text
        second = response.json()
        assert second["id"] != initial["id"]
        assert _call(second)["call_id"] != _call(initial)["call_id"]
        assert second["previous_response_id"] == initial["id"]
        assert _call(second)["name"] == "get-active-incidents"
        assert client.post("/responses", json=first_result, headers=CONTINUATION_AUTH).json() == second
        assert len(fake.requests) == 2
        assert json.loads(fake.requests[1]["input"][-1]["output"]) == first_output
    # Capacity is per workflow, so advancing remains possible with one entry.
    assert len(json.loads(state_file.read_text())["states"]) == 1
    resumed_model = _FakeChatTransport([_chat_response({"role": "assistant", "content": "Incident INC-17 explains the outage."})])
    with TestClient(_model_loop_app(spec, resumed_model, max_pending_responses=1, response_state_file=state_file)) as client:
        assert client.post("/responses", json=first_result, headers=CONTINUATION_AUTH).json() == second
        second_result = _result(second, {"approved": True, "output": {"incident": "INC-17"}})
        second_result["agent_session_id"] = "session-a"
        wrong_pair = dict(first_result, previous_response_id=second["id"])
        assert client.post("/responses", json=wrong_pair, headers=CONTINUATION_AUTH).status_code == 400
        wrong_session = dict(second_result, agent_session_id="session-b")
        assert client.post("/responses", json=wrong_session, headers=CONTINUATION_AUTH).status_code == 409
        completed = client.post("/responses", json=second_result, headers=CONTINUATION_AUTH)
        assert completed.status_code == 200, completed.text
        assert _message_text(completed.json()) == "Incident INC-17 explains the outage."
        assert completed.json()["previous_response_id"] == second["id"]
        assert client.post("/responses", json=first_result, headers=CONTINUATION_AUTH).json() == second
        assert client.post("/responses", json=second_result, headers=CONTINUATION_AUTH).json() == completed.json()
        assert len(resumed_model.requests) == 1
        returned = [json.loads(item["output"]) for item in resumed_model.requests[0]["input"] if item["type"] == "function_call_output"]
        assert returned == [first_output, {"approved": True, "output": {"incident": "INC-17"}}]


def test_every_new_round_validates_tool_and_arguments_before_export():
    fake = _FakeChatTransport([
        _tool("check-network-telemetry", {"site": "site-a"}),
        _tool("check-network-telemetry", {"site": 42}),
        _tool("get-active-incidents"),
    ])
    with TestClient(_model_loop_app(_multi_tool_spec(), fake)) as client:
        initial = client.post("/responses", json={"input": "Inspect the outage"}).json()
        payload = _result(initial, {"approved": True, "output": {}})
        rejected = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)
        assert rejected.status_code == 400
        assert rejected.json()["error"]["code"] == "InvalidToolArguments"
        retried = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)
        assert retried.status_code == 200, retried.text
        assert _call(retried.json())["name"] == "get-active-incidents"


def test_chained_round_write_failure_retries_cached_transition_without_model_work(tmp_path, monkeypatch):
    fake = _FakeChatTransport([_tool("conformance_read"), _tool("conformance_read")])
    original = foundry_module._FoundryResponseStateStore._persist
    fail = False

    def persist(store, data):
        if fail and b'"continuationPayloads"' in data:
            raise foundry_module._StatePersistenceError("injected transition failure")
        return original(store, data)

    monkeypatch.setattr(foundry_module._FoundryResponseStateStore, "_persist", persist)
    with TestClient(_model_loop_app(_spec(), fake, response_state_file=tmp_path / "responses.json")) as client:
        initial = client.post("/responses", json={"input": "Read twice"}).json()
        payload = _result(initial, {"approved": True, "output": {}})
        fail = True
        assert client.post("/responses", json=payload, headers=CONTINUATION_AUTH).status_code == 503
        fail = False
        retried = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)
        assert retried.status_code == 200, retried.text
        assert _call(retried.json())["call_id"] != _call(initial)["call_id"]
        assert len(fake.requests) == 2


def test_earlier_round_replays_after_later_capacity_failure_and_restart(tmp_path):
    state_file = tmp_path / "responses.json"
    fake = _FakeChatTransport([
        _tool("conformance_read"),
        _tool("conformance_read"),
        _tool("conformance_read"),
        _chat_response({"role": "assistant", "content": "x" * 6_000}),
    ])
    options = {"response_state_file": state_file, "max_response_state_bytes": 10_000}
    with TestClient(_model_loop_app(_spec(), fake, **options)) as client:
        initial = client.post("/responses", json={"input": "Read twice"}).json()
        first_result = _result(initial, {"approved": True, "output": {}})
        second = client.post("/responses", json=first_result, headers=CONTINUATION_AUTH)
        assert second.status_code == 200, second.text
        unrelated = client.post("/responses", json={"input": "Another pending read"})
        assert unrelated.status_code == 200, unrelated.text
        second_result = _result(second.json(), {"approved": True, "output": {}})
        failed = client.post("/responses", json=second_result, headers=CONTINUATION_AUTH)
        assert failed.status_code == 429, failed.text
        assert failed.json()["error"]["code"] == "brokered_response_state_full"
        assert client.post("/responses", json=second_result, headers=CONTINUATION_AUTH).json() == failed.json()
        replayed = client.post("/responses", json=first_result, headers=CONTINUATION_AUTH)
        assert replayed.status_code == 200, replayed.text
        assert replayed.json() == second.json()
        assert len(fake.requests) == 4

    restarted = _FakeChatTransport([])
    with TestClient(_model_loop_app(_spec(), restarted, **options)) as client:
        replayed = client.post("/responses", json=first_result, headers=CONTINUATION_AUTH)
        assert replayed.status_code == 200, replayed.text
        assert replayed.json() == second.json()
        assert client.post("/responses", json=second_result, headers=CONTINUATION_AUTH).json() == failed.json()
        assert not restarted.requests


def test_brokered_workflow_has_a_finite_tool_budget():
    fake = _FakeChatTransport([_tool("conformance_read") for _ in range(17)])
    with TestClient(_model_loop_app(_spec(), fake, max_pending_responses=1)) as client:
        response = client.post("/responses", json={"input": "Keep reading"})
        for _ in range(16):
            assert response.status_code == 200, response.text
            response = client.post("/responses", json=_result(response.json(), {"approved": True, "output": {}}), headers=CONTINUATION_AUTH)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "tool_loop_limit_exceeded"
        assert fake.requests[-1]["tool_choice"] == "none"
        assert fake.requests[-1]["tools"] == fake.requests[0]["tools"]


def test_hosted_followups_retain_dialogue_without_tool_data_across_restart(tmp_path):
    state_file = tmp_path / "responses.json"
    fake = _FakeChatTransport([
        _chat_response({"role": "assistant", "content": "I will inspect site-a."}),
        _tool("conformance_read"),
        _chat_response({"role": "assistant", "content": "Site-a has incident INC-17."}),
    ])
    with TestClient(_model_loop_app(_spec(), fake, response_state_file=state_file)) as client:
        first = client.post("/responses", json={"input": "My site is site-a", "agent_session_id": "session-a"})
        assert first.status_code == 200, first.text
        second = client.post("/responses", json={"input": "Check it", "agent_session_id": "session-a", "previous_response_id": first.json()["id"]})
        assert second.status_code == 200, second.text
        request = _result(second.json(), {"approved": True, "output": {"incident": "INC-17", "internal_note": "tool-data-only"}})
        request["agent_session_id"] = "session-a"
        completed = client.post("/responses", json=request, headers=CONTINUATION_AUTH)
        assert completed.status_code == 200, completed.text
        context = fake.requests[1]["input"]
        assert context[-3:] == [
            {"type": "message", "role": "user", "content": "My site is site-a"},
            {"type": "message", "role": "assistant", "content": "I will inspect site-a."},
            {"type": "message", "role": "user", "content": "Check it"},
        ]
    restarted = _FakeChatTransport([_chat_response({"role": "assistant", "content": "INC-17 is the incident at site-a."})])
    with TestClient(_model_loop_app(_spec(), restarted, response_state_file=state_file)) as client:
        payload = {"input": "Which incident was that?", "agent_session_id": "session-a", "previous_response_id": completed.json()["id"]}
        for session in [None, "session-b"]:
            wrong = dict(payload)
            if session is None:
                del wrong["agent_session_id"]
            else:
                wrong["agent_session_id"] = session
            assert client.post("/responses", json=wrong).status_code == 409
        followup = client.post("/responses", json=payload)
        assert followup.status_code == 200, followup.text
        assert _message_text(followup.json()) == "INC-17 is the incident at site-a."
        items = restarted.requests[0]["input"]
        assert any(item.get("content") == "Site-a has incident INC-17." for item in items)
        assert all(item["type"] == "message" for item in items)
        assert "tool-data-only" not in json.dumps(items)
        assert len(restarted.requests) == 1


def test_hosted_followup_with_unknown_history_fails_closed():
    fake = _FakeChatTransport([])
    with TestClient(_model_loop_app(_spec(), fake)) as client:
        response = client.post("/responses", json={"input": "Continue", "agent_session_id": "session-a", "previous_response_id": "unknown-response"})
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "unknown_previous_response_id"
        assert not fake.requests


@pytest.mark.parametrize("session_id", [None, "session-a", "session-b"])
def test_expired_hosted_history_rejects_followup_even_without_session_id(session_id):
    fake = _FakeChatTransport([
        _chat_response({"role": "assistant", "content": "Your site is site-a."}),
        _chat_response({"role": "assistant", "content": "Must not run without the expired context."}),
    ])
    with TestClient(_model_loop_app(_spec(), fake, state_ttl_seconds=0)) as client:
        first = client.post("/responses", json={"input": "My site is site-a", "agent_session_id": "session-a"})
        assert first.status_code == 200, first.text
        payload = {"input": "Which site was that?", "previous_response_id": first.json()["id"]}
        if session_id is not None:
            payload["agent_session_id"] = session_id
        followup = client.post("/responses", json=payload)
        assert followup.status_code == 410, followup.text
        assert followup.json()["error"]["code"] == "response_state_expired"
        assert len(fake.requests) == 1
