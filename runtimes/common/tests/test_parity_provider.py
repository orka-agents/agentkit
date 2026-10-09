"""Bounded wire fixture checks, independent of any runtime adapter or model SDK."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from agentkit_serve_common.parity import (
    _answer,
    _canary_env,
    _conversation,
    _model_wire_response,
    _Reply,
    _request_messages,
    _responses_body,
    _ScriptedProvider,
    _tool_calls,
    _tool_definitions,
    _tool_results,
)
from agentkit_serve_common.parity import (
    pytest_generate_tests as generate_tests,
)
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient


def _request(api, prompt="plain:MARK", *, stream=False):
    body = {"model": "parity-model", "stream": stream}
    message = {"role": "user", "content": prompt}
    tool = {"name": "parity_echo", "parameters": {"type": "object", "required": ["value"]}}
    if api == "chat_completions":
        return {**body, "messages": [message], "tools": [{"type": "function", "function": tool}]}
    return {**body, "input": [message], "tools": [{"type": "function", **tool}], "store": False}


def _events(text):
    return [json.loads(line.removeprefix("data: ")) for line in text.splitlines() if line.startswith("data: {")]


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_calls", [False, True])
def test_loopback_provider_emits_native_json_and_sse(api, stream, tool_calls):
    request = _request(api, stream=stream)
    reply = (
        _tool_calls(("parity_echo", '{"value":"A"}'), ("parity_echo", '{"value":"B"}'))
        if tool_calls
        else _answer("MARK")
    )
    path = "/responses" if api == "responses" else "/chat/completions"
    with _ScriptedProvider(api) as provider:
        provider.script = lambda body: reply
        response = httpx.post(provider.base_url + path, json=request, headers={"authorization": "Bearer test-key"})
        assert provider.requests == [("Bearer test-key", request)]
        assert provider.paths == ["/v1" + path]
    assert response.status_code == 200
    if stream:
        assert response.headers["content-type"] == "text/event-stream"
        events = _events(response.text)
        if api == "responses":
            assert [event["sequence_number"] for event in events] == list(range(len(events)))
            assert events[0]["type"] == "response.created"
            assert events[0]["response"]["status"] == "in_progress"
            assert events[0]["response"]["output"] == []
            assert events[-1]["type"] == "response.completed"
            result = events[-1]["response"]
            added = [event for event in events if event["type"] == "response.output_item.added"]
            done = [event for event in events if event["type"] == "response.output_item.done"]
            assert [event["item"]["id"] for event in added] == [event["item"]["id"] for event in done]
            delta_type = "response.function_call_arguments.delta" if tool_calls else "response.output_text.delta"
            deltas = [event for event in events if event["type"] == delta_type]
            assert [event["delta"] for event in deltas] == (
                ['{"value":"A"}', '{"value":"B"}'] if tool_calls else ["MARK"]
            )
            assert all(f"event: {event['type']}\n" in response.text for event in events)
        else:
            assert response.text.endswith("data: [DONE]\n\n")
            assert len(events) == 2
            assert events[-1]["choices"][0]["finish_reason"] == ("tool_calls" if tool_calls else "stop")
            message = events[0]["choices"][0]["delta"]
            assert [call["index"] for call in message.get("tool_calls", [])] == ([0, 1] if tool_calls else [])
    else:
        assert response.headers["content-type"] == "application/json"
        result = response.json()
        if api == "chat_completions":
            message = result["choices"][0]["message"]
            assert result["choices"][0]["finish_reason"] == ("tool_calls" if tool_calls else "stop")
    if api == "responses":
        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["model"] == "parity-model"
        assert result["usage"]["total_tokens"] == 5
        if tool_calls:
            assert [(item["type"], item["call_id"], item["name"], item["arguments"]) for item in result["output"]] == [
                ("function_call", "call_parity_0", "parity_echo", '{"value":"A"}'),
                ("function_call", "call_parity_1", "parity_echo", '{"value":"B"}'),
            ]
        else:
            assert result["output"][0]["role"] == "assistant"
            assert result["output"][0]["content"][0]["text"] == "MARK"
    else:
        expected = (
            reply.message
            if not stream or not tool_calls
            else {
                **reply.message,
                "tool_calls": [{**call, "index": index} for index, call in enumerate(reply.message["tool_calls"])],
            }
        )
        assert message == expected


@pytest.mark.parametrize("response_status", ["completed", "incomplete", "failed", "in_progress"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_calls", [False, True])
def test_responses_provider_preserves_terminal_status_and_unfinished_eof(response_status, stream, tool_calls):
    reply = _tool_calls(("parity_echo", '{"value":"MUST-NOT-RUN"}')) if tool_calls else _answer("partial text")
    body = _responses_body("parity-model", reply.message)
    body["status"] = response_status
    if response_status == "incomplete":
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    elif response_status == "failed":
        body["error"] = {"code": "server_error", "message": "scripted failure"}
    original = json.dumps(body)
    with _ScriptedProvider("responses") as provider:
        provider.script = lambda request: _Reply(body=body)
        response = httpx.post(provider.base_url + "/responses", json=_request("responses", stream=stream))
    assert response.status_code == 200
    assert json.dumps(body) == original
    if not stream:
        assert response.headers["content-type"] == "application/json"
        assert response.json() == body
        return
    assert response.headers["content-type"] == "text/event-stream"
    events = _events(response.text)
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert events[0]["type"] == "response.created"
    assert events[0]["response"]["status"] == "in_progress"
    assert events[0]["response"]["error"] is None
    assert events[0]["response"]["incomplete_details"] is None
    terminal = [
        event for event in events if event["type"] in {"response.completed", "response.incomplete", "response.failed"}
    ]
    if response_status == "in_progress":
        assert terminal == []
        assert events[-1]["type"] == "response.output_item.done"
    else:
        assert terminal == [
            {"type": "response." + response_status, "sequence_number": len(events) - 1, "response": body}
        ]
        assert events[-1] == terminal[0]
        assert f"event: response.{response_status}\n" in response.text
    delta_type = "response.function_call_arguments.delta" if tool_calls else "response.output_text.delta"
    assert [event["delta"] for event in events if event["type"] == delta_type] == (
        ['{"value":"MUST-NOT-RUN"}'] if tool_calls else ["partial text"]
    )
    assert "data: [DONE]" not in response.text


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.parametrize(
    "reply", [_Reply(status=401, body={"error": {"message": "test-key"}}), _Reply(body=b"not-json")]
)
def test_provider_passes_scripted_error_and_malformed_payloads_unchanged(api, reply):
    content_type, payload = _model_wire_response(_request(api, stream=True), reply, api)
    assert content_type == "application/json"
    assert payload == (reply.body if isinstance(reply.body, bytes) else json.dumps(reply.body).encode())


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_provider_rejects_wrong_api_endpoint(api):
    with _ScriptedProvider(api) as provider:
        path = "/chat/completions" if api == "responses" else "/responses"
        response = httpx.post(provider.base_url + path, json=_request(api))
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "wrong upstream API"


def test_responses_wire_history_and_function_outputs_are_read_without_mutation():
    body = {
        "instructions": "baked instructions",
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": "client note"}]},
            {"role": "user", "content": "first"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
            {"type": "function_call", "call_id": "call_1", "name": "parity_echo", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "receipt-MARK"},
            {"role": "user", "content": "second"},
        ],
    }
    original = json.dumps(body)
    assert _conversation(body) == [
        ("system", "baked instructions"),
        ("system", "client note"),
        ("user", "first"),
        ("assistant", "answer"),
        ("user", "second"),
    ]
    assert _tool_results(body) == {"call_1": "receipt-MARK"}
    assert _request_messages(body)[4]["tool_calls"][0]["id"] == "call_1"
    assert json.dumps(body) == original
    chat_tool = _tool_definitions(_request("chat_completions"))[0]
    responses_tool = _tool_definitions(_request("responses"))[0]
    assert responses_tool["type"] == "function"
    assert responses_tool["name"] == chat_tool["name"]
    assert responses_tool["parameters"] == chat_tool["parameters"]


def test_inherited_suite_collects_both_startup_apis_and_responses_statuses(monkeypatch):
    calls = []
    generate_tests(
        SimpleNamespace(fixturenames=["model_api", "response_status"], parametrize=lambda *args: calls.append(args))
    )
    assert calls == [
        ("model_api", ["chat_completions", "responses"]),
        ("response_status", ["incomplete", "failed", "in_progress"]),
    ]
    monkeypatch.setenv("AGENTKIT_MODEL_API", "original")
    with _canary_env("responses"):
        assert os.environ["AGENTKIT_MODEL_API"] == "responses"
    assert os.environ["AGENTKIT_MODEL_API"] == "original"


@pytest.fixture
def container_fixture():
    path = Path(__file__).resolve().parents[3] / "test/parity/fixture.py"
    spec = importlib.util.spec_from_file_location("parity_container_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture_client(container_fixture):
    app = Starlette(
        routes=[
            Route("/v1/chat/completions", container_fixture.chat_completions, methods=["POST"]),
            Route("/v1/responses", container_fixture.responses, methods=["POST"]),
        ]
    )
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_container_fixture_uses_same_native_wire_encoders(fixture_client, api, stream):
    path = "/v1/responses" if api == "responses" else "/v1/chat/completions"
    for prompt, expected in [("plain:MARK", "parity-answer: MARK"), ("api:responses", "parity-api: " + api)]:
        response = fixture_client.post(path, json=_request(api, prompt, stream=stream))
        assert response.status_code == 200
        assert expected in response.text
    response = fixture_client.post(
        path, json=_request(api, "auth-echo", stream=stream), headers={"authorization": "Bearer test-key"}
    )
    assert response.status_code == 401
    assert response.json() == {"error": {"message": "invalid key Bearer test-key"}}


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_container_fixture_native_tool_roundtrip(fixture_client, api, stream):
    path = "/v1/responses" if api == "responses" else "/v1/chat/completions"
    request = _request(api, "tool:MARK", stream=stream)
    response = fixture_client.post(path, json=request)
    assert response.status_code == 200
    if api == "responses":
        result = _events(response.text)[-1]["response"] if stream else response.json()
        call = result["output"][0]
        assert (call["type"], call["call_id"], call["name"]) == ("function_call", "call_parity_0", "parity_echo")
        assert json.loads(call["arguments"]) == {"value": "MARK"}
        request["input"].extend(
            [call, {"type": "function_call_output", "call_id": call["call_id"], "output": "receipt-MARK"}]
        )
    else:
        message = (
            _events(response.text)[0]["choices"][0]["delta"] if stream else response.json()["choices"][0]["message"]
        )
        call = message["tool_calls"][0]
        assert (call["id"], call["function"]["name"]) == ("call_parity_0", "parity_echo")
        assert json.loads(call["function"]["arguments"]) == {"value": "MARK"}
        request["messages"].extend([message, {"role": "tool", "tool_call_id": call["id"], "content": "receipt-MARK"}])
    response = fixture_client.post(path, json=request)
    assert response.status_code == 200
    assert "tool-result: receipt-MARK" in response.text


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_container_fixture_tool_receipts_are_current_turn_only(container_fixture, api):
    request = _request(api, "tool:NEW")
    key = "input" if api == "responses" else "messages"
    old = [{"role": "user", "content": "tool:OLD"}]
    old.extend(
        [{"type": "function_call_output", "call_id": "old_call", "output": "receipt-OLD"}]
        if api == "responses"
        else [{"role": "tool", "tool_call_id": "old_call", "content": "receipt-OLD"}]
    )
    request[key] = old + request[key]
    reply = container_fixture._script_reply(request, api, "")
    assert reply.message["tool_calls"][0]["function"]["arguments"] == '{"value": "NEW"}'
    request[key].append(
        {"type": "function_call_output", "call_id": "call_parity_0", "output": "receipt-NEW"}
        if api == "responses"
        else {"role": "tool", "tool_call_id": "call_parity_0", "content": "receipt-NEW"}
    )
    reply = container_fixture._script_reply(request, api, "")
    assert reply.message["content"] == "tool-result: receipt-NEW"


@pytest.mark.parametrize(
    "invalid",
    [
        {"store": True},
        {"store": None},
        {"previous_response_id": "resp_old"},
        {"conversation": "conv_old"},
        {"messages": []},
    ],
)
def test_container_fixture_rejects_provider_managed_responses_state(fixture_client, invalid):
    response = fixture_client.post("/v1/responses", json={**_request("responses"), **invalid})
    assert response.status_code == 400


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_container_fixture_reports_exact_text_history(container_fixture, api):
    request = _request(api, "history:MARK")
    key = "input" if api == "responses" else "messages"
    request[key] = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ] + request[key]
    reply = container_fixture._script_reply(request, api, "")
    assert json.loads(reply.message["content"].removeprefix("parity-history: ")) == [
        ["system", "instructions"],
        ["user", "q1"],
        ["assistant", "a1"],
        ["user", "history:MARK"],
    ]
