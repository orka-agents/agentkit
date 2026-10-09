"""Startup API selection and public brokered workflow transport regressions."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from agentkit_serve_common.adapter_support import AgentBuildError, MODEL_API_ENV, resolve_model_api
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.foundry_model_loop import BrokeredChatModelLoop
from test_foundry_brokered_protocol import (
    CONTINUATION_AUTH, _FakeChatTransport, _call, _chat_response, _continuation,
    _message_text, _model_loop_app, _spec,
)


@pytest.fixture(autouse=True)
def clear_model_api(monkeypatch):
    monkeypatch.delenv(MODEL_API_ENV, raising=False)


def _reply(api, *, tool=False):
    message = {"role": "assistant", "content": None if tool else "Done."}
    if tool:
        message["tool_calls"] = [{
            "id": "provider-call", "type": "function",
            "function": {"name": "conformance_read", "arguments": '{"probe":true}'},
        }]
    if api == "responses":
        return _chat_response(message, prompt_tokens=1, completion_tokens=2)
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if tool else "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


def test_default_api_is_chat_completions():
    assert resolve_model_api() == "chat_completions"


@pytest.mark.parametrize("value", ["", "chat", "automatic", "anthropic_messages", " responses ", "private-invalid-value"])
def test_invalid_api_is_rejected_without_echoing_value(monkeypatch, value):
    monkeypatch.setenv(MODEL_API_ENV, value)
    with pytest.raises(AgentBuildError) as error:
        resolve_model_api()
    assert MODEL_API_ENV in str(error.value)
    if value == "private-invalid-value":
        assert value not in str(error.value)


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_supported_api_is_resolved(monkeypatch, api):
    monkeypatch.setenv(MODEL_API_ENV, api)
    assert resolve_model_api() == api


def test_runtime_cannot_silently_ignore_api_selection(monkeypatch):
    monkeypatch.setenv(MODEL_API_ENV, "responses")
    with pytest.raises(AgentBuildError, match="does not support"):
        resolve_model_api(supported={"chat_completions"}, runtime="chat-only test runtime")


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.parametrize("persisted", [False, True])
def test_brokered_workflow_uses_selected_api_after_resume(monkeypatch, tmp_path, api, persisted):
    monkeypatch.setenv(MODEL_API_ENV, api)
    fake = _FakeChatTransport([_reply(api, tool=True), _reply(api)])
    kwargs = {"response_state_file": tmp_path / "state.json"} if persisted else {}
    app = _model_loop_app(_spec(), fake, **kwargs)
    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "Read the probe."})
    assert initial.status_code == 200, initial.text
    call = _call(initial.json())
    if persisted:
        serialized = json.loads(kwargs["response_state_file"].read_text())
        assert serialized["states"][initial.json()["id"]]["modelAPI"] == api
        app = _model_loop_app(_spec(), fake, **kwargs)
    with TestClient(app) as client:
        final = client.post("/responses", headers=CONTINUATION_AUTH, json=_continuation(
            initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}},
        ))
    assert final.status_code == 200, final.text
    assert _message_text(final.json()) == "Done."
    assert final.json()["usage"] == {"input_tokens": 2, "output_tokens": 4, "total_tokens": 6}
    suffix = "/responses" if api == "responses" else "/chat/completions"
    assert fake.urls == ["https://api.openai.com/v1" + suffix] * 2
    for request in fake.requests:
        assert request["parallel_tool_calls"] is False
        assert request["tool_choice"] == "auto"
    if api == "responses":
        assert fake.requests[0]["store"] is False
        assert fake.requests[0]["tools"][0]["name"] == "conformance_read"
        assert fake.requests[1]["input"][-1]["type"] == "function_call_output"
        assert fake.requests[1]["input"][-1]["call_id"] == call["call_id"]
        assert "messages" not in fake.requests[0]
    else:
        assert "input" not in fake.requests[0]
        assert fake.requests[0]["tools"][0]["function"]["name"] == "conformance_read"
        assert fake.requests[1]["messages"][-1]["role"] == "tool"
        assert fake.requests[1]["messages"][-1]["tool_call_id"] == call["call_id"]


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_pending_workflow_rejects_api_change_before_upstream_call(monkeypatch, tmp_path, api):
    monkeypatch.setenv(MODEL_API_ENV, api)
    path = tmp_path / "state.json"
    fake = _FakeChatTransport([_reply(api, tool=True), _reply(api)])
    with TestClient(_model_loop_app(_spec(), fake, response_state_file=path)) as client:
        initial = client.post("/responses", json={"input": "Read the probe."})
    assert initial.status_code == 200, initial.text
    call = _call(initial.json())
    continuation = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
    before = path.read_bytes()
    other = "responses" if api == "chat_completions" else "chat_completions"
    monkeypatch.setenv(MODEL_API_ENV, other)
    with TestClient(_model_loop_app(_spec(), fake, response_state_file=path)) as client:
        rejected = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)
    assert rejected.status_code == 409, rejected.text
    assert rejected.json()["error"]["code"] == "brokered_model_api_mismatch"
    assert len(fake.requests) == 1
    assert path.read_bytes() == before
    monkeypatch.setenv(MODEL_API_ENV, api)
    with TestClient(_model_loop_app(_spec(), fake, response_state_file=path)) as client:
        resumed = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)
    assert resumed.status_code == 200, resumed.text
    assert len(fake.requests) == 2


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_model_api_does_not_fallback_on_endpoint_error(monkeypatch, api):
    monkeypatch.setenv(MODEL_API_ENV, api)
    requests = []
    def reject(request):
        requests.append(request)
        return httpx.Response(404, json={"error": "endpoint unsupported"})
    fake = _FakeChatTransport([])
    fake.handler = reject
    with TestClient(_model_loop_app(_spec(), fake)) as client:
        response = client.post("/responses", json={"input": "Read the probe."})
    assert response.status_code == 502, response.text
    assert response.json()["error"]["upstream_status"] == 404
    assert len(requests) == 1
    suffix = "/responses" if api == "responses" else "/chat/completions"
    assert str(requests[0].url).endswith(suffix)


def test_model_api_is_captured_once_and_cannot_be_overridden_per_turn(monkeypatch):
    fake = _FakeChatTransport([_reply("chat_completions")])
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    loop = BrokeredChatModelLoop(_spec(), [], http_client=client)
    monkeypatch.setenv(MODEL_API_ENV, "responses")
    async def run():
        async with client:
            return await loop.start(RunRequest(prompt="Hello.", env={MODEL_API_ENV: "responses"}), call_id="host-call")
    assert asyncio.run(run()).text == "Done."
    assert loop.model_api == "chat_completions"
    assert fake.urls == ["https://api.openai.com/v1/chat/completions"]


@pytest.mark.parametrize("finish_reason", [None, "length", "content_filter", "unknown"])
def test_chat_incomplete_output_does_not_execute_tools(finish_reason):
    reply = _reply("chat_completions", tool=True)
    reply["choices"][0]["finish_reason"] = finish_reason
    fake = _FakeChatTransport([reply])
    with TestClient(_model_loop_app(_spec(), fake)) as client:
        response = client.post("/responses", json={"input": "Read the probe."})
    assert response.status_code == 502, response.text
    assert response.json()["error"]["code"] == "InvalidModelResponse"


def test_unmarked_legacy_state_resumes_with_chat_default(tmp_path):
    path = tmp_path / "legacy.json"
    fake = _FakeChatTransport([_reply("chat_completions", tool=True), _reply("chat_completions")])
    with TestClient(_model_loop_app(_spec(), fake, response_state_file=path)) as client:
        initial = client.post("/responses", json={"input": "Read the probe."})
    assert initial.status_code == 200, initial.text
    stored = json.loads(path.read_text())
    del stored["states"][initial.json()["id"]]["modelAPI"]
    path.write_text(json.dumps(stored))
    call = _call(initial.json())
    with TestClient(_model_loop_app(_spec(), fake, response_state_file=path)) as client:
        final = client.post("/responses", headers=CONTINUATION_AUTH, json=_continuation(
            initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}},
        ))
    assert final.status_code == 200, final.text
    assert fake.urls == ["https://api.openai.com/v1/chat/completions"] * 2


def test_unknown_model_api_is_rejected_by_cli_before_loading_config(monkeypatch, capsys):
    from agentkit_serve_common import cli
    monkeypatch.setenv(MODEL_API_ENV, "private-invalid-value")
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid startup API must be rejected before loading the agent")
    monkeypatch.setattr(cli, "_load_spec_or_exit", unexpected)
    with pytest.raises(SystemExit) as error:
        cli.run(object(), ["--config", "unused.yaml"])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert MODEL_API_ENV in stderr
    assert "private-invalid-value" not in stderr
