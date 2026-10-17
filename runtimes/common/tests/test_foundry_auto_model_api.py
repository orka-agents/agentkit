"""Raw HTTP auto negotiation and concrete hosted continuation pinning."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.foundry_model_loop import BrokeredChatModelLoop, ModelLoopFinal
from agentkit_serve_common.runtime import AgentRunError
from _foundry_model_api_cases import _assert_model_request, _model_response
from test_foundry_model_retries import RejectedBody
from test_foundry_brokered_protocol import (
    CONTINUATION_AUTH,
    _app,
    _call,
    _chat_response,
    _continuation,
    _message_text,
    _spec,
)

UNSUPPORTED = {"error": {"code": "unsupported_endpoint", "message": "The /v1/responses endpoint is not supported."}}


@pytest.fixture(autouse=True)
def auto_model_api(monkeypatch):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "auto")


def _reply(api, *, tool=False):
    message = {"role": "assistant", "content": "Done."}
    if tool:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "provider-call", "type": "function", "function": {
                "name": "conformance_read", "arguments": '{"probe":true}',
            }}],
        }
    return _model_response(_chat_response(message), model_api=api)


def _model_app(handler, **kwargs):
    return _app(
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


@pytest.mark.parametrize("chat_only", [False, True], ids=["both", "chat-only"])
def test_auto_first_real_request_selects_and_caches_wire_api(chat_only, tmp_path):
    paths = []
    accepted = []
    api = "chat_completions" if chat_only else "responses"

    def model(request):
        paths.append(request.url.path)
        if chat_only and request.url.path.endswith("/responses"):
            _assert_model_request(request, model_api="responses")
            return httpx.Response(404, json=UNSUPPORTED)
        _assert_model_request(request, model_api=api)
        body = json.loads(request.content)
        accepted.append(body)
        assert body["parallel_tool_calls"] is False
        return httpx.Response(200, json=_reply(api, tool=len(accepted) == 1))

    state_file = tmp_path / "states.json"
    with TestClient(_model_app(model, response_state_file=state_file)) as client:
        assert client.get("/readiness").status_code == 200
        assert paths == []
        initial = client.post("/responses", json={"input": "Read telemetry."})
        assert initial.status_code == 200, initial.text
        pending = initial.json()
        stored = json.loads(state_file.read_text())["states"][pending["id"]]
        assert stored["modelAPI"] == api
        continuation = _continuation(pending["id"], _call(pending)["call_id"], {"approved": True, "output": {"ok": True}})
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)
        assert completed.status_code == 200, completed.text
        assert _message_text(completed.json()) == "Done."
        again = client.post("/responses", json={"input": "New turn."})
        assert again.status_code == 200, again.text

    assert paths == (["/v1/responses"] if chat_only else []) + [
        "/v1/chat/completions" if chat_only else "/v1/responses",
    ] * 3
    if chat_only:
        assert accepted[1]["messages"][-1]["role"] == "tool"
    else:
        assert accepted[1]["input"][-1]["type"] == "function_call_output"


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (400, {"error": {"message": "Invalid parameter: input"}}, "ModelUpstreamError"),
        (404, {"error": {"message": "Not found"}}, "ModelUpstreamError"),
        (404, {"error": {"code": "model_not_found", "message": "Model not found"}}, "ModelUpstreamError"),
        (401, UNSUPPORTED, "ModelAuthRejected"),
        (403, UNSUPPORTED, "ModelAuthRejected"),
        (408, UNSUPPORTED, "ModelUpstreamError"),
        (429, UNSUPPORTED, "ModelUnavailable"),
        (500, UNSUPPORTED, "ModelUnavailable"),
        (503, UNSUPPORTED, "ModelUnavailable"),
        (None, None, "ModelUpstreamError"),
    ],
)
def test_auto_does_not_fallback_for_other_errors(status, body, code):
    paths = []

    def model(request):
        paths.append(request.url.path)
        _assert_model_request(request, model_api="responses")
        if status is None:
            raise httpx.ReadTimeout("private-upstream-detail", request=request)
        return httpx.Response(status, json=body, headers={"retry-after": "61"})

    with TestClient(_model_app(model)) as client:
        response = client.post("/responses", json={"input": "Read telemetry."})
    assert response.status_code >= 400
    assert response.json()["error"]["code"] == code
    assert "private-upstream-detail" not in response.text
    assert paths == ["/v1/responses"]


@pytest.mark.parametrize("status", [401, 403, 408, 429, 500, 503])
def test_auto_unrelated_error_bodies_remain_unread(status):
    body = RejectedBody()

    def model(request):
        _assert_model_request(request, model_api="responses")
        return httpx.Response(status, stream=body, headers={"retry-after": "61"})

    with TestClient(_model_app(model)) as client:
        assert client.post("/responses", json={"input": "Read telemetry."}).status_code >= 400
    assert body.closed and not body.read


def test_auto_chat_fallback_is_cached_even_when_chat_fails():
    async def run():
        paths = []

        def model(request):
            paths.append(request.url.path)
            if len(paths) == 1:
                return httpx.Response(404, json=UNSUPPORTED)
            _assert_model_request(request, model_api="chat_completions")
            if len(paths) == 2:
                raise httpx.ReadTimeout("private-upstream-detail", request=request)
            return httpx.Response(200, json=_reply("chat_completions"))

        async with httpx.AsyncClient(transport=httpx.MockTransport(model)) as client:
            loop = BrokeredChatModelLoop(_spec(), [], http_client=client)
            with pytest.raises(AgentRunError):
                await loop.start(RunRequest(prompt="First."), call_id="first")
            assert loop.model_api == "chat_completions"
            result = await loop.start(RunRequest(prompt="Second."), call_id="second")
            assert isinstance(result, ModelLoopFinal) and result.text == "Done."
        assert paths == ["/v1/responses", "/v1/chat/completions", "/v1/chat/completions"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "body",
    [b"not-json", b"[]", b"{}", b'{"status":"incomplete","output":[]}', b" " * 257],
    ids=["invalid-json", "non-object", "missing-output", "incomplete", "too-large"],
)
def test_auto_acceptance_pins_responses_before_body_validation(body):
    async def run():
        paths = []

        def model(request):
            paths.append(request.url.path)
            if len(paths) == 1:
                return httpx.Response(200, content=body)
            return httpx.Response(404, json=UNSUPPORTED)

        async with httpx.AsyncClient(transport=httpx.MockTransport(model)) as client:
            loop = BrokeredChatModelLoop(_spec(), [], http_client=client, max_response_bytes=256)
            with pytest.raises(AgentRunError) as initial:
                await loop.start(RunRequest(prompt="First."), call_id="first")
            assert initial.value.code in {"InvalidModelResponse", "ModelResponseTooLarge"}
            with pytest.raises(AgentRunError) as later:
                await loop.start(RunRequest(prompt="Second."), call_id="second")
            assert later.value.code == "ModelUpstreamError"
            assert loop.model_api == "responses"
        assert paths == ["/v1/responses", "/v1/responses"]

    asyncio.run(run())


def test_auto_serializes_first_requests_and_rebuilds_waiting_chat_payloads():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        paths = []

        async def model(request):
            paths.append(request.url.path)
            if request.url.path.endswith("/responses"):
                _assert_model_request(request, model_api="responses")
                started.set()
                await release.wait()
                return httpx.Response(404, json=UNSUPPORTED)
            _assert_model_request(request, model_api="chat_completions")
            return httpx.Response(200, json=_reply("chat_completions"))

        async with httpx.AsyncClient(transport=httpx.MockTransport(model)) as client:
            loop = BrokeredChatModelLoop(_spec(), [], http_client=client)
            first = asyncio.create_task(loop.start(RunRequest(prompt="First."), call_id="first"))
            await asyncio.wait_for(started.wait(), 2)
            second = asyncio.create_task(loop.start(RunRequest(prompt="Second."), call_id="second"))
            await asyncio.sleep(0)
            assert paths == ["/v1/responses"]
            release.set()
            results = await asyncio.wait_for(asyncio.gather(first, second), 2)
            assert all(isinstance(result, ModelLoopFinal) and result.text == "Done." for result in results)
            assert loop.model_api == "chat_completions"
        assert paths == ["/v1/responses", "/v1/chat/completions", "/v1/chat/completions"]

    asyncio.run(run())


@pytest.mark.parametrize(("api", "legacy_chat"), [("chat_completions", False), ("responses", False), ("chat_completions", True)])
def test_auto_restart_honors_pending_api_and_existing_guards(api, legacy_chat, monkeypatch, tmp_path):
    state_file = tmp_path / "states.json"
    monkeypatch.setenv("AGENTKIT_MODEL_API", api)
    with TestClient(_model_app(lambda request: httpx.Response(200, json=_reply(api, tool=True)), response_state_file=state_file)) as client:
        response = client.post("/responses", json={"input": "Read telemetry.", "agent_session_id": "session-a"})
        assert response.status_code == 200, response.text
        initial = response.json()
    if legacy_chat:
        data = json.loads(state_file.read_text())
        del data["states"][initial["id"]]["modelAPI"]
        state_file.write_text(json.dumps(data))
    continuation = {
        **_continuation(initial["id"], _call(initial)["call_id"], {"approved": True, "output": {"ok": True}}),
        "agent_session_id": "session-a",
    }
    paths = []

    def model(request):
        paths.append(request.url.path)
        _assert_model_request(request, model_api=api)
        return httpx.Response(200, json=_reply(api, tool=len(paths) == 1))

    # Explicit mismatch remains rejected without consuming the continuation.
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses" if api == "chat_completions" else "chat_completions")
    with TestClient(_model_app(model, response_state_file=state_file)) as client:
        mismatch = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)
        assert mismatch.status_code == 409
        assert mismatch.json()["error"]["code"] == "brokered_model_api_mismatch"
    assert paths == []

    monkeypatch.setenv("AGENTKIT_MODEL_API", "auto")
    with TestClient(_model_app(model, response_state_file=state_file, max_brokered_output_bytes=128)) as client:
        forbidden = client.post("/responses", json=continuation)
        assert forbidden.status_code == 403
        wrong_session = client.post("/responses", headers=CONTINUATION_AUTH, json={**continuation, "agent_session_id": "session-b"})
        assert wrong_session.status_code == 409
        oversized = _continuation(initial["id"], _call(initial)["call_id"], {"output": "x" * 256})
        oversized["agent_session_id"] = "session-a"
        assert client.post("/responses", headers=CONTINUATION_AUTH, json=oversized).status_code == 413
        assert paths == []
        advanced = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)
        assert advanced.status_code == 200, advanced.text
        stored = json.loads(state_file.read_text())["states"][initial["id"]]
        assert stored["modelAPI"] == api
        replay = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)
        assert replay.json() == advanced.json()
        assert len(paths) == 1
        pending = advanced.json()
        final = client.post("/responses", headers=CONTINUATION_AUTH, json={
            **_continuation(pending["id"], _call(pending)["call_id"], {"approved": True, "output": {"ok": True}}),
            "agent_session_id": "session-a",
        })
        assert final.status_code == 200, final.text
        assert _message_text(final.json()) == "Done."
    assert paths == ["/v1/chat/completions" if api == "chat_completions" else "/v1/responses"] * 2


@pytest.mark.parametrize("cached_api", [None, "chat_completions"])
def test_auto_pinned_responses_continuation_never_falls_back(cached_api, monkeypatch, tmp_path):
    state_file = tmp_path / "states.json"
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")
    with TestClient(_model_app(lambda request: httpx.Response(200, json=_reply("responses", tool=True)), response_state_file=state_file)) as client:
        initial = client.post("/responses", json={"input": "Read telemetry."}).json()
    monkeypatch.setenv("AGENTKIT_MODEL_API", "auto")
    paths = []

    def model(request):
        paths.append(request.url.path)
        api = "responses" if request.url.path.endswith("/responses") else "chat_completions"
        _assert_model_request(request, model_api=api)
        return httpx.Response(404, json=UNSUPPORTED) if api == "responses" else httpx.Response(200, json=_reply(api))

    with TestClient(_model_app(model, response_state_file=state_file)) as client:
        if cached_api:
            assert client.post("/responses", json={"input": "New turn."}).status_code == 200
            assert paths == ["/v1/responses", "/v1/chat/completions"]
            paths.clear()
        continuation = _continuation(initial["id"], _call(initial)["call_id"], {"approved": True, "output": {"ok": True}})
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)
        assert failed.status_code == 502
        assert failed.json()["error"]["code"] == "ModelUpstreamError"
        stored = json.loads(state_file.read_text())["states"][initial["id"]]
        assert stored["modelAPI"] == "responses" and stored["status"] == "pending"
        assert paths == ["/v1/responses"]
        if cached_api:
            assert client.post("/responses", json={"input": "Another new turn."}).status_code == 200
            assert paths == ["/v1/responses", "/v1/chat/completions"]


@pytest.mark.parametrize("api", [None, "chat_completions", "responses"])
def test_auto_does_not_change_explicit_or_default_selection(api, monkeypatch):
    if api is None:
        monkeypatch.delenv("AGENTKIT_MODEL_API")
    else:
        monkeypatch.setenv("AGENTKIT_MODEL_API", api)
    paths = []

    def model(request):
        paths.append(request.url.path)
        _assert_model_request(request, model_api=api or "chat_completions")
        return httpx.Response(404, json=UNSUPPORTED)

    with TestClient(_model_app(model)) as client:
        assert client.post("/responses", json={"input": "Read telemetry."}).status_code == 502
    assert paths == ["/v1/responses" if api == "responses" else "/v1/chat/completions"]
