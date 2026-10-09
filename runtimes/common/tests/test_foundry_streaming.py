"""Hosted Responses acknowledgements, identity, and disconnect regressions."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from agentkit_serve_common.foundry import create_foundry_app
from agentkit_serve_common.runtime import AgentRunError, RunResult
from test_foundry_brokered_protocol import (
    responses_model_api,
    CONTINUATION_AUTH,
    _app,
    _call,
    _chat_response,
    _continuation,
    _spec,
)
from test_foundry_protocol import EchoFactory, EchoRuntime
from test_foundry_protocol import _spec as native_spec


def _tool_response():
    return _chat_response(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "model-call",
                    "type": "function",
                    "function": {
                        "name": "conformance_read",
                        "arguments": '{"probe":true}',
                    },
                }
            ],
        }
    )


class HeldModel(httpx.AsyncBaseTransport):
    def __init__(self, result="text", *, continuation=False):
        self.result = result
        self.held_call = 2 if continuation else 1
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def handle_async_request(self, request):
        self.calls += 1
        if self.calls < self.held_call:
            return httpx.Response(200, request=request, json=_tool_response())
        if self.calls == self.held_call:
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        if self.result == "error":
            return httpx.Response(
                401, request=request, json={"error": "private-upstream-detail"}
            )
        if self.result == "exception":
            raise RuntimeError("private-upstream-detail")
        payload = (
            _tool_response()
            if self.result == "tool"
            else _chat_response({"role": "assistant", "content": "Verified response."})
        )
        return httpx.Response(200, request=request, json=payload)


@asynccontextmanager
async def _exchange(app, payload, *, hold_created=None, headers=None):
    incoming, outgoing = asyncio.Queue(), asyncio.Queue()
    await incoming.put(
        {
            "type": "http.request",
            "body": json.dumps(payload).encode(),
            "more_body": False,
        }
    )

    async def send(message):
        await outgoing.put(message)
        if (
            hold_created is not None
            and message["type"] == "http.response.body"
            and message.get("more_body")
        ):
            await hold_created.wait()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/responses",
        "raw_path": b"/responses",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            *[
                (key.encode(), value.encode())
                for key, value in (
                    CONTINUATION_AUTH if headers is None else headers
                ).items()
            ],
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    task = asyncio.create_task(app(scope, incoming.get, send))
    try:
        yield incoming, outgoing, task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _created(outgoing):
    headers = await asyncio.wait_for(outgoing.get(), 2)
    assert headers["type"] == "http.response.start" and headers["status"] == 200
    assert dict(headers["headers"])[b"content-type"].startswith(b"text/event-stream")
    assert b"content-length" not in dict(headers["headers"])
    return await _event(outgoing)


async def _event(outgoing):
    message = await asyncio.wait_for(outgoing.get(), 2)
    assert message["type"] == "http.response.body"
    assert message["body"].endswith(b"\n\n")
    data = next(
        line.removeprefix(b"data: ")
        for line in message["body"].splitlines()
        if line.startswith(b"data: ")
    )
    event = json.loads(data)
    assert type(event["sequence_number"]) is int
    assert event["sequence_number"] == (0 if event["type"] == "response.created" else 1)
    return event


@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize("result", ["text", "tool"])
def test_brokered_stream_ack_precedes_model_and_matches_completion_and_replay(
    continuation, result, monkeypatch, tmp_path
):
    clock = [1_700_000_000]
    monkeypatch.setattr("agentkit_serve_common.foundry.time.time", lambda: clock[0])
    state_file = tmp_path / "responses.json"

    async def exercise():
        model = HeldModel(result, continuation=continuation)
        async with httpx.AsyncClient(transport=model) as upstream:
            app = _app(
                _spec(),
                brokered_model_loop_enabled=True,
                brokered_model_http_client=upstream,
                response_state_file=state_file,
            )
            payload = {
                "input": "Read the synthetic data",
                "agent_session_id": "session-a",
                "stream": True,
            }
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                if continuation:
                    initial = (
                        await client.post(
                            "/responses",
                            json={
                                "input": "Read data",
                                "agent_session_id": "session-a",
                            },
                        )
                    ).json()
                    payload = {
                        **_continuation(
                            initial["id"],
                            _call(initial)["call_id"],
                            {"approved": True, "output": {"ok": True}},
                        ),
                        "agent_session_id": "session-a",
                        "stream": True,
                    }
                delivered = asyncio.Event()
                async with _exchange(app, payload, hold_created=delivered) as (
                    _,
                    outgoing,
                    request,
                ):
                    created = await _created(outgoing)
                    assert created["type"] == "response.created"
                    assert created["response"]["status"] == "in_progress"
                    assert created["response"]["created_at"] == clock[0]
                    assert created["response"]["agent_session_id"] == "session-a"
                    assert created["response"]["output"] == []
                    assert not model.started.is_set()
                    delivered.set()
                    await asyncio.wait_for(model.started.wait(), 2)
                    assert outgoing.empty() and not request.done()
                    clock[0] += 10
                    model.release.set()
                    completed = await _event(outgoing)
                    await asyncio.wait_for(request, 2)
                assert completed["type"] == "response.completed"
                response = completed["response"]
                assert response["id"] == created["response"]["id"]
                assert response["created_at"] == created["response"]["created_at"]
                assert response["agent_session_id"] == "session-a"
                assert response["output"][0]["response_id"] == response["id"]
                assert response["output"][0]["type"] == (
                    "function_call" if result == "tool" else "message"
                )
                if continuation:
                    clock[0] += 10
                    restarted_app = _app(
                        _spec(),
                        brokered_model_loop_enabled=True,
                        brokered_model_http_client=upstream,
                        response_state_file=state_file,
                    )
                    async with _exchange(restarted_app, payload) as (
                        _,
                        outgoing,
                        replay_request,
                    ):
                        replay_created = await _created(outgoing)
                        replay_completed = await _event(outgoing)
                        await asyncio.wait_for(replay_request, 2)
                    assert replay_created["response"]["id"] == response["id"]
                    assert (
                        replay_created["response"]["created_at"]
                        == response["created_at"]
                    )
                    assert replay_completed == completed
                    buffered = await client.post(
                        "/responses",
                        json={**payload, "stream": False},
                        headers=CONTINUATION_AUTH,
                    )
                    assert buffered.status_code == 200
                    assert buffered.json() == {
                        key: value
                        for key, value in response.items()
                        if key != "agent_session_id"
                    }
                    assert model.calls == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("continuation", [False, True])
def test_brokered_stream_error_retains_acknowledged_identity(continuation):
    async def exercise():
        model = HeldModel("error", continuation=continuation)
        async with httpx.AsyncClient(transport=model) as upstream:
            app = _app(
                _spec(),
                brokered_model_loop_enabled=True,
                brokered_model_http_client=upstream,
            )
            payload = {
                "input": "Read data",
                "agent_session_id": "session-a",
                "stream": True,
            }
            if continuation:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://testserver"
                ) as client:
                    initial = (
                        await client.post(
                            "/responses",
                            json={
                                "input": "Read data",
                                "agent_session_id": "session-a",
                            },
                        )
                    ).json()
                payload = {
                    **_continuation(
                        initial["id"],
                        _call(initial)["call_id"],
                        {"approved": True, "output": {"ok": True}},
                    ),
                    "agent_session_id": "session-a",
                    "stream": True,
                }
            async with _exchange(app, payload) as (_, outgoing, request):
                created = await _created(outgoing)
                await asyncio.wait_for(model.started.wait(), 2)
                model.release.set()
                failed = await _event(outgoing)
                await asyncio.wait_for(request, 2)
            assert failed["type"] == "response.failed" and "error" not in failed
            assert failed["response"]["id"] == created["response"]["id"]
            assert failed["response"]["agent_session_id"] == "session-a"
            assert failed["response"]["status"] == "failed"
            assert failed["response"]["error"]["code"]
            assert "private-upstream-detail" not in json.dumps(failed)

    asyncio.run(exercise())


@pytest.mark.parametrize("continuation", [False, True])
def test_brokered_stream_disconnect_cancels_model_and_releases_initial_or_resume_state(
    continuation,
):
    async def exercise():
        model = HeldModel(continuation=continuation)
        async with httpx.AsyncClient(transport=model) as upstream:
            app = _app(
                _spec(),
                brokered_model_loop_enabled=True,
                brokered_model_http_client=upstream,
                max_pending_responses=1,
            )
            payload = {"input": "Read data", "stream": True}
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                if continuation:
                    initial = (
                        await client.post("/responses", json={"input": "Read data"})
                    ).json()
                    payload = {
                        **_continuation(
                            initial["id"],
                            _call(initial)["call_id"],
                            {"approved": True, "output": {"ok": True}},
                        ),
                        "stream": True,
                    }
                async with _exchange(app, payload) as (incoming, outgoing, request):
                    await _created(outgoing)
                    await asyncio.wait_for(model.started.wait(), 2)
                    await incoming.put({"type": "http.disconnect"})
                    await asyncio.wait_for(request, 2)
                    assert model.cancelled.is_set()
                    assert outgoing.empty()
                retry = await client.post(
                    "/responses",
                    json={**payload, "stream": False},
                    headers=CONTINUATION_AUTH,
                )
                assert retry.status_code == 200, retry.text
                assert (
                    retry.json()["output"][0]["content"][0]["text"]
                    == "Verified response."
                )
                assert model.calls == (3 if continuation else 2)

    asyncio.run(exercise())


def test_brokered_stream_disconnect_before_created_is_delivered_starts_no_model_work():
    async def exercise():
        model = HeldModel()
        async with httpx.AsyncClient(transport=model) as upstream:
            app = _app(
                _spec(),
                brokered_model_loop_enabled=True,
                brokered_model_http_client=upstream,
                max_pending_responses=1,
            )
            async with _exchange(
                app,
                {"input": "Read data", "stream": True},
                hold_created=asyncio.Event(),
            ) as (incoming, outgoing, request):
                await _created(outgoing)
                await incoming.put({"type": "http.disconnect"})
                await asyncio.wait_for(request, 2)
            assert model.calls == 0
            model.release.set()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                retry = await client.post("/responses", json={"input": "Read data"})
            assert retry.status_code == 200, retry.text

    asyncio.run(exercise())


def test_brokered_stream_validation_failure_keeps_http_error_without_ack():
    async def exercise():
        model = HeldModel()
        async with httpx.AsyncClient(transport=model) as upstream:
            app = _app(
                _spec(),
                brokered_model_loop_enabled=True,
                brokered_model_http_client=upstream,
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/responses",
                    json={
                        "input": "Read data",
                        "stream": True,
                        "tools": [{"type": "function"}],
                    },
                )
            assert response.status_code == 400
            assert response.json()["error"]["code"] == "tools_unsupported"
            assert model.calls == 0

    asyncio.run(exercise())


class NativeModelRuntime(EchoRuntime):
    def __init__(self, client, url):
        super().__init__()
        self.client = client
        self.url = url

    async def run(self, request):
        self.requests.append(request)
        response = await self.client.post(
            self.url,
            json={"messages": [{"role": "user", "content": request.prompt}]},
        )
        if response.status_code >= 400:
            raise AgentRunError(response.text, status=502, code="UpstreamFailure")
        body = response.json()
        return RunResult(text=body["output"][0]["content"][0]["text"])


def _native_app(client, *, url="http://model/v1/chat/completions", **options):
    factory = EchoFactory()
    factory.runtime = NativeModelRuntime(client, url)
    return create_foundry_app(native_spec(), factory, **options), factory.runtime


@pytest.mark.parametrize("session_id", [None, "native-session"])
def test_native_stream_ack_precedes_runtime_and_keeps_completion_identity(
    session_id, monkeypatch
):
    clock = [1_700_000_000]
    monkeypatch.setattr("agentkit_serve_common.foundry.time.time", lambda: clock[0])

    async def exercise():
        model = HeldModel()
        async with httpx.AsyncClient(transport=model) as upstream:
            app, runtime = _native_app(upstream)
            payload = {"input": "Summarize the evidence", "stream": True}
            if session_id:
                payload["agent_session_id"] = session_id
            delivered = asyncio.Event()
            async with (
                app.router.lifespan_context(app),
                _exchange(app, payload, hold_created=delivered, headers={}) as (
                    _,
                    outgoing,
                    request,
                ),
            ):
                created = await _created(outgoing)
                assert created["response"]["status"] == "in_progress"
                assert created["response"]["created_at"] == clock[0]
                assert created["response"].get("agent_session_id") == session_id
                assert created["response"]["output"] == []
                assert runtime.requests == [] and model.calls == 0
                delivered.set()
                await asyncio.wait_for(model.started.wait(), 2)
                assert not request.done() and outgoing.empty()
                clock[0] += 10
                model.release.set()
                completed = await _event(outgoing)
                await asyncio.wait_for(request, 2)
            assert completed["type"] == "response.completed"
            response = completed["response"]
            assert response["id"] == created["response"]["id"]
            assert response["created_at"] == created["response"]["created_at"]
            assert response.get("agent_session_id") == session_id
            assert response["output"][0]["response_id"] == response["id"]
            assert response["output"][0]["content"][0]["text"] == "Verified response."
            assert runtime.requests[0].session_id == session_id
            assert model.calls == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("result", ["error", "exception"])
def test_native_stream_error_keeps_identity_and_sanitizes_runtime_details(result):
    async def exercise():
        model = HeldModel(result)
        async with httpx.AsyncClient(transport=model) as upstream:
            app, _ = _native_app(upstream)
            async with (
                app.router.lifespan_context(app),
                _exchange(
                    app,
                    {
                        "input": "Summarize the evidence",
                        "agent_session_id": "native-session",
                        "stream": True,
                    },
                    headers={},
                ) as (_, outgoing, request),
            ):
                created = await _created(outgoing)
                await asyncio.wait_for(model.started.wait(), 2)
                model.release.set()
                failed = await _event(outgoing)
                await asyncio.wait_for(request, 2)
            assert failed["type"] == "response.failed" and "error" not in failed
            assert failed["response"]["id"] == created["response"]["id"]
            assert failed["response"]["agent_session_id"] == "native-session"
            assert failed["response"]["status"] == "failed"
            assert failed["response"]["error"]["code"] == "RuntimeFailure"
            assert "private-upstream-detail" not in json.dumps(failed)

    asyncio.run(exercise())


def test_native_stream_disconnect_before_ack_delivery_starts_no_runtime():
    async def exercise():
        model = HeldModel()
        async with httpx.AsyncClient(transport=model) as upstream:
            app, runtime = _native_app(upstream)
            async with (
                app.router.lifespan_context(app),
                _exchange(
                    app,
                    {"input": "Summarize the evidence", "stream": True},
                    hold_created=asyncio.Event(),
                    headers={},
                ) as (incoming, outgoing, request),
            ):
                await _created(outgoing)
                await incoming.put({"type": "http.disconnect"})
                await asyncio.wait_for(request, 2)
            assert runtime.requests == [] and model.calls == 0

    asyncio.run(exercise())


def test_native_stream_disconnect_closes_model_http_socket_and_allows_next_request():
    async def exercise():
        model_started, model_disconnected = asyncio.Event(), asyncio.Event()
        model_requests, handlers = [], set()

        async def serve_model(reader, writer):
            handler = asyncio.current_task()
            handlers.add(handler)
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                length = next(
                    int(line.split(b":", 1)[1])
                    for line in headers.splitlines()
                    if line.lower().startswith(b"content-length:")
                )
                model_requests.append(await reader.readexactly(length))
                if len(model_requests) == 1:
                    model_started.set()
                    assert await reader.read(1) == b""
                    model_disconnected.set()
                else:
                    body = json.dumps(
                        _chat_response({"role": "assistant", "content": "Recovered."})
                    ).encode()
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                        + body
                    )
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.remove(handler)

        server = await asyncio.start_server(serve_model, "127.0.0.1", 0)
        url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/chat/completions"
        try:
            async with (
                server,
                httpx.AsyncClient(trust_env=False, timeout=5) as upstream,
            ):
                app, runtime = _native_app(upstream, url=url)
                async with app.router.lifespan_context(app):
                    async with _exchange(
                        app,
                        {"input": "Wait for cancellation", "stream": True},
                        headers={},
                    ) as (incoming, outgoing, request):
                        await _created(outgoing)
                        await asyncio.wait_for(model_started.wait(), 2)
                        await incoming.put({"type": "http.disconnect"})
                        await asyncio.wait_for(request, 2)
                        await asyncio.wait_for(model_disconnected.wait(), 2)
                        assert outgoing.empty()
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app), base_url="http://hosted"
                    ) as client:
                        response = await client.post(
                            "/responses",
                            json={"input": "Next request", "stream": False},
                        )
                    assert response.status_code == 200
                    assert (
                        response.json()["output"][0]["content"][0]["text"]
                        == "Recovered."
                    )
                    assert len(runtime.requests) == len(model_requests) == 2
        finally:
            for handler in tuple(handlers):
                handler.cancel()
            await asyncio.gather(*handlers, return_exceptions=True)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("payload", "authorized", "status", "code"),
    [
        ({"input": "Read data", "stream": True}, False, 401, None),
        ({"stream": True}, True, 400, "missing_input"),
        ({"input": "x", "stream": True, "tools": [{}]}, True, 400, "tools_unsupported"),
        (
            {"input": [{"role": "assistant", "content": "x"}], "stream": True},
            True,
            400,
            "invalid_input",
        ),
        ({"input": "x" * 300, "stream": True}, True, 413, "request_too_large"),
    ],
)
def test_native_stream_rejection_keeps_http_error_before_runtime(
    payload, authorized, status, code
):
    async def exercise():
        model = HeldModel()
        async with httpx.AsyncClient(transport=model) as upstream:
            app, runtime = _native_app(
                upstream, auth_token="local-test", max_request_body_bytes=256
            )
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://hosted"
                ) as client,
            ):
                response = await client.post(
                    "/responses",
                    json=payload,
                    headers={"authorization": "Bearer local-test"}
                    if authorized
                    else {},
                )
            assert response.status_code == status
            assert response.headers["content-type"] == "application/json"
            if code:
                assert response.json()["error"]["code"] == code
            assert runtime.requests == [] and model.calls == 0

    asyncio.run(exercise())
