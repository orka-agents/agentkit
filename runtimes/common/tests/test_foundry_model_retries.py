"""Bounded rate-limit recovery without replaying hosted invocations."""

from __future__ import annotations

import asyncio
import json
from email.utils import formatdate
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from agentkit_serve_common import foundry_model_loop
from test_foundry_brokered_protocol import (
    CONTINUATION_AUTH,
    _app,
    _call,
    _chat_response,
    _continuation,
    _spec,
)
from test_foundry_streaming import _created, _event, _exchange, _tool_response


class RejectedBody(httpx.AsyncByteStream):
    def __init__(self):
        self.read = False
        self.closed = False

    async def __aiter__(self):
        self.read = True
        yield b"private-upstream-detail"

    async def aclose(self):
        self.closed = True


def _reject(request, bodies, *, headers=None):
    body = RejectedBody()
    bodies.append(body)
    return httpx.Response(
        429,
        request=request,
        headers={"x-request-id": "private-upstream-detail", **(headers or {})},
        stream=body,
    )


def _capture_waits(monkeypatch, bodies):
    waits = []

    async def sleep(delay):
        assert bodies and all(body.closed and not body.read for body in bodies)
        waits.append(delay)
        await asyncio.sleep(0)

    # Keep the real ASGI/event-loop scheduling untouched while advancing only
    # this model client's retry waits without spending minutes in each test.
    monkeypatch.setattr(
        foundry_model_loop,
        "asyncio",
        SimpleNamespace(sleep=sleep, to_thread=asyncio.to_thread),
    )
    return waits


async def _payload(client, *, continuation):
    payload = {
        "input": "Read synthetic data",
        "agent_session_id": "session-a",
        "stream": True,
    }
    if continuation:
        initial = await client.post("/responses", json={**payload, "stream": False})
        assert initial.status_code == 200, initial.text
        body = initial.json()
        payload = {
            **_continuation(
                body["id"],
                _call(body)["call_id"],
                {"approved": True, "output": {"ok": True}},
            ),
            "agent_session_id": "session-a",
            "stream": True,
        }
    return payload


@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize("completion", ["text", "tool", "exhausted"])
def test_hosted_rate_limit_retries_keep_one_ack_and_unchanged_model_input(
    continuation, completion, monkeypatch, caplog
):
    bodies, requests, credentials = [], [], []
    waits = _capture_waits(monkeypatch, bodies)
    first_attempt = 2 if continuation else 1

    async def auth(self):
        credentials.append(len(credentials) + 1)
        return {"Authorization": f"Bearer synthetic-attempt-{len(credentials)}"}

    monkeypatch.setattr(foundry_model_loop.BrokeredChatModelLoop, "_auth_headers", auth)

    def model(request):
        requests.append(request)
        if len(requests) < first_attempt:
            return httpx.Response(200, request=request, json=_tool_response())
        if len(requests) < first_attempt + 2 or completion == "exhausted":
            return _reject(request, bodies, headers={"retry-after": "60"})
        result = (
            _tool_response()
            if completion == "tool"
            else _chat_response({"role": "assistant", "content": "Verified response."})
        )
        return httpx.Response(200, request=request, json=result)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(model)) as upstream:
            app = _app(
                brokered_model_loop_enabled=True,
                brokered_model_http_client=upstream,
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://hosted"
            ) as client:
                payload = await _payload(client, continuation=continuation)
                async with _exchange(app, payload) as (_, outgoing, request):
                    created = await _created(outgoing)
                    terminal = await _event(outgoing)
                    await asyncio.wait_for(request, 2)
                    assert outgoing.empty()
                assert created["type"] == "response.created"
                assert terminal["response"]["id"] == created["response"]["id"]
                assert (
                    terminal["response"]["created_at"]
                    == created["response"]["created_at"]
                )
                assert terminal["response"]["agent_session_id"] == "session-a"
                assert "private-upstream-detail" not in json.dumps(terminal)
                if completion == "exhausted":
                    assert terminal["type"] == "response.failed"
                    assert terminal["response"]["error"] == {
                        "code": "ModelUnavailable",
                        "message": "model service is unavailable",
                        "upstream_status": 429,
                    }
                else:
                    assert terminal["type"] == "response.completed"
                    assert terminal["response"]["output"][0]["type"] == (
                        "function_call" if completion == "tool" else "message"
                    )
                    if continuation:
                        replay = await client.post(
                            "/responses",
                            json={**payload, "stream": False},
                            headers=CONTINUATION_AUTH,
                        )
                        assert replay.status_code == 200
                        assert replay.json()["id"] == terminal["response"]["id"]

    asyncio.run(exercise())
    assert len(requests) == first_attempt + 2
    attempts = requests[first_attempt - 1 :]
    assert attempts[0].content == attempts[1].content == attempts[2].content
    assert [request.headers["authorization"] for request in requests] == [
        f"Bearer synthetic-attempt-{number}" for number in credentials
    ]
    assert waits == [60, 60]
    assert all(body.closed and not body.read for body in bodies)
    assert "private-upstream-detail" not in caplog.text


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after-ms": "60000", "retry-after": "1"}, 60),
        ({"retry-after-ms": "1250"}, 1.25),
        ({"retry-after-ms": "invalid", "retry-after": "2"}, 2),
        ({"retry-after-ms": "nan", "retry-after": "3"}, 3),
        ({"retry-after-ms": "-1", "retry-after": "4"}, 4),
        ({"retry-after": "60"}, 60),
        ({"retry-after": "0"}, 0),
        ({"retry-after": formatdate(1_700_000_060, usegmt=True)}, 60),
        ({"retry-after": formatdate(1_699_999_999, usegmt=True)}, 0),
        ({"retry-after": "61"}, None),
        ({"retry-after-ms": "60001", "retry-after": "1"}, None),
        ({"retry-after": formatdate(1_700_000_061, usegmt=True)}, None),
    ],
)
def test_model_retry_respects_server_delay_and_never_shortens_long_windows(
    headers, expected, monkeypatch
):
    monkeypatch.setattr(foundry_model_loop.time, "time", lambda: 1_700_000_000)
    bodies, requests = [], []
    waits = _capture_waits(monkeypatch, bodies)

    def model(request):
        requests.append(request)
        if len(requests) == 1:
            return _reject(request, bodies, headers=headers)
        return httpx.Response(
            200,
            request=request,
            json=_chat_response({"role": "assistant", "content": "Recovered."}),
        )

    app = _app(
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(model)
        ),
    )
    with TestClient(app) as client:
        result = client.post("/responses", json={"input": "Read data"})
    if expected is None:
        assert len(requests) == 1 and waits == []
        assert result.status_code == 503
        assert result.json()["error"]["upstream_status"] == 429
    else:
        assert len(requests) == 2 and waits == [expected]
        assert result.status_code == 200
        assert result.json()["output"][0]["content"][0]["text"] == "Recovered."
    assert all(body.closed and not body.read for body in bodies)


@pytest.mark.parametrize("header", [None, "invalid", "nan", "inf", "-1"])
def test_model_missing_or_malformed_retry_header_uses_bounded_backoff(
    header, monkeypatch
):
    bodies = []
    waits = _capture_waits(monkeypatch, bodies)
    headers = (
        {} if header is None else {"retry-after-ms": header, "retry-after": header}
    )

    def model(request):
        return _reject(request, bodies, headers=headers)

    app = _app(
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(model)
        ),
    )
    with TestClient(app) as client:
        result = client.post("/responses", json={"input": "Read data"})
    assert result.status_code == 503
    assert result.json()["error"]["upstream_status"] == 429
    assert len(bodies) == 3 and len(waits) == 2
    assert 0.75 <= waits[0] <= 1 and 1.5 <= waits[1] <= 2


@pytest.mark.parametrize(
    "failure", [400, 401, 403, 408, 409, 500, 503, "connect", "read", "json"]
)
def test_model_does_not_retry_other_rejections_or_ambiguous_failures(
    failure, monkeypatch
):
    requests = []
    waits = _capture_waits(monkeypatch, [])

    class IncompleteBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"output":'
            raise httpx.ReadError("private-upstream-detail")

    def model(request):
        requests.append(request)
        if failure == "connect":
            raise httpx.ConnectError("private-upstream-detail", request=request)
        if failure == "read":
            return httpx.Response(200, request=request, stream=IncompleteBody())
        return httpx.Response(
            200 if failure == "json" else failure,
            request=request,
            headers={"retry-after-ms": "1"},
            content=b"private-upstream-detail",
        )

    app = _app(
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(model)
        ),
    )
    with TestClient(app) as client:
        result = client.post("/responses", json={"input": "Read data"})
    assert result.status_code >= 400
    assert len(requests) == 1 and waits == []
    assert "private-upstream-detail" not in result.text


@pytest.mark.parametrize("continuation", [False, True])
def test_hosted_disconnect_during_real_rate_limit_wait_stops_retry_and_releases_state(
    continuation, tmp_path
):
    async def exercise():
        requests, handlers = [], set()
        rejected_connection_closed = asyncio.Event()
        reject_at = 2 if continuation else 1

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
                requests.append(await reader.readexactly(length))
                if len(requests) == reject_at:
                    # A definitive 429 is enough. Its body must not hold the
                    # connection open during the actual 60-second retry wait.
                    writer.write(
                        b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 60\r\n"
                        b"Content-Length: 65536\r\nConnection: close\r\n\r\n"
                    )
                    await writer.drain()
                    assert await reader.read(1) == b""
                    rejected_connection_closed.set()
                else:
                    response = (
                        _tool_response()
                        if len(requests) < reject_at
                        else _chat_response(
                            {"role": "assistant", "content": "Recovered."}
                        )
                    )
                    body = json.dumps(response).encode()
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
        spec = _spec()
        spec.model.base_url = (
            f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1"
        )
        state_file = tmp_path / "responses.json"
        try:
            async with (
                server,
                httpx.AsyncClient(trust_env=False, timeout=5) as upstream,
            ):
                app = _app(
                    spec,
                    brokered_model_loop_enabled=True,
                    brokered_model_http_client=upstream,
                    response_state_file=state_file,
                    max_pending_responses=1,
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://hosted"
                ) as client:
                    payload = await _payload(client, continuation=continuation)
                    async with _exchange(app, payload) as (incoming, outgoing, request):
                        created = await _created(outgoing)
                        assert created["type"] == "response.created"
                        await asyncio.wait_for(rejected_connection_closed.wait(), 2)
                        assert not request.done() and outgoing.empty()
                        await incoming.put({"type": "http.disconnect"})
                        await asyncio.wait_for(request, 2)
                        assert outgoing.empty() and len(requests) == reject_at
                    # A new client operation can use the released initial slot or
                    # resume the explicitly cancelled continuation. No automatic
                    # model retry remains alive after the disconnected request.
                    result = await client.post(
                        "/responses",
                        json={**payload, "stream": False},
                        headers=CONTINUATION_AUTH,
                    )
                    assert result.status_code == 200, result.text
                    assert (
                        result.json()["output"][0]["content"][0]["text"] == "Recovered."
                    )
                    assert len(requests) == reject_at + 1
        finally:
            for handler in tuple(handlers):
                handler.cancel()
            await asyncio.gather(*handlers, return_exceptions=True)

    asyncio.run(exercise())
