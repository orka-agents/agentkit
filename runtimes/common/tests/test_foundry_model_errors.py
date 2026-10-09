"""Safe model diagnostics on the public hosted Responses boundary."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from agentkit_serve_common import foundry_model_loop
from agentkit_serve_common.runtime import AgentRunError
from test_foundry_brokered_protocol import (
    responses_model_api,
    CONTINUATION_AUTH,
    _app,
    _call,
    _continuation,
    _spec,
)
from test_foundry_streaming import _tool_response


def _response_error(app, *, continuation, stream=True, status=400):
    payload = {
        "input": "Read synthetic data.",
        "agent_session_id": "synthetic-session",
        "stream": stream,
    }
    with TestClient(app) as client:
        if continuation:
            initial = client.post("/responses", json={**payload, "stream": False})
            assert initial.status_code == 200
            payload = {
                **_continuation(
                    initial.json()["id"],
                    _call(initial.json())["call_id"],
                    {"approved": True, "output": {"ok": True}},
                ),
                "agent_session_id": "synthetic-session",
                "stream": stream,
            }
        response = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)

    assert "private-upstream-detail" not in response.text
    if not stream:
        assert response.status_code == status
        return response.json()["error"]
    assert response.status_code == 200
    frames = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [frame["type"] for frame in frames] == [
        "response.created",
        "response.failed",
    ]
    created, failed = (frame["response"] for frame in frames)
    assert failed["id"] == created["id"]
    assert failed["created_at"] == created["created_at"]
    assert failed["agent_session_id"] == created["agent_session_id"] == "synthetic-session"
    assert failed["status"] == "failed"
    assert "error" not in frames[-1]
    return failed["error"]


@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("kind", "code", "message"),
    [
        ("unknown-name", "unknown_brokered_tool", "model requested unknown brokered tool"),
        ("duplicate-key", "InvalidToolArguments", "model tool arguments are invalid"),
        ("unicode-path", "InvalidToolArguments", "model tool arguments are invalid"),
        ("schema-path", "InvalidToolArguments", "model tool arguments are invalid"),
        ("unsafe-path", "UnsafeBrokeredArguments", "model tool arguments are unsafe"),
    ],
)
def test_model_tool_validation_never_reflects_names_keys_or_paths(
    continuation, stream, kind, code, message, caplog
):
    calls = 0
    marker = "private-upstream-detail"
    spec = _spec()
    payload = _tool_response()
    function = payload["output"][0]
    if kind == "unknown-name":
        function["name"] = marker
    elif kind == "duplicate-key":
        function["arguments"] = f'{{"{marker}":true,"{marker}":false}}'
    elif kind == "unicode-path":
        function["arguments"] = json.dumps({marker: "\ud800"})
    elif kind == "schema-path":
        function["arguments"] = json.dumps({marker: True})
        spec.brokered_tools[0].parameters["additionalProperties"] = False
    else:
        function["arguments"] = json.dumps({marker: {"api_key": "synthetic-value"}})

    def model(request):
        nonlocal calls
        calls += 1
        result = _tool_response() if continuation and calls == 1 else payload
        return httpx.Response(200, request=request, json=result)

    app = _app(
        spec,
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(transport=httpx.MockTransport(model)),
    )
    assert _response_error(app, continuation=continuation, stream=stream) == {
        "code": code,
        "message": message,
    }
    assert calls == (2 if continuation else 1)
    assert marker not in caplog.text


@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize(
    ("status", "code", "message"),
    [
        (401, "ModelAuthRejected", "model service rejected configured credentials"),
        (403, "ModelAuthRejected", "model service rejected configured credentials"),
        (429, "ModelUnavailable", "model service is unavailable"),
        (500, "ModelUnavailable", "model service is unavailable"),
        (599, "ModelUnavailable", "model service is unavailable"),
        (400, "ModelUpstreamError", "model service request failed"),
    ],
)
def test_model_http_error_retains_only_normalized_code_and_status(
    continuation, status, code, message, caplog
):
    calls = 0

    def model(request):
        nonlocal calls
        calls += 1
        if continuation and calls == 1:
            return httpx.Response(200, request=request, json=_tool_response())
        return httpx.Response(
            status,
            request=request,
            headers={"x-request-id": "private-upstream-detail"},
            json={
                "error": {
                    "code": "private-upstream-detail",
                    "message": "private-upstream-detail",
                }
            },
        )

    app = _app(
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(transport=httpx.MockTransport(model)),
    )
    error = _response_error(app, continuation=continuation)
    assert error == {"code": code, "message": message, "upstream_status": status}
    assert "private-upstream-detail" not in caplog.text


@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize("kind", ["transport", "invalid-json"])
def test_non_http_model_failures_have_no_upstream_status(continuation, kind):
    calls = 0

    def model(request):
        nonlocal calls
        calls += 1
        if continuation and calls == 1:
            return httpx.Response(200, request=request, json=_tool_response())
        if kind == "transport":
            raise httpx.ConnectError("private-upstream-detail", request=request)
        return httpx.Response(200, request=request, content=b"private-upstream-detail")

    app = _app(
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(transport=httpx.MockTransport(model)),
    )
    error = _response_error(app, continuation=continuation)
    assert set(error) == {"code", "message"}
    assert error["code"] == (
        "ModelUpstreamError" if kind == "transport" else "InvalidModelResponse"
    )


def _raise_model_error(monkeypatch, *, continuation, error):
    calls = 0

    async def fail(self, messages, *, tools, tool_choice):
        nonlocal calls
        calls += 1
        if continuation and calls == 1:
            return _tool_response()
        raise error

    monkeypatch.setattr(foundry_model_loop.BrokeredChatModelLoop, "_create_response", fail)
    return _app(brokered_model_loop_enabled=True)


@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("ModelAuthMissing", "model authentication is not configured"),
        ("ModelAuthRejected", "model service rejected configured credentials"),
        ("ModelUnavailable", "model service is unavailable"),
        ("ModelUpstreamError", "model service request failed"),
        ("InvalidModelResponse", "model service returned an invalid response"),
        ("ModelResponseTooLarge", "model response is too large to retain safely"),
        ("InvalidToolArguments", "model tool arguments are invalid"),
        ("UnsafeBrokeredArguments", "model tool arguments are unsafe"),
    ],
)
def test_known_model_codes_use_fixed_messages_and_do_not_trust_status_attributes(
    continuation, code, message, monkeypatch, caplog
):
    error = AgentRunError("private-upstream-detail", status=999, code=code)
    error.upstream_status = 401
    app = _raise_model_error(monkeypatch, continuation=continuation, error=error)
    assert _response_error(app, continuation=continuation) == {
        "code": code,
        "message": message,
    }
    assert "private-upstream-detail" not in caplog.text


@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize("upstream_status", [True, "401", 401.0, 399, 600, None])
def test_normalized_http_error_rejects_invalid_upstream_status(
    continuation, upstream_status, monkeypatch
):
    error = foundry_model_loop._normalized_model_http_error(401)
    error.upstream_status = upstream_status
    app = _raise_model_error(monkeypatch, continuation=continuation, error=error)
    assert _response_error(app, continuation=continuation) == {
        "code": "ModelAuthRejected",
        "message": "model service rejected configured credentials",
    }


@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("private-upstream-detail", 503),
        ("private-upstream-detail", 400),
        (["private-upstream-detail"], 413),
        (None, 999),
        (["private-upstream-detail"], "503"),
    ],
)
def test_unknown_model_codes_and_statuses_are_sanitized(
    continuation, code, status, monkeypatch, caplog
):
    error = AgentRunError("private-upstream-detail", status=status, code=code)
    error.upstream_status = 401
    app = _raise_model_error(monkeypatch, continuation=continuation, error=error)
    assert _response_error(app, continuation=continuation) == {
        "code": "ModelResumeError",
        "message": "model resume failed",
    }
    assert "private-upstream-detail" not in caplog.text


@pytest.mark.parametrize("continuation", [False, True])
def test_unexpected_model_exception_remains_generic(continuation, monkeypatch, caplog):
    app = _raise_model_error(
        monkeypatch,
        continuation=continuation,
        error=RuntimeError("private-upstream-detail"),
    )
    assert _response_error(app, continuation=continuation) == {
        "code": "ModelResumeError",
        "message": "model resume failed",
    }
    assert "private-upstream-detail" not in caplog.text
