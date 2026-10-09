"""Wire-level cross-runtime parity suite for AgentKit runtime adapters.

``conformance.py`` swaps each adapter's model client for an in-process double, so
it cannot see what an adapter actually sends to a model. This suite builds the
adapter's REAL runtime from an AgentSpec whose ``model.baseURL`` is a scripted
OpenAI-compatible endpoint on loopback and whose tool is a stdio MCP fixture
server, then drives the shared protocol skins. Every adapter must send the model
the same conversation, return the same client-visible results and errors, and
keep secret canaries out of everything a client can read.

An adapter inherits the suite by re-exporting it from ``tests/test_parity.py``::

    from agentkit_serve_common.parity import *  # noqa: F401,F403

The adapter package (``agentkit_serve``), the FastAPI test client, and the MCP
fixture server's SDK are imported lazily or in a subprocess, so the shared core
stays framework-agnostic.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator

from .config import AgentSpec
from .conversation import RunRequest, ToolCallEvent

__all__ = [
    "test_parity_model_receives_baked_instructions_then_client_history",
    "test_parity_client_history_is_authoritative_with_session_header",
    "test_parity_orka_turns_carry_runtime_session_history",
    "test_parity_tool_roundtrip",
    "test_parity_parallel_tool_calls",
    "test_parity_tool_events_pair_each_call",
    "test_parity_tool_env_is_allowlisted",
    "test_parity_tool_errors_reach_model_without_tool_details",
    "test_parity_invalid_tool_arguments_are_returned_to_model",
    "test_parity_unknown_tool_is_returned_to_model",
    "test_parity_model_http_errors_are_normalized",
    "test_parity_model_auth_errors_never_echo_credentials",
    "test_parity_model_transport_errors_are_normalized",
    "test_parity_malformed_model_response_fails_closed",
    "test_parity_crashed_stdio_tool_fails_health",
    "test_parity_orka_runtime_startup_failure_hides_tool_credentials",
    "test_parity_orka_rebuilds_runtime_after_crashed_stdio_tool",
]

_MODEL_NAME = "parity-model"
_MODEL_KEY_ENV = "PARITY_MODEL_KEY"
_MODEL_KEY = "sk-parity-model-canary"
_UNDECLARED_ENV = "PARITY_UNDECLARED_SECRET"
_UNDECLARED_SECRET = "parity-undeclared-canary"
_TOOL_ENV = "PARITY_TOOL_VISIBLE"
_TOOL_VALUE = "parity-tool-visible"
_TOOL_ERROR_DETAIL = "parity-tool-internal-detail"
_REMOTE_TOOL_TOKEN_ENV = "PARITY_REMOTE_TOOL_TOKEN"
_REMOTE_TOOL_TOKEN = "parity-remote-tool-token-canary"
_REMOTE_TOOL_URL_ENV = "PARITY_REMOTE_TOOL_URL"
_REMOTE_TOOL_URL_KEY = "parity-remote-url-key-canary"
_INSTRUCTIONS = "Parity baked instructions."
_ORKA_TOKEN = "parity-orka-token"
_TOOL_NAMES = {"probe_echo", "probe_fail", "probe_env_dump", "probe_crash"}
_CANARIES = (_MODEL_KEY, _UNDECLARED_SECRET, _TOOL_ERROR_DETAIL, _REMOTE_TOOL_TOKEN, _REMOTE_TOOL_URL_KEY)

# Run in a subprocess with the adapter's interpreter; the MCP SDK is an adapter
# dependency, not a shared-core one.
_MCP_SERVER = f'''
import json
import os

from mcp.server.fastmcp import FastMCP

server = FastMCP("probe", log_level="ERROR")


@server.tool()
def echo(value: str) -> str:
    """Return a receipt for value."""
    return "receipt-" + value


@server.tool()
def fail() -> str:
    """Always fail."""
    raise ValueError("{_TOOL_ERROR_DETAIL}")


@server.tool()
def env_dump() -> str:
    """Return this tool process environment."""
    return json.dumps(dict(os.environ))


@server.tool()
def crash() -> str:
    """Exit the tool server."""
    os._exit(1)


server.run()
'''


# --------------------------------------------------------------------------- #
# Scripted OpenAI-compatible provider
# --------------------------------------------------------------------------- #
@dataclass
class _Reply:
    status: int = 200
    message: dict[str, Any] | None = None
    finish: str = "stop"
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)


def _answer(text: str) -> _Reply:
    return _Reply(message={"role": "assistant", "content": text})


def _tool_calls(*calls: tuple[str, str]) -> _Reply:
    return _Reply(
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": f"call_parity_{index}", "type": "function", "function": {"name": name, "arguments": arguments}}
                for index, (name, arguments) in enumerate(calls)
            ],
        },
        finish="tool_calls",
    )


def _call_then_answer(*calls: tuple[str, str], answer: str = "parity-final") -> Callable[[dict[str, Any]], _Reply]:
    """Request tool calls once, then answer after any tool result arrives."""

    def script(body: dict[str, Any]) -> _Reply:
        if body["messages"][-1]["role"] == "tool":
            return _answer(answer)
        return _tool_calls(*calls)

    return script


class _ScriptedProvider:
    """Loopback Chat Completions endpoint that records requests and replays a script."""

    def __init__(self) -> None:
        # A script returning None drops the connection without a response.
        self.script: Callable[[dict[str, Any]], _Reply | None] = lambda body: _answer("parity-answer")
        self.requests: list[tuple[str | None, dict[str, Any]]] = []
        self._lock = threading.Lock()
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
                provider._handle(self)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __enter__(self) -> "_ScriptedProvider":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def bodies(self) -> list[dict[str, Any]]:
        with self._lock:
            return [body for _, body in self.requests]

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        raw = handler.rfile.read(int(handler.headers.get("content-length") or 0))
        if handler.path.startswith("/mcp"):
            # A remote MCP endpoint that rejects the tool's credential and, like
            # some gateways, echoes it and the request URL back.
            echoed = {"error": f"unauthorized {handler.headers.get('authorization')} at {handler.path}"}
            self._send(handler, 401, "application/json", json.dumps(echoed).encode())
            return
        body = json.loads(raw)
        with self._lock:
            self.requests.append((handler.headers.get("authorization"), body))
        reply = self.script(body)
        if reply is None:
            handler.close_connection = True
            return
        if reply.message is None:
            payload = reply.body if isinstance(reply.body, bytes) else json.dumps(reply.body).encode()
            self._send(handler, reply.status, "application/json", payload, reply.headers)
            return
        base = {"id": f"chatcmpl-{uuid.uuid4().hex}", "created": 1, "model": body["model"]}
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        if not body.get("stream"):
            choice = {"index": 0, "message": reply.message, "finish_reason": reply.finish}
            payload = json.dumps({**base, "object": "chat.completion", "choices": [choice], "usage": usage}).encode()
            self._send(handler, 200, "application/json", payload)
            return
        delta = json.loads(json.dumps(reply.message))
        for index, call in enumerate(delta.get("tool_calls") or []):
            call["index"] = index
        chunks = [
            {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {
                **base,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": reply.finish}],
                "usage": usage,
            },
        ]
        payload = b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)
        self._send(handler, 200, "text/event-stream", payload + b"data: [DONE]\n\n")

    @staticmethod
    def _send(
        handler: BaseHTTPRequestHandler,
        status: int,
        content_type: str,
        payload: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        handler.send_response(status)
        for name, value in (headers or {}).items():
            handler.send_header(name, value)
        handler.send_header("content-type", content_type)
        handler.send_header("content-length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)


# --------------------------------------------------------------------------- #
# Harness helpers
# --------------------------------------------------------------------------- #
@contextmanager
def _canary_env() -> Iterator[None]:
    values = {
        _MODEL_KEY_ENV: _MODEL_KEY,
        _UNDECLARED_ENV: _UNDECLARED_SECRET,
        _TOOL_ENV: _TOOL_VALUE,
        _REMOTE_TOOL_TOKEN_ENV: _REMOTE_TOOL_TOKEN,
        _REMOTE_TOOL_URL_ENV: "",
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _harness(*, tools: bool = False, rejecting_remote_tool: bool = False) -> Iterator[tuple[_ScriptedProvider, AgentSpec]]:
    with tempfile.TemporaryDirectory() as tmp, _ScriptedProvider() as provider, _canary_env():
        tool_specs: list[dict[str, Any]] = []
        if tools:
            server = os.path.join(tmp, "parity_mcp_server.py")
            with open(server, "w", encoding="utf-8") as fh:
                fh.write(_MCP_SERVER)
            tool_specs.append({"name": "probe", "command": [sys.executable, server], "env": [_TOOL_ENV]})
        if rejecting_remote_tool:
            remote_url = provider.base_url.removesuffix("/v1") + f"/mcp?key={_REMOTE_TOOL_URL_KEY}"
            os.environ[_REMOTE_TOOL_URL_ENV] = remote_url
            tool_specs.append(
                {
                    "name": "remote",
                    "type": "mcp",
                    "transport": "streamable-http",
                    "urlEnv": _REMOTE_TOOL_URL_ENV,
                    "auth": {"type": "bearer", "tokenEnv": _REMOTE_TOOL_TOKEN_ENV},
                }
            )
        spec = AgentSpec.model_validate(
            {
                "abiVersion": "v0",
                "metadata": {"name": "parity-agent"},
                "model": {
                    "provider": "openai-compatible",
                    "baseURL": provider.base_url,
                    "name": _MODEL_NAME,
                    "apiKeyEnv": _MODEL_KEY_ENV,
                },
                "instructions": _INSTRUCTIONS,
                "tools": tool_specs,
                "expose": {"openai": True, "port": 8080},
            }
        )
        yield provider, spec


def _factory() -> Any:
    from agentkit_serve import agent_factory  # the adapter package running this suite

    return agent_factory


@contextmanager
def _openai(spec: AgentSpec) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from .server import create_app

    with TestClient(create_app(spec, _factory())) as client:
        yield client


@contextmanager
def _foundry(spec: AgentSpec) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from .foundry import create_foundry_app

    with TestClient(create_foundry_app(spec, _factory())) as client:
        yield client


@contextmanager
def _orka(spec: AgentSpec) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from .orka import create_orka_app

    with TestClient(create_orka_app(spec, _factory(), auth_token=_ORKA_TOKEN)) as client:
        yield client


def _chat(client: Any, messages: list[dict[str, Any]], headers: dict[str, str] | None = None) -> Any:
    response = client.post("/v1/chat/completions", json={"model": "x", "messages": messages}, headers=headers or {})
    _assert_no_canary(response.text)
    return response


def _foundry_respond(client: Any, messages: list[dict[str, Any]]) -> Any:
    response = client.post("/responses", json={"input": messages})
    _assert_no_canary(response.text)
    return response


def _orka_turn(client: Any, prompt: str, *, runtime_session_id: str = "parity-session") -> list[dict[str, Any]]:
    from .orka import ORKA_HARNESS_VERSION

    auth = {"authorization": f"Bearer {_ORKA_TOKEN}"}
    turn_id = f"turn-{uuid.uuid4().hex}"
    deadline = (datetime.now(UTC) + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    started = client.post(
        "/v1/turns",
        headers=auth,
        json={
            "version": ORKA_HARNESS_VERSION,
            "namespace": "default",
            "taskName": "parity-task",
            "sessionName": "parity-session",
            "runtimeSessionID": runtime_session_id,
            "turnID": turn_id,
            "correlationID": f"corr-{turn_id}",
            "deadline": deadline,
            "authIdentity": {"subject": "system:serviceaccount:default:orka"},
            "input": {"prompt": prompt, "contextRefs": [], "env": []},
            "toolExecutionMode": "observed",
            "metadata": {},
        },
    )
    assert started.status_code == 202, started.text
    events = client.get(f"/v1/turns/{turn_id}/events", headers=auth)
    _assert_no_canary(events.text)
    return [json.loads(line.removeprefix("data: ")) for line in events.text.splitlines() if line.startswith("data: ")]


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", "")) for part in content if isinstance(part, (str, dict))
        )
    return json.dumps(content)


def _conversation(body: dict[str, Any]) -> list[tuple[str, str]]:
    """The system/user/assistant text turns one model request carried."""
    return [
        (message["role"], _content_text(message.get("content")))
        for message in body["messages"]
        if message["role"] in {"system", "user", "assistant"} and _content_text(message.get("content"))
    ]


def _tool_results(body: dict[str, Any]) -> dict[str, str]:
    return {
        message.get("tool_call_id"): _content_text(message.get("content"))
        for message in body["messages"]
        if message["role"] == "tool"
    }


def _assert_no_canary(text: str) -> None:
    leaked = [canary for canary in _CANARIES if canary in text]
    assert not leaked, f"client-visible output leaked {leaked}"


# --------------------------------------------------------------------------- #
# Conversation parity
# --------------------------------------------------------------------------- #
_CLIENT_CONVERSATION = [
    {"role": "system", "content": "client system note"},
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1"},
    {"role": "tool", "content": "client-supplied tool turn"},
    {"role": "user", "content": "q2"},
]
_EXPECTED_CONVERSATION = [
    ("system", _INSTRUCTIONS),
    ("system", "client system note"),
    ("user", "q1"),
    ("assistant", "a1"),
    ("user", "q2"),
]


def test_parity_model_receives_baked_instructions_then_client_history():
    """Baked instructions lead; client system/user/assistant turns follow in order."""
    with _harness() as (provider, spec):
        with _openai(spec) as client:
            response = _chat(client, _CLIENT_CONVERSATION)
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "parity-answer"
        with _foundry(spec) as client:
            response = _foundry_respond(client, _CLIENT_CONVERSATION)
        assert response.status_code == 200, response.text
        assert response.json()["output"][0]["content"][0]["text"] == "parity-answer"

    assert [auth for auth, _ in provider.requests] == [f"Bearer {_MODEL_KEY}"] * 2
    for body in provider.bodies():
        assert body["model"] == _MODEL_NAME
        assert "tools" not in body
        assert _conversation(body) == _EXPECTED_CONVERSATION


def test_parity_client_history_is_authoritative_with_session_header():
    """A session header must not replace the history the client actually sent."""
    headers = {"X-AgentKit-Session-Id": "parity-session"}
    with _harness() as (provider, spec), _openai(spec) as client:
        assert _chat(client, [{"role": "user", "content": "u1 original"}], headers).status_code == 200
        edited = [
            {"role": "user", "content": "u1 edited"},
            {"role": "assistant", "content": "a1 edited"},
            {"role": "user", "content": "u2"},
        ]
        assert _chat(client, edited, headers).status_code == 200

    assert _conversation(provider.bodies()[1]) == [
        ("system", _INSTRUCTIONS),
        ("user", "u1 edited"),
        ("assistant", "a1 edited"),
        ("user", "u2"),
    ]


def test_parity_orka_turns_carry_runtime_session_history():
    """Orka sends only the new prompt; the harness supplies committed history."""
    replies = iter(["answer-1", "answer-2", "answer-3"])
    with _harness() as (provider, spec), _orka(spec) as client:
        provider.script = lambda body: _answer(next(replies))
        first = _orka_turn(client, "first")
        second = _orka_turn(client, "second")
        other = _orka_turn(client, "other", runtime_session_id="parity-other-session")

    assert first[-1]["type"] == second[-1]["type"] == other[-1]["type"] == "TurnCompleted"
    assert second[-1]["completed"]["result"] == "answer-2"
    first_body, second_body, other_body = provider.bodies()
    assert _conversation(first_body) == [("system", _INSTRUCTIONS), ("user", "first")]
    assert _conversation(second_body) == [
        ("system", _INSTRUCTIONS),
        ("user", "first"),
        ("assistant", "answer-1"),
        ("user", "second"),
    ]
    assert _conversation(other_body) == [("system", _INSTRUCTIONS), ("user", "other")]


# --------------------------------------------------------------------------- #
# Tool parity
# --------------------------------------------------------------------------- #
def test_parity_tool_roundtrip():
    with _harness(tools=True) as (provider, spec), _openai(spec) as client:
        provider.script = _call_then_answer(("probe_echo", json.dumps({"value": "MARK"})))
        response = _chat(client, [{"role": "user", "content": "use the tool"}])

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "parity-final"
    first, second = provider.bodies()
    assert {tool["function"]["name"] for tool in first["tools"]} == _TOOL_NAMES
    echo = next(tool for tool in first["tools"] if tool["function"]["name"] == "probe_echo")
    assert echo["function"]["parameters"]["required"] == ["value"]
    assert "receipt-MARK" in _tool_results(second)["call_parity_0"]


def test_parity_parallel_tool_calls():
    with _harness(tools=True) as (provider, spec), _openai(spec) as client:
        provider.script = _call_then_answer(
            ("probe_echo", json.dumps({"value": "A"})),
            ("probe_echo", json.dumps({"value": "B"})),
        )
        response = _chat(client, [{"role": "user", "content": "use the tool twice"}])

    assert response.status_code == 200, response.text
    results = _tool_results(provider.bodies()[-1])
    assert "receipt-A" in results["call_parity_0"]
    assert "receipt-B" in results["call_parity_1"]


def test_parity_tool_events_pair_each_call():
    """Each executed tool call starts in_progress and ends under the same ID."""

    async def exercise(spec: AgentSpec) -> tuple[str, list[ToolCallEvent]]:
        events: list[ToolCallEvent] = []

        async def observe(event: ToolCallEvent) -> None:
            events.append(event)

        async with _factory().build_runtime(spec) as runtime:
            result = await runtime.run(RunRequest("use the tools", on_tool_event=observe))
        return result.text, events

    with _harness(tools=True) as (provider, spec):
        provider.script = _call_then_answer(("probe_echo", json.dumps({"value": "A"})), ("probe_fail", "{}"))
        text, events = asyncio.run(exercise(spec))

    assert text == "parity-final"
    by_id: dict[str, list[tuple[str, str]]] = {}
    for event in events:
        by_id.setdefault(event.tool_call_id, []).append((event.tool_name, event.status))
    assert sorted(by_id.values()) == [
        [("probe_echo", "in_progress"), ("probe_echo", "completed")],
        [("probe_fail", "in_progress"), ("probe_fail", "failed")],
    ]


def test_parity_tool_env_is_allowlisted():
    with _harness(tools=True) as (provider, spec), _openai(spec) as client:
        provider.script = _call_then_answer(("probe_env_dump", "{}"))
        response = _chat(client, [{"role": "user", "content": "dump env"}])

    assert response.status_code == 200, response.text
    tool_output = _tool_results(provider.bodies()[-1])["call_parity_0"]
    assert _TOOL_VALUE in tool_output
    assert _MODEL_KEY not in tool_output
    assert _UNDECLARED_SECRET not in tool_output


def test_parity_tool_errors_reach_model_without_tool_details():
    with _harness(tools=True) as (provider, spec), _openai(spec) as client:
        provider.script = _call_then_answer(("probe_fail", "{}"))
        response = _chat(client, [{"role": "user", "content": "use the failing tool"}])

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "parity-final"
    tool_output = _tool_results(provider.bodies()[-1])["call_parity_0"]
    assert tool_output
    assert _TOOL_ERROR_DETAIL not in tool_output


def test_parity_invalid_tool_arguments_are_returned_to_model():
    """Unparseable model arguments must not end the run with an empty answer."""
    with _harness(tools=True) as (provider, spec), _openai(spec) as client:
        provider.script = _call_then_answer(("probe_echo", "{not json"), answer="recovered")
        response = _chat(client, [{"role": "user", "content": "use the tool"}])

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "recovered"
    assert len(provider.requests) == 2
    assert _tool_results(provider.bodies()[-1])["call_parity_0"]


def test_parity_unknown_tool_is_returned_to_model():
    with _harness(tools=True) as (provider, spec), _openai(spec) as client:
        provider.script = _call_then_answer(("probe_missing", "{}"), answer="recovered")
        response = _chat(client, [{"role": "user", "content": "use a missing tool"}])

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "recovered"
    assert _tool_results(provider.bodies()[-1])["call_parity_0"]


# --------------------------------------------------------------------------- #
# Error parity
# --------------------------------------------------------------------------- #
def _upstream_error(status: int, message: str) -> Callable[[dict[str, Any]], _Reply]:
    return lambda body: _Reply(status=status, body={"error": {"message": message, "type": "upstream_error"}})


def test_parity_model_http_errors_are_normalized():
    """Upstream statuses map to runtime-owned codes after the SDK's retry budget."""
    cases = [
        (429, 503, "ModelUnavailable", "model service is unavailable", 3),
        (500, 503, "ModelUnavailable", "model service is unavailable", 3),
        (400, 502, "ModelUpstreamError", "model service request failed", 1),
    ]
    for upstream, status, code, message, attempts in cases:
        with _harness() as (provider, spec), _openai(spec) as client:
            provider.script = _upstream_error(upstream, f"upstream detail {_UNDECLARED_SECRET}")
            response = _chat(client, [{"role": "user", "content": "hi"}])
        assert response.status_code == status, (upstream, response.text)
        assert response.json() == {"error": {"message": message, "type": "agent_error", "code": code}}
        assert len(provider.requests) == attempts, upstream


def test_parity_model_auth_errors_never_echo_credentials():
    """A gateway that echoes the presented key must not leak it on any skin."""

    def echo_auth(body: dict[str, Any]) -> _Reply:
        return _Reply(status=401, body={"error": {"message": f"invalid key Bearer {_MODEL_KEY}"}})

    message = "model service rejected configured credentials"
    with _harness() as (provider, spec):
        provider.script = echo_auth
        with _openai(spec) as client:
            response = _chat(client, [{"role": "user", "content": "hi"}])
        assert response.status_code == 503
        assert response.json() == {"error": {"message": message, "type": "agent_error", "code": "ModelAuthRejected"}}
        with _foundry(spec) as client:
            response = _foundry_respond(client, [{"role": "user", "content": "hi"}])
        assert response.status_code == 503
        assert response.json() == {"error": {"message": message, "code": "ModelAuthRejected", "upstream_status": 401}}
        with _orka(spec) as client:
            frames = _orka_turn(client, "hi")
        assert frames[-1]["type"] == "TurnFailed"
        assert frames[-1]["failed"] == {"reason": "ModelAuthRejected", "message": message, "retryable": False}


def test_parity_model_transport_errors_are_normalized():
    """A dropped model connection is reported without connection details."""
    with _harness() as (provider, spec), _openai(spec) as client:
        provider.script = lambda body: None
        response = _chat(client, [{"role": "user", "content": "hi"}])

    assert response.status_code == 502, response.text
    assert response.json() == {
        "error": {"message": "model service request failed", "type": "agent_error", "code": "ModelUpstreamError"}
    }


def test_parity_malformed_model_response_fails_closed():
    with _harness() as (provider, spec), _openai(spec) as client:
        provider.script = lambda body: _Reply(status=200, body={"unexpected": _UNDECLARED_SECRET})
        response = _chat(client, [{"role": "user", "content": "hi"}])

    assert response.status_code == 502, response.text
    assert response.json() == {"error": {"message": "agent run failed", "type": "agent_error", "code": "AgentRunFailed"}}


# --------------------------------------------------------------------------- #
# Tool session health parity
# --------------------------------------------------------------------------- #
def test_parity_crashed_stdio_tool_fails_health():
    """A dead stdio tool session cannot recover in place, so health must fail."""
    with _harness(tools=True) as (provider, spec):
        provider.script = _call_then_answer(("probe_crash", "{}"))
        with _openai(spec) as client:
            assert client.get("/healthz").status_code == 200
            response = _chat(client, [{"role": "user", "content": "crash the tool"}])
            assert response.status_code == 502, response.text
            assert response.json()["error"] == {
                "message": "MCP tool protocol failed",
                "type": "agent_error",
                "code": "MCPToolProtocolError",
            }
            assert client.get("/healthz").status_code == 503
        with _foundry(spec) as client:
            assert client.get("/readiness").status_code == 200
            _foundry_respond(client, [{"role": "user", "content": "crash the tool"}])
            assert client.get("/readiness").status_code == 503


def test_parity_orka_runtime_startup_failure_hides_tool_credentials(caplog):
    """Orka builds runtimes per turn, so startup errors reach the turn's frames and logs."""
    with _harness(rejecting_remote_tool=True) as (provider, spec), _orka(spec) as client:
        frames = _orka_turn(client, "hi")

    assert frames[-1]["type"] == "TurnFailed"
    assert frames[-1]["failed"] == {"reason": "RuntimeStartFailed", "message": "runtime failed to start", "retryable": False}
    assert provider.requests == []
    assert "runtime session failed to start" in caplog.text
    _assert_no_canary(caplog.text)


def test_parity_orka_rebuilds_runtime_after_crashed_stdio_tool():
    """Orka runtimes are per runtime session, so the next turn starts a fresh tool."""
    with _harness(tools=True) as (provider, spec), _orka(spec) as client:
        provider.script = _call_then_answer(("probe_crash", "{}"))
        crashed = _orka_turn(client, "crash the tool")
        provider.script = _call_then_answer(("probe_echo", json.dumps({"value": "AFTER"})))
        recovered = _orka_turn(client, "use the tool")

    assert crashed[-1]["type"] == "TurnFailed"
    assert crashed[-1]["failed"]["reason"] == "MCPToolProtocolError"
    assert recovered[-1]["type"] == "TurnCompleted"
    assert "receipt-AFTER" in _tool_results(provider.bodies()[-1])["call_parity_0"]
