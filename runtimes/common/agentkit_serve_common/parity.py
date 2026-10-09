"""Wire-level cross-runtime parity suite for AgentKit runtime adapters.

``conformance.py`` swaps each adapter's model client for an in-process double, so
it cannot see what an adapter actually sends to a model. This suite builds the
adapter's REAL runtime from an AgentSpec whose ``model.baseURL`` is a scripted
OpenAI-compatible endpoint on loopback and whose tool is a stdio MCP fixture
server, then drives the shared protocol skins with both startup
``AGENTKIT_MODEL_API`` values. Every adapter must send the same conversation,
return the same client-visible results and errors, and keep secret canaries out
of everything a client can read.

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
    "pytest_generate_tests",
    "test_parity_model_receives_baked_instructions_then_client_history",
    "test_parity_dual_openai_endpoints_with_sdk",
    "test_parity_responses_tool_roundtrip",
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
    "test_parity_non_completed_responses_fail_before_tool_execution",
    "test_parity_crashed_stdio_tool_fails_health",
    "test_parity_orka_runtime_startup_failure_hides_tool_credentials",
    "test_parity_orka_rebuilds_runtime_after_crashed_stdio_tool",
]


def pytest_generate_tests(metafunc: Any) -> None:
    """Run every inherited parity assertion against both startup API selections."""
    if "model_api" in metafunc.fixturenames:
        metafunc.parametrize("model_api", ["chat_completions", "responses"])
    if "response_status" in metafunc.fixturenames:
        metafunc.parametrize("response_status", ["incomplete", "failed", "in_progress"])


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
        if _request_messages(body)[-1]["role"] == "tool":
            return _answer(answer)
        return _tool_calls(*calls)

    return script


def _request_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Read native wire shapes without changing the recorded upstream request."""
    if "messages" in body:
        return body["messages"]
    messages: list[dict[str, Any]] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    inputs = body["input"]
    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    for item in inputs:
        if item.get("type") == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item["call_id"],
                            "type": "function",
                            "function": {"name": item["name"], "arguments": item["arguments"]},
                        }
                    ],
                }
            )
        elif item.get("type") == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
        elif "role" in item:
            role = "system" if item["role"] == "developer" else item["role"]
            messages.append({**item, "role": role})
    return messages


def _tool_definitions(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Chat wraps function definitions; Responses uses flat definitions."""
    return [tool.get("function", tool) for tool in body.get("tools", [])]


def _model_wire_response(body: dict[str, Any], reply: _Reply, model_api: str) -> tuple[str, bytes]:
    """Encode the bounded script as native JSON or SDK-consumable SSE for either API."""
    if reply.message is None:
        if (
            model_api == "responses"
            and body.get("stream")
            and reply.status == 200
            and isinstance(reply.body, dict)
            and reply.body.get("object") == "response"
        ):
            return "text/event-stream", _responses_sse(reply.body)
        payload = reply.body if isinstance(reply.body, bytes) else json.dumps(reply.body).encode()
        return "application/json", payload
    if model_api == "responses":
        response = _responses_body(body["model"], reply.message)
        if body.get("stream"):
            return "text/event-stream", _responses_sse(response)
        return "application/json", json.dumps(response).encode()
    base = {"id": f"chatcmpl-{uuid.uuid4().hex}", "created": 1, "model": body["model"]}
    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    if not body.get("stream"):
        choice = {"index": 0, "message": reply.message, "finish_reason": reply.finish}
        return "application/json", json.dumps(
            {**base, "object": "chat.completion", "choices": [choice], "usage": usage}
        ).encode()
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
    return "text/event-stream", payload + b"data: [DONE]\n\n"


def _responses_body(model: str, message: dict[str, Any]) -> dict[str, Any]:
    output = []
    if message.get("content") is not None:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": message["content"], "annotations": [], "logprobs": []}],
            }
        )
    for call in message.get("tool_calls") or []:
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            }
        )
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": 1,
        "model": model,
        "status": "completed",
        "output": output,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "store": False,
        "metadata": {},
        "text": {"format": {"type": "text"}},
        "usage": {
            "input_tokens": 3,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 5,
        },
    }


def _responses_sse(response: dict[str, Any]) -> bytes:
    events: list[dict[str, Any]] = []

    def event(kind: str, **fields: Any) -> None:
        events.append({"type": kind, "sequence_number": len(events), **fields})

    started = {
        **response,
        "status": "in_progress",
        "output": [],
        "usage": None,
        "error": None,
        "incomplete_details": None,
    }
    event("response.created", response=started)
    event("response.in_progress", response=started)
    for index, item in enumerate(response["output"]):
        added = {**item, "status": "in_progress"}
        added["content" if item["type"] == "message" else "arguments"] = [] if item["type"] == "message" else ""
        event("response.output_item.added", output_index=index, item=added)
        ref = {"item_id": item["id"], "output_index": index}
        if item["type"] == "message":
            part = item["content"][0]
            ref["content_index"] = 0
            event("response.content_part.added", **ref, part={**part, "text": ""})
            event("response.output_text.delta", **ref, delta=part["text"], logprobs=[])
            event("response.output_text.done", **ref, text=part["text"], logprobs=[])
            event("response.content_part.done", **ref, part=part)
        else:
            event("response.function_call_arguments.delta", **ref, delta=item["arguments"])
            event("response.function_call_arguments.done", **ref, name=item["name"], arguments=item["arguments"])
        event("response.output_item.done", output_index=index, item=item)
    if response.get("status") in {"completed", "incomplete", "failed"}:
        event(f"response.{response['status']}", response=response)
    # An unfinished response ends at EOF, without inventing a completion event.
    return b"".join(f"event: {item['type']}\ndata: {json.dumps(item)}\n\n".encode() for item in events)


class _ScriptedProvider:
    """Loopback Chat/Responses endpoint that records native requests and replays a script."""

    def __init__(self, model_api: str = "chat_completions") -> None:
        self.model_api = model_api
        self.paths: list[str] = []
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
            self.paths.append(handler.path)
        expected_path = "/v1/responses" if self.model_api == "responses" else "/v1/chat/completions"
        if handler.path != expected_path:
            self._send(handler, 400, "application/json", b'{"error":{"message":"wrong upstream API"}}')
            return
        reply = self.script(body)
        if reply is None:
            handler.close_connection = True
            return
        content_type, payload = _model_wire_response(body, reply, self.model_api)
        self._send(handler, reply.status, content_type, payload, reply.headers)

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
def _canary_env(model_api: str) -> Iterator[None]:
    values = {
        "AGENTKIT_MODEL_API": model_api,
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
def _harness(
    *, model_api: str, tools: bool = False, rejecting_remote_tool: bool = False
) -> Iterator[tuple[_ScriptedProvider, AgentSpec]]:
    with tempfile.TemporaryDirectory() as tmp, _ScriptedProvider(model_api) as provider, _canary_env(model_api):
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
        expected_path = "/v1/responses" if model_api == "responses" else "/v1/chat/completions"
        assert provider.paths == [expected_path] * len(provider.requests)
        if model_api == "responses":
            for body in provider.bodies():
                assert "messages" not in body
                assert body.get("store") is False
                assert not body.get("previous_response_id")
                assert not body.get("conversation")


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
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
            if isinstance(part, (str, dict))
        )
    return json.dumps(content)


def _conversation(body: dict[str, Any]) -> list[tuple[str, str]]:
    """The system/user/assistant text turns one model request carried."""
    return [
        (message["role"], _content_text(message.get("content")))
        for message in _request_messages(body)
        if message["role"] in {"system", "user", "assistant"} and _content_text(message.get("content"))
    ]


def _tool_results(body: dict[str, Any]) -> dict[str, str]:
    return {
        message.get("tool_call_id"): _content_text(message.get("content"))
        for message in _request_messages(body)
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


def test_parity_model_receives_baked_instructions_then_client_history(model_api):
    """Baked instructions lead; client system/user/assistant turns follow in order."""
    with _harness(model_api=model_api) as (provider, spec):
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


def test_parity_dual_openai_endpoints_with_sdk(model_api, openai_client_factory):
    """Both client APIs use the same configured upstream API and baked agent."""
    messages = [message for message in _CLIENT_CONVERSATION if message["role"] != "tool"]
    with _harness(model_api=model_api) as (provider, spec), _openai(spec) as client:
        sdk = openai_client_factory(
            base_url="http://testserver/v1", api_key="unused", http_client=client,
            _strict_response_validation=True,
        )
        chat = sdk.chat.completions.create(model=_MODEL_NAME, messages=messages)
        response = sdk.responses.create(model=_MODEL_NAME, input=messages, store=False)
        assert chat.choices[0].message.content == response.output_text == "parity-answer"
        assert response.status == "completed"
        assert response.usage.total_tokens > 0

    assert [auth for auth, _ in provider.requests] == [f"Bearer {_MODEL_KEY}"] * 2
    for body in provider.bodies():
        assert body["model"] == _MODEL_NAME
        assert _conversation(body) == _EXPECTED_CONVERSATION


def test_parity_client_history_is_authoritative_with_session_header(model_api):
    """A session header must not replace the history the client actually sent."""
    headers = {"X-AgentKit-Session-Id": "parity-session"}
    with _harness(model_api=model_api) as (provider, spec), _openai(spec) as client:
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


def test_parity_orka_turns_carry_runtime_session_history(model_api):
    """Orka sends only the new prompt; the harness supplies committed history."""
    replies = iter(["answer-1", "answer-2", "answer-3"])
    with _harness(model_api=model_api) as (provider, spec), _orka(spec) as client:
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
def test_parity_tool_roundtrip(model_api):
    with _harness(model_api=model_api, tools=True) as (provider, spec), _openai(spec) as client:
        provider.script = _call_then_answer(("probe_echo", json.dumps({"value": "MARK"})))
        response = _chat(client, [{"role": "user", "content": "use the tool"}])

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "parity-final"
    first, second = provider.bodies()
    assert {tool["name"] for tool in _tool_definitions(first)} == _TOOL_NAMES
    echo = next(tool for tool in _tool_definitions(first) if tool["name"] == "probe_echo")
    assert echo["parameters"]["required"] == ["value"]
    assert "receipt-MARK" in _tool_results(second)["call_parity_0"]


def test_parity_responses_tool_roundtrip(model_api):
    """Client Responses requests retain baked tool execution with either upstream API."""
    with _harness(model_api=model_api, tools=True) as (provider, spec), _openai(spec) as client:
        provider.script = _call_then_answer(("probe_echo", json.dumps({"value": "MARK"})))
        response = client.post("/v1/responses", json={"input": "use the tool", "store": False})

    assert response.status_code == 200, response.text
    assert response.json()["output"][0]["content"][0]["text"] == "parity-final"
    first, second = provider.bodies()
    assert {tool["name"] for tool in _tool_definitions(first)} == _TOOL_NAMES
    assert "receipt-MARK" in _tool_results(second)["call_parity_0"]


def test_parity_parallel_tool_calls(model_api):
    with _harness(model_api=model_api, tools=True) as (provider, spec), _openai(spec) as client:
        provider.script = _call_then_answer(
            ("probe_echo", json.dumps({"value": "A"})),
            ("probe_echo", json.dumps({"value": "B"})),
        )
        response = _chat(client, [{"role": "user", "content": "use the tool twice"}])

    assert response.status_code == 200, response.text
    results = _tool_results(provider.bodies()[-1])
    assert "receipt-A" in results["call_parity_0"]
    assert "receipt-B" in results["call_parity_1"]


def test_parity_tool_events_pair_each_call(model_api):
    """Each executed tool call starts in_progress and ends under the same ID."""

    async def exercise(spec: AgentSpec) -> tuple[str, list[ToolCallEvent]]:
        events: list[ToolCallEvent] = []

        async def observe(event: ToolCallEvent) -> None:
            events.append(event)

        async with _factory().build_runtime(spec) as runtime:
            result = await runtime.run(RunRequest("use the tools", on_tool_event=observe))
        return result.text, events

    with _harness(model_api=model_api, tools=True) as (provider, spec):
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


def test_parity_tool_env_is_allowlisted(model_api):
    with _harness(model_api=model_api, tools=True) as (provider, spec), _openai(spec) as client:
        provider.script = _call_then_answer(("probe_env_dump", "{}"))
        response = _chat(client, [{"role": "user", "content": "dump env"}])

    assert response.status_code == 200, response.text
    tool_output = _tool_results(provider.bodies()[-1])["call_parity_0"]
    assert _TOOL_VALUE in tool_output
    assert _MODEL_KEY not in tool_output
    assert _UNDECLARED_SECRET not in tool_output


def test_parity_tool_errors_reach_model_without_tool_details(model_api):
    with _harness(model_api=model_api, tools=True) as (provider, spec), _openai(spec) as client:
        provider.script = _call_then_answer(("probe_fail", "{}"))
        response = _chat(client, [{"role": "user", "content": "use the failing tool"}])

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "parity-final"
    tool_output = _tool_results(provider.bodies()[-1])["call_parity_0"]
    assert tool_output
    assert _TOOL_ERROR_DETAIL not in tool_output


def test_parity_invalid_tool_arguments_are_returned_to_model(model_api):
    """Unparseable model arguments must not end the run with an empty answer."""
    with _harness(model_api=model_api, tools=True) as (provider, spec), _openai(spec) as client:
        provider.script = _call_then_answer(("probe_echo", "{not json"), answer="recovered")
        response = _chat(client, [{"role": "user", "content": "use the tool"}])

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "recovered"
    assert len(provider.requests) == 2
    assert _tool_results(provider.bodies()[-1])["call_parity_0"]


def test_parity_unknown_tool_is_returned_to_model(model_api):
    with _harness(model_api=model_api, tools=True) as (provider, spec), _openai(spec) as client:
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


def test_parity_model_http_errors_are_normalized(model_api):
    """Upstream statuses map to runtime-owned codes after the SDK's retry budget."""
    cases = [
        (429, 503, "ModelUnavailable", "model service is unavailable", 3),
        (500, 503, "ModelUnavailable", "model service is unavailable", 3),
        (400, 502, "ModelUpstreamError", "model service request failed", 1),
    ]
    for upstream, status, code, message, attempts in cases:
        with _harness(model_api=model_api) as (provider, spec), _openai(spec) as client:
            provider.script = _upstream_error(upstream, f"upstream detail {_UNDECLARED_SECRET}")
            response = _chat(client, [{"role": "user", "content": "hi"}])
        assert response.status_code == status, (upstream, response.text)
        assert response.json() == {"error": {"message": message, "type": "agent_error", "code": code}}
        assert len(provider.requests) == attempts, upstream


def test_parity_model_auth_errors_never_echo_credentials(model_api):
    """A gateway that echoes the presented key must not leak it on any skin."""

    def echo_auth(body: dict[str, Any]) -> _Reply:
        return _Reply(status=401, body={"error": {"message": f"invalid key Bearer {_MODEL_KEY}"}})

    message = "model service rejected configured credentials"
    with _harness(model_api=model_api) as (provider, spec):
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


def test_parity_model_transport_errors_are_normalized(model_api):
    """A dropped model connection is reported without connection details."""
    with _harness(model_api=model_api) as (provider, spec), _openai(spec) as client:
        provider.script = lambda body: None
        response = _chat(client, [{"role": "user", "content": "hi"}])

    assert response.status_code == 502, response.text
    assert response.json() == {
        "error": {"message": "model service request failed", "type": "agent_error", "code": "ModelUpstreamError"}
    }


def test_parity_malformed_model_response_fails_closed(model_api):
    with _harness(model_api=model_api) as (provider, spec), _openai(spec) as client:
        provider.script = lambda body: _Reply(status=200, body={"unexpected": _UNDECLARED_SECRET})
        response = _chat(client, [{"role": "user", "content": "hi"}])

    assert response.status_code == 502, response.text
    assert response.json() == {
        "error": {"message": "agent run failed", "type": "agent_error", "code": "AgentRunFailed"}
    }


def test_parity_non_completed_responses_fail_before_tool_execution(response_status):
    """Partial/failed Responses must not authorize execution of their function calls."""
    from .runtime import AgentRunError

    async def exercise(spec: AgentSpec) -> list[ToolCallEvent]:
        events: list[ToolCallEvent] = []
        rejection: AgentRunError | None = None

        async def observe(event: ToolCallEvent) -> None:
            events.append(event)

        async with _factory().build_runtime(spec) as runtime:
            try:
                await runtime.run(RunRequest("use the tool", on_tool_event=observe))
            except AgentRunError as exc:
                rejection = exc
        assert rejection is not None, f"{response_status} Responses result was accepted after {len(events)} tool events"
        assert rejection.status == 502
        _assert_no_canary(str(rejection))
        return events

    with _harness(model_api="responses", tools=True) as (provider, spec):
        reply = _tool_calls(("probe_echo", json.dumps({"value": "MUST-NOT-RUN"})))
        body = _responses_body(_MODEL_NAME, reply.message)
        body["status"] = response_status
        if response_status == "incomplete":
            body["incomplete_details"] = {"reason": "max_output_tokens"}
        elif response_status == "failed":
            body["error"] = {"code": "server_error", "message": _UNDECLARED_SECRET}
        # If an adapter executes the partial call, finish its next model request
        # rather than let a bad adapter loop indefinitely inside this fixture.
        provider.script = lambda request: (
            _answer("unexpected tool execution") if _tool_results(request) else _Reply(body=body)
        )
        events = asyncio.run(exercise(spec))
        assert events == []
        assert len(provider.requests) == 1


# --------------------------------------------------------------------------- #
# Tool session health parity
# --------------------------------------------------------------------------- #
def test_parity_crashed_stdio_tool_fails_health(model_api):
    """A dead stdio tool session cannot recover in place, so health must fail."""
    with _harness(model_api=model_api, tools=True) as (provider, spec):
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


def test_parity_orka_runtime_startup_failure_hides_tool_credentials(model_api, caplog):
    """Orka builds runtimes per turn, so startup errors reach the turn's frames and logs."""
    with _harness(model_api=model_api, rejecting_remote_tool=True) as (provider, spec), _orka(spec) as client:
        frames = _orka_turn(client, "hi")

    assert frames[-1]["type"] == "TurnFailed"
    assert frames[-1]["failed"] == {
        "reason": "RuntimeStartFailed",
        "message": "runtime failed to start",
        "retryable": False,
    }
    assert provider.requests == []
    assert "runtime session failed to start" in caplog.text
    _assert_no_canary(caplog.text)


def test_parity_orka_rebuilds_runtime_after_crashed_stdio_tool(model_api):
    """Orka runtimes are per runtime session, so the next turn starts a fresh tool."""
    with _harness(model_api=model_api, tools=True) as (provider, spec), _orka(spec) as client:
        provider.script = _call_then_answer(("probe_crash", "{}"))
        crashed = _orka_turn(client, "crash the tool")
        provider.script = _call_then_answer(("probe_echo", json.dumps({"value": "AFTER"})))
        recovered = _orka_turn(client, "use the tool")

    assert crashed[-1]["type"] == "TurnFailed"
    assert crashed[-1]["failed"]["reason"] == "MCPToolProtocolError"
    assert recovered[-1]["type"] == "TurnCompleted"
    assert "receipt-AFTER" in _tool_results(provider.bodies()[-1])["call_parity_0"]
