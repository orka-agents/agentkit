"""Startup-only upstream API selection through real SDK requests and agent runs."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from functools import partial, wraps
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import ConversationTurn, RunRequest
from agentkit_serve_common.runtime import AgentRunError
from agentkit_serve_common.server import create_app

_BASE_URL = "http://localhost:18491/compatible/v1"
_API_KEY_ENV = "TEST_UPSTREAM_API_KEY"
_API_KEY = "offline-upstream-key"
_TOOL_NAME = "probe_echo"
_HISTORY = (
    ConversationTurn("system", "request system"),
    ConversationTurn("user", "first question"),
    ConversationTurn("assistant", "first answer"),
    ConversationTurn("tool", "untrusted client tool result"),
    ConversationTurn("user", ""),
    ConversationTurn("unknown", "ignored"),
)
_MESSAGES = [
    {"role": "system", "content": "Be helpful."},
    {"role": "system", "content": "request system"},
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "hello"},
]


def _spec(*, tools=False, api_key_env=_API_KEY_ENV) -> AgentSpec:
    model = {
        "provider": "openai-compatible",
        "baseURL": _BASE_URL,
        "name": "test-model",
    }
    if api_key_env is not None:
        model["apiKeyEnv"] = api_key_env
    return AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "model-api-test"},
        "model": model,
        "instructions": "Be helpful.",
        "tools": [{"name": "probe", "command": ["unused-mcp"]}] if tools else [],
        "expose": {"openai": True, "port": 8080},
    })


def _select(monkeypatch, model_api):
    monkeypatch.delenv("AGENTKIT_MODEL_API", raising=False)
    if model_api is not None:
        monkeypatch.setenv("AGENTKIT_MODEL_API", model_api)
    monkeypatch.setenv(_API_KEY_ENV, _API_KEY)


@asynccontextmanager
async def _mock_upstream(monkeypatch, handle, **model_options):
    """Replace HTTP transports only, keeping real SDK routing and serialization."""
    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as sync_client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False) as async_client:
            monkeypatch.setattr(agent_factory, "ChatOpenAI", partial(
                ChatOpenAI, http_client=sync_client, http_async_client=async_client, max_retries=0, **model_options,
            ))
            yield


def _reply(model_api, *, arguments=None, call_id="call-1"):
    if model_api != "responses":
        message = {"role": "assistant", "content": "offline reply"}
        if arguments is not None:
            message.update(content=None, tool_calls=[{
                "id": call_id,
                "type": "function",
                "function": {"name": _TOOL_NAME, "arguments": arguments},
            }])
        return {
            "id": f"chatcmpl-{call_id}" if arguments is not None else "chatcmpl-final",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if arguments is not None else "stop",
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    output = [{
        "id": "msg-offline",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "offline reply", "annotations": []}],
    }]
    if arguments is not None:
        output = [{
            "id": f"fc-{call_id}",
            "type": "function_call",
            "call_id": call_id,
            "name": _TOOL_NAME,
            "arguments": arguments,
            "status": "completed",
        }]
    return {
        "id": f"resp-{call_id}" if arguments is not None else "resp-final",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "test-model",
        "error": None,
        "incomplete_details": None,
        "output": output,
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _payload(request, model_api, *, streaming=False):
    endpoint = "responses" if model_api == "responses" else "chat/completions"
    assert str(request.url) == f"{_BASE_URL}/{endpoint}"
    assert request.method == "POST"
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload.get("stream", False) is streaming
    if model_api == "responses":
        assert "messages" not in payload
        # Keep complete explicit history; never rely on provider-hosted state.
        assert "previous_response_id" not in payload
        assert "conversation" not in payload
        assert payload["store"] is False
    else:
        assert "input" not in payload
        assert "store" not in payload
    return payload


def _history(payload, model_api):
    if model_api != "responses":
        return payload["messages"]
    messages = []
    for item in payload["input"]:
        if "role" not in item:
            continue
        content = item["content"]
        if isinstance(content, list):
            content = "".join(block["text"] for block in content)
        messages.append({"role": item["role"], "content": content})
    return messages


@pytest.mark.parametrize("model_api", [None, "chat_completions", "responses", "auto"])
def test_build_model_selects_upstream_api_explicitly(monkeypatch, model_api):
    _select(monkeypatch, model_api)
    model = mock.Mock()
    monkeypatch.setattr(agent_factory, "ChatOpenAI", model)

    assert agent_factory.build_model(_spec(api_key_env=None)) is model.return_value

    model.assert_called_once_with(
        model="test-model",
        base_url=_BASE_URL,
        api_key="not-needed",
        use_responses_api=model_api in {"responses", "auto"},
        store=False if model_api in {"responses", "auto"} else None,
    )


@pytest.mark.parametrize("model_api", ["chat_completions", "responses"])
def test_build_model_uses_concrete_auto_candidate_and_wraps_only_async(monkeypatch, model_api):
    _select(monkeypatch, "invalid-after-startup")
    resolver = mock.Mock(side_effect=AssertionError("candidate must not resolve the environment"))
    constructor = mock.Mock()
    monkeypatch.setattr(agent_factory, "resolve_model_api", resolver)
    monkeypatch.setattr(agent_factory, "ChatOpenAI", constructor)
    model = constructor.return_value
    sync_responses = model.root_client.responses
    async_responses = model.root_async_client.responses
    state = mock.Mock(spec=agent_factory.AutoModelAPIState)

    assert agent_factory.build_model(_spec(), model_api=model_api, auto_state=state) is model

    resolver.assert_not_called()
    constructor.assert_called_once_with(
        model="test-model",
        base_url=_BASE_URL,
        api_key=_API_KEY,
        use_responses_api=model_api == "responses",
        store=False if model_api == "responses" else None,
    )
    if model_api == "responses":
        # Validation stays outside negotiation; only async SDK calls select an API.
        assert isinstance(model.root_client.responses, agent_factory._ValidatedResponsesResource)
        assert model.root_client.responses._resource is sync_responses
        state.wrap_responses.assert_called_once_with(async_responses, client=model.root_async_client)
        assert isinstance(model.root_async_client.responses, agent_factory._ValidatedResponsesResource)
        assert model.root_async_client.responses._resource is state.wrap_responses.return_value
    else:
        state.wrap_responses.assert_not_called()
        assert model.root_client.responses is sync_responses
        assert model.root_async_client.responses is async_responses


@pytest.mark.parametrize("model_api", [None, "chat_completions", "responses"])
def test_runtime_forwards_candidate_or_preserves_default_model_call(monkeypatch, model_api):
    _select(monkeypatch, None)
    spec = _spec()
    state = agent_factory.AutoModelAPIState() if model_api == "responses" else None
    model = SimpleNamespace(use_responses_api=model_api == "responses")
    build = mock.Mock(return_value=model)
    graph = mock.Mock()
    load_tools = mock.AsyncMock(return_value=[])
    monkeypatch.setattr(agent_factory, "build_model", build)
    monkeypatch.setattr(agent_factory, "create_agent", graph)
    monkeypatch.setattr(agent_factory.LangGraphRuntime, "_load_tools", load_tools)
    runtime = agent_factory.LangGraphRuntime(spec, model_api=model_api, auto_state=state)

    async def exercise():
        async with runtime as entered:
            assert entered is runtime
            assert entered.graph is graph.return_value

    asyncio.run(exercise())

    if model_api is None:
        build.assert_called_once_with(spec)
    else:
        build.assert_called_once_with(spec, model_api=model_api, auto_state=state)
    assert graph.call_args.kwargs["model"] is model
    assert graph.call_args.kwargs["middleware"][0].responses_api is (model_api == "responses")
    load_tools.assert_awaited_once_with()
    assert runtime.graph is None


@pytest.mark.parametrize("model_api", [None, "chat_completions", "responses"])
def test_build_runtime_preserves_explicit_and_default_constructor_calls(monkeypatch, model_api):
    _select(monkeypatch, model_api)
    spec = _spec()
    concrete = mock.Mock()
    auto = mock.Mock()
    monkeypatch.setattr(agent_factory, "LangGraphRuntime", concrete)
    monkeypatch.setattr(agent_factory, "AutoModelRuntime", auto)

    assert agent_factory.build_runtime(spec) is concrete.return_value

    concrete.assert_called_once_with(spec)
    auto.assert_not_called()


def test_build_runtime_auto_delegates_concrete_candidate_construction(monkeypatch):
    _select(monkeypatch, "auto")
    spec = _spec()
    concrete = mock.Mock()
    auto = mock.Mock()
    monkeypatch.setattr(agent_factory, "LangGraphRuntime", concrete)
    monkeypatch.setattr(agent_factory, "AutoModelRuntime", auto)

    assert agent_factory.build_runtime(spec) is auto.return_value

    concrete.assert_not_called()
    factory, = auto.call_args.args
    auto.assert_called_once_with(factory)
    state = agent_factory.AutoModelAPIState()
    assert state.selected is None
    assert factory("responses", state) is concrete.return_value
    concrete.assert_called_once_with(spec, model_api="responses", auto_state=state)
    concrete.reset_mock()
    assert factory("chat_completions", None) is concrete.return_value
    concrete.assert_called_once_with(spec, model_api="chat_completions", auto_state=None)
    assert os.environ["AGENTKIT_MODEL_API"] == "auto"


@pytest.mark.parametrize("model_api", [None, "chat_completions", "responses", "auto"])
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
def test_real_agent_uses_selected_wire_format_and_preserves_history(monkeypatch, model_api, output_version):
    _select(monkeypatch, model_api)
    upstream_api = "responses" if model_api == "auto" else model_api
    monkeypatch.setenv("LC_OUTPUT_VERSION", output_version)
    requests = []

    def handle(request):
        payload = _payload(request, upstream_api)
        assert _history(payload, upstream_api) == _MESSAGES
        requests.append(request)
        return httpx.Response(200, json=_reply(upstream_api))

    async def exercise():
        async with _mock_upstream(monkeypatch, handle):
            async with agent_factory.build_runtime(_spec()) as runtime:
                return await runtime.run(RunRequest("hello", history=_HISTORY))

    result = asyncio.run(exercise())

    assert result.text == "offline reply"
    assert result.usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert len(requests) == 1


@pytest.mark.parametrize("include_headers", [False, True])
def test_auto_selects_responses_before_output_validation_rejects_it(monkeypatch, include_headers):
    _select(monkeypatch, "auto")
    requests = []
    state = agent_factory.AutoModelAPIState()

    def handle(request):
        _payload(request, "responses")
        requests.append(request)
        response = _reply("responses")
        response["output"][0]["status"] = "in_progress"
        return httpx.Response(200, json=response)

    async def exercise():
        async with _mock_upstream(monkeypatch, handle, include_response_headers=include_headers):
            model = agent_factory.build_model(_spec(), model_api="responses", auto_state=state)
            with pytest.raises(AgentRunError, match="agent run failed"):
                await model.ainvoke("hello")
            assert state.selected == "responses"

    asyncio.run(exercise())
    assert len(requests) == 1
    assert os.environ["AGENTKIT_MODEL_API"] == "auto"


@pytest.mark.parametrize("model_api", ["", "chat", "Responses", "invalid-api"])
def test_build_model_rejects_invalid_api_before_client_or_auth(monkeypatch, model_api):
    _select(monkeypatch, model_api)
    auth = mock.Mock()
    model = mock.Mock()
    monkeypatch.setattr(agent_factory, "resolve_api_key", auth)
    monkeypatch.setattr(agent_factory, "ChatOpenAI", model)

    with pytest.raises(agent_factory.AgentBuildError, match="AGENTKIT_MODEL_API"):
        agent_factory.build_model(_spec())

    auth.assert_not_called()
    model.assert_not_called()


@pytest.mark.parametrize("model_api", ["chat_completions", "responses"])
@pytest.mark.parametrize("turn_model_api", ["chat_completions", "responses", "invalid-api"])
def test_turn_and_later_process_env_cannot_override_startup_api(monkeypatch, model_api, turn_model_api):
    _select(monkeypatch, model_api)
    requests = []

    def handle(request):
        _payload(request, model_api)
        requests.append(request)
        return httpx.Response(200, json=_reply(model_api))

    async def exercise():
        async with _mock_upstream(monkeypatch, handle):
            async with agent_factory.build_runtime(_spec()) as runtime:
                monkeypatch.setenv("AGENTKIT_MODEL_API", "invalid-after-startup")
                for _ in range(2):
                    result = await runtime.run(RunRequest("hello", env={"AGENTKIT_MODEL_API": turn_model_api}))
                    assert result.text == "offline reply"
                    assert os.environ["AGENTKIT_MODEL_API"] == "invalid-after-startup"

    asyncio.run(exercise())
    assert len(requests) == 2


@pytest.mark.parametrize("model_api", ["chat_completions", "responses"])
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
@pytest.mark.parametrize("invalid_arguments", [False, True])
def test_model_mcp_tool_roundtrip_and_retry(monkeypatch, model_api, output_version, invalid_arguments):
    _select(monkeypatch, model_api)
    monkeypatch.setenv("LC_OUTPUT_VERSION", output_version)
    payloads = []
    tool_calls = []
    lifecycle = []
    events = []
    owner_tasks = []

    class Session:
        async def initialize(self):
            lifecycle.append("initialize")

        async def list_tools(self, **kwargs):
            return ListToolsResult(tools=[Tool(
                name="echo",
                description="Echo the supplied text.",
                inputSchema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )])

        async def call_tool(self, name, arguments, **kwargs):
            tool_calls.append((name, arguments))
            return CallToolResult(content=[TextContent(type="text", text="MCP says hello")])

    class MCPClient:
        def __init__(self, connections, *, tool_name_prefix):
            assert tool_name_prefix is True
            assert connections["probe"]["env"] == {}

        @asynccontextmanager
        async def session(self, name, *, auto_initialize):
            assert name == "probe" and auto_initialize is False
            lifecycle.append("enter")
            owner_tasks.append(asyncio.current_task())
            try:
                yield Session()
            finally:
                owner_tasks.append(asyncio.current_task())
                lifecycle.append("exit")

    monkeypatch.setattr(agent_factory, "MultiServerMCPClient", MCPClient)

    def handle(request):
        payload = _payload(request, model_api)
        assert _history(payload, model_api)[:len(_MESSAGES)] == _MESSAGES
        tool = payload["tools"][0]
        assert tool["type"] == "function"
        function = tool if model_api == "responses" else tool["function"]
        assert function["name"] == _TOOL_NAME
        assert function["parameters"]["properties"]["text"]["type"] == "string"
        payloads.append(payload)
        if invalid_arguments and len(payloads) == 1:
            reply = _reply(model_api, arguments="{invalid-json", call_id="call-invalid")
        elif len(payloads) == 1 + int(invalid_arguments):
            reply = _reply(model_api, arguments='{"text":"hello"}')
        else:
            reply = _reply(model_api)
        return httpx.Response(200, json=reply)

    async def exercise():
        async def observe(event):
            events.append(event)

        async with _mock_upstream(monkeypatch, handle):
            async with agent_factory.build_runtime(_spec(tools=True)) as runtime:
                result = await runtime.run(RunRequest("hello", history=_HISTORY, on_tool_event=observe))
                assert lifecycle == ["enter", "initialize"]
                return result

    result = asyncio.run(exercise())

    assert lifecycle == ["enter", "initialize", "exit"]
    assert owner_tasks[0] is owner_tasks[1]
    assert tool_calls == [("echo", {"text": "hello"})]
    assert result.text == "offline reply"
    count = 2 + int(invalid_arguments)
    assert len(payloads) == count
    assert result.usage == {"prompt_tokens": 3 * count, "completion_tokens": 2 * count, "total_tokens": 5 * count}
    assert [event.status for event in events] == ["in_progress", "completed"]
    assert all(event.tool_name == _TOOL_NAME for event in events)
    assert events[0].tool_call_id == events[1].tool_call_id

    if model_api == "responses":
        inputs = payloads[-1]["input"]
        calls = [item for item in inputs if item.get("type") == "function_call"]
        outputs = [item for item in inputs if item.get("type") == "function_call_output"]
        assert calls[-1]["call_id"] == outputs[-1]["call_id"] == "call-1"
        assert calls[-1]["name"] == _TOOL_NAME
        assert json.loads(calls[-1]["arguments"]) == {"text": "hello"}
        tool_content = outputs[-1]["output"]
        if invalid_arguments:
            assert outputs[0]["call_id"] == "call-invalid"
            assert outputs[0]["output"] == agent_factory._INVALID_TOOL_ARGUMENTS
    else:
        messages = payloads[-1]["messages"]
        assert messages[-2]["tool_calls"][0]["id"] == messages[-1]["tool_call_id"] == "call-1"
        assert messages[-2]["tool_calls"][0]["function"]["name"] == _TOOL_NAME
        assert json.loads(messages[-2]["tool_calls"][0]["function"]["arguments"]) == {"text": "hello"}
        tool_content = messages[-1]["content"]
        if invalid_arguments:
            invalid_result = next(msg for msg in messages if msg.get("tool_call_id") == "call-invalid")
            assert invalid_result["content"] == agent_factory._INVALID_TOOL_ARGUMENTS
    assert "MCP says hello" in str(tool_content)


@pytest.mark.parametrize("status", ["incomplete", "in_progress", "queued", "failed", None])
@pytest.mark.parametrize("tool_call", [False, True])
@pytest.mark.parametrize("response_object", [False, True])
def test_non_completed_responses_fail_before_text_or_tool_execution(monkeypatch, status, tool_call, response_object):
    _select(monkeypatch, "responses")
    requests = []
    tool_calls = []
    events = []

    async def echo(text: str):
        tool_calls.append(text)
        return "must not execute"

    tool = StructuredTool.from_function(coroutine=echo, name=_TOOL_NAME, description="Echo text.")
    monkeypatch.setattr(agent_factory.LangGraphRuntime, "_load_tools", mock.AsyncMock(return_value=[tool]))

    def handle(request):
        _payload(request, "responses")
        requests.append(request)
        if len(requests) > 1:
            return httpx.Response(200, json=_reply("responses"))
        body = _reply("responses", arguments='{"text":"hello"}' if tool_call else None)
        if not response_object:
            body.pop("object")
        if status is None:
            body.pop("status")
        else:
            body["status"] = status
        if status == "incomplete":
            body["incomplete_details"] = {"reason": "max_output_tokens"}
        elif status == "failed":
            body["error"] = {"code": "server_error", "message": f"private {_API_KEY}"}
        return httpx.Response(200, json=body)

    async def exercise():
        async def observe(event):
            events.append(event)

        async with _mock_upstream(monkeypatch, handle):
            async with agent_factory.build_runtime(_spec()) as runtime:
                with pytest.raises(AgentRunError) as caught:
                    await runtime.run(RunRequest("hello", on_tool_event=observe))
                assert caught.value.status == 502
                assert caught.value.code == "AgentRunFailed"
                assert str(caught.value) == "agent run failed"

    asyncio.run(exercise())
    assert len(requests) == 1
    assert not tool_calls
    assert not events


@pytest.mark.parametrize("model_api", ["chat_completions", "responses"])
@pytest.mark.parametrize("status", [404, 429])
def test_real_upstream_errors_are_normalized_without_api_fallback(monkeypatch, model_api, status):
    _select(monkeypatch, model_api)
    requests = []

    def handle(request):
        _payload(request, model_api)
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": f"private {_API_KEY}", "type": "fixture"}})

    async def exercise():
        async with _mock_upstream(monkeypatch, handle):
            async with agent_factory.build_runtime(_spec()) as runtime:
                with pytest.raises(AgentRunError) as caught:
                    await runtime.run(RunRequest("hello"))
                assert caught.value.status == (503 if status == 429 else 502)
                assert caught.value.code == ("ModelUnavailable" if status == 429 else "ModelUpstreamError")
                assert caught.value.upstream_status == status
                assert _API_KEY not in str(caught.value)
                assert "private" not in str(caught.value)

    asyncio.run(exercise())
    assert len(requests) == 1


@pytest.mark.parametrize("model_api", ["chat_completions", "responses"])
def test_chat_serving_protocol_and_stream_guard_are_independent_of_upstream_api(monkeypatch, model_api):
    _select(monkeypatch, model_api)
    requests = []

    def handle(request):
        _payload(request, model_api)
        requests.append(request)
        return httpx.Response(200, json=_reply(model_api))

    async def exercise():
        async with _mock_upstream(monkeypatch, handle):
            with TestClient(create_app(_spec(), agent_factory)) as client:
                payload = {"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}
                rejected = client.post("/v1/chat/completions", json={**payload, "stream": True})
                assert rejected.status_code == 400
                assert rejected.json()["error"]["code"] == "stream_unsupported"
                assert not requests
                return client.post("/v1/chat/completions", json=payload)

    response = asyncio.run(exercise())

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "offline reply"
    assert response.json()["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert len(requests) == 1



def _responses_stream(response, termination, *, done_status=None):
    """Emit native SSE; only a completed response gets a completion event."""
    events = []

    def event(kind, **fields):
        events.append({"type": kind, "sequence_number": len(events), **fields})

    started = {**response, "status": "in_progress", "output": [], "usage": None}
    event("response.created", response=started)
    event("response.in_progress", response=started)
    for index, item in enumerate(response["output"]):
        added = {**item, "status": "in_progress"}
        if item["type"] == "function_call":
            added["arguments"] = ""
        else:
            added["content"] = []
        event("response.output_item.added", output_index=index, item=added)
        ref = {"item_id": item["id"], "output_index": index}
        if item["type"] == "function_call":
            arguments = item["arguments"]
            midpoint = len(arguments) // 2
            for delta in (arguments[:midpoint], arguments[midpoint:]):
                event("response.function_call_arguments.delta", **ref, delta=delta)
            if termination != "eof_after_delta":
                event("response.function_call_arguments.done", **ref, name=item["name"], arguments=arguments)
        else:
            ref["content_index"] = 0
            part = item["content"][0]
            event("response.content_part.added", **ref, part={**part, "text": ""})
            event("response.output_text.delta", **ref, delta=part["text"], logprobs=[])
            if termination != "eof_after_delta":
                event("response.output_text.done", **ref, text=part["text"], logprobs=[])
                event("response.content_part.done", **ref, part=part)
        if termination != "eof_after_delta":
            done_item = {**item, "status": done_status} if done_status is not None else item
            event("response.output_item.done", output_index=index, item=done_item)
    if termination in {"completed", "incomplete", "failed"}:
        terminal = {**response, "status": termination}
        if termination == "incomplete":
            terminal["incomplete_details"] = {"reason": "max_output_tokens"}
        elif termination == "failed":
            terminal["error"] = {"code": "server_error", "message": "private provider error"}
        event(f"response.{termination}", response=terminal)
    return b"".join(
        f"event: {item['type']}\ndata: {json.dumps(item)}\n\n".encode()
        for item in events
    )


@pytest.mark.parametrize("termination", ["completed", "incomplete", "failed", "eof_after_delta", "eof_after_item"])
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
@pytest.mark.parametrize("tool_call", [False, True])
def test_responses_stream_requires_completion_before_text_or_tools(monkeypatch, termination, output_version, tool_call):
    _select(monkeypatch, "responses")
    monkeypatch.setenv("LC_OUTPUT_VERSION", output_version)
    requests = []
    tool_calls = []
    events = []
    statuses = []
    after_model = agent_factory._ModelResultMiddleware.after_model

    @wraps(after_model)
    def observe_model(self, state, runtime):
        statuses.append(state["messages"][-1].response_metadata.get("status"))
        return after_model(self, state, runtime)

    monkeypatch.setattr(agent_factory._ModelResultMiddleware, "after_model", observe_model)

    async def echo(text: str):
        tool_calls.append(text)
        return "tool ran after response completed"

    tool = StructuredTool.from_function(coroutine=echo, name=_TOOL_NAME, description="Echo text.")
    monkeypatch.setattr(agent_factory.LangGraphRuntime, "_load_tools", mock.AsyncMock(return_value=[tool]))

    def handle(request):
        _payload(request, "responses", streaming=True)
        requests.append(request)
        if len(requests) == 1:
            response = _reply("responses", arguments='{"text":"hello"}' if tool_call else None)
            content = _responses_stream(response, termination)
        else:
            # A guard regression finishes rather than looping through more calls.
            content = _responses_stream(_reply("responses"), "completed")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    async def exercise():
        async def observe(event):
            events.append(event)

        async with _mock_upstream(monkeypatch, handle, streaming=True):
            async with agent_factory.build_runtime(_spec()) as runtime:
                request = RunRequest("hello", on_tool_event=observe)
                if termination == "completed":
                    result = await runtime.run(request)
                    assert result.text == "offline reply"
                    count = 2 if tool_call else 1
                    assert result.usage == {
                        "prompt_tokens": 3 * count, "completion_tokens": 2 * count, "total_tokens": 5 * count,
                    }
                else:
                    with pytest.raises(AgentRunError) as caught:
                        await runtime.run(request)
                    assert caught.value.status == 502
                    assert caught.value.code == "AgentRunFailed"
                    assert str(caught.value) == "agent run failed"

    asyncio.run(exercise())
    if termination == "completed":
        assert len(requests) == (2 if tool_call else 1)
        assert statuses == (["completed", "completed"] if tool_call else ["completed"])
        assert tool_calls == (["hello"] if tool_call else [])
        assert [event.status for event in events] == (["in_progress", "completed"] if tool_call else [])
        if tool_call:
            assert events[0].tool_call_id == events[1].tool_call_id
    else:
        assert len(requests) == 1
        assert not tool_calls
        assert not events
        # SDK 1.0 ignores incomplete/failed terminal events. Its missing status
        # must fail closed just like newer SDKs that retain the incomplete status.
        assert statuses in ([], [None], ["incomplete"])



@pytest.mark.parametrize("include_headers", [False, True])
@pytest.mark.parametrize("item_status", ["incomplete", "in_progress", "completed", None])
@pytest.mark.parametrize("wire", ["json", "sse_item_done", "sse_response_completed"])
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
@pytest.mark.parametrize("tool_call", [False, True])
def test_completed_response_requires_completed_output_items(
    monkeypatch, item_status, wire, output_version, tool_call, include_headers,
):
    _select(monkeypatch, "responses")
    monkeypatch.setenv("LC_OUTPUT_VERSION", output_version)
    requests = []
    tool_calls = []
    events = []
    accepted = item_status in {"completed", None}
    streaming = wire != "json"

    async def echo(text: str):
        tool_calls.append(text)
        return "tool ran after output item completed"

    tool = StructuredTool.from_function(coroutine=echo, name=_TOOL_NAME, description="Echo text.")
    monkeypatch.setattr(agent_factory.LangGraphRuntime, "_load_tools", mock.AsyncMock(return_value=[tool]))

    def handle(request):
        _payload(request, "responses", streaming=streaming)
        requests.append(request)
        response = _reply("responses", arguments='{"text":"hello"}' if tool_call and len(requests) == 1 else None)
        if len(requests) == 1:
            if item_status is None:
                response["output"][0].pop("status")
            else:
                response["output"][0]["status"] = item_status
        if not streaming:
            return httpx.Response(200, json=response)
        # Isolate final-snapshot validation from output_item.done validation.
        done_status = "completed" if wire == "sse_response_completed" else None
        content = _responses_stream(response, "completed", done_status=done_status)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    async def exercise():
        async def observe(event):
            events.append(event)

        async with _mock_upstream(monkeypatch, handle, streaming=streaming, include_response_headers=include_headers):
            async with agent_factory.build_runtime(_spec()) as runtime:
                request = RunRequest("hello", on_tool_event=observe)
                if accepted:
                    result = await runtime.run(request)
                    assert result.text == "offline reply"
                    count = 2 if tool_call else 1
                    assert result.usage == {
                        "prompt_tokens": 3 * count, "completion_tokens": 2 * count, "total_tokens": 5 * count,
                    }
                else:
                    with pytest.raises(AgentRunError) as caught:
                        await runtime.run(request)
                    assert caught.value.status == 502
                    assert caught.value.code == "AgentRunFailed"
                    assert str(caught.value) == "agent run failed"

    asyncio.run(exercise())
    if accepted:
        assert len(requests) == (2 if tool_call else 1)
        assert tool_calls == (["hello"] if tool_call else [])
        assert [event.status for event in events] == (["in_progress", "completed"] if tool_call else [])
        if tool_call:
            assert events[0].tool_call_id == events[1].tool_call_id
    else:
        assert len(requests) == 1
        assert not tool_calls
        assert not events



@pytest.mark.parametrize("item_status", ["in_progress", None])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
@pytest.mark.parametrize("tool_call", [False, True])
def test_sync_model_validates_raw_response_items(monkeypatch, item_status, streaming, output_version, tool_call):
    _select(monkeypatch, "responses")
    monkeypatch.setenv("LC_OUTPUT_VERSION", output_version)
    requests = []

    def handle(request):
        _payload(request, "responses", streaming=streaming)
        requests.append(request)
        response = _reply("responses", arguments='{"text":"hello"}' if tool_call else None)
        if item_status is None:
            response["output"][0].pop("status")
        else:
            response["output"][0]["status"] = item_status
        if not streaming:
            return httpx.Response(200, json=response)
        content = _responses_stream(response, "completed", done_status="completed")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    async def exercise():
        async with _mock_upstream(monkeypatch, handle, streaming=streaming, include_response_headers=True):
            model = agent_factory.build_model(_spec())
            if item_status is None:
                result = model.invoke("hello")
                if tool_call:
                    assert result.tool_calls == [{
                        "name": _TOOL_NAME, "args": {"text": "hello"}, "id": "call-1", "type": "tool_call",
                    }]
                else:
                    assert agent_factory._message_text(result) == "offline reply"
            else:
                with pytest.raises(AgentRunError) as caught:
                    model.invoke("hello")
                assert caught.value.code == "AgentRunFailed"
                assert str(caught.value) == "agent run failed"

    asyncio.run(exercise())
    assert len(requests) == 1


@pytest.mark.parametrize("model_api", ["responses", "auto"])
@pytest.mark.parametrize("include_headers", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_validated_response_stream_closes_on_item_rejection_or_cancellation(
    monkeypatch, model_api, include_headers, cancel,
):
    _select(monkeypatch, model_api)
    responses = []
    streams = []
    tool_calls = []
    events = []

    async def exercise():
        waiting = asyncio.Event()

        class ResponseBody(httpx.AsyncByteStream):
            def __init__(self, content):
                self.content = content
                self.closes = 0

            async def __aiter__(self):
                yield self.content
                if cancel:
                    waiting.set()
                    await asyncio.Future()

            async def aclose(self):
                self.closes += 1

        async def echo(text: str):
            tool_calls.append(text)
            return "must not execute"

        tool = StructuredTool.from_function(coroutine=echo, name=_TOOL_NAME, description="Echo text.")
        monkeypatch.setattr(agent_factory.LangGraphRuntime, "_load_tools", mock.AsyncMock(return_value=[tool]))

        def handle(request):
            _payload(request, "responses", streaming=True)
            body = _reply("responses", arguments='{"text":"hello"}')
            if not cancel:
                body["output"][0]["status"] = "incomplete"
            content = _responses_stream(body, "eof_after_delta" if cancel else "completed")
            stream = ResponseBody(content)
            streams.append(stream)
            response = httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
            responses.append(response)
            return response

        async def observe(event):
            events.append(event)

        async with _mock_upstream(monkeypatch, handle, streaming=True, include_response_headers=include_headers):
            async with agent_factory.build_runtime(_spec()) as runtime:
                run = asyncio.create_task(runtime.run(RunRequest("hello", on_tool_event=observe)))
                if cancel:
                    try:
                        await asyncio.wait_for(waiting.wait(), timeout=3)
                    finally:
                        run.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await asyncio.wait_for(run, timeout=3)
                else:
                    with pytest.raises(AgentRunError) as caught:
                        await run
                    assert caught.value.code == "AgentRunFailed"
                # Verify SDK stream cleanup before closing the runtime/HTTP client.
                assert len(responses) == 1 and responses[0].is_closed
                assert streams[0].closes == 1
                if model_api == "auto":
                    assert runtime.state.selected == "responses"

    asyncio.run(exercise())
    assert not tool_calls
    assert not events
