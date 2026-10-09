"""Direct upstream API selection through the real Pydantic AI/OpenAI HTTP clients."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import ConversationTurn, RunRequest
from agentkit_serve_common.orka import ORKA_HARNESS_VERSION, create_orka_app
from agentkit_serve_common.runtime import AgentRunError
from agentkit_serve_common.server import create_app


BASE_URL = "http://docker-model.test:8080/custom/v1"
KEY = "synthetic-upstream-key"
PRIVATE = "private-model-output"
ARGUMENTS = '{"value":"hello"}'


def _spec() -> AgentSpec:
    return AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "model-api-test"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": BASE_URL,
            "name": "local-model",
            "apiKeyEnv": "MODEL_API_TEST_KEY",
        },
        "instructions": "Baked rules first.",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })


def _chat_frame(delta, finish=None, usage=None):
    return (
        "data: " + json.dumps({
            "id": "completion-test", "object": "chat.completion.chunk", "created": 1,
            "model": "local-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            "usage": usage,
        }) + "\n\n"
    ).encode()


def _responses_body(*, tool=False, status="completed"):
    item = (
        {"type": "function_call", "id": "fc-probe", "call_id": "call-probe",
         "name": "probe", "arguments": ARGUMENTS, "status": "completed"}
        if tool else
        {"type": "message", "id": "msg-test", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": PRIVATE, "annotations": [], "logprobs": []}]}
    )
    return {
        "id": "resp-test", "object": "response", "created_at": 1, "model": "local-model",
        "status": status, "output": [item], "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "instructions": None, "max_output_tokens": None, "parallel_tool_calls": True,
        "temperature": 1, "tool_choice": "auto", "tools": [], "top_p": 1,
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}},
    }


def _responses_events(*, tool=False, terminal="response.completed"):
    body = _responses_body(tool=tool)
    item = body["output"][0]
    events = [("response.created", {"response": {**body, "status": "in_progress", "output": [], "usage": None}})]
    if tool:
        events.extend([
            ("response.output_item.added", {"output_index": 0, "item": {**item, "arguments": "", "status": "in_progress"}}),
            ("response.function_call_arguments.delta", {"item_id": item["id"], "output_index": 0, "delta": ARGUMENTS}),
            ("response.function_call_arguments.done", {"item_id": item["id"], "output_index": 0, "arguments": ARGUMENTS}),
        ])
    else:
        events.extend([
            ("response.output_item.added", {"output_index": 0, "item": {**item, "content": [], "status": "in_progress"}}),
            ("response.content_part.added", {"item_id": item["id"], "output_index": 0, "content_index": 0,
                                             "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}}),
            ("response.output_text.delta", {"item_id": item["id"], "output_index": 0, "content_index": 0,
                                            "delta": PRIVATE, "logprobs": []}),
            ("response.output_text.done", {"item_id": item["id"], "output_index": 0, "content_index": 0,
                                           "text": PRIVATE, "logprobs": []}),
            ("response.content_part.done", {"item_id": item["id"], "output_index": 0, "content_index": 0,
                                            "part": item["content"][0]}),
        ])
    events.append(("response.output_item.done", {"output_index": 0, "item": item}))
    if terminal == "error":
        events.append((terminal, {"code": "server_error", "message": PRIVATE + KEY, "param": None}))
    elif terminal is not None:
        status = terminal.removeprefix("response.")
        events.append((terminal, {"response": _responses_body(tool=tool, status=status)}))
    return b"".join(
        ("event: " + kind + "\ndata: " + json.dumps({"type": kind, "sequence_number": index, **data}) + "\n\n").encode()
        for index, (kind, data) in enumerate(events)
    )


def _reply(model_api, *, tool=False, stream=False):
    if model_api == "responses":
        if not stream:
            return httpx.Response(200, json=_responses_body(tool=tool))
        content = _responses_events(tool=tool)
    else:
        delta = (
            {"tool_calls": [{"index": 0, "id": "call-probe", "type": "function",
                             "function": {"name": "probe", "arguments": ARGUMENTS}}]}
            if tool else {"content": PRIVATE}
        )
        usage = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
        finish = "tool_calls" if tool else "stop"
        if not stream:
            message = {"role": "assistant", "content": None if tool else PRIVATE}
            if tool:
                message["tool_calls"] = [{k: v for k, v in call.items() if k != "index"} for call in delta["tool_calls"]]
            return httpx.Response(200, json={
                "id": "completion-test", "object": "chat.completion", "created": 1, "model": "local-model",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage,
            })
        content = _chat_frame({"role": "assistant"}) + _chat_frame(delta) + _chat_frame({}, finish, usage)
        content += b"data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)


def _inject_http(monkeypatch, client):
    # Replace only the transport seam; both the model and OpenAI SDK stay real.
    monkeypatch.setattr(agent_factory, "OpenAIProvider", lambda **kwargs: OpenAIProvider(http_client=client, **kwargs))
    monkeypatch.setenv("MODEL_API_TEST_KEY", KEY)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-ambient-key")


def _input_text(body, model_api):
    items = body["input"] if model_api == "responses" else body["messages"]
    messages = []
    for item in items:
        if "role" not in item:
            continue
        content = item["content"]
        text = content if isinstance(content, str) else "".join(part["text"] for part in (content or []))
        messages.append((item["role"], text))
    return messages


@pytest.mark.parametrize("model_api", [None, "chat_completions"])
def test_build_model_accepts_chat_completions(monkeypatch, model_api):
    monkeypatch.delenv("AGENTKIT_MODEL_API", raising=False)
    if model_api is not None:
        monkeypatch.setenv("AGENTKIT_MODEL_API", model_api)
    monkeypatch.setenv("MODEL_API_TEST_KEY", KEY)
    provider = mock.Mock()
    model = mock.Mock()
    monkeypatch.setattr(agent_factory, "OpenAIProvider", provider)
    monkeypatch.setattr(agent_factory, "OpenAIChatModel", model)

    result = agent_factory.build_model(_spec())

    assert result is model.return_value
    provider.assert_called_once_with(base_url=BASE_URL, api_key=KEY)
    model.assert_called_once_with(
        "local-model", provider=provider.return_value,
        profile={"openai_chat_streaming_requires_finish_reason": True},
    )


@pytest.mark.parametrize("model_api", ["invalid-api", "", "chat", "Responses"])
def test_build_model_rejects_invalid_model_api_before_client_or_auth(monkeypatch, model_api):
    monkeypatch.setenv("AGENTKIT_MODEL_API", model_api)
    auth = mock.Mock()
    provider = mock.Mock()
    monkeypatch.setattr(agent_factory, "resolve_api_key", auth)
    monkeypatch.setattr(agent_factory, "OpenAIProvider", provider)

    with pytest.raises(agent_factory.AgentBuildError, match="AGENTKIT_MODEL_API"):
        agent_factory.build_model(_spec())
    auth.assert_not_called()
    provider.assert_not_called()


@pytest.mark.parametrize("selection", [None, "chat_completions", "responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_actual_wire_tool_roundtrip_preserves_history_events_and_usage(monkeypatch, selection, stream):
    monkeypatch.delenv("AGENTKIT_MODEL_API", raising=False)
    if selection is not None:
        monkeypatch.setenv("AGENTKIT_MODEL_API", selection)
    model_api = selection or "chat_completions"

    async def run():
        requests, invocations, events = [], [], []

        async def probe(value: str) -> str:
            if stream:
                assert events[-1].status == "in_progress"
            invocations.append(value)
            return "tool-success"

        async def observe(event):
            events.append(event)

        def handle(request):
            assert request.method == "POST"
            assert str(request.url) == BASE_URL + ("/responses" if model_api == "responses" else "/chat/completions")
            assert request.headers["authorization"] == "Bearer " + KEY
            body = json.loads(request.content)
            assert body["model"] == "local-model" and body["stream"] is stream
            requests.append(body)
            return _reply(model_api, tool=len(requests) == 1, stream=stream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            _inject_http(monkeypatch, client)
            model = agent_factory.build_model(_spec())
            assert isinstance(model, OpenAIResponsesModel if model_api == "responses" else OpenAIChatModel)
            agent = Agent(model, tools=[probe])
            request = RunRequest("current question", history=(
                ConversationTurn("system", "Client rules second."),
                ConversationTurn("user", "past question"), ConversationTurn("assistant", "past answer"),
                ConversationTurn("tool", "untrusted result"), ConversationTurn("user", ""),
            ), on_tool_event=observe if stream else None)
            async with agent:
                result = await agent_factory.run_agent(agent, request, instructions=_spec().instructions)
        assert result.text == PRIVATE
        assert result.usage == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
        assert invocations == ["hello"] and len(requests) == 2
        expected = [("system", "Baked rules first."), ("system", "Client rules second."),
                    ("user", "past question"), ("assistant", "past answer"), ("user", "current question")]
        for body in requests:
            assert _input_text(body, model_api)[:5] == expected
            assert "untrusted result" not in json.dumps(body)
            tool = body["tools"][0]
            function = tool if model_api == "responses" else tool["function"]
            assert tool["type"] == "function" and function["name"] == "probe"
            assert function["parameters"]["properties"]["value"]["type"] == "string"
        if model_api == "responses":
            assert "messages" not in requests[0]
            assert all(body["store"] is False for body in requests)
            call, output = requests[1]["input"][-2:]
            assert call["type"] == "function_call" and call["call_id"] == "call-probe"
            assert json.loads(call["arguments"]) == {"value": "hello"}
            assert output == {"type": "function_call_output", "call_id": "call-probe", "output": "tool-success"}
            assert "previous_response_id" not in requests[1]
        else:
            assert "input" not in requests[0]
            call, output = requests[1]["messages"][-2:]
            assert call["tool_calls"][0]["id"] == "call-probe"
            assert output == {"role": "tool", "tool_call_id": "call-probe", "content": "tool-success"}
        assert [event.status for event in events] == (["in_progress", "completed"] if stream else [])
        if events:
            assert events[0].tool_call_id == events[1].tool_call_id
            assert [event.tool_name for event in events] == ["probe", "probe"]

    asyncio.run(run())


@pytest.mark.parametrize("model_api", ["chat_completions", "responses"])
@pytest.mark.parametrize("turn_model_api", ["chat_completions", "responses", "invalid-api"])
def test_turn_env_cannot_override_startup_model_api(monkeypatch, model_api, turn_model_api):
    monkeypatch.setenv("AGENTKIT_MODEL_API", model_api)

    async def run():
        urls = []

        def handle(request):
            urls.append(str(request.url))
            return _reply(model_api)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            _inject_http(monkeypatch, client)
            async with agent_factory.build_runtime(_spec()) as runtime:
                # Selection is fixed by model construction, not by each turn.
                monkeypatch.setenv("AGENTKIT_MODEL_API", "invalid-after-startup")
                result = await runtime.run(RunRequest("hello", env={"AGENTKIT_MODEL_API": turn_model_api}))
        assert result.text == PRIVATE
        assert urls == [BASE_URL + ("/responses" if model_api == "responses" else "/chat/completions")]
        assert os.environ["AGENTKIT_MODEL_API"] == "invalid-after-startup"

    asyncio.run(run())


@pytest.mark.parametrize("selection", ["chat_completions", "responses"])
def test_orka_rejects_per_turn_model_api_override(monkeypatch, selection):
    monkeypatch.setenv("AGENTKIT_MODEL_API", selection)
    monkeypatch.setattr(agent_factory, "build_runtime", mock.Mock(side_effect=AssertionError("must not run")))
    app = create_orka_app(_spec(), agent_factory, auth_token="test-token")
    with TestClient(app) as client:
        response = client.post("/v1/turns", headers={"authorization": "Bearer test-token"}, json={
            "version": ORKA_HARNESS_VERSION, "namespace": "default", "taskName": "task",
            "sessionName": "session", "runtimeSessionID": "runtime", "turnID": "turn",
            "correlationID": "correlation", "deadline": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
            "authIdentity": {"subject": "test"}, "toolExecutionMode": "observed", "metadata": {},
            "input": {"prompt": "hello", "contextRefs": [],
                      "env": [{"name": "AGENTKIT_MODEL_API", "value": "responses"}]},
        })
    assert response.status_code == 400
    assert "AGENTKIT_MODEL_API" in response.text and "reserved" in response.text
    agent_factory.build_runtime.assert_not_called()


def test_responses_upstream_does_not_change_chat_serving_contract(monkeypatch):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")
    requests = []

    def handle(request):
        assert str(request.url) == BASE_URL + "/responses"
        requests.append(json.loads(request.content))
        return _reply("responses")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    _inject_http(monkeypatch, http)
    try:
        with TestClient(create_app(_spec(), agent_factory)) as client:
            response = client.post("/v1/chat/completions", json={
                "model": "local-model", "messages": [{"role": "user", "content": "hello"}],
            })
            assert response.status_code == 200
            assert response.json()["object"] == "chat.completion"
            assert response.json()["choices"][0]["message"]["content"] == PRIVATE
            streamed = client.post("/v1/chat/completions", json={
                "model": "local-model", "messages": [{"role": "user", "content": "hello"}], "stream": True,
            })
            assert streamed.status_code == 400
        assert len(requests) == 1 and requests[0]["stream"] is False
    finally:
        asyncio.run(http.aclose())


@pytest.mark.parametrize("model_api", ["chat_completions", "responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_model_http_error_remains_secret_safe(monkeypatch, model_api, stream):
    monkeypatch.setenv("AGENTKIT_MODEL_API", model_api)

    async def run():
        requests = []

        async def observe(event):
            raise AssertionError("failed request must not execute tools")

        def handle(request):
            requests.append(request)
            return httpx.Response(401, json={"error": {"message": PRIVATE + KEY, "type": "authentication_error"}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            _inject_http(monkeypatch, client)
            async with agent_factory.build_runtime(_spec()) as runtime:
                with pytest.raises(AgentRunError) as caught:
                    await runtime.run(RunRequest("hello", on_tool_event=observe if stream else None))
        assert caught.value.code == "ModelAuthRejected" and caught.value.status == 503
        assert PRIVATE not in str(caught.value) and KEY not in str(caught.value)
        assert len(requests) == 1
        assert json.loads(requests[0].content)["stream"] is stream

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("terminal", [None, "response.failed", "response.incomplete", "error"])
@pytest.mark.parametrize("tool", [False, True])
@pytest.mark.parametrize("after_tool", [False, True])
def test_responses_unfinished_output_cannot_commit_text_or_execute_tools(monkeypatch, stream, terminal, tool, after_tool):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")

    async def run():
        requests, invocations, events = [], [], []

        async def probe(value: str) -> str:
            invocations.append(value)
            return "tool-success"

        async def observe(event):
            events.append(event)

        def handle(request):
            requests.append(request)
            if after_tool and len(requests) == 1:
                return _reply("responses", tool=True, stream=stream)
            if stream:
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                      content=_responses_events(tool=tool, terminal=terminal))
            status = terminal.removeprefix("response.") if terminal and terminal != "error" else "in_progress"
            body = _responses_body(tool=tool, status=status)
            if terminal == "error":
                body["status"] = "failed"
                body["error"] = {"code": "server_error", "message": PRIVATE + KEY}
            return httpx.Response(200, json=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            _inject_http(monkeypatch, client)
            agent = Agent(agent_factory.build_model(_spec()), tools=[probe])
            async with agent:
                with pytest.raises(AgentRunError) as caught:
                    await agent_factory.run_agent(agent, RunRequest("hello", on_tool_event=observe if stream else None))
        assert PRIVATE not in str(caught.value)
        assert KEY not in str(caught.value)
        assert len(requests) == (2 if after_tool else 1)
        assert invocations == (["hello"] if after_tool else [])
        assert [event.status for event in events] == (["in_progress", "completed"] if stream and after_tool else [])

    asyncio.run(run())


def test_cancelling_responses_stream_preserves_cancellation_and_closes_transport(monkeypatch):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")

    async def run():
        started, closed = asyncio.Event(), asyncio.Event()

        class PendingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield _responses_events(terminal=None)
                started.set()
                await asyncio.Future()

            async def aclose(self):
                closed.set()

        async def observe(event):
            pass

        def handle(request):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=PendingStream())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            _inject_http(monkeypatch, client)
            async with agent_factory.build_runtime(_spec()) as runtime:
                task = asyncio.create_task(runtime.run(RunRequest("hello", on_tool_event=observe)))
                await asyncio.wait_for(started.wait(), 3)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 3)
        assert closed.is_set()

    asyncio.run(run())


def _item_status_reply(*, tool, source, item_status):
    """Keep top-level completion valid while changing raw output-item evidence."""
    if source == "json":
        body = _responses_body(tool=tool)
        body["output"][0]["status"] = item_status
        return httpx.Response(200, json=body)

    events = [json.loads(frame.split("\ndata: ", 1)[1])
              for frame in _responses_events(tool=tool).decode().strip().split("\n\n")]
    for event in events:
        if event["type"] == "response.output_item.added":
            if source == "added":
                event["item"]["status"] = item_status
            elif source == "unterminated_omitted":
                event["item"].pop("status", None)
        if event["type"] == "response.output_item.done" and source == "item_done":
            event["item"]["status"] = item_status
        if event["type"] == "response.completed":
            if source == "terminal":
                event["response"]["output"][0]["status"] = item_status
            elif source in {"unterminated", "unterminated_omitted", "done_only"}:
                event["response"]["output"] = []
    if source in {"unterminated", "unterminated_omitted", "terminal_only"}:
        events = [event for event in events if event["type"] != "response.output_item.done"]
    content = b"".join(
        ("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n").encode()
        for event in events
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)


@pytest.mark.parametrize("source,item_status", [
    ("json", "in_progress"), ("json", "incomplete"),
    ("item_done", "in_progress"), ("item_done", "incomplete"),
    ("terminal", "in_progress"), ("terminal", "incomplete"),
    ("added", "incomplete"), ("unterminated", "in_progress"),
    ("unterminated_omitted", None),
])
@pytest.mark.parametrize("tool", [False, True])
@pytest.mark.parametrize("after_tool", [False, True])
def test_completed_response_with_unfinished_raw_item_fails_before_effects(
    monkeypatch, source, item_status, tool, after_tool,
):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")
    stream = source != "json"

    async def run():
        requests, invocations, events = [], [], []

        async def probe(value: str) -> str:
            invocations.append(value)
            return "tool-success"

        async def observe(event):
            events.append(event)

        def handle(request):
            requests.append(request)
            if after_tool and len(requests) == 1:
                return _reply("responses", tool=True, stream=stream)
            if len(requests) == (2 if after_tool else 1):
                return _item_status_reply(tool=tool, source=source, item_status=item_status)
            # Negative control must finish promptly if partial tools are admitted.
            return _reply("responses", stream=stream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            _inject_http(monkeypatch, client)
            agent = Agent(agent_factory.build_model(_spec()), tools=[probe])
            async with agent:
                with pytest.raises(AgentRunError) as caught:
                    await agent_factory.run_agent(agent, RunRequest("hello", on_tool_event=observe if stream else None))
        assert PRIVATE not in str(caught.value) and KEY not in str(caught.value)
        assert len(requests) == (2 if after_tool else 1)
        assert invocations == (["hello"] if after_tool else [])
        assert [event.status for event in events] == (["in_progress", "completed"] if stream and after_tool else [])

    asyncio.run(run())


@pytest.mark.parametrize("source", ["done_only", "terminal_only"])
@pytest.mark.parametrize("tool", [False, True])
def test_completed_raw_item_can_be_confirmed_by_done_event_or_terminal_output(monkeypatch, source, tool):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")

    async def run():
        requests, invocations = [], []

        async def probe(value: str) -> str:
            invocations.append(value)
            return "tool-success"

        async def observe(event):
            pass

        def handle(request):
            requests.append(request)
            if len(requests) == 1:
                return _item_status_reply(tool=tool, source=source, item_status="completed")
            return _reply("responses", stream=True)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            _inject_http(monkeypatch, client)
            agent = Agent(agent_factory.build_model(_spec()), tools=[probe])
            async with agent:
                result = await agent_factory.run_agent(agent, RunRequest("hello", on_tool_event=observe))
        assert result.text == PRIVATE
        assert len(requests) == (2 if tool else 1)
        assert invocations == (["hello"] if tool else [])

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("item_type", ["function_call", "reasoning"])
def test_responses_allows_omitted_optional_raw_item_status(monkeypatch, stream, item_type):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")
    tool = item_type == "function_call"

    async def run():
        requests, invocations = [], []

        async def probe(value: str) -> str:
            invocations.append(value)
            return "tool-success"

        async def observe(event):
            pass

        def omit_status(body):
            if tool:
                body["output"][0].pop("status", None)
            else:
                body["output"].insert(0, {"type": "reasoning", "id": "rs-test", "summary": []})
            return body

        def handle(request):
            requests.append(request)
            if len(requests) > 1:
                return _reply("responses", stream=stream)
            if not stream:
                return httpx.Response(200, json=omit_status(_responses_body(tool=tool)))
            events = [json.loads(frame.split("\ndata: ", 1)[1])
                      for frame in _responses_events(tool=tool).decode().strip().split("\n\n")]
            for event in events:
                if tool and event["type"] in {"response.output_item.added", "response.output_item.done"}:
                    event["item"].pop("status", None)
                if event["type"] == "response.completed":
                    omit_status(event["response"])
            content = b"".join(
                ("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n").encode()
                for event in events
            )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            _inject_http(monkeypatch, client)
            agent = Agent(agent_factory.build_model(_spec()), tools=[probe])
            async with agent:
                result = await agent_factory.run_agent(agent, RunRequest("hello", on_tool_event=observe if stream else None))
        assert result.text == PRIVATE
        assert len(requests) == (2 if tool else 1)
        assert invocations == (["hello"] if tool else [])

    asyncio.run(run())
