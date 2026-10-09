from __future__ import annotations

import asyncio
import json
from unittest import mock

import httpx
from agent_framework import ContextProvider, Message, tool
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import ConversationTurn, RunRequest


def _spec(*, with_tool=False):
    return AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "package-agent"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": "https://provider.example/v1",
            "name": "test-model",
        },
        "instructions": "Follow the user's instructions.",
        "tools": [{"name": "probe", "command": ["unused-fixture"]}] if with_tool else [],
        "expose": {"openai": True, "port": 8080},
    })


def _completion(message):
    return httpx.Response(200, json={
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def _reject_speaker_names(body):
    # A chat-to-Responses gateway cannot translate named speakers. Package
    # metadata must not turn into a speaker identity on subsequent model calls.
    assert all("name" not in message for message in body["messages"])


def test_continuation_preserves_history_without_promoting_package_name_to_speaker():
    async def exercise():
        requests = []

        def respond(request):
            body = json.loads(request.content)
            _reject_speaker_names(body)
            requests.append(body)
            return _completion({"role": "assistant", "content": "remembered-é-世界"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            sdk = AsyncOpenAI(base_url="https://provider.example/v1", api_key="test-key", http_client=http,
                              max_retries=0)
            client = agent_factory.OpenAIChatCompletionClient(model="test-model", async_client=sdk)
            with mock.patch.object(agent_factory, "build_client", return_value=client):
                async with agent_factory.MAFRuntime(_spec()) as runtime:
                    first = await runtime.run(RunRequest("Remember é-世界.", session_id="conversation"))
                    second = await runtime.run(RunRequest(
                        "Recall it.", session_id="conversation",
                        history=(ConversationTurn("user", "Remember é-世界."),
                                 ConversationTurn("assistant", first.text)),
                    ))
        assert second.text == first.text == "remembered-é-世界"
        assert len(requests) == 2
        assert [(message["role"], message["content"]) for message in requests[1]["messages"]] == [
            ("system", "Follow the user's instructions."),
            ("user", "Remember é-世界."),
            ("assistant", "remembered-é-世界"),
            ("user", "Recall it."),
        ]

    asyncio.run(exercise())


def test_context_providers_see_only_the_new_turn_as_input():
    class RecordingProvider(ContextProvider):
        def __init__(self):
            super().__init__("recorder")
            self.inputs = []

        async def after_run(self, *, agent, session, context, state):  # noqa: ANN001
            # Memory providers persist exactly these messages after each run.
            self.inputs.append([(message.role, message.text) for message in context.input_messages])

    async def exercise():
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return _completion({"role": "assistant", "content": f"answer-{len(requests)}"})

        recorder = RecordingProvider()
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            sdk = AsyncOpenAI(base_url="https://provider.example/v1", api_key="test-key", http_client=http,
                              max_retries=0)
            client = agent_factory.OpenAIChatCompletionClient(model="test-model", async_client=sdk)
            async with agent_factory.build_agent(_spec(), client=client, context_providers=[recorder]) as agent:
                session = agent.create_session()
                await agent_factory.run_agent(agent, RunRequest("first"), session=session)
                await agent_factory.run_agent(agent, RunRequest(
                    "second", history=(ConversationTurn("user", "first"), ConversationTurn("assistant", "answer-1")),
                ), session=session)

        assert recorder.inputs == [[("user", "first")], [("user", "second")]]
        assert [(message["role"], message["content"]) for message in requests[1]["messages"]] == [
            ("system", "Follow the user's instructions."),
            ("user", "first"),
            ("assistant", "answer-1"),
            ("user", "second"),
        ]

    asyncio.run(exercise())


def test_tool_loop_preserves_function_name_arguments_and_correlation_without_speaker_name():
    async def exercise():
        requests = []
        calls = []
        events = []
        payload = {"text": "héllo 世界 🌍", "nested": [42, True, None]}

        async def probe(payload: dict):
            calls.append(payload)
            return json.dumps(payload, ensure_ascii=False)

        async def observe(event):
            events.append(event)

        def respond(request):
            body = json.loads(request.content)
            _reject_speaker_names(body)
            requests.append(body)
            if len(requests) == 1:
                assert body["tools"][0]["function"]["name"] == "probe"
                return _completion({"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call-fixture", "type": "function",
                    "function": {"name": "probe", "arguments": json.dumps({"payload": payload})},
                }]})
            assistant, result = body["messages"][-2:]
            assert assistant["role"] == "assistant"
            assert assistant["tool_calls"][0]["function"]["name"] == "probe"
            assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"payload": payload}
            assert result["role"] == "tool" and result["tool_call_id"] == "call-fixture"
            assert json.loads(result["content"]) == payload
            return _completion({"role": "assistant", "content": "done"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            sdk = AsyncOpenAI(base_url="https://provider.example/v1", api_key="test-key", http_client=http,
                              max_retries=0)
            client = agent_factory.OpenAIChatCompletionClient(model="test-model", async_client=sdk)
            with (
                mock.patch.object(agent_factory, "build_client", return_value=client),
                mock.patch.object(agent_factory, "build_tool", return_value=tool(probe, name="probe")),
            ):
                async with agent_factory.MAFRuntime(_spec(with_tool=True)) as runtime:
                    result = await runtime.run(RunRequest("Call probe.", session_id="tools", on_tool_event=observe))
        assert result.text == "done" and len(requests) == 2 and calls == [payload]
        assert [event.status for event in events] == ["in_progress", "completed"]
        assert events[0].tool_call_id == events[1].tool_call_id

    asyncio.run(exercise())


def test_storing_responses_client_does_not_chain_stored_conversation():
    async def exercise():
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": f"resp_{len(requests)}", "object": "response", "created_at": 1, "model": "test-model",
                "status": "completed", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
                "output": [{"type": "message", "id": f"msg_{len(requests)}", "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "remembered", "annotations": []}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}},
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            sdk = AsyncOpenAI(base_url="https://provider.example/v1", api_key="test-key", http_client=http,
                              max_retries=0)
            # The Foundry client inherits this Responses client's store-by-default behavior.
            client = OpenAIChatClient(model="test-model", async_client=sdk)
            assert client.STORES_BY_DEFAULT
            async with agent_factory.build_agent(_spec(), client=client) as agent:
                session = agent.create_session()
                await agent.run([Message(role="user", contents=["first"])], session=session)
                await agent.run([
                    Message(role="user", contents=["first"]),
                    Message(role="assistant", contents=["remembered"]),
                    Message(role="user", contents=["second"]),
                ], session=session)

        # AgentKit sends the full conversation on every run; chaining the stored
        # response as well would give the model that history twice.
        assert [request.get("store") for request in requests] == [False, False]
        assert all("previous_response_id" not in request and "conversation" not in request for request in requests)

    asyncio.run(exercise())


def test_outbound_projection_preserves_original_internal_message_metadata():
    async def exercise():
        original = Message(role="assistant", contents=["remembered-é-世界"], author_name="package-agent")
        before = original.to_dict()
        requests = []

        def respond(request):
            body = json.loads(request.content)
            _reject_speaker_names(body)
            requests.append(body)
            return _completion({"role": "assistant", "content": "done"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            sdk = AsyncOpenAI(base_url="https://provider.example/v1", api_key="test-key", http_client=http,
                              max_retries=0)
            client = agent_factory.OpenAIChatCompletionClient(model="test-model", async_client=sdk)
            async with agent_factory.build_agent(_spec(), client=client) as agent:
                result = await agent.run([original, Message(role="user", contents=["Continue."])])
                assert agent.name == "package-agent"
        assert result.text == "done" and len(requests) == 1
        assert original.to_dict() == before
        assert any(message["role"] == "assistant" and message["content"] == "remembered-é-世界"
                   for message in requests[0]["messages"])

    asyncio.run(exercise())
