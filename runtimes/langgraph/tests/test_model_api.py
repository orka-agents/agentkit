"""Startup model API selection for the non-brokered LangGraph adapter."""

from __future__ import annotations

import asyncio
import json
import os
from functools import partial
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest


def _spec() -> AgentSpec:
    return AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "model-api-test"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": "https://example.test/v1",
            "name": "test-model",
        },
        "instructions": "Be helpful.",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })


@pytest.mark.parametrize("model_api", [None, "chat_completions"])
def test_build_model_accepts_chat_completions(monkeypatch, model_api):
    monkeypatch.delenv("AGENTKIT_MODEL_API", raising=False)
    if model_api is not None:
        monkeypatch.setenv("AGENTKIT_MODEL_API", model_api)
    model = mock.Mock()
    monkeypatch.setattr(agent_factory, "ChatOpenAI", model)

    result = agent_factory.build_model(_spec())

    assert result is model.return_value
    model.assert_called_once_with(
        model="test-model",
        base_url="https://example.test/v1",
        api_key="not-needed",
        use_responses_api=False,
    )


@pytest.mark.parametrize("model_api", [None, "chat_completions"])
def test_build_model_uses_chat_wire_format_despite_responses_output_version(monkeypatch, model_api):
    monkeypatch.delenv("AGENTKIT_MODEL_API", raising=False)
    if model_api is not None:
        monkeypatch.setenv("AGENTKIT_MODEL_API", model_api)
    monkeypatch.setenv("LC_OUTPUT_VERSION", "responses/v1")
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={
            "id": "chatcmpl-offline",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "offline reply"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    async def exercise():
        with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as sync_client:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False) as async_client:
                # Inject only transports; retain real SDK routing and serialization.
                monkeypatch.setattr(agent_factory, "ChatOpenAI", partial(
                    ChatOpenAI, http_client=sync_client, http_async_client=async_client, max_retries=0,
                ))
                model = agent_factory.build_model(_spec())
                assert isinstance(model, ChatOpenAI)
                assert model.output_version == "responses/v1"
                return await model.ainvoke([("system", "Be helpful."), ("human", "hello")])

    result = asyncio.run(exercise())

    assert result.content == "offline reply"
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/v1/chat/completions"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "test-model"
    assert payload["messages"] == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "hello"},
    ]
    assert "input" not in payload


@pytest.mark.parametrize("model_api", ["responses", "invalid-api"])
def test_build_model_rejects_model_api_before_client_or_auth(monkeypatch, model_api):
    monkeypatch.setenv("AGENTKIT_MODEL_API", model_api)
    auth = mock.Mock()
    model = mock.Mock()
    monkeypatch.setattr(agent_factory, "resolve_api_key", auth)
    monkeypatch.setattr(agent_factory, "ChatOpenAI", model)

    with pytest.raises(agent_factory.AgentBuildError, match="AGENTKIT_MODEL_API") as exc:
        agent_factory.build_model(_spec())

    if model_api == "responses":
        assert "LangGraph" in str(exc.value)
    auth.assert_not_called()
    model.assert_not_called()


@pytest.mark.parametrize("turn_model_api", ["responses", "invalid-api"])
def test_turn_env_cannot_override_startup_model_api(monkeypatch, turn_model_api):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "chat_completions")
    model = mock.Mock()
    graph = SimpleNamespace(ainvoke=mock.AsyncMock(return_value={"messages": [AIMessage(content="offline reply")]}))
    monkeypatch.setattr(agent_factory, "ChatOpenAI", model)
    create_agent = mock.Mock(return_value=graph)
    monkeypatch.setattr(agent_factory, "create_agent", create_agent)

    async def run():
        async with agent_factory.build_runtime(_spec()) as runtime:
            return await runtime.run(RunRequest(prompt="hello", env={"AGENTKIT_MODEL_API": turn_model_api}))

    result = asyncio.run(run())

    assert result.text == "offline reply"
    assert os.environ["AGENTKIT_MODEL_API"] == "chat_completions"
    model.assert_called_once()
    create_agent.assert_called_once()
    assert create_agent.call_args.kwargs["model"] is model.return_value
