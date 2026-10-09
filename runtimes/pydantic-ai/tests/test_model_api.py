"""Startup model API selection for the non-brokered pydantic-ai adapter."""

from __future__ import annotations

import asyncio
import os
from unittest import mock

import pytest
from pydantic_ai.models.test import TestModel

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
    provider = mock.Mock()
    model = mock.Mock()
    monkeypatch.setattr(agent_factory, "OpenAIProvider", provider)
    monkeypatch.setattr(agent_factory, "OpenAIChatModel", model)

    result = agent_factory.build_model(_spec())

    assert result is model.return_value
    provider.assert_called_once_with(base_url="https://example.test/v1", api_key="not-needed")
    model.assert_called_once_with(
        "test-model",
        provider=provider.return_value,
        profile={"openai_chat_streaming_requires_finish_reason": True},
    )


@pytest.mark.parametrize("model_api", ["responses", "invalid-api"])
def test_build_model_rejects_model_api_before_client_or_auth(monkeypatch, model_api):
    monkeypatch.setenv("AGENTKIT_MODEL_API", model_api)
    auth = mock.Mock()
    provider = mock.Mock()
    model = mock.Mock()
    monkeypatch.setattr(agent_factory, "resolve_api_key", auth)
    monkeypatch.setattr(agent_factory, "OpenAIProvider", provider)
    monkeypatch.setattr(agent_factory, "OpenAIChatModel", model)

    with pytest.raises(agent_factory.AgentBuildError, match="AGENTKIT_MODEL_API") as exc:
        agent_factory.build_model(_spec())

    if model_api == "responses":
        assert "pydantic-ai" in str(exc.value)
    auth.assert_not_called()
    provider.assert_not_called()
    model.assert_not_called()


@pytest.mark.parametrize("turn_model_api", ["responses", "invalid-api"])
def test_turn_env_cannot_override_startup_model_api(monkeypatch, turn_model_api):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "chat_completions")
    provider = mock.Mock()
    model = mock.Mock(return_value=TestModel(custom_output_text="offline reply"))
    monkeypatch.setattr(agent_factory, "OpenAIProvider", provider)
    monkeypatch.setattr(agent_factory, "OpenAIChatModel", model)

    async def run():
        async with agent_factory.build_runtime(_spec()) as runtime:
            return await runtime.run(RunRequest(prompt="hello", env={"AGENTKIT_MODEL_API": turn_model_api}))

    result = asyncio.run(run())

    assert result.text == "offline reply"
    assert os.environ["AGENTKIT_MODEL_API"] == "chat_completions"
    provider.assert_called_once()
    model.assert_called_once()
