"""Startup model API selection across non-brokered MAF auth paths."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest


def _spec(*, workload_identity: bool = False) -> AgentSpec:
    model = {
        "provider": "openai-compatible",
        "baseURL": "https://example.services.ai.azure.com/api/projects/test/openai/v1",
        "name": "test-model",
    }
    if workload_identity:
        model["auth"] = {"type": "workload-identity-token", "audience": "https://ai.azure.com/.default"}
    return AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "model-api-test"},
        "model": model,
        "instructions": "Be helpful.",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })


@pytest.fixture(params=["api-key", "model-token", "generic-token", "foundry"])
def auth_path(monkeypatch, request):
    for name in (
        "AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    if request.param == "model-token":
        monkeypatch.setenv("AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN", "test-token")
    elif request.param == "generic-token":
        monkeypatch.setenv("AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND", "/unused/token-command")
    return request.param


@pytest.fixture
def model_dependencies(monkeypatch):
    dependencies = SimpleNamespace(
        chat=mock.Mock(),
        foundry=mock.Mock(),
        credential=mock.Mock(),
        api_key=mock.Mock(return_value="test-key"),
        token_provider=mock.Mock(),
    )
    monkeypatch.setattr(agent_factory, "OpenAIChatCompletionClient", dependencies.chat)
    monkeypatch.setattr("agent_framework.foundry.FoundryChatClient", dependencies.foundry)
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", dependencies.credential)
    monkeypatch.setattr(agent_factory, "resolve_api_key", dependencies.api_key)
    monkeypatch.setattr(agent_factory, "_model_workload_api_key_provider", dependencies.token_provider)
    return dependencies


@pytest.mark.parametrize("model_api", [None, "chat_completions", "responses", "invalid-api"])
def test_build_client_honors_auth_path_api_capability(monkeypatch, model_api, auth_path, model_dependencies):
    monkeypatch.delenv("AGENTKIT_MODEL_API", raising=False)
    if model_api is not None:
        monkeypatch.setenv("AGENTKIT_MODEL_API", model_api)
    spec = _spec(workload_identity=auth_path != "api-key")
    supported = "responses" if auth_path == "foundry" else "chat_completions"
    effective = model_api or "chat_completions"
    deps = model_dependencies
    if effective != supported:
        with pytest.raises(agent_factory.AgentBuildError, match="AGENTKIT_MODEL_API"):
            agent_factory.build_client(spec)
        for dependency in vars(deps).values():
            dependency.assert_not_called()
        return

    client = agent_factory.build_client(spec)
    if auth_path == "foundry":
        assert client is deps.foundry.return_value
        deps.credential.assert_called_once_with()
        deps.foundry.assert_called_once_with(
            project_endpoint="https://example.services.ai.azure.com/api/projects/test",
            model="test-model", credential=deps.credential.return_value,
        )
        deps.chat.assert_not_called()
    else:
        assert client is deps.chat.return_value
        key = deps.api_key.return_value if auth_path == "api-key" else deps.token_provider.return_value
        deps.chat.assert_called_once_with(model="test-model", base_url=spec.model.base_url, api_key=key)
        deps.credential.assert_not_called()
        deps.foundry.assert_not_called()


@pytest.mark.parametrize("model_api", ["chat_completions", "invalid-api"])
def test_runtime_fallback_rejects_model_api_before_credential(monkeypatch, model_api, model_dependencies):
    monkeypatch.setenv("AGENTKIT_MODEL_API", model_api)
    for name in (
        "AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)

    async def start():
        async with agent_factory.build_runtime(_spec(workload_identity=True)):
            pytest.fail("unsupported model API must fail during startup")

    with pytest.raises(agent_factory.AgentBuildError, match="AGENTKIT_MODEL_API"):
        asyncio.run(start())

    for dependency in vars(model_dependencies).values():
        dependency.assert_not_called()


@pytest.mark.parametrize("turn_model_api", ["responses", "invalid-api"])
def test_turn_env_cannot_override_startup_model_api(monkeypatch, turn_model_api):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "chat_completions")
    client = mock.Mock()
    agent = mock.MagicMock()
    agent.__aenter__ = mock.AsyncMock(return_value=agent)
    agent.__aexit__ = mock.AsyncMock(return_value=False)
    agent.run = mock.AsyncMock(return_value=SimpleNamespace(text="offline reply"))
    monkeypatch.setattr(agent_factory, "OpenAIChatCompletionClient", client)
    constructor = mock.Mock(return_value=agent)
    monkeypatch.setattr(agent_factory, "Agent", constructor)

    async def run():
        async with agent_factory.build_runtime(_spec()) as runtime:
            return await runtime.run(RunRequest(prompt="hello", env={"AGENTKIT_MODEL_API": turn_model_api}))

    result = asyncio.run(run())

    assert result.text == "offline reply"
    assert os.environ["AGENTKIT_MODEL_API"] == "chat_completions"
    client.assert_called_once()
    constructor.assert_called_once()
    assert constructor.call_args.kwargs["client"] is client.return_value


@pytest.mark.parametrize("path", ["chat-default", "chat-explicit", "foundry-responses"])
def test_real_model_client_uses_selected_wire_api(monkeypatch, path):
    import functools
    import json
    import httpx
    from openai import AsyncOpenAI
    from agent_framework import Content, Message
    from agent_framework.openai import OpenAIChatCompletionClient

    for name in (
        "AGENTKIT_MODEL_API", "AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN", "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    foundry = path == "foundry-responses"
    if foundry:
        monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")
    elif path == "chat-explicit":
        monkeypatch.setenv("AGENTKIT_MODEL_API", "chat_completions")
    requests = []
    def handler(request):
        requests.append((str(request.url), json.loads(request.content)))
        if foundry:
            body = {
                "id": "resp_test", "object": "response", "created_at": 1,
                "model": "test-model", "status": "completed",
                "output": [{"type": "message", "id": "msg_test", "status": "completed", "role": "assistant",
                            "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        else:
            body = {
                "id": "chatcmpl_test", "object": "chat.completion", "created": 1, "model": "test-model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Done."}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return httpx.Response(200, json=body)

    async def run():
        spec = _spec(workload_identity=foundry)
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        openai = AsyncOpenAI(api_key="test-key", base_url=spec.model.base_url, http_client=http_client)
        if foundry:
            # Keep the real FoundryChatClient; only its Azure discovery/credential boundary is local.
            class Project:
                def __init__(self, **kwargs):
                    pass
                def get_openai_client(self, **kwargs):
                    return openai
            monkeypatch.setattr("agent_framework_foundry._chat_client.AIProjectClient", Project)
            # Older supported SDKs do not allocate a separate feature-usage client.
            monkeypatch.setattr(
                "agent_framework_foundry._chat_client.create_foundry_feature_usage_http_client",
                lambda: http_client,
                raising=False,
            )
            monkeypatch.setattr("azure.identity.DefaultAzureCredential", lambda: object())
        else:
            monkeypatch.setattr(agent_factory, "OpenAIChatCompletionClient", functools.partial(
                OpenAIChatCompletionClient, async_client=openai,
            ))
        try:
            client = agent_factory.build_client(spec)
            response = await client.get_response([Message(role="user", contents=[Content.from_text("Hello.")])])
            assert response.text == "Done."
        finally:
            await openai.close()
    asyncio.run(run())
    assert len(requests) == 1
    url, payload = requests[0]
    if foundry:
        assert url.endswith("/responses")
        assert "input" in payload and "messages" not in payload
    else:
        assert url.endswith("/chat/completions")
        assert "messages" in payload and "input" not in payload


@pytest.mark.parametrize("workload_identity,selection", [
    (False, "responses"), (False, "invalid-api"),
    (True, None), (True, "chat_completions"), (True, "invalid-api"),
])
def test_runtime_rejects_api_before_context_and_tool_startup(monkeypatch, workload_identity, selection):
    for name in (
        "AGENTKIT_MODEL_API", "AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN", "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    if selection is not None:
        monkeypatch.setenv("AGENTKIT_MODEL_API", selection)
    runtime = agent_factory.MAFRuntime(_spec(workload_identity=workload_identity))
    contexts = mock.AsyncMock(side_effect=AssertionError("context initialization must not run"))
    agent = mock.Mock(side_effect=AssertionError("agent/tool construction must not run"))
    monkeypatch.setattr(runtime, "_build_context_providers", contexts)
    monkeypatch.setattr(agent_factory, "build_agent", agent)
    async def run():
        async with runtime:
            pytest.fail("invalid model API must fail startup")
    with pytest.raises(agent_factory.AgentBuildError, match="AGENTKIT_MODEL_API"):
        asyncio.run(run())
    contexts.assert_not_called()
    agent.assert_not_called()
