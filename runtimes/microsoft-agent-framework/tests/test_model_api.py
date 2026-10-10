"""Model API selection and auto-runtime wiring across MAF auth paths."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest


def _spec(*, workload_identity: bool = False, api_key: bool = False) -> AgentSpec:
    model = {
        "provider": "openai-compatible",
        "baseURL": "https://example.services.ai.azure.com/api/projects/test/openai/v1",
        "name": "test-model",
    }
    if api_key:
        model["apiKeyEnv"] = "OPENAI_API_KEY"
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


@pytest.fixture(params=["no-auth", "api-key", "model-token", "generic-token", "generic-token-command", "foundry"])
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
        monkeypatch.setenv("AGENTKIT_WORKLOAD_IDENTITY_TOKEN", "test-token")
    elif request.param == "generic-token-command":
        monkeypatch.setenv("AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND", "/unused/token-command")
    return request.param


@pytest.fixture
def model_dependencies(monkeypatch):
    dependencies = SimpleNamespace(
        chat=mock.Mock(),
        responses=mock.Mock(),
        foundry=mock.Mock(),
        credential=mock.Mock(),
        api_key=mock.Mock(return_value="test-key"),
        token_provider=mock.Mock(),
    )
    monkeypatch.setattr(agent_factory, "OpenAIChatCompletionClient", dependencies.chat)
    monkeypatch.setattr(agent_factory, "OpenAIChatClient", dependencies.responses)
    monkeypatch.setattr("agent_framework.foundry.FoundryChatClient", dependencies.foundry)
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", dependencies.credential)
    monkeypatch.setattr(agent_factory, "resolve_api_key", dependencies.api_key)
    monkeypatch.setattr(agent_factory, "_model_workload_api_key_provider", dependencies.token_provider)
    return dependencies


@pytest.mark.parametrize("model_api", [None, "chat_completions", "responses", "auto", "invalid-api"])
def test_build_client_honors_auth_path_api_capability(monkeypatch, model_api, auth_path, model_dependencies):
    monkeypatch.delenv("AGENTKIT_MODEL_API", raising=False)
    if model_api is not None:
        monkeypatch.setenv("AGENTKIT_MODEL_API", model_api)
    spec = _spec(workload_identity=auth_path not in {"no-auth", "api-key"}, api_key=auth_path == "api-key")
    effective = "responses" if model_api == "auto" else model_api or "chat_completions"
    deps = model_dependencies
    if effective == "invalid-api":
        with pytest.raises(agent_factory.AgentBuildError, match="AGENTKIT_MODEL_API"):
            agent_factory.build_client(spec)
        for dependency in vars(deps).values():
            dependency.assert_not_called()
        return

    client = agent_factory.build_client(spec)
    selected = deps.responses if effective == "responses" else deps.chat
    unselected = deps.chat if effective == "responses" else deps.responses
    unselected.assert_not_called()
    if auth_path == "foundry":
        deps.credential.assert_called_once_with()
        deps.foundry.assert_called_once_with(
            project_endpoint="https://example.services.ai.azure.com/api/projects/test",
            model="test-model", credential=deps.credential.return_value,
        )
        if effective == "responses":
            assert client is deps.foundry.return_value
            selected.assert_not_called()
        else:
            assert client is selected.return_value
            selected.assert_called_once_with(model="test-model", async_client=deps.foundry.return_value.client)
            assert client.project_client is deps.foundry.return_value.project_client
    else:
        assert client is selected.return_value
        key = deps.api_key.return_value if auth_path in {"no-auth", "api-key"} else deps.token_provider.return_value
        selected.assert_called_once_with(model="test-model", base_url=spec.model.base_url, api_key=key)
        deps.credential.assert_not_called()
        deps.foundry.assert_not_called()


@pytest.mark.parametrize("model_api", ["invalid-api"])
def test_runtime_fallback_rejects_invalid_model_api_before_credential(monkeypatch, model_api, model_dependencies):
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


@pytest.mark.parametrize("startup_model_api", ["chat_completions", "responses"])
@pytest.mark.parametrize("turn_model_api", ["chat_completions", "responses", "invalid-api"])
def test_turn_env_cannot_override_startup_model_api(monkeypatch, startup_model_api, turn_model_api):
    monkeypatch.setenv("AGENTKIT_MODEL_API", startup_model_api)
    client = mock.Mock()
    agent = mock.MagicMock()
    agent.__aenter__ = mock.AsyncMock(return_value=agent)
    agent.__aexit__ = mock.AsyncMock(return_value=False)
    agent.run = mock.AsyncMock(return_value=SimpleNamespace(text="offline reply"))
    client_name = "OpenAIChatClient" if startup_model_api == "responses" else "OpenAIChatCompletionClient"
    monkeypatch.setattr(agent_factory, client_name, client)
    constructor = mock.Mock(return_value=agent)
    monkeypatch.setattr(agent_factory, "Agent", constructor)

    async def run():
        async with agent_factory.build_runtime(_spec()) as runtime:
            return await runtime.run(RunRequest(prompt="hello", env={"AGENTKIT_MODEL_API": turn_model_api}))

    result = asyncio.run(run())

    assert result.text == "offline reply"
    assert os.environ["AGENTKIT_MODEL_API"] == startup_model_api
    client.assert_called_once()
    constructor.assert_called_once()
    assert constructor.call_args.kwargs["client"] is client.return_value


@pytest.mark.parametrize("path", [
    "chat-default", "chat-explicit", "responses", "token-responses", "foundry-chat", "foundry-responses",
])
def test_real_model_client_uses_selected_wire_api(monkeypatch, path):
    import functools
    import json
    import httpx
    from openai import AsyncOpenAI
    from agent_framework import Content, Message
    from agent_framework.openai import OpenAIChatClient, OpenAIChatCompletionClient

    for name in (
        "AGENTKIT_MODEL_API", "AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN", "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    foundry = path.startswith("foundry-")
    responses = path in {"responses", "token-responses", "foundry-responses"}
    if path == "token-responses":
        monkeypatch.setenv("AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN", "test-token")
    if responses:
        monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")
    elif path in {"chat-explicit", "foundry-chat"}:
        monkeypatch.setenv("AGENTKIT_MODEL_API", "chat_completions")
    requests = []
    def handler(request):
        requests.append((str(request.url), json.loads(request.content)))
        if responses:
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
        spec = _spec(workload_identity=foundry or path == "token-responses")
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
            client_type = OpenAIChatClient if responses else OpenAIChatCompletionClient
            client_name = "OpenAIChatClient" if responses else "OpenAIChatCompletionClient"
            monkeypatch.setattr(agent_factory, client_name, functools.partial(client_type, async_client=openai))
        try:
            client = agent_factory.build_client(spec)
            response = await client.get_response([Message(role="user", contents=[Content.from_text("Hello.")])])
            assert response.text == "Done."
        finally:
            await openai.close()
    asyncio.run(run())
    assert len(requests) == 1
    url, payload = requests[0]
    if responses:
        assert url.endswith("/responses")
        assert "input" in payload and "messages" not in payload
    else:
        assert url.endswith("/chat/completions")
        assert "messages" in payload and "input" not in payload


@pytest.mark.parametrize("workload_identity,selection", [
    (False, "invalid-api"), (True, "invalid-api"),
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


@pytest.mark.parametrize("model_api", ["chat_completions", "responses"])
def test_project_auth_keeps_model_and_project_lifetimes_for_both_apis(monkeypatch, model_api):
    monkeypatch.setenv("AGENTKIT_MODEL_API", model_api)
    for name in (
        "AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    events = []

    class Credential:
        def close(self):
            events.append("credential-close")

    class Resource:
        def __init__(self, name):
            self.name = name

        async def __aenter__(self):
            events.append(f"{self.name}-enter")
            return self

        async def __aexit__(self, *args):
            events.append(f"{self.name}-exit")

    class FoundryClient:
        def __init__(self, **kwargs):
            self.client = Resource("model")
            self.project_client = Resource("project")
            self._prepare_message_for_openai = mock.Mock(return_value=[])

    class ChatClient:
        def __init__(self, *, model, async_client):
            self.client = async_client

    monkeypatch.setattr("azure.identity.DefaultAzureCredential", Credential)
    monkeypatch.setattr("agent_framework.foundry.FoundryChatClient", FoundryClient)
    monkeypatch.setattr(agent_factory, "OpenAIChatCompletionClient", ChatClient)

    async def run():
        runtime = agent_factory.MAFRuntime(_spec(workload_identity=True))
        async with runtime.stack:
            client = await runtime._build_model_fallback_client()
            assert isinstance(client, ChatClient if model_api == "chat_completions" else FoundryClient)
            assert client.project_client.name == "project"
            assert events == ["project-enter", "model-enter"]

    asyncio.run(run())
    assert events == ["project-enter", "model-enter", "model-exit", "project-exit", "credential-close"]


@pytest.mark.parametrize("status,item_status", [
    ("incomplete", "completed"), ("failed", "completed"),
    ("completed", "incomplete"), ("completed", "in_progress"),
])
def test_responses_does_not_execute_tools_from_an_unfinished_result(monkeypatch, status, item_status):
    import httpx
    from openai import AsyncOpenAI
    from agent_framework import FunctionTool
    from agent_framework.openai import OpenAIChatClient
    from agentkit_serve_common.config import ToolSpec
    from agentkit_serve_common.runtime import AgentRunError

    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")
    calls = []
    requests = []

    def echo(value: str) -> str:
        calls.append(value)
        return value

    def handle(request):
        requests.append(request)
        output = [{"type": "function_call", "id": "fc_test", "call_id": "call_test",
                   "name": "probe_echo", "arguments": '{"value":"must-not-run"}', "status": item_status}]
        if len(requests) > 1:
            output = [{"type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}]
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 1, "model": "test-model",
            "status": status if len(requests) == 1 else "completed", "output": output,
            "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        })

    async def run():
        spec = _spec().model_copy(update={"tools": [ToolSpec(name="probe", type="mcp", command=["unused"])]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            sdk = AsyncOpenAI(api_key="test-key", base_url=spec.model.base_url, http_client=http)
            client = OpenAIChatClient(model=spec.model.name, async_client=sdk)
            monkeypatch.setattr(agent_factory, "OpenAIChatClient", lambda **kwargs: client)
            monkeypatch.setattr(agent_factory, "build_tool", lambda *args, **kwargs:
                                FunctionTool(name="probe_echo", description="Echo value", func=echo))
            async with agent_factory.build_runtime(spec) as runtime:
                with pytest.raises(AgentRunError) as error:
                    await runtime.run(RunRequest(prompt="call echo"))
                assert error.value.code == "InvalidModelResponse"

    asyncio.run(run())
    assert calls == []
    assert len(requests) == 1


@pytest.mark.parametrize("model_api", ["responses", "chat_completions"])
def test_build_client_wraps_only_sdk_responses_resource(
    monkeypatch, model_api, auth_path, model_dependencies,
):
    # The concrete candidate is selected by the runtime, not a later env read.
    monkeypatch.setenv("AGENTKIT_MODEL_API", "invalid-api")
    spec = _spec(workload_identity=auth_path not in {"no-auth", "api-key"}, api_key=auth_path == "api-key")
    deps = model_dependencies
    selected = deps.responses if model_api == "responses" else deps.chat
    if auth_path == "foundry":
        sdk_client = deps.foundry.return_value.client
        selected.return_value.client = sdk_client
        project_client = deps.foundry.return_value.project_client
    else:
        sdk_client = selected.return_value.client
    resource = sdk_client.responses
    proxy = object()
    state = SimpleNamespace(selected=None, wrap_responses=mock.Mock(return_value=proxy))

    client = agent_factory.build_client(spec, model_api=model_api, auto_state=state)

    state.wrap_responses.assert_called_once_with(resource, client=sdk_client)
    assert client.client is sdk_client
    assert sdk_client.responses is proxy
    assert state.selected is None
    assert os.environ["AGENTKIT_MODEL_API"] == "invalid-api"
    if auth_path == "foundry":
        assert client.project_client is project_client
        assert deps.foundry.return_value.client is sdk_client
    (deps.chat if model_api == "responses" else deps.responses).assert_not_called()


@pytest.mark.parametrize("model_api", ["responses", "chat_completions"])
@pytest.mark.parametrize("prebuilt_client", [False, True])
def test_build_agent_forwards_auto_state_preserving_context_and_store(
    monkeypatch, model_api, prebuilt_client,
):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "invalid-api")
    spec = _spec()
    state = SimpleNamespace(wrap_responses=mock.Mock())
    client = SimpleNamespace(STORES_BY_DEFAULT=model_api == "responses")
    client_builder = mock.Mock(return_value=client)
    constructor = mock.Mock()
    monkeypatch.setattr(agent_factory, "build_client", client_builder)
    monkeypatch.setattr(agent_factory, "Agent", constructor)
    context = object()

    agent = agent_factory.build_agent(
        spec,
        client=client if prebuilt_client else None,
        context_providers=[context],
        model_api=model_api,
        auto_state=state,
    )

    assert agent is constructor.return_value
    if prebuilt_client:
        client_builder.assert_not_called()
    else:
        client_builder.assert_called_once_with(spec, model_api=model_api, auto_state=state)
    state.wrap_responses.assert_not_called()
    options = constructor.call_args.kwargs
    assert options["client"] is client
    assert options["default_options"] == ({"store": False} if model_api == "responses" else None)
    assert isinstance(options["context_providers"][0], agent_factory._RequestHistoryProvider)
    assert options["context_providers"][1:] == [context]


def test_auto_build_runtime_supplies_concrete_candidates_without_env_mutation(monkeypatch):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "auto")
    monkeypatch.setattr(agent_factory, "offline_orka_echo_enabled", lambda: False)
    candidate = mock.Mock()
    wrapper = mock.Mock()
    monkeypatch.setattr(agent_factory, "MAFRuntime", candidate)
    monkeypatch.setattr(agent_factory, "AutoModelRuntime", wrapper)
    spec = _spec()
    state = object()
    environment = dict(os.environ)

    runtime = agent_factory.build_runtime(spec)

    assert runtime is wrapper.return_value
    candidate.assert_not_called()
    factory = wrapper.call_args.args[0]
    factory("responses", state)
    factory("chat_completions", None)
    assert candidate.call_args_list == [
        mock.call(spec, model_api="responses", auto_state=state),
        mock.call(spec, model_api="chat_completions", auto_state=None),
    ]
    assert dict(os.environ) == environment


@pytest.mark.parametrize("model_api", ["responses", "chat_completions"])
def test_runtime_forwards_concrete_selection_and_auto_state(monkeypatch, model_api):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "invalid-api")
    state = object()
    runtime = agent_factory.MAFRuntime(_spec(), model_api=model_api, auto_state=state)
    contexts = [object()]
    client = object()
    monkeypatch.setattr(runtime, "_build_context_providers", mock.AsyncMock(return_value=contexts))
    monkeypatch.setattr(runtime, "_build_model_fallback_client", mock.AsyncMock(return_value=client))
    agent = mock.MagicMock()
    builder = mock.Mock(return_value=agent)
    monkeypatch.setattr(agent_factory, "build_agent", builder)

    async def run():
        async with runtime:
            builder.assert_called_once_with(
                runtime.spec,
                context_providers=contexts,
                stack=runtime.stack,
                client=client,
                model_api=model_api,
                auto_state=state,
            )

    asyncio.run(run())
    agent.__aenter__.assert_awaited_once()
    agent.__aexit__.assert_awaited_once()


@pytest.mark.parametrize("model_api", ["responses", "chat_completions"])
@pytest.mark.parametrize("fail_at", [None, "project", "model", "agent"])
def test_auto_project_candidate_preserves_owned_resource_cleanup(monkeypatch, model_api, fail_at):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "auto")
    for name in (
        "AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    events = []
    resources = {}
    startup_error = RuntimeError("partial enter failed")
    proxy = object()
    state = SimpleNamespace(selected=None, wrap_responses=mock.Mock(return_value=proxy))

    class Credential:
        def close(self):
            events.append("credential-close")

    class Resource:
        def __init__(self, name):
            self.name = name
            resources[name] = self

        async def __aenter__(self):
            events.append(f"{self.name}-enter")
            if fail_at == self.name:
                raise startup_error
            return self

        async def __aexit__(self, *args):
            events.append(f"{self.name}-exit")

    class FoundryClient:
        STORES_BY_DEFAULT = True

        def __init__(self, **kwargs):
            self.client = Resource("model")
            self.client.responses = object()
            self.project_client = Resource("project")
            self._prepare_message_for_openai = mock.Mock(return_value=[])

    class ChatClient:
        STORES_BY_DEFAULT = False

        def __init__(self, *, model, async_client):
            self.client = async_client

    constructor = mock.Mock(side_effect=lambda **kwargs: Resource("agent"))
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", Credential)
    monkeypatch.setattr("agent_framework.foundry.FoundryChatClient", FoundryClient)
    monkeypatch.setattr(agent_factory, "OpenAIChatCompletionClient", ChatClient)
    monkeypatch.setattr(agent_factory, "Agent", constructor)

    async def run():
        runtime = agent_factory.MAFRuntime(
            _spec(workload_identity=True), model_api=model_api, auto_state=state,
        )

        async def build_contexts():
            return [await runtime._enter_owned_async_context(Resource("context"))]

        monkeypatch.setattr(runtime, "_build_context_providers", build_contexts)
        async with runtime:
            client = constructor.call_args.kwargs["client"]
            assert client.client is resources["model"]
            assert client.project_client is resources["project"]
            assert client.client.responses is proxy
            assert constructor.call_args.kwargs["context_providers"][1:] == [resources["context"]]

    if fail_at is None:
        asyncio.run(run())
    else:
        with pytest.raises(RuntimeError) as error:
            asyncio.run(run())
        assert error.value is startup_error
    order = ["context", "project", "model", "agent"]
    entered = order if fail_at is None else order[:order.index(fail_at) + 1]
    assert events == [
        *[f"{name}-enter" for name in entered],
        *[f"{name}-exit" for name in reversed(entered[1:])],
        "credential-close", "context-exit",
    ]
    state.wrap_responses.assert_called_once()
    assert os.environ["AGENTKIT_MODEL_API"] == "auto"


def test_auto_runtime_first_request_uses_real_responses_client_for_each_auth(monkeypatch, auth_path):
    import json

    import httpx
    from openai import AsyncOpenAI

    from agentkit_serve_common.model_api_auto import AutoModelRuntime

    monkeypatch.setenv("AGENTKIT_MODEL_API", "auto")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    token_hook = mock.Mock(return_value="test-token")
    monkeypatch.setattr(agent_factory, "resolve_workload_identity_token", token_hook)
    chat_constructor = mock.Mock(side_effect=AssertionError("Responses is supported"))
    monkeypatch.setattr(agent_factory, "OpenAIChatCompletionClient", chat_constructor)
    spec = _spec(workload_identity=auth_path not in {"no-auth", "api-key"}, api_key=auth_path == "api-key")
    requests = []
    sdk_clients = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 1,
            "model": "test-model", "status": "completed",
            "output": [{"type": "message", "id": "msg_test", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}],
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            def sdk_client(**kwargs):
                kwargs.setdefault("http_client", http)
                client = AsyncOpenAI(**kwargs, max_retries=0)
                sdk_clients.append(client)
                return client

            monkeypatch.setattr("agent_framework_openai._shared.AsyncOpenAI", sdk_client)
            if auth_path == "foundry":
                class Project:
                    def __init__(self, **kwargs):
                        pass

                    def get_openai_client(self, **kwargs):
                        return sdk_client(api_key="test-token", base_url=spec.model.base_url, **kwargs)

                monkeypatch.setattr("agent_framework_foundry._chat_client.AIProjectClient", Project)
                monkeypatch.setattr(
                    "agent_framework_foundry._chat_client.create_foundry_feature_usage_http_client",
                    lambda: http,
                    raising=False,
                )
                monkeypatch.setattr("azure.identity.DefaultAzureCredential", lambda: object())
            environment = dict(os.environ)
            try:
                async with agent_factory.build_runtime(spec) as runtime:
                    assert isinstance(runtime, AutoModelRuntime)
                    assert runtime.state.selected is None
                    assert requests == []
                    assert len(sdk_clients) == 1
                    for _ in range(2):
                        result = await runtime.run(RunRequest(prompt="Hello."))
                        assert result.text == "Done."
                        assert runtime.state.selected == "responses"
                    assert dict(os.environ) == environment
            finally:
                for client in sdk_clients:
                    await client.close()

    asyncio.run(run())
    chat_constructor.assert_not_called()
    assert len(sdk_clients) == 1
    assert len(requests) == 2
    expected_key = "test-key" if auth_path == "api-key" else "not-needed" if auth_path == "no-auth" else "test-token"
    for request in requests:
        assert request.url.path.endswith("/responses")
        assert request.headers["authorization"] == f"Bearer {expected_key}"
        assert json.loads(request.content)["store"] is False
    if auth_path in {"generic-token", "generic-token-command"}:
        assert token_hook.call_args_list == [mock.call("https://ai.azure.com/.default")] * 2
    else:
        token_hook.assert_not_called()


def test_responses_phase_requires_message_serializer():
    with pytest.raises(agent_factory.AgentBuildError, match="required message serialization hook"):
        agent_factory._preserve_responses_phase(SimpleNamespace())


def _phase_reply(model_api):
    import httpx

    if model_api == "responses":
        body = {
            "id": "resp_phase", "object": "response", "created_at": 1,
            "model": "test-model", "status": "completed",
            "output": [{"type": "message", "id": "msg_phase", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}],
        }
    else:
        body = {
            "id": "chatcmpl_phase", "object": "chat.completion", "created": 1, "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Done."}}],
        }
    return httpx.Response(200, json=body)


def _inject_phase_http(monkeypatch, handler, spec, *, foundry=False):
    import httpx
    from openai import AsyncOpenAI

    clients = []

    def http_client():
        # Fallback closes the Responses candidate before constructing Chat.
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    def sdk_client(**kwargs):
        if "http_client" not in kwargs:
            kwargs["http_client"] = http_client()
        return AsyncOpenAI(**kwargs, max_retries=0)

    # Keep both MAF clients and SDK serialization real, including auto fallback.
    monkeypatch.setattr("agent_framework_openai._shared.AsyncOpenAI", sdk_client)
    if foundry:
        class Project:
            def __init__(self, **kwargs):
                pass

            def get_openai_client(self, **kwargs):
                return sdk_client(api_key="test-key", base_url=spec.model.base_url, **kwargs)

        monkeypatch.setattr("agent_framework_foundry._chat_client.AIProjectClient", Project)
        monkeypatch.setattr(
            "agent_framework_foundry._chat_client.create_foundry_feature_usage_http_client",
            http_client,
            raising=False,
        )
        monkeypatch.setattr("azure.identity.DefaultAzureCredential", lambda: object())
    return clients


@pytest.mark.parametrize("phase", [None, "commentary", "final_answer"])
@pytest.mark.parametrize("selection", ["responses", "auto", "chat_completions", "auto-chat"])
@pytest.mark.parametrize("foundry", [False, True], ids=["openai", "foundry"])
def test_responses_facade_preserves_assistant_phase_on_upstream_wire(monkeypatch, selection, phase, foundry):
    import json
    import httpx
    from fastapi.testclient import TestClient
    from agentkit_serve_common.server import create_app

    for name in (
        "AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN", "AGENTKIT_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGENTKIT_MODEL_API", "auto" if selection == "auto-chat" else selection)
    spec = _spec(workload_identity=foundry)
    requests = []
    expected_api = "chat_completions" if selection in {"chat_completions", "auto-chat"} else "responses"

    def handle(request):
        requests.append((request.url.path, json.loads(request.content)))
        if selection == "auto-chat" and request.url.path.endswith("/responses"):
            return httpx.Response(404, json={"error": {"code": "unsupported_endpoint"}})
        assert request.url.path.endswith("/responses" if expected_api == "responses" else "/chat/completions")
        return _phase_reply(expected_api)

    clients = _inject_phase_http(monkeypatch, handle, spec, foundry=foundry)
    text = "Checking the logs."
    try:
        with TestClient(create_app(spec, agent_factory)) as client:
            response = client.post("/v1/responses", json={
                "input": [
                    {"role": "user", "content": "Prior request"},
                    {"type": "message", "role": "assistant", "phase": phase,
                     "content": [{"type": "output_text", "text": text}]},
                    {"role": "user", "content": "Anything else?"},
                    # Identical unphased text must remain a separate, unphased turn.
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "Continue."},
                ],
            })
            assert response.status_code == 200, response.text
            assert response.json()["output"][0]["content"][0]["text"] == "Done."
        assert len(requests) == (2 if selection == "auto-chat" else 1)
        for path, body in requests:
            responses = path.endswith("/responses")
            items = body["input"] if responses else body["messages"]
            assert [item["role"] for item in items if item["role"] != "system"] == [
                "user", "assistant", "user", "assistant", "user",
            ]
            prior = [item for item in items if item["role"] == "assistant"]
            expected_content = [{"type": "output_text", "text": text, "annotations": []}] if responses else text
            assert all(item["content"] == expected_content for item in prior)
            if responses and phase is not None:
                assert prior[0]["phase"] == phase
            else:
                assert "phase" not in prior[0]
            assert all("phase" not in item for item in items if item is not prior[0])
    finally:
        for http in clients:
            asyncio.run(http.aclose())
