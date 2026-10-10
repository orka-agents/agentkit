from __future__ import annotations

import asyncio
import json
from unittest import mock

import pytest
from mcp import McpError

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec, ToolSpec


def _runtime() -> agent_factory.MAFRuntime:
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "lifecycle-test"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "expose": {"openai": True, "port": 8080},
        }
    )
    return agent_factory.MAFRuntime(spec)


def test_agent_partial_enter_failure_closes_agent_and_context_without_masking_startup_error():
    runtime = _runtime()
    startup_error = RuntimeError("agent startup failed")
    agent_cleanup_error = RuntimeError("agent cleanup failed")
    events: list[str] = []
    owner_tasks: dict[str, asyncio.Task] = {}

    class _ContextResource:
        async def __aenter__(self):
            events.append("context-enter")
            owner_tasks["context"] = asyncio.current_task()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks["context"]
            events.append("context-exit")

    class _PartialAgent:
        async def __aenter__(self):
            events.append("agent-enter")
            owner_tasks["agent"] = asyncio.current_task()
            raise startup_error

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks["agent"]
            events.append("agent-exit")
            raise agent_cleanup_error

    async def _build_context_providers():
        await runtime.stack.enter_async_context(_ContextResource())
        return None

    runtime._build_context_providers = _build_context_providers  # type: ignore[method-assign]
    with mock.patch("agentkit_serve.agent_factory.build_agent", return_value=_PartialAgent()):
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(runtime.__aenter__())

    assert exc_info.value is startup_error
    assert events == ["context-enter", "agent-enter", "agent-exit", "context-exit"]
    assert runtime.agent is None


def test_runtime_exit_closes_context_stack_even_when_agent_exit_raises():
    runtime = _runtime()
    agent_exit_error = RuntimeError("agent exit failed")
    events: list[str] = []
    owner_tasks: dict[str, asyncio.Task] = {}

    class _ContextResource:
        async def __aenter__(self):
            events.append("context-enter")
            owner_tasks["context"] = asyncio.current_task()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks["context"]
            events.append("context-exit")

    class _Agent:
        async def __aenter__(self):
            events.append("agent-enter")
            owner_tasks["agent"] = asyncio.current_task()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks["agent"]
            events.append("agent-exit")
            raise agent_exit_error

    async def _build_context_providers():
        await runtime.stack.enter_async_context(_ContextResource())
        return None

    async def exercise() -> None:
        runtime._build_context_providers = _build_context_providers  # type: ignore[method-assign]
        with mock.patch("agentkit_serve.agent_factory.build_agent", return_value=_Agent()):
            await runtime.__aenter__()
        with pytest.raises(RuntimeError) as exc_info:
            await runtime.__aexit__(None, None, None)
        assert exc_info.value is agent_exit_error

    asyncio.run(exercise())

    assert events == ["context-enter", "agent-enter", "agent-exit", "context-exit"]
    assert runtime.agent is None


def test_cancellation_while_opening_second_context_resource_finishes_cleanup():
    runtime = _runtime()
    second_opened = asyncio.Event()
    second_exit_started = asyncio.Event()
    allow_second_exit = asyncio.Event()
    exited: list[str] = []
    owner_tasks: dict[str, asyncio.Task] = {}

    class _ContextResource:
        def __init__(self, name: str) -> None:
            self.name = name

        async def __aenter__(self):
            owner_tasks[self.name] = asyncio.current_task()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks[self.name]
            if self.name == "second":
                second_exit_started.set()
                await allow_second_exit.wait()
            exited.append(self.name)

    async def _build_context_providers():
        await runtime.stack.enter_async_context(_ContextResource("first"))
        await runtime.stack.enter_async_context(_ContextResource("second"))
        second_opened.set()
        await asyncio.Event().wait()

    async def exercise() -> None:
        runtime._build_context_providers = _build_context_providers  # type: ignore[method-assign]
        task = asyncio.create_task(runtime.__aenter__())
        try:
            await asyncio.wait_for(second_opened.wait(), timeout=1)
            task.cancel()
            await asyncio.wait_for(second_exit_started.wait(), timeout=1)

            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()

            allow_second_exit.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            allow_second_exit.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())

    assert exited == ["second", "first"]


def test_exit_cancellation_waits_for_owner_cleanup_and_keeps_cancellation_primary():
    runtime = _runtime()
    cleanup_error = RuntimeError("context cleanup failed")
    exit_started = asyncio.Event()
    allow_exit = asyncio.Event()
    events: list[str] = []
    owner_tasks: dict[str, asyncio.Task] = {}

    class _ContextResource:
        async def __aenter__(self):
            events.append("context-enter")
            owner_tasks["context"] = asyncio.current_task()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks["context"]
            exit_started.set()
            await allow_exit.wait()
            events.append("context-exit")
            raise cleanup_error

    class _Agent:
        async def __aenter__(self):
            events.append("agent-enter")
            owner_tasks["agent"] = asyncio.current_task()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            assert asyncio.current_task() is owner_tasks["agent"]
            events.append("agent-exit")

    async def _build_context_providers():
        await runtime.stack.enter_async_context(_ContextResource())
        return None

    async def exercise() -> None:
        runtime._build_context_providers = _build_context_providers  # type: ignore[method-assign]
        with mock.patch("agentkit_serve.agent_factory.build_agent", return_value=_Agent()):
            await runtime.__aenter__()

        exit_task = asyncio.create_task(runtime.__aexit__(None, None, None))
        try:
            await asyncio.wait_for(exit_started.wait(), timeout=1)
            exit_task.cancel()
            allow_exit.set()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await exit_task
            assert exc_info.value.__cause__ is cleanup_error
        finally:
            allow_exit.set()
            if not exit_task.done():
                exit_task.cancel()
            await asyncio.gather(exit_task, return_exceptions=True)

    asyncio.run(exercise())

    assert events == ["context-enter", "agent-enter", "agent-exit", "context-exit"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 120), ("garbage", 120), ("-1", 120), ("7.9", 7)],
)
def test_stdio_mcp_timeout_defaults_to_120_seconds_and_honors_override(
    monkeypatch,
    raw: str | None,
    expected: int,
):
    tool = ToolSpec(name="fetch", command=["fake-mcp"], env=[])

    if raw is None:
        monkeypatch.delenv("AGENTKIT_MCP_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("AGENTKIT_MCP_TIMEOUT", raw)
    assert agent_factory.build_tool(tool).request_timeout == expected


def test_mcp_skills_initialize_timeout_closes_live_transport(monkeypatch):
    request_bodies: list[bytes] = []
    active_writers: set[asyncio.StreamWriter] = set()
    handler_tasks: set[asyncio.Task[None]] = set()
    accepted = asyncio.Event()
    disconnected = asyncio.Event()

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        handler_tasks.add(task)
        active_writers.add(writer)
        try:
            header_bytes = await reader.readuntil(b"\r\n\r\n")
            headers = header_bytes.decode("ascii").split("\r\n")
            content_length = next(
                int(line.split(":", 1)[1].strip())
                for line in headers
                if line.lower().startswith("content-length:")
            )
            request_bodies.append(await reader.readexactly(content_length))
            writer.write(
                b"HTTP/1.1 202 Accepted\r\n"
                b"Content-Length: 0\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\n"
            )
            await writer.drain()
            accepted.set()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            active_writers.discard(writer)
            handler_tasks.discard(task)
            disconnected.set()

    async def exercise() -> None:
        server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
        assert server.sockets
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("AGENTKIT_MCP_TIMEOUT", "1")
        monkeypatch.setenv("TOOLBOX_ENDPOINT", f"http://127.0.0.1:{port}/mcp")
        spec = AgentSpec.model_validate(
            {
                "abiVersion": "v0",
                "metadata": {"name": "mcp-skills-timeout"},
                "model": {
                    "provider": "openai-compatible",
                    "baseURL": "https://api.openai.com/v1",
                    "name": "gpt-4o-mini",
                },
                "instructions": "Be helpful.",
                "tools": [
                    {
                        "name": "toolbox",
                        "type": "mcp",
                        "transport": "streamable-http",
                        "urlEnv": "TOOLBOX_ENDPOINT",
                    }
                ],
                "context": {
                    "providers": [
                        {
                            "type": "skills",
                            "source": "mcp",
                            "toolRef": "toolbox",
                        }
                    ]
                },
                "expose": {"openai": True, "port": 8080},
            }
        )
        runtime = agent_factory.MAFRuntime(spec)

        try:
            with pytest.raises(McpError, match="Timed out while waiting for response"):
                await asyncio.wait_for(runtime.__aenter__(), timeout=3)

            await asyncio.wait_for(accepted.wait(), timeout=1)
            await asyncio.wait_for(disconnected.wait(), timeout=1)
            assert runtime.agent is None
            assert not active_writers
            assert json.loads(request_bodies[0])["method"] == "initialize"
        finally:
            server.close()
            for writer in tuple(active_writers):
                writer.close()
            if active_writers:
                await asyncio.gather(
                    *(writer.wait_closed() for writer in tuple(active_writers)),
                    return_exceptions=True,
                )
            if handler_tasks:
                await asyncio.gather(*tuple(handler_tasks), return_exceptions=True)
            await server.wait_closed()

    asyncio.run(exercise())


def test_runtime_owns_remote_mcp_http_client_and_closes_it_after_agent(monkeypatch):
    monkeypatch.setenv("TOOLBOX_ENDPOINT", "http://127.0.0.1:8765/mcp")
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "remote-client-lifecycle"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
            },
            "instructions": "Be helpful.",
            "tools": [
                {
                    "name": "toolbox",
                    "type": "mcp",
                    "transport": "streamable-http",
                    "urlEnv": "TOOLBOX_ENDPOINT",
                }
            ],
            "expose": {"openai": True, "port": 8080},
        }
    )
    runtime = agent_factory.MAFRuntime(spec)
    agents = []
    events: list[str] = []

    class _Agent:
        def __init__(self, **kwargs):
            self.tools = kwargs["tools"]
            agents.append(self)

        async def __aenter__(self):
            events.append("agent-enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            http_client = self.tools[0]._httpx_client
            assert http_client is not None
            assert http_client.is_closed is False
            events.append("agent-exit")

    async def exercise() -> None:
        with (
            mock.patch("agentkit_serve.agent_factory.Agent", _Agent),
            mock.patch("agentkit_serve.agent_factory.build_client", return_value=object()),
        ):
            await runtime.__aenter__()
        http_client = agents[0].tools[0]._httpx_client
        assert http_client is not None
        assert http_client.is_closed is False
        await runtime.__aexit__(None, None, None)
        assert http_client.is_closed is True

    asyncio.run(exercise())
    assert events == ["agent-enter", "agent-exit"]


def test_runtime_enters_async_context_provider_and_closes_it_before_credential(monkeypatch):
    monkeypatch.setenv("SEARCH_ENDPOINT", "https://example.search.windows.net")
    monkeypatch.setenv("SEARCH_INDEX", "knowledge")
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "context-lifecycle"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "context": {
                "providers": [
                    {
                        "name": "knowledge",
                        "type": "search",
                        "endpointEnv": "SEARCH_ENDPOINT",
                        "indexEnv": "SEARCH_INDEX",
                    }
                ]
            },
            "expose": {"openai": True, "port": 8080},
        }
    )
    runtime = agent_factory.MAFRuntime(spec)
    events: list[str] = []

    class _Credential:
        async def close(self):
            events.append("credential-close")

    class _Provider:
        def __init__(self, **kwargs):
            self.credential = kwargs["credential"]
            events.append("provider-create")

        async def __aenter__(self):
            events.append("provider-enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("provider-exit")

    class _AzureModule:
        AzureAISearchContextProvider = _Provider

    class _Agent:
        async def __aenter__(self):
            events.append("agent-enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("agent-exit")

    def _build_agent(spec, *, context_providers=None, stack=None, client=None):
        assert context_providers and isinstance(context_providers[0], _Provider)
        return _Agent()

    async def exercise() -> None:
        with (
            mock.patch("agentkit_serve.agent_factory._credential_for_context", return_value=_Credential()),
            mock.patch("agentkit_serve.agent_factory.build_agent", side_effect=_build_agent),
            mock.patch("agentkit_serve.agent_factory.importlib.import_module", return_value=_AzureModule),
        ):
            await runtime.__aenter__()
        await runtime.__aexit__(None, None, None)

    asyncio.run(exercise())
    assert events == [
        "provider-create",
        "provider-enter",
        "agent-enter",
        "agent-exit",
        "provider-exit",
        "credential-close",
    ]


def test_runtime_owns_memory_provider_project_client_and_sync_credential(monkeypatch):
    monkeypatch.setenv("MEMORY_ENDPOINT", "https://example.services.ai.azure.com/api/projects/proj")
    monkeypatch.setenv("MEMORY_STORE_NAME", "agentkit-memory")
    monkeypatch.setenv("AGENTKIT_MEMORY_SCOPE", "scope-1")
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "memory-lifecycle"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "context": {
                "providers": [
                    {
                        "name": "memory",
                        "type": "memory",
                        "endpointEnv": "MEMORY_ENDPOINT",
                        "storeNameEnv": "MEMORY_STORE_NAME",
                    }
                ]
            },
            "expose": {"openai": True, "port": 8080},
        }
    )
    runtime = agent_factory.MAFRuntime(spec)
    events: list[str] = []

    class _Credential:
        def close(self):
            events.append("credential-close")

    class _ProjectClient:
        async def __aenter__(self):
            events.append("project-enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("project-exit")

    class _MemoryProvider:
        def __init__(self, **kwargs):
            self.credential = kwargs["credential"]
            self.project_client = _ProjectClient()
            events.append("provider-create")

        async def __aenter__(self):
            events.append("provider-enter")
            await self.project_client.__aenter__()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("provider-exit-start")
            await self.project_client.__aexit__(exc_type, exc, tb)
            events.append("provider-exit")

    class _FoundryModule:
        FoundryMemoryProvider = _MemoryProvider

    class _Agent:
        async def __aenter__(self):
            events.append("agent-enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("agent-exit")

    def _build_agent(spec, *, context_providers=None, stack=None, client=None):
        assert context_providers and isinstance(context_providers[0], _MemoryProvider)
        return _Agent()

    async def exercise() -> None:
        with (
            mock.patch("agentkit_serve.agent_factory._credential_for_context", return_value=_Credential()),
            mock.patch("agentkit_serve.agent_factory.build_agent", side_effect=_build_agent),
            mock.patch("agentkit_serve.agent_factory.importlib.import_module", return_value=_FoundryModule),
        ):
            await runtime.__aenter__()
        await runtime.__aexit__(None, None, None)

    asyncio.run(exercise())
    assert events == [
        "provider-create",
        "provider-enter",
        "project-enter",
        "agent-enter",
        "agent-exit",
        "provider-exit-start",
        "project-exit",
        "provider-exit",
        "credential-close",
    ]


def test_runtime_owns_model_fallback_credential_and_project_client_without_double_closing_framework_client(
    monkeypatch,
):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")
    for name in (
        "AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "model-lifecycle"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://example.services.ai.azure.com/api/projects/proj/openai/v1",
                "name": "gpt-4.1-mini",
                "auth": {
                    "type": "workload-identity-token",
                    "audience": "https://ai.azure.com/.default",
                },
            },
            "instructions": "Be helpful.",
            "tools": [],
            "expose": {"openai": True, "port": 8080},
        }
    )
    runtime = agent_factory.MAFRuntime(spec)
    events: list[str] = []

    class _Credential:
        def __init__(self):
            events.append("credential-create")

        def close(self):
            events.append("credential-close")

    class _ProjectClient:
        async def __aenter__(self):
            events.append("project-enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("project-exit")

    class _ModelHTTPClient:
        async def close(self):
            events.append("model-http-close")

    class _FoundryClient:
        _prepare_message_for_openai = mock.Mock(return_value=[])

        def __init__(self, **kwargs):
            assert isinstance(kwargs["credential"], _Credential)
            self.project_client = _ProjectClient()
            self.client = _ModelHTTPClient()
            events.append("client-create")

        async def __aenter__(self):
            events.append("client-enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            await self.client.close()
            events.append("client-exit")

    class _Agent:
        def __init__(self, **kwargs):
            self.client = kwargs["client"]

        async def __aenter__(self):
            events.append("agent-enter")
            await self.client.__aenter__()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("agent-exit-start")
            await self.client.__aexit__(exc_type, exc, tb)
            events.append("agent-exit")

    async def exercise() -> None:
        with (
            mock.patch("azure.identity.DefaultAzureCredential", _Credential),
            mock.patch("agent_framework.foundry.FoundryChatClient", _FoundryClient),
            mock.patch("agentkit_serve.agent_factory.Agent", _Agent),
        ):
            await runtime.__aenter__()
        await runtime.__aexit__(None, None, None)

    asyncio.run(exercise())
    assert events == [
        "credential-create",
        "client-create",
        "project-enter",
        "agent-enter",
        "client-enter",
        "agent-exit-start",
        "model-http-close",
        "client-exit",
        "agent-exit",
        "project-exit",
        "credential-close",
    ]


def test_partial_agent_startup_closes_all_runtime_owned_resources_in_dependency_order(monkeypatch):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")
    for name in (
        "AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SEARCH_ENDPOINT", "https://example.search.windows.net")
    monkeypatch.setenv("SEARCH_INDEX", "knowledge")
    monkeypatch.setenv("TOOLBOX_ENDPOINT", "http://127.0.0.1:8765/mcp")
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "partial-startup-lifecycle"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://example.services.ai.azure.com/api/projects/proj/openai/v1",
                "name": "gpt-4.1-mini",
                "auth": {
                    "type": "workload-identity-token",
                    "audience": "https://ai.azure.com/.default",
                },
            },
            "instructions": "Be helpful.",
            "tools": [
                {
                    "name": "toolbox",
                    "type": "mcp",
                    "transport": "streamable-http",
                    "urlEnv": "TOOLBOX_ENDPOINT",
                }
            ],
            "context": {
                "providers": [
                    {
                        "name": "knowledge",
                        "type": "search",
                        "endpointEnv": "SEARCH_ENDPOINT",
                        "indexEnv": "SEARCH_INDEX",
                    }
                ]
            },
            "expose": {"openai": True, "port": 8080},
        }
    )
    runtime = agent_factory.MAFRuntime(spec)
    startup_error = RuntimeError("agent startup failed")
    events: list[str] = []

    class _ContextCredential:
        def __init__(self):
            events.append("context-credential-create")

        async def close(self):
            events.append("context-credential-close")

    class _ContextProvider:
        def __init__(self, **kwargs):
            events.append("context-provider-create")

        async def __aenter__(self):
            events.append("context-provider-enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("context-provider-exit")

    class _AzureModule:
        AzureAISearchContextProvider = _ContextProvider

    class _ModelCredential:
        def __init__(self):
            events.append("model-credential-create")

        def close(self):
            events.append("model-credential-close")

    class _ProjectClient:
        async def __aenter__(self):
            events.append("project-enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("project-exit")

    class _ModelHTTPClient:
        async def close(self):
            events.append("model-http-close")

    class _FoundryClient:
        _prepare_message_for_openai = mock.Mock(return_value=[])

        def __init__(self, **kwargs):
            self.project_client = _ProjectClient()
            self.client = _ModelHTTPClient()
            events.append("model-client-create")

    class _HTTPClient:
        def __init__(self, **kwargs):
            events.append("http-client-create")

        async def aclose(self):
            events.append("http-client-close")

    class _PartialAgent:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            events.append("agent-enter")
            raise startup_error

        async def __aexit__(self, exc_type, exc, tb):
            events.append("agent-exit")

    async def exercise() -> None:
        with (
            mock.patch(
                "agentkit_serve.agent_factory._credential_for_context",
                side_effect=lambda *args, **kwargs: _ContextCredential(),
            ),
            mock.patch("azure.identity.DefaultAzureCredential", _ModelCredential),
            mock.patch("agent_framework.foundry.FoundryChatClient", _FoundryClient),
            mock.patch("agentkit_serve.agent_factory.AsyncClient", _HTTPClient),
            mock.patch("agentkit_serve.agent_factory.Agent", _PartialAgent),
            mock.patch("agentkit_serve.agent_factory.importlib.import_module", return_value=_AzureModule),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await runtime.__aenter__()
        assert exc_info.value is startup_error

    asyncio.run(exercise())
    assert events == [
        "context-credential-create",
        "context-provider-create",
        "context-provider-enter",
        "model-credential-create",
        "model-client-create",
        "project-enter",
        "http-client-create",
        "agent-enter",
        "agent-exit",
        "http-client-close",
        "model-http-close",
        "project-exit",
        "model-credential-close",
        "context-provider-exit",
        "context-credential-close",
    ]
    assert runtime.agent is None


def test_startup_cancellation_waits_for_all_runtime_owned_resource_cleanup(monkeypatch):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")
    for name in (
        "AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SEARCH_ENDPOINT", "https://example.search.windows.net")
    monkeypatch.setenv("SEARCH_INDEX", "knowledge")
    monkeypatch.setenv("TOOLBOX_ENDPOINT", "http://127.0.0.1:8765/mcp")
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "cancelled-startup-lifecycle"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://example.services.ai.azure.com/api/projects/proj/openai/v1",
                "name": "gpt-4.1-mini",
                "auth": {
                    "type": "workload-identity-token",
                    "audience": "https://ai.azure.com/.default",
                },
            },
            "instructions": "Be helpful.",
            "tools": [
                {
                    "name": "toolbox",
                    "type": "mcp",
                    "transport": "streamable-http",
                    "urlEnv": "TOOLBOX_ENDPOINT",
                }
            ],
            "context": {
                "providers": [
                    {
                        "name": "knowledge",
                        "type": "search",
                        "endpointEnv": "SEARCH_ENDPOINT",
                        "indexEnv": "SEARCH_INDEX",
                    }
                ]
            },
            "expose": {"openai": True, "port": 8080},
        }
    )
    runtime = agent_factory.MAFRuntime(spec)
    events: list[str] = []
    agent_started = asyncio.Event()
    http_close_started = asyncio.Event()
    allow_http_close = asyncio.Event()

    class _ContextCredential:
        async def close(self):
            events.append("context-credential-close")

    class _ContextProvider:
        def __init__(self, **kwargs):
            events.append("context-provider-create")

        async def __aenter__(self):
            events.append("context-provider-enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("context-provider-exit")

    class _AzureModule:
        AzureAISearchContextProvider = _ContextProvider

    class _ModelCredential:
        def close(self):
            events.append("model-credential-close")

    class _ProjectClient:
        async def __aenter__(self):
            events.append("project-enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("project-exit")

    class _ModelHTTPClient:
        async def close(self):
            events.append("model-http-close")

    class _FoundryClient:
        _prepare_message_for_openai = mock.Mock(return_value=[])

        def __init__(self, **kwargs):
            self.project_client = _ProjectClient()
            self.client = _ModelHTTPClient()
            events.append("model-client-create")

    class _HTTPClient:
        def __init__(self, **kwargs):
            events.append("http-client-create")

        async def aclose(self):
            events.append("http-client-close-start")
            http_close_started.set()
            await allow_http_close.wait()
            events.append("http-client-close")

    class _BlockingAgent:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            events.append("agent-enter")
            agent_started.set()
            await asyncio.Event().wait()

        async def __aexit__(self, exc_type, exc, tb):
            events.append("agent-exit")

    async def exercise() -> None:
        with (
            mock.patch(
                "agentkit_serve.agent_factory._credential_for_context",
                side_effect=lambda *args, **kwargs: _ContextCredential(),
            ),
            mock.patch("azure.identity.DefaultAzureCredential", _ModelCredential),
            mock.patch("agent_framework.foundry.FoundryChatClient", _FoundryClient),
            mock.patch("agentkit_serve.agent_factory.AsyncClient", _HTTPClient),
            mock.patch("agentkit_serve.agent_factory.Agent", _BlockingAgent),
            mock.patch("agentkit_serve.agent_factory.importlib.import_module", return_value=_AzureModule),
        ):
            task = asyncio.create_task(runtime.__aenter__())
            try:
                await asyncio.wait_for(agent_started.wait(), timeout=1)
                task.cancel()
                await asyncio.wait_for(http_close_started.wait(), timeout=1)

                task.cancel()
                await asyncio.sleep(0)
                assert task.done() is False

                allow_http_close.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                allow_http_close.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())
    assert events == [
        "context-provider-create",
        "context-provider-enter",
        "model-client-create",
        "project-enter",
        "http-client-create",
        "agent-enter",
        "agent-exit",
        "http-client-close-start",
        "http-client-close",
        "model-http-close",
        "project-exit",
        "model-credential-close",
        "context-provider-exit",
        "context-credential-close",
    ]
    assert runtime.agent is None


def test_runtime_closes_model_fallback_http_client_before_project_and_credential(monkeypatch):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")
    for name in (
        "AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN",
        "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "model-http-lifecycle"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://example.services.ai.azure.com/api/projects/proj/openai/v1",
                "name": "gpt-4.1-mini",
                "auth": {
                    "type": "workload-identity-token",
                    "audience": "https://ai.azure.com/.default",
                },
            },
            "instructions": "Be helpful.",
            "tools": [],
            "expose": {"openai": True, "port": 8080},
        }
    )
    runtime = agent_factory.MAFRuntime(spec)
    events: list[str] = []

    class _Credential:
        def __init__(self):
            events.append("credential-create")

        def close(self):
            events.append("credential-close")

    class _ProjectClient:
        async def __aenter__(self):
            events.append("project-enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("project-exit")

    class _ModelHTTPClient:
        async def close(self):
            events.append("model-http-close")

    class _FoundryClient:
        _prepare_message_for_openai = mock.Mock(return_value=[])

        def __init__(self, **kwargs):
            assert isinstance(kwargs["credential"], _Credential)
            self.project_client = _ProjectClient()
            self.client = _ModelHTTPClient()
            events.append("client-create")

    class _Agent:
        def __init__(self, **kwargs):
            self.client = kwargs["client"]

        async def __aenter__(self):
            events.append("agent-enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("agent-exit")

    async def exercise() -> None:
        with (
            mock.patch("azure.identity.DefaultAzureCredential", _Credential),
            mock.patch("agent_framework.foundry.FoundryChatClient", _FoundryClient),
            mock.patch("agentkit_serve.agent_factory.Agent", _Agent),
        ):
            await runtime.__aenter__()
        await runtime.__aexit__(None, None, None)

    asyncio.run(exercise())
    assert events == [
        "credential-create",
        "client-create",
        "project-enter",
        "agent-enter",
        "agent-exit",
        "model-http-close",
        "project-exit",
        "credential-close",
    ]
