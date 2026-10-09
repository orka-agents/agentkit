"""Build a Microsoft Agent Framework (MAF) agent from a validated AgentSpec.

This module is the framework-specific translation layer behind the frozen
``/agent/agent.yaml`` ABI. The ABI loader, the ``/v1`` facade, Foundry protocol
wrapper, and CLI live in ``agentkit_serve_common``.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from contextvars import ContextVar
from datetime import timedelta
from types import TracebackType
from urllib.parse import urlsplit
from uuid import uuid4

from agent_framework import (
    Agent,
    AgentSession,
    ChatContext,
    ChatMiddleware,
    FileSkillsSource,
    FunctionInvocationContext,
    FunctionMiddleware,
    FunctionTool,
    HistoryProvider,
    MCPSkillsSource,
    MCPStdioTool,
    MCPStreamableHTTPTool,
    Message,
    MiddlewareTermination,
    SkillsProvider,
)
from agent_framework.openai import OpenAIChatCompletionClient
from httpx import AsyncClient, URL
from mcp.types import CallToolResult
from agentkit_serve_common.adapter_support import (
    FORWARDED_ROLES,
    AsyncExitStackLifecycle,
    AgentBuildError,
    declared_tool_env,
    mcp_tool_protocol_error,
    normalize_agent_run_error,
    positive_int_env,
    resolve_api_key,
    resolve_model_api,
    resolve_workload_identity_token,
    resolve_tool_headers,
    resolve_tool_url,
    same_origin_mcp_httpx_client_factory,
    split_tool_command,
    upstream_status_code,
)
from agentkit_serve_common.config import AgentSpec, ContextProviderSpec, ToolSpec
from agentkit_serve_common.conversation import RunRequest, ToolCallEvent
from agentkit_serve_common.runtime import (
    AgentRunError,
    OfflineEchoRuntimeFactory,
    RunResult,
    RuntimeSession,
    offline_orka_echo_enabled,
)
from agentkit_serve_common.tool_errors import orka_tool_error_details

_AUTH_WORKLOAD_IDENTITY = "workload-identity-token"
_CONTEXT_TYPE_SEARCH = "search"
_CONTEXT_TYPE_SKILLS = "skills"
_CONTEXT_TYPE_MEMORY = "memory"
_CONTEXT_SOURCE_FILESYSTEM = "filesystem"
_CONTEXT_SOURCE_MCP = "mcp"
_DEFAULT_SEARCH_AUDIENCE = "https://search.azure.com/.default"
_DEFAULT_FOUNDRY_AUDIENCE = "https://ai.azure.com/.default"
_DEFAULT_MCP_REQUEST_TIMEOUT = 120
_DEFAULT_SESSION_CACHE_MAX = 256
class _RunFailure:
    """The error that ends a run, shared by its concurrent tool calls."""

    def __init__(self) -> None:
        self.signal: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.error: AgentRunError | None = None

    def record(self, error: AgentRunError) -> None:
        # The first failure ends the run, but a sibling call that then finds
        # its session dead must still mark the runtime unhealthy.
        if self.error is None or (error.fatal and not self.error.fatal):
            self.error = error
        if not self.signal.done():
            self.signal.set_result(None)


# Invocation tasks share a fatal-error signal with their run owner. Concurrent
# Sessions on the same Agent have independent signals.
_run_failure: ContextVar[_RunFailure | None] = ContextVar("agentkit_maf_run_failure", default=None)
_ORKA_TOOL_ERROR_CONTEXT_KEY = "agentkit_orka_tool_error"


def _fail_run(message: str, *, code: str | None = None) -> None:
    _fail_run_with(AgentRunError(message, code=code))


def _fail_run_with(error: AgentRunError) -> None:
    failure = _run_failure.get()
    if failure is not None:
        failure.record(error)


def _mcp_request_timeout() -> int:
    """MCP request timeout (seconds), overridable via ``AGENTKIT_MCP_TIMEOUT``."""
    return positive_int_env(default=_DEFAULT_MCP_REQUEST_TIMEOUT) or _DEFAULT_MCP_REQUEST_TIMEOUT


def _remote_mcp_timeout() -> float:
    """Use the shared bounded MCP timeout for network-backed calls."""
    return float(_mcp_request_timeout())


def _resolve_api_key(spec: AgentSpec) -> str:
    """Compatibility wrapper around the shared adapter support module."""
    return resolve_api_key(spec)


def _project_endpoint_from_openai_base_url(base_url: str) -> str:
    """Extract a Foundry project endpoint from ``.../openai/v1`` base URLs."""
    marker = "/openai/v1"
    if marker not in base_url:
        raise AgentBuildError(
            "model.auth workload-identity-token for the MAF runtime requires "
            "model.baseURL to be a Foundry project OpenAI endpoint ending in /openai/v1"
        )
    return base_url.split(marker, 1)[0].rstrip("/")


def _env_required(name: str | None, *, field: str) -> str:
    if not name:
        raise AgentBuildError(f"{field} is required")
    value = os.environ.get(name)
    if value is None or value == "":
        raise AgentBuildError(
            f"{field} {name!r} is declared in agent.yaml but env var {name!r} is not set; "
            "inject it at runtime"
        )
    return value


class _BearerTokenCredential:
    """Tiny Azure TokenCredential over AgentKit's generic workload token hook."""

    def __init__(self, audience: str) -> None:
        self._audience = audience

    def get_token(self, *scopes: str, **kwargs):  # noqa: ANN003 - Azure credential protocol
        credentials_mod = importlib.import_module("azure.core.credentials")
        access_token = getattr(credentials_mod, "AccessToken")

        audience = scopes[0] if scopes else self._audience
        token = resolve_workload_identity_token(audience or self._audience)
        return access_token(token, int(time.time()) + 300)


class _AsyncBearerTokenCredential:
    """Async TokenCredential over AgentKit's generic workload token hook."""

    def __init__(self, audience: str) -> None:
        self._audience = audience

    async def get_token(self, *scopes: str, **kwargs):  # noqa: ANN003 - Azure async credential protocol
        credentials_mod = importlib.import_module("azure.core.credentials")
        access_token = getattr(credentials_mod, "AccessToken")

        audience = scopes[0] if scopes else self._audience
        token = await asyncio.to_thread(resolve_workload_identity_token, audience or self._audience)
        return access_token(token, int(time.time()) + 300)


def _credential_for_context(
    provider: ContextProviderSpec,
    *,
    default_audience: str,
    async_credential: bool = False,
):
    auth = provider.auth
    if auth is None or auth.type == _AUTH_WORKLOAD_IDENTITY:
        audience = auth.audience if auth and auth.audience else default_audience
        # If AgentKit's generic token hook is configured, use it. Otherwise fall
        # back to DefaultAzureCredential for local az login / hosted MI flows.
        if (
            os.environ.get("AGENTKIT_WORKLOAD_IDENTITY_TOKEN")
            or os.environ.get("AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND")
        ):
            return _AsyncBearerTokenCredential(audience) if async_credential else _BearerTokenCredential(audience)
        try:
            if async_credential:
                identity_mod = importlib.import_module("azure.identity.aio")
            else:
                identity_mod = importlib.import_module("azure.identity")
            credential_type = getattr(identity_mod, "DefaultAzureCredential")
        except (ImportError, AttributeError) as exc:  # pragma: no cover - dependency guard.
            raise AgentBuildError("context workload identity auth requires azure-identity") from exc
        return credential_type()
    raise AgentBuildError(f"context provider auth type {auth.type!r} is not supported by the MAF runtime")


def _memory_update_delay() -> int:
    return positive_int_env("AGENTKIT_MEMORY_UPDATE_DELAY", default=0) or 0


def _session_cache_max() -> int:
    return positive_int_env("AGENTKIT_SESSION_CACHE_MAX", default=_DEFAULT_SESSION_CACHE_MAX) or _DEFAULT_SESSION_CACHE_MAX


def _memory_scope() -> str:
    value = os.environ.get("AGENTKIT_MEMORY_SCOPE")
    if not value:
        raise AgentBuildError(
            "memory context provider requires AGENTKIT_MEMORY_SCOPE; choose a per-user/session-safe scope"
        )
    return value


def _model_workload_api_key_provider(audience: str):
    async def _provider() -> str:
        explicit = os.environ.get("AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN")
        if explicit:
            return explicit
        return await asyncio.to_thread(resolve_workload_identity_token, audience)

    return _provider


def _uses_model_workload_identity_fallback(spec: AgentSpec) -> bool:
    auth = spec.model.auth
    return bool(
        auth is not None
        and auth.type == _AUTH_WORKLOAD_IDENTITY
        and not (
            os.environ.get("AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN")
            or os.environ.get("AGENTKIT_WORKLOAD_IDENTITY_TOKEN")
            or os.environ.get("AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND")
        )
    )


def _disable_mcp_ping(mcp_tool: object) -> None:
    if hasattr(mcp_tool, "_ping_available"):
        setattr(mcp_tool, "_ping_available", False)


async def _close_resource(resource: object) -> None:
    """Close a sync or async runtime-owned resource."""
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _validate_model_api(spec: AgentSpec) -> None:
    # The project-credential Foundry client is Responses-based despite its name.
    supported = {"responses"} if _uses_model_workload_identity_fallback(spec) else {"chat_completions"}
    resolve_model_api(supported=supported, runtime="Microsoft Agent Framework model client")


def build_client(spec: AgentSpec, *, workload_identity_credential: object | None = None):
    """Construct the chat client for the configured model auth mode."""
    _validate_model_api(spec)
    auth = spec.model.auth
    if auth is not None and auth.type == _AUTH_WORKLOAD_IDENTITY:
        if (
            os.environ.get("AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN")
            or os.environ.get("AGENTKIT_WORKLOAD_IDENTITY_TOKEN")
            or os.environ.get("AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND")
        ):
            return OpenAIChatCompletionClient(
                model=spec.model.name,
                base_url=spec.model.base_url,
                api_key=_model_workload_api_key_provider(auth.audience or _DEFAULT_FOUNDRY_AUDIENCE),
            )
        try:
            from agent_framework.foundry import FoundryChatClient
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise AgentBuildError(
                "model workload identity auth requires agent-framework-foundry and azure-identity"
            ) from exc
        credential = (
            workload_identity_credential
            if workload_identity_credential is not None
            else DefaultAzureCredential()
        )
        return FoundryChatClient(
            project_endpoint=_project_endpoint_from_openai_base_url(spec.model.base_url),
            model=spec.model.name,
            credential=credential,
        )

    return OpenAIChatCompletionClient(
        model=spec.model.name,
        base_url=spec.model.base_url,
        api_key=resolve_api_key(spec),
    )


def _tool_env(tool: ToolSpec) -> dict[str, str]:
    """Compatibility wrapper around the shared secret-safe tool env projection."""
    return declared_tool_env(tool)


class _MCPToolError(Exception):
    """A validated MCP result reports an admitted tool execution failure."""


class _OrkaToolError(_MCPToolError):
    """A broker result whose code and message come from the fixed allowlist."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class _MCPProtocolError(Exception):
    """An MCP call failed without an admitted tool error result."""

    def __init__(self, error: AgentRunError) -> None:
        super().__init__(str(error))
        self.error = error


class _MCPCallBoundary:
    """Keep MCP protocol failures out of MAF's recoverable tool-error loop."""

    async def call_tool(self, tool_name, **kwargs):
        try:
            return await super().call_tool(tool_name, **kwargs)
        except _MCPToolError:
            raise
        except Exception as exc:
            # Do not attach upstream exceptions: framework tool logging and
            # detailed-error options must never expose transport credentials.
            error = mcp_tool_protocol_error(exc)
            raise _MCPProtocolError(error) from None

    async def _call_tool_with_retries(self, tool_name, filtered_kwargs, meta, parser, span):
        # MAF 1.9+ retries tools/call after a lost connection. An attempted call
        # may already have executed, so leave retry decisions to the caller.
        result = await self.session.call_tool(tool_name, arguments=filtered_kwargs, meta=meta)
        return parser(result)

    def _parse_tool_result_from_mcp(self, result):
        if not isinstance(result, CallToolResult):
            raise _MCPProtocolError(mcp_tool_protocol_error())
        if result.isError:
            details = orka_tool_error_details(result.structuredContent)
            if details is not None:
                raise _OrkaToolError(*details)
            raise _MCPToolError("MCP tool execution failed")
        return super()._parse_tool_result_from_mcp(result)


class _MCPStreamableHTTPTool(_MCPCallBoundary, MCPStreamableHTTPTool):
    pass


class _MCPStdioTool(_MCPCallBoundary, MCPStdioTool):
    pass


def build_tool(tool: ToolSpec, *, stack: AsyncExitStack | None = None):
    """Create a stdio or Streamable HTTP MCP server for one tool spec."""
    timeout = _mcp_request_timeout()
    if tool.url_env:
        url = resolve_tool_url(tool)
        target_url = URL(url)
        target_origin = (target_url.scheme, target_url.host, target_url.port)
        remote_timeout = _remote_mcp_timeout()

        async def inject_headers(request):  # noqa: ANN001
            request_origin = (request.url.scheme, request.url.host, request.url.port)
            if request_origin != target_origin:
                return
            for key, value in (await asyncio.to_thread(resolve_tool_headers, tool)).items():
                request.headers[key] = value

        http_client = AsyncClient(
            event_hooks={"request": [inject_headers]},
            follow_redirects=False,
            timeout=remote_timeout,
        )
        if stack is not None:
            # MAF owns the MCP tool/session lifecycle, but the MCP SDK deliberately
            # leaves caller-supplied HTTP clients open. Register the client before
            # the Agent so LIFO cleanup disconnects the tool before closing HTTP.
            stack.push_async_callback(http_client.aclose)
        kwargs: dict[str, object] = {
            "name": tool.name,
            "url": url,
            "tool_name_prefix": tool.name,
            "load_prompts": False,
            "http_client": http_client,
        }
        kwargs["request_timeout"] = int(remote_timeout)
        mcp_tool = _MCPStreamableHTTPTool(**kwargs)
        # Some Streamable HTTP MCP services, including Foundry Toolbox, do not
        # implement MCP ping. The framework handles request-time connection
        # errors separately, so skip proactive pings for remote HTTP tools.
        _disable_mcp_ping(mcp_tool)
        return mcp_tool

    command, args = split_tool_command(tool, example='["uvx", "mcp-server-fetch"]')
    kwargs = {
        "name": tool.name,
        "command": command,
        "args": args,
        "env": declared_tool_env(tool),
        "tool_name_prefix": tool.name,
    }
    kwargs["request_timeout"] = timeout
    return _MCPStdioTool(**kwargs)


class _MCPFailureMiddleware(FunctionMiddleware):
    async def process(
        self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
    ) -> None:
        failure = _run_failure.get()
        if failure is not None and failure.error is not None:
            raise MiddlewareTermination(str(failure.error))
        try:
            await call_next()
        except _OrkaToolError as exc:
            context.metadata[_ORKA_TOOL_ERROR_CONTEXT_KEY] = exc
            if exc.code == "tool_outcome_unknown":
                _fail_run(str(exc), code=exc.code)
                raise MiddlewareTermination(str(exc)) from None
            # The SDK hides ordinary exception messages from the model. Project
            # only these fixed broker outcomes through its normal tool result.
            context.result = {"isError": True, "code": exc.code, "message": exc.message}
        except _MCPProtocolError as exc:
            _fail_run_with(exc.error)
            # MiddlewareTermination stops the model loop even on MAF 1.9,
            # which predates MiddlewareFailure. run_agent makes it a failure.
            raise MiddlewareTermination("MCP tool protocol failed") from None


# The current run's request history, which _RequestHistoryProvider loads.
_request_history: ContextVar[tuple[Message, ...]] = ContextVar("agentkit_maf_request_history", default=())


class _RequestHistoryProvider(HistoryProvider):
    """Make each run's request history the only conversation the model sees.

    MAF injects an in-memory history provider into session-backed runs unless a
    loading history provider is registered, and that stored copy would replace
    the history the protocol layer sends. Loading the request history here, not
    passing it as run input, also keeps context providers that persist input
    messages, such as Foundry memory, to the new turn.
    """

    def __init__(self) -> None:
        super().__init__("agentkit-request-history", store_inputs=False, store_outputs=False)

    async def get_messages(self, session_id, *, state=None, **kwargs) -> list[Message]:  # noqa: ANN001, ANN003
        return list(_request_history.get())

    async def save_messages(self, session_id, messages, *, state=None, **kwargs) -> None:  # noqa: ANN001, ANN003
        return None


class _ModelMessageMiddleware(ChatMiddleware):
    """Keep package and framework author labels out of model speaker fields."""

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        # AgentKit's inputs do not define named speakers. MAF adds its agent name
        # to history, but chat-to-Responses gateways cannot translate that field.
        context.messages = [copy.copy(message) for message in context.messages]
        for message in context.messages:
            message.author_name = None
        await call_next()


def build_agent(
    spec: AgentSpec,
    *,
    context_providers=None,
    stack: AsyncExitStack | None = None,
    client=None,
) -> Agent:
    """Assemble the MAF agent: client + system prompt + tools + context."""
    instructions = spec.instructions
    if client is None:
        _validate_model_api(spec)
    tools = [build_tool(t, stack=stack) for t in spec.tools]
    if spec._packaged_skill_catalog:
        skills = spec._packaged_skill_catalog
        instructions += "\n\n" + skills.instructions
        definition = skills.tool_schema()["function"]
        tools.append(FunctionTool(
            name=definition["name"],
            description=definition["description"],
            input_model=definition["parameters"],
            func=skills.load_skill,
            approval_mode="never_require",
        ))
    chat_client = client if client is not None else build_client(spec)
    return Agent(
        client=chat_client,
        instructions=instructions,
        name=spec.metadata.name,
        tools=tools,
        context_providers=[_RequestHistoryProvider(), *(context_providers or [])],
        middleware=[_ModelMessageMiddleware(), _MCPFailureMiddleware()],
        # Each run already carries the full request history. A client that
        # stores responses by default (the Foundry Responses API) would also
        # chain the stored conversation and repeat that history. Other
        # OpenAI-compatible servers may reject an unknown store field.
        default_options={"store": False} if getattr(chat_client, "STORES_BY_DEFAULT", False) else None,
    )


class MAFRuntime:
    """RuntimeSession Adapter around a Microsoft Agent Framework Agent."""

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec
        self.stack = AsyncExitStack()
        self.lifecycle = AsyncExitStackLifecycle(self.stack)
        self.agent: Agent | None = None
        self.sessions: dict[str, AgentSession] = {}
        self.session_locks: dict[str, asyncio.Lock] = {}
        # Claims cover both lock holders and queued waiters during eviction.
        self.session_claims: dict[str, int] = {}
        self.most_recent_session_id: str | None = None
        self.session_cache_max = _session_cache_max()

    async def __aenter__(self) -> RuntimeSession:
        async def start() -> RuntimeSession:
            _validate_model_api(self.spec)
            context_providers = await self._build_context_providers()
            client = await self._build_model_fallback_client()
            self.agent = build_agent(
                self.spec,
                context_providers=context_providers,
                stack=self.stack,
                client=client,
            )
            # Register before entering so a partially failed Agent.__aenter__ still
            # unwinds the Agent's own internal AsyncExitStack.
            self.stack.push_async_exit(self.agent)
            await self.agent.__aenter__()
            return self

        try:
            return await self.lifecycle.enter(start)
        except BaseException:
            self.agent = None
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        try:
            return await self.lifecycle.exit(exc_type, exc, tb)
        finally:
            self.agent = None

    async def run(self, request: RunRequest) -> RunResult:
        if self.agent is None:
            raise AgentBuildError("MAF runtime session is not initialized")
        session_id = request.session_id
        session, lock = self._session_for(session_id)
        if lock is None:
            return await run_agent(self.agent, request, session=session)
        assert session_id
        try:
            # Sessions keep context-provider state; the request still carries the
            # whole conversation, so runs in one session stay ordered.
            async with lock:
                self._touch_session(session_id)
                return await run_agent(self.agent, request, session=self.sessions[session_id])
        finally:
            self._release_session_claim(session_id)

    async def discard_session(self, session_id: str) -> None:
        """Discard framework state for an ACP prompt that did not commit."""

        lock = self.session_locks.get(session_id)
        if lock is None:
            return
        async with lock:
            if self.session_locks.get(session_id) is lock and session_id in self.sessions:
                self._reset_session(session_id)

    def _session_for(self, session_id: str | None) -> tuple[AgentSession | None, asyncio.Lock | None]:
        if not session_id:
            return None, None
        session = self.sessions.get(session_id)
        lock = self.session_locks.get(session_id)
        if session is None:
            session = AgentSession(session_id=session_id)
            self.sessions[session_id] = session
        if lock is None:
            lock = asyncio.Lock()
            self.session_locks[session_id] = lock
        self.session_claims[session_id] = self.session_claims.get(session_id, 0) + 1
        self._evict_idle_sessions()
        return session, lock

    def _touch_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id)
        lock = self.session_locks.pop(session_id)
        self.sessions[session_id] = session
        self.session_locks[session_id] = lock
        self.most_recent_session_id = session_id
        self._evict_idle_sessions()

    def _reset_session(self, session_id: str) -> None:
        self.sessions[session_id] = AgentSession(session_id=session_id)

    def _release_session_claim(self, session_id: str) -> None:
        claims = self.session_claims.get(session_id, 0)
        if claims <= 1:
            self.session_claims.pop(session_id, None)
            self._evict_idle_sessions()
        else:
            self.session_claims[session_id] = claims - 1

    def _evict_idle_sessions(self) -> None:
        for session_id in list(self.sessions):
            if len(self.sessions) <= self.session_cache_max:
                return
            if session_id == self.most_recent_session_id:
                continue
            if self.session_claims.get(session_id, 0) > 0:
                continue
            lock = self.session_locks.get(session_id)
            if lock is not None and lock.locked():
                continue
            self.sessions.pop(session_id, None)
            self.session_locks.pop(session_id, None)
            self.session_claims.pop(session_id, None)

    async def _enter_owned_async_context(self, resource):
        """Enter an adapter-owned async context with partial-enter cleanup."""
        enter = getattr(resource, "__aenter__", None)
        exit_ = getattr(resource, "__aexit__", None)
        if not callable(enter) or not callable(exit_):
            return resource
        self.stack.push_async_exit(resource)
        return await enter()

    async def _build_model_fallback_client(self):
        if not _uses_model_workload_identity_fallback(self.spec):
            return None
        resolve_model_api(supported={"responses"}, runtime="Microsoft Agent Framework Foundry-project credential path")
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise AgentBuildError(
                "model workload identity auth requires agent-framework-foundry and azure-identity"
            ) from exc

        credential = DefaultAzureCredential()
        if callable(getattr(credential, "close", None)):
            self.stack.push_async_callback(_close_resource, credential)
        client = build_client(self.spec, workload_identity_credential=credential)
        # FoundryChatClient exposes its internally-created AIProjectClient, but
        # MAF's Agent does not enter that project client. Own it here while leaving
        # the framework chat client itself exclusively under Agent ownership.
        project_client = getattr(client, "project_client", None)
        if callable(getattr(project_client, "__aenter__", None)) and callable(
            getattr(project_client, "__aexit__", None)
        ):
            await self._enter_owned_async_context(project_client)

        # Current FoundryChatClient is not an async context manager, while the
        # AsyncOpenAI client it creates is. Close that model HTTP pool after the
        # Agent but before the project client. If a future framework version owns
        # the chat client lifecycle, leave its internals exclusively to Agent.
        if not isinstance(client, AbstractAsyncContextManager):
            model_http_client = getattr(client, "client", None)
            if model_http_client is not None and model_http_client is not project_client:
                enter = getattr(model_http_client, "__aenter__", None)
                exit_ = getattr(model_http_client, "__aexit__", None)
                if callable(enter) and callable(exit_):
                    await self._enter_owned_async_context(model_http_client)
                elif callable(getattr(model_http_client, "close", None)):
                    self.stack.push_async_callback(_close_resource, model_http_client)
        return client

    async def _build_context_providers(self):
        providers = []
        for provider in self.spec.context.providers:
            if provider.type == _CONTEXT_TYPE_SEARCH:
                providers.append(await self._build_search_provider(provider))
            elif provider.type == _CONTEXT_TYPE_SKILLS:
                if provider.source == _CONTEXT_SOURCE_FILESYSTEM:
                    if self.spec._packaged_skill_catalog is None:
                        providers.append(SkillsProvider(FileSkillsSource(provider.path)))
                elif provider.source == _CONTEXT_SOURCE_MCP:
                    providers.append(await self._build_mcp_skills_provider(provider))
            elif provider.type == _CONTEXT_TYPE_MEMORY:
                providers.append(await self._build_memory_provider(provider))
        return providers or None

    async def _build_search_provider(self, provider: ContextProviderSpec):
        try:
            azure_mod = importlib.import_module("agent_framework.azure")
            search_provider = getattr(azure_mod, "AzureAISearchContextProvider")
        except (ImportError, AttributeError) as exc:
            raise AgentBuildError(
                "search context provider requires agent-framework-azure-ai-search in the MAF runtime"
            ) from exc

        endpoint = _env_required(provider.endpoint_env, field="context.providers[].endpointEnv")
        index = _env_required(provider.index_env, field="context.providers[].indexEnv")
        credential = _credential_for_context(
            provider,
            default_audience=_DEFAULT_SEARCH_AUDIENCE,
            async_credential=True,
        )
        if callable(getattr(credential, "close", None)):
            self.stack.push_async_callback(_close_resource, credential)
        # Agent invokes context providers but does not manage their async
        # lifecycle, so the runtime must close their internally-created clients.
        context_provider = search_provider(
            endpoint=endpoint,
            index_name=index,
            credential=credential,
        )
        return await self._enter_owned_async_context(context_provider)

    async def _build_memory_provider(self, provider: ContextProviderSpec):
        try:
            foundry_mod = importlib.import_module("agent_framework.foundry")
            memory_provider = getattr(foundry_mod, "FoundryMemoryProvider")
        except (ImportError, AttributeError) as exc:
            raise AgentBuildError(
                "memory context provider requires agent-framework-foundry in the MAF runtime"
            ) from exc

        endpoint = _env_required(provider.endpoint_env, field="context.providers[].endpointEnv")
        store_name = _env_required(provider.store_name_env, field="context.providers[].storeNameEnv")
        credential = _credential_for_context(provider, default_audience=_DEFAULT_FOUNDRY_AUDIENCE)
        if callable(getattr(credential, "close", None)):
            self.stack.push_async_callback(_close_resource, credential)
        # Entering the provider enters/closes its internally-created project
        # client. Keep the credential below it on the stack so it closes last.
        context_provider = memory_provider(
            source_id=provider.name or "memory",
            project_endpoint=endpoint,
            credential=credential,
            memory_store_name=store_name,
            scope=_memory_scope(),
            update_delay=_memory_update_delay(),
        )
        return await self._enter_owned_async_context(context_provider)

    async def _build_mcp_skills_provider(self, provider: ContextProviderSpec):
        tool = next((t for t in self.spec.tools if t.name == provider.tool_ref), None)
        if tool is None:
            raise AgentBuildError(f"skills provider references unknown toolRef {provider.tool_ref!r}")
        if not tool.url_env:
            raise AgentBuildError("MCP skills provider currently requires a streamable-http MCP toolRef")

        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        url = resolve_tool_url(tool)
        timeout = _remote_mcp_timeout()
        http_client_factory = same_origin_mcp_httpx_client_factory(
            tool,
            url,
            timeout=timeout,
        )
        # This compatibility API creates the AsyncClient from the factory inside
        # its own async context. Owning the transport context on our stack also
        # owns that client; registering it separately would double-close it.
        read, write, _ = await self.stack.enter_async_context(
            streamablehttp_client(url=url, httpx_client_factory=http_client_factory)
        )
        session = await self.stack.enter_async_context(
            ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=timeout),
            )
        )
        await session.initialize()
        return SkillsProvider(MCPSkillsSource(client=session))


def supports_brokered_read() -> bool:
    return offline_orka_echo_enabled()


def supports_brokered_write() -> bool:
    return offline_orka_echo_enabled()


def supports_brokered_coordination() -> bool:
    return offline_orka_echo_enabled()


def supports_acp_http_mcp() -> bool:
    return True


def supports_acp_packaged_skills() -> bool:
    return True


def build_runtime(spec: AgentSpec) -> RuntimeSession:
    """Build the runtime session consumed by the shared server."""
    if offline_orka_echo_enabled():
        return OfflineEchoRuntimeFactory().build_runtime(spec)
    return MAFRuntime(spec)


def _status_of(exc: Exception) -> int:
    """Compatibility wrapper around shared upstream status extraction."""
    return upstream_status_code(exc)


def _result_text(result: object) -> str:
    """Extract the final assistant text from a MAF ``AgentResponse``."""
    text = getattr(result, "text", None)
    return text if isinstance(text, str) else str(result)


def _usage_value(details: object, name: str, default: int | None = 0) -> int | None:
    get = getattr(details, "get", None)
    if callable(get):
        value = get(name, default)
    else:
        to_dict = getattr(details, "to_dict", None)
        if callable(to_dict):
            value = to_dict().get(name, default)
        else:
            value = getattr(details, name, default)
    return int(value) if value is not None else None


def _result_usage(result: object) -> dict[str, int]:
    """Map MAF ``usage_details`` to the OpenAI usage block (zeros if unknown)."""
    details = getattr(result, "usage_details", None)
    if details is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = _usage_value(details, "input_token_count", 0) or 0
    completion_tokens = _usage_value(details, "output_token_count", 0) or 0
    total = _usage_value(details, "total_token_count", None)
    total_tokens = total if total is not None else prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _history_messages(request: RunRequest) -> tuple[Message, ...]:
    """Map a neutral RunRequest's prior turns to MAF messages."""
    return tuple(
        Message(role=turn.role, contents=[turn.text])
        for turn in request.history
        if turn.role in FORWARDED_ROLES and turn.text
    )


class _ToolEventMiddleware(FunctionMiddleware):
    """Await payload-free observations around each actual tool invocation."""

    def __init__(self, observe: Callable[[ToolCallEvent], Awaitable[None]]) -> None:
        self.observe = observe

    async def _emit(self, event: ToolCallEvent) -> None:
        try:
            await self.observe(event)
        except Exception:
            # MAF absorbs ordinary function exceptions and continues the model
            # loop. Stop it, then fail run_agent, even on supported versions
            # predating MiddlewareFailure.
            _fail_run("tool lifecycle observer failed")
            raise MiddlewareTermination("tool lifecycle observer failed") from None

    async def process(
        self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
    ) -> None:
        call_id = uuid4().hex
        name = context.function.name
        await self._emit(ToolCallEvent(call_id, name, "in_progress"))
        try:
            await call_next()
        except Exception:
            await self._emit(ToolCallEvent(call_id, name, "failed"))
            raise
        status = "failed" if isinstance(context.metadata.get(_ORKA_TOOL_ERROR_CONTEXT_KEY), _OrkaToolError) else "completed"
        await self._emit(ToolCallEvent(call_id, name, status))


async def run_agent(
    agent: Agent,
    request: RunRequest,
    *,
    session: AgentSession | None = None,
) -> RunResult:
    """Run the MAF agent and return the neutral result shape."""
    messages = [Message(role="user", contents=[request.prompt])]
    kwargs = {}
    if request.on_tool_event is not None:
        kwargs["middleware"] = [_ToolEventMiddleware(request.on_tool_event)]
    failure = _RunFailure()
    token = _run_failure.set(failure)
    history_token = _request_history.set(_history_messages(request))

    async def execute():
        return await agent.run(messages, session=session, **kwargs)

    running = asyncio.create_task(execute())
    try:
        try:
            # MiddlewareTermination stops the next model step, but supported
            # MAF versions first join all calls in the current batch. Cancel
            # and join the SDK run so a fatal call also stops pending siblings.
            await asyncio.wait((running, failure.signal), return_when=asyncio.FIRST_COMPLETED)
            if failure.error is None:
                try:
                    result = await running
                except Exception as exc:  # noqa: BLE001 — normalized for the façade
                    raise normalize_agent_run_error(exc) from exc
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        if failure.error is not None:
            raise failure.error
        return RunResult(text=_result_text(result), usage=_result_usage(result))
    finally:
        failure.signal.cancel()
        _run_failure.reset(token)
        _request_history.reset(history_token)
