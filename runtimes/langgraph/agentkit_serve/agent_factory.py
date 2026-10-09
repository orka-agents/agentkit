"""Build a LangChain/LangGraph agent from a validated :class:`AgentSpec`.

This adapter keeps AgentKit's generic runtime boundary intact: it consumes the
same frozen ``/agent/agent.yaml`` ABI as the pydantic-ai and MAF adapters and
serves the same non-streaming OpenAI ``/v1/chat/completions`` façade through
``agentkit_serve_common``. LangGraph is used internally via LangChain's
``create_agent`` helper; arbitrary user-authored graphs and Foundry
``/responses``/``/invocations`` hosting are intentionally out of scope here.

Verified during implementation against the installed package set:

* ``langchain.agents.create_agent(model=..., tools=..., system_prompt=...)``
  returns a compiled LangGraph with ``ainvoke``.
* ``ChatOpenAI(model=..., base_url=..., api_key=...)`` is the generic
  OpenAI-compatible chat model client.
* ``MultiServerMCPClient.session(server_name, auto_initialize=False)`` plus
  ``load_mcp_tools(..., server_name=..., tool_name_prefix=True)`` keeps stdio MCP
  sessions open for the server lifespan and namespaces tool names.

THE LOCK-IN BOUNDARY: imports here are confined to LangChain/LangGraph/OpenAI/MCP
packages and ``agentkit_serve_common``. NEVER import Azure / Foundry hosting
packages from this generic runtime; a future Foundry mode should be a separate
adapter/target.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from datetime import timedelta
from types import TracebackType
from typing import Any
from uuid import UUID

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

from agentkit_serve_common.adapter_support import (
    FORWARDED_ROLES,
    AsyncExitStackLifecycle,
    AgentBuildError,
    declared_tool_env,
    mcp_tool_protocol_error,
    normalize_agent_run_error,
    positive_float_env,
    resolve_api_key,
    resolve_model_api,
    resolve_tool_url,
    same_origin_mcp_httpx_client_factory,
    split_tool_command,
    upstream_status_code,
)
from agentkit_serve_common.config import AgentSpec, ToolSpec
from agentkit_serve_common.conversation import RunRequest, ToolCallEvent
from agentkit_serve_common.runtime import (
    AgentRunError,
    OfflineEchoRuntimeFactory,
    RunResult,
    RuntimeSession,
    offline_orka_echo_enabled,
)

# Seconds to wait for a stdio MCP server's initialize handshake. A cold `uvx` or
# `npx` tool may download/install before speaking MCP, so match pydantic-ai's
# generous default and let operators tune via env.
_DEFAULT_MCP_INIT_TIMEOUT = 120.0
# Fixed tool results the model sees instead of upstream diagnostics.
_MCP_TOOL_ERROR = "MCP tool execution failed"
_INVALID_TOOL_ARGUMENTS = "tool call arguments must be a JSON object"


def _mcp_init_timeout() -> float:
    """MCP stdio init timeout (seconds), overridable via AGENTKIT_MCP_TIMEOUT."""
    return positive_float_env(default=_DEFAULT_MCP_INIT_TIMEOUT)


def _resolve_api_key(spec: AgentSpec) -> str:
    """Compatibility wrapper over the shared API-key resolver."""
    return resolve_api_key(spec)


def build_model(spec: AgentSpec) -> ChatOpenAI:
    """Construct the OpenAI-compatible chat model pointed at ``model.baseURL``."""
    resolve_model_api(supported={"chat_completions"}, runtime="LangGraph runtime")
    return ChatOpenAI(
        model=spec.model.name,
        base_url=spec.model.base_url,
        api_key=_resolve_api_key(spec),
        # LC_OUTPUT_VERSION must not override the supported model API.
        use_responses_api=False,
    )


def _tool_env(tool: ToolSpec) -> dict[str, str]:
    """Compatibility wrapper over the shared declared-only tool env helper."""
    return declared_tool_env(tool)


def build_mcp_connection(tool: ToolSpec) -> dict[str, Any]:
    """Convert an AgentKit tool declaration into a LangChain MCP connection."""
    timeout = _mcp_init_timeout()
    if tool.url_env:
        url = resolve_tool_url(tool)
        return {
            "transport": "streamable_http",
            "url": url,
            "timeout": timeout,
            "sse_read_timeout": timeout,
            "session_kwargs": {"read_timeout_seconds": timedelta(seconds=timeout)},
            "httpx_client_factory": same_origin_mcp_httpx_client_factory(tool, url, timeout=timeout),
        }

    command, args = split_tool_command(tool, example='["uvx", "mcp-server-fetch"]')
    return {
        "transport": "stdio",
        "command": command,
        "args": args,
        # Declared-only env, even when empty; never let the MCP SDK inherit the
        # model key or process env by omission.
        "env": _tool_env(tool),
        # MCP ClientSession exposes a read timeout as a timedelta. Use the same
        # operator knob for request/read waits that we use for initialize.
        "session_kwargs": {"read_timeout_seconds": timedelta(seconds=timeout)},
    }


class _MCPSessionBoundary:
    """End the run on MCP transport/protocol failures without upstream detail."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await self._session.call_tool(*args, **kwargs)
        except Exception as exc:
            raise mcp_tool_protocol_error(exc) from None


class _InvalidToolCallMiddleware(AgentMiddleware):
    """Answer tool calls whose arguments are not JSON so the model can retry.

    LangChain parks them in ``invalid_tool_calls``, which the agent loop would
    otherwise treat as a final (often empty) answer.
    """

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        message = state["messages"][-1]
        if not isinstance(message, AIMessage) or not message.invalid_tool_calls:
            return None
        update: dict[str, Any] = {
            "messages": [
                ToolMessage(
                    content=_INVALID_TOOL_ARGUMENTS,
                    tool_call_id=call["id"],
                    name=call.get("name") or "",
                    status="error",
                )
                for call in message.invalid_tool_calls
            ]
        }
        if not message.tool_calls:
            update["jump_to"] = "model"
        return update


class LangGraphRuntime:
    """Async lifespan wrapper around a compiled LangGraph agent.

    The shared FastAPI server enters this object once for the process lifespan,
    so stdio MCP sessions stay warm across requests and close on shutdown.
    """

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec
        self.stack = AsyncExitStack()
        self.lifecycle = AsyncExitStackLifecycle(self.stack)
        self.graph: Any | None = None
        self.client: MultiServerMCPClient | None = None

    async def __aenter__(self) -> RuntimeSession:
        async def start() -> RuntimeSession:
            model = build_model(self.spec)
            tools = await self._load_tools()
            self.graph = create_agent(
                model=model,
                tools=tools,
                system_prompt=self.spec.instructions,
                middleware=[_InvalidToolCallMiddleware()],
            )
            return self

        try:
            return await self.lifecycle.enter(start)
        except BaseException:
            self.graph = None
            self.client = None
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
            self.graph = None
            self.client = None

    async def run(self, request: RunRequest) -> RunResult:
        return await run_agent(self, request)

    async def _load_tools(self) -> list[Any]:
        if not self.spec.tools:
            return []

        connections = {tool.name: build_mcp_connection(tool) for tool in self.spec.tools}
        self.client = MultiServerMCPClient(connections, tool_name_prefix=True)

        tools: list[Any] = []
        for tool in self.spec.tools:
            session_cm = self.client.session(tool.name, auto_initialize=False)
            session = await self.stack.enter_async_context(session_cm)
            await asyncio.wait_for(session.initialize(), timeout=_mcp_init_timeout())
            loaded = await load_mcp_tools(
                _MCPSessionBoundary(session),
                server_name=tool.name,
                tool_name_prefix=True,
            )
            for loaded_tool in loaded:
                # Admitted MCP errors still return to the model so it can recover,
                # but as a fixed result: tool error text can carry upstream detail.
                loaded_tool.handle_tool_error = _MCP_TOOL_ERROR
            tools.extend(loaded)
        return tools


def validate_supported_spec(spec: AgentSpec) -> None:
    if spec.model.auth is not None:
        raise AgentBuildError("langgraph runtime does not support model.auth; use apiKeyEnv")
    if spec.context.providers:
        raise AgentBuildError("langgraph runtime does not support context providers")


def supports_brokered_read() -> bool:
    return offline_orka_echo_enabled()


def supports_brokered_write() -> bool:
    return offline_orka_echo_enabled()


def supports_brokered_coordination() -> bool:
    return offline_orka_echo_enabled()


def supports_acp_http_mcp() -> bool:
    return True


def build_runtime(spec: AgentSpec) -> RuntimeSession:
    """Build the runtime session consumed by the shared server."""
    if offline_orka_echo_enabled():
        return OfflineEchoRuntimeFactory().build_runtime(spec)
    validate_supported_spec(spec)
    return LangGraphRuntime(spec)


def build_agent(spec: AgentSpec) -> LangGraphRuntime:
    """Compatibility alias for wrappers that build an adapter runtime directly."""
    return build_runtime(spec)


def _to_messages(request: RunRequest) -> list[BaseMessage]:
    """Map a neutral RunRequest to LangChain messages.

    The agent's own ``spec.instructions`` is passed as ``system_prompt`` when the
    graph is created; do not duplicate it here.
    """
    messages: list[BaseMessage] = []
    for turn in request.history:
        if turn.role not in FORWARDED_ROLES or not turn.text:
            continue
        if turn.role == "system":
            messages.append(SystemMessage(content=turn.text))
        elif turn.role == "user":
            messages.append(HumanMessage(content=turn.text))
        elif turn.role == "assistant":
            messages.append(AIMessage(content=turn.text))
    messages.append(HumanMessage(content=request.prompt))
    return messages


def _status_of(exc: Exception) -> int:
    """Compatibility wrapper over shared upstream status unwrapping."""
    return upstream_status_code(exc)


def _state_messages(state: Any) -> list[Any]:
    """Extract the LangGraph ``messages`` list or raise a normalized run error."""
    messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
    if not isinstance(messages, list):
        raise AgentRunError(
            "agent run failed: LangGraph result did not contain a messages list",
            status=502,
            code="LangGraphResultError",
        )
    return messages


def _last_ai_message(state: Any) -> AIMessage:
    """Extract the final AIMessage from a LangGraph ``ainvoke`` state."""
    for msg in reversed(_state_messages(state)):
        if isinstance(msg, AIMessage):
            return msg
    raise AgentRunError(
        "agent run failed: LangGraph result did not contain an assistant message",
        status=502,
        code="LangGraphResultError",
    )


def _message_text(message: AIMessage) -> str:
    """Extract assistant text from LangChain message content.

    LangChain may surface content as a string or as a list of content blocks. Text
    blocks are joined; non-text blocks are ignored unless there is no text at all,
    in which case we fall back to ``str(content)`` for debuggability.
    """
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
                continue
            part_type = getattr(part, "type", None)
            part_text = getattr(part, "text", None)
            if part_type == "text" and part_text is not None:
                text_parts.append(str(part_text))
        if text_parts:
            return "".join(text_parts)
    return str(content)


def _message_usage(message: AIMessage) -> dict[str, int]:
    """Map one LangChain ``usage_metadata`` block to the OpenAI usage shape."""
    usage = getattr(message, "usage_metadata", None)
    get = getattr(usage, "get", None)
    if not callable(get):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt_tokens = int(get("input_tokens", 0) or 0)
    completion_tokens = int(get("output_tokens", 0) or 0)
    total = get("total_tokens", None)
    total_tokens = int(total) if total is not None else prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _state_usage(state: Any) -> dict[str, int]:
    """Aggregate token usage across every model call in a LangGraph run.

    A tool-using LangGraph agent can make multiple model calls: one AIMessage may
    request a tool, and a later AIMessage contains the final answer. Each message
    carries per-call ``usage_metadata`` when available, so the OpenAI facade should
    report the sum for the whole agent turn rather than only the final answer.
    """
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for msg in _state_messages(state):
        if not isinstance(msg, AIMessage):
            continue
        usage = _message_usage(msg)
        totals["prompt_tokens"] += usage["prompt_tokens"]
        totals["completion_tokens"] += usage["completion_tokens"]
        totals["total_tokens"] += usage["total_tokens"]
    return totals


class _ToolEventCallback(AsyncCallbackHandler):
    """Observe real tool execution without forwarding its payload or errors."""

    raise_error = True
    run_inline = True

    def __init__(self, observe: Callable[[ToolCallEvent], Awaitable[None]]) -> None:
        self.observe = observe
        self.names: dict[UUID, str] = {}

    async def _emit(self, event: ToolCallEvent) -> None:
        try:
            await self.observe(event)
        except Exception:
            # LangChain logs callback exceptions even when raise_error is set.
            raise AgentRunError("tool lifecycle observer failed") from None

    async def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        name = serialized.get("name") or "Tool call"
        self.names[run_id] = name
        await self._emit(ToolCallEvent(str(run_id), name, "in_progress"))

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        name = self.names.pop(run_id)
        # Handled MCP failures arrive as ToolMessage(status="error"), without
        # invoking on_tool_error, so a completed callback is not always success.
        failed = isinstance(output, ToolMessage) and output.status == "error"
        await self._emit(ToolCallEvent(str(run_id), name, "failed" if failed else "completed"))

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        name = self.names.pop(run_id)
        await self._emit(ToolCallEvent(str(run_id), name, "failed"))


async def run_agent(agent: LangGraphRuntime, request: RunRequest) -> RunResult:
    """Run the compiled LangGraph once and return a neutral ``RunResult``."""
    if agent.graph is None:
        raise AgentRunError("agent graph is not initialized", status=500, code="AgentNotInitialized")

    kwargs = {}
    if request.on_tool_event is not None:
        kwargs["config"] = {"callbacks": [_ToolEventCallback(request.on_tool_event)]}
    try:
        state = await agent.graph.ainvoke({"messages": _to_messages(request)}, **kwargs)
    except AgentRunError:
        raise
    except Exception as exc:  # noqa: BLE001 — normalized for the façade
        raise normalize_agent_run_error(exc) from exc

    msg = _last_ai_message(state)
    return RunResult(text=_message_text(msg), usage=_state_usage(state))
