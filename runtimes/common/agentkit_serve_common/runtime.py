"""The framework-neutral run contract shared by all AgentKit runtime adapters.

This is the seam that makes the OpenAI /v1 facade (``server.py``) framework-
agnostic: the server depends only on these neutral types and the
:class:`RuntimeFactory` protocol, never on a concrete agent framework. Each
adapter ships an ``agent_factory`` module that satisfies the protocol — that
module is the ONLY place a framework (pydantic-ai, agent-framework, …) is
imported, which is what keeps the lock-in boundary (plan §12) per-adapter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from .config import AgentSpec
from .conversation import RunRequest

OFFLINE_ORKA_ECHO_ENV = "AGENTKIT_ORKA_OFFLINE_ECHO"
OFFLINE_ORKA_DELEGATE_AGENT_ENV = "AGENTKIT_ORKA_OFFLINE_DELEGATE_AGENT"
_BROKERED_TOOL_OUTPUT_ABSENT = object()


def offline_orka_echo_enabled() -> bool:
    """Whether adapter factories should use the no-provider Orka/ACP echo runtime."""

    return os.environ.get("AGENTKIT_PROTOCOL", "").strip().lower() in {"acp", "orka"} and (
        os.environ.get(OFFLINE_ORKA_ECHO_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    )


def _offline_wait_task_name(prompt: str) -> str:
    marker = "wait_for_tasks:"
    if marker not in prompt:
        return ""
    parts = prompt.split(marker, 1)[1].strip().split()
    if not parts:
        return ""
    return parts[0].strip("`'\"")


class AgentRunError(Exception):
    """A runtime/model failure during a run, carrying an HTTP status.

    The server maps this to the OpenAI error envelope WITHOUT importing any
    framework or model-SDK type — keeping ``server.py`` framework-agnostic. The
    message and optional ``code`` are runtime-owned and reach clients verbatim, so
    they must never carry framework or upstream text; when ``code`` is ``None``,
    the server falls back to this class name (``AgentRunError``). ``fatal`` marks
    a runtime session that cannot serve another run, such as one whose stdio MCP
    subprocess exited.
    """

    def __init__(self, message: str, status: int = 502, code: str | None = None, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.fatal = fatal


class RuntimeHealth:
    """Liveness of a protocol skin's long-lived runtime session.

    Skins record each run failure; once a fatal one is seen, their health probes
    fail so the platform replaces the container instead of routing more requests
    to a runtime that cannot recover in place.
    """

    def __init__(self) -> None:
        self.failure_code: str | None = None

    @property
    def healthy(self) -> bool:
        return self.failure_code is None

    def record(self, exc: BaseException) -> None:
        if isinstance(exc, AgentRunError) and exc.fatal and self.failure_code is None:
            self.failure_code = exc.code or exc.__class__.__name__


@dataclass(frozen=True)
class RunResult:
    """Framework-neutral result of one agent run (what ``server.py`` reads)."""

    text: str
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BrokeredToolDefinition:
    """Safe tool schema a brokered runtime may request through Orka.

    This is intentionally schema-only: execution URLs, auth headers, Kubernetes
    Secret refs, bearer tokens, TxTokens, and other credential-bearing fields must
    never cross this Interface. Orka remains the broker and policy authority.
    """

    name: str
    description: str
    brokered_class: Literal["read", "write", "coordination"]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    schema_digest: str | None = None


@dataclass(frozen=True)
class BrokeredToolCall:
    """One framework-neutral request to execute an Orka-brokered tool."""

    tool_call_id: str
    name: str
    arguments: Mapping[str, Any]
    brokered_class: Literal["read", "write", "coordination"]


@dataclass(frozen=True, init=False)
class BrokeredToolResult:
    """Result returned by Orka after policy, approval, and tool execution.

    ``output_present`` mirrors Go ``json.RawMessage`` presence semantics so an
    omitted output remains distinct from a present JSON ``null`` value.
    """

    tool_call_id: str
    approved: bool
    output: Any
    error: Mapping[str, Any] | None
    output_present: bool

    def __init__(
        self,
        tool_call_id: str,
        approved: bool,
        output: Any = _BROKERED_TOOL_OUTPUT_ABSENT,
        error: Mapping[str, Any] | None = None,
    ) -> None:
        output_present = output is not _BROKERED_TOOL_OUTPUT_ABSENT
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "approved", approved)
        object.__setattr__(self, "output", None if not output_present else output)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "output_present", output_present)


@runtime_checkable
class ToolBroker(Protocol):
    """Callback owned by the Orka HTTP skin for brokered tool execution."""

    async def request_tool(self, call: BrokeredToolCall) -> BrokeredToolResult:
        ...


@runtime_checkable
class RuntimeSession(Protocol):
    """A live runtime Adapter session owned by the adapter implementation.

    The shared server enters this session once for the FastAPI lifespan and calls
    :meth:`run` for each normalized RunRequest. Framework-specific agent objects
    and lifecycle quirks stay behind this Interface.
    """

    async def __aenter__(self) -> "RuntimeSession":
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        ...

    async def run(self, request: RunRequest) -> RunResult:
        ...


@runtime_checkable
class BrokeredRuntimeSession(RuntimeSession, Protocol):
    """Optional runtime capability for Orka-governed brokered tools.

    Framework adapters must implement this before AgentKit advertises brokered
    Orka support. Direct AgentKit-owned tools should be disabled or clearly
    separated while a brokered run is active so Orka governance cannot be bypassed.
    """

    async def run_brokered(
        self,
        request: RunRequest,
        tools: Sequence[BrokeredToolDefinition],
        broker: ToolBroker,
    ) -> RunResult:
        ...


class OfflineEchoRuntime:
    """No-provider runtime used only for Orka conformance and offline demos."""

    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None

    async def run(self, request: RunRequest) -> RunResult:
        return RunResult(
            text=f"offline echo: {request.prompt}",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    async def run_brokered(
        self,
        request: RunRequest,
        tools: Sequence[BrokeredToolDefinition],
        broker: ToolBroker,
    ) -> RunResult:
        if not tools:
            raise AgentRunError("brokered run requires at least one Orka tool schema", status=400, code="NoBrokeredTools")
        by_name = {tool.name: tool for tool in tools}
        coordination_tool_names = {name for name, tool in by_name.items() if tool.brokered_class == "coordination"}
        if {"delegate_task", "wait_for_tasks"}.issubset(coordination_tool_names):
            return await self._run_offline_coordinator(request, by_name, broker)
        if "delegate_task" in coordination_tool_names:
            delegate_result = await self._request_offline_delegate(request, by_name["delegate_task"], broker)
            if delegate_result.error is not None:
                return RunResult(text=f"offline coordinator delegate error: {delegate_result.error}")
            return RunResult(text=f"offline coordinator delegate result: {delegate_result.output}")
        if "wait_for_tasks" in coordination_tool_names:
            task_name = _offline_wait_task_name(request.prompt)
            if task_name:
                wait_result = await broker.request_tool(
                    BrokeredToolCall(
                        tool_call_id="wait-call-1",
                        name="wait_for_tasks",
                        arguments={"tasks": [task_name], "timeout": "30s"},
                        brokered_class=by_name["wait_for_tasks"].brokered_class,
                    )
                )
                if wait_result.error is not None:
                    return RunResult(text=f"offline coordinator wait error: {wait_result.error}")
                return RunResult(text=f"offline coordinator waited for {task_name}; wait result: {wait_result.output}")
        tool = tools[0]
        result = await broker.request_tool(
            BrokeredToolCall(
                tool_call_id="tool-call-1",
                name=tool.name,
                arguments={"prompt": request.prompt},
                brokered_class=tool.brokered_class,
            )
        )
        if result.error is not None:
            return RunResult(text=f"offline brokered echo tool error: {result.error}")
        return RunResult(text=f"offline brokered echo: approved={result.approved}; output={result.output}")

    async def _request_offline_delegate(
        self,
        request: RunRequest,
        delegate_tool: BrokeredToolDefinition,
        broker: ToolBroker,
    ) -> BrokeredToolResult:
        delegate_agent = os.environ.get(OFFLINE_ORKA_DELEGATE_AGENT_ENV, "fibey-agentkit-worker").strip() or "fibey-agentkit-worker"
        return await broker.request_tool(
            BrokeredToolCall(
                tool_call_id="delegate-call-1",
                name="delegate_task",
                arguments={"agent": delegate_agent, "prompt": f"Offline child proof for: {request.prompt}"},
                brokered_class=delegate_tool.brokered_class,
            )
        )

    async def _run_offline_coordinator(
        self,
        request: RunRequest,
        tools_by_name: Mapping[str, BrokeredToolDefinition],
        broker: ToolBroker,
    ) -> RunResult:
        delegate_result = await self._request_offline_delegate(request, tools_by_name["delegate_task"], broker)
        if delegate_result.error is not None:
            return RunResult(text=f"offline coordinator delegate error: {delegate_result.error}")
        task_name = ""
        if delegate_result.output:
            task_name = str(delegate_result.output.get("taskName") or delegate_result.output.get("task_name") or "")
        if not task_name:
            raise AgentRunError("delegate_task result did not include taskName", status=502, code="DelegateResultMissingTaskName")

        wait_tool = tools_by_name["wait_for_tasks"]
        wait_result = await broker.request_tool(
            BrokeredToolCall(
                tool_call_id="wait-call-1",
                name="wait_for_tasks",
                arguments={"tasks": [task_name], "timeout": "2m"},
                brokered_class=wait_tool.brokered_class,
            )
        )
        if wait_result.error is not None:
            return RunResult(text=f"offline coordinator wait error: {wait_result.error}")
        return RunResult(text=f"offline coordinator delegated {task_name}; wait result: {wait_result.output}")


class OfflineEchoRuntimeFactory:
    """RuntimeFactory that never imports or calls a model/framework provider."""

    def supports_brokered_read(self) -> bool:
        return True

    def supports_brokered_write(self) -> bool:
        return True

    def supports_brokered_coordination(self) -> bool:
        return True

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:  # noqa: ARG002 - offline runtime is spec-independent.
        return OfflineEchoRuntime()


@runtime_checkable
class RuntimeFactory(Protocol):
    """The interface an adapter's ``agent_factory`` module must satisfy.

    ``server.py`` is handed a value of this shape (the adapter's module) and uses
    only ``build_runtime(spec)``, so it never imports a framework or touches raw
    framework agent lifecycle.
    """

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        ...
