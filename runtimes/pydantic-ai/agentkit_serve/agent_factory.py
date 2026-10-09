"""Build a pydantic-ai :class:`Agent` from a validated :class:`AgentSpec`.

Verified against modern pydantic-ai (1.107.x through 2.x). Key facts baked in here:

* Direct upstream connections use ``OpenAIChatModel`` or ``OpenAIResponsesModel``
  selected at startup; both take base_url/api_key from an ``OpenAIProvider``.
* stdio MCP servers are passed to the agent as ``toolsets`` (the old
  ``mcp_servers=`` kwarg is gone). pydantic-ai 1.x exposes ``MCPServerStdio``;
  pydantic-ai 2.x uses ``MCPToolset(StdioTransport(...))``.
* The agent is itself an async context manager: ``async with agent:`` starts the
  MCP subprocesses; that is the modern replacement for ``run_mcp_servers()``.

The agent OWNS its system prompt (spec.instructions) and its tools — request-side
tools are rejected by the server, never merged here.

This module is the ONLY framework-specific surface of the adapter. It exposes a
NEUTRAL run contract that ``agentkit_serve_common.server`` consumes —
``build_runtime`` (the :class:`RuntimeFactory` protocol) — so the
shared server imports nothing from ``pydantic_ai``. Cross-runtime invariants such
as API-key resolution, secret-safe tool env projection, MCP timeout parsing, and
error normalization live in ``agentkit_serve_common.adapter_support``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, AsyncIterable, AsyncIterator, cast

from openai import AsyncStream
from openai.types import responses
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.settings import ModelSettings

try:  # pydantic-ai 1.x
    from pydantic_ai.mcp import MCPServerStdio
except ImportError:  # pydantic-ai 2.x
    MCPServerStdio = None  # type: ignore[assignment]

try:
    from pydantic_ai.mcp import MCPToolset
except ImportError:  # pragma: no cover - older pydantic-ai without MCPToolset
    MCPToolset = None  # type: ignore[assignment]

try:
    from fastmcp.client.transports.stdio import StdioTransport
    from fastmcp.client.transports.http import StreamableHttpTransport
    from fastmcp.exceptions import ToolError
except ImportError:  # pragma: no cover - older dependency set without FastMCP transports
    StdioTransport = None  # type: ignore[assignment]
    StreamableHttpTransport = None  # type: ignore[assignment]
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
    OpenAIResponsesStreamedResponse,
)
from pydantic_ai.providers.openai import OpenAIProvider

from agentkit_serve_common.adapter_support import (
    FORWARDED_ROLES,
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
)
from agentkit_serve_common.agentsessions import ExecutionExchange, VerifiedAgentsessionsBinding
from agentkit_serve_common.config import AgentSpec, ToolSpec
from agentkit_serve_common.conversation import RunRequest, ToolCallEvent
from agentkit_serve_common.runtime import (
    OfflineEchoRuntimeFactory,
    RunResult,
    RuntimeSession,
    offline_orka_echo_enabled,
)

# Seconds to wait for stdio MCP initialization and each subsequent request.
# pydantic-ai's 5s initialization default is too tight for a COLD `uvx`/`npx`
# tool: the first launch resolves, downloads, and installs the server package
# before it speaks MCP. Default generously and let operators tune both phases.
_DEFAULT_MCP_INIT_TIMEOUT = 120.0


def _mcp_init_timeout() -> float:
    """MCP stdio init/read timeout, overridable via AGENTKIT_MCP_TIMEOUT."""
    return positive_float_env(default=_DEFAULT_MCP_INIT_TIMEOUT)


def validate_supported_spec(spec: AgentSpec) -> None:
    if spec.model.auth is not None:
        raise AgentBuildError("pydantic-ai runtime does not support model.auth; use apiKeyEnv")
    if spec.context.providers:
        raise AgentBuildError("pydantic-ai runtime does not support context providers")


class _ValidatedResponsesStream:
    """Validate decoded native events before Pydantic AI drops item statuses.

    The OpenAI SDK still owns SSE decoding, HTTP errors and transport cleanup.
    This proxy supplies the async-iteration/close interface its model consumes.
    """

    def __init__(
        self, source: AsyncStream[responses.ResponseStreamEvent], model: _CompletedResponsesModel,
    ) -> None:
        self._source = source
        self._iterator = aiter(source)
        self._model = model
        self._unfinished: set[int] = set()
        self.response = source.response

    def __aiter__(self) -> AsyncIterator[responses.ResponseStreamEvent]:
        return self

    async def __anext__(self) -> responses.ResponseStreamEvent:
        event = await anext(self._iterator)
        if event.type == "response.output_item.added":
            if getattr(event.item, "status", None) == "incomplete":
                self._model._raise_incomplete()
            # Omitted optional status is not evidence that an added item finished.
            self._unfinished.add(event.output_index)
        elif event.type == "response.output_item.done":
            self._model._require_finished_item(event.item)
            self._unfinished.discard(event.output_index)
        elif event.type in {"response.failed", "response.incomplete", "error"}:
            self._model._raise_incomplete()
        elif event.type == "response.completed":
            self._model._require_raw_completed(event.response)
            self._unfinished.difference_update(range(len(event.response.output)))
            if self._unfinished:
                self._model._raise_incomplete()
        return event

    async def close(self) -> None:
        await self._source.close()


class _CompletedResponsesModel(OpenAIResponsesModel):
    """Use the SDK's Responses model, but fail closed on unfinished output.

    Pydantic AI accepts Responses EOF and, in older supported versions, failed or
    incomplete terminal events as partial model output. Check before the agent
    can commit text or execute tools, matching the Chat stream policy. Raw
    output-item status checks run before the SDK loses that completion evidence.
    """

    def _raise_incomplete(self) -> None:
        raise ModelAPIError(
            model_name=self.model_name,
            message="Responses API response did not complete",
        )

    def _require_finished_item(self, item: Any) -> None:
        # Function-call and reasoning status fields are optional in the native
        # schema. Reject explicit unfinished evidence, not legitimate omission.
        if getattr(item, "status", None) in {"in_progress", "incomplete"}:
            self._raise_incomplete()

    def _require_raw_completed(self, response: responses.Response) -> None:
        if response.status != "completed":
            self._raise_incomplete()
        for item in response.output:
            self._require_finished_item(item)

    def _process_response(
        self,
        response: responses.Response,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self._require_raw_completed(response)
        return super()._process_response(response, model_settings, model_request_parameters)

    async def _process_streamed_response(
        self,
        response: AsyncStream[responses.ResponseStreamEvent],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        expected_model_name: str | None = None,
        expected_response_id: str | None = None,
    ) -> OpenAIResponsesStreamedResponse:
        # This SDK hook retains native decoded items on both supported dependency
        # versions. Its consumer needs only iteration, response and close.
        validated = cast(
            AsyncStream[responses.ResponseStreamEvent], _ValidatedResponsesStream(response, self),
        )
        return await super()._process_streamed_response(
            validated, model_settings, model_request_parameters,
            expected_model_name=expected_model_name, expected_response_id=expected_response_id,
        )

    def _require_completed(self, response: ModelResponse) -> None:
        if (response.provider_details or {}).get("finish_reason") != "completed":
            self._raise_incomplete()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await super().request(messages, model_settings, model_request_parameters)
        self._require_completed(response)
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        async with super().request_stream(
            messages, model_settings, model_request_parameters, run_context=run_context,
        ) as response:
            yield response
            self._require_completed(response.get())


def build_model(spec: AgentSpec) -> OpenAIChatModel | OpenAIResponsesModel:
    """Select the direct upstream API independently of the serving protocol."""
    model_api = resolve_model_api()
    provider = OpenAIProvider(
        base_url=spec.model.base_url,
        api_key=resolve_api_key(spec),
    )
    if model_api == "responses":
        # Each request carries authoritative history; never depend on upstream
        # response storage or a provider-managed conversation.
        return _CompletedResponsesModel(
            spec.model.name, provider=provider, settings={"openai_store": False},
        )
    # EOF alone must not commit partial text or execute partial tool calls.
    # Without this, pydantic-ai treats a stream that ends without a
    # finish_reason as 'stop'.
    return OpenAIChatModel(
        spec.model.name,
        provider=provider,
        profile={"openai_chat_streaming_requires_finish_reason": True},
    )


async def _process_mcp_tool_call(ctx: Any, call_tool: Any, name: str, args: dict[str, Any]) -> Any:
    try:
        return await call_tool(name, args)
    except ToolError:
        # FastMCP raises ToolError only for an admitted isError result. Preserve
        # model recovery for that case, without forwarding upstream diagnostics.
        raise ModelRetry("MCP tool execution failed") from None
    except ExceptionGroup as exc:
        # FastMCP can group completed tool errors during session teardown.
        # Mixed protocol/transport failures must still stop the run.
        _, remaining = exc.split(ToolError)
        if remaining is None:
            raise ModelRetry("MCP tool execution failed") from None
        raise mcp_tool_protocol_error(remaining) from None
    except Exception as exc:
        # Recent Pydantic AI versions also retry JSON-RPC errors by default.
        # Authorization, protocol and transport failures must end this run.
        raise mcp_tool_protocol_error(exc) from None


def build_tool_server(tool: ToolSpec) -> Any:
    """Create an MCP toolset for one stdio or Streamable HTTP tool spec."""
    timeout = _mcp_init_timeout()

    if tool.url_env:
        url = resolve_tool_url(tool)
        if MCPToolset is None or StreamableHttpTransport is None:
            raise AgentBuildError(
                f"tool {tool.name!r} requires streamable-http MCP, but this pydantic-ai "
                "build does not expose a Streamable HTTP MCP transport"
            )
        transport = StreamableHttpTransport(
            url,
            httpx_client_factory=same_origin_mcp_httpx_client_factory(tool, url, timeout=timeout),
        )
        return MCPToolset(
            transport,
            init_timeout=timeout,
            read_timeout=timeout,
            tool_error_behavior="error",
            process_tool_call=_process_mcp_tool_call,
        ).prefixed(tool.name)

    command, args = split_tool_command(tool, example='["npx", "-y", "..."]')

    # A single operator timeout bounds both cold initialization and later tool
    # calls, preventing a blocked stdio server from hanging a request forever.
    env = declared_tool_env(tool)

    if MCPServerStdio is not None:
        return MCPServerStdio(
            command=command,
            args=args,
            env=env,
            timeout=timeout,
            read_timeout=timeout,
            # tool_prefix namespaces tool names so two servers can't collide.
            tool_prefix=tool.name,
        )

    if MCPToolset is None or StdioTransport is None:
        raise AgentBuildError("this pydantic-ai build does not expose an MCP stdio transport")

    # pydantic-ai 2.x replaced MCPServerStdio with a transport + toolset pair.
    # Prefixing moved to Toolset.prefixed(...), preserving the same namespacing
    # interface as the 1.x tool_prefix argument.
    transport = StdioTransport(
        command=command,
        args=args,
        env=env,
        # Match the agent lifespan: when pydantic-ai exits the toolset context,
        # the stdio subprocess should be torn down instead of kept alive.
        keep_alive=False,
    )
    return MCPToolset(
        transport,
        init_timeout=timeout,
        read_timeout=timeout,
        tool_error_behavior="error",
        process_tool_call=_process_mcp_tool_call,
    ).prefixed(tool.name)


def build_agent(spec: AgentSpec) -> Agent:
    """Assemble the pydantic-ai agent: model + stdio MCP toolsets.

    The baked system prompt is sent per run by :class:`PydanticRuntime` instead of
    as pydantic-ai ``instructions``, which the OpenAI model inserts after any
    leading client system messages.
    """
    model = build_model(spec)
    toolsets = [build_tool_server(t) for t in spec.tools]
    return Agent(model, toolsets=toolsets)


class PydanticRuntime:
    """RuntimeSession Adapter around a pydantic-ai Agent."""

    def __init__(self, agent: Agent, instructions: str = "") -> None:
        self.agent = agent
        self.instructions = instructions

    async def __aenter__(self) -> RuntimeSession:
        await self.agent.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return await self.agent.__aexit__(exc_type, exc, tb)

    async def run(self, request: RunRequest) -> RunResult:
        return await run_agent(self.agent, request, instructions=self.instructions)


def supports_brokered_read() -> bool:
    return offline_orka_echo_enabled()


def supports_brokered_write() -> bool:
    return offline_orka_echo_enabled()


def supports_brokered_coordination() -> bool:
    return offline_orka_echo_enabled()


def supports_acp_http_mcp() -> bool:
    return (
        offline_orka_echo_enabled()
        or (MCPToolset is not None and StreamableHttpTransport is not None)
    )


def build_runtime(spec: AgentSpec) -> RuntimeSession:
    """Build the runtime session consumed by the shared server."""
    if offline_orka_echo_enabled():
        return OfflineEchoRuntimeFactory().build_runtime(spec)
    validate_supported_spec(spec)
    return PydanticRuntime(build_agent(spec), instructions=spec.instructions)


async def run_agentsessions(
    binding: VerifiedAgentsessionsBinding, request: RunRequest, exchange: ExecutionExchange,
) -> None:
    """Fresh text-only SDK execution bound exclusively to the local host bridge.

    The host bridge has a fixed Chat RPC contract. AGENTKIT_MODEL_API selects
    direct upstream connections only and does not change this mediated path.
    Config remains opaque on RunRequest; this policy does not interpret it.
    The controller owns model OUTPUT and usage, so return None, not RunResult.
    """
    import httpx
    from openai import AsyncOpenAI
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    from agentkit_serve_common.agentsessions.bridge import loopback_bridge

    # Unlike the other protocols' conversion, empty turns are authoritative.
    history = [
        ModelRequest(parts=[UserPromptPart(content=turn.text)])
        if turn.role == "user" else ModelResponse(parts=[TextPart(content=turn.text)])
        for turn in request.history
    ]
    prompt = request.prompt if exchange.input_count else None
    if prompt is None:
        # SDK run(None) may adopt an existing assistant response. An empty
        # request forces an invocation without manufacturing an empty user turn.
        history.append(ModelRequest(parts=[]))
    async with loopback_bridge(exchange) as local:
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as http:
            async with AsyncOpenAI(
                base_url=local.base_url, api_key=local.token,
                organization="", project="", http_client=http,
                # Explicit Authorization excludes ambient OPENAI_CUSTOM_HEADERS
                # overrides, including differently cased authorization names.
                default_headers={"Authorization": "Bearer " + local.token},
                # The host owns effect completion/cancellation, not an
                # unjournaled wall-clock read deadline. Bound loopback connect.
                max_retries=0, timeout=httpx.Timeout(None, connect=5),
            ) as client:
                model = OpenAIChatModel(
                    binding.spec.model.name,
                    provider=OpenAIProvider(openai_client=client),
                    profile={
                        "openai_chat_streaming_requires_finish_reason": True,
                        "openai_system_prompt_role": "system",
                    },
                )
                agent = Agent(model, instructions=binding.spec.instructions, retries=0)
                async with agent:
                    async with agent.iter(prompt, message_history=history) as run:
                        async for node in run:
                            if agent.is_call_tools_node(node):
                                # The full SDK model request has completed. The
                                # host already owns its validated text OUTPUT,
                                # including empty text. Do not enter framework
                                # output validation/retry or tool execution.
                                break
                        else:
                            raise AgentRunError("agentsessions model execution failed")
    return None


def _to_message_history(request: RunRequest, instructions: str = "") -> list:
    """Map a neutral RunRequest to a pydantic-ai message_history list.

    The agent's baked ``instructions`` lead, followed by prior conversation turns,
    so a client system message never precedes the agent's own system prompt.
    """
    # Imported lazily so config-only consumers don't pull the messages module.
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        UserPromptPart,
    )

    out: list = []
    if instructions:
        out.append(ModelRequest(parts=[SystemPromptPart(content=instructions)]))
    for turn in request.history:
        if not turn.text or turn.role not in FORWARDED_ROLES:
            continue
        if turn.role == "user":
            out.append(ModelRequest(parts=[UserPromptPart(content=turn.text)]))
        elif turn.role == "system":
            out.append(ModelRequest(parts=[SystemPromptPart(content=turn.text)]))
        elif turn.role == "assistant":
            out.append(ModelResponse(parts=[TextPart(content=turn.text)]))
    return out


def _result_text(result: object) -> str:
    """Extract the final assistant text from a pydantic-ai run result."""
    output = getattr(result, "output", None)
    return output if isinstance(output, str) else str(output)


def _result_usage(result: object) -> dict[str, int]:
    """Best-effort OpenAI usage block from the pydantic-ai run result (zeros if unknown)."""
    try:
        usage = result.usage
        # In current pydantic-ai ``usage`` is a property returning a RunUsage; in
        # older builds it was a method. Prefer the property value; only call it if
        # we got a bare callable WITHOUT the token attributes (the real method).
        if not hasattr(usage, "input_tokens") and callable(usage):
            usage = usage()
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    except Exception:
        prompt_tokens = completion_tokens = 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


async def run_agent(agent: Agent, request: RunRequest, *, instructions: str = "") -> RunResult:
    """Run the pydantic-ai agent and return the neutral result shape."""
    message_history = _to_message_history(request, instructions)
    run_options: dict[str, Any] = {"message_history": message_history}
    on_tool_event = request.on_tool_event
    if on_tool_event is not None:
        from pydantic_ai.messages import (
            AgentStreamEvent,
            FunctionToolCallEvent,
            FunctionToolResultEvent,
            RetryPromptPart,
        )

        async def observe_tools(_: Any, events: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in events:
                if isinstance(event, FunctionToolCallEvent):
                    await on_tool_event(
                        ToolCallEvent(event.part.tool_call_id, event.part.tool_name, "in_progress")
                    )
                elif isinstance(event, FunctionToolResultEvent):
                    failed = (
                        isinstance(event.part, RetryPromptPart)
                        or getattr(event.part, "outcome", "success") != "success"
                    )
                    await on_tool_event(
                        ToolCallEvent(
                            event.part.tool_call_id,
                            event.part.tool_name or "",
                            "failed" if failed else "completed",
                        )
                    )

        run_options["event_stream_handler"] = observe_tools
    try:
        result = await agent.run(request.prompt, **run_options)
    except Exception as exc:  # noqa: BLE001 — normalized for the façade
        raise normalize_agent_run_error(exc) from exc
    return RunResult(text=_result_text(result), usage=_result_usage(result))
