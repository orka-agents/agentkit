from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp import types
from mcp.shared.exceptions import McpError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from agentkit_serve import agent_factory
from agentkit_serve_common.config import ToolSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.runtime import AgentRunError


PRIVATE = "PRIVATE_MCP_DIAGNOSTIC_0ca0cb"


class CountingModel(TestModel):
    requests = 0

    async def request(self, *args, **kwargs):
        self.requests += 1
        return await super().request(*args, **kwargs)

    @asynccontextmanager
    async def request_stream(self, *args, **kwargs):
        self.requests += 1
        async with super().request_stream(*args, **kwargs) as response:
            yield response


def _failure(kind):
    if kind == "jsonrpc":
        return McpError(types.ErrorData(code=-32000, message=PRIVATE))
    if kind == "auth":
        return httpx.HTTPStatusError(
            PRIVATE,
            request=httpx.Request("POST", "https://example.invalid"),
            response=httpx.Response(403),
        )
    if kind == "timeout":
        return httpx.ReadTimeout(PRIVATE)
    if kind == "malformed":
        try:
            types.CallToolResult.model_validate({"content": PRIVATE})
        except ValueError as exc:
            return exc
    if kind == "grouped":
        return ExceptionGroup("private transport group", [_failure("jsonrpc")])
    if kind == "mixed":
        return ExceptionGroup(
            "private mixed group", [ToolError(PRIVATE), _failure("jsonrpc")]
        )
    raise AssertionError("unknown controlled failure")


def _setup(monkeypatch, outcomes, *, transport="streamable-http", tools=1, model=None):
    monkeypatch.setenv("TEST_MCP_URL", "http://example.invalid/mcp")
    spec = (
        ToolSpec(
            name="fixture", type="mcp", url_env="TEST_MCP_URL", transport=transport
        )
        if transport == "streamable-http"
        else ToolSpec(name="fixture", command=["unused-fixture"])
    )
    toolset = agent_factory.build_tool_server(spec)
    client = toolset.wrapped.client
    initialized = types.InitializeResult(
        protocolVersion="2025-11-25",
        capabilities=types.ServerCapabilities(tools=types.ToolsCapability()),
        serverInfo=types.Implementation(name="agentkit-mcp-fixture", version="1"),
    )

    async def enter(self):
        return self

    async def exit(self, *args):
        return None

    async def list_tools():
        return [
            types.Tool(
                name=f"probe_{index}", inputSchema={"type": "object", "properties": {}}
            )
            for index in range(tools)
        ]

    # Keep real FastMCP result parsing, Pydantic toolsets, and agent execution.
    # Substitute connection establishment and the remote MCP responses only.
    monkeypatch.setattr(Client, "__aenter__", enter)
    monkeypatch.setattr(Client, "__aexit__", exit)
    monkeypatch.setattr(Client, "initialize_result", property(lambda self: initialized))
    monkeypatch.setattr(
        Client,
        "session",
        property(
            lambda self: SimpleNamespace(
                _tool_output_schemas={},
                list_tools=list_tools,
            )
        ),
    )
    monkeypatch.setattr(client, "list_tools", list_tools)
    pending = iter(outcomes)
    calls = []

    async def call_tool_mcp(name, arguments, **kwargs):
        calls.append({"name": name, "arguments": arguments, "meta": kwargs.get("meta")})
        result = next(pending)
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return await result()
        return result

    monkeypatch.setattr(client, "call_tool_mcp", call_tool_mcp)
    model = model or CountingModel(custom_output_text="done")
    return Agent(model, toolsets=[toolset]), model, calls


def _success():
    return types.CallToolResult(
        content=[types.TextContent(type="text", text="héllo 世界 🌍")]
    )


@pytest.mark.parametrize("transport", ["streamable-http", "stdio"])
@pytest.mark.parametrize(
    "kind", ["jsonrpc", "auth", "timeout", "malformed", "grouped", "mixed"]
)
@pytest.mark.parametrize("observed", [True, False])
def test_protocol_failures_end_the_run_without_retry_or_model_continuation(
    monkeypatch, transport, kind, observed
):
    async def exercise():
        events = []

        async def observe(event):
            events.append(event)

        agent, model, calls = _setup(
            monkeypatch, [_failure(kind), _success()], transport=transport
        )
        async with agent:
            with pytest.raises(
                AgentRunError, match="MCP tool protocol failed"
            ) as caught:
                await agent_factory.run_agent(
                    agent,
                    RunRequest("fatal", on_tool_event=observe if observed else None),
                )
        assert len(calls) == 1 and model.requests == 1
        assert PRIVATE not in str(caught.value)
        assert [event.status for event in events] == (
            ["in_progress"] if observed else []
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("transport", ["streamable-http", "stdio"])
@pytest.mark.parametrize("observed", [True, False])
@pytest.mark.parametrize("grouped", [True, False])
def test_admitted_iserror_still_allows_model_recovery(
    monkeypatch, transport, observed, grouped
):
    async def exercise():
        events = []

        async def observe(event):
            events.append(event)

        admitted = types.CallToolResult(
            content=[types.TextContent(type="text", text=PRIVATE)], isError=True
        )
        if grouped:
            admitted = ExceptionGroup(
                "private group", [ExceptionGroup("nested group", [ToolError(PRIVATE)])]
            )
        model_requests = []

        def recover(messages, info):
            model_requests.append(messages)
            if len(model_requests) <= 3:
                return ModelResponse(parts=[
                    ToolCallPart("fixture_probe_0", {}, tool_call_id=f"call-{len(model_requests)}")
                ])
            return ModelResponse(parts=[TextPart("done")])

        async def recover_stream(messages, info):
            response = recover(messages, info)
            if isinstance(response.parts[0], ToolCallPart):
                yield {0: DeltaToolCall(name="fixture_probe_0", json_args="{}")}
            else:
                yield "done"

        agent, _model, calls = _setup(
            monkeypatch, [admitted, admitted, _success()], transport=transport,
            model=FunctionModel(recover, stream_function=recover_stream),
        )
        async with agent:
            result = await agent_factory.run_agent(
                agent,
                RunRequest("recover", on_tool_event=observe if observed else None),
            )
        assert result.text == "done" and len(calls) == 3 and len(model_requests) == 4
        assert [event.status for event in events] == (
            ["in_progress", "failed", "in_progress", "failed", "in_progress", "completed"] if observed else []
        )
        assert PRIVATE not in str(events)
        assert PRIVATE not in str(model_requests)

    asyncio.run(exercise())


def test_grouped_cancellation_is_not_converted_to_model_retry(monkeypatch):
    async def exercise():
        cancellation = BaseExceptionGroup(
            "cancelled transport", [ToolError(PRIVATE), asyncio.CancelledError()]
        )
        agent, model, calls = _setup(monkeypatch, [cancellation])
        async with agent:
            with pytest.raises(BaseExceptionGroup) as caught:
                await agent_factory.run_agent(agent, RunRequest("cancel"))
        assert caught.value.subgroup(asyncio.CancelledError) is not None
        assert len(calls) == 1 and model.requests == 1

    asyncio.run(exercise())


def test_parallel_protocol_failure_cancels_other_call_without_model_continuation(
    monkeypatch,
):
    async def exercise():
        started, stopped = asyncio.Event(), asyncio.Event()

        async def pending():
            started.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        async def fail():
            await started.wait()
            raise _failure("jsonrpc")

        async def observe(event):
            pass

        agent, model, calls = _setup(monkeypatch, [pending, fail], tools=2)
        async with agent:
            with pytest.raises(AgentRunError, match="MCP tool protocol failed"):
                await asyncio.wait_for(
                    agent_factory.run_agent(
                        agent, RunRequest("parallel", on_tool_event=observe)
                    ),
                    3,
                )
        assert len(calls) == 2 and model.requests == 1 and stopped.is_set()

    asyncio.run(exercise())


def test_cancellation_is_not_converted_to_model_retry(monkeypatch):
    async def exercise():
        started, stopped = asyncio.Event(), asyncio.Event()

        async def pending():
            started.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        agent, model, calls = _setup(monkeypatch, [pending])
        async with agent:
            run = asyncio.create_task(
                agent_factory.run_agent(agent, RunRequest("cancel"))
            )
            await asyncio.wait_for(started.wait(), 3)
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(run, 3)
        assert len(calls) == 1 and model.requests == 1 and stopped.is_set()

    asyncio.run(exercise())
