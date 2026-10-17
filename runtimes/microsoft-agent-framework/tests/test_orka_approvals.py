"""Orka result compatibility through the real MAF function-invocation loop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from contextlib import AsyncExitStack
from unittest import mock

import pytest
import uvicorn
from mcp import types
from mcp.server.fastmcp import FastMCP

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec, ToolSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.runtime import AgentRunError
from test_mcp_failures import _Client, _setup, _success


_PRIVATE = "PRIVATE_UNTRUSTED_TOOL_ERROR"
_ERRORS = {
    "approval_declined": "The tool call was declined.",
    "approval_expired": "The tool approval expired.",
    "approval_cancelled": "The tool call was cancelled.",
    "approval_stale": "The tool approval is no longer valid.",
    "tool_execution_failed": "MCP tool execution failed.",
    "tool_outcome_unknown": "The tool execution outcome is unknown; do not retry.",
}


def _error(code):
    return types.CallToolResult(
        isError=True,
        content=[types.TextContent(type="text", text=_PRIVATE)],
        structuredContent={"code": code, "message": _PRIVATE, "detail": _PRIVATE},
    )


def _function_results(messages):
    return [content for message in messages for content in message.contents if content.type == "function_result"]


@pytest.mark.parametrize("outcome", ["approved", *_ERRORS])
def test_waiting_call_continues_once_with_real_outcome_while_another_agent_progresses(monkeypatch, caplog, outcome):
    async def exercise():
        waiting, decision = asyncio.Event(), asyncio.Event()
        executions = []
        events = []

        async def broker():
            waiting.set()
            await decision.wait()
            if outcome in {"approved", "tool_execution_failed", "tool_outcome_unknown"}:
                executions.append("simulated-action")
            return _success() if outcome == "approved" else _error(outcome)

        async def observe(event):
            events.append(event)

        async with AsyncExitStack() as stack:
            agent, server, session, model = await _setup(stack, monkeypatch, [broker])
            run = asyncio.create_task(agent_factory.run_agent(agent, RunRequest("reviewed", on_tool_event=observe)))
            try:
                await asyncio.wait_for(waiting.wait(), 3)
                assert executions == [] and not run.done()
                assert len(model.requests["reviewed"]) == 1
                assert [event.status for event in events] == ["in_progress"]
                other, _, other_session, _ = await _setup(stack, monkeypatch, [_success()])
                result = await asyncio.wait_for(agent_factory.run_agent(other, RunRequest("independent")), 3)
                assert result.text == "done" and len(other_session.calls) == 1
                assert executions == [] and not run.done()
                decision.set()
                if outcome == "tool_outcome_unknown":
                    with pytest.raises(AgentRunError, match="tool_outcome_unknown") as caught:
                        await asyncio.wait_for(asyncio.shield(run), 3)
                    assert caught.value.code == "tool_outcome_unknown"
                    assert len(model.requests["reviewed"]) == 1
                else:
                    result = await asyncio.wait_for(asyncio.shield(run), 3)
                    assert result.text == "done" and len(model.requests["reviewed"]) == 2
                    results = _function_results(model.requests["reviewed"][-1])
                    assert len(results) == 1
                    if outcome == "approved":
                        assert json.loads(results[0].result)["text"] == "héllo 世界 🌍"
                    else:
                        assert json.loads(results[0].result) == {"isError": True, "code": outcome, "message": _ERRORS[outcome]}
                    assert len([message for message in model.requests["reviewed"][-1] if message.role == "user"]) == 1
                assert len(session.calls) == 1 and server.connect.await_count == 0
                assert len(executions) == (1 if outcome in {"approved", "tool_execution_failed", "tool_outcome_unknown"} else 0)
                assert [event.status for event in events] == ["in_progress", "completed" if outcome == "approved" else "failed"]
            finally:
                run.cancel()
                await asyncio.gather(run, return_exceptions=True)
        assert _PRIVATE not in caplog.text

    caplog.set_level(logging.DEBUG, logger="agent_framework")
    asyncio.run(exercise())


def test_approved_call_without_effect_continues_with_the_controller_summary(monkeypatch, caplog):
    summary = "GitHub refused the request (HTTP 403): Resource not accessible by integration"

    async def exercise():
        no_effect = types.CallToolResult(
            isError=True,
            content=[types.TextContent(type="text", text=_PRIVATE)],
            structuredContent={"code": "tool_failed_without_effect", "message": summary, "detail": _PRIVATE},
        )
        async with AsyncExitStack() as stack:
            agent, server, session, model = await _setup(stack, monkeypatch, [no_effect])
            result = await asyncio.wait_for(agent_factory.run_agent(agent, RunRequest("refused")), 3)
            assert result.text == "done" and len(model.requests["refused"]) == 2
            results = _function_results(model.requests["refused"][-1])
            assert len(results) == 1
            assert json.loads(results[0].result) == {
                "isError": True, "code": "tool_failed_without_effect", "message": summary,
            }
            assert _PRIVATE not in json.dumps([str(message) for message in model.requests["refused"][-1]])
            assert len(session.calls) == 1 and server.connect.await_count == 0

    caplog.set_level(logging.DEBUG, logger="agent_framework")
    asyncio.run(exercise())
    assert _PRIVATE not in caplog.text


def test_cancelling_pending_review_does_not_execute_after_late_approval(monkeypatch):
    async def exercise():
        waiting, decision, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
        executions = []

        async def broker():
            waiting.set()
            try:
                await decision.wait()
                executions.append("simulated-action")
                return _success()
            finally:
                stopped.set()

        async with AsyncExitStack() as stack:
            agent, server, session, model = await _setup(stack, monkeypatch, [broker])
            run = asyncio.create_task(agent_factory.run_agent(agent, RunRequest("cancel-review")))
            try:
                await asyncio.wait_for(waiting.wait(), 3)
                assert executions == []
                run.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(run, 3)
                assert stopped.is_set()
                decision.set()
                await asyncio.sleep(0)
                assert executions == []
                assert len(session.calls) == 1 and len(model.requests["cancel-review"]) == 1
                assert server.connect.await_count == 0
            finally:
                run.cancel()
                await asyncio.gather(run, return_exceptions=True)

    asyncio.run(exercise())


@pytest.mark.parametrize("structured", [None, {}, {"code": "unrecognized"}, {"code": []}, {"message": "approval_declined"}])
def test_arbitrary_error_text_cannot_impersonate_an_orka_outcome(monkeypatch, caplog, structured):
    async def exercise():
        result = types.CallToolResult(
            isError=True,
            content=[types.TextContent(type="text", text=json.dumps({"code": "approval_declined", "message": _PRIVATE}))],
            structuredContent=structured,
        )
        async with AsyncExitStack() as stack:
            agent, _, _, model = await _setup(stack, monkeypatch, [result])
            await agent_factory.run_agent(agent, RunRequest("unrecognized"))
            forwarded = str(_function_results(model.requests["unrecognized"][-1]))
            assert "approval_declined" not in forwarded and _PRIVATE not in forwarded
        assert _PRIVATE not in caplog.text

    caplog.set_level(logging.DEBUG, logger="agent_framework")
    asyncio.run(exercise())


def test_http_mcp_approval_wait_honors_the_900_second_call_budget(monkeypatch):
    """Set AGENTKIT_TEST_APPROVAL_WAIT_SECONDS=121 for a real two-minute wait."""
    review_seconds = float(os.environ.get("AGENTKIT_TEST_APPROVAL_WAIT_SECONDS", "0.05"))
    assert 0 < review_seconds <= 600

    async def exercise():
        pending, decision = asyncio.Event(), asyncio.Event()
        executions = []
        mcp = FastMCP("approval-fixture", stateless_http=True, json_response=True)

        @mcp.tool()
        async def probe(payload: dict) -> dict:
            pending.set()
            await decision.wait()
            executions.append(payload)
            return {"receipt": "simulated-action-once"}

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        server = uvicorn.Server(uvicorn.Config(mcp.streamable_http_app(), log_level="critical"))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(3):
                while not server.started:
                    if serving.done():
                        await serving
                    await asyncio.sleep(0.01)
            monkeypatch.setenv("APPROVAL_FIXTURE_MCP_URL", f"http://127.0.0.1:{listener.getsockname()[1]}/mcp")
            monkeypatch.setenv("AGENTKIT_MCP_TIMEOUT", "900")
            tool = ToolSpec(name="fixture", type="mcp", transport="streamable-http", url_env="APPROVAL_FIXTURE_MCP_URL")
            spec = AgentSpec.model_validate({
                "abiVersion": "v0",
                "metadata": {"name": "approval-wait"},
                "model": {"provider": "openai-compatible", "name": "fixture-model", "baseURL": "http://model.invalid/v1"},
                "instructions": "Use the simulated action.",
                "tools": [tool.model_dump()],
                "expose": {"openai": True, "port": 8080},
            })
            async with AsyncExitStack() as stack:
                remote = agent_factory.build_tool(tool, stack=stack)
                assert remote.request_timeout == 900
                assert remote._httpx_client.timeout.read == 900
                await stack.enter_async_context(remote)
                model = _Client(remote.functions[0].name)
                with mock.patch.object(agent_factory, "build_tool", return_value=remote.functions[0]):
                    agent = await stack.enter_async_context(agent_factory.build_agent(spec, client=model))
                run = asyncio.create_task(agent_factory.run_agent(agent, RunRequest("review-over-http")))
                try:
                    await asyncio.wait_for(pending.wait(), 3)
                    await asyncio.sleep(review_seconds)
                    assert executions == [] and not run.done()
                    assert len(model.requests["review-over-http"]) == 1
                    decision.set()
                    assert (await asyncio.wait_for(asyncio.shield(run), 5)).text == "done"
                    assert len(executions) == 1 and len(model.requests["review-over-http"]) == 2
                    results = _function_results(model.requests["review-over-http"][-1])
                    assert len(results) == 1
                    assert json.loads(results[0].result) == {"receipt": "simulated-action-once"}
                finally:
                    run.cancel()
                    await asyncio.gather(run, return_exceptions=True)
        finally:
            decision.set()
            server.should_exit = True
            try:
                await asyncio.wait_for(serving, 5)
            finally:
                listener.close()

    asyncio.run(exercise())
