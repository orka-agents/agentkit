from __future__ import annotations

import asyncio
import traceback
from dataclasses import asdict

import pytest
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.messages import ModelResponse, ToolCallPart

from agentkit_serve import agent_factory
from agentkit_serve_common.conversation import RunRequest, ToolCallEvent
from agentkit_serve_common.runtime import AgentRunError


def test_real_tool_execution_awaits_start_and_result_observations():
    async def exercise() -> None:
        events: list[ToolCallEvent] = []
        order: list[str] = []

        async def echo() -> dict:
            order.append("side-effect")
            assert events[-1].status == "in_progress"
            return {"nested": ["héllo 世界 🌍", {"private-value": "not an event"}]}

        async def observe(event: ToolCallEvent) -> None:
            await asyncio.sleep(0)
            events.append(event)
            order.append(event.status)

        agent = Agent(TestModel(custom_output_text="done"), tools=[echo])
        async with agent:
            result = await agent_factory.run_agent(agent, RunRequest("run echo", on_tool_event=observe))
        assert result.text == "done"
        assert order == ["in_progress", "side-effect", "completed"]
        assert events[0].tool_call_id == events[1].tool_call_id
        assert [event.tool_name for event in events] == ["echo", "echo"]
        assert all(set(asdict(event)) == {"tool_call_id", "tool_name", "status"} for event in events)
        assert "private-value" not in str(events)

    asyncio.run(exercise())


def test_model_retry_emits_failed_tool_result_then_successful_recovery():
    async def exercise() -> None:
        events: list[ToolCallEvent] = []
        calls = 0

        async def recover() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ModelRetry("private error details must not enter tool events")
            return "private success payload"

        async def observe(event: ToolCallEvent) -> None:
            events.append(event)

        agent = Agent(TestModel(custom_output_text="recovered"), tools=[recover])
        async with agent:
            result = await agent_factory.run_agent(agent, RunRequest("recover", on_tool_event=observe))
        assert result.text == "recovered" and calls == 2
        assert [event.status for event in events] == ["in_progress", "failed", "in_progress", "completed"]
        assert events[0].tool_call_id == events[1].tool_call_id
        assert events[2].tool_call_id == events[3].tool_call_id
        assert "private" not in str(events)

    asyncio.run(exercise())


def test_cancellation_interrupts_tool_and_propagates_to_acp_owner():
    async def exercise() -> None:
        events: list[ToolCallEvent] = []
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def delay() -> str:
            started.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        async def observe(event: ToolCallEvent) -> None:
            events.append(event)

        agent = Agent(TestModel(custom_output_text="unreachable"), tools=[delay])
        async with agent:
            run = asyncio.create_task(agent_factory.run_agent(agent, RunRequest("delay", on_tool_event=observe)))
            await asyncio.wait_for(started.wait(), timeout=2)
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(run, timeout=2)
        assert stopped.is_set()
        assert [event.status for event in events] == ["in_progress"]

    asyncio.run(exercise())


def test_without_tool_observer_keeps_nonstreaming_run_contract():
    async def exercise() -> None:
        calls: list[dict] = []

        class PlainAgent:
            async def run(self, prompt, *, message_history):
                calls.append({"prompt": prompt, "history": message_history})
                return type("Result", (), {"output": "plain"})()

        result = await agent_factory.run_agent(PlainAgent(), RunRequest("ordinary HTTP request"))
        assert result.text == "plain"
        assert calls == [{"prompt": "ordinary HTTP request", "history": []}]

    asyncio.run(exercise())


def test_observer_failure_stops_tool_execution():
    async def exercise() -> None:
        calls: list[str] = []

        async def echo() -> str:
            calls.append("unexpected")
            return "unreachable"

        async def observe(event: ToolCallEvent) -> None:
            raise RuntimeError("private observer failure")

        agent = Agent(TestModel(custom_output_text="unreachable"), tools=[echo])
        async with agent:
            with pytest.raises(AgentRunError):
                await agent_factory.run_agent(agent, RunRequest("echo", on_tool_event=observe))
        assert calls == []

    asyncio.run(exercise())


@pytest.mark.parametrize("observed", [True, False])
def test_model_retry_exhaustion_has_a_safe_distinct_error_code(observed):
    async def exercise():
        calls = 0
        events = []

        async def needs_correction():
            nonlocal calls
            calls += 1
            raise ModelRetry("private validation details")

        async def observe(event):
            events.append(event)

        agent = Agent(TestModel(), tools=[needs_correction])
        async with agent:
            with pytest.raises(AgentRunError) as caught:
                await agent_factory.run_agent(
                    agent, RunRequest("test", on_tool_event=observe if observed else None)
                )
        assert calls == 2
        assert caught.value.code == "AgentRetryLimitExceeded"
        assert caught.value.status == 502
        assert str(caught.value) == "agent retry limit exceeded"
        assert not caught.value.fatal
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__
        assert "private" not in "".join(traceback.format_exception(caught.value))
        assert "private" not in str(events)

    asyncio.run(exercise())


def test_unclassified_unexpected_model_behavior_remains_a_runtime_failure():
    async def exercise():
        class BrokenAgent:
            async def run(self, *args, **kwargs):
                raise UnexpectedModelBehavior("private framework diagnostic")

        with pytest.raises(AgentRunError) as caught:
            await agent_factory.run_agent(BrokenAgent(), RunRequest("test"))
        assert caught.value.code == "AgentRunFailed"
        assert str(caught.value) == "agent run failed"

    asyncio.run(exercise())


@pytest.mark.parametrize("observed", [True, False])
def test_invalid_tool_argument_retry_exhaustion_is_a_quality_error(observed):
    async def exercise():
        requests = []
        events = []

        async def needs_number(quantity: int):
            raise AssertionError("invalid arguments must not execute the tool")

        def invalid_arguments(messages, info):
            requests.append(messages)
            return ModelResponse(parts=[ToolCallPart("needs_number", {"quantity": "private invalid value"})])

        async def invalid_arguments_stream(messages, info):
            invalid_arguments(messages, info)
            yield {0: DeltaToolCall(name="needs_number", json_args='{"quantity":"private invalid value"}')}

        async def observe(event):
            events.append(event)

        agent = Agent(
            FunctionModel(invalid_arguments, stream_function=invalid_arguments_stream), tools=[needs_number]
        )
        async with agent:
            with pytest.raises(AgentRunError) as caught:
                await agent_factory.run_agent(
                    agent, RunRequest("test", on_tool_event=observe if observed else None)
                )
        assert len(requests) == 2
        assert caught.value.code == "AgentRetryLimitExceeded"
        assert caught.value.status == 502
        assert str(caught.value) == "agent retry limit exceeded"
        assert not caught.value.fatal
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__
        assert "private" not in "".join(traceback.format_exception(caught.value))
        assert "private" not in str(events)

    asyncio.run(exercise())
