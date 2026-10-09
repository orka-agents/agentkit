"""END is the admission barrier; clients need not drain the stream to EOF."""

from __future__ import annotations

import asyncio

import pytest
from agentkit_serve_common.adapter_support import _wait_for_owner_task
from agentkit_serve_common.agentsessions._generated import common_pb2 as c
from agentkit_serve_common.agentsessions._generated import harness_pb2 as h
from agentkit_serve_common.agentsessions.service import HarnessService
from agentkit_serve_common.runtime import RunResult
from test_agentsessions_protocol import (
    binding_file as binding_file,  # noqa: PLC0414 -- pytest fixture re-export
)
from test_agentsessions_protocol import live, protocol, text


@pytest.mark.parametrize("finish", ["close", "eof"])
@pytest.mark.parametrize(
    "outcome,state,code",
    [
        ("none", "COMPLETED", 0),
        ("output", "COMPLETED", 0),
        ("failure", "FAILED", 13),
        ("cancel", "CANCELED", 1),
        ("cancel-and-complete", "CANCELED", 1),
        ("reader-error", "FAILED", 13),
        ("reader-cancel", "CANCELED", 1),
        ("unsupported", "FAILED", 12),
        ("negative-resume", "FAILED", 12),
        ("runner-none", "FAILED", 12),
        ("oversized", "FAILED", 8),
        ("invalid-text", "FAILED", 13),
    ],
)
def test_end_releases_admission_before_eof_without_releasing_next_owner(
    binding_file,
    caplog,
    outcome,
    state,
    code,
    finish,
):
    async def check():
        exchanges = {}
        execution_entered = asyncio.Event()
        execution_cleaned = asyncio.Event()
        readers_closed = set()
        streams = []

        async def runner(binding, request, exchange):
            exchanges[request.turn_id] = exchange
            if request.turn_id == "second":
                await exchange.call([text(c, "user", "hold the next reservation")])
                await asyncio.Future()
            if request.turn_id != "first":
                return None
            if outcome == "cancel-and-complete":
                return RunResult(text="control must win over this output")
            if outcome == "failure":
                raise RuntimeError("PRIVATE-PROMPT-CONFIG-TOKEN")
            if outcome in {"cancel", "reader-error", "reader-cancel"}:
                execution_entered.set()
                try:
                    await asyncio.Future()
                finally:
                    execution_cleaned.set()
            if outcome in {"output", "oversized", "invalid-text"}:
                value = {
                    "output": "answer",
                    "oversized": "x" * (4 * 1024 * 1024),
                    "invalid-text": "private-\ud800",
                }[outcome]
                return RunResult(text=value)
            return None

        async def frames(execution_id, start=None):
            try:
                yield h.ControllerFrame(
                    execution_id=execution_id, start=start or h.Start()
                )
                if execution_id == "first" and outcome in {
                    "cancel",
                    "cancel-and-complete",
                }:
                    if outcome == "cancel":
                        await execution_entered.wait()
                    yield h.ControllerFrame(
                        execution_id=execution_id, cancel=h.Cancel()
                    )
                if execution_id == "first" and outcome in {
                    "reader-error",
                    "reader-cancel",
                }:
                    await execution_entered.wait()
                    if outcome == "reader-cancel":
                        raise asyncio.CancelledError("PRIVATE-PROMPT-CONFIG-TOKEN")
                    raise RuntimeError("PRIVATE-PROMPT-CONFIG-TOKEN")
                await asyncio.Future()
            finally:
                readers_closed.add(execution_id)

        binding = protocol().load_verified_agentsessions_binding(binding_file[0])
        service = HarnessService(
            binding, runner=None if outcome == "runner-none" else runner
        )
        first_start = h.Start()
        if outcome == "unsupported":
            first_start.inputs.append(text(c, "tool", "unsupported"))
        elif outcome == "negative-resume":
            first_start.resume_from_seq = -1
        first = service.Connect(frames("first", first_start), None)
        streams.append(first)
        try:
            events = []
            while True:
                event = await asyncio.wait_for(anext(first), 2)
                events.append(event)
                if event.kind == c.EVENT_END:
                    break
            assert event.end.state == state
            assert event.end.error.code == code
            assert [item.kind for item in events] == (
                [c.EVENT_OUTPUT, c.EVENT_END] if outcome == "output" else [c.EVENT_END]
            )
            assert "PRIVATE" not in str(events)
            assert "PRIVATE" not in caplog.text
            assert not any(record.exc_info for record in caplog.records)
            if outcome in {"reader-error", "reader-cancel"}:
                assert execution_cleaned.is_set()
                assert event.end.error.description == (
                    "execution failed"
                    if outcome == "reader-error"
                    else "execution canceled"
                )

            # Do not resume/close the first generator until the next reservation
            # is proven live. END, not StopAsyncIteration, is the client barrier.
            service.runner = runner
            second = service.Connect(frames("second"), None)
            streams.append(second)
            next_event = await asyncio.wait_for(anext(second), 2)
            assert next_event.kind == c.EVENT_MODEL_CALL
            if outcome not in {"unsupported", "negative-resume", "runner-none"}:
                if outcome not in {"cancel", "cancel-and-complete"}:
                    assert "first" in readers_closed
                with pytest.raises(ValueError, match="unavailable"):
                    await exchanges["first"].call([text(c, "user", "after END")])
            if finish == "close":
                await first.aclose()
            else:
                with pytest.raises(StopAsyncIteration):
                    await anext(first)
            third = service.Connect(frames("third"), None)
            streams.append(third)
            rejected = await asyncio.wait_for(anext(third), 2)
            assert rejected.kind == c.EVENT_END
            assert rejected.end.state == "FAILED"
            assert rejected.end.error.code == 8
            await second.aclose()
            fourth = service.Connect(frames("fourth"), None)
            streams.append(fourth)
            admitted = await asyncio.wait_for(anext(fourth), 2)
            assert admitted.kind == c.EVENT_END
            assert admitted.end.state == "COMPLETED"
        finally:
            for stream in streams:
                await stream.aclose()

    asyncio.run(check())


@pytest.mark.parametrize("output", [False, True])
def test_reader_teardown_failure_is_a_safe_failed_end(binding_file, caplog, output):
    async def check():
        reader_entered = asyncio.Event()

        async def frames(execution_id):
            yield h.ControllerFrame(execution_id=execution_id, start=h.Start())
            try:
                reader_entered.set()
                await asyncio.Future()
            finally:
                if execution_id == "first":
                    raise RuntimeError("PRIVATE-PROMPT-CONFIG-TOKEN")

        async def runner(binding, request, exchange):
            await reader_entered.wait()
            return RunResult(text="answer") if output else None

        binding = protocol().load_verified_agentsessions_binding(binding_file[0])
        service = HarnessService(binding, runner=runner)
        first = service.Connect(frames("first"), None)
        second = service.Connect(frames("second"), None)
        try:
            events = []
            while True:
                event = await asyncio.wait_for(anext(first), 2)
                events.append(event)
                if event.kind == c.EVENT_END:
                    break
            assert [event.kind for event in events] == (
                [c.EVENT_OUTPUT, c.EVENT_END] if output else [c.EVENT_END]
            )
            assert event.end.state == "FAILED"
            assert event.end.error.code == 13
            assert event.end.error.description == "execution cleanup failed"
            assert "PRIVATE" not in str(events) + caplog.text
            assert not any(record.exc_info for record in caplog.records)
            assert [record.getMessage() for record in caplog.records].count(
                "agentsessions execution cleanup failed"
            ) == 1
            # The failed END still releases admission without waiting for EOF.
            next_events = []
            while True:
                following = await asyncio.wait_for(anext(second), 2)
                next_events.append(following)
                if following.kind == c.EVENT_END:
                    break
            assert next_events[-1].end.state == "COMPLETED"
        finally:
            await first.aclose()
            await second.aclose()

    asyncio.run(check())


@pytest.mark.parametrize(
    "owner", ["execution", "reader", "reader-error", "reader-cancel"]
)
@pytest.mark.parametrize("cleanup_raises", [False, True])
def test_repeated_handler_cancel_waits_for_owned_cleanup_before_releasing(
    binding_file,
    caplog,
    owner,
    cleanup_raises,
):
    async def check():
        reader_entered = asyncio.Event()
        execution_entered = asyncio.Event()
        closing = asyncio.Event()
        release = asyncio.Event()
        cleaned = asyncio.Event()
        interrupted = asyncio.Event()
        controls = asyncio.Queue()
        streams = []

        async def close_resource():
            closing.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                interrupted.set()
                raise
            cleaned.set()
            if cleanup_raises:
                raise RuntimeError("PRIVATE-PROMPT-CONFIG-TOKEN")

        async def runner(binding, request, exchange):
            if request.turn_id == "first":
                if owner == "reader":
                    await reader_entered.wait()
                    return
                execution_entered.set()
                try:
                    await asyncio.Future()
                finally:
                    await close_resource()

        async def frames(execution_id):
            yield h.ControllerFrame(execution_id=execution_id, start=h.Start())
            try:
                reader_entered.set()
                if execution_id == "first" and owner == "execution":
                    yield await controls.get()
                if execution_id == "first" and owner in {
                    "reader-error",
                    "reader-cancel",
                }:
                    await execution_entered.wait()
                    if owner == "reader-cancel":
                        raise asyncio.CancelledError("PRIVATE-PROMPT-CONFIG-TOKEN")
                    raise RuntimeError("PRIVATE-PROMPT-CONFIG-TOKEN")
                await asyncio.Future()
            finally:
                if execution_id == "first" and owner == "reader":
                    await close_resource()

        binding = protocol().load_verified_agentsessions_binding(binding_file[0])
        service = HarnessService(binding, runner=runner)
        first = service.Connect(frames("first"), None)
        streams.append(first)
        consumer = asyncio.create_task(anext(first))
        try:
            if owner == "execution":
                await asyncio.wait_for(execution_entered.wait(), 2)
                controls.put_nowait(
                    h.ControllerFrame(execution_id="first", cancel=h.Cancel())
                )
            await asyncio.wait_for(closing.wait(), 2)
            assert not consumer.done()  # No END while any owned cleanup is pending.
            for _ in range(2):
                consumer.cancel()
                await asyncio.sleep(0)
                assert not consumer.done()
            blocked = service.Connect(frames("blocked"), None)
            streams.append(blocked)
            rejected = await asyncio.wait_for(anext(blocked), 2)
            assert rejected.end.error.code == 8
            assert not interrupted.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await consumer
            assert cleaned.is_set()
            healthy = service.Connect(frames("healthy"), None)
            streams.append(healthy)
            admitted = await asyncio.wait_for(anext(healthy), 2)
            assert admitted.end.state == "COMPLETED"
            messages = [record.getMessage() for record in caplog.records]
            assert messages.count("agentsessions execution cleanup failed") == int(
                cleanup_raises
            )
            assert not any(record.exc_info for record in caplog.records)
            assert "PRIVATE-PROMPT-CONFIG-TOKEN" not in caplog.text
        finally:
            release.set()
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            for stream in streams:
                await stream.aclose()

    asyncio.run(check())


@pytest.mark.parametrize("outcome", ["none", "unsupported", "runner-none", "cancel"])
def test_native_grpc_next_start_after_end_does_not_wait_for_eof(
    binding_file,
    monkeypatch,
    outcome,
):
    async def check():
        release_eof = asyncio.Event()
        second_entered = asyncio.Event()
        first_entered = asyncio.Event()
        original_connect = HarnessService.Connect

        async def paused_connect(self, frames, context):
            stream = original_connect(self, frames, context)
            paused = False
            try:
                async for event in stream:
                    paused = event.execution_id == "first" and event.kind == c.EVENT_END
                    yield event
                    if paused:
                        await release_eof.wait()
            finally:
                try:
                    if paused:
                        # Keep the original generator at END even when native
                        # gRPC cancels its wrapper before the next Start arrives.
                        await _wait_for_owner_task(
                            asyncio.create_task(release_eof.wait())
                        )
                finally:
                    await stream.aclose()

        monkeypatch.setattr(HarnessService, "Connect", paused_connect)

        async def runner(binding, request, exchange):
            if request.turn_id == "first" and outcome == "cancel":
                first_entered.set()
                await asyncio.Future()
            if request.turn_id == "second":
                second_entered.set()
                await exchange.call([text(c, "user", "next")])

        async with live(binding_file, None if outcome == "runner-none" else runner) as (
            _,
            _,
            _,
            stub,
            _,
        ):
            first = stub.Connect()
            second = stub.Connect()
            third = stub.Connect()
            try:
                first_start = h.Start()
                if outcome == "unsupported":
                    first_start.inputs.append(text(c, "tool", "unsupported"))
                await first.write(
                    h.ControllerFrame(execution_id="first", start=first_start)
                )
                if outcome == "cancel":
                    await asyncio.wait_for(first_entered.wait(), 2)
                    await first.write(
                        h.ControllerFrame(execution_id="first", cancel=h.Cancel())
                    )
                end = await asyncio.wait_for(first.read(), 2)
                assert end.kind == c.EVENT_END
                assert end.end.state == (
                    "CANCELED"
                    if outcome == "cancel"
                    else "FAILED"
                    if outcome in {"unsupported", "runner-none"}
                    else "COMPLETED"
                )
                # Mirror ClientHarness.Run: read END, cancel, immediately Start.
                first.cancel()
                await second.write(
                    h.ControllerFrame(execution_id="second", start=h.Start())
                )
                next_event = await asyncio.wait_for(second.read(), 2)
                assert next_event.kind == (
                    c.EVENT_END if outcome == "runner-none" else c.EVENT_MODEL_CALL
                )
                if outcome == "runner-none":
                    assert next_event.end.error.code == 12  # Not RESOURCE_EXHAUSTED.
                else:
                    assert second_entered.is_set()
                    await third.write(
                        h.ControllerFrame(execution_id="third", start=h.Start())
                    )
                    rejected = await asyncio.wait_for(third.read(), 2)
                    assert rejected.kind == c.EVENT_END
                    assert rejected.end.error.code == 8
            finally:
                release_eof.set()
                first.cancel()
                second.cancel()
                third.cancel()

    asyncio.run(check())
