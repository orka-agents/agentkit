"""END releases admission, without waiting for clients to drain the RPC to EOF."""

from __future__ import annotations

import asyncio

import grpc
import pytest
from agentkit_serve_common.agentsessions._generated import common_pb2 as c
from agentkit_serve_common.agentsessions._generated import harness_pb2 as h
from agentkit_serve_common.agentsessions.service import HarnessService
from agentkit_serve_common.runtime import RunResult
from test_agentsessions_protocol import binding_file as binding_file  # noqa: PLC0414
from test_agentsessions_protocol import protocol, text


async def until_end(stream):
    events = []
    while not events or events[-1].kind != c.EVENT_END:
        events.append(await asyncio.wait_for(anext(stream), 2))
    return events


@pytest.mark.parametrize("finish", ["close", "eof"])
@pytest.mark.parametrize("outcome,state,code", [
    ("runner-none", "FAILED", 12), ("unsupported", "FAILED", 12),
    ("negative-resume", "FAILED", 12), ("none", "COMPLETED", 0),
    ("output", "COMPLETED", 0), ("failure", "FAILED", 13),
    ("cancel", "CANCELED", 1), ("cancel-and-complete", "CANCELED", 1),
    ("reader-error", "FAILED", 13), ("reader-cancel", "CANCELED", 1),
    ("reader-cleanup-error", "FAILED", 13),
])
def test_end_releases_admission_before_eof_and_old_handler_cannot_release_next(
    binding_file, caplog, outcome, state, code, finish,
):
    async def check():
        entered, second_entered, cleaned = (asyncio.Event() for _ in range(3))
        readers_closed, streams = set(), []

        async def runner(binding, request):
            if request.turn_id == "second":
                second_entered.set()
                await asyncio.Future()
            if request.turn_id == "first":
                if outcome == "failure":
                    raise RuntimeError("PRIVATE-PROMPT-CONFIG-TOKEN")
                if outcome in {"cancel", "reader-error", "reader-cancel"}:
                    entered.set()
                    try:
                        await asyncio.Future()
                    finally:
                        cleaned.set()
                if outcome in {"output", "cancel-and-complete"}:
                    return RunResult(text="answer")

        async def frames(execution_id, start=None):
            try:
                yield h.ControllerFrame(execution_id=execution_id, start=start or h.Start())
                if execution_id == "first":
                    if outcome in {"cancel", "reader-error", "reader-cancel"}:
                        await entered.wait()
                    if outcome in {"cancel", "cancel-and-complete"}:
                        yield h.ControllerFrame(execution_id=execution_id, cancel=h.Cancel())
                    if outcome == "reader-error":
                        raise RuntimeError("PRIVATE-PROMPT-CONFIG-TOKEN")
                    if outcome == "reader-cancel":
                        raise asyncio.CancelledError("PRIVATE-PROMPT-CONFIG-TOKEN")
                await asyncio.Future()
            finally:
                readers_closed.add(execution_id)
                if execution_id == "first" and outcome == "reader-cleanup-error":
                    raise RuntimeError("PRIVATE-PROMPT-CONFIG-TOKEN")

        binding = protocol().load_verified_agentsessions_binding(binding_file[0])
        # Exercise a genuinely fresh default-None service, not a runner that returns None.
        service = HarnessService(binding) if outcome == "runner-none" else HarnessService(binding, runner=runner)
        first_start = h.Start(resume_from_seq=-1) if outcome == "negative-resume" else h.Start()
        if outcome == "unsupported":
            first_start.inputs.append(text(c, "tool", "unsupported"))
        first = service.Connect(frames("first", first_start), None)
        streams.append(first)
        consumer = None
        try:
            events = await until_end(first)
            assert (events[-1].end.state, events[-1].end.error.code) == (state, code)
            assert [event.kind for event in events] == ([c.EVENT_OUTPUT, c.EVENT_END] if outcome == "output" else [c.EVENT_END])
            assert "PRIVATE" not in str(events) + caplog.text
            assert not any(record.exc_info for record in caplog.records)
            if outcome in {"reader-error", "reader-cancel"}:
                assert cleaned.is_set()
                assert events[-1].end.error.description == ("execution failed" if outcome == "reader-error" else "execution canceled")
            if outcome == "reader-cleanup-error":
                assert events[-1].end.error.description == "execution cleanup failed"
                assert [record.getMessage() for record in caplog.records].count("agentsessions execution cleanup failed") == 1
            if outcome in {"none", "output", "failure", "reader-error", "reader-cancel", "reader-cleanup-error"}:
                assert "first" in readers_closed
            # Leave first suspended at END while a second runner owns admission.
            service.runner = runner
            second = service.Connect(frames("second"), None)
            streams.append(second)
            consumer = asyncio.create_task(anext(second))
            await asyncio.wait_for(second_entered.wait(), 2)
            if finish == "close":
                await first.aclose()
            else:
                with pytest.raises(StopAsyncIteration):
                    await anext(first)
            assert not service._idle.is_set()
            third = service.Connect(frames("third"), None)
            streams.append(third)
            assert (await until_end(third))[-1].end.error.code == 8
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            fourth = service.Connect(frames("fourth"), None)
            streams.append(fourth)
            assert (await until_end(fourth))[-1].end.state == "COMPLETED"
        finally:
            if consumer is not None:
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)
            for stream in streams:
                await stream.aclose()
    asyncio.run(check())


@pytest.mark.parametrize("owner", ["execution", "reader", "reader-error", "reader-cancel"])
@pytest.mark.parametrize("cleanup_raises", [False, True])
def test_repeated_cancel_does_not_interrupt_owned_cleanup(binding_file, caplog, owner, cleanup_raises):
    async def check():
        entered, reading, closing, release, cleaned, interrupted = (asyncio.Event() for _ in range(6))
        controls = asyncio.Queue()

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

        async def runner(binding, request):
            if request.turn_id == "first":
                if owner == "reader":
                    await reading.wait()
                    return
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    await close_resource()

        async def frames(execution_id):
            yield h.ControllerFrame(execution_id=execution_id, start=h.Start())
            try:
                reading.set()
                if execution_id == "first":
                    if owner == "execution":
                        yield await controls.get()
                    if owner in {"reader-error", "reader-cancel"}:
                        await entered.wait()
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
        consumer = asyncio.create_task(anext(first))
        try:
            if owner == "execution":
                await asyncio.wait_for(entered.wait(), 2)
                controls.put_nowait(h.ControllerFrame(execution_id="first", cancel=h.Cancel()))
            await asyncio.wait_for(closing.wait(), 2)
            assert not consumer.done()  # No END before resource teardown finishes.
            for _ in range(2):
                consumer.cancel()
                await asyncio.sleep(0)
                assert not consumer.done()
            blocked = service.Connect(frames("blocked"), None)
            assert (await until_end(blocked))[-1].end.error.code == 8
            await blocked.aclose()
            assert not interrupted.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await consumer
            assert cleaned.is_set()
            healthy = service.Connect(frames("healthy"), None)
            assert (await until_end(healthy))[-1].end.state == "COMPLETED"
            await healthy.aclose()
            assert [record.getMessage() for record in caplog.records].count("agentsessions execution cleanup failed") == int(cleanup_raises)
            assert "PRIVATE" not in caplog.text
            assert not any(record.exc_info for record in caplog.records)
        finally:
            release.set()
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            await first.aclose()
    asyncio.run(check())


def test_native_grpc_malformed_control_ends_once_without_private_diagnostics(binding_file, caplog):
    async def check():
        entered, cleaned = asyncio.Event(), asyncio.Event()

        async def runner(binding, request):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

        binding = protocol().load_verified_agentsessions_binding(binding_file[0])
        server = protocol().create_server(binding, runner=runner)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                connect = channel.stream_stream("/agentsessions.v1.Harness/Connect", response_deserializer=c.Event.FromString)
                call = connect(timeout=2)
                try:
                    await call.write(h.ControllerFrame(execution_id="first", start=h.Start()).SerializeToString())
                    await asyncio.wait_for(entered.wait(), 2)
                    # A truncated length-delimited protobuf, after a valid Start.
                    await call.write(b"\x0a\xffPRIVATE-PROMPT-CONFIG-TOKEN")
                    events = [event async for event in call]
                    assert [event.kind for event in events] == [c.EVENT_END]
                    assert (events[0].end.state, events[0].end.error.code) == ("FAILED", 13)
                    assert events[0].end.error.description == "execution failed"
                    assert cleaned.is_set()
                    assert "PRIVATE" not in str(events) + caplog.text
                    assert not any(record.exc_info for record in caplog.records)
                finally:
                    call.cancel()
        finally:
            await server.stop(0)
    asyncio.run(check())
