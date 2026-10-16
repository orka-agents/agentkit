"""Admission and protected cleanup observed through the SDK and runtime seam."""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

pytest.importorskip("a2a")
from a2a.client.errors import A2AClientJSONRPCError
from a2a.types import Task, TaskIdParams, TaskQueryParams, TaskState, TaskStatus
from a2a.utils.errors import ServerError
from agentkit_serve_common.runtime import AgentRunError, RunResult
from test_a2a_client import connect, live_server, message
from test_a2a_protocol import Factory, Runtime, app, params


class BlockingRuntime(Runtime):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.cleanup = threading.Event()
        self.release_run = threading.Event()
        self.release_cleanup = threading.Event()
        self.cleaned = False

    async def run(self, request):
        if request.prompt != "block":
            return await super().run(request)
        self.requests.append(request)
        self.started.set()
        try:
            while not self.release_run.is_set():
                await asyncio.sleep(0.005)
            return RunResult(text="finished block")
        finally:
            self.cleanup.set()
            while not self.release_cleanup.is_set():
                await asyncio.sleep(0.005)
            self.cleaned = True


async def wait(event):
    assert await asyncio.to_thread(event.wait, 3), "runtime phase not reached"


def test_cancel_fanout_resubscribe_cleanup_and_immediate_next_send():
    runtime = BlockingRuntime()
    with live_server(app(Factory(runtime))) as url:

        async def exercise():
            try:
                async with httpx.AsyncClient(timeout=5) as http:
                    client = await connect(url, http)
                    stream = client.send_message(message("block"))
                    initial = await anext(stream)
                    task = initial[0]
                    await wait(runtime.started)
                    subscriber = client.resubscribe(TaskIdParams(id=task.id))
                    snapshot = await anext(subscriber)
                    assert snapshot[0].id == task.id
                    cancel = asyncio.create_task(
                        client.cancel_task(TaskIdParams(id=task.id))
                    )
                    await wait(runtime.cleanup)
                    assert not cancel.done() and not runtime.cleaned
                    # Supplied active task IDs must never spawn a second producer.
                    with pytest.raises(A2AClientJSONRPCError) as named:
                        async for _ in client.send_message(
                            message("duplicate", task_id=task.id)
                        ):
                            pass
                    assert named.value.error.code == -32602
                    with pytest.raises(A2AClientJSONRPCError):
                        async for _ in client.send_message(message("overlap")):
                            pass
                    assert len(runtime.requests) == 1
                    assert (
                        await client.get_task(TaskQueryParams(id=task.id))
                    ).status.state == TaskState.working
                    runtime.release_cleanup.set()
                    canceled = await asyncio.wait_for(cancel, 3)
                    assert (
                        runtime.cleaned and canceled.status.state == TaskState.canceled
                    )
                    # Cancel response guarantees capacity is available, not merely requested.
                    following = [
                        event
                        async for event in client.send_message(
                            message("next", context_id=task.context_id)
                        )
                    ]
                    assert following[-1][0].status.state == TaskState.completed
                    assert runtime.requests[-1].history == ()
                    original = [event async for event in stream]
                    resumed = [event async for event in subscriber]
                    assert (
                        original[-1][0].status.state
                        == resumed[-1][0].status.state
                        == TaskState.canceled
                    )
                    assert original[-1][1].final and resumed[-1][1].final
                    assert (
                        await client.get_task(TaskQueryParams(id=task.id))
                    ).status.state == TaskState.canceled
            finally:
                runtime.release_run.set()
                runtime.release_cleanup.set()

        asyncio.run(exercise())


def test_stream_disconnect_does_not_cancel_and_nonblocking_poll_works():
    runtime = BlockingRuntime()
    with live_server(app(Factory(runtime))) as url:

        async def exercise():
            try:
                async with httpx.AsyncClient(timeout=5) as http:
                    client = await connect(url, http)
                    stream = client.send_message(message("block"))
                    task = (await anext(stream))[0]
                    await wait(runtime.started)
                    await stream.aclose()
                    assert not runtime.cleanup.is_set()
                    subscriber = client.resubscribe(TaskIdParams(id=task.id))
                    await anext(subscriber)
                    # Abandoned tapped queues must not prevent producer cleanup.
                    await subscriber.aclose()
                    runtime.release_run.set()
                    runtime.release_cleanup.set()
                    for _ in range(100):
                        fetched = await client.get_task(TaskQueryParams(id=task.id))
                        if fetched.status.state == TaskState.completed:
                            break
                        await asyncio.sleep(0.01)
                    assert (
                        fetched.status.state == TaskState.completed and runtime.cleaned
                    )
                    polling = await connect(url, http, streaming=False, polling=True)
                    results = [
                        event async for event in polling.send_message(message("next"))
                    ]
                    assert results[0][0].id != task.id
            finally:
                runtime.release_run.set()
                runtime.release_cleanup.set()

        asyncio.run(exercise())


def test_cancel_without_live_producer_refuses_instead_of_hanging():
    application = app()

    async def exercise():
        async with application.router.lifespan_context(application):
            handler = application.state.a2a_handler
            await handler.task_store.publish(
                Task(
                    id="orphan",
                    context_id="context",
                    status=TaskStatus(state=TaskState.working),
                )
            )
            with pytest.raises(ServerError) as error:
                await asyncio.wait_for(
                    handler.on_cancel_task(TaskIdParams(id="orphan")), 1
                )
            assert error.value.error.code == -32002

    asyncio.run(exercise())


def test_outer_producer_cancellation_cannot_cancel_owned_cleanup():
    runtime = BlockingRuntime()
    application = app(Factory(runtime))

    async def exercise():
        async with application.router.lifespan_context(application):
            handler = application.state.a2a_handler
            from a2a.types import MessageSendParams

            send = asyncio.create_task(
                handler.on_message_send(
                    MessageSendParams.model_validate(params("block"))
                )
            )
            try:
                await wait(runtime.started)
                task_id, producer = next(iter(handler._running_agents.items()))
                producer.cancel()
                await asyncio.sleep(0.02)
                assert not runtime.cleanup.is_set()
                cancel = asyncio.create_task(
                    handler.on_cancel_task(TaskIdParams(id=task_id))
                )
                await wait(runtime.cleanup)
                cancel.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await cancel
                assert not runtime.cleaned
                runtime.release_cleanup.set()
                result = await asyncio.wait_for(send, 3)
                assert result.status.state == TaskState.canceled and runtime.cleaned
                assert (
                    await handler.task_store.get(task_id)
                ).status.state == TaskState.canceled
            finally:
                runtime.release_run.set()
                runtime.release_cleanup.set()

    asyncio.run(exercise())


def test_repeated_cancel_does_not_recancel_cleanup_and_queue_settles():
    runtime = BlockingRuntime()
    application = app(Factory(runtime))

    async def exercise():
        async with application.router.lifespan_context(application):
            handler = application.state.a2a_handler
            from a2a.types import MessageSendParams

            send = asyncio.create_task(
                handler.on_message_send(
                    MessageSendParams.model_validate(params("block"))
                )
            )
            try:
                await wait(runtime.started)
                task_id = next(iter(handler._running_agents))
                queue = await handler._queue_manager.get(task_id)
                first = asyncio.create_task(
                    handler.on_cancel_task(TaskIdParams(id=task_id))
                )
                await wait(runtime.cleanup)
                second = asyncio.create_task(
                    handler.on_cancel_task(TaskIdParams(id=task_id))
                )
                await asyncio.sleep(0.02)
                assert not runtime.cleaned and not first.done() and not second.done()
                runtime.release_cleanup.set()
                results = await asyncio.wait_for(asyncio.gather(first, second, send), 3)
                assert all(
                    result.status.state == TaskState.canceled for result in results
                )
                assert queue.is_closed() and all(
                    child.is_closed() for child in queue._children
                )
                assert await handler._queue_manager.get(task_id) is None
            finally:
                runtime.release_run.set()
                runtime.release_cleanup.set()

    asyncio.run(exercise())


def test_retention_does_not_evict_live_task(monkeypatch):
    from agentkit_serve_common.a2a import core

    monkeypatch.setattr(core, "MAX_TASKS", 1)
    runtime = BlockingRuntime()
    application = app(Factory(runtime))

    async def exercise():
        async with application.router.lifespan_context(application):
            handler = application.state.a2a_handler
            from a2a.types import MessageSendParams

            previous = await handler.on_message_send(
                MessageSendParams.model_validate(params())
            )
            send = asyncio.create_task(
                handler.on_message_send(
                    MessageSendParams.model_validate(params("block"))
                )
            )
            try:
                await wait(runtime.started)
                task_id = next(
                    key for key in handler._running_agents if key != previous.id
                )
                assert (
                    await handler.on_get_task(TaskQueryParams(id=task_id))
                ).status.state == TaskState.working
                with pytest.raises(ServerError) as error:
                    await handler.on_get_task(TaskQueryParams(id=previous.id))
                assert error.value.error.code == -32001
                runtime.release_run.set()
                runtime.release_cleanup.set()
                assert (await send).status.state == TaskState.completed
                assert len(handler.task_store.tasks) == 1
            finally:
                runtime.release_run.set()
                runtime.release_cleanup.set()

    asyncio.run(exercise())


def test_subscriber_limit_does_not_prevent_cancel(monkeypatch):
    from agentkit_serve_common.a2a import core

    monkeypatch.setattr(core, "MAX_SUBSCRIBERS", 2)
    runtime = BlockingRuntime()
    application = app(Factory(runtime))

    async def exercise():
        async with application.router.lifespan_context(application):
            handler = application.state.a2a_handler
            from a2a.types import MessageSendParams

            send = asyncio.create_task(
                handler.on_message_send(
                    MessageSendParams.model_validate(params("block"))
                )
            )
            subscribers = []
            try:
                await wait(runtime.started)
                task_id = next(iter(handler._running_agents))
                for _ in range(2):
                    stream = handler.on_resubscribe_to_task(TaskIdParams(id=task_id))
                    await anext(stream)
                    subscribers.append(stream)
                excess = handler.on_resubscribe_to_task(TaskIdParams(id=task_id))
                with pytest.raises(ServerError) as error:
                    await anext(excess)
                assert error.value.error.code == -32004
                runtime.release_cleanup.set()
                canceled = await asyncio.wait_for(
                    handler.on_cancel_task(TaskIdParams(id=task_id)), 3
                )
                assert canceled.status.state == TaskState.canceled
                assert (await send).status.state == TaskState.canceled
            finally:
                runtime.release_run.set()
                runtime.release_cleanup.set()
                for stream in subscribers:
                    await stream.aclose()

    asyncio.run(exercise())


def test_cancellation_cleanup_failure_settles_failed_and_fails_closed():
    class BrokenCleanup(BlockingRuntime):
        async def run(self, request):
            try:
                return await super().run(request)
            except asyncio.CancelledError:
                raise AgentRunError("secret cleanup exception", fatal=True)

    runtime = BrokenCleanup()
    application = app(Factory(runtime))

    async def exercise():
        async with application.router.lifespan_context(application):
            handler = application.state.a2a_handler
            from a2a.types import MessageSendParams

            send = asyncio.create_task(
                handler.on_message_send(
                    MessageSendParams.model_validate(params("block"))
                )
            )
            try:
                await wait(runtime.started)
                task_id = next(iter(handler._running_agents))
                cancel = asyncio.create_task(
                    handler.on_cancel_task(TaskIdParams(id=task_id))
                )
                await wait(runtime.cleanup)
                runtime.release_cleanup.set()
                with pytest.raises(ServerError) as error:
                    await asyncio.wait_for(cancel, 3)
                assert error.value.error.code == -32002
                task = await send
                assert task.status.state == TaskState.failed
                assert "secret" not in str(task)
                assert not handler.agent_executor.health.healthy
            finally:
                runtime.release_run.set()
                runtime.release_cleanup.set()

    asyncio.run(exercise())


def test_shutdown_cancels_indefinitely_blocked_run_without_external_release():
    runtime = BlockingRuntime()
    runtime.release_cleanup.set()
    application = app(Factory(runtime))

    async def exercise():
        from a2a.types import MessageSendParams

        lifespan = application.router.lifespan_context(application)
        await lifespan.__aenter__()
        send = asyncio.create_task(
            application.state.a2a_handler.on_message_send(
                MessageSendParams.model_validate(params("block"))
            )
        )
        shutdown = None
        try:
            await wait(runtime.started)
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
            done, _ = await asyncio.wait({shutdown}, timeout=1)
            assert shutdown in done, "shutdown waited for external runtime release"
            await shutdown
            assert not runtime.release_run.is_set()
            assert runtime.cleaned and runtime.exited
            assert (await send).status.state == TaskState.canceled
        finally:
            runtime.release_run.set()
            runtime.release_cleanup.set()
            if shutdown is not None:
                await shutdown

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel_shutdown", [False, True])
def test_shutdown_drains_active_execution_before_runtime_exit(cancel_shutdown):
    runtime = BlockingRuntime()
    application = app(Factory(runtime))

    async def exercise():
        lifespan = application.router.lifespan_context(application)
        await lifespan.__aenter__()
        from a2a.types import MessageSendParams

        send = asyncio.create_task(
            application.state.a2a_handler.on_message_send(
                MessageSendParams.model_validate(params("block"))
            )
        )
        try:
            await wait(runtime.started)
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await asyncio.sleep(0.02)
            assert not shutdown.done() and not runtime.exited
            if cancel_shutdown:
                shutdown.cancel()
                await asyncio.sleep(0.02)
                assert not shutdown.done() and not runtime.exited
            runtime.release_run.set()
            await wait(runtime.cleanup)
            assert not shutdown.done() and not runtime.exited
            if cancel_shutdown:
                shutdown.cancel()
                await asyncio.sleep(0.02)
                assert not shutdown.done() and not runtime.exited
            runtime.release_cleanup.set()
            if cancel_shutdown:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(shutdown, 3)
            else:
                await asyncio.wait_for(shutdown, 3)
            assert runtime.cleaned and runtime.exited
            assert (await send).status.state == TaskState.canceled
        finally:
            runtime.release_run.set()
            runtime.release_cleanup.set()

    asyncio.run(exercise())
