"""Native agentsessions Harness SPI. PR1 has no model or tool execution path."""

from __future__ import annotations

import asyncio
import logging
import secrets
import signal
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

import grpc

from ..adapter_support import _wait_for_owner_task
from ..conversation import ConversationTurn, RunRequest
from ..runtime import RunResult
from ._generated import common_pb2 as common
from ._generated import harness_pb2 as harness
from ._generated import harness_pb2_grpc
from .binding import VerifiedAgentsessionsBinding

MAX_MESSAGE_BYTES = 4 * 1024 * 1024
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"})
_LOG = logging.getLogger(__name__)
# A fresh, explicitly supplied async execution hook. It owns any per-Start
# resources and must close them under cancellation. Never use build_runtime.
ExecutionRunner = Callable[
    [VerifiedAgentsessionsBinding, RunRequest], Awaitable[RunResult | None]
]
_HISTORY_METADATA_BODIES = {
    common.EVENT_MODEL_CALL: "model",
    common.EVENT_USAGE: "usage",
    common.EVENT_LIFECYCLE: "lifecycle",
    common.EVENT_END: "end",
    common.EVENT_ERROR: "error",
    common.EVENT_EXECUTION_START: "execution_start",
}


class _Unsupported(ValueError):
    pass


async def _settle(tasks: list[asyncio.Task]) -> bool:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return any(
        isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError)
        for result in results
    )


async def _invoke_runner(
    runner: ExecutionRunner, binding: VerifiedAgentsessionsBinding, request: RunRequest
) -> RunResult | None:
    # Invocation also belongs inside the task: callbacks may return any
    # Awaitable or raise synchronously before returning one.
    return await runner(binding, request)


def _text(message: common.Message, role: str) -> str:
    if message.role != role or any(part.WhichOneof("part") != "text" for part in message.parts):
        raise _Unsupported("only role-correct plain text is supported")
    return "".join(part.text.text for part in message.parts)


def _request(first: harness.ControllerFrame) -> RunRequest:
    start = first.start
    history: list[ConversationTurn] = []
    for event in start.history:
        if event.kind in (common.EVENT_INPUT, common.EVENT_OUTPUT):
            role = "user" if event.kind == common.EVENT_INPUT else "assistant"
            if event.WhichOneof("body") != "message":
                raise _Unsupported("history message payload is required")
            history.append(ConversationTurn(role=role, text=_text(event.message, role)))
        elif event.kind in _HISTORY_METADATA_BODIES:
            # Host-owned journal metadata is not conversation input. Never feed
            # ExecutionStart.Config or model requests into framework prompts.
            if event.WhichOneof("body") != _HISTORY_METADATA_BODIES[event.kind]:
                raise _Unsupported("unsupported history payload")
            if event.kind == common.EVENT_MODEL_CALL:
                for message in event.model.messages:
                    if message.role not in {"user", "assistant"}:
                        raise _Unsupported("unsupported model history role")
                    _text(message, message.role)
        else:
            raise _Unsupported("tool or unknown history is unsupported")
    inputs = [_text(message, "user") for message in start.inputs]
    history.extend(ConversationTurn(role="user", text=value) for value in inputs[:-1])
    return RunRequest(
        prompt=inputs[-1] if inputs else "",
        history=tuple(history),
        session_id=first.session or None,
        turn_id=first.execution_id,
        config=bytes(start.config),
    )


def _end(state: str, code: grpc.StatusCode | None = None, description: str = "") -> common.HarnessEnd:
    end = common.HarnessEnd(state=state)
    if code is not None:
        end.error.CopyFrom(common.Error(code=code.value[0], description=description))
    return end


def _event(execution_id: str, **kwargs) -> common.Event:
    return common.Event(execution_id=execution_id, schema_version=1, **kwargs)


class HarnessService(harness_pb2_grpc.HarnessServicer):
    """One admitted execution per service, with a reader independent of the runner."""

    def __init__(
        self,
        binding: VerifiedAgentsessionsBinding,
        *,
        runner: ExecutionRunner | None = None,
        auth_token: str | None = None,
    ) -> None:
        self.binding = binding
        self.runner = runner
        self.auth_token = auth_token
        self._active: object | None = None
        self._stopping = False
        self._owners: set[asyncio.Task] = set()
        self._idle = asyncio.Event()
        self._idle.set()

    async def _drain(self) -> None:
        # Transport termination does not wait for canceled RPC handlers' shielded
        # execution owners. Keep the loop alive until their cleanup settles.
        tasks = list(self._owners)
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        await _settle(tasks)
        await self._idle.wait()

    async def _authenticate(self, context: grpc.aio.ServicerContext) -> None:
        if self.auth_token is None:
            return
        values = [value for key, value in context.invocation_metadata() if key == "authorization"]
        expected = ("Bearer " + self.auth_token).encode("utf-8")
        if len(values) != 1 or not secrets.compare_digest(values[0].encode("utf-8"), expected):
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "authentication required")

    async def Describe(
        self, request: harness.DescribeRequest, context: grpc.aio.ServicerContext
    ) -> harness.HarnessDescriptor:
        await self._authenticate(context)
        return harness.HarnessDescriptor(
            id=self.binding.descriptor_id,
            models=[self.binding.spec.model.name],
            capabilities=harness.Capabilities(
                resumability=harness.RESUMABILITY_STATELESS_REPLAY,
                fork_safe=True,
                requires_gpu=False,
                streaming=False,
                reasoning_replay=False,
            ),
        )

    async def _read_controls(
        self,
        frames: AsyncIterator[harness.ControllerFrame],
        first: harness.ControllerFrame,
    ) -> common.HarnessEnd:
        async for frame in frames:
            if frame.execution_id != first.execution_id or frame.session != first.session:
                return _end("FAILED", grpc.StatusCode.INVALID_ARGUMENT, "control frame identity mismatch")
            kind = frame.WhichOneof("frame")
            if kind == "cancel":
                return _end("CANCELED", grpc.StatusCode.CANCELLED, "execution canceled")
            if kind in {"model", "tool", "approval"}:
                return _end("FAILED", grpc.StatusCode.UNIMPLEMENTED, "model, tool and approval replies are unsupported")
            return _end("FAILED", grpc.StatusCode.INVALID_ARGUMENT, "unexpected control frame")
        # A half-close means the controller can no longer service mediated effects.
        return _end("CANCELED", grpc.StatusCode.CANCELLED, "controller disconnected")

    async def Connect(
        self,
        request_iterator: AsyncIterator[harness.ControllerFrame],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[common.Event]:
        await self._authenticate(context)
        frames = request_iterator.__aiter__()
        try:
            first = await anext(frames)
        except StopAsyncIteration:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "first frame must be Start")
            return
        # Leave ample bounded space for an END carrying the exact same ID.
        if (
            first.WhichOneof("frame") != "start"
            or not first.execution_id
            or len(first.execution_id.encode("utf-8")) > 1024
        ):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Start first and execution_id of 1..1024 bytes required",
            )
            return
        execution_id = first.execution_id
        if self._stopping:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "service is shutting down")
            return
        if self._active:
            yield _event(execution_id, kind=common.EVENT_END, end=_end("FAILED", grpc.StatusCode.RESOURCE_EXHAUSTED, "an execution is already active"))
            return
        # No await between checking and reserving admission on this asyncio loop.
        reservation = object()
        self._active = reservation
        self._idle.clear()
        reader = None
        execution = None
        cleanup_failed = False
        settlement = None

        async def cleanup() -> None:
            nonlocal settlement, cleanup_failed
            if settlement is None:
                tasks = [task for task in (reader, execution) if task is not None]
                # Retrieve already-failed outcomes even when caller cancellation
                # preempts result handling; classify cleanup failures separately.
                cleanup_tasks = [task for task in tasks if not task.done() or task.cancelling()]
                for task in tasks:
                    if task.done() and not task.cancelled():
                        task.exception()
                    elif not task.done() and not task.cancelling():
                        task.cancel()
                settlement = asyncio.create_task(_settle(cleanup_tasks))
            try:
                # Reuse the settlement if cancellation sends us through finally:
                # never cancel a resource owner's teardown a second time.
                await _wait_for_owner_task(settlement)
            finally:
                if not settlement.cancelled() and settlement.result() and not cleanup_failed:
                    _LOG.error("agentsessions execution cleanup failed")
                    cleanup_failed = True
                self._owners.difference_update(task for task in (reader, execution) if task is not None)
                if self._active is reservation:
                    self._active = None
                    self._idle.set()

        try:
            try:
                request = _request(first)
                if first.start.resume_from_seq < 0:
                    raise _Unsupported("negative resume cursor")
            except _Unsupported:
                await cleanup()
                yield _event(execution_id, kind=common.EVENT_END, end=_end("FAILED", grpc.StatusCode.UNIMPLEMENTED, "unsupported Start content"))
                return
            if self.runner is None:
                await cleanup()
                yield _event(execution_id, kind=common.EVENT_END, end=_end("FAILED", grpc.StatusCode.UNIMPLEMENTED, "agentsessions execution is not implemented"))
                return
            reader = asyncio.create_task(self._read_controls(frames, first))
            execution = asyncio.create_task(_invoke_runner(self.runner, self.binding, request))
            self._owners.update((reader, execution))
            await asyncio.wait((reader, execution), return_when=asyncio.FIRST_COMPLETED)
            result = None
            if reader.done():
                try:
                    end = reader.result()
                except asyncio.CancelledError:
                    end = _end("CANCELED", grpc.StatusCode.CANCELLED, "execution canceled")
                except Exception:
                    # Iterator/transport errors can contain private request data.
                    end = _end("FAILED", grpc.StatusCode.INTERNAL, "execution failed")
                if not execution.done() and not execution.cancelling():
                    execution.cancel()
                # A disconnect during awaited teardown must not cancel the
                # resource owner's finally block a second time.
                cleanup_failed = await _wait_for_owner_task(asyncio.create_task(_settle([execution])))
                if cleanup_failed:
                    end = _end("FAILED", grpc.StatusCode.INTERNAL, "execution cleanup failed")
            else:
                try:
                    result = execution.result()
                    end = _end("COMPLETED")
                except asyncio.CancelledError:
                    end = _end("CANCELED", grpc.StatusCode.CANCELLED, "execution canceled")
                except Exception:
                    # Exception text can contain prompts, Config, URLs or tokens.
                    cleanup_failed = bool(execution.cancelling())
                    description = "execution cleanup failed" if cleanup_failed else "execution failed"
                    end = _end("FAILED", grpc.StatusCode.INTERNAL, description)
            if cleanup_failed:
                _LOG.error("agentsessions execution cleanup failed")
            if result is not None:
                try:
                    output = _event(
                        execution_id,
                        kind=common.EVENT_OUTPUT,
                        message=common.Message(
                            role="assistant",
                            parts=[common.Part(text=common.TextPart(text=result.text))],
                        ),
                    )
                    oversized = output.ByteSize() > MAX_MESSAGE_BYTES
                except Exception:
                    # Protobuf text conversion can fail (e.g. lone surrogates).
                    # Keep both the RPC and its diagnostics free of raw output.
                    end = _end("FAILED", grpc.StatusCode.INTERNAL, "execution failed")
                else:
                    if oversized:
                        end = _end("FAILED", grpc.StatusCode.RESOURCE_EXHAUSTED, "output exceeds message limit")
                    else:
                        yield output
            # ClientHarness.Run returns on END and cancels without draining EOF.
            # Finish all owned cleanup and release this reservation before END.
            await cleanup()
            if cleanup_failed:
                end = _end("FAILED", grpc.StatusCode.INTERNAL, "execution cleanup failed")
            yield _event(execution_id, kind=common.EVENT_END, end=end)
        finally:
            await cleanup()


class _ManagedServer(grpc.aio.Server):
    """Preserve the gRPC server API while owning execution cleanup on shutdown."""

    def __init__(self, server: grpc.aio.Server, service: HarnessService) -> None:
        self._server = server
        self._service = service
        self._draining: asyncio.Task[None] | None = None
        self._termination: asyncio.Task[None] | None = None
        self._started = False
        self._stopped: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def add_generic_rpc_handlers(self, generic_rpc_handlers: Sequence[grpc.GenericRpcHandler]) -> None:
        self._server.add_generic_rpc_handlers(generic_rpc_handlers)

    def add_registered_method_handlers(
        self, service_name: str, method_handlers: dict[str, grpc.RpcMethodHandler]
    ) -> None:
        self._server.add_registered_method_handlers(service_name, method_handlers)

    def _check_bind(self, address: str) -> None:
        # Unix sockets are local, like loopback; other address forms fail closed.
        if address.startswith(("unix:", "unix-abstract:")):
            return
        host = address.lower()
        if host not in _LOOPBACK_HOSTS:
            prefix = address.rpartition(":")[0]
            if address.startswith("[") and prefix.endswith("]"):
                host = prefix[1:-1].lower()
            elif address.count(":") == 1:
                host = prefix.lower()
            else:
                # Bare IPv6 is the entire host, not loopback plus a port.
                host = ""
        if host not in _LOOPBACK_HOSTS and not self._service.auth_token:
            raise ValueError("nonloopback agentsessions bind requires authentication")

    def add_insecure_port(self, address: str) -> int:
        self._check_bind(address)
        return self._server.add_insecure_port(address)

    def add_secure_port(self, address: str, server_credentials: grpc.ServerCredentials) -> int:
        self._check_bind(address)
        return self._server.add_secure_port(address, server_credentials)

    async def start(self) -> None:
        if self._service._stopping:
            raise grpc.aio.UsageError("server is shutting down")
        # Own shutdown even if cancellation preempts native startup completion.
        self._started = True
        await self._server.start()

    async def _drain(self) -> None:
        if self._draining is None:
            self._draining = asyncio.create_task(self._service._drain())
        await _wait_for_owner_task(self._draining)

    async def stop(self, grace: float | None) -> None:
        if not self._started:
            # Native stop is a no-op before the first start. In particular, do
            # not close admission or cancel a pre-start termination monitor.
            await self._server.stop(grace)
            return
        self._service._stopping = True

        async def shutdown() -> None:
            try:
                await self._server.stop(grace)
            finally:
                await self._drain()
            if not self._stopped.done():
                self._stopped.set_result(None)
            if self._termination is not None:
                # Transport and execution cleanup are complete; retire the
                # monitor without retaining a second lifecycle owner.
                if not self._termination.done():
                    self._termination.cancel()
                await asyncio.gather(self._termination, return_exceptions=True)

        # A canceled stop caller must still wait until resource owners settle.
        # Separate transport stop calls preserve gRPC's most-restrictive grace.
        await _wait_for_owner_task(asyncio.create_task(shutdown()))

    async def wait_for_termination(self, timeout: float | None = None) -> bool:
        if self._stopped.done():
            return False
        if self._termination is None:
            async def terminated() -> None:
                await self._server.wait_for_termination()
                self._service._stopping = True
                await self._drain()

            self._termination = asyncio.create_task(terminated())
        # A timeout or canceled waiter must not cancel gRPC's shared termination
        # future or the resource drain. Completion includes execution cleanup.
        done, _ = await asyncio.wait(
            (self._termination, self._stopped), timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            return True
        if not self._stopped.done():
            self._termination.result()
        return False


def _registered_server(service: HarnessService) -> grpc.aio.Server:
    server = grpc.aio.server(options=[
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ])
    harness_pb2_grpc.add_HarnessServicer_to_server(service, server)
    return _ManagedServer(server, service)


def create_server(
    binding: VerifiedAgentsessionsBinding,
    *,
    runner: ExecutionRunner | None = None,
    auth_token: str | None = None,
) -> grpc.aio.Server:
    """Create an unbound server in its asyncio loop, with managed shutdown."""
    return _registered_server(HarnessService(binding, runner=runner, auth_token=auth_token))


async def serve(
    binding: VerifiedAgentsessionsBinding,
    *,
    bind: str = "127.0.0.1",
    port: int = 8080,
    auth_token: str | None = None,
    runner: ExecutionRunner | None = None,
) -> None:
    """Serve h2c; nonloopback requires auth AND deployment-private networking."""
    bind = bind.strip().lower()
    if bind not in _LOOPBACK_HOSTS and not auth_token:
        raise ValueError("nonloopback agentsessions bind requires authentication")
    service = HarnessService(binding, runner=runner, auth_token=auth_token)
    server = _registered_server(service)
    host = f"[{bind}]" if ":" in bind else bind
    server.add_insecure_port(f"{host}:{port}")
    termination = None
    try:
        await server.start()
        termination = asyncio.create_task(server.wait_for_termination())
        # gRPC shares its termination future with shutdown. Canceling that
        # future would make stop() fail instead of completing server cleanup.
        await asyncio.shield(termination)
    finally:
        # Close admission before yielding to stop(). One shielded lifecycle owner
        # must finish every shutdown step, even if serve is canceled again.
        service._stopping = True

        async def shutdown() -> None:
            try:
                await server.stop(0)
                if termination is not None:
                    await termination
            finally:
                await service._drain()

        await _wait_for_owner_task(asyncio.create_task(shutdown()))


def run(
    binding: VerifiedAgentsessionsBinding,
    *,
    bind: str = "127.0.0.1",
    port: int = 8080,
    auth_token: str | None = None,
    runner: ExecutionRunner | None = None,
) -> None:
    """Synchronous CLI entrypoint; does not receive a default RuntimeFactory."""
    received_signal: int | None = None

    async def main() -> None:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        previous_handlers = {}

        def request_shutdown(signum: int) -> None:
            nonlocal received_signal
            if received_signal is None:
                received_signal = signum
                task.cancel()

        try:
            # Embedded serve never owns process signals. Synchronous run may also
            # be embedded off the main thread or on a loop without signal support.
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    previous = signal.getsignal(signum)
                    try:
                        loop.add_signal_handler(signum, request_shutdown, signum)
                    except (NotImplementedError, RuntimeError, ValueError):
                        continue
                    previous_handlers[signum] = previous
            await serve(binding, bind=bind, port=port, auth_token=auth_token, runner=runner)
        finally:
            for signum, previous in previous_handlers.items():
                loop.remove_signal_handler(signum)
                signal.signal(signum, previous)

    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        if received_signal == signal.SIGINT:
            raise KeyboardInterrupt from None
        if received_signal != signal.SIGTERM:
            raise
