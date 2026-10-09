from __future__ import annotations

import asyncio
import subprocess
import sys
from contextlib import asynccontextmanager

import grpc
import pytest

from test_agentsessions_protocol import binding_file, protocol, start  # noqa: F401


_TIMEOUT = 5


class _CleanupGate:
    def __init__(self):
        self.entered = asyncio.Event()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.interrupted = asyncio.Event()
        self.settled = asyncio.Event()
        self.waiters = []

    def watch(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.waiters.append(task)
        return task


@asynccontextmanager
async def _running_server(binding_file):
    p = protocol()
    from agentkit_serve_common.agentsessions._generated import common_pb2 as c
    from agentkit_serve_common.agentsessions._generated import harness_pb2 as h
    from agentkit_serve_common.agentsessions._generated import harness_pb2_grpc as g

    gate = _CleanupGate()

    async def runner(binding, request):
        gate.entered.set()
        try:
            await asyncio.Future()
        finally:
            gate.started.set()
            try:
                await gate.release.wait()
                gate.finished.set()
            except asyncio.CancelledError:
                gate.interrupted.set()
                raise
            finally:
                gate.settled.set()

    binding = p.load_verified_agentsessions_binding(binding_file[0])
    server = p.create_server(binding, runner=runner)
    port = server.add_insecure_port("127.0.0.1:0")
    assert port > 0
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    call = None
    try:
        await server.start()
        stub = g.HarnessStub(channel)
        assert (await stub.Describe(h.DescribeRequest(), timeout=_TIMEOUT)).id == binding.descriptor_id
        call = stub.Connect()
        await asyncio.wait_for(call.write(start(h, c)), _TIMEOUT)
        await asyncio.wait_for(gate.entered.wait(), _TIMEOUT)
        yield server, stub, gate
    finally:
        # Release first so failures cannot leave a stop or runner waiting on us.
        gate.release.set()
        if call is not None:
            call.cancel()
        try:
            await asyncio.wait_for(server.stop(0), _TIMEOUT)
            if gate.entered.is_set():
                await asyncio.wait_for(gate.settled.wait(), _TIMEOUT)
        finally:
            await channel.close()
            for task in gate.waiters:
                if not task.done():
                    task.cancel()
            await asyncio.wait_for(
                asyncio.gather(*gate.waiters, return_exceptions=True), _TIMEOUT
            )


async def _assert_transport_stopped(stub):
    from agentkit_serve_common.agentsessions._generated import harness_pb2 as h

    # A rejected native RPC proves transport shutdown has taken effect without
    # relying on a scheduling sleep or inspecting the service's owner tasks.
    with pytest.raises(grpc.aio.AioRpcError) as failure:
        await stub.Describe(h.DescribeRequest(), timeout=_TIMEOUT)
    assert failure.value.code() in {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.CANCELLED}


def test_create_server_stop_waits_for_runner_cleanup(binding_file):
    async def check():
        async with _running_server(binding_file) as (server, stub, gate):
            stopping = gate.watch(server.stop(0))
            await asyncio.wait_for(gate.started.wait(), _TIMEOUT)
            await _assert_transport_stopped(stub)

            assert not stopping.done(), "stop returned while runner cleanup was held"
            assert not gate.finished.is_set()
            assert not gate.interrupted.is_set()

            gate.release.set()
            await asyncio.wait_for(asyncio.shield(stopping), _TIMEOUT)
            assert gate.finished.is_set()
            assert not gate.interrupted.is_set()

    asyncio.run(check())


@pytest.mark.parametrize("cancellations", [1, 3])
def test_create_server_stop_caller_cancellation_preserves_cleanup(binding_file, cancellations):
    async def check():
        async with _running_server(binding_file) as (server, stub, gate):
            stopping = gate.watch(server.stop(0))
            await asyncio.wait_for(gate.started.wait(), _TIMEOUT)
            await _assert_transport_stopped(stub)

            for _ in range(cancellations):
                assert stopping.cancel(), "stop must remain pending until runner cleanup finishes"
                await _assert_transport_stopped(stub)
                assert not gate.interrupted.is_set()
                assert not gate.finished.is_set()
                assert not stopping.done()

            gate.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(stopping), _TIMEOUT)
            assert gate.finished.is_set()
            assert not gate.interrupted.is_set()

    asyncio.run(check())


def test_create_server_wait_for_termination_includes_runner_cleanup(binding_file):
    async def check():
        async with _running_server(binding_file) as (server, stub, gate):
            termination = gate.watch(server.wait_for_termination())
            stopping = gate.watch(server.stop(0))
            await asyncio.wait_for(gate.started.wait(), _TIMEOUT)
            await _assert_transport_stopped(stub)

            assert await asyncio.wait_for(
                server.wait_for_termination(timeout=0.01), _TIMEOUT
            ) is True
            assert not termination.done(), "termination reported before runner cleanup finished"
            assert not gate.finished.is_set()

            gate.release.set()
            await asyncio.wait_for(asyncio.shield(stopping), _TIMEOUT)
            assert await asyncio.wait_for(asyncio.shield(termination), _TIMEOUT) is False
            assert gate.finished.is_set()
            assert not gate.interrupted.is_set()
            assert await asyncio.wait_for(
                server.wait_for_termination(timeout=0), _TIMEOUT
            ) is False

    asyncio.run(check())


@pytest.mark.parametrize("wait_before_stop", [False, True])
def test_create_server_pre_start_stop_preserves_later_start(binding_file, wait_before_stop):
    # asyncio.wait_for cannot bound a stop that defers caller cancellation.
    # A subprocess timeout also bounds asyncio.run's cancellation during exit.
    script = """
import asyncio
import sys

from agentkit_serve_common.agentsessions import create_server, load_verified_agentsessions_binding

async def check():
    server_binding = load_verified_agentsessions_binding(sys.argv[1])
    server = create_server(server_binding)
    termination = None
    if sys.argv[2] == "wait":
        waiting = asyncio.Event()

        async def wait_for_termination():
            waiting.set()
            return await server.wait_for_termination()

        termination = asyncio.create_task(wait_for_termination())
        await asyncio.wait_for(waiting.wait(), 1)
        assert await server.wait_for_termination(timeout=0.01) is True
        assert not termination.done()

    await server.stop(0)
    await server.stop(None)
    assert await server.wait_for_termination(timeout=0.01) is True
    if termination is not None:
        assert not termination.done()
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    import grpc
    from agentkit_serve_common.agentsessions._generated import harness_pb2 as h, harness_pb2_grpc as g
    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        descriptor = await g.HarnessStub(channel).Describe(h.DescribeRequest(), timeout=1)
        assert descriptor.id == server_binding.descriptor_id
        call = g.HarnessStub(channel).Connect()
        await call.write(h.ControllerFrame(execution_id="post-pre-start-stop", start=h.Start()))
        end = await asyncio.wait_for(call.read(), 1)
        assert end.end.state == "FAILED" and end.end.error.code == 12
        call.cancel()
    await server.stop(None)
    if termination is not None:
        assert await asyncio.wait_for(asyncio.shield(termination), 1) is False
    assert await server.wait_for_termination(timeout=0) is False
    try:
        await server.start()
    except grpc.aio.UsageError:
        pass
    else:
        raise AssertionError("stopped server restarted")
    print("pre-start stop preserved first start", flush=True)

asyncio.run(check())
"""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(binding_file[0]),
                "wait" if wait_before_stop else "stop",
            ],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("pre-start stop or termination wait hung", pytrace=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "pre-start stop preserved first start\n"


def test_create_server_canceled_termination_waiter_does_not_poison_shutdown(binding_file):
    async def check():
        from agentkit_serve_common.agentsessions._generated import harness_pb2 as h

        async with _running_server(binding_file) as (server, stub, gate):
            waiting = asyncio.Event()

            async def wait_for_termination():
                waiting.set()
                return await server.wait_for_termination()

            canceled = gate.watch(wait_for_termination())
            await asyncio.wait_for(waiting.wait(), _TIMEOUT)
            # Complete a native RPC while the termination waiter is pending.
            await stub.Describe(h.DescribeRequest(), timeout=_TIMEOUT)
            assert not canceled.done()
            assert canceled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(canceled), _TIMEOUT)

            stopping = gate.watch(server.stop(0))
            await asyncio.wait_for(gate.started.wait(), _TIMEOUT)
            await _assert_transport_stopped(stub)
            termination = gate.watch(server.wait_for_termination())
            assert await asyncio.wait_for(
                server.wait_for_termination(timeout=0.01), _TIMEOUT
            ) is True
            assert not stopping.done()
            assert not termination.done()
            assert not gate.interrupted.is_set()

            gate.release.set()
            await asyncio.wait_for(asyncio.shield(stopping), _TIMEOUT)
            assert await asyncio.wait_for(asyncio.shield(termination), _TIMEOUT) is False
            assert gate.finished.is_set()
            assert not gate.interrupted.is_set()
            assert await asyncio.wait_for(
                server.wait_for_termination(timeout=0), _TIMEOUT
            ) is False

    asyncio.run(check())


def test_create_server_immediate_stop_shortens_concurrent_graceful_stop(binding_file):
    async def check():
        async with _running_server(binding_file) as (server, stub, gate):
            graceful = gate.watch(server.stop(_TIMEOUT * 4))
            # Graceful shutdown rejects new native RPCs but keeps the admitted
            # runner alive. This proves the long grace period has begun.
            await _assert_transport_stopped(stub)
            assert not graceful.done()
            assert not gate.started.is_set()

            immediate = gate.watch(server.stop(0))
            # A serialized or ignored stop(0) would exhaust this timeout before
            # the original grace period expired and before cleanup could start.
            await asyncio.wait_for(gate.started.wait(), _TIMEOUT)
            await _assert_transport_stopped(stub)
            assert await asyncio.wait_for(
                server.wait_for_termination(timeout=0.01), _TIMEOUT
            ) is True
            assert not graceful.done()
            assert not immediate.done()
            assert not gate.finished.is_set()
            assert not gate.interrupted.is_set()

            gate.release.set()
            await asyncio.wait_for(asyncio.shield(immediate), _TIMEOUT)
            await asyncio.wait_for(asyncio.shield(graceful), _TIMEOUT)
            assert gate.finished.is_set()
            assert not gate.interrupted.is_set()
            assert await asyncio.wait_for(
                server.wait_for_termination(timeout=0), _TIMEOUT
            ) is False

    asyncio.run(check())


@pytest.mark.parametrize("secure", [False, True])
@pytest.mark.parametrize("address", ["0.0.0.0:0", "[::]:0", "example.invalid:0", "::1:8080", "::ffff:127.0.0.1:8080"])
@pytest.mark.parametrize("auth_token", [None, "", "test-bind-token"])
def test_create_server_port_binding_requires_nonloopback_auth(binding_file, monkeypatch, secure, address, auth_token):
    async def check():
        from agentkit_serve_common.agentsessions import service

        native = grpc.aio.server()
        delegated = []
        credentials = object()

        def bind(address, *args):
            delegated.append((address, args))
            return 12345

        # Inspect delegation without exposing a listener on the test machine.
        monkeypatch.setattr(native, "add_insecure_port", bind)
        monkeypatch.setattr(native, "add_secure_port", bind)
        monkeypatch.setattr(grpc.aio, "server", lambda **kwargs: native)
        server = service.create_server(
            protocol().load_verified_agentsessions_binding(binding_file[0]), auth_token=auth_token
        )
        try:
            if secure:
                operation = lambda: server.add_secure_port(address, credentials)
            else:
                operation = lambda: server.add_insecure_port(address)
            if auth_token:
                assert operation() == 12345
                assert delegated == [(address, (credentials,) if secure else ())]
            else:
                with pytest.raises(ValueError, match="authentication"):
                    operation()
                assert delegated == []
        finally:
            await server.stop(0)

    asyncio.run(check())


@pytest.mark.parametrize("address", ["127.0.0.1:0", "LOCALHOST:0", "[::1]:0", "[::ffff:127.0.0.1]:0", "::1", "::ffff:127.0.0.1", "unix:/local-test.sock", "unix-abstract:local-test"])
def test_create_server_local_port_binding_remains_keyless(binding_file, monkeypatch, address):
    async def check():
        from agentkit_serve_common.agentsessions import service

        native = grpc.aio.server()
        delegated = []
        monkeypatch.setattr(native, "add_insecure_port", lambda address: delegated.append(address) or 12345)
        monkeypatch.setattr(grpc.aio, "server", lambda **kwargs: native)
        server = service.create_server(protocol().load_verified_agentsessions_binding(binding_file[0]))
        try:
            assert server.add_insecure_port(address) == 12345
            assert delegated == [address]
        finally:
            await server.stop(0)

    asyncio.run(check())
