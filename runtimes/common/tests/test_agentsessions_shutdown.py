from __future__ import annotations

import asyncio
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import pytest

from agentkit_serve_common.agentsessions import service
from agentkit_serve_common.agentsessions._generated import harness_pb2 as h, harness_pb2_grpc as g
from test_agentsessions_protocol import binding_file, protocol  # noqa: F401


def _observed_server_factory(on_port, on_server=None):
    original_factory = grpc.aio.server

    def factory(*args, **kwargs):
        server = original_factory(*args, **kwargs)
        original_add_port = server.add_insecure_port

        def add_port(address):
            port = original_add_port(address)
            on_port(port)
            return port

        server.add_insecure_port = add_port
        if on_server is not None:
            on_server(server)
        return server

    return factory


# This file doubles as an offline child-process probe. Only the explicit neutral
# runner owns resources; no adapter factory, SDK, or provider is involved.
def _run_child(config: str) -> None:
    binding = protocol().load_verified_agentsessions_binding(config)
    grpc.aio.server = _observed_server_factory(lambda port: print(f"PORT {port}", flush=True))

    async def runner(binding, request):
        print("ENTERED", flush=True)
        try:
            await asyncio.Future()
        finally:
            loop = asyncio.get_running_loop()
            release = loop.create_future() if request.config == b"hold-cleanup" else None
            if release is not None:
                # Hold the fixture until the parent has sent its second signal;
                # don't race a loaded scheduler against the finite cleanup timer.
                def release_cleanup():
                    loop.remove_reader(sys.stdin.fileno())
                    value = os.read(sys.stdin.fileno(), 1)
                    if not release.done():
                        release.set_result(value)

                loop.add_reader(sys.stdin.fileno(), release_cleanup)
            print("CLOSING", flush=True)
            try:
                if release is not None:
                    await release
                # Deliberately finite asynchronous resource cleanup: the bug
                # interrupts this await when asyncio.run cancels leftover tasks.
                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                print("INTERRUPTED", flush=True)
                raise
            finally:
                if release is not None:
                    loop.remove_reader(sys.stdin.fileno())
            print("CLEANED", flush=True)
            if request.config == b"raise-cleanup":
                raise RuntimeError("PRIVATE-CLEANUP-CONFIG-TOKEN")

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        service.run(binding, port=0, runner=runner)
    except KeyboardInterrupt:
        print("SIGINT EXIT", flush=True)
    finally:
        if all(signal.getsignal(sig) == handler for sig, handler in previous.items()):
            print("HANDLERS RESTORED", flush=True)
    print("RETURNED", flush=True)


@pytest.mark.skipif(os.name != "posix", reason="process signals require POSIX")
@pytest.mark.parametrize("shutdown_signal", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("mode", ["single", "repeat", "mixed", "already-closing", "cleanup-error"])
def test_run_process_drains_finite_cleanup_before_return(binding_file, shutdown_signal, mode):
    # Removing the drain loses CLEANED on SIGINT. Removing the process handler
    # exits with -SIGTERM. Repeated signals must not interrupt awaited teardown.
    async def check():
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).resolve()), str(binding_file[0]),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        lines = []

        async def marker(expected):
            while True:
                line = await asyncio.wait_for(process.stdout.readline(), 5)
                assert line, f"child exited before {expected}: {lines}"
                value = line.decode().strip()
                lines.append(value)
                if value.startswith(expected):
                    return value

        try:
            port = int((await marker("PORT ")).split()[1])
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                await asyncio.wait_for(channel.channel_ready(), 5)
                call = g.HarnessStub(channel).Connect()
                config = b"raise-cleanup" if mode == "cleanup-error" else (
                    b"hold-cleanup" if mode in {"repeat", "mixed", "already-closing"} else b""
                )
                await call.write(h.ControllerFrame(execution_id="shutdown", start=h.Start(config=config)))
                await marker("ENTERED")
                if mode == "already-closing":
                    await call.write(h.ControllerFrame(execution_id="shutdown", cancel=h.Cancel()))
                    await marker("CLOSING")
                process.send_signal(shutdown_signal)
                if mode in {"repeat", "mixed"}:
                    await marker("CLOSING")
                    second = shutdown_signal if mode == "repeat" else (
                        signal.SIGTERM if shutdown_signal == signal.SIGINT else signal.SIGINT
                    )
                    process.send_signal(second)
                # Release only after every requested signal has been sent. The
                # marker/gate controls ordering; 300ms is resource teardown only.
                stdout, stderr = await asyncio.wait_for(process.communicate(b"release"), 5)
                lines.extend(stdout.decode().splitlines())
                call.cancel()
            assert "CLEANED" in lines, (lines, stderr.decode())
            assert "INTERRUPTED" not in lines, (lines, stderr.decode())
            assert "HANDLERS RESTORED" in lines
            assert "RETURNED" in lines
            assert lines.index("CLEANED") < lines.index("RETURNED")
            assert process.returncode == 0, stderr.decode()
            expected_error = "agentsessions execution cleanup failed\n" if mode == "cleanup-error" else ""
            assert stderr.decode() == expected_error
            assert "PRIVATE-CLEANUP-CONFIG-TOKEN" not in stderr.decode()
            assert ("SIGINT EXIT" in lines) == (shutdown_signal == signal.SIGINT)
        finally:
            if process.returncode is None:
                process.kill()
            await asyncio.wait_for(process.communicate(), 5)

    asyncio.run(check())


@pytest.mark.parametrize("shutdown_kind", ["cancel", "transport-stop"])
def test_serve_native_rpc_waits_for_execution_cleanup(binding_file, monkeypatch, shutdown_kind):
    # Transport stop alone must not make serve return while the resource owner
    # remains blocked in its finally. Cancellation of serve's drain must not
    # forward another cancellation to that owner.
    async def check():
        binding = protocol().load_verified_agentsessions_binding(binding_file[0])
        entered = asyncio.Event()
        closing = asyncio.Event()
        release = asyncio.Event()
        cleaned = asyncio.Event()
        interrupted = asyncio.Event()
        bound = asyncio.get_running_loop().create_future()
        server_ready = asyncio.get_running_loop().create_future()
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        monkeypatch.setattr(
            grpc.aio, "server", _observed_server_factory(bound.set_result, server_ready.set_result)
        )

        async def runner(binding, request):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                closing.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    interrupted.set()
                    raise
                cleaned.set()

        running = asyncio.create_task(service.serve(binding, port=0, runner=runner))
        try:
            port = await asyncio.wait_for(bound, 5)
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                await asyncio.wait_for(channel.channel_ready(), 5)
                call = g.HarnessStub(channel).Connect()
                await call.write(h.ControllerFrame(execution_id="shutdown", start=h.Start()))
                await asyncio.wait_for(entered.wait(), 5)
                assert all(signal.getsignal(sig) == handler for sig, handler in previous_handlers.items())
                if shutdown_kind == "cancel":
                    running.cancel()
                else:
                    await asyncio.wait_for(server_ready.result().stop(0), 5)
                await asyncio.wait_for(closing.wait(), 5)
                # Describe's rejection is an actual transport shutdown barrier,
                # rather than a sleep guessing whether stop(0) has completed.
                with pytest.raises(grpc.aio.AioRpcError) as caught:
                    await g.HarnessStub(channel).Describe(h.DescribeRequest(), timeout=5)
                assert caught.value.code() == grpc.StatusCode.UNAVAILABLE
                assert not running.done(), "serve returned before execution cleanup"
                running.cancel()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(running, 5)
                assert cleaned.is_set()
                assert not interrupted.is_set()
                assert all(signal.getsignal(sig) == handler for sig, handler in previous_handlers.items())
                call.cancel()
        finally:
            release.set()
            running.cancel()
            await asyncio.wait_for(asyncio.gather(running, return_exceptions=True), 5)

    asyncio.run(check())


def test_serve_cancellation_during_startup_stops_transport(binding_file, monkeypatch):
    async def check():
        binding = protocol().load_verified_agentsessions_binding(binding_file[0])
        started = asyncio.Event()
        stopped = asyncio.Event()
        original_factory = grpc.aio.server
        servers = []

        def factory(*args, **kwargs):
            server = original_factory(*args, **kwargs)
            original_start = server.start
            original_stop = server.stop
            servers.append(server)

            async def start():
                await original_start()
                started.set()
                await asyncio.Future()

            async def stop(grace):
                stopped.set()
                await original_stop(grace)

            server.start = start
            server.stop = stop
            return server

        monkeypatch.setattr(grpc.aio, "server", factory)
        running = asyncio.create_task(service.serve(binding, port=0))
        try:
            await asyncio.wait_for(started.wait(), 5)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, 5)
            assert stopped.is_set(), "startup cancellation left the transport running"
        finally:
            running.cancel()
            await asyncio.wait_for(asyncio.gather(running, return_exceptions=True), 5)
            for server in servers:
                await asyncio.wait_for(server.stop(0), 5)

    asyncio.run(check())


def test_shutdown_closes_admission_before_transport_stop(binding_file, monkeypatch):
    async def check():
        binding = protocol().load_verified_agentsessions_binding(binding_file[0])
        stopping = asyncio.Event()
        release_stop = asyncio.Event()
        bound = asyncio.get_running_loop().create_future()
        original_registered = service._registered_server
        calls = []

        async def runner(binding, request):
            calls.append(request)

        def registered(servicer):
            server = original_registered(servicer)
            original_stop = server.stop

            async def stop(grace):
                # Keep the actual transport accepting RPCs after shutdown begins
                # to prove admission, not transport rejection, protects runners.
                stopping.set()
                await release_stop.wait()
                await original_stop(grace)

            server.stop = stop
            return server

        monkeypatch.setattr(service, "_registered_server", registered)
        monkeypatch.setattr(grpc.aio, "server", _observed_server_factory(bound.set_result))
        running = asyncio.create_task(service.serve(binding, port=0, runner=runner))
        try:
            port = await asyncio.wait_for(bound, 5)
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                await asyncio.wait_for(channel.channel_ready(), 5)
                running.cancel()
                await asyncio.wait_for(stopping.wait(), 5)
                call = g.HarnessStub(channel).Connect()
                await call.write(h.ControllerFrame(execution_id="late", start=h.Start()))
                with pytest.raises(grpc.aio.AioRpcError) as caught:
                    await asyncio.wait_for(call.read(), 5)
                assert caught.value.code() == grpc.StatusCode.UNAVAILABLE
                assert calls == []
        finally:
            release_stop.set()
            running.cancel()
            await asyncio.wait_for(asyncio.gather(running, return_exceptions=True), 5)

    asyncio.run(check())


def test_run_restores_custom_signal_handlers_on_startup_failure(binding_file):
    binding = protocol().load_verified_agentsessions_binding(binding_file[0])
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def handler(signum, frame):
        pytest.fail("restoration must not invoke the previous handler")

    try:
        for sig in previous:
            signal.signal(sig, handler)
        with pytest.raises(ValueError, match="authentication"):
            service.run(binding, bind="0.0.0.0")
        assert all(signal.getsignal(sig) is handler for sig in previous)
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


@pytest.mark.parametrize("off_thread", [False, True])
def test_run_without_signal_support_preserves_embedded_use(binding_file, monkeypatch, off_thread):
    binding = protocol().load_verified_agentsessions_binding(binding_file[0])
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    original_factory = grpc.aio.server

    def factory(*args, **kwargs):
        server = original_factory(*args, **kwargs)
        original_start = server.start

        async def start():
            await original_start()
            await server.stop(0)

        server.start = start
        return server

    monkeypatch.setattr(grpc.aio, "server", factory)
    original_new_loop = asyncio.DefaultEventLoopPolicy.new_event_loop

    def new_loop(policy):
        loop = original_new_loop(policy)

        def unsupported(*args):
            if off_thread:
                pytest.fail("off-thread run must not try to own process signals")
            raise NotImplementedError

        loop.add_signal_handler = unsupported
        return loop

    monkeypatch.setattr(asyncio.DefaultEventLoopPolicy, "new_event_loop", new_loop)
    if off_thread:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(service.run, binding, port=0).result(timeout=5)
    else:
        service.run(binding, port=0)
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())


if __name__ == "__main__":
    _run_child(sys.argv[1])
