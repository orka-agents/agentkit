from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from agentkit_serve_common.adapter_support import _wait_for_owner_task
from test_agentsessions_protocol import binding_file, protocol  # noqa: F401


async def _communicate(process, timeout=30):
    communication = asyncio.create_task(process.communicate())
    try:
        return await asyncio.wait_for(asyncio.shield(communication), timeout)
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        # Keep the pipe drain alive and reap the child even under repeated
        # cancellation. wait() alone can deadlock with full PIPE buffers.
        await _wait_for_owner_task(communication)


@pytest.mark.parametrize("control", ["timeout", "cancel"])
def test_interop_child_is_killed_and_drained_on_interruption(
    control, monkeypatch, tmp_path,
):
    async def check():
        ready = tmp_path / "child-ready"
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            "import pathlib, sys, time; "
            "sys.stdout.buffer.write(b'x' * 262144); sys.stdout.flush(); "
            "sys.stderr.buffer.write(b'y' * 262144); sys.stderr.flush(); "
            "pathlib.Path(sys.argv[1]).touch(); time.sleep(300)",
            str(ready),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        started = asyncio.Event()
        closing = asyncio.Event()
        release = asyncio.Event()
        interrupted = asyncio.Event()
        captured = []
        real_communicate = process.communicate

        async def owned_communicate():
            started.set()
            try:
                result = await real_communicate()
                captured.append(result)
                closing.set()
                if control == "cancel":
                    await release.wait()
                return result
            except asyncio.CancelledError:
                interrupted.set()
                raise

        monkeypatch.setattr(process, "communicate", owned_communicate)
        consumer = asyncio.create_task(
            _communicate(process, timeout=0.05 if control == "timeout" else 30)
        )
        try:
            await asyncio.wait_for(started.wait(), 2)
            if control == "timeout":
                with pytest.raises(TimeoutError):
                    await consumer
            else:
                async def wait_for_output():
                    while not ready.exists():
                        await asyncio.sleep(0.005)

                await asyncio.wait_for(wait_for_output(), 2)
                consumer.cancel()
                await asyncio.wait_for(closing.wait(), 1)
                assert not consumer.done()
                for _ in range(2):
                    consumer.cancel()
                    await asyncio.sleep(0)
                    assert not consumer.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await consumer
            assert process.returncode is not None
            assert not interrupted.is_set()
            assert len(captured) == 1
            if control == "cancel":
                assert captured[0] == (b"x" * 262144, b"y" * 262144)
        finally:
            release.set()
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            if process.returncode is None:
                process.kill()
            await real_communicate()

    asyncio.run(check())


def test_pinned_go_client_uses_python_native_harness(binding_file, tmp_path):
    repo = Path(__file__).resolve().parents[3]
    fixture = repo / "test/agentsessions"
    assert (fixture / "main.go").is_file(), "pinned Go agentsessions wire fixture is missing"
    assert shutil.which("go"), "Go is required for agentsessions interoperability"
    binary = tmp_path / "wire-client"
    built = subprocess.run(["go", "build", "-o", str(binary), "."], cwd=fixture, capture_output=True, text=True, timeout=180)
    assert built.returncode == 0, built.stdout + built.stderr

    async def check():
        p = protocol()
        from agentkit_serve_common.runtime import RunResult
        seen = []
        async def runner(binding, request, exchange):
            seen.append(request)
            assert request.config == b"\x00go-config\xff"
            assert request.prompt == "from Go"
            assert request.turn_id == "go-execution"
            return RunResult(text="from Python")
        binding = p.load_verified_agentsessions_binding(binding_file[0])
        server = p.create_server(binding, runner=runner, auth_token="interop-local-token")
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            process = await asyncio.create_subprocess_exec(str(binary), f"127.0.0.1:{port}", binding.descriptor_id, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await _communicate(process)
            assert process.returncode == 0, (stdout + stderr).decode()
            assert stdout == b"Go/Python Harness wire interoperable\n"
            assert len(seen) == 1
        finally:
            await server.stop(0)
    asyncio.run(check())
