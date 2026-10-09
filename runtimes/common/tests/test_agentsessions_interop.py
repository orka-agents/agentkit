from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

from test_agentsessions_protocol import binding_file, protocol  # noqa: F401


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
        async def runner(binding, request):
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
            stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
            assert process.returncode == 0, (stdout + stderr).decode()
            assert stdout == b"Go/Python Harness wire interoperable\n"
            assert len(seen) == 1
        finally:
            await server.stop(0)
    asyncio.run(check())
