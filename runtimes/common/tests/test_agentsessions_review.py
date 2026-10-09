from __future__ import annotations

import asyncio
import hashlib
import traceback

import grpc
import pytest
import yaml

from agentkit_serve_common.agentsessions import service
from agentkit_serve_common.agentsessions._generated import harness_pb2, harness_pb2_grpc
from test_agentsessions_protocol import binding_file, protocol  # noqa: F401


def test_programmatic_binding_error_traceback_hides_invalid_yaml(binding_file):
    path, _ = binding_file
    marker = "PRIVATE_CONFIG_MARKER"
    path.write_text("instructions: [" + marker)
    with pytest.raises(protocol().AgentsessionsConfigurationError) as caught:
        protocol().load_verified_agentsessions_binding(path)
    diagnostic = "".join(traceback.format_exception(caught.value))
    assert marker not in diagnostic
    assert "cannot load agentsessions agent configuration" in diagnostic


def test_programmatic_binding_error_traceback_hides_invalid_model_url(binding_file, monkeypatch):
    path, data = binding_file
    marker = "PRIVATE_URL_MARKER"
    data["model"]["baseURL"] = "https://[" + marker + "]/v1"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv(
        "AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST",
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(protocol().AgentsessionsConfigurationError) as caught:
        protocol().load_verified_agentsessions_binding(path)
    diagnostic = "".join(traceback.format_exception(caught.value))
    assert marker not in diagnostic
    assert "credential-bearing model URLs" in diagnostic


@pytest.mark.parametrize("bind", ["LOCALHOST", " Localhost "])
def test_direct_server_accepts_case_insensitive_loopback(binding_file, monkeypatch, bind):
    async def check():
        p = protocol()
        binding = p.load_verified_agentsessions_binding(binding_file[0])
        server = p.create_server(binding)
        original_add_port = server.add_insecure_port
        bound = asyncio.get_running_loop().create_future()

        def add_port(address):
            port = original_add_port(address)
            bound.set_result(port)
            return port

        monkeypatch.setattr(server, "add_insecure_port", add_port)
        monkeypatch.setattr(service, "_registered_server", lambda _: server)
        running = asyncio.create_task(service.serve(binding, bind=bind, port=0))
        try:
            await asyncio.wait((running, bound), return_when=asyncio.FIRST_COMPLETED)
            if running.done():
                await running
                pytest.fail("server terminated instead of binding loopback")
            port = bound.result()
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                descriptor = await harness_pb2_grpc.HarnessStub(channel).Describe(
                    harness_pb2.DescribeRequest(), timeout=3
                )
            assert descriptor.id == binding.descriptor_id
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
            await server.stop(0)

    asyncio.run(check())
