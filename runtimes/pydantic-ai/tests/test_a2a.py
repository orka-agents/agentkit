"""Qualify the normal Pydantic loop under the A2A skin, not sessions mediation."""

from __future__ import annotations

import asyncio

import httpx
from agentkit_serve import agent_factory
from agentkit_serve_common import cli
from agentkit_serve_common.config import AgentSpec
from pydantic_ai.models.test import TestModel


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "pydantic-a2a"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "http://localhost:1234/v1",
                "name": "test",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "expose": {"openai": True, "port": 8080},
        }
    )


def test_pydantic_cli_enables_a2a_without_allocating_provider_before_startup(
    monkeypatch,
):
    captured = {}
    monkeypatch.setattr(cli, "load_or_exit", lambda _: _spec())
    monkeypatch.setattr(
        cli,
        "_create_protocol_app",
        lambda protocol, spec, factory, token: (
            captured.update(protocol=protocol) or object()
        ),
    )
    monkeypatch.setattr(
        cli.uvicorn, "run", lambda app, **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(
        agent_factory,
        "build_runtime",
        lambda _: (_ for _ in ()).throw(AssertionError("startup only")),
    )
    for name in (
        "AGENTKIT_PROTOCOL",
        "AGENTKIT_BIND",
        "AGENTKIT_AUTH_TOKEN",
        "AGENTKIT_PORT",
        "AGENTKIT_A2A_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    cli.run(agent_factory, ["--protocol", "a2a", "--config", "agent.yaml"])
    assert captured["protocol"] == "a2a"
    assert captured["port"] == 8080


def test_a2a_runs_real_pydantic_agent_and_preserves_text_context(monkeypatch):
    from agentkit_serve_common.a2a import create_a2a_app

    monkeypatch.setattr(
        agent_factory,
        "build_model",
        lambda _: TestModel(custom_output_text="pydantic answer"),
    )
    app = create_a2a_app(_spec(), agent_factory, advertised_url="http://agent.test/")

    async def exercise():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://agent.test"
            ) as client:
                first = await client.post(
                    "/",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "message/send",
                        "params": {
                            "message": {
                                "kind": "message",
                                "role": "user",
                                "messageId": "one",
                                "parts": [{"kind": "text", "text": "hello"}],
                            },
                        },
                    },
                )
                task = first.json()["result"]
                assert task["status"]["state"] == "completed"
                assert task["artifacts"][0]["parts"][0]["text"] == "pydantic answer"
                second = await client.post(
                    "/",
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "message/send",
                        "params": {
                            "message": {
                                "kind": "message",
                                "role": "user",
                                "messageId": "two",
                                "contextId": task["contextId"],
                                "parts": [{"kind": "text", "text": "follow up"}],
                            },
                        },
                    },
                )
                followup = second.json()["result"]
                assert followup["status"]["state"] == "completed"
                assert followup["contextId"] == task["contextId"]
                assert followup["id"] != task["id"]

    asyncio.run(exercise())


def test_a2a_cancels_real_stdio_mcp_run_then_accepts_another_turn(
    monkeypatch, tmp_path
):
    import os
    import sys

    from agentkit_serve_common.a2a import create_a2a_app

    script = tmp_path / "mcp_fixture.py"
    pid_file = tmp_path / "pid"
    started = tmp_path / "started"
    stopped = tmp_path / "stopped"
    script.write_text(
        """
import asyncio
import os
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP
pid, started, stopped = map(Path, sys.argv[1:])
pid.write_text(str(os.getpid()))
server = FastMCP("a2a-cancel-fixture")
@server.tool()
async def block_once() -> str:
    if started.exists():
        return "tool recovered"
    started.write_text("started")
    try:
        await asyncio.Future()
    finally:
        stopped.write_text("stopped")
server.run(transport="stdio")
""",
        encoding="utf-8",
    )
    data = _spec().model_dump(by_alias=True)
    data["tools"] = [
        {
            "name": "fixture",
            "command": [
                sys.executable,
                str(script),
                str(pid_file),
                str(started),
                str(stopped),
            ],
            "env": [],
        }
    ]
    spec = AgentSpec.model_validate(data)
    monkeypatch.setattr(
        agent_factory,
        "build_model",
        lambda _: TestModel(custom_output_text="recovered answer"),
    )
    app = create_a2a_app(spec, agent_factory, advertised_url="http://agent.test/")
    child_pid = None

    async def wait_for_file(path):
        async with asyncio.timeout(10):
            while not path.exists():
                await asyncio.sleep(0.01)

    async def exercise():
        nonlocal child_pid
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://agent.test"
            ) as client:
                response = await client.post(
                    "/",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "message/send",
                        "params": {
                            "message": {
                                "kind": "message",
                                "role": "user",
                                "messageId": "blocked",
                                "parts": [{"kind": "text", "text": "use the tool"}],
                            },
                            "configuration": {"blocking": False},
                        },
                    },
                )
                task = response.json()["result"]
                await wait_for_file(started)
                child_pid = int(pid_file.read_text())
                async with asyncio.timeout(10):
                    canceled = await client.post(
                        "/",
                        json={
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tasks/cancel",
                            "params": {"id": task["id"]},
                        },
                    )
                assert canceled.json()["result"]["status"]["state"] == "canceled"
                # MCP ClientSession cancellation drops the local waiter; it does
                # not guarantee a notifications/cancelled acknowledgement.
                async with asyncio.timeout(10):
                    recovered = await client.post(
                        "/",
                        json={
                            "jsonrpc": "2.0",
                            "id": 3,
                            "method": "message/send",
                            "params": {
                                "message": {
                                    "kind": "message",
                                    "role": "user",
                                    "messageId": "after-cancel",
                                    "parts": [
                                        {"kind": "text", "text": "use the tool again"}
                                    ],
                                },
                            },
                        },
                    )
                assert recovered.json()["result"]["status"]["state"] == "completed"
                assert (
                    recovered.json()["result"]["artifacts"][0]["parts"][0]["text"]
                    == "recovered answer"
                )
        # The lifespan owns the MCP subprocess; cancellation is not a claim that
        # remote side effects rolled back or the shared subprocess exited early.
        if child_pid is not None:
            async with asyncio.timeout(5):
                while True:
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        break
                    await asyncio.sleep(0.01)

    asyncio.run(exercise())
