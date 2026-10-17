from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel as PydanticTestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.providers.openai import OpenAIProvider

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec, ToolSpec
from agentkit_serve_common.conversation import RunRequest


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _wait_for_pid(pid_path: Path, *, timeout: float = 3.0) -> int:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if pid_path.exists():
            return int(pid_path.read_text(encoding="utf-8"))
        await asyncio.sleep(0.01)
    raise AssertionError("stdio MCP child did not publish its PID")


async def _wait_for_process_exit(pid: int, *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if not _process_exists(pid):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"stdio MCP child process {pid} was not cleaned up")


def test_stdio_tool_call_honors_mcp_timeout_and_cleans_up_child(monkeypatch, tmp_path):
    server_script = tmp_path / "blocked_mcp_server.py"
    pid_path = tmp_path / "blocked_mcp_server.pid"
    server_script.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")

for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "blocked-tool-test", "version": "1.0"},
            },
        }
        print(json.dumps(response), flush=True)
    elif method == "tools/list":
        response = {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "tools": [
                    {
                        "name": "block_forever",
                        "description": "Never returns",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            },
        }
        print(json.dumps(response), flush=True)
    elif method == "tools/call":
        while True:
            time.sleep(3600)
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTKIT_MCP_TIMEOUT", "0.3")
    tool = ToolSpec(
        name="blocked",
        command=[sys.executable, str(server_script), str(pid_path)],
        env=[],
    )
    child_pid: int | None = None

    async def exercise() -> None:
        nonlocal child_pid
        toolset = agent_factory.build_tool_server(tool)
        wrapped = toolset.wrapped
        async with wrapped:
            child_pid = await _wait_for_pid(pid_path)
            started = time.monotonic()
            with pytest.raises(Exception):
                await asyncio.wait_for(
                    wrapped.direct_call_tool("block_forever", {}),
                    timeout=2.0,
                )
            assert time.monotonic() - started < 1.2
            assert _process_exists(child_pid)

        await _wait_for_process_exit(child_pid)

    try:
        asyncio.run(exercise())
    finally:
        if child_pid is not None and _process_exists(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_legacy_stdio_server_receives_init_and_read_timeout(monkeypatch):
    captured: dict[str, object] = {}

    class _LegacyServer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(agent_factory, "MCPServerStdio", _LegacyServer)
    monkeypatch.setenv("AGENTKIT_MCP_TIMEOUT", "1.25")

    server = agent_factory.build_tool_server(
        ToolSpec(name="legacy", command=["legacy-mcp", "--stdio"], env=[])
    )

    assert isinstance(server, _LegacyServer)
    assert captured["timeout"] == 1.25
    assert captured["read_timeout"] == 1.25


class _TaskBoundToolset(AbstractToolset[None]):
    """Exercise the same task-bound cancel-scope contract as MCP clients."""

    def __init__(self, *, close_gate: asyncio.Event | None = None) -> None:
        self.close_gate = close_gate
        self.closing = asyncio.Event()
        self.enter_task = None
        self.exit_task = None
        self.close_count = 0

    @property
    def id(self) -> str:
        return f"task-bound-{id(self)}"

    async def __aenter__(self):
        self.enter_task = asyncio.current_task()
        self.scope = anyio.CancelScope()
        self.scope.__enter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_task = asyncio.current_task()
        self.closing.set()
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.close_count += 1
        return self.scope.__exit__(exc_type, exc, tb)

    async def get_tools(self, ctx: Any) -> dict[str, Any]:
        return {}

    async def call_tool(self, name, tool_args, ctx, tool):
        raise AssertionError("lifecycle tests must not execute tools")


def _task_bound_runtime(toolsets):
    return agent_factory.PydanticRuntime(
        Agent(PydanticTestModel(custom_output_text="ok"), toolsets=toolsets),
    )


def test_pydantic_runtime_exits_task_bound_scopes_from_a_different_request_task():
    async def run():
        toolset = _TaskBoundToolset()
        runtime = _task_bound_runtime([toolset])
        assert await runtime.__aenter__() is runtime
        caller = asyncio.current_task()
        await asyncio.create_task(runtime.__aexit__(None, None, None))
        assert toolset.enter_task is toolset.exit_task
        assert toolset.enter_task is not caller
        assert toolset.close_count == 1
        await runtime.__aexit__(None, None, None)
        assert toolset.close_count == 1

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_pydantic_runtime_partial_entry_unwinds_on_owner_task(cancel):
    async def run():
        started = asyncio.Event()
        primary = RuntimeError("toolset startup failed")

        class FailingToolset(_TaskBoundToolset):
            async def __aenter__(self):
                started.set()
                if cancel:
                    await asyncio.Future()
                raise primary

        toolset = _TaskBoundToolset()
        runtime = _task_bound_runtime([toolset, FailingToolset()])
        entering = asyncio.create_task(runtime.__aenter__())
        await asyncio.wait_for(started.wait(), 3)
        if cancel:
            entering.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError) as caught:
            await asyncio.wait_for(entering, 3)
        if not cancel:
            assert caught.value is primary
        assert toolset.enter_task is toolset.exit_task
        assert toolset.close_count == 1
        await runtime.__aexit__(None, None, None)
        assert toolset.close_count == 1

    asyncio.run(run())


def test_pydantic_runtime_cancelled_exit_waits_for_owner_cleanup():
    async def run():
        gate = asyncio.Event()
        toolset = _TaskBoundToolset(close_gate=gate)
        runtime = _task_bound_runtime([toolset])
        await runtime.__aenter__()
        exiting = asyncio.create_task(runtime.__aexit__(None, None, None))
        await asyncio.wait_for(toolset.closing.wait(), 3)
        exiting.cancel()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(exiting, 3)
        assert toolset.enter_task is toolset.exit_task
        assert toolset.close_count == 1

    asyncio.run(run())


def test_auto_fallback_closes_responses_mcp_child_before_chat_request(monkeypatch, tmp_path):
    server_script = tmp_path / "lifecycle_mcp_server.py"
    pid_path = tmp_path / "lifecycle_mcp_server.pids"
    server_script.write_text(
        """
import json
import os
import sys

with open(sys.argv[1], "a", encoding="utf-8") as pids:
    pids.write(str(os.getpid()) + "\\n")
for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        result = {
            "protocolVersion": message["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "lifecycle-test", "version": "1.0"},
        }
    elif method == "tools/list":
        result = {"tools": []}
    else:
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
""".lstrip(),
        encoding="utf-8",
    )
    spec = AgentSpec.model_validate({
        "abiVersion": "v0", "metadata": {"name": "auto-lifecycle-test"},
        "model": {"provider": "openai-compatible", "baseURL": "http://model.test/v1", "name": "local-model"},
        "instructions": "Be helpful.",
        "tools": [{"name": "lifecycle", "command": [sys.executable, str(server_script), str(pid_path)], "env": []}],
        "expose": {"openai": True, "port": 8080},
    })
    monkeypatch.setenv("AGENTKIT_MODEL_API", "auto")
    requests = []

    def pids():
        return [int(pid) for pid in pid_path.read_text(encoding="utf-8").splitlines()]

    def handle(request):
        requests.append(request.url.path)
        children = pids()
        if request.url.path == "/v1/responses":
            assert len(children) == 1 and _process_exists(children[0])
            return httpx.Response(404, text="Not Found")
        assert request.url.path == "/v1/chat/completions"
        assert len(children) == 2
        assert not _process_exists(children[0])
        assert _process_exists(children[1])
        return httpx.Response(200, json={
            "id": "chat-test", "object": "chat.completion", "created": 1, "model": "local-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            monkeypatch.setattr(agent_factory, "OpenAIProvider", lambda **kwargs: OpenAIProvider(http_client=client, **kwargs))
            async with agent_factory.build_runtime(spec) as runtime:
                result = await asyncio.wait_for(asyncio.create_task(runtime.run(RunRequest("hello"))), 5)
                assert result.text == "ok"
                assert runtime.state.selected == "chat_completions"
            for pid in pids():
                await _wait_for_process_exit(pid)
        assert requests == ["/v1/responses", "/v1/chat/completions"]
        assert os.environ["AGENTKIT_MODEL_API"] == "auto"

    try:
        asyncio.run(run())
    finally:
        if pid_path.exists():
            for pid in pids():
                if _process_exists(pid):
                    os.kill(pid, signal.SIGKILL)
