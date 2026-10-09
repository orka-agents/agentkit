from __future__ import annotations

from types import TracebackType

from fastapi.testclient import TestClient

from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.runtime import AgentRunError, RunResult, RuntimeSession
from agentkit_serve_common.server import create_app


def _spec() -> AgentSpec:
    return AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "server-test"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })


class Runtime:
    def __init__(self):
        self.requests: list[RunRequest] = []

    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None

    async def run(self, request: RunRequest) -> RunResult:
        self.requests.append(request)
        return RunResult(text="ok")


class Factory:
    def __init__(self):
        self.runtime = Runtime()

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


def test_openai_facade_forwards_agentkit_session_header():
    factory = Factory()
    app = create_app(_spec(), factory)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"X-AgentKit-Session-Id": "local-session"},
        )

    assert resp.status_code == 200
    assert factory.runtime.requests[0].session_id == "local-session"


class FailingRuntime(Runtime):
    def __init__(self, *errors: AgentRunError):
        super().__init__()
        self.errors = list(errors)

    async def run(self, request: RunRequest) -> RunResult:
        raise self.errors.pop(0)


def test_openai_healthz_fails_only_after_fatal_run_error():
    factory = Factory()
    factory.runtime = FailingRuntime(
        AgentRunError("model service is unavailable", status=503, code="ModelUnavailable"),
        AgentRunError("MCP tool protocol failed", code="MCPToolProtocolError", fatal=True),
    )
    app = create_app(_spec(), factory)
    body = {"messages": [{"role": "user", "content": "hello"}]}
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=body).status_code == 503
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.post("/v1/chat/completions", json=body).status_code == 502
        health = client.get("/healthz")

    assert health.status_code == 503
    assert health.json() == {"status": "unhealthy"}
