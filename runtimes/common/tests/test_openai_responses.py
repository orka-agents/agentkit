from __future__ import annotations

from types import TracebackType

import pytest
from fastapi.testclient import TestClient

from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.runtime import AgentRunError, RunResult, RuntimeSession
from agentkit_serve_common.server import create_app


def _spec() -> AgentSpec:
    return AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "dual-api"},
        "model": {
            "provider": "openai-compatible", "baseURL": "http://localhost:11434/v1", "name": "local-model",
        },
        "instructions": "Baked instructions.",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })


class Runtime:
    def __init__(self) -> None:
        self.requests: list[RunRequest] = []
        self.entered = self.exited = 0
        self.error: AgentRunError | None = None

    async def __aenter__(self) -> RuntimeSession:
        self.entered += 1
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None,
    ) -> None:
        self.exited += 1

    async def run(self, request: RunRequest) -> RunResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return RunResult(
            text=f"echo: {request.prompt}",
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )


class Factory:
    def __init__(self) -> None:
        self.runtime = Runtime()
        self.built = 0

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        self.built += 1
        return self.runtime


def test_both_endpoints_share_one_runtime_lifespan_and_auth():
    factory = Factory()
    with TestClient(create_app(_spec(), factory, auth_token="test-token")) as client:
        assert client.get("/healthz").status_code == 200
        for path, data in (
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "chat"}]}),
            ("/v1/responses", {"input": "responses"}),
        ):
            before = len(factory.runtime.requests)
            assert client.post(path, json=data).status_code == 401
            assert client.post(path, json=data, headers={"Authorization": "Bearer wrong"}).status_code == 401
            assert len(factory.runtime.requests) == before
            response = client.post(path, json=data, headers={"Authorization": "Bearer test-token"})
            assert response.status_code == 200, response.text
        assert factory.built == factory.runtime.entered == 1
        assert factory.runtime.exited == 0
    assert factory.runtime.exited == 1
    assert [request.prompt for request in factory.runtime.requests] == ["chat", "responses"]


def test_responses_completed_output_and_usage():
    with TestClient(create_app(_spec(), Factory())) as client:
        body = client.post("/v1/responses", json={"model": "ignored-client-model", "input": "hello"}).json()
        other = client.post("/v1/responses", json={"input": "other", "store": False}).json()
    assert body["id"].startswith("resp_") and body["id"] != other["id"]
    assert body["object"] == "response"
    assert body["model"] == "local-model"
    assert body["status"] == "completed"
    assert isinstance(body["created_at"], int)
    assert body["store"] is False
    assert body["parallel_tool_calls"] is False
    assert body["tool_choice"] == "auto"
    assert body["tools"] == []
    assert body["error"] is body["incomplete_details"] is None
    assert body["output"] == [{
        "id": body["output"][0]["id"], "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "echo: hello", "annotations": []}],
    }]
    assert body["output"][0]["id"].startswith("msg_")
    assert body["usage"] == {
        "input_tokens": 2, "output_tokens": 3, "total_tokens": 5,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def test_responses_history_instructions_phase_and_session(monkeypatch):
    monkeypatch.setenv("FOUNDRY_AGENT_SESSION_ID", "must-not-use-foundry-identity")
    factory = Factory()
    data = {
        "input": [
            {"role": "system", "content": "client system"},
            {"role": "developer", "content": [{"type": "input_text", "text": "client developer"}]},
            {"role": "user", "content": "old user"},
            {"type": "message", "role": "assistant", "phase": "final_answer", "content": [
                {"type": "output_text", "text": "old assistant", "annotations": []},
            ]},
            {"role": "user", "content": [
                {"type": "input_text", "text": "new "}, {"type": "input_text", "text": "user"},
            ]},
        ],
        "instructions": "request instructions",
        "session_id": "ignored-body-identity",
    }
    with TestClient(create_app(_spec(), factory)) as client:
        response = client.post("/v1/responses?session_id=ignored-query-identity", json=data,
                               headers={"X-AgentKit-Session-Id": "  shared-session  "})
        assert response.status_code == 200, response.text
        # A transport identity must never replace the explicit caller history.
        assert client.post("/v1/responses", json={"input": "fresh"},
                           headers={"X-AgentKit-Session-Id": "shared-session"}).status_code == 200
    request = factory.runtime.requests[0]
    assert request.prompt == "new user"
    assert request.session_id == "shared-session"
    assert [(turn.role, turn.text, turn.phase) for turn in request.history] == [
        ("system", "request instructions", None), ("system", "client system", None),
        ("system", "client developer", None), ("user", "old user", None),
        ("assistant", "old assistant", "final_answer"),
    ]
    assert factory.runtime.requests[1].history == ()


@pytest.mark.parametrize("input_value", [
    "", [{"role": "user", "content": ""}],
    [{"role": "user", "content": [{"type": "input_text", "text": "", "input_text": "ignored-extra"}]}],
])
def test_responses_allows_empty_text_like_chat(input_value):
    factory = Factory()
    with TestClient(create_app(_spec(), factory)) as client:
        assert client.post("/v1/responses", json={"input": input_value}).status_code == 200
    assert factory.runtime.requests[0].prompt == ""


@pytest.mark.parametrize(("options", "code"), [
    ({"stream": True}, "stream_unsupported"),
    ({"tools": [{"type": "function", "name": "client_tool"}]}, "tools_unsupported"),
    ({"tool_choice": "required"}, "tool_choice_unsupported"),
    ({"tool_choice": {"type": "function", "name": "client_tool"}}, "tool_choice_unsupported"),
    ({"previous_response_id": "resp_old"}, "response_state_unsupported"),
    ({"conversation": "conv_old"}, "response_state_unsupported"),
    ({"conversation": {}}, "response_state_unsupported"),
    ({"background": True}, "background_unsupported"),
    ({"store": True}, "store_unsupported"),
])
def test_responses_rejects_unsupported_features_before_execution(options, code):
    factory = Factory()
    with TestClient(create_app(_spec(), factory)) as client:
        response = client.post("/v1/responses", json={"input": "do not execute", **options})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    assert factory.runtime.requests == []


@pytest.mark.parametrize("tool_choice", [None, "", "none", "auto"])
def test_responses_allows_empty_tool_selection(tool_choice):
    with TestClient(create_app(_spec(), Factory())) as client:
        response = client.post("/v1/responses", json={"input": "hello", "tools": [], "tool_choice": tool_choice})
    assert response.status_code == 200


@pytest.mark.parametrize("input_value", [
    None, 1, {}, [], ["raw item"], [{"role": "user"}],
    [{"role": "assistant", "content": "not a final user"}],
    [{"role": "unknown", "content": "secret-canary"}],
    [{"type": "function_call_output", "call_id": "call_1", "output": "secret-canary"}],
    [{"type": "item_reference", "id": "secret-canary"}],
    [{"role": "user", "content": [{"type": "input_image", "image_url": "secret-canary"}]}],
    [{"role": "user", "content": [{"type": "input_file", "file_data": "secret-canary"}]}],
    [{"role": "user", "content": [{"type": "input_text", "text": {"secret": "secret-canary"}}]}],
    [{"role": "assistant", "content": "old", "phase": "secret-canary"}, {"role": "user", "content": "new"}],
])
def test_responses_rejects_invalid_or_unsupported_input_without_echoing_it(input_value):
    factory = Factory()
    with TestClient(create_app(_spec(), factory)) as client:
        response = client.post("/v1/responses", json={"input": input_value})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "secret-canary" not in response.text
    assert factory.runtime.requests == []


@pytest.mark.parametrize("body", ['{"input":"secret-canary",', '{}', '[]'])
def test_responses_invalid_body_uses_safe_openai_error_envelope(body):
    factory = Factory()
    with TestClient(create_app(_spec(), factory)) as client:
        response = client.post("/v1/responses", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "secret-canary" not in response.text
    assert factory.runtime.requests == []


@pytest.mark.parametrize("fatal", [False, True])
def test_responses_uses_shared_runtime_error_and_health_handling(fatal):
    factory = Factory()
    factory.runtime.error = AgentRunError("model unavailable", status=503, code="ModelUnavailable", fatal=fatal)
    with TestClient(create_app(_spec(), factory)) as client:
        response = client.post("/v1/responses", json={"input": "hi"})
        assert response.status_code == 503
        assert response.json()["error"] == {
            "message": "model unavailable", "type": "agent_error", "code": "ModelUnavailable",
        }
        assert client.get("/healthz").status_code == (503 if fatal else 200)
