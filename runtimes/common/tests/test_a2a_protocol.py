"""Standalone A2A boundaries; SDK models are the wire contract."""

from __future__ import annotations

import json
import logging
from uuid import uuid4

import pytest

pytest.importorskip("a2a")
from a2a.types import Message, MessageSendParams, Part, TextPart
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.runtime import AgentRunError, RunResult
from fastapi.testclient import TestClient


def spec():
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "a2a-test"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://secret-model.example/v1",
                "name": "secret-model",
            },
            "instructions": "secret instructions",
            "env": [{"name": "SECRET_TOKEN"}],
            "expose": {"openai": True, "port": 8080},
        }
    )


class Runtime:
    def __init__(self):
        self.requests = []
        self.entered = self.exited = False
        self.failure = None
        self.output = None

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *args):
        self.exited = True

    async def run(self, request):
        self.requests.append(request)
        if self.failure:
            raise self.failure
        return RunResult(
            text=self.output if self.output is not None else f"answer: {request.prompt}"
        )


class Factory:
    def __init__(self, runtime=None):
        self.runtime = runtime or Runtime()

    def supports_a2a(self):
        return True

    def build_runtime(self, spec):
        return self.runtime


def app(factory=None, **kwargs):
    # Lazy import allows a meaningful RED assertion instead of collection failure.
    import agentkit_serve_common.a2a as core

    return core.create_a2a_app(spec(), factory or Factory(), **kwargs)


def params(text="hello", **kwargs):
    return MessageSendParams(
        message=Message(
            message_id=str(uuid4()),
            role="user",
            parts=[Part(root=TextPart(text=text))],
            **kwargs,
        )
    ).model_dump(mode="json", by_alias=True, exclude_none=True)


def rpc(client, method="message/send", payload=None, **kwargs):
    headers = {"content-type": "application/json", **kwargs.pop("headers", {})}
    body = {
        "jsonrpc": "2.0",
        "id": "test",
        "method": method,
        "params": payload if payload is not None else params(),
    }
    return client.post("/", content=json.dumps(body), headers=headers, **kwargs)


def test_explicit_capability_required_before_runtime_creation():
    class Unsupported:
        def build_runtime(self, spec):
            pytest.fail("allocated unsupported runtime")

    with pytest.raises(ValueError, match="support.*A2A"):
        app(Unsupported())


def test_public_cards_aliases_auth_and_secret_filtering():
    factory = Factory()
    with TestClient(
        app(
            factory,
            auth_token="private-token",
            advertised_url="https://agent.example/rpc",
        )
    ) as client:
        first = client.get(
            "/.well-known/agent-card.json", headers={"host": "attacker.example"}
        )
        assert first.status_code == 200
        card = first.json()
        assert card == client.get("/.well-known/agent.json").json()
        assert card["url"] == "https://agent.example/rpc"
        assert card["protocolVersion"] == "0.3.0"
        assert card["defaultInputModes"] == card["defaultOutputModes"] == ["text/plain"]
        assert card["capabilities"]["streaming"] is True
        assert card["capabilities"].get("pushNotifications", False) is False
        assert card["securitySchemes"]["bearer"]["scheme"] == "bearer"
        for secret in (
            "secret instructions",
            "secret-model",
            "SECRET_TOKEN",
            "private-token",
        ):
            assert secret not in first.text
        for headers in (
            {},
            {"authorization": "Bearer wrong"},
            {"x-user-id": "trusted"},
        ):
            assert rpc(client, headers=headers).status_code == 401
        headers = {"authorization": "Bearer private-token"}
        task = rpc(client, headers=headers).json()["result"]
        for method in ("tasks/get", "tasks/cancel", "tasks/resubscribe"):
            assert rpc(client, method, {"id": task["id"]}).status_code == 401
        assert (
            rpc(client, "tasks/get", {"id": task["id"]}, headers=headers).json()[
                "result"
            ]["id"]
            == task["id"]
        )
        assert factory.runtime.entered
    assert factory.runtime.exited


def test_send_get_context_and_neutral_history():
    factory = Factory()
    with TestClient(app(factory)) as client:
        first = rpc(client).json()["result"]
        assert first["status"]["state"] == "completed"
        assert first["artifacts"][0]["parts"] == [
            {"kind": "text", "text": "answer: hello"}
        ]
        assert rpc(client, "tasks/get", {"id": first["id"]}).json()["result"] == first
        second = rpc(
            client, payload=params("again", context_id=first["contextId"])
        ).json()["result"]
        assert second["id"] != first["id"]
        assert second["contextId"] == first["contextId"]
        request = factory.runtime.requests[-1]
        assert [(turn.role, turn.text) for turn in request.history] == [
            ("user", "hello"),
            ("assistant", "answer: hello"),
        ]
        assert (
            request.session_id == first["contextId"] and request.turn_id == second["id"]
        )
        assert not request.env and not request.config


@pytest.mark.parametrize(
    "change",
    [
        {"message": {"role": "agent"}},
        {"message": {"contextId": "unknown"}},
        {"message": {"taskId": "named"}},
        {"message": {"parts": [{"kind": "data", "data": {"secret": "x"}}]}},
        {
            "message": {
                "parts": [{"kind": "file", "file": {"uri": "https://example/file"}}]
            }
        },
        {"message": {"parts": [{"kind": "text", "text": "   "}]}},
        {"message": {"metadata": {"env": {"TOKEN": "secret"}}}},
        {"message": {"parts": [{"kind": "text", "text": "\ud800"}]}},
        {"message": {"referenceTaskIds": ["client-tool-task"]}},
        {
            "message": {
                "parts": [
                    {"kind": "text", "text": "hello", "metadata": {"tool": "payload"}}
                ]
            }
        },
        {"message": {"extensions": ["https://example/credentials"]}},
        {"metadata": {"credentials": "secret"}},
        {"configuration": {"acceptedOutputModes": ["application/json"]}},
        {
            "configuration": {
                "pushNotificationConfig": {"url": "https://callback.example"}
            }
        },
    ],
)
def test_rejects_unsupported_input_before_execution(change):
    factory = Factory()
    with TestClient(app(factory)) as client:
        payload = params()
        for key, value in change.items():
            if key == "message":
                payload[key].update(value)
            else:
                payload[key] = value
        response = rpc(client, payload=payload).json()
        assert response["error"]["code"] in (-32602, -32004, -32005)
        assert not factory.runtime.requests


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("secret upstream detail"), AgentRunError("secret fatal", fatal=True)],
)
def test_sanitized_failure_does_not_commit_and_fatal_fails_closed(failure, caplog):
    factory = Factory()
    with TestClient(app(factory)) as client:
        first = rpc(client).json()["result"]
        factory.runtime.failure = failure
        failed = rpc(
            client, payload=params("bad", context_id=first["contextId"])
        ).json()["result"]
        assert failed["status"]["state"] == "failed"
        assert "secret" not in str(failed)
        assert failed["id"] in caplog.text and type(failure).__name__ in caplog.text
        assert (
            "secret upstream detail" not in caplog.text
            and "secret fatal" not in caplog.text
        )
        factory.runtime.failure = None
        following = rpc(
            client, payload=params("next", context_id=first["contextId"])
        ).json()
        if isinstance(failure, AgentRunError):
            assert "error" in following
            assert client.get("/.well-known/agent-card.json").status_code == 503
        else:
            assert following["result"]["status"]["state"] == "completed"
            assert [turn.text for turn in factory.runtime.requests[-1].history] == [
                "hello",
                "answer: hello",
            ]


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example/",
        "https://user:secret@example/",
        "https://example/#x",
        "https://example/?token=secret",
        "/relative",
    ],
)
def test_rejects_unsafe_advertised_url(url):
    with pytest.raises(ValueError, match="URL"):
        app(advertised_url=url)


def test_startup_failure_cannot_serve_ready_card():
    class Broken(Runtime):
        async def __aenter__(self):
            raise RuntimeError("secret startup")

    with pytest.raises(RuntimeError), TestClient(app(Factory(Broken()))):
        pytest.fail("startup succeeded")
    application = app()
    # TestClient without its context manager deliberately omits ASGI lifespan.
    client = TestClient(application)
    assert client.get("/.well-known/agent-card.json").status_code == 503
    assert rpc(client).status_code == 503
    client.close()


def test_task_context_history_and_output_limits(monkeypatch):
    from agentkit_serve_common.a2a import core

    monkeypatch.setattr(core, "MAX_TASKS", 2)
    monkeypatch.setattr(core, "MAX_CONTEXTS", 2)
    monkeypatch.setattr(core, "MAX_HISTORY_TURNS", 4)
    monkeypatch.setattr(core, "MAX_TEXT_BYTES", 24)
    factory = Factory()
    with TestClient(app(factory)) as client:
        first = rpc(client, payload=params("1")).json()["result"]
        context = first["contextId"]
        for text in ("2", "3", "4"):
            task = rpc(client, payload=params(text, context_id=context)).json()[
                "result"
            ]
            assert task["status"]["state"] == "completed"
        assert [turn.text for turn in factory.runtime.requests[-1].history] == [
            "2",
            "answer: 2",
            "3",
            "answer: 3",
        ]
        assert (
            rpc(client, "tasks/get", {"id": first["id"]}).json()["error"]["code"]
            == -32001
        )
        rpc(client, payload=params("new"))
        rpc(client, payload=params("newer"))
        assert (
            rpc(client, payload=params("expired", context_id=context)).json()["error"][
                "code"
            ]
            == -32602
        )
        assert rpc(client, payload=params("x" * 25)).json()["error"]["code"] == -32602
        factory.runtime.output = "y" * 25
        assert rpc(client).json()["result"]["status"]["state"] == "failed"


def test_oversized_output_has_payload_free_diagnostic(monkeypatch, caplog):
    from agentkit_serve_common.a2a import core

    monkeypatch.setattr(core, "MAX_TEXT_BYTES", 24)
    factory = Factory()
    factory.runtime.output = "private-model-output" * 3
    caplog.set_level("WARNING")
    with TestClient(app(factory)) as client:
        task = rpc(client).json()["result"]
    assert task["status"]["state"] == "failed"
    assert "A2A output exceeded limit" in caplog.text
    assert task["id"] in caplog.text
    assert "private-model-output" not in caplog.text


def test_push_get_unknown_and_terminal_cancel_have_specific_errors():
    with TestClient(app()) as client:
        task = rpc(client).json()["result"]
        assert (
            rpc(client, "tasks/get", {"id": "missing"}).json()["error"]["code"]
            == -32001
        )
        assert (
            rpc(client, "tasks/cancel", {"id": task["id"]}).json()["error"]["code"]
            == -32002
        )
        assert (
            rpc(client, "tasks/pushNotificationConfig/get", {"id": task["id"]}).json()[
                "error"
            ]["code"]
            == -32004
        )


def test_malformed_input_and_debug_logs_do_not_echo_secrets(caplog):
    with TestClient(app()) as client:
        with caplog.at_level(logging.DEBUG, logger="a2a"):
            malformed = params()
            malformed["message"]["role"] = "secret-invalid-role"
            response = rpc(client, payload=malformed)
            assert response.json()["error"]["code"] == -32602
            assert "secret-invalid-role" not in response.text
            successful = rpc(client, payload=params("secret prompt"))
            assert successful.json()["result"]["status"]["state"] == "completed"
        assert "secret prompt" not in caplog.text
        assert "secret-invalid-role" not in caplog.text


def test_request_byte_limit_includes_chunked_bodies(monkeypatch):
    from agentkit_serve_common.a2a import core

    monkeypatch.setattr(core, "MAX_REQUEST_BYTES", 1024)
    factory = Factory()
    with TestClient(app(factory)) as client:
        response = client.post(
            "/",
            content=iter([b"x" * 600, b"y" * 600]),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413
        assert not factory.runtime.requests


def test_history_byte_bound_keeps_complete_pairs(monkeypatch):
    from agentkit_serve_common.a2a import core

    monkeypatch.setattr(core, "MAX_HISTORY_BYTES", 19)
    factory = Factory()
    with TestClient(app(factory)) as client:
        first = rpc(client, payload=params("1")).json()["result"]
        rpc(client, payload=params("2", context_id=first["contextId"]))
        rpc(client, payload=params("3", context_id=first["contextId"]))
        assert [turn.text for turn in factory.runtime.requests[-1].history] == [
            "2",
            "answer: 2",
        ]
