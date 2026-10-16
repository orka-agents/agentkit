"""A2A startup selection must not weaken the existing network posture."""

from __future__ import annotations

import pytest
from agentkit_serve_common import cli
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.runtime import OfflineEchoRuntimeFactory


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "a2a-cli"},
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


class UnsupportedFactory:
    def build_runtime(self, spec):
        raise AssertionError("unsupported A2A must not allocate runtime resources")


class SupportedFactory(OfflineEchoRuntimeFactory):
    def supports_a2a(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "AGENTKIT_PROTOCOL",
        "AGENTKIT_BIND",
        "AGENTKIT_AUTH_TOKEN",
        "AGENTKIT_PORT",
        "AGENTKIT_A2A_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "load_or_exit", lambda _: _spec())


def test_a2a_flag_is_recognized():
    assert cli._parse_args(["--protocol", "a2a"]).protocol == "a2a"


def test_a2a_environment_is_recognized(monkeypatch):
    monkeypatch.setenv("AGENTKIT_PROTOCOL", "a2a")
    assert cli._resolve_protocol(None) == "a2a"


def test_unsupported_a2a_factory_fails_before_server_or_runtime(monkeypatch, capsys):
    monkeypatch.setenv("AGENTKIT_PROTOCOL", "a2a")
    monkeypatch.setattr(
        cli.uvicorn,
        "run",
        lambda *_args, **_kwargs: pytest.fail("server must not start"),
    )
    with pytest.raises(SystemExit) as exc:
        cli.run(UnsupportedFactory(), ["--config", "agent.yaml"])
    assert exc.value.code == 2
    assert "does not support A2A" in capsys.readouterr().err


def test_a2a_keeps_nonloopback_bearer_requirement(monkeypatch, capsys):
    monkeypatch.setenv("AGENTKIT_PROTOCOL", "a2a")
    monkeypatch.setenv("AGENTKIT_BIND", "0.0.0.0")
    monkeypatch.setattr(
        cli.uvicorn,
        "run",
        lambda *_args, **_kwargs: pytest.fail("server must not start"),
    )
    with pytest.raises(SystemExit) as exc:
        cli.run(SupportedFactory(), ["--config", "agent.yaml"])
    assert exc.value.code == 2
    assert "without authentication" in capsys.readouterr().err


def test_nonloopback_a2a_requires_operator_advertised_url(monkeypatch, capsys):
    monkeypatch.setenv("AGENTKIT_PROTOCOL", "a2a")
    monkeypatch.setenv("AGENTKIT_BIND", "0.0.0.0")
    monkeypatch.setenv("AGENTKIT_AUTH_TOKEN", "test-token")
    with pytest.raises(SystemExit) as exc:
        cli.run(SupportedFactory(), ["--config", "agent.yaml"])
    assert exc.value.code == 2
    assert "AGENTKIT_A2A_URL" in capsys.readouterr().err


def test_a2a_invalid_operator_url_has_normal_cli_error(monkeypatch, capsys):
    pytest.importorskip("a2a")
    monkeypatch.setenv("AGENTKIT_A2A_URL", "agent:8080")
    with pytest.raises(SystemExit) as exc:
        cli.run(SupportedFactory(), ["--protocol", "a2a", "--config", "agent.yaml"])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "agentkit-serve:" in error and "absolute HTTP(S) URL" in error
    assert "Traceback" not in error


@pytest.mark.parametrize(
    "port,url", [("9191", None), ("9191", "https://agents.example/a2a/")]
)
def test_a2a_cli_passes_resolved_or_operator_owned_discovery_url(
    monkeypatch, port, url
):
    pytest.importorskip("a2a")
    captured = {}
    monkeypatch.setenv("AGENTKIT_PORT", port)
    if url:
        monkeypatch.setenv("AGENTKIT_A2A_URL", url)
    monkeypatch.setattr(
        cli.uvicorn, "run", lambda app, **kwargs: captured.update(app=app, **kwargs)
    )
    cli.run(SupportedFactory(), ["--protocol", "a2a", "--config", "agent.yaml"])
    assert captured["port"] == 9191
    # The SDK card is exposed publicly; test through its registered HTTP endpoint.
    import asyncio

    import httpx

    async def fetch_card():
        async with captured["app"].router.lifespan_context(captured["app"]):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=captured["app"]),
                base_url="http://localhost",
            ) as client:
                return await client.get("/.well-known/agent-card.json")

    response = asyncio.run(fetch_card())
    assert response.status_code == 200
    assert response.json()["url"] == (url or "http://localhost:9191/")
