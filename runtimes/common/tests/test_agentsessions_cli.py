from __future__ import annotations

import os

import pytest

from agentkit_serve_common import cli
from test_agentsessions_protocol import binding_file  # noqa: F401


def test_cli_help_explains_agentsessions_constraints_and_guide(capsys):
    assert cli._parse_args(["--protocol", "agentsessions"]).protocol == "agentsessions"
    with pytest.raises(SystemExit) as error:
        cli._parse_args(["--help"])
    assert error.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    help_text = " ".join(captured.out.lower().split())
    assert "agentsessions" in help_text
    assert "grpc" in help_text
    assert "text-only" in help_text
    assert "configuration" in help_text and "implementation" in help_text
    assert "digest" in help_text
    assert "no tools" in help_text
    assert "docs/agentsessions.md" in help_text


class ProviderTrap:
    def build_runtime(self, spec):
        pytest.fail("normal provider runtime must never be built")


@pytest.mark.parametrize("via_env", [False, True])
def test_cli_agentsessions_dispatch_skips_http_acp_and_required_env(binding_file, monkeypatch, via_env):
    captured = {}
    def trap(*args, **kwargs):
        pytest.fail("HTTP, ACP, or required/provider env resolution must not run")
    for name in ("load_or_exit", "run_acp_stdio", "_create_protocol_app"):
        monkeypatch.setattr(cli, name, trap)
    monkeypatch.setattr(cli.uvicorn, "run", trap)
    def serve(binding, **kwargs):
        captured.update(binding=binding, **kwargs)
    monkeypatch.setattr(cli, "run_agentsessions", serve, raising=False)
    monkeypatch.delenv("AGENTKIT_PROTOCOL", raising=False)
    monkeypatch.delenv("AGENTKIT_BIND", raising=False)
    monkeypatch.delenv("AGENTKIT_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("AGENTKIT_PORT", "12345")
    if via_env:
        monkeypatch.setenv("AGENTKIT_PROTOCOL", "agentsessions")
    argv = ["--config", str(binding_file[0])]
    if not via_env:
        argv += ["--protocol", "agentsessions"]
    cli.run(ProviderTrap(), argv)
    assert captured["binding"].spec.model.name == "host-model"
    assert captured["bind"] == "127.0.0.1"
    assert captured["port"] == 12345
    assert captured["auth_token"] is None
    assert captured["runner"] is None
    assert os.environ["AGENTKIT_PROTOCOL"] == "agentsessions"


@pytest.mark.parametrize("bind,token", [("0.0.0.0", None), ("::", None), ("0.0.0.0", "local-only-token")])
def test_cli_nonloopback_requires_auth(binding_file, monkeypatch, bind, token):
    captured = []
    monkeypatch.setattr(cli, "run_agentsessions", lambda *args, **kwargs: captured.append(kwargs), raising=False)
    monkeypatch.setenv("AGENTKIT_BIND", bind)
    monkeypatch.delenv("AGENTKIT_PORT", raising=False)
    monkeypatch.delenv("AGENTKIT_PROTOCOL", raising=False)
    if token:
        monkeypatch.setenv("AGENTKIT_AUTH_TOKEN", token)
        cli.run(ProviderTrap(), ["--config", str(binding_file[0]), "--protocol", "agentsessions"])
        assert captured[0]["auth_token"] == token
    else:
        monkeypatch.delenv("AGENTKIT_AUTH_TOKEN", raising=False)
        with pytest.raises(SystemExit) as error:
            cli.run(ProviderTrap(), ["--config", str(binding_file[0]), "--protocol", "agentsessions"])
        assert error.value.code == 2
        assert captured == []


def test_cli_passes_only_explicit_agentsessions_runner(binding_file, monkeypatch):
    captured = {}
    class Factory(ProviderTrap):
        async def run_agentsessions(self, binding, request, exchange):
            raise AssertionError("server entrypoint mocked only")
    factory = Factory()
    monkeypatch.setattr(cli, "run_agentsessions", lambda *args, **kwargs: captured.update(kwargs), raising=False)
    monkeypatch.setenv("AGENTKIT_BIND", "127.0.0.1")
    monkeypatch.delenv("AGENTKIT_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("AGENTKIT_PROTOCOL", raising=False)
    cli.run(factory, ["--config", str(binding_file[0]), "--protocol", "agentsessions"])
    assert captured["runner"] == factory.run_agentsessions
