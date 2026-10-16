"""Shared CLI / network-posture core for AgentKit runtime adapters.

Loads ``/agent/agent.yaml`` and selects one protocol skin. HTTP modes apply the
network posture and run uvicorn; ACP uses stdio. Each adapter's console script
calls :func:`run` with its own framework-specific ``agent_factory`` module, the
only per-adapter input.

Protocol modes:

* ``openai`` (default): ``/healthz``, ``/v1/models``, ``/v1/chat/completions``.
* ``foundry``: ``/readiness``, ``/invocations``, minimal non-streaming
  ``/responses``.
* ``orka``: observed-mode ``orka.harness.v1`` over HTTP+SSE.
* ``acp``: Orka-owned ACP protocol v1 over newline-delimited JSON-RPC on stdio.
* ``agentsessions``: keyless native protobuf Harness SPI over gRPC (h2c).
* ``a2a``: text-only A2A 0.3 JSON-RPC and task-lifecycle SSE (Pydantic AI).

Network posture:

* Bind ``127.0.0.1`` by default (``AGENTKIT_BIND`` overrides).
* A NON-loopback bind (e.g. ``0.0.0.0``) REQUIRES ``AGENTKIT_AUTH_TOKEN``.
* ``orka`` protocol always requires ``AGENTKIT_AUTH_TOKEN`` because its turn,
  event-stream, cancel, and output endpoints are bearer-authenticated.
* In ``openai`` mode, ``/v1/*`` requires ``Authorization: Bearer <token>`` when a
  token is set; ``/healthz`` stays open.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import NoReturn

import uvicorn

from .acp import (
    ACPConfigurationError,
    ACPProtocolError,
    load_verified_acp_runtime_binding,
    run_acp_stdio,
)
from .agentsessions import (
    AgentsessionsConfigurationError,
    load_verified_agentsessions_binding,
    run as run_agentsessions,
)
from .config import ConfigError, load, load_or_exit
from .foundry import create_foundry_app
from .orka import create_orka_app
from .runtime import RuntimeFactory
from .server import create_app

# Hosts that mean "loopback only" — a bind to any of these needs no auth token.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"})
_PROTOCOLS = frozenset({"a2a", "acp", "agentsessions", "openai", "foundry", "orka"})

DEFAULT_CONFIG_PATH = "/agent/agent.yaml"
DEFAULT_PORT = 8080
DEFAULT_FOUNDRY_PORT = 8088


def _is_loopback(bind: str) -> bool:
    return bind.strip().lower() in _LOOPBACK_HOSTS


def _fail(message: str) -> NoReturn:
    print(f"agentkit-serve: {message}", file=sys.stderr)
    raise SystemExit(2)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agentkit-serve",
        description="Serve an AgentKit agent over the selected protocol skin.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"path to agent.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--protocol",
        choices=sorted(_PROTOCOLS),
        default=None,
        help="protocol skin to expose (default: AGENTKIT_PROTOCOL or openai)",
    )
    return parser.parse_args(argv)


def _resolve_protocol(cli_protocol: str | None) -> str:
    protocol = (cli_protocol or os.environ.get("AGENTKIT_PROTOCOL") or "openai").strip().lower()
    if protocol not in _PROTOCOLS:
        _fail(
            f"unsupported protocol {protocol!r}; expected one of "
            + ", ".join(sorted(_PROTOCOLS))
        )
    return protocol


def _resolve_port(protocol: str, spec_port: int | None) -> int:
    raw = os.environ.get("AGENTKIT_PORT")
    if raw:
        try:
            port = int(raw)
        except ValueError:
            _fail(f"AGENTKIT_PORT must be an integer, got {raw!r}")
        if port < 1 or port > 65535:
            _fail(f"AGENTKIT_PORT {port} is out of range")
        return port

    if protocol == "foundry" and (spec_port in (None, 0, DEFAULT_PORT)):
        return DEFAULT_FOUNDRY_PORT
    return spec_port or DEFAULT_PORT

def _load_spec_or_exit(path: str, protocol: str):  # noqa: ANN001
    if protocol not in {"acp", "orka"}:
        return load_or_exit(path)
    try:
        return load(path)
    except ConfigError as exc:
        print(f"agentkit-serve: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _create_protocol_app(protocol: str, spec, factory: RuntimeFactory, auth_token: str | None):  # noqa: ANN001
    if protocol == "openai":
        return create_app(spec, factory, auth_token=auth_token)
    if protocol == "foundry":
        return create_foundry_app(spec, factory, auth_token=auth_token)
    if protocol == "orka":
        return create_orka_app(spec, factory, auth_token=auth_token)
    if protocol == "a2a":
        # Other protocols do not need the optional SDK, including in adapter images.
        try:
            from .a2a import create_a2a_app
        except ModuleNotFoundError as exc:
            if exc.name == "a2a" or (exc.name and exc.name.startswith("a2a.")):
                _fail("A2A requires the agentkit-serve-common[a2a] extra")
            raise
        url = os.environ.get("AGENTKIT_A2A_URL") or f"http://localhost:{_resolve_port(protocol, spec.expose.port)}/"
        try:
            return create_a2a_app(spec, factory, auth_token=auth_token, advertised_url=url)
        except ValueError as exc:
            _fail(str(exc))
    raise AssertionError(f"unknown protocol: {protocol}")


def run(factory: RuntimeFactory, argv: list[str] | None = None) -> None:
    """Entry point: serve an agent built by ``factory`` (the adapter's module)."""
    args = _parse_args(argv)

    protocol = _resolve_protocol(args.protocol)
    # Keep the resolved protocol visible to adapter factories for the full server
    # lifetime. Some runtime gates, such as Orka's offline conformance runtime,
    # are intentionally adapter-owned and read AGENTKIT_PROTOCOL when a turn later
    # builds a runtime session.
    os.environ["AGENTKIT_PROTOCOL"] = protocol
    if protocol == "a2a":
        supports_a2a = getattr(factory, "supports_a2a", None)
        if supports_a2a is None or not supports_a2a():
            _fail("this runtime adapter does not support A2A")
    if protocol == "acp":
        try:
            spec = load_verified_acp_runtime_binding(args.config)
        except (ConfigError, ACPConfigurationError, ACPProtocolError) as exc:
            _fail(str(exc))
        run_acp_stdio(spec, factory)
        return
    binding = None
    if protocol == "agentsessions":
        try:
            binding = load_verified_agentsessions_binding(args.config)
        except AgentsessionsConfigurationError as exc:
            _fail(str(exc))
        spec = binding.spec
    else:
        spec = _load_spec_or_exit(args.config, protocol)
    if spec.brokered_tools and protocol != "foundry":
        _fail(
            "brokeredTools require AGENTKIT_PROTOCOL=foundry (or --protocol foundry); "
            f"the {protocol!r} protocol cannot broker Foundry Responses tool calls"
        )

    # --- resolve bind/port ------------------------------------------------
    bind = os.environ.get("AGENTKIT_BIND", "127.0.0.1").strip()
    if not bind:
        bind = "127.0.0.1"
    port = _resolve_port(protocol, spec.expose.port)

    # --- network posture gate --------------------------------------------
    auth_token = os.environ.get("AGENTKIT_AUTH_TOKEN") or None
    if protocol == "orka" and not auth_token:
        _fail("AGENTKIT_PROTOCOL=orka requires AGENTKIT_AUTH_TOKEN for turn/event/cancel endpoints")
    if not _is_loopback(bind) and not auth_token:
        _fail(
            f"refusing to bind {bind!r} without authentication: set "
            f"AGENTKIT_AUTH_TOKEN to require `Authorization: Bearer <token>` on "
            f"protected endpoints, or bind 127.0.0.1 (the default) for loopback-only access"
        )

    if protocol == "a2a" and not _is_loopback(bind) and not os.environ.get("AGENTKIT_A2A_URL"):
        _fail("non-loopback A2A requires AGENTKIT_A2A_URL for client discovery")

    if binding is not None:
        # Optional, explicit per-Start capability. No default provider runtime
        # fallback; adapters without this hook fail closed in the gRPC skeleton.
        runner = getattr(factory, "run_agentsessions", None)
        run_agentsessions(
            binding, bind=bind, port=port, auth_token=auth_token,
            runner=runner if callable(runner) else None,
        )
        return

    app = _create_protocol_app(protocol, spec, factory, auth_token)

    # uvicorn's default access log can echo paths/headers; keep it but never log
    # bodies. The app itself never logs secrets.
    uvicorn.run(app, host=bind, port=port, log_level="info", access_log=True)
