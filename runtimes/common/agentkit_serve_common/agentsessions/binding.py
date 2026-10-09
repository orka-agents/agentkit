"""Immutable deployment binding; never resolves model/provider credentials."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from ..config import AgentSpec, ConfigError, load_with_bytes

AGENT_CONFIGURATION_DIGEST_ENV = "AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST"
IMPLEMENTATION_DIGEST_ENV = "AGENTKIT_AGENTSESSIONS_IMPLEMENTATION_DIGEST"


class AgentsessionsConfigurationError(ValueError):
    """Safe startup failure, without configuration or credential values."""


@dataclass(frozen=True)
class VerifiedAgentsessionsBinding:
    spec: AgentSpec
    configuration_digest: str
    implementation_digest: str

    @property
    def descriptor_id(self) -> str:
        return f"agentkit:{self.configuration_digest}:{self.implementation_digest}"


def _digest_from_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise AgentsessionsConfigurationError(f"{name} must be a lowercase sha256 digest")
    return value


def load_verified_agentsessions_binding(path: str | Path) -> VerifiedAgentsessionsBinding:
    """Read once, verify exact ABI bytes and restrict the keyless text-only profile.

    Implementation digest is deployment-owned: use the immutable adapter image
    digest (including its dependency closure), not a moving tag or agent name.
    """
    try:
        spec, raw = load_with_bytes(path)
    except ConfigError:
        # Suppress parser chaining even for callers outside the CLI: tracebacks
        # must not expose YAML snippets or values from the invalid input.
        raise AgentsessionsConfigurationError("cannot load agentsessions agent configuration") from None
    configuration_digest = _digest_from_env(AGENT_CONFIGURATION_DIGEST_ENV)
    actual = "sha256:" + hashlib.sha256(raw).hexdigest()
    if not secrets.compare_digest(configuration_digest, actual):
        raise AgentsessionsConfigurationError(
            f"{AGENT_CONFIGURATION_DIGEST_ENV} does not match the exact agent config bytes"
        )
    implementation_digest = _digest_from_env(IMPLEMENTATION_DIGEST_ENV)
    if spec.tools:
        raise AgentsessionsConfigurationError("agentsessions rejects baked direct tools")
    if spec.brokered_tools:
        raise AgentsessionsConfigurationError("agentsessions rejects baked brokeredTools")
    if spec.context.providers:
        raise AgentsessionsConfigurationError("agentsessions rejects baked context providers")
    if spec.model.api_key_env and os.environ.get(spec.model.api_key_env):
        raise AgentsessionsConfigurationError("agentsessions rejects supplied baked model credentials")
    try:
        url = urlsplit(spec.model.base_url)
        if url.username is not None or url.password is not None or url.query or url.fragment:
            raise ValueError("credential-bearing URL")
    except ValueError:
        raise AgentsessionsConfigurationError("agentsessions rejects credential-bearing model URLs") from None
    return VerifiedAgentsessionsBinding(spec, configuration_digest, implementation_digest)
