"""Wire-level parity suite for the LangGraph adapter.

The shared suite drives this adapter's real runtime against a scripted loopback
model and a stdio MCP fixture, holding it to the same model-visible and
client-visible behavior as every other adapter.
"""

from __future__ import annotations

import pytest

from agentkit_serve_common.parity import *  # noqa: F401,F403


@pytest.fixture
def openai_client_factory():
    # Model SDK imports stay in adapter-owned tests, outside the shared core.
    from openai import OpenAI

    return OpenAI
