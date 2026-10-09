"""Wire-level parity suite for the Microsoft Agent Framework adapter.

The shared suite drives this adapter's real runtime against a scripted loopback
model and a stdio MCP fixture, holding it to the same model-visible and
client-visible behavior as every other adapter.
"""

from __future__ import annotations

from agentkit_serve_common.parity import *  # noqa: F401,F403
