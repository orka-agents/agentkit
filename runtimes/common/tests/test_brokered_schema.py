from __future__ import annotations

from agentkit_serve_common.brokered import brokered_tool_definitions
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.runtime import BrokeredToolDefinition


def test_brokered_tool_definitions_preserve_only_safe_runtime_fields():
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "brokered-test"},
            "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
            "instructions": "Be helpful.",
            "tools": [],
            "brokeredTools": [
                {
                    "name": "check-network-telemetry",
                    "description": "Read telemetry.",
                    "brokeredClass": "read",
                    "parameters": {"type": "object"},
                }
            ],
            "expose": {"openai": True, "port": 8080},
        }
    )

    assert brokered_tool_definitions(spec) == [
        BrokeredToolDefinition(
            name="check-network-telemetry",
            description="Read telemetry.",
            brokered_class="read",
            parameters={"type": "object"},
        )
    ]
