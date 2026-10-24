"""Helpers for safe Orka-brokered tool schemas.

The hosted Foundry `/responses` endpoint cannot receive request-level tool
schemas, so brokered mode uses static schema-only declarations baked into
`agent.yaml`. Only name, description, brokered class, JSON parameters schema,
and an optional schema digest may cross into hosted AgentKit. Export those
declarations from Orka Tool CRDs with `agentkit render --target brokered-tools`.
"""

from __future__ import annotations

from .config import AgentSpec
from .runtime import BrokeredToolDefinition


def brokered_tool_definitions(spec: AgentSpec) -> list[BrokeredToolDefinition]:
    """Return runtime-safe brokered tool definitions from an AgentSpec."""

    return [
        BrokeredToolDefinition(
            name=tool.name,
            description=tool.description,
            brokered_class=tool.brokered_class,
            parameters=dict(tool.parameters),
            schema_digest=tool.schema_digest,
        )
        for tool in spec.brokered_tools
    ]
