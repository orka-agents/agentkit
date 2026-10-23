from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentkit_serve import agent_factory


@pytest.mark.parametrize("extra", [{}, {"history": None}, {"history": object()}])
def test_responses_stream_forwards_sdk_keywords(monkeypatch, extra):
    process = AsyncMock()
    monkeypatch.setattr(agent_factory.OpenAIResponsesModel, "_process_streamed_response", process)
    model = agent_factory._CompletedResponsesModel(
        "test", provider=agent_factory.OpenAIProvider(api_key="test"),
    )
    source = MagicMock()
    settings, parameters = {}, agent_factory.ModelRequestParameters()

    result = asyncio.run(model._process_streamed_response(
        source, settings, parameters,
        expected_model_name="test", expected_response_id="resp_test", **extra,
    ))

    assert result is process.return_value
    validated, forwarded_settings, forwarded_parameters = process.call_args.args
    assert isinstance(validated, agent_factory._ValidatedResponsesStream)
    assert validated._source is source
    assert forwarded_settings is settings
    assert forwarded_parameters is parameters
    assert process.call_args.kwargs == {
        "expected_model_name": "test", "expected_response_id": "resp_test", **extra,
    }


def test_mcp_compatibility_symbols_available():
    from mcp import McpError

    assert McpError.__name__ == "McpError"
