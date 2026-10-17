"""Test-only wire selection for API-neutral hosted model-loop scenarios."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest

from test_foundry_brokered_protocol import (
    _FakeChatTransport as _ResponsesTransport,
    _chat_response as _responses_payload,
)


@pytest.fixture(params=[None, "responses"], ids=["default-chat", "responses"])
def model_api(request, monkeypatch):
    if request.param is None:
        # Prove the shipping default, not just an explicitly selected Chat path.
        monkeypatch.delenv("AGENTKIT_MODEL_API", raising=False)
        return "chat_completions"
    assert request.param == "responses"
    monkeypatch.setenv("AGENTKIT_MODEL_API", request.param)
    return request.param


def _selected_model_api() -> str:
    value = os.environ.get("AGENTKIT_MODEL_API", "chat_completions")
    assert value in {"chat_completions", "responses"}
    return value


def _assert_model_request(request: httpx.Request, *, model_api: str | None = None) -> None:
    api = model_api or _selected_model_api()
    body = json.loads(request.content)
    if api == "responses":
        assert request.url.path.endswith("/responses")
        assert "input" in body and "messages" not in body
        assert body["store"] is False
        assert body["include"] == ["reasoning.encrypted_content"]
        assert not body.get("previous_response_id") and not body.get("conversation")
        assert all(tool["type"] == "function" and "name" in tool for tool in body.get("tools", []))
    else:
        assert request.url.path.endswith("/chat/completions")
        assert "messages" in body and "input" not in body
        assert "store" not in body and "include" not in body
        assert all(tool["type"] == "function" and "function" in tool for tool in body.get("tools", []))
        assert all("phase" not in message and "reasoning" not in message for message in body["messages"])


def _model_response(payload: dict[str, Any], *, model_api: str | None = None) -> dict[str, Any]:
    """Convert the existing scripted assistant output to the selected wire API."""
    if (model_api or _selected_model_api()) == "responses":
        return payload
    message: dict[str, Any] = {"role": "assistant", "content": None}
    calls = []
    for item in payload["output"]:
        # These have no Chat equivalent. Keep their tests explicitly Responses-only.
        assert not item.get("phase") and item["type"] != "reasoning"
        if item["type"] == "function_call":
            calls.append(
                {
                    "id": item.get("call_id"),
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments")},
                }
            )
        else:
            assert item["type"] == "message" and item["role"] == "assistant"
            content = item["content"]
            if isinstance(content, list) and all(part.get("type") == "output_text" for part in content):
                message["content"] = "".join(part["text"] for part in content)
            elif isinstance(content, list) and len(content) == 1 and content[0].get("type") == "refusal":
                message["refusal"] = content[0]["refusal"]
            else:
                message["content"] = content
    if calls:
        message["tool_calls"] = calls
    usage = payload.get("usage", {})
    return {
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def _chat_response(message: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return _model_response(_responses_payload(message, **kwargs))


def _tool_response() -> dict[str, Any]:
    # Lazy import avoids a cycle with the streaming suite, which also uses this
    # helper for its brokered model transport but keeps its native runtime double.
    from test_foundry_streaming import _tool_response as responses_tool_response

    return _model_response(responses_tool_response())


def _response_function(payload: dict[str, Any]) -> dict[str, Any]:
    if "choices" in payload:
        return payload["choices"][0]["message"]["tool_calls"][0]["function"]
    return payload["output"][0]


def _model_input(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare conversation/tool pairing without replacing native recorded requests."""
    if "input" in body:
        return body["input"]
    items = []
    for message in body["messages"]:
        if message["role"] == "tool":
            items.append(
                {"type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"]}
            )
            continue
        if message.get("content") is not None:
            items.append({"type": "message", "role": message["role"], "content": message["content"]})
        for call in message.get("tool_calls", []):
            items.append({"type": "function_call", "call_id": call["id"], **call["function"]})
    return items


def _model_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [tool.get("function", tool) for tool in body["tools"]]


class _FakeChatTransport(_ResponsesTransport):
    def handler(self, request: httpx.Request) -> httpx.Response:
        _assert_model_request(request)
        response = super().handler(request)
        if response.status_code != 200 or _selected_model_api() == "responses":
            return response
        return httpx.Response(200, request=request, json=_model_response(response.json()))
