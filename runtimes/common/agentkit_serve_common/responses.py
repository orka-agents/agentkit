"""Shared Responses text/usage encoding and stateless OpenAI input normalization.

Foundry retains its own hosted session, ID, and brokered continuation contract.
These helpers have no dependency on that hosting protocol or a model SDK.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Mapping

from .conversation import ConversationError, ConversationTurn, RunRequest
from .runtime import RunResult


def responses_usage(result: RunResult | None = None, usage: Mapping[str, int] | None = None) -> dict[str, int]:
    raw = dict(usage or (result.usage if result is not None else {}) or {})
    input_tokens = int(raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0)
    output_tokens = int(raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0)
    total_tokens = int(raw.get("total_tokens", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def run_request_from_responses(
    value: str | list[dict[str, Any]],
    *,
    instructions: str | None = None,
    session_id: str | None = None,
) -> RunRequest:
    """Normalize the text-only, stateless input supported by the OpenAI facade.

    Reject unsupported items before any runtime/tool execution. In particular,
    caller-supplied function results are not instructions or agent-owned history.
    """
    history = [ConversationTurn(role="system", text=instructions)] if instructions else []
    if isinstance(value, str):
        return RunRequest(prompt=value, history=tuple(history), session_id=session_id)
    if not value:
        raise ConversationError("Responses input must be a string or a non-empty message array")

    for item in value:
        if item.get("type", "message") != "message":
            raise ConversationError("Responses input supports message items only")
        role = item.get("role")
        if role not in ("system", "developer", "user", "assistant"):
            raise ConversationError("Responses message role must be system, developer, user, or assistant")
        content = item.get("content")
        if not isinstance(content, str):
            if not isinstance(content, list) or any(
                not isinstance(part, dict)
                or part.get("type") not in ("input_text", "output_text")
                or not isinstance(part.get("text"), str)
                for part in content
            ):
                raise ConversationError("Responses message content must be text or text parts")
        phase = item.get("phase") if role == "assistant" else None
        if phase not in (None, "commentary", "final_answer"):
            raise ConversationError("Responses assistant phase must be commentary or final_answer")
        history.append(ConversationTurn(
            role="system" if role == "developer" else role,
            text=content if isinstance(content, str) else "".join(part["text"] for part in content),
            phase=phase,
        ))

    last = history.pop()
    if last.role != "user":
        raise ConversationError("Responses input list final message must have role 'user'")
    return RunRequest(prompt=last.text, history=tuple(history), session_id=session_id)


def responses_payload(
    model_name: str,
    result: RunResult,
    *,
    previous_response_id: str | None = None,
    response_id: str | None = None,
    created_at: int | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    response_id = response_id or f"resp_{uuid.uuid4().hex}"
    message_id = message_id or f"msg_{uuid.uuid4().hex}"
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()) if created_at is None else created_at,
        "status": "completed",
        "model": model_name,
        "output": [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": result.text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": responses_usage(result),
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    return payload
