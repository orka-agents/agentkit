"""Conversation normalization for the OpenAI Chat-Completions facade.

This Module owns the framework-neutral request shape that runtime Adapters consume.
It translates lenient OpenAI messages into an explicit RunRequest so server.py and
adapter agent_factory.py files do not pass around loosely-typed ``(history,
prompt)`` pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Literal, Mapping, Protocol, Sequence

FORWARDED_ROLES = frozenset({"system", "user", "assistant"})


class ConversationError(ValueError):
    """Raised when an OpenAI message list cannot form a valid agent run."""


@dataclass(frozen=True)
class ConversationTurn:
    """One prior conversation turn forwarded to a runtime Adapter."""

    role: str
    text: str
    phase: Literal["commentary", "final_answer"] | None = None


@dataclass(frozen=True)
class ToolCallEvent:
    """One tool lifecycle observation, without arguments, results, or errors.

    Adapters await the observer before continuing the run. A call starts with
    ``in_progress`` and ends with ``completed`` or ``failed`` under the same ID.
    The ID is adapter-local and must not be used as a host authorization token.
    """

    tool_call_id: str
    tool_name: str
    status: Literal["in_progress", "completed", "failed"]


@dataclass(frozen=True)
class RunRequest:
    """Framework-neutral request for one agent run.

    ``session_id`` is optional and provider-neutral. HTTP adapters may set it
    from their transport/session headers so runtimes with memory or durable
    context providers can correlate turns without baking a provider-specific
    concept into the ABI.

    The remaining optional fields carry per-turn protocol metadata used by
    orchestration hosts such as Orka. Runtime adapters may ignore fields they do
    not support, but helpers should prefer ``env`` over process environment when
    resolving per-run credentials.
    """

    prompt: str
    history: tuple[ConversationTurn, ...] = ()
    session_id: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    deadline: datetime | None = None
    turn_id: str | None = None
    correlation_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    on_tool_event: Callable[[ToolCallEvent], Awaitable[None]] | None = field(
        default=None, repr=False, compare=False
    )
    # Journal-restored opaque agentsessions Start.Config; never env or prompt input.
    config: bytes = field(default=b"", repr=False)


class OpenAIMessage(Protocol):
    role: str
    content: Any


def text_of(content: Any) -> str:
    """Flatten OpenAI message content into plain text.

    Accepts a string, a list of content parts (``{"type":"text","text":...}``),
    or ``None``. Non-text parts (images, etc.) are ignored in v0.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                out.append(str(part.get("text", "")))
        return "".join(out)
    return str(content)


def run_request_from_messages(messages: Sequence[OpenAIMessage]) -> RunRequest:
    """Map OpenAI messages to an explicit framework-neutral RunRequest.

    The conversation must end with a ``user`` message; its text is the prompt for
    this turn. Earlier ``system``/``user``/``assistant`` messages with non-empty
    text become history. ``tool`` and unknown roles are dropped by design: the
    agent owns its tools, so client-supplied tool results are not meaningful in v0.
    """
    if not messages:
        raise ConversationError("messages must be a non-empty array")

    last = messages[-1]
    if last.role != "user":
        raise ConversationError("the final message must have role 'user'")

    history: list[ConversationTurn] = []
    for msg in messages[:-1]:
        turn_text = text_of(msg.content)
        if turn_text and msg.role in FORWARDED_ROLES:
            history.append(ConversationTurn(role=msg.role, text=turn_text))

    return RunRequest(prompt=text_of(last.content), history=tuple(history))
