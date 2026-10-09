"""Restricted journal text projection; invocation Config is never prompt data."""
from __future__ import annotations

from ..conversation import ConversationTurn, RunRequest
from ._generated import common_pb2 as common
from ._generated import harness_pb2 as harness
from .exchange import text_content

_METADATA_BODIES = {
    common.EVENT_MODEL_CALL: "model",
    common.EVENT_USAGE: "usage",
    common.EVENT_LIFECYCLE: "lifecycle",
    common.EVENT_END: "end",
    common.EVENT_ERROR: "error",
    common.EVENT_EXECUTION_START: "execution_start",
}


class UnsupportedContent(ValueError):
    pass


def run_request(first: harness.ControllerFrame) -> RunRequest:
    """History once; earlier Inputs once; final Input as prompt, even if empty.

    The execution exchange separately retains the raw input count, because
    RunRequest's empty prompt cannot distinguish inputless from one empty Input.
    """
    start = first.start
    history: list[ConversationTurn] = []
    try:
        for event in start.history:
            if event.kind in (common.EVENT_INPUT, common.EVENT_OUTPUT):
                role = "user" if event.kind == common.EVENT_INPUT else "assistant"
                if event.WhichOneof("body") != "message":
                    raise UnsupportedContent("history message payload is required")
                history.append(ConversationTurn(role=role, text=text_content(event.message, {role})))
            elif event.kind in _METADATA_BODIES:
                if event.WhichOneof("body") != _METADATA_BODIES[event.kind]:
                    raise UnsupportedContent("unsupported history payload")
                if event.kind == common.EVENT_MODEL_CALL:
                    if event.model.params:
                        raise UnsupportedContent("unsupported model history options")
                    for message in event.model.messages:
                        text_content(message, {"system", "user", "assistant"})
            else:
                raise UnsupportedContent("tool or unknown history is unsupported")
        inputs = [text_content(message, {"user"}) for message in start.inputs]
    except ValueError as exc:
        raise UnsupportedContent("unsupported Start content") from exc
    history.extend(ConversationTurn(role="user", text=value) for value in inputs[:-1])
    return RunRequest(
        prompt=inputs[-1] if inputs else "", history=tuple(history),
        session_id=first.session or None, turn_id=first.execution_id,
        config=bytes(start.config),
    )
