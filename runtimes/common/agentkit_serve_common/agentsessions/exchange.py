"""One bounded, correlated controller-mediated model effect per execution."""
from __future__ import annotations

import asyncio
from collections.abc import Sequence

from ._generated import common_pb2 as common
from ._generated import harness_pb2 as harness

MAX_MESSAGE_BYTES = 4 * 1024 * 1024
MAX_MODEL_RESULT_BYTES = 1024 * 1024


def text_content(message: common.Message, roles: set[str]) -> str:
    if message.role not in roles or any(p.WhichOneof("part") != "text" for p in message.parts):
        raise ValueError("only role-correct plain text is supported")
    return "".join(p.text.text for p in message.parts)


class ExecutionExchange:
    """Explicit per-Start transport; no callbacks hidden in request metadata.

    Input count retains zero inputs versus a single empty prompt. The neutral
    RunRequest still contains earlier Inputs in history and the last as prompt.
    The service owns the independent receiver and drains the one-slot event queue.
    """

    def __init__(self, execution_id: str, model: str, input_count: int) -> None:
        self.execution_id = execution_id
        self.model = model
        self.input_count = input_count
        self.events: asyncio.Queue[common.Event] = asyncio.Queue(maxsize=1)
        self._pending: asyncio.Future[common.Message] | None = None
        self._call_id = ""
        self._emitted = False
        self._counter = 0
        self._closed = False

    async def call(self, messages: Sequence[common.Message]) -> common.Message:
        if self._closed or self._pending is not None:
            raise ValueError("model exchange unavailable")
        for message in messages:
            text_content(message, {"system", "user", "assistant"})
        call_id = f"model-{self._counter + 1}"
        event = common.Event(
            execution_id=self.execution_id, schema_version=1, kind=common.EVENT_MODEL_CALL,
            model=common.ModelCall(model=self.model, id=call_id, messages=messages),
        )
        if event.ByteSize() > MAX_MESSAGE_BYTES:
            raise ValueError("model request exceeds message limit")
        self._counter += 1
        self._call_id = call_id
        pending = asyncio.get_running_loop().create_future()
        self._pending = pending
        try:
            self.events.put_nowait(event)
            return await pending
        finally:
            pending.cancel()
            self._pending = None
            self._call_id = ""
            self._emitted = False

    def mark_emitted(self, call_id: str) -> None:
        """Called immediately before the service yields the pending model call."""
        if self._pending is None or call_id != self._call_id:
            raise ValueError("unexpected model call")
        self._emitted = True

    def accept(self, result: harness.ModelResult) -> None:
        pending = self._pending
        if pending is None or pending.done() or not self._emitted or result.model_call_id != self._call_id:
            raise ValueError("unexpected model result")
        if not result.HasField("message") or not result.message.parts:
            raise ValueError("model result message required")
        text_content(result.message, {"assistant"})
        if result.ByteSize() > MAX_MODEL_RESULT_BYTES:
            raise ValueError("model result exceeds message limit")
        if result.HasField("usage"):
            usage = result.usage
            if (usage.model and usage.model != self.model) or min(usage.input_tokens, usage.output_tokens, usage.reasoning_tokens) < 0:
                raise ValueError("invalid model result usage")
        # Usage stays host-owned; neither invent counts nor emit duplicate OUTPUT.
        message = common.Message()
        message.CopyFrom(result.message)
        pending.set_result(message)

    def close(self) -> None:
        self._closed = True
        if self._pending is not None:
            self._pending.cancel()
