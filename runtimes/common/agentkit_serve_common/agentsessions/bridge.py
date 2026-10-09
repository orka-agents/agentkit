"""Restricted Chat Completions on authenticated per-execution loopback only.

No provider networking lives here. A complete controller result is required
before a JSON response or one synthetic terminal SSE chunk can be returned.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import socket
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response

from ..adapter_support import _wait_for_owner_task
from ._generated import common_pb2 as common
from .exchange import ExecutionExchange, text_content

MAX_HTTP_BYTES = 1024 * 1024


@dataclass(frozen=True)
class LoopbackBinding:
    base_url: str
    token: str = field(repr=False)


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _invalid_constant(value):
    raise ValueError("invalid JSON constant")


def _messages(body: object, model: str) -> tuple[list[common.Message], bool]:
    if not isinstance(body, dict) or set(body) - {"model", "messages", "stream"}:
        raise ValueError("unsupported request options")
    if body.get("model") != model or type(body.get("stream", False)) is not bool:
        raise ValueError("unsupported model or stream option")
    raw = body.get("messages")
    if not isinstance(raw, list):
        raise ValueError("messages required")
    messages = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise ValueError("unsupported message options")
        role, content = item["role"], item["content"]
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise ValueError("unsupported message role")
        if isinstance(content, list):
            if any(
                not isinstance(p, dict) or set(p) != {"type", "text"}
                or p["type"] != "text" or not isinstance(p["text"], str)
                for p in content
            ):
                raise ValueError("unsupported message content")
            content = "".join(p["text"] for p in content)
        if not isinstance(content, str):
            raise ValueError("plain text required")
        messages.append(common.Message(role=role, parts=[common.Part(text=common.TextPart(text=content))]))
    return messages, body.get("stream", False)


def _error(status: int) -> JSONResponse:
    return JSONResponse({"error": {"message": "agentsessions model bridge request failed"}}, status_code=status)


def _app(exchange: ExecutionExchange, token: str) -> FastAPI:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    busy = False

    @app.post("/v1/chat/completions")
    async def completion(request: Request):
        nonlocal busy
        auth = request.headers.getlist("authorization")
        if len(auth) != 1 or not secrets.compare_digest(auth[0].encode(), ("Bearer " + token).encode()):
            return _error(401)
        if busy:
            return _error(409)
        # Reserve before reading the body: concurrent clients cannot build an
        # unbounded collection of bodies or pending effects in this execution.
        busy = True
        try:
            raw = bytearray()
            async for chunk in request.stream():
                if len(raw) + len(chunk) > MAX_HTTP_BYTES:
                    return _error(413)
                raw.extend(chunk)
            try:
                body = json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid_constant)
                messages, stream = _messages(body, exchange.model)
            except (ValueError, TypeError, UnicodeError, RecursionError):
                return _error(400)
            try:
                result = await exchange.call(messages)
                content = text_content(result, {"assistant"})
                choice = {"index": 0, "finish_reason": "stop"}
                choice["delta" if stream else "message"] = {"role": "assistant", "content": content}
                response = {
                    "id": "agentkit-model",
                    "object": "chat.completion.chunk" if stream else "chat.completion",
                    "created": 0, "model": exchange.model, "choices": [choice],
                }
                encoded = json.dumps(response, ensure_ascii=False).encode("utf-8")
                if stream:
                    encoded = b"data: " + encoded + b"\n\ndata: [DONE]\n\n"
                if len(encoded) > MAX_HTTP_BYTES:
                    return _error(502)
                return Response(encoded, media_type="text/event-stream" if stream else "application/json")
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
                # Closing the exchange cancels its pending future, not this ASGI
                # task; return a fixed failure rather than leaking exceptions.
                return _error(502)
            except Exception:
                return _error(502)
        finally:
            busy = False

    return app


class _ExecutionServer(uvicorn.Server):
    async def serve(self, sockets: list[socket.socket] | None = None) -> None:
        try:
            await super().serve(sockets=sockets)
        finally:
            # Uvicorn 0.29 can skip shutdown if should_exit arrives in startup;
            # exceptions also bypass it. Close asyncio's listeners before the
            # raw socket context exits, or FD reuse leaves a stale selector.
            # loopback_bridge shields this owner task, including these awaits.
            listeners = getattr(self, "servers", ())
            for listener in listeners:
                listener.close()
            for listener in listeners:
                await listener.wait_closed()

    @contextmanager
    def capture_signals(self):
        # This nested listener does not own process-global signal handlers.
        yield


@asynccontextmanager
async def loopback_bridge(exchange: ExecutionExchange):
    """Own a fresh listener/token, closing all requests before runner teardown."""
    token = secrets.token_urlsafe(32)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = _ExecutionServer(uvicorn.Config(
            _app(exchange, token), host="127.0.0.1", port=port,
            lifespan="off", access_log=False, log_config=None, log_level=None,
            limit_concurrency=8, timeout_graceful_shutdown=1, ws="none",
        ))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("agentsessions bridge startup failed")
                    await asyncio.sleep(0.001)
            yield LoopbackBinding(f"http://127.0.0.1:{port}/v1/", token)
        finally:
            exchange.close()
            server.should_exit = True
            await _wait_for_owner_task(task)
