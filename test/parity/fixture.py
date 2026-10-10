"""Scripted Chat/Responses model and remote MCP tool for the container parity smoke.

Runs with an AgentKit adapter image's interpreter and serves both dependencies
on one port. The model routes reuse the wire provider's native JSON and SDK SSE
encoders; no model, provider-managed conversation, or external service is used.

``plain:<marker>`` answers directly; ``tool:<marker>`` requests the ``*_echo``
tool, then answers with its receipt. ``history:<marker>`` reports the text turns
received and ``api:<selection>`` reports the upstream API actually used.
``auth-echo`` returns a 401 that echoes the presented credential.
The task-only ``PARITY_DISABLE_RESPONSES=1`` flag rejects Responses with a known
route 404 before reading the body or executing model/tool work.
``GET /parity/requests`` exposes only request counts for smoke assertions.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agentkit_serve_common.parity import (
    _answer,
    _conversation,
    _model_wire_response,
    _Reply,
    _request_messages,
    _tool_calls,
    _tool_definitions,
    _tool_results,
)
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


_request_counts = {"chat_completions": 0, "responses": 0, "responses_rejected": 0}


def echo(value: str) -> str:
    """Return a receipt for value."""
    return "receipt-" + value


def _script_reply(body: dict[str, Any], model_api: str, presented: str) -> _Reply:
    if model_api == "responses" and (
        "messages" in body
        or body.get("store") is not False
        or body.get("previous_response_id")
        or body.get("conversation")
    ):
        return _Reply(status=400, body={"error": {"message": "Responses must carry stateless input with store=false"}})
    messages = _request_messages(body)
    last_user = next(index for index in reversed(range(len(messages))) if messages[index]["role"] == "user")
    prompt = _conversation({"messages": [messages[last_user]]})[0][1]
    if prompt == "auth-echo":
        return _Reply(status=401, body={"error": {"message": f"invalid key {presented}"}})
    if prompt.startswith("api:"):
        return _answer("parity-api: " + model_api)
    if prompt.startswith("history:"):
        return _answer("parity-history: " + json.dumps(_conversation(body)))
    if prompt.startswith("tool:"):
        # Old Orka tool receipts must not satisfy a new turn's tool request.
        results = _tool_results({"messages": messages[last_user + 1 :]})
        if results:
            return _answer("tool-result: " + " ".join(results.values()))
        name = next(tool["name"] for tool in _tool_definitions(body) if tool["name"].endswith("_echo"))
        return _tool_calls((name, json.dumps({"value": prompt.removeprefix("tool:")})))
    return _answer("parity-answer: " + prompt.removeprefix("plain:"))


async def _model_response(request: Request, model_api: str) -> Response:
    _request_counts[model_api] += 1
    if model_api == "responses" and os.environ.get("PARITY_DISABLE_RESPONSES") == "1":
        _request_counts["responses_rejected"] += 1
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    body = await request.json()
    reply = _script_reply(body, model_api, request.headers.get("authorization", ""))
    content_type, payload = _model_wire_response(body, reply, model_api)
    return Response(payload, status_code=reply.status, media_type=content_type)


async def chat_completions(request: Request) -> Response:
    return await _model_response(request, "chat_completions")


async def responses(request: Request) -> Response:
    return await _model_response(request, "responses")


async def request_counts(request: Request) -> Response:
    return JSONResponse(_request_counts.copy())


def create_server() -> Any:
    # The common fixture tests need no MCP dependency. The adapter image already
    # ships FastMCP, and registers these same handlers when run as the fixture.
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("parity", host="0.0.0.0", port=8090, log_level="WARNING", stateless_http=True, json_response=True)
    server.tool()(echo)
    server.custom_route("/v1/chat/completions", methods=["POST"])(chat_completions)
    server.custom_route("/v1/responses", methods=["POST"])(responses)
    server.custom_route("/parity/requests", methods=["GET"])(request_counts)
    return server


if __name__ == "__main__":
    create_server().run(transport="streamable-http")
