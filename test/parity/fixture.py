"""Scripted model and remote MCP tool for the container parity smoke.

Runs with an AgentKit adapter image's interpreter, which already ships FastMCP,
and serves both dependencies of the parity agent images on one port:

* ``POST /v1/chat/completions`` answers from the last user message:
  ``plain:<marker>`` replies ``parity-answer: <marker>``; ``tool:<marker>`` calls
  the ``*_echo`` tool, then replies with its result; ``auth-echo`` returns a 401
  that echoes the presented credential, as some gateways do.
* ``/mcp`` is a stateless Streamable HTTP MCP server with an ``echo`` tool.
"""

from __future__ import annotations

import json
import uuid

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

server = FastMCP("parity", host="0.0.0.0", port=8090, log_level="WARNING", stateless_http=True, json_response=True)


@server.tool()
def echo(value: str) -> str:
    """Return a receipt for value."""
    return "receipt-" + value


def _text(content: object) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def _completion(model: str, message: dict) -> JSONResponse:
    finish = "tool_calls" if message.get("tool_calls") else "stop"
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )


@server.custom_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request) -> JSONResponse:
    body = await request.json()
    if body.get("stream"):
        return JSONResponse({"error": {"message": "streaming is not scripted"}}, status_code=400)
    messages = body["messages"]
    prompt = next(_text(m.get("content")) for m in reversed(messages) if m["role"] == "user")
    if prompt == "auth-echo":
        presented = request.headers.get("authorization", "")
        return JSONResponse({"error": {"message": f"invalid key {presented}"}}, status_code=401)
    if prompt.startswith("tool:"):
        results = [_text(m.get("content")) for m in messages if m["role"] == "tool"]
        if results:
            return _completion(body["model"], {"role": "assistant", "content": "tool-result: " + " ".join(results)})
        name = next(t["function"]["name"] for t in body.get("tools", []) if t["function"]["name"].endswith("_echo"))
        call = {
            "id": "call_parity_0",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps({"value": prompt.removeprefix("tool:")})},
        }
        return _completion(body["model"], {"role": "assistant", "content": None, "tool_calls": [call]})
    return _completion(body["model"], {"role": "assistant", "content": "parity-answer: " + prompt.removeprefix("plain:")})


if __name__ == "__main__":
    server.run(transport="streamable-http")
