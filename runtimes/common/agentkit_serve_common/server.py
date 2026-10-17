"""FastAPI app: a non-streaming OpenAI Chat and Responses facade over an agent.

Endpoints (see ``docs/agent-abi.md`` §4):

* ``POST /v1/chat/completions`` — runs the agent once and returns a single
  ``chat.completion`` object. Rejects ``stream: true`` and request-supplied
  ``tools`` / ``tool_choice`` (the agent owns its tools).
* ``POST /v1/responses``       — stateless text input and a completed response.
  Shares the same runtime and authentication as Chat Completions.
* ``GET  /v1/models``           — SDK-compatibility listing of the one model.
* ``GET  /healthz``             — liveness (always open, even under auth).

The runtime Adapter session is entered ONCE in the lifespan and reused across
requests, then torn down on shutdown.

THIS MODULE IS FRAMEWORK-AGNOSTIC and lives in the shared core. It imports nothing
from a runtime framework (``agent_framework``, ``pydantic_ai``) or any model SDK:
the run is driven through a :class:`~agentkit_serve_common.runtime.RuntimeFactory`
passed into :func:`create_app`, which returns a neutral
:class:`~agentkit_serve_common.runtime.RunResult` and raises a neutral
:class:`~agentkit_serve_common.runtime.AgentRunError`. The lock-in import boundary
(plan §12) therefore lives only in each adapter's ``agent_factory.py``.

Secret hygiene: this module never logs the request body, the API key, or any tool
env value.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .config import AgentSpec
from .conversation import ConversationError, RunRequest, run_request_from_messages
from .responses import responses_payload, run_request_from_responses
from .runtime import AgentRunError, RunResult, RuntimeFactory, RuntimeHealth


# --------------------------------------------------------------------------- #
# Request models (lenient: ignore unknown OpenAI fields like name/logit_bias).
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    # content is a string, a list of content parts, or null (tool-call turns).
    content: Any = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessage] = []
    stream: bool | None = False
    tools: list[Any] | None = None
    tool_choice: Any = None


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    stream: bool | None = False
    tools: list[Any] | None = None
    tool_choice: Any = None
    previous_response_id: Any = None
    conversation: Any = None
    background: bool | None = False
    store: bool | None = False


# tool_choice values that mean "no specific tool requested" — treated as absent so
# generic OpenAI SDK clients that always attach one are not rejected outright.
_EMPTY_TOOL_CHOICE = (None, "", "none", "auto")


def _completion_response(model_name: str, result: RunResult) -> dict:
    """Assemble a single OpenAI ``chat.completion`` object from a neutral result."""
    usage = result.usage or {}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        },
    }


def _error_response(status: int, message: str, err_type: str, code: str | None = None) -> JSONResponse:
    """OpenAI-shaped error envelope."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": code}},
    )


def make_auth_dependency(auth_token: str | None):
    """Build a FastAPI dependency enforcing ``Authorization: Bearer <token>``.

    When ``auth_token`` is falsy, auth is disabled and the dependency is a no-op.
    Applied to ``/v1/*`` only; ``/healthz`` stays open.
    """

    async def _require_auth(authorization: str | None = Header(default=None)) -> None:
        if not auth_token:
            return
        expected = f"Bearer {auth_token}"
        # Constant-ish comparison; tokens are short-lived deploy secrets.
        if authorization != expected:
            raise HTTPException(
                status_code=401,
                detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return _require_auth


def _clean_session_id(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    return value or None


def create_app(spec: AgentSpec, factory: RuntimeFactory, auth_token: str | None = None) -> FastAPI:
    """Construct the FastAPI app for one validated :class:`AgentSpec`.

    ``factory`` is the adapter's framework-specific runtime factory (its
    ``agent_factory`` module): it builds a RuntimeSession that owns framework
    lifecycle and returns neutral results. This injection is what keeps THIS
    module framework-agnostic.
    """
    runtime = factory.build_runtime(spec)
    health = RuntimeHealth()
    model_name = spec.model.name

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Enter the runtime context ONCE: starts stdio MCP subprocesses and keeps
        # them warm for the life of the server. Torn down on shutdown.
        async with runtime:
            app.state.runtime = runtime
            yield

    app = FastAPI(title="agentkit-serve", lifespan=lifespan)
    auth = Depends(make_auth_dependency(auth_token))

    @app.get("/healthz")
    async def healthz():
        if not health.healthy:
            return JSONResponse(status_code=503, content={"status": "unhealthy"})
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[auth])
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "agentkit",
                }
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[auth])
    async def chat_completions(
        req: ChatCompletionRequest,
        request: Request,
        x_agentkit_session_id: str | None = Header(default=None, alias="X-AgentKit-Session-Id"),
    ):
        # --- reject unsupported request features (ABI §4) -------------------
        if req.stream:
            return _error_response(
                400,
                "streaming is not supported; this agent serves non-streaming "
                "chat completions only",
                "invalid_request_error",
                "stream_unsupported",
            )
        if req.tools:
            return _error_response(
                400,
                "request-supplied tools are not allowed; this agent owns its tools",
                "invalid_request_error",
                "tools_unsupported",
            )
        if req.tool_choice not in _EMPTY_TOOL_CHOICE:
            return _error_response(
                400,
                "request-supplied tool_choice is not allowed; this agent owns its tools",
                "invalid_request_error",
                "tool_choice_unsupported",
            )

        # --- map conversation & run the agent ------------------------------
        try:
            normalized = run_request_from_messages(req.messages)
            run_request = RunRequest(
                prompt=normalized.prompt,
                history=normalized.history,
                session_id=_clean_session_id(x_agentkit_session_id),
            )
        except ConversationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = await run_agent(run_request, request)
        if isinstance(result, JSONResponse):
            return result
        return _completion_response(model_name, result)

    async def run_agent(run_request: RunRequest, request: Request) -> RunResult | JSONResponse:
        try:
            return await request.app.state.runtime.run(run_request)
        except HTTPException:
            raise
        except AgentRunError as exc:  # neutral error: framework/model failure
            health.record(exc)
            return _error_response(
                exc.status,
                str(exc),
                "agent_error",
                # Preserve the adapter's original framework exception class name
                # when supplied; otherwise fall back to the neutral class name.
                exc.code or exc.__class__.__name__,
            )

    @app.post("/v1/responses", dependencies=[auth])
    async def responses(
        req: ResponsesRequest,
        request: Request,
        x_agentkit_session_id: str | None = Header(default=None, alias="X-AgentKit-Session-Id"),
    ):
        if req.stream:
            return _error_response(
                400, "streaming is not supported; this agent serves non-streaming responses only",
                "invalid_request_error", "stream_unsupported",
            )
        if req.tools:
            return _error_response(
                400, "request-supplied tools are not allowed; this agent owns its tools",
                "invalid_request_error", "tools_unsupported",
            )
        if req.tool_choice not in _EMPTY_TOOL_CHOICE:
            return _error_response(
                400, "request-supplied tool_choice is not allowed; this agent owns its tools",
                "invalid_request_error", "tool_choice_unsupported",
            )
        if req.previous_response_id is not None or req.conversation is not None:
            return _error_response(
                400, "response state is not supported; resend conversation history in input",
                "invalid_request_error", "response_state_unsupported",
            )
        if req.background:
            return _error_response(
                400, "background responses are not supported",
                "invalid_request_error", "background_unsupported",
            )
        if req.store:
            return _error_response(
                400, "response storage is not supported; use store=false",
                "invalid_request_error", "store_unsupported",
            )
        try:
            run_request = run_request_from_responses(
                req.input, instructions=req.instructions,
                session_id=_clean_session_id(x_agentkit_session_id),
            )
        except ConversationError as exc:
            return _error_response(400, str(exc), "invalid_request_error", "invalid_request")

        result = await run_agent(run_request, request)
        if isinstance(result, JSONResponse):
            return result
        payload = responses_payload(model_name, result)
        payload.update(
            store=False, error=None, incomplete_details=None,
            parallel_tool_calls=False, tool_choice="auto", tools=[],
        )
        # The runtime seam reports aggregate usage, not cache/reasoning breakdowns.
        payload["usage"].update(
            input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
            output_tokens_details={"reasoning_tokens": 0},
        )
        return payload

    @app.exception_handler(RequestValidationError)
    async def _request_validation_exc_handler(request: Request, exc: RequestValidationError):
        # Pydantic validation details can embed request inputs. Inspect only the
        # machine-readable error type and return fixed, secret-safe messages.
        if any(error.get("type") == "json_invalid" for error in exc.errors()):
            return _error_response(
                400,
                "request body must be valid JSON",
                "invalid_request_error",
                "invalid_json",
            )
        return _error_response(
            400,
            "request body is invalid",
            "invalid_request_error",
            "invalid_request",
        )

    # Map HTTPExceptions raised in helpers to the OpenAI error envelope too.
    @app.exception_handler(HTTPException)
    async def _http_exc_handler(request: Request, exc: HTTPException):
        return _error_response(
            exc.status_code,
            str(exc.detail),
            "invalid_request_error" if exc.status_code < 500 else "internal_error",
        )

    return app
