"""Foundry Hosted Agent protocol adapters over the AgentKit RuntimeSession seam.

The adapter intentionally stays provider-light: it exposes the container HTTP
contract expected by Foundry Hosted Agents (``/readiness``, ``/invocations`` and
``/responses``) while reusing the same ``RuntimeFactory`` / ``RunRequest`` seam
as the native OpenAI facade.

When ``agent.yaml`` contains static ``brokeredTools`` declarations, the hosted
``/responses`` route enters a deterministic brokered function-call loop. The
container emits Responses ``function_call`` output items but never executes the
Orka-governed tool locally. A later continuation must provide a matching
``function_call_output`` for the same ``previous_response_id`` and pending
``call_id``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .adapter_support import MODEL_APIS, ModelAPI
from .brokered import brokered_tool_definitions
from .config import AgentSpec, _unsafe_brokered_key, _unsafe_brokered_text
from .foundry_model_loop import (
    BrokeredChatModelLoop,
    ModelLoopFinal,
    ModelLoopToolRequest,
)
from .model_errors import normalized_model_error_details
from .foundry_streaming import BrokeredResponseStream, brokered_stream_response
from .conversation import FORWARDED_ROLES, ConversationTurn, RunRequest
from .runtime import AgentRunError, BrokeredToolDefinition, RunResult, RuntimeFactory, RuntimeHealth
from .responses import (
    responses_payload,
    responses_usage as _responses_usage,
)
from .server import make_auth_dependency
from .tool_errors import orka_tool_error_details

logger = logging.getLogger(__name__)

try:  # Prefer the official hosted Responses SDK ID format/state-compatible prefix.
    from azure.ai.agentserver.responses._id_generator import IdGenerator as _AzureResponsesIdGenerator
except Exception:  # pragma: no cover - the SDK is optional outside conformance installs.
    _AzureResponsesIdGenerator = None

_DEFAULT_STATE_TTL_SECONDS = 15 * 60
_DEFAULT_MAX_PENDING_RESPONSES = 128
_DEFAULT_MAX_ARGUMENT_BYTES = 8192
_DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024
_DEFAULT_MAX_RESPONSE_STATE_BYTES = 4 * 1024 * 1024
_DEFAULT_MAX_REQUEST_BODY_BYTES = 1024 * 1024
_DEFAULT_MAX_MODEL_MESSAGES_BYTES = 1024 * 1024
_REQUEST_BODY_OVERHEAD_BYTES = 16 * 1024
_MAX_SYNTHETIC_ARRAY_ITEMS = 32
_MAX_SYNTHETIC_STRING_LENGTH = 4096
_MAX_SYNTHETIC_VALUES = 256
_MAX_OUTPUT_DEPTH = 128
_STATE_TTL_ENV = "AGENTKIT_FOUNDRY_RESPONSE_STATE_TTL_SECONDS"
_MAX_PENDING_ENV = "AGENTKIT_FOUNDRY_RESPONSE_STATE_MAX_PENDING"
_MAX_ARGUMENT_BYTES_ENV = "AGENTKIT_FOUNDRY_BROKERED_MAX_ARGUMENT_BYTES"
_MAX_OUTPUT_BYTES_ENV = "AGENTKIT_FOUNDRY_BROKERED_MAX_OUTPUT_BYTES"
_MAX_RESPONSE_STATE_BYTES_ENV = "AGENTKIT_FOUNDRY_RESPONSE_STATE_MAX_BYTES"
_MAX_REQUEST_BODY_BYTES_ENV = "AGENTKIT_FOUNDRY_MAX_REQUEST_BODY_BYTES"
_LEGACY_MAX_REQUEST_BODY_BYTES_ENV = "AGENTKIT_FOUNDRY_REQUEST_BODY_MAX_BYTES"
_MAX_MODEL_MESSAGES_BYTES_ENV = "AGENTKIT_FOUNDRY_BROKERED_MAX_MODEL_MESSAGES_BYTES"
_CONTINUATION_PROOF_ENV = "AGENTKIT_FOUNDRY_BROKERED_CONTINUATION_PROOF"
_CONTINUATION_PROOF_HEADER = "x-agentkit-brokered-continuation-proof"
_CONTINUATION_PROOF_BODY_FIELD = "brokered_continuation_proof"
_MODEL_LOOP_ENV = "AGENTKIT_FOUNDRY_BROKERED_MODEL_LOOP"
_STATE_FILE_ENV = "AGENTKIT_FOUNDRY_RESPONSE_STATE_FILE"
_FOUNDRY_SESSION_ENV = "FOUNDRY_AGENT_SESSION_ID"
_TERMINAL_STATE_FULL = "state_full"
_BROKERED_MODEL_VALIDATION_ERRORS = {
    "InvalidToolArguments": (400, "model tool arguments are invalid"),
    "InvalidToolOutput": (400, "brokered tool output is invalid"),
    "UnsafeBrokeredArguments": (400, "model tool arguments are unsafe"),
    "UnsupportedBrokeredSchema": (400, "brokered tool schema is unsupported"),
    "unknown_brokered_tool": (400, "model requested unknown brokered tool"),
    "invalid_tool_call": (400, "model tool call is invalid"),
    "unsupported_tool_call": (400, "model returned an unsupported tool call"),
    "multiple_tool_calls_unsupported": (400, "brokered mode requires sequential tool calls"),
    "tool_loop_limit_exceeded": (400, "model exceeded the tool call limit"),
    "brokered_arguments_too_large": (413, "model tool arguments are too large"),
    "brokered_output_too_large": (413, "brokered tool output is too large for model resume"),
    "brokered_model_messages_too_large": (413, "model loop messages are too large"),
}


def _new_response_id(previous_response_id: str | None = None) -> str:
    if _AzureResponsesIdGenerator is not None:
        try:
            return _AzureResponsesIdGenerator.new_response_id(previous_response_id or "")
        except TypeError:  # Older/newer SDKs may expose this as a zero-argument helper.
            return _AzureResponsesIdGenerator.new_response_id()
    return f"caresp_{uuid.uuid4().hex}{uuid.uuid4().hex[:18]}"


def _new_message_id(response_id: str) -> str:
    if _AzureResponsesIdGenerator is not None:
        message_id = getattr(_AzureResponsesIdGenerator, "new_message_item_id", None)
        if callable(message_id):
            return message_id(response_id)
    return f"msg_{uuid.uuid4().hex}"


def _new_function_call_id(response_id: str) -> str:
    if _AzureResponsesIdGenerator is not None:
        function_call_id = getattr(_AzureResponsesIdGenerator, "new_function_call_item_id", None)
        if callable(function_call_id):
            return function_call_id(response_id)
    return f"fc_{uuid.uuid4().hex}"


def _usage(result: RunResult) -> dict[str, int]:
    usage = result.usage or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _combine_usage(*usages: Mapping[str, int] | None) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for usage in usages:
        if not usage:
            continue
        prompt_count = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion_count = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        prompt_tokens += prompt_count
        completion_tokens += completion_count
        total_tokens += int(usage.get("total_tokens", prompt_count + completion_count) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _error(message: str, status: int = 400, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "code": code}},
        status_code=status,
    )


def _state_storage_error() -> JSONResponse:
    return _error(
        "brokered response state storage unavailable",
        status=503,
        code="brokered_response_state_storage_error",
    )


def _state_too_large_error() -> JSONResponse:
    return _error(
        "brokered response state exceeds the configured byte limit",
        status=413,
        code="brokered_response_state_too_large",
    )


def _state_full_error() -> JSONResponse:
    return _error(
        "too many pending brokered responses",
        status=429,
        code="brokered_response_state_full",
    )


def _request_too_large_error() -> JSONResponse:
    return _error(
        "Foundry request body is too large",
        status=413,
        code="request_too_large",
    )


def _model_response_too_large_error() -> JSONResponse:
    return _error(
        "model response is too large to retain safely",
        status=502,
        code="ModelResponseTooLarge",
    )


def _brokered_model_run_error(exc: Exception) -> JSONResponse:
    status = 502
    if isinstance(exc, AgentRunError):
        normalized = normalized_model_error_details(exc)
        if normalized is not None:
            status, error = normalized
            return JSONResponse({"error": error}, status_code=status)
        # Validation exceptions may include model-supplied names or argument paths.
        validation = _BROKERED_MODEL_VALIDATION_ERRORS.get(exc.code) if type(exc.code) is str else None
        if validation is not None:
            status, message = validation
            return _error(message, status=status, code=exc.code)
        if type(exc.status) is int and 400 <= exc.status <= 599:
            status = exc.status
    return _error("model resume failed", status=status, code="ModelResumeError")


def _non_brokered_agent_run_error(exc: AgentRunError) -> JSONResponse:
    normalized = normalized_model_error_details(exc)
    if normalized is not None:
        status, error = normalized
        return JSONResponse({"error": error}, status_code=status)
    if exc.status < 500:
        return _error(str(exc), status=exc.status, code=exc.code)
    logger.warning("non-brokered Foundry runtime request failed: %s", exc, exc_info=True)
    status = exc.status if 500 <= exc.status <= 599 else 502
    return _error(
        "agent runtime request failed",
        status=status,
        code="RuntimeFailure",
    )


def _non_brokered_unexpected_runtime_error() -> JSONResponse:
    logger.exception("non-brokered Foundry runtime request failed unexpectedly")
    return _error(
        "agent runtime request failed",
        status=502,
        code="RuntimeFailure",
    )


def _message_to_prompt(message: Any) -> str:
    if isinstance(message, str):
        return message
    return json.dumps(message, separators=(",", ":"), sort_keys=True)


def _session_id_from_request(request: Request) -> str | None:
    # The hosted sandbox identity is authoritative when present. Query and
    # header carriers remain ordered compatibility fallbacks for local use.
    value = os.environ.get(_FOUNDRY_SESSION_ENV)
    if value and value.strip():
        return value.strip()
    for name in ("agent_session_id", "session_id"):
        value = request.query_params.get(name)
        if value and value.strip():
            return value.strip()
    for name in ("x-agent-session-id", "x-agentkit-session-id"):
        value = request.headers.get(name)
        if value and value.strip():
            return value.strip()
    return None


class _SessionIdentityConflict(ValueError):
    pass


def _clean_session_id(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _effective_responses_session_id(
    request: Request,
    data: Mapping[str, Any],
    *,
    enforce_trusted_precedence: bool = False,
) -> str | None:
    # The hosted platform identity describes the sandbox that actually received
    # the request, so prefer it over caller-controlled routing fields. Body
    # fields remain useful for direct/local protocol fidelity when the hosted
    # runtime environment is unavailable.
    hosted = _clean_session_id(os.environ.get(_FOUNDRY_SESSION_ENV))
    if not enforce_trusted_precedence:
        if hosted:
            return hosted
        for name in ("agent_session_id", "session_id"):
            value = _clean_session_id(data.get(name))
            if value:
                return value
        return _session_id_from_request(request)

    # Brokered continuations persist a session binding. The hosted ingress
    # contract strips/replaces x-agent-session-id, so that gateway-owned header
    # and the sandbox environment must not be silently replaced by local
    # body/query compatibility fields. Matching duplicates remain valid.
    gateway = _clean_session_id(request.headers.get("x-agent-session-id"))
    compatibility = [
        _clean_session_id(data.get("agent_session_id")),
        _clean_session_id(data.get("session_id")),
        _clean_session_id(request.query_params.get("agent_session_id")),
        _clean_session_id(request.query_params.get("session_id")),
        _clean_session_id(request.headers.get("x-agentkit-session-id")),
    ]
    trusted = hosted or gateway
    if trusted is not None and any(
        value is not None and value != trusted
        for value in (hosted, gateway, *compatibility)
    ):
        raise _SessionIdentityConflict("conflicting Foundry session identities")
    if trusted is not None:
        return trusted
    return next((value for value in compatibility if value), None)


def _continuation_proof_matches(
    *,
    configured: str,
    request: Request,
    data: Mapping[str, Any],
) -> bool:
    candidates: list[str] = []
    header = request.headers.get(_CONTINUATION_PROOF_HEADER)
    if isinstance(header, str) and header:
        candidates.append(header)
    body = data.get(_CONTINUATION_PROOF_BODY_FIELD)
    if isinstance(body, str) and body:
        candidates.append(body)
    configured_bytes = configured.encode("utf-8", errors="surrogatepass")
    return any(
        hmac.compare_digest(candidate.encode("utf-8", errors="surrogatepass"), configured_bytes)
        for candidate in candidates
    )


def _responses_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("input_text") or block.get("output_text")
                if text is not None:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def _responses_input_to_run_request(value: Any, *, session_id: str | None) -> RunRequest:
    """Extract a RunRequest from common non-streaming Responses API input shapes."""
    if isinstance(value, str):
        return RunRequest(prompt=value, session_id=session_id)

    if isinstance(value, list) and all(isinstance(item, dict) and "role" in item for item in value):
        history: list[ConversationTurn] = []
        for item in value:
            role = str(item.get("role") or "")
            phase = item.get("phase") if role == "assistant" else None
            if phase not in (None, "commentary", "final_answer"):
                raise ValueError("Responses assistant phase must be commentary or final_answer")
            text = _responses_content_to_text(item.get("content"))
            if role in FORWARDED_ROLES and text:
                history.append(ConversationTurn(role=role, text=text, phase=phase))
        if not history:
            return RunRequest(prompt="", session_id=session_id)
        last = history[-1]
        if last.role != "user":
            raise ValueError("Responses input list final message must have role 'user'")
        return RunRequest(prompt=last.text, history=tuple(history[:-1]), session_id=session_id)

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = _responses_content_to_text(item.get("content"))
                if text:
                    parts.append(text)
                    continue
            parts.append(str(item))
        if parts:
            return RunRequest(prompt="\n".join(parts), session_id=session_id)

    return RunRequest(prompt=json.dumps(value, separators=(",", ":"), sort_keys=True), session_id=session_id)


def _responses_payload(
    spec: AgentSpec,
    result: RunResult,
    *,
    previous_response_id: str | None = None,
    response_id: str | None = None,
    created_at: int | None = None,
) -> dict[str, Any]:
    response_id = response_id or _new_response_id(previous_response_id)
    payload = responses_payload(
        spec.model.name,
        result,
        previous_response_id=previous_response_id,
        response_id=response_id,
        message_id=_new_message_id(response_id),
        created_at=created_at,
    )
    # The hosted SDK associates each message with its Foundry response ID.
    payload["output"][0]["response_id"] = response_id
    return payload


@dataclass(frozen=True)
class _PendingCall:
    call_id: str
    item_id: str
    tool: BrokeredToolDefinition
    arguments: dict[str, Any]


@dataclass
class _HostedResponseState:
    response_id: str
    session_id: str | None
    pending_calls: dict[str, _PendingCall]
    expires_at: float
    status: str = "pending"
    accepted_output_digests: dict[str, str] = field(default_factory=dict)
    accepted_output_sizes: dict[str, int] = field(default_factory=dict)
    final_payload: dict[str, Any] | None = None
    terminal_error: str | None = None
    model_messages: list[dict[str, Any]] | None = None
    model_api: ModelAPI | None = None
    initial_usage: dict[str, int] = field(default_factory=dict)
    final_persistence_pending: bool = False
    # All rounds share one bounded store entry. A completed round retains its
    # response/call pairing and exact reply so retries cannot repeat model work.
    response_calls: dict[str, str] = field(default_factory=dict)
    continuation_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    conversation_history: list[dict[str, str]] = field(default_factory=list)


class _StateExpired(KeyError):
    def __init__(self, response_id: str, state: _HostedResponseState) -> None:
        super().__init__(response_id)
        self.state = state


class _StateStoreFull(Exception):
    pass


class _StatePersistenceError(Exception):
    pass


class _StateSizeLimitExceeded(Exception):
    pass


class _SerializedPayloadTooLarge(ValueError):
    pass


class _InvalidUnicodeValue(ValueError):
    pass


class _RequestBodyTooLarge(ValueError):
    pass


def _tool_to_state_payload(tool: BrokeredToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "brokeredClass": tool.brokered_class,
        "parameters": dict(tool.parameters),
        "schemaDigest": tool.schema_digest,
    }


def _tool_from_state_payload(data: Mapping[str, Any]) -> BrokeredToolDefinition:
    brokered_class = data.get("brokeredClass")
    if brokered_class not in {"read", "write", "coordination"}:
        raise ValueError("stored brokered tool class is invalid")
    parameters = data.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError("stored brokered tool parameters must be an object")
    schema_digest = data.get("schemaDigest")
    return BrokeredToolDefinition(
        name=str(data.get("name") or ""),
        description=str(data.get("description") or ""),
        brokered_class=brokered_class,  # type: ignore[arg-type]
        parameters=dict(parameters),
        schema_digest=str(schema_digest) if schema_digest else None,
    )


def _pending_call_to_state_payload(call: _PendingCall) -> dict[str, Any]:
    return {
        "callID": call.call_id,
        "itemID": call.item_id,
        "tool": _tool_to_state_payload(call.tool),
        "arguments": dict(call.arguments),
    }


def _pending_call_from_state_payload(data: Mapping[str, Any]) -> _PendingCall:
    tool = data.get("tool")
    if not isinstance(tool, Mapping):
        raise ValueError("stored pending call tool must be an object")
    arguments = data.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise ValueError("stored pending call arguments must be an object")
    return _PendingCall(
        call_id=str(data.get("callID") or ""),
        item_id=str(data.get("itemID") or ""),
        tool=_tool_from_state_payload(tool),
        arguments=dict(arguments),
    )


def _persistence_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _state_to_payload(state: _HostedResponseState) -> dict[str, Any]:
    payload = {
        "responseID": state.response_id,
        "sessionID": state.session_id,
        "pendingCalls": {call_id: _pending_call_to_state_payload(call) for call_id, call in state.pending_calls.items()},
        "expiresAt": state.expires_at,
        "status": state.status,
        "acceptedOutputDigests": dict(state.accepted_output_digests),
        "acceptedOutputSizes": dict(state.accepted_output_sizes),
        "finalPayload": state.final_payload,
        "modelMessages": state.model_messages,
        "initialUsage": dict(state.initial_usage),
    }
    if state.model_api is not None:
        payload["modelAPI"] = state.model_api
    if state.terminal_error is not None:
        payload["terminalError"] = state.terminal_error
    if state.response_calls:
        payload["responseCalls"] = dict(state.response_calls)
    if state.continuation_payloads:
        payload["continuationPayloads"] = dict(state.continuation_payloads)
    if state.conversation_history:
        payload["conversationHistory"] = list(state.conversation_history)
    return payload


def _state_from_payload(data: Mapping[str, Any]) -> _HostedResponseState:
    pending_calls_raw = data.get("pendingCalls", {})
    if not isinstance(pending_calls_raw, Mapping):
        raise ValueError("stored pendingCalls must be an object")
    accepted_output_digests = data.get("acceptedOutputDigests")
    accepted_output_sizes = data.get("acceptedOutputSizes")
    if accepted_output_digests is None:
        accepted_outputs = data.get("acceptedOutputs", {})
        if not isinstance(accepted_outputs, Mapping):
            raise ValueError("stored acceptedOutputs must be an object")
        accepted_output_digests = {
            str(key): _output_digest(str(value)) for key, value in accepted_outputs.items()
        }
        accepted_output_sizes = {
            str(key): len(str(value).encode("utf-8")) for key, value in accepted_outputs.items()
        }
    elif not isinstance(accepted_output_digests, Mapping):
        raise ValueError("stored acceptedOutputDigests must be an object")
    if accepted_output_sizes is None:
        accepted_output_sizes = {}
    if not isinstance(accepted_output_sizes, Mapping):
        raise ValueError("stored acceptedOutputSizes must be an object")
    final_payload = data.get("finalPayload")
    terminal_error = data.get("terminalError")
    model_messages = data.get("modelMessages")
    # Unmarked released model-loop state predates Responses support.
    model_api = data.get("modelAPI", "chat_completions" if model_messages is not None else None)
    if model_api is not None and model_api not in MODEL_APIS:
        raise ValueError("stored modelAPI must be chat_completions or responses")
    initial_usage = data.get("initialUsage", {})
    response_calls = data.get("responseCalls", {})
    continuation_payloads = data.get("continuationPayloads", {})
    conversation_history = data.get("conversationHistory", [])
    if not isinstance(response_calls, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in response_calls.items()):
        raise ValueError("stored responseCalls must map response IDs to call IDs")
    if not isinstance(continuation_payloads, dict) or not all(isinstance(k, str) and isinstance(v, dict) for k, v in continuation_payloads.items()):
        raise ValueError("stored continuationPayloads must map call IDs to response objects")
    if not isinstance(conversation_history, list) or not all(
        isinstance(turn, dict) and set(turn) in ({"role", "content"}, {"role", "content", "phase"})
        and turn["role"] in {"user", "assistant"} and isinstance(turn["content"], str)
        and ("phase" not in turn or (
            turn["role"] == "assistant" and turn["phase"] in (None, "commentary", "final_answer")
        ))
        for turn in conversation_history
    ):
        raise ValueError("stored conversationHistory must contain user and assistant text")
    if final_payload is not None and not isinstance(final_payload, dict):
        raise ValueError("stored finalPayload must be an object")
    if terminal_error is not None and not isinstance(terminal_error, str):
        raise ValueError("stored terminalError must be a string")
    if terminal_error not in {None, _TERMINAL_STATE_FULL}:
        raise ValueError("stored terminalError is invalid")
    if model_messages is not None and not isinstance(model_messages, list):
        raise ValueError("stored modelMessages must be an array")
    if not isinstance(initial_usage, Mapping):
        raise ValueError("stored initialUsage must be an object")
    status = str(data.get("status") or "pending")
    expires_at = float(data.get("expiresAt") or 0)
    if not math.isfinite(expires_at):
        raise ValueError("stored expiresAt must be finite")
    accepted = {str(key): str(value) for key, value in accepted_output_digests.items()}
    accepted_sizes: dict[str, int] = {}
    for key, value in accepted_output_sizes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("stored acceptedOutputSizes values must be non-negative integers")
        accepted_sizes[str(key)] = value
    if any(key not in accepted for key in accepted_sizes):
        raise ValueError("stored acceptedOutputSizes contains an unknown call id")
    if final_payload is None and terminal_error is None and (accepted or status == "resuming"):
        # Persisted in-progress state without a final payload means the process
        # stopped between accepting the continuation and completing the resume.
        # Clear it on load so Orka can retry instead of being stuck behind
        # duplicate_continuation_in_progress until TTL expiry.
        status = "pending"
        accepted = {key: value for key, value in accepted.items() if key in continuation_payloads}
        accepted_sizes = {key: value for key, value in accepted_sizes.items() if key in accepted}
    return _HostedResponseState(
        response_id=str(data.get("responseID") or ""),
        session_id=str(data["sessionID"]) if data.get("sessionID") is not None else None,
        pending_calls={str(call_id): _pending_call_from_state_payload(call) for call_id, call in pending_calls_raw.items() if isinstance(call, Mapping)},
        expires_at=expires_at,
        status=status,
        accepted_output_digests=accepted,
        accepted_output_sizes=accepted_sizes,
        final_payload=final_payload,
        terminal_error=terminal_error,
        model_messages=model_messages,
        model_api=model_api,
        initial_usage={str(key): int(value or 0) for key, value in initial_usage.items()},
        response_calls=response_calls,
        continuation_payloads=continuation_payloads,
        conversation_history=conversation_history,
    )


def _state_has_replay(state: _HostedResponseState) -> bool:
    return state.final_payload is not None or state.terminal_error is not None or bool(state.continuation_payloads)


class _FoundryResponseStateStore:
    """Continuation store for hosted Responses brokered calls.

    Without a file path this is in-memory only. With a file path, the store
    persists pending/final response state using atomic JSON writes so a restarted
    single-replica/sticky deployment can resume known response IDs. Initial model
    work reserves capacity in memory; a reservation may provisionally claim a
    completed replay entry, but eviction is committed only when pending state is
    installed in its place.
    """

    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int,
        max_bytes: int,
        state_file: str | Path | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.state_file = Path(state_file) if state_file else None
        self._states: dict[str, _HostedResponseState] = {}
        self._entry_sizes: dict[str, int] = {}
        self._active_resumes: dict[str, asyncio.Task[Any]] = {}
        self._reservations: dict[str, str | None] = {}
        self._lock = threading.RLock()
        self._load()

    @property
    def backend_name(self) -> str:
        return "file" if self.state_file else "memory"

    def max_accepted_output_bytes(self) -> int:
        with self._lock:
            now = time.time()
            active_resume_ids = self._active_resume_ids()
            return max(
                (
                    size
                    for response_id, state in self._states.items()
                    if state.expires_at > now
                    or (state.status == "resuming" and response_id in active_resume_ids)
                    for size in state.accepted_output_sizes.values()
                ),
                default=0,
            )

    def _active_resume_ids(self) -> set[str]:
        finished = [response_id for response_id, task in self._active_resumes.items() if task.done()]
        for response_id in finished:
            self._active_resumes.pop(response_id, None)
        return set(self._active_resumes)

    def _claimed_response_ids(
        self,
        states: Mapping[str, _HostedResponseState],
        *,
        exclude_reservation_id: str | None = None,
    ) -> set[str]:
        return {
            claimed_response_id
            for reservation_id, claimed_response_id in self._reservations.items()
            if reservation_id != exclude_reservation_id
            and claimed_response_id is not None
            and claimed_response_id in states
        }

    def _reservation_slots(
        self,
        states: Mapping[str, _HostedResponseState],
        *,
        exclude_reservation_id: str | None = None,
    ) -> int:
        return sum(
            1
            for reservation_id, claimed_response_id in self._reservations.items()
            if reservation_id != exclude_reservation_id
            and (claimed_response_id is None or claimed_response_id not in states)
        )

    @staticmethod
    def _purge_expired_from(
        states: dict[str, _HostedResponseState],
        *,
        now: float,
        active_resume_ids: set[str],
    ) -> bool:
        expired = [
            response_id
            for response_id, state in states.items()
            if state.expires_at <= now and not (state.status == "resuming" and response_id in active_resume_ids)
        ]
        for response_id in expired:
            states.pop(response_id, None)
        return bool(expired)

    @staticmethod
    def _evict_completed_from(
        states: dict[str, _HostedResponseState],
        *,
        target: int,
        excluded_response_ids: set[str] | None = None,
    ) -> bool:
        if len(states) <= target:
            return False
        excluded = excluded_response_ids or set()
        completed = sorted(
            (
                entry
                for entry in states.values()
                if entry.status == "completed"
                and _state_has_replay(entry)
                and not entry.final_persistence_pending
                and entry.response_id not in excluded
            ),
            key=lambda entry: entry.expires_at,
        )
        changed = False
        for entry in completed:
            states.pop(entry.response_id, None)
            changed = True
            if len(states) <= target:
                break
        return changed

    @staticmethod
    def _completed_reservation_candidate(
        states: Mapping[str, _HostedResponseState],
        *,
        excluded_response_ids: set[str],
    ) -> str | None:
        candidates = sorted(
            (
                entry
                for entry in states.values()
                if entry.status == "completed"
                and _state_has_replay(entry)
                and not entry.final_persistence_pending
                and entry.response_id not in excluded_response_ids
            ),
            key=lambda entry: entry.expires_at,
        )
        return candidates[0].response_id if candidates else None

    def _commit(
        self,
        states: dict[str, _HostedResponseState],
        *,
        data: bytes | None = None,
        entry_sizes: Mapping[str, int] | None = None,
    ) -> None:
        if data is None:
            data = self._serialize(states)
        if entry_sizes is None:
            entry_sizes = self._entry_sizes_for(states)
        self._persist(data)
        installed = states
        pending_finals = [
            response_id
            for response_id, state in states.items()
            if state.final_persistence_pending and _state_has_replay(state)
        ]
        if pending_finals:
            installed = dict(states)
            for response_id in pending_finals:
                durable = deepcopy(states[response_id])
                durable.final_persistence_pending = False
                installed[response_id] = durable
        self._states = installed
        self._entry_sizes = {response_id: entry_sizes[response_id] for response_id in installed}

    def _serialized_state_entry_size(self, response_id: str, state: _HostedResponseState) -> int:
        try:
            key_data = _bounded_json_bytes(response_id, max_bytes=self.max_bytes)
            state_data = _bounded_json_bytes(_state_to_payload(state), max_bytes=self.max_bytes)
        except _SerializedPayloadTooLarge as exc:
            raise _StateSizeLimitExceeded("Foundry response state exceeds configured byte limit") from exc
        except (TypeError, ValueError) as exc:
            raise _StatePersistenceError("Foundry response state serialization failed") from exc
        return len(key_data) + 1 + len(state_data)

    def _entry_sizes_for(
        self,
        states: Mapping[str, _HostedResponseState],
        *,
        recompute_response_id: str | None = None,
    ) -> dict[str, int]:
        return {
            response_id: (
                self._serialized_state_entry_size(response_id, state)
                if response_id == recompute_response_id or response_id not in self._entry_sizes
                else self._entry_sizes[response_id]
            )
            for response_id, state in states.items()
        }

    def _serialize_candidate(
        self,
        states: dict[str, _HostedResponseState],
        *,
        response_id: str,
        excluded_response_ids: set[str],
    ) -> tuple[bytes, dict[str, int]]:
        entry_sizes = self._entry_sizes_for(states, recompute_response_id=response_id)

        envelope_bytes = len(b'{"states":{}}')
        candidate_bytes = envelope_bytes + entry_sizes[response_id]
        if candidate_bytes > self.max_bytes:
            raise _StateSizeLimitExceeded("Foundry response state exceeds configured byte limit")

        state_count = len(states)
        total_bytes = envelope_bytes + sum(entry_sizes.values()) + max(state_count - 1, 0)
        excluded = {*excluded_response_ids, response_id}
        evictable = sorted(
            (
                state
                for state in states.values()
                if state.status == "completed"
                and _state_has_replay(state)
                and not state.final_persistence_pending
                and state.response_id not in excluded
            ),
            key=lambda state: state.expires_at,
        )
        for completed in evictable:
            if total_bytes <= self.max_bytes:
                break
            states.pop(completed.response_id, None)
            total_bytes -= entry_sizes[completed.response_id]
            if state_count > 1:
                total_bytes -= 1
            state_count -= 1
        if total_bytes > self.max_bytes:
            raise _StateStoreFull("brokered response state byte capacity is full")
        return self._serialize(states), {current_response_id: entry_sizes[current_response_id] for current_response_id in states}

    def _add_locked(self, state: _HostedResponseState, *, reservation_id: str | None = None) -> None:
        if reservation_id is not None:
            if reservation_id != state.response_id or reservation_id not in self._reservations:
                raise RuntimeError("Foundry response state reservation is missing")
        states = dict(self._states)
        self._purge_expired_from(states, now=time.time(), active_resume_ids=self._active_resume_ids())
        is_new_state = state.response_id not in states
        reservation_slots = self._reservation_slots(
            states,
            exclude_reservation_id=reservation_id,
        )
        if (
            is_new_state
            and len(states) + reservation_slots >= self.max_entries
            and any(
                entry.final_persistence_pending and entry.status == "completed" and _state_has_replay(entry)
                for entry in states.values()
            )
        ):
            self._commit(states)
            states = dict(self._states)

        claimed_response_id = self._reservations.get(reservation_id) if reservation_id is not None else None
        if claimed_response_id is not None:
            states.pop(claimed_response_id, None)
        reservation_slots = self._reservation_slots(
            states,
            exclude_reservation_id=reservation_id,
        )
        protected_response_ids = self._claimed_response_ids(
            states,
            exclude_reservation_id=reservation_id,
        )
        if len(states) + reservation_slots >= self.max_entries and is_new_state:
            self._evict_completed_from(
                states,
                target=max(self.max_entries - reservation_slots - 1, 0),
                excluded_response_ids=protected_response_ids,
            )
        if len(states) + reservation_slots >= self.max_entries and is_new_state:
            raise _StateStoreFull("too many pending brokered responses")
        states[state.response_id] = state
        data, entry_sizes = self._serialize_candidate(
            states,
            response_id=state.response_id,
            excluded_response_ids=protected_response_ids,
        )
        states[state.response_id] = deepcopy(state)
        self._commit(states, data=data, entry_sizes=entry_sizes)
        if reservation_id is not None:
            self._reservations.pop(reservation_id, None)

    def reserve(self, response_id: str) -> None:
        with self._lock:
            if response_id in self._states or response_id in self._reservations:
                raise RuntimeError("Foundry response state ID is already in use")
            states = dict(self._states)
            changed = self._purge_expired_from(
                states,
                now=time.time(),
                active_resume_ids=self._active_resume_ids(),
            )
            reservation_slots = self._reservation_slots(states)
            if (
                len(states) + reservation_slots >= self.max_entries
                and any(
                    entry.final_persistence_pending and entry.status == "completed" and _state_has_replay(entry)
                    for entry in states.values()
                )
            ):
                self._commit(states)
                states = dict(self._states)
                changed = False
                reservation_slots = self._reservation_slots(states)

            if len(states) + reservation_slots < self.max_entries:
                if changed:
                    self._commit(states)
                self._reservations[response_id] = None
                return

            claimed_response_id = self._completed_reservation_candidate(
                states,
                excluded_response_ids=self._claimed_response_ids(states),
            )
            if claimed_response_id is None:
                raise _StateStoreFull("too many pending brokered responses")
            if changed:
                self._commit(states)
            self._reservations[response_id] = claimed_response_id

    def release_reservation(self, response_id: str) -> None:
        with self._lock:
            self._reservations.pop(response_id, None)

    def add(self, state: _HostedResponseState) -> None:
        with self._lock:
            self._add_locked(state)

    def add_reserved(self, state: _HostedResponseState) -> None:
        with self._lock:
            self._add_locked(state, reservation_id=state.response_id)

    def save(self, state: _HostedResponseState) -> None:
        with self._lock:
            if state.response_id not in self._states:
                return
            states = dict(self._states)
            states[state.response_id] = state
            data, entry_sizes = self._serialize_candidate(
                states,
                response_id=state.response_id,
                excluded_response_ids=self._claimed_response_ids(states),
            )
            states[state.response_id] = deepcopy(state)
            self._commit(states, data=data, entry_sizes=entry_sizes)

    def cache_in_memory(self, state: _HostedResponseState) -> bool:
        """Cache one entry after a durable transition could not be written.

        Persisted unfinalized continuations are normalized back to pending by
        ``_state_from_payload`` on restart. In-process caching also preserves a
        successfully computed final payload until an identical retry can persist it.
        """

        with self._lock:
            if state.response_id not in self._states:
                return False
            states = dict(self._states)
            states[state.response_id] = state
            try:
                _, entry_sizes = self._serialize_candidate(
                    states,
                    response_id=state.response_id,
                    excluded_response_ids=self._claimed_response_ids(states),
                )
            except (_StateSizeLimitExceeded, _StateStoreFull, _StatePersistenceError):
                return False
            states[state.response_id] = deepcopy(state)
            self._states = states
            self._entry_sizes = entry_sizes
            return True

    def mark_resume_active(self, response_id: str) -> None:
        with self._lock:
            task = asyncio.current_task()
            if task is not None:
                self._active_resumes[response_id] = task

    def mark_resume_inactive(self, response_id: str) -> None:
        with self._lock:
            self._active_resumes.pop(response_id, None)

    def evict_completed_to_capacity(self, *, reserve_slots: int = 0) -> None:
        with self._lock:
            states = dict(self._states)
            reservation_slots = self._reservation_slots(states)
            target = max(self.max_entries - reservation_slots - reserve_slots, 0)
            if self._evict_completed_from(
                states,
                target=target,
                excluded_response_ids=self._claimed_response_ids(states),
            ):
                self._commit(states)

    def get(self, response_id: str) -> _HostedResponseState:
        with self._lock:
            state = self._states.get(response_id)
            if state is None:
                state = next((entry for entry in self._states.values() if response_id in entry.response_calls or (
                    entry.session_id and entry.final_payload and entry.final_payload.get("id") == response_id
                )), None)
            if state is None:
                raise KeyError(response_id)
            root_id = state.response_id
            states = dict(self._states)
            now = time.time()
            active_resume_ids = self._active_resume_ids()
            target_expired = state.expires_at <= now and not (
                state.status == "resuming" and root_id in active_resume_ids
            )
            changed = self._purge_expired_from(states, now=now, active_resume_ids=active_resume_ids)
            if changed:
                self._commit(states)
            if target_expired:
                raise _StateExpired(response_id, deepcopy(state))
            current = self._states.get(root_id)
            if current is None:
                raise KeyError(response_id)
            return deepcopy(current)

    def purge_expired(self) -> None:
        with self._lock:
            states = dict(self._states)
            if self._purge_expired_from(states, now=time.time(), active_resume_ids=self._active_resume_ids()):
                self._commit(states)

    def _load(self) -> None:
        if self.state_file is None or not self.state_file.exists():
            return
        try:
            if self.state_file.stat().st_size > self.max_bytes:
                raise _StateSizeLimitExceeded("Foundry response state file exceeds configured byte limit")
            with self.state_file.open("rb") as handle:
                raw = handle.read(self.max_bytes + 1)
            if len(raw) > self.max_bytes:
                raise _StateSizeLimitExceeded("Foundry response state file exceeds configured byte limit")
            data = json.loads(raw)
            if not isinstance(data, Mapping):
                raise ValueError("Foundry response state file root must be an object")
            states = data.get("states", {})
            if not isinstance(states, Mapping):
                raise ValueError("Foundry response state file states must be an object")
            if len(states) > self.max_entries:
                raise ValueError("Foundry response state file exceeds configured entry limit")
            if not all(isinstance(state, Mapping) for state in states.values()):
                raise ValueError("Foundry response state entries must be objects")
            loaded_states = {
                str(response_id): _state_from_payload(state)
                for response_id, state in states.items()
            }
            self._serialize(loaded_states)
            self._states = loaded_states
            self._entry_sizes = {}
            self.purge_expired()
        except (
            OSError,
            _StatePersistenceError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            ValueError,
            _StateSizeLimitExceeded,
        ) as exc:
            raise RuntimeError(f"invalid Foundry response state file {self.state_file}: {exc}") from exc

    def _serialize(self, states: Mapping[str, _HostedResponseState]) -> bytes:
        payload = {"states": {response_id: _state_to_payload(state) for response_id, state in states.items()}}
        try:
            return _bounded_json_bytes(payload, max_bytes=self.max_bytes)
        except _SerializedPayloadTooLarge as exc:
            raise _StateSizeLimitExceeded("Foundry response state exceeds configured byte limit") from exc
        except (TypeError, ValueError) as exc:
            raise _StatePersistenceError("Foundry response state serialization failed") from exc

    def _persist(self, data: bytes) -> None:
        if self.state_file is None:
            return
        tmp = self.state_file.with_name(f".{self.state_file.name}.tmp")
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(tmp, flags, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o600)
            tmp.replace(self.state_file)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise _StatePersistenceError("Foundry response state persistence failed") from exc


def _state_ttl_seconds(value: float | None = None) -> float:
    if value is not None:
        parsed = float(value)
        return max(parsed, 0.0) if math.isfinite(parsed) else float(_DEFAULT_STATE_TTL_SECONDS)
    raw = os.environ.get(_STATE_TTL_ENV)
    if not raw:
        return float(_DEFAULT_STATE_TTL_SECONDS)
    try:
        parsed = float(raw)
    except ValueError:
        return float(_DEFAULT_STATE_TTL_SECONDS)
    return max(parsed, 0.0) if math.isfinite(parsed) else float(_DEFAULT_STATE_TTL_SECONDS)


def _positive_int_setting(value: int | None, *, env_name: str, default: int) -> int:
    if value is not None:
        return max(int(value), 1)
    raw = os.environ.get(env_name)
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _max_pending_responses(value: int | None = None) -> int:
    return _positive_int_setting(value, env_name=_MAX_PENDING_ENV, default=_DEFAULT_MAX_PENDING_RESPONSES)


def _max_argument_bytes(value: int | None = None) -> int:
    return _positive_int_setting(value, env_name=_MAX_ARGUMENT_BYTES_ENV, default=_DEFAULT_MAX_ARGUMENT_BYTES)


def _max_output_bytes(value: int | None = None) -> int:
    return _positive_int_setting(value, env_name=_MAX_OUTPUT_BYTES_ENV, default=_DEFAULT_MAX_OUTPUT_BYTES)


def _max_response_state_bytes(value: int | None = None) -> int:
    return _positive_int_setting(
        value,
        env_name=_MAX_RESPONSE_STATE_BYTES_ENV,
        default=_DEFAULT_MAX_RESPONSE_STATE_BYTES,
    )


def _max_request_body_bytes(value: int | None = None) -> int:
    if value is not None or os.environ.get(_MAX_REQUEST_BODY_BYTES_ENV):
        return _positive_int_setting(
            value,
            env_name=_MAX_REQUEST_BODY_BYTES_ENV,
            default=_DEFAULT_MAX_REQUEST_BODY_BYTES,
        )
    return _positive_int_setting(
        None,
        env_name=_LEGACY_MAX_REQUEST_BODY_BYTES_ENV,
        default=_DEFAULT_MAX_REQUEST_BODY_BYTES,
    )


def _max_model_messages_bytes(value: int | None = None) -> int:
    return _positive_int_setting(
        value,
        env_name=_MAX_MODEL_MESSAGES_BYTES_ENV,
        default=_DEFAULT_MAX_MODEL_MESSAGES_BYTES,
    )


def _brokered_model_loop_enabled(value: bool | None = None) -> bool:
    if value is not None:
        return value
    return os.environ.get(_MODEL_LOOP_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _response_state_file(value: str | Path | None = None) -> str | Path | None:
    if value is not None:
        return value
    raw = os.environ.get(_STATE_FILE_ENV)
    return raw.strip() if raw and raw.strip() else None


def _function_call_outputs_from_input(input_value: Any) -> list[dict[str, Any]]:
    items: list[Any]
    if isinstance(input_value, dict):
        items = [input_value]
    elif isinstance(input_value, list):
        items = input_value
    else:
        return []
    outputs: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            outputs.append(item)
    return outputs


def _mapping_value_paths(value: Mapping[Any, Any], parent_path: str):
    for key, child in value.items():
        yield child, f"{parent_path}.{key}"


def _list_value_paths(value: list[Any], parent_path: str):
    for idx, child in enumerate(value):
        yield child, f"{parent_path}[{idx}]"


def _reject_nonfinite_json_values(value: Any, *, path: str = "value") -> None:
    pending: list[tuple[Any, int]] = [(iter(((value, path),)), 0)]
    while pending:
        iterator, depth = pending[-1]
        try:
            current, current_path = next(iterator)
        except StopIteration:
            pending.pop()
            continue
        if depth > _MAX_OUTPUT_DEPTH:
            raise ValueError(f"{path} is nested too deeply")
        if isinstance(current, float) and not math.isfinite(current):
            raise ValueError(f"{current_path} must be finite")
        if isinstance(current, Mapping):
            pending.append((_mapping_value_paths(current, current_path), depth + 1))
        elif isinstance(current, list):
            pending.append((_list_value_paths(current, current_path), depth + 1))


def _parse_output_float(raw: str) -> float:
    try:
        decimal = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError("function_call_output.output must contain valid JSON numbers") from exc
    parsed = float(decimal)
    if not math.isfinite(parsed) or Decimal(str(parsed)) != decimal:
        raise ValueError("function_call_output.output contains a number that cannot be represented exactly")
    return parsed


def _reject_output_constant(raw: str) -> None:
    raise ValueError(f"function_call_output.output contains non-finite number {raw}")


def _reject_duplicate_output_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"function_call_output.output contains duplicate key {key!r}")
        out[key] = value
    return out


def _json_object_from_output(output: Any) -> dict[str, Any]:
    if not isinstance(output, str):
        raise ValueError("function_call_output.output must be a JSON object string")
    try:
        parsed = json.loads(
            output,
            parse_float=_parse_output_float,
            parse_constant=_reject_output_constant,
            object_pairs_hook=_reject_duplicate_output_keys,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("function_call_output.output must be a JSON object string with bounded nesting") from exc
    if not isinstance(parsed, dict):
        raise ValueError("function_call_output.output must be a JSON object")
    approved = parsed.get("approved")
    if not isinstance(approved, bool):
        raise ValueError("function_call_output.output.approved must be a boolean")
    if approved:
        unexpected = set(parsed) - {"approved", "output"}
        if unexpected:
            raise ValueError("approved function_call_output.output contains unsupported fields")
        if "output" not in parsed or not isinstance(parsed["output"], dict):
            raise ValueError("approved function_call_output.output.output must be an object")
    else:
        unexpected = set(parsed) - {"approved", "error"}
        if unexpected:
            raise ValueError("denied function_call_output.output contains unsupported fields")
        if "error" not in parsed or not isinstance(parsed["error"], dict):
            raise ValueError("denied function_call_output.output.error must be an object")
    _reject_nonfinite_json_values(parsed, path="function_call_output.output")
    return parsed


def _utf8_exceeds_limit(value: str, max_bytes: int) -> bool:
    if len(value) > max_bytes:
        return True
    try:
        return len(value.encode("utf-8")) > max_bytes
    except UnicodeEncodeError as exc:
        raise _InvalidUnicodeValue("function_call_output.output must contain valid Unicode") from exc


def _mapping_children(value: Mapping[Any, Any]):
    for key, child in value.items():
        yield key
        yield child


def _json_string_encoded_size(value: str, *, max_bytes: int) -> int:
    size = 2
    for char in value:
        codepoint = ord(char)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise _InvalidUnicodeValue("JSON strings must contain valid Unicode")
        if char in {'"', "\\", "\b", "\f", "\n", "\r", "\t"}:
            size += 2
        elif codepoint < 0x20 or codepoint == 0x7F or 0x80 <= codepoint <= 0xFFFF:
            size += 6
        elif codepoint > 0xFFFF:
            size += 12
        else:
            size += 1
        if size > max_bytes:
            raise _SerializedPayloadTooLarge
    return size


def _encode_json_bounded(value: Any, *, max_bytes: int, sort_keys: bool, collect: bool) -> bytes:
    encoded = bytearray() if collect else None
    total = 0
    encoder = json.JSONEncoder(allow_nan=False, separators=(",", ":"), sort_keys=sort_keys)
    try:
        for chunk in encoder.iterencode(value):
            remaining = max_bytes - total
            if len(chunk) > remaining:
                raise _SerializedPayloadTooLarge
            chunk_bytes = chunk.encode("utf-8")
            if len(chunk_bytes) > remaining:
                raise _SerializedPayloadTooLarge
            total += len(chunk_bytes)
            if encoded is not None:
                encoded.extend(chunk_bytes)
    except RecursionError as exc:
        raise _SerializedPayloadTooLarge from exc
    return bytes(encoded or b"")


async def _read_json_request_bounded(request: Request, *, max_bytes: int) -> tuple[Any, int]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            declared_length = -1
        if declared_length > max_bytes:
            raise _RequestBodyTooLarge

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_bytes:
            raise _RequestBodyTooLarge
        body.extend(chunk)
    return json.loads(body), len(body)


def _bounded_json_bytes(value: Any, *, max_bytes: int) -> bytes:
    pending = [iter((value,))]
    visited = 0
    while pending:
        try:
            current = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        visited += 1
        if visited > max_bytes:
            raise _SerializedPayloadTooLarge
        if isinstance(current, str):
            _json_string_encoded_size(current, max_bytes=max_bytes)
        elif isinstance(current, Mapping):
            pending.append(iter(_mapping_children(current)))
        elif isinstance(current, (list, tuple)):
            pending.append(iter(current))

    _encode_json_bounded(value, max_bytes=max_bytes, sort_keys=False, collect=False)
    return _encode_json_bounded(value, max_bytes=max_bytes, sort_keys=True, collect=True)


def _canonical_output_json(output: dict[str, Any]) -> str:
    _reject_nonfinite_json_values(output)
    return json.dumps(output, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _output_digest(output_json: str) -> str:
    return "sha256:" + hashlib.sha256(output_json.encode("utf-8")).hexdigest()


def _first_typed_value(schema: Mapping[str, Any], key: str, expected_type: type) -> Any:
    if key in schema and isinstance(schema[key], expected_type):
        return schema[key]
    return None


def _enum_value(schema: Mapping[str, Any], expected_type: type) -> Any:
    values = schema.get("enum")
    if isinstance(values, list):
        for value in values:
            if isinstance(value, expected_type):
                return value
    return None


def _integer_schema_bound(value: Any, *, lower: bool, exclusive: bool) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        if exclusive:
            return value + 1 if lower else value - 1
        return value
    if isinstance(value, float) and math.isfinite(value):
        if lower:
            return math.floor(value) + 1 if exclusive else math.ceil(value)
        return math.ceil(value) - 1 if exclusive else math.floor(value)
    return None


def _effective_numeric_bounds(schema: Mapping[str, Any]) -> tuple[Fraction | None, bool, Fraction | None, bool]:
    lower_value: Fraction | None = None
    lower_open = False
    for key, exclusive in (("minimum", False), ("exclusiveMinimum", True)):
        raw = schema.get(key)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            continue
        if isinstance(raw, float) and not math.isfinite(raw):
            raise AgentRunError("brokered tool schema has a non-finite numeric bound", status=400, code="UnsupportedBrokeredSchema")
        try:
            candidate = Fraction(str(raw))
        except (ValueError, ZeroDivisionError) as exc:
            raise AgentRunError("brokered tool schema has an invalid numeric bound", status=400, code="UnsupportedBrokeredSchema") from exc
        if lower_value is None or candidate > lower_value:
            lower_value = candidate
            lower_open = exclusive
        elif candidate == lower_value and exclusive:
            lower_open = True

    upper_value: Fraction | None = None
    upper_open = False
    for key, exclusive in (("maximum", False), ("exclusiveMaximum", True)):
        raw = schema.get(key)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            continue
        if isinstance(raw, float) and not math.isfinite(raw):
            raise AgentRunError("brokered tool schema has a non-finite numeric bound", status=400, code="UnsupportedBrokeredSchema")
        try:
            candidate = Fraction(str(raw))
        except (ValueError, ZeroDivisionError) as exc:
            raise AgentRunError("brokered tool schema has an invalid numeric bound", status=400, code="UnsupportedBrokeredSchema") from exc
        if upper_value is None or candidate < upper_value:
            upper_value = candidate
            upper_open = exclusive
        elif candidate == upper_value and exclusive:
            upper_open = True
    return lower_value, lower_open, upper_value, upper_open


def _fraction_floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def _fraction_ceil(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _fraction_json_candidate(value: Fraction, *, name: str, require_exact: bool) -> int | float:
    if value.denominator == 1:
        integer_text = str(value.numerator)
        try:
            compact_float = float(value.numerator)
        except OverflowError:
            compact_float = math.inf
        if math.isfinite(compact_float) and Fraction(str(compact_float)) == value and len(str(compact_float)) < len(integer_text):
            return compact_float
        return value.numerator
    try:
        candidate = float(value)
    except OverflowError as exc:
        raise AgentRunError(
            f"brokered tool schema for {name!r} has no representable numeric value in bounds",
            status=400,
            code="UnsupportedBrokeredSchema",
        ) from exc
    if not math.isfinite(candidate) or (require_exact and Fraction(str(candidate)) != value):
        raise AgentRunError(
            f"brokered tool schema for {name!r} has no representable numeric value in bounds",
            status=400,
            code="UnsupportedBrokeredSchema",
        )
    return candidate


def _fraction_in_numeric_bounds(
    candidate: Fraction,
    *,
    lower_value: Fraction | None,
    lower_open: bool,
    upper_value: Fraction | None,
    upper_open: bool,
) -> bool:
    if lower_value is not None and (candidate < lower_value or (candidate == lower_value and lower_open)):
        return False
    if upper_value is not None and (candidate > upper_value or (candidate == upper_value and upper_open)):
        return False
    return True


def _integer_candidate_from_bounds(
    *,
    lower_value: Fraction | None,
    lower_open: bool,
    upper_value: Fraction | None,
    upper_open: bool,
    step: int = 1,
) -> int:
    if lower_value is not None:
        quotient = lower_value / step
        multiplier = _fraction_ceil(quotient)
        if lower_open and quotient.denominator == 1:
            multiplier += 1
        return multiplier * step
    if upper_value is not None:
        quotient = upper_value / step
        multiplier = _fraction_floor(quotient)
        if upper_open and quotient.denominator == 1:
            multiplier -= 1
        return multiplier * step
    return 0


def _required_property_names(schema: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    required = schema.get("required")
    if isinstance(required, list):
        names.extend(name for name in required if isinstance(name, str))

    dependent_required = schema.get("dependentRequired")
    min_properties = schema.get("minProperties")
    properties = schema.get("properties")

    changed = True
    while changed:
        changed = False
        if isinstance(dependent_required, Mapping):
            present = set(names)
            for trigger, dependent_names in dependent_required.items():
                if trigger not in present or not isinstance(dependent_names, list):
                    continue
                for dependent_name in dependent_names:
                    if isinstance(dependent_name, str) and dependent_name not in present:
                        names.append(dependent_name)
                        present.add(dependent_name)
                        changed = True
        if isinstance(min_properties, int) and not isinstance(min_properties, bool) and isinstance(properties, Mapping):
            for name in properties:
                if len(names) >= min_properties:
                    break
                if isinstance(name, str) and name not in names:
                    names.append(name)
                    changed = True
                    break

    if isinstance(min_properties, int) and not isinstance(min_properties, bool) and len(names) < min_properties:
        raise AgentRunError(
            "brokered tool schema has minProperties that cannot be synthesized",
            status=400,
            code="UnsupportedBrokeredSchema",
        )
    return names


def _bounded_synthetic_literal(name: str, value: Any, *, depth: int) -> Any:
    pending: list[tuple[Any, int]] = [(value, depth)]
    while pending:
        current, current_depth = pending.pop()
        if current_depth > _MAX_OUTPUT_DEPTH:
            raise AgentRunError(
                f"brokered tool schema for {name!r} exceeds deterministic synthesis depth limit",
                status=413,
                code="brokered_arguments_too_large",
            )
        if isinstance(current, Mapping):
            pending.extend((child, current_depth + 1) for child in current.values())
        elif isinstance(current, list):
            pending.extend((child, current_depth + 1) for child in current)
    return value


def _sample_argument_value(
    name: str,
    schema: Any,
    run_request: RunRequest,
    *,
    budget: list[int] | None = None,
    depth: int = 1,
) -> Any:
    if depth > _MAX_OUTPUT_DEPTH:
        raise AgentRunError(
            f"brokered tool schema for {name!r} exceeds deterministic synthesis depth limit",
            status=413,
            code="brokered_arguments_too_large",
        )
    if budget is None:
        budget = [_MAX_SYNTHETIC_VALUES]
    if budget[0] <= 0:
        raise AgentRunError(
            f"brokered tool schema for {name!r} exceeds deterministic synthesis value budget",
            status=413,
            code="brokered_arguments_too_large",
        )
    budget[0] -= 1
    if not isinstance(schema, Mapping):
        return run_request.prompt
    if "multipleOf" in schema:
        raise AgentRunError(
            f"brokered tool schema for {name!r} has unsupported numeric multipleOf",
            status=400,
            code="UnsupportedBrokeredSchema",
        )
    if "const" in schema:
        return _bounded_synthetic_literal(name, schema["const"], depth=depth)
    if "default" in schema:
        return _bounded_synthetic_literal(name, schema["default"], depth=depth)
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return _bounded_synthetic_literal(name, enum_values[0], depth=depth)

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), schema_type[0] if schema_type else None)

    if schema_type == "null":
        return None

    if schema_type == "boolean":
        for key in ("const", "default"):
            value = _first_typed_value(schema, key, bool)
            if value is not None:
                return value
        value = _enum_value(schema, bool)
        return value if value is not None else True

    if schema_type == "integer":
        for key in ("const", "default"):
            value = _first_typed_value(schema, key, int)
            if value is not None and not isinstance(value, bool):
                return value
        value = _enum_value(schema, int)
        if value is not None and not isinstance(value, bool):
            return value
        lower_candidates = [
            candidate
            for candidate in (
                _integer_schema_bound(schema.get("minimum"), lower=True, exclusive=False),
                _integer_schema_bound(schema.get("exclusiveMinimum"), lower=True, exclusive=True),
            )
            if candidate is not None
        ]
        upper_candidates = [
            candidate
            for candidate in (
                _integer_schema_bound(schema.get("maximum"), lower=False, exclusive=False),
                _integer_schema_bound(schema.get("exclusiveMaximum"), lower=False, exclusive=True),
            )
            if candidate is not None
        ]
        lower = max(lower_candidates, default=0)
        upper = min(upper_candidates) if upper_candidates else None
        if upper is not None and lower > upper:
            if lower_candidates:
                raise AgentRunError(
                    f"brokered tool schema for {name!r} has incompatible integer bounds",
                    status=400,
                    code="UnsupportedBrokeredSchema",
                )
            lower = upper
        multiple_of = schema.get("multipleOf")
        if multiple_of is not None:
            raise AgentRunError(
                f"brokered tool schema for {name!r} has unsupported integer multipleOf",
                status=400,
                code="UnsupportedBrokeredSchema",
            )
        return lower

    if schema_type == "number":
        for key in ("const", "default"):
            value = schema.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
        value = _enum_value(schema, (int, float))
        if value is not None and not isinstance(value, bool):
            return value
        lower_fraction, lower_open, upper_fraction, upper_open = _effective_numeric_bounds(schema)
        has_lower = lower_fraction is not None
        has_upper = upper_fraction is not None
        if lower_fraction is not None and upper_fraction is not None:
            if lower_fraction > upper_fraction or (lower_fraction == upper_fraction and (lower_open or upper_open)):
                raise AgentRunError(
                    f"brokered tool schema for {name!r} has incompatible numeric bounds",
                    status=400,
                    code="UnsupportedBrokeredSchema",
                )
            if lower_fraction == upper_fraction:
                return _fraction_json_candidate(lower_fraction, name=name, require_exact=True)

        multiple_of = schema.get("multipleOf")
        if multiple_of is not None:
            raise AgentRunError(
                f"brokered tool schema for {name!r} has unsupported numeric multipleOf",
                status=400,
                code="UnsupportedBrokeredSchema",
            )

        candidates: list[int | float] = []

        def add_candidate(candidate: int | float) -> None:
            if isinstance(candidate, float) and not math.isfinite(candidate):
                return
            if _fraction_in_numeric_bounds(
                Fraction(str(candidate)),
                lower_value=lower_fraction,
                lower_open=lower_open,
                upper_value=upper_fraction,
                upper_open=upper_open,
            ):
                candidates.append(candidate)

        add_candidate(0)

        for boundary, is_open in (
            (lower_fraction, lower_open),
            (upper_fraction, upper_open),
        ):
            if boundary is None or is_open:
                continue
            try:
                add_candidate(_fraction_json_candidate(boundary, name=name, require_exact=True))
            except AgentRunError:
                pass

        for boundary, is_open, direction in (
            (lower_fraction, lower_open, math.inf),
            (upper_fraction, upper_open, -math.inf),
        ):
            if boundary is None or not is_open:
                continue
            try:
                boundary_float = float(boundary)
            except OverflowError:
                continue
            if math.isfinite(boundary_float):
                add_candidate(math.nextafter(boundary_float, direction))

        integer_candidate = _integer_candidate_from_bounds(
            lower_value=lower_fraction,
            lower_open=lower_open,
            upper_value=upper_fraction,
            upper_open=upper_open,
            step=1,
        )
        add_candidate(integer_candidate)

        if lower_fraction is not None and upper_fraction is not None:
            midpoint = (lower_fraction + upper_fraction) / 2
            try:
                add_candidate(_fraction_json_candidate(midpoint, name=name, require_exact=False))
            except AgentRunError:
                pass

        if candidates:
            return min(
                candidates,
                key=lambda candidate: len(json.dumps(candidate, allow_nan=False, separators=(",", ":"))),
            )

        raise AgentRunError(
            f"brokered tool schema for {name!r} has no representable numeric value in bounds",
            status=400,
            code="UnsupportedBrokeredSchema",
        )

    if schema_type == "array":
        for key in ("const", "default"):
            value = _first_typed_value(schema, key, list)
            if value is not None:
                return value
        value = _enum_value(schema, list)
        if value is not None:
            return value
        min_items = schema.get("minItems", 0)
        if isinstance(min_items, int) and min_items > 0:
            if min_items > _MAX_SYNTHETIC_ARRAY_ITEMS or min_items > budget[0]:
                raise AgentRunError(
                    f"brokered tool schema for {name!r} has minItems too large for deterministic synthesis",
                    status=413,
                    code="brokered_arguments_too_large",
                )
            item_schema = schema.get("items", {})
            return [
                _sample_argument_value(name, item_schema, run_request, budget=budget, depth=depth + 1)
                for _ in range(min_items)
            ]
        return []

    if schema_type == "object":
        for key in ("const", "default"):
            value = _first_typed_value(schema, key, dict)
            if value is not None:
                return value
        nested: dict[str, Any] = {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        for child_name in _required_property_names(schema):
            nested[child_name] = _sample_argument_value(
                child_name,
                properties.get(child_name, {}),
                run_request,
                budget=budget,
                depth=depth + 1,
            )
        return nested

    for key in ("const", "default"):
        value = _first_typed_value(schema, key, str)
        if value is not None:
            return value
    value = _enum_value(schema, str)
    if value is not None:
        return value
    if schema.get("pattern"):
        raise AgentRunError(
            f"brokered tool schema for {name!r} uses pattern without const/default/enum",
            status=400,
            code="UnsupportedBrokeredSchema",
        )
    sample = run_request.prompt if name in {"prompt", "site"} else name
    min_length = schema.get("minLength", 0)
    if isinstance(min_length, int) and min_length > _MAX_SYNTHETIC_STRING_LENGTH:
        raise AgentRunError(
            f"brokered tool schema for {name!r} has minLength too large for deterministic synthesis",
            status=413,
            code="brokered_arguments_too_large",
        )
    if isinstance(min_length, int) and len(sample) < min_length:
        sample = sample + ("x" * (min_length - len(sample)))
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and max_length >= 0 and len(sample) > max_length:
        if isinstance(min_length, int) and min_length > max_length:
            raise AgentRunError(
                f"brokered tool schema for {name!r} has incompatible minLength/maxLength",
                status=400,
                code="UnsupportedBrokeredSchema",
            )
        sample = sample[:max_length]
    return sample


def _schema_has_literal_value(schema: Any) -> bool:
    return isinstance(schema, Mapping) and ("const" in schema or "default" in schema or (isinstance(schema.get("enum"), list) and len(schema.get("enum")) == 1) or schema.get("type") == "null")


def _deterministic_tool_arguments(tool: BrokeredToolDefinition, run_request: RunRequest) -> dict[str, Any]:
    parameters = tool.parameters if isinstance(tool.parameters, Mapping) else {}
    for key in ("const", "default"):
        literal = parameters.get(key)
        if isinstance(literal, Mapping):
            return dict(_bounded_synthetic_literal(tool.name, literal, depth=0))
    enum = parameters.get("enum")
    if isinstance(enum, list):
        if tool.brokered_class != "read" and len(enum) > 1:
            raise AgentRunError(
                f"brokered {tool.brokered_class} tool {tool.name!r} has multiple root enum payloads; deterministic mode refuses to choose side-effecting arguments",
                status=400,
                code="UnsupportedBrokeredSchema",
            )
        for item in enum:
            if isinstance(item, Mapping):
                return dict(_bounded_synthetic_literal(tool.name, item, depth=0))
    properties = parameters.get("properties") if isinstance(parameters.get("properties"), Mapping) else {}
    required_names = _required_property_names(parameters)
    if tool.brokered_class != "read":
        for name in required_names:
            if not _schema_has_literal_value(properties.get(name, {})):
                raise AgentRunError(
                    f"brokered {tool.brokered_class} tool {tool.name!r} requires non-literal argument {name!r}; deterministic mode refuses to synthesize side-effecting arguments",
                    status=400,
                    code="UnsupportedBrokeredSchema",
                )
    arguments: dict[str, Any] = {}
    synthesis_budget = [_MAX_SYNTHETIC_VALUES]
    for name in required_names:
        arguments[name] = _sample_argument_value(name, properties.get(name, {}), run_request, budget=synthesis_budget)
    if tool.name == "conformance_read" and "probe" in properties and "probe" not in arguments:
        arguments["probe"] = True
    if not arguments and "prompt" in properties:
        if tool.brokered_class != "read" and not _schema_has_literal_value(properties["prompt"]):
            raise AgentRunError(
                f"brokered {tool.brokered_class} tool {tool.name!r} has a non-literal optional prompt; deterministic mode refuses to synthesize side-effecting arguments",
                status=400,
                code="UnsupportedBrokeredSchema",
            )
        arguments["prompt"] = _sample_argument_value("prompt", properties["prompt"], run_request, budget=synthesis_budget)
    return arguments


def _prompt_mentions_tool(prompt: str, tool_name: str) -> bool:
    pattern = r"(?<![A-Za-z0-9_-])" + re.escape(tool_name) + r"(?![A-Za-z0-9_-])"
    return re.search(pattern, prompt, flags=re.IGNORECASE) is not None


def _select_brokered_tool(tools: list[BrokeredToolDefinition], run_request: RunRequest) -> BrokeredToolDefinition | None:
    matches = [tool for tool in tools if _prompt_mentions_tool(run_request.prompt, tool.name)]
    if len(matches) == 1:
        return matches[0]
    if len(tools) == 1 and tools[0].name == "conformance_read":
        return tools[0]
    return None




def _validate_model_brokered_arguments(value: Any, *, path: str = "arguments") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise AgentRunError("model brokered tool argument keys must be strings", status=400, code="InvalidToolArguments")
            if _unsafe_brokered_key(key) is not None:
                raise AgentRunError(
                    f"model brokered tool argument {path}.{key} is not safe",
                    status=400,
                    code="UnsafeBrokeredArguments",
                )
            _validate_model_brokered_arguments(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _validate_model_brokered_arguments(child, path=f"{path}[{idx}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AgentRunError(
            f"model brokered tool argument {path} must be finite",
            status=400,
            code="InvalidToolArguments",
        )
    elif isinstance(value, str) and _unsafe_brokered_text(value):
        raise AgentRunError(
            f"model brokered tool argument {path} contains unsafe text",
            status=400,
            code="UnsafeBrokeredArguments",
        )




def _schema_types(schema: Mapping[str, Any]) -> list[str]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return [schema_type]
    if isinstance(schema_type, list):
        return [item for item in schema_type if isinstance(item, str)]
    return []


def _decimal_json_number(value: int | float) -> Decimal:
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as exc:
        raise AgentRunError("model brokered tool argument is not a finite JSON number", status=400, code="InvalidToolArguments") from exc
    if not decimal.is_finite():
        raise AgentRunError("model brokered tool argument is not a finite JSON number", status=400, code="InvalidToolArguments")
    return decimal


def _json_type_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _decimal_json_number(left) == _decimal_json_number(right)
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_json_type_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_json_type_equal(left[key], right[key]) for key in left)
        )
    return left == right


def _value_matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "null":
        return value is None
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and math.isfinite(value) and value.is_integer()
        )
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, Mapping)
    return True


def _validate_model_argument_against_schema(value: Any, schema: Any, *, path: str) -> None:
    if not isinstance(schema, Mapping):
        return
    if "const" in schema and not _json_type_equal(value, schema["const"]):
        raise AgentRunError(f"model brokered tool argument {path} does not match const", status=400, code="InvalidToolArguments")
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and not any(_json_type_equal(value, enum_value) for enum_value in enum_values):
        raise AgentRunError(f"model brokered tool argument {path} is not in enum", status=400, code="InvalidToolArguments")
    types = _schema_types(schema)
    if types and not any(_value_matches_type(value, schema_type) for schema_type in types):
        raise AgentRunError(f"model brokered tool argument {path} has wrong type", status=400, code="InvalidToolArguments")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            raise AgentRunError(f"model brokered tool argument {path} is too short", status=400, code="InvalidToolArguments")
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and len(value) > max_length:
            raise AgentRunError(f"model brokered tool argument {path} is too long", status=400, code="InvalidToolArguments")
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                matches = re.search(pattern, value) is not None
            except re.error as exc:
                raise AgentRunError(
                    f"brokered tool schema pattern for {path} is not supported by the model-loop validator",
                    status=400,
                    code="UnsupportedBrokeredSchema",
                ) from exc
            if not matches:
                raise AgentRunError(f"model brokered tool argument {path} does not match pattern", status=400, code="InvalidToolArguments")

    if isinstance(value, int) and not isinstance(value, bool) or isinstance(value, float):
        number = _decimal_json_number(value)
        for key, compare in (("minimum", lambda a, b: a >= b), ("exclusiveMinimum", lambda a, b: a > b), ("maximum", lambda a, b: a <= b), ("exclusiveMaximum", lambda a, b: a < b)):
            bound = schema.get(key)
            if isinstance(bound, (int, float)) and not isinstance(bound, bool) and not compare(number, _decimal_json_number(bound)):
                raise AgentRunError(f"model brokered tool argument {path} violates {key}", status=400, code="InvalidToolArguments")
        multiple_of = schema.get("multipleOf")
        if isinstance(multiple_of, (int, float)) and not isinstance(multiple_of, bool) and multiple_of > 0:
            divisor = _decimal_json_number(multiple_of)
            if divisor == 0 or number % divisor != 0:
                raise AgentRunError(f"model brokered tool argument {path} violates multipleOf", status=400, code="InvalidToolArguments")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise AgentRunError(f"model brokered tool argument {path} has too few items", status=400, code="InvalidToolArguments")
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            raise AgentRunError(f"model brokered tool argument {path} has too many items", status=400, code="InvalidToolArguments")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for idx, item in enumerate(value):
                _validate_model_argument_against_schema(item, item_schema, path=f"{path}[{idx}]")

    if isinstance(value, Mapping):
        required = schema.get("required")
        if isinstance(required, list):
            for name in required:
                if isinstance(name, str) and name not in value:
                    raise AgentRunError(f"model brokered tool argument {path}.{name} is required", status=400, code="InvalidToolArguments")
        dependent_required = schema.get("dependentRequired")
        if isinstance(dependent_required, Mapping):
            for trigger, dependent_names in dependent_required.items():
                if trigger not in value or not isinstance(dependent_names, list):
                    continue
                for dependent_name in dependent_names:
                    if isinstance(dependent_name, str) and dependent_name not in value:
                        raise AgentRunError(f"model brokered tool argument {path}.{dependent_name} is required", status=400, code="InvalidToolArguments")
        min_properties = schema.get("minProperties")
        if isinstance(min_properties, int) and len(value) < min_properties:
            raise AgentRunError(f"model brokered tool argument {path} has too few properties", status=400, code="InvalidToolArguments")
        max_properties = schema.get("maxProperties")
        if isinstance(max_properties, int) and len(value) > max_properties:
            raise AgentRunError(f"model brokered tool argument {path} has too many properties", status=400, code="InvalidToolArguments")
        properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        additional = schema.get("additionalProperties")
        if additional is False:
            for key in value:
                if key not in properties:
                    raise AgentRunError(f"model brokered tool argument {path}.{key} is not declared", status=400, code="InvalidToolArguments")
        elif isinstance(additional, Mapping):
            for key, child in value.items():
                if key not in properties:
                    _validate_model_argument_against_schema(child, additional, path=f"{path}.{key}")
        for key, child_schema in properties.items():
            if isinstance(key, str) and key in value:
                _validate_model_argument_against_schema(value[key], child_schema, path=f"{path}.{key}")


def _validate_model_arguments_for_tool(arguments: Mapping[str, Any], tool: BrokeredToolDefinition) -> None:
    _validate_model_argument_against_schema(dict(arguments), tool.parameters, path="arguments")


def _function_call_response_payload(
    spec: AgentSpec,
    *,
    response_id: str,
    call: _PendingCall,
    previous_response_id: str | None = None,
    usage: Mapping[str, int] | None = None,
    created_at: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()) if created_at is None else created_at,
        "status": "completed",
        "model": spec.model.name,
        "output": [
            {
                "id": call.item_id,
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.tool.name,
                "arguments": _canonical_output_json(call.arguments),
                "status": "completed",
                "response_id": response_id,
            }
        ],
        "usage": _responses_usage(usage=usage),
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    return payload


def _model_pending_call(
    result: ModelLoopToolRequest,
    *,
    model_loop: BrokeredChatModelLoop,
    response_id: str,
    call_id: str,
    max_argument_bytes: int,
    max_messages_bytes: int,
) -> _PendingCall:
    tool = model_loop.tools_by_name.get(result.name)
    if tool is None:
        raise AgentRunError("model requested unknown brokered tool", status=400, code="unknown_brokered_tool")
    _validate_model_brokered_arguments(result.arguments)
    _validate_model_arguments_for_tool(result.arguments, tool)
    if len(_canonical_output_json(result.arguments).encode("utf-8")) > max_argument_bytes:
        raise AgentRunError("brokered function_call arguments are too large for pending state", status=413, code="brokered_arguments_too_large")
    try:
        _bounded_json_bytes(result.messages, max_bytes=max_messages_bytes)
    except _SerializedPayloadTooLarge as exc:
        raise AgentRunError("model loop messages are too large for pending state", status=413, code="brokered_model_messages_too_large") from exc
    except (_InvalidUnicodeValue, TypeError, ValueError) as exc:
        raise AgentRunError("model loop messages are invalid", status=502, code="InvalidModelResponse") from exc
    return _PendingCall(call_id=call_id, item_id=_new_function_call_id(response_id), tool=tool, arguments=result.arguments)


def _bounded_conversation_history(messages: list[dict[str, Any]], *, max_bytes: int) -> list[dict[str, str]]:
    """Retain complete recent exchanges, without system prompts or tool data."""
    turns = [
        {
            "role": message["role"], "content": message["content"],
            **({"phase": message["phase"]} if message.get("phase") is not None else {}),
        }
        for message in messages
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str) and message["content"]
        and not message.get("tool_calls")
    ]
    while turns:
        try:
            _bounded_json_bytes(turns, max_bytes=max_bytes)
            return turns
        except _SerializedPayloadTooLarge:
            # Drop an entire exchange rather than leave an orphan answer.
            next_user = next((i for i in range(1, len(turns)) if turns[i]["role"] == "user"), len(turns))
            turns = turns[next_user:]
    return []


def _advance_brokered_state(
    *,
    spec: AgentSpec,
    store: _FoundryResponseStateStore,
    state: _HostedResponseState,
    previous_response_id: str,
    call_id: str,
    result: ModelLoopToolRequest,
    model_loop: BrokeredChatModelLoop,
    response_id: str,
    next_call_id: str,
    created_at: int | None = None,
) -> JSONResponse:
    try:
        call = _model_pending_call(
            result,
            model_loop=model_loop,
            response_id=response_id,
            call_id=next_call_id,
            max_argument_bytes=model_loop.max_argument_bytes,
            max_messages_bytes=model_loop.max_messages_bytes,
        )
    except AgentRunError as exc:
        if not _reset_unfinalized_continuation(store, state, call_id=call_id):
            return _state_storage_error()
        return _brokered_model_run_error(exc)
    payload = _function_call_response_payload(spec, response_id=response_id, call=call, previous_response_id=previous_response_id, usage=result.usage, created_at=created_at)
    following = deepcopy(state)
    if not following.response_calls:
        following.response_calls[state.response_id] = call_id
    following.response_calls[response_id] = next_call_id
    following.pending_calls[next_call_id] = call
    following.continuation_payloads[call_id] = payload
    following.model_messages = result.messages
    following.initial_usage = _combine_usage(state.initial_usage, result.usage)
    following.status = "pending"
    following.expires_at = time.time() + store.ttl_seconds
    try:
        store.save(following)
    except (_StateStoreFull, _StateSizeLimitExceeded) as exc:
        if not _reset_unfinalized_continuation(store, state, call_id=call_id):
            return _state_storage_error()
        return _state_full_error() if isinstance(exc, _StateStoreFull) else _model_response_too_large_error()
    except _StatePersistenceError:
        following.final_persistence_pending = True
        store.cache_in_memory(following)
        return _state_storage_error()
    return JSONResponse(payload)


def _final_text_from_tool_output(call: _PendingCall, output: dict[str, Any]) -> str:
    if not output.get("approved"):
        error = output.get("error") if isinstance(output.get("error"), dict) else {}
        details = orka_tool_error_details(error)
        if details is not None:
            code, message = details
            return f"Brokered tool {call.tool.name}: {code}: {message}"
        code = str(error.get("code") or "brokered_tool_denied")
        message = str(error.get("message") or "brokered tool was not performed")
        return f"Brokered tool {call.tool.name} was not performed: {code}: {message}"
    tool_output = output.get("output") if isinstance(output.get("output"), dict) else {}
    return f"Brokered tool {call.tool.name} completed with output: {_canonical_output_json(tool_output)}"


def _reset_unfinalized_continuation(
    store: _FoundryResponseStateStore,
    state: _HostedResponseState,
    *,
    call_id: str,
) -> bool:
    state.accepted_output_digests.pop(call_id, None)
    state.accepted_output_sizes.pop(call_id, None)
    state.status = "pending"
    state.final_payload = None
    state.terminal_error = None
    state.final_persistence_pending = False
    state.conversation_history = []
    state.expires_at = time.time() + store.ttl_seconds
    try:
        store.save(state)
    except _StatePersistenceError as exc:
        logger.warning("failed to persist Foundry continuation rollback: %s", exc)
        store.cache_in_memory(state)
        return False
    return True


def _complete_with_state_full(
    store: _FoundryResponseStateStore,
    state: _HostedResponseState,
    *,
    call_id: str,
    resume_model_messages: list[dict[str, Any]] | None,
    resume_initial_usage: Mapping[str, int],
) -> JSONResponse:
    state.status = "completed"
    state.final_payload = None
    state.terminal_error = _TERMINAL_STATE_FULL
    state.model_messages = None
    state.initial_usage = {}
    state.conversation_history = []
    state.final_persistence_pending = False
    state.expires_at = time.time() + store.ttl_seconds
    try:
        store.save(state)
    except (_StateSizeLimitExceeded, _StateStoreFull):
        state.model_messages = resume_model_messages
        state.initial_usage = dict(resume_initial_usage)
        if not _reset_unfinalized_continuation(store, state, call_id=call_id):
            return _state_storage_error()
        return _state_full_error()
    except _StatePersistenceError as exc:
        logger.warning("failed to persist terminal Foundry brokered capacity state: %s", exc)
        state.final_persistence_pending = True
        try:
            store.cache_in_memory(state)
        except (_StateSizeLimitExceeded, _StateStoreFull):
            return _state_storage_error()
        return _state_storage_error()
    return _state_full_error()


async def _handle_brokered_continuation(
    *,
    spec: AgentSpec,
    store: _FoundryResponseStateStore,
    previous_response_id: str | None,
    input_value: Any,
    continuation_proof: str | None,
    request: Request,
    request_data: Mapping[str, Any],
    session_id: str | None,
    max_output_bytes: int,
    model_loop: BrokeredChatModelLoop | None = None,
    stream: BrokeredResponseStream | None = None,
) -> JSONResponse:
    if not continuation_proof:
        return _error(
            "brokered continuation proof is not configured; refusing function_call_output",
            status=503,
            code="brokered_continuation_auth_required",
        )
    if not _continuation_proof_matches(
        configured=continuation_proof,
        request=request,
        data=request_data,
    ):
        return _error(
            "function_call_output is restricted to the Orka broker continuation path",
            status=403,
            code="brokered_continuation_forbidden",
        )
    outputs = _function_call_outputs_from_input(input_value)
    if not outputs:
        return _error(
            "Responses continuation requires a function_call_output input item",
            status=400,
            code="missing_function_call_output",
        )
    if previous_response_id is None or not str(previous_response_id).strip():
        return _error(
            "function_call_output requires previous_response_id",
            status=400,
            code="missing_previous_response_id",
        )
    input_items = input_value if isinstance(input_value, list) else [input_value] if isinstance(input_value, dict) else []
    if len(outputs) != 1:
        return _error(
            "multiple function_call_output items are not supported by this deterministic brokered adapter",
            status=400,
            code="multiple_tool_outputs_unsupported",
        )
    if len(input_items) != 1:
        return _error(
            "continuation input must contain exactly one function_call_output item",
            status=400,
            code="multiple_tool_outputs_unsupported",
        )
    try:
        state = store.get(str(previous_response_id))
    except _StatePersistenceError as exc:
        logger.warning("failed to access Foundry brokered response state: %s", exc)
        return _state_storage_error()
    except _StateExpired:
        return _error("previous_response_id state has expired", status=410, code="response_state_expired")
    except KeyError:
        return _error("unknown previous_response_id", status=404, code="unknown_previous_response_id")

    stored_session_id = state.session_id.strip() if isinstance(state.session_id, str) else ""
    if stored_session_id and session_id != stored_session_id:
        return _error(
            "previous_response_id requires the same effective Foundry session",
            status=409,
            code="response_session_mismatch",
        )

    item = outputs[0]
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        return _error("function_call_output.call_id is required", status=400, code="missing_call_id")
    if state.response_calls and state.response_calls.get(str(previous_response_id)) != call_id:
        return _error("function_call_output does not match previous_response_id", status=400, code="unknown_call_id")
    if not state.response_calls and previous_response_id != state.response_id:
        return _error("function_call_output does not match previous_response_id", status=400, code="unknown_call_id")
    call = state.pending_calls.get(call_id)
    if call is None:
        return _error("unknown function_call_output call_id", status=400, code="unknown_call_id")
    raw_output = item.get("output")
    if not isinstance(raw_output, str):
        return _error(
            "function_call_output.output must be a JSON object string",
            status=400,
            code="invalid_function_call_output",
        )
    output_exceeds_current_limit = False
    existing_output_digest = state.accepted_output_digests.get(call_id)
    persisted_output_size = state.accepted_output_sizes.get(call_id, 0)
    replay_parse_limit = (
        max_output_bytes
        if existing_output_digest is None
        else max(store.max_bytes, max_output_bytes, persisted_output_size)
    )
    raw_output_size = 0
    try:
        if _utf8_exceeds_limit(raw_output, replay_parse_limit):
            return _error(
                    "brokered function_call_output is too large",
                    status=413,
                    code="brokered_output_too_large",
                )
        raw_output_size = len(raw_output.encode("utf-8"))
        output_exceeds_current_limit = raw_output_size > max_output_bytes
    except _InvalidUnicodeValue as exc:
        return _error(str(exc), status=400, code="invalid_function_call_output")
    try:
        parsed_output = _json_object_from_output(raw_output)
    except ValueError as exc:
        return _error(str(exc), status=400, code="invalid_function_call_output")
    try:
        output_bytes = _bounded_json_bytes(parsed_output, max_bytes=replay_parse_limit)
    except _SerializedPayloadTooLarge:
        return _error(
            "brokered function_call_output is too large",
            status=413,
            code="brokered_output_too_large",
        )
    except _InvalidUnicodeValue as exc:
        return _error(str(exc), status=400, code="invalid_function_call_output")
    if len(output_bytes) > max_output_bytes:
        output_exceeds_current_limit = True
    accepted_output_size = max(raw_output_size, len(output_bytes))
    output_json = output_bytes.decode("utf-8")
    output_digest = _output_digest(output_json)

    if existing_output_digest is not None:
        replay_payload = state.continuation_payloads.get(call_id, state.final_payload)
        if existing_output_digest == output_digest and (replay_payload is not None or state.terminal_error is not None):
            if state.final_persistence_pending:
                state.final_persistence_pending = False
                try:
                    store.save(state)
                except _StateStoreFull:
                    state.final_persistence_pending = True
                    return _state_full_error()
                except _StateSizeLimitExceeded:
                    state.final_persistence_pending = True
                    return _state_too_large_error()
                except _StatePersistenceError as exc:
                    logger.warning("failed to persist cached Foundry brokered completion: %s", exc)
                    state.final_persistence_pending = True
                    store.cache_in_memory(state)
                    return _state_storage_error()
            # A later workflow failure must not replace an earlier round's reply.
            if replay_payload is None and state.terminal_error == _TERMINAL_STATE_FULL:
                return _state_full_error()
            assert replay_payload is not None
            return JSONResponse(replay_payload)
        if existing_output_digest == output_digest:
            return _error(
                "matching function_call_output is already being processed",
                status=409,
                code="duplicate_continuation_in_progress",
            )
        return _error(
            "conflicting duplicate function_call_output for call_id",
            status=409,
            code="conflicting_duplicate_continuation",
        )
    if output_exceeds_current_limit:
        return _error(
            "brokered function_call_output is too large",
            status=413,
            code="brokered_output_too_large",
        )
    if state.model_messages is not None and model_loop is None:
        return _error(
            "brokered model-loop continuation is unavailable for this pending response",
            status=503,
            code="brokered_model_loop_unavailable",
        )
    if state.model_messages is not None and model_loop is not None:
        if (state.model_api or "chat_completions") != model_loop.model_api:
            return _error(
                "pending response requires the model API it started with",
                status=409, code="brokered_model_api_mismatch",
            )
    if state.status != "pending":
        return _error("previous response is not pending a tool result", status=409, code="response_not_pending")

    created_at = None
    if state.model_messages is not None and model_loop is not None:
        try:
            model_loop.validate_static_credentials()
        except AgentRunError as exc:
            return _brokered_model_run_error(exc)
        state.model_api = model_loop.model_api
        state.accepted_output_digests[call_id] = output_digest
        state.accepted_output_sizes[call_id] = accepted_output_size
        state.status = "resuming"
        state.expires_at = time.time() + store.ttl_seconds
        try:
            store.save(state)
        except _StateStoreFull:
            return _state_full_error()
        except _StateSizeLimitExceeded:
            return _state_too_large_error()
        except _StatePersistenceError as exc:
            logger.warning("failed to persist Foundry brokered continuation state: %s", exc)
            return _state_storage_error()
        store.mark_resume_active(state.response_id)
        try:
            try:
                next_response_id = _new_response_id(str(previous_response_id))
                next_call_id = f"call_{next_response_id}_1"
                if stream is not None:
                    created_at = await stream.accept(next_response_id)
                model_result = await model_loop.resume(state.model_messages, call_id=call_id, output=output_json, next_call_id=next_call_id)
            except AgentRunError as exc:
                if not _reset_unfinalized_continuation(store, state, call_id=call_id):
                    return _state_storage_error()
                logger.warning("brokered model-loop resume failed")
                return _brokered_model_run_error(exc)
            except asyncio.CancelledError:
                _reset_unfinalized_continuation(store, state, call_id=call_id)
                raise
            except Exception as exc:  # noqa: BLE001 - reset continuation state before surfacing unexpected model failures.
                logger.warning("brokered model-loop resume failed unexpectedly")
                if not _reset_unfinalized_continuation(store, state, call_id=call_id):
                    return _state_storage_error()
                return _brokered_model_run_error(exc)
            if isinstance(model_result, ModelLoopToolRequest):
                return _advance_brokered_state(
                    spec=spec,
                    store=store,
                    state=state,
                    previous_response_id=str(previous_response_id),
                    call_id=call_id,
                    result=model_result,
                    model_loop=model_loop,
                    response_id=next_response_id,
                    next_call_id=next_call_id,
                    created_at=created_at,
                )
            result = RunResult(text=model_result.text, usage=_combine_usage(state.initial_usage, model_result.usage))
        finally:
            store.mark_resume_inactive(state.response_id)
    else:
        result = RunResult(text=_final_text_from_tool_output(call, parsed_output))
    resume_model_messages = state.model_messages
    resume_initial_usage = dict(state.initial_usage)
    has_resume_transcript = resume_model_messages is not None
    used_model_resume = has_resume_transcript and model_loop is not None
    final_payload = _responses_payload(
        spec,
        result,
        previous_response_id=str(previous_response_id),
        response_id=stream.created.result()["id"] if stream is not None and stream.created.done() else None,
        created_at=created_at,
    )
    state.accepted_output_digests[call_id] = output_digest
    state.accepted_output_sizes[call_id] = accepted_output_size
    state.status = "completed"
    state.final_payload = final_payload
    state.terminal_error = None
    if has_resume_transcript:
        state.model_messages = None
        state.initial_usage = {}
        if state.session_id and isinstance(model_result, ModelLoopFinal):
            state.conversation_history = _bounded_conversation_history(
                model_result.messages, max_bytes=min(model_loop.max_messages_bytes // 2, store.max_bytes // 8),
            )
    state.final_persistence_pending = False
    state.expires_at = time.time() + store.ttl_seconds
    try:
        store.save(state)
    except _StateStoreFull:
        return _complete_with_state_full(
            store,
            state,
            call_id=call_id,
            resume_model_messages=resume_model_messages,
            resume_initial_usage=resume_initial_usage,
        )
    except _StateSizeLimitExceeded:
        if has_resume_transcript:
            state.model_messages = resume_model_messages
            state.initial_usage = resume_initial_usage
            if not _reset_unfinalized_continuation(store, state, call_id=call_id):
                return _state_storage_error()
        if used_model_resume:
            return _model_response_too_large_error()
        return _state_too_large_error()
    except _StatePersistenceError as exc:
        logger.warning("failed to persist completed Foundry brokered response state: %s", exc)
        state.final_persistence_pending = True
        store.cache_in_memory(state)
        return _state_storage_error()
    try:
        store.evict_completed_to_capacity()
    except _StatePersistenceError as exc:
        logger.warning("failed to evict completed Foundry brokered response state: %s", exc)
        return _state_storage_error()
    return JSONResponse(final_payload)


def create_foundry_app(
    spec: AgentSpec,
    factory: RuntimeFactory,
    auth_token: str | None = None,
    *,
    state_ttl_seconds: float | None = None,
    brokered_continuation_proof: str | None = None,
    max_pending_responses: int | None = None,
    max_brokered_argument_bytes: int | None = None,
    max_brokered_output_bytes: int | None = None,
    max_response_state_bytes: int | None = None,
    max_request_body_bytes: int | None = None,
    max_model_messages_bytes: int | None = None,
    brokered_model_loop_enabled: bool | None = None,
    brokered_model_http_client: Any | None = None,
    response_state_file: str | Path | None = None,
) -> FastAPI:
    """Create a Foundry-compatible wrapper app for one AgentKit runtime."""
    brokered_tools = brokered_tool_definitions(spec)
    runtime = None if brokered_tools else factory.build_runtime(spec)
    health = RuntimeHealth()
    continuation_proof = brokered_continuation_proof or os.environ.get(_CONTINUATION_PROOF_ENV) or None
    max_argument_bytes = _max_argument_bytes(max_brokered_argument_bytes)
    max_output_bytes = _max_output_bytes(max_brokered_output_bytes)
    request_body_limit = _max_request_body_bytes(max_request_body_bytes)
    model_messages_limit = _max_model_messages_bytes(max_model_messages_bytes)
    response_states = _FoundryResponseStateStore(
        ttl_seconds=_state_ttl_seconds(state_ttl_seconds),
        max_entries=_max_pending_responses(max_pending_responses),
        max_bytes=_max_response_state_bytes(max_response_state_bytes),
        state_file=_response_state_file(response_state_file) if brokered_tools else None,
    )

    def max_brokered_request_body_bytes() -> int:
        replay_ceiling = max(
            response_states.max_bytes,
            max_output_bytes,
            response_states.max_accepted_output_bytes(),
        )
        return max(6 * replay_ceiling, max_argument_bytes) + _REQUEST_BODY_OVERHEAD_BYTES

    model_loop = (
        BrokeredChatModelLoop(
            spec,
            brokered_tools,
            http_client=brokered_model_http_client,
            max_argument_bytes=max_argument_bytes,
            max_output_bytes=max_output_bytes,
            max_response_bytes=response_states.max_bytes,
            max_messages_bytes=model_messages_limit,
        )
        if brokered_tools and _brokered_model_loop_enabled(brokered_model_loop_enabled)
        else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if runtime is None:
            yield
            return
        async with runtime:
            app.state.runtime = runtime
            yield

    app = FastAPI(lifespan=lifespan)
    auth = Depends(make_auth_dependency(auth_token))

    @app.get("/readiness")
    async def readiness():
        body: dict[str, Any] = {"ready": True}
        if not health.healthy:
            return JSONResponse({"ready": False}, status_code=503)
        if brokered_tools:
            body["foundryResponses"] = {
                "brokeredTools": len(brokered_tools),
                "ownedToolsDisabled": len(spec.tools),
                "stateBackend": response_states.backend_name,
                "stateTtlSeconds": response_states.ttl_seconds,
                "stateMaxPending": response_states.max_entries,
                "continuationAuth": "configured" if continuation_proof else "missing",
                "runtime": "model-loop" if model_loop is not None else "deterministic",
                "scaling": "single-replica-or-sticky-routing-required",
            }
            if not continuation_proof:
                body["ready"] = False
            if model_loop is not None:
                try:
                    await model_loop.validate_credentials()
                except AgentRunError:
                    body["ready"] = False
                    body["foundryResponses"]["modelAuth"] = "missing"
            if not body["ready"]:
                return JSONResponse(body, status_code=503)
        return body

    @app.post("/invocations", dependencies=[auth])
    async def invocations(request: Request):
        if brokered_tools:
            return _error(
                "Foundry /invocations is disabled when brokeredTools are configured; use /responses so Orka can broker tools",
                status=400,
                code="invocations_disabled_in_brokered_mode",
            )
        try:
            data, _ = await _read_json_request_bounded(request, max_bytes=request_body_limit)
        except _RequestBodyTooLarge:
            return _request_too_large_error()
        except (UnicodeDecodeError, RecursionError, ValueError):
            return Response("Request body must be JSON", status_code=400)

        if not isinstance(data, dict):
            return Response("Request body must be a JSON object", status_code=400)
        if "message" not in data:
            return Response("Missing 'message' in request", status_code=400)
        prompt = _message_to_prompt(data["message"])

        try:
            result = await request.app.state.runtime.run(
                RunRequest(prompt=prompt, session_id=_session_id_from_request(request))
            )
        except AgentRunError as exc:
            health.record(exc)
            return _non_brokered_agent_run_error(exc)
        except Exception:  # noqa: BLE001 - deterministic protocol envelope.
            return _non_brokered_unexpected_runtime_error()

        return JSONResponse({"response": result.text, "usage": _usage(result)})

    @app.post("/responses", dependencies=[auth])
    async def responses(request: Request):
        try:
            data, request_body_size = await _read_json_request_bounded(
                request,
                max_bytes=max_brokered_request_body_bytes() if brokered_tools else request_body_limit,
            )
        except _RequestBodyTooLarge:
            if not brokered_tools:
                return _request_too_large_error()
            return _error(
                "brokered Responses request body is too large",
                status=413,
                code="brokered_request_too_large",
            )
        except (UnicodeDecodeError, RecursionError, ValueError):
            return _error("Request body must be JSON", status=400, code="invalid_json")

        if isinstance(data, dict) and data.get("stream") is True:
            return await brokered_stream_response(
                spec.model.name,
                lambda stream: execute_responses(request, data, request_body_size, stream),
            )
        return await execute_responses(request, data, request_body_size)

    async def execute_responses(
        request: Request,
        data: Any,
        request_body_size: int,
        stream: BrokeredResponseStream | None = None,
    ) -> JSONResponse:
        if not isinstance(data, dict):
            return _error("Request body must be a JSON object", status=400, code="invalid_request")
        if data.get("tools"):
            return _error(
                "request-supplied Responses tools are not allowed; hosted brokered mode uses static safe schemas",
                status=400,
                code="tools_unsupported",
            )
        tool_choice = data.get("tool_choice")
        if tool_choice not in (None, "", "none", "auto") or (brokered_tools and tool_choice == "none"):
            return _error(
                "request-supplied Responses tool_choice is not allowed; hosted brokered mode owns tool selection",
                status=400,
                code="tool_choice_unsupported",
            )
        if "input" not in data:
            return _error("Missing 'input' in request", status=400, code="missing_input")

        try:
            session_id = _effective_responses_session_id(
                request,
                data,
                enforce_trusted_precedence=bool(brokered_tools),
            )
        except _SessionIdentityConflict:
            return _error(
                "request contains conflicting Foundry session identities",
                status=409,
                code="response_session_mismatch",
            )
        if stream is not None:
            stream.session_id = session_id
        previous_response_id = data.get("previous_response_id")
        function_outputs = _function_call_outputs_from_input(data["input"])
        if brokered_tools and not function_outputs and request_body_size > request_body_limit:
            return _error(
                "Request body is too large",
                status=413,
                code="request_body_too_large",
            )
        if brokered_tools and function_outputs:
            return await _handle_brokered_continuation(
                spec=spec,
                store=response_states,
                previous_response_id=previous_response_id if isinstance(previous_response_id, str) else None,
                input_value=data["input"],
                continuation_proof=continuation_proof,
                request=request,
                request_data=data,
                session_id=session_id,
                max_output_bytes=max_output_bytes,
                model_loop=model_loop,
                stream=stream,
            )
        previous_state = None
        if brokered_tools and isinstance(previous_response_id, str) and previous_response_id:
            try:
                previous_state = response_states.get(previous_response_id)
            except _StatePersistenceError as exc:
                logger.warning("failed to access Foundry brokered response state: %s", exc)
                return _state_storage_error()
            except _StateExpired as exc:
                if exc.state.status in {"pending", "resuming"} or (
                    model_loop is not None and (session_id or exc.state.session_id)
                ):
                    return _error("previous_response_id state has expired", status=410, code="response_state_expired")
                previous_state = None
            except KeyError:
                previous_state = None
            if previous_state is not None and previous_state.status in {"pending", "resuming"}:
                return _error(
                    "previous_response_id is pending a brokered function_call_output",
                    status=409,
                    code="response_pending_function_call_output",
                )
            if model_loop is not None and (session_id or (previous_state is not None and previous_state.session_id)):
                if previous_state is None:
                    return _error("unknown previous_response_id", status=404, code="unknown_previous_response_id")
                if previous_state.session_id != session_id:
                    return _error("previous_response_id requires the same effective Foundry session", status=409, code="response_session_mismatch")
                if not previous_state.final_payload or previous_state.final_payload.get("id") != previous_response_id:
                    return _error("follow-up requires the completed response ID", status=409, code="response_not_completed")
                if not previous_state.conversation_history:
                    return _error("previous conversation no longer fits retained history", status=409, code="response_history_unavailable")

        try:
            run_request = _responses_input_to_run_request(
                data["input"],
                session_id=session_id,
            )
        except ValueError as exc:
            return _error(str(exc), status=400, code="invalid_input")
        if model_loop is not None and session_id and previous_state is not None:
            run_request = replace(run_request, history=(
                *(ConversationTurn(role=turn["role"], text=turn["content"], phase=turn.get("phase"))
                  for turn in previous_state.conversation_history),
                *run_request.history,
            ))

        if brokered_tools:
            if not continuation_proof:
                return _error(
                    "brokered continuation proof is not configured; refusing to start an uncontinuable brokered response",
                    status=503,
                    code="brokered_continuation_auth_required",
                )
            previous_response_id_for_output = previous_response_id if isinstance(previous_response_id, str) and previous_response_id else None
            if model_loop is not None:
                try:
                    model_loop.validate_static_credentials()
                except AgentRunError as exc:
                    return _brokered_model_run_error(exc)
                response_id = _new_response_id(previous_response_id_for_output)
                call_id = f"call_{response_id}_1"
                try:
                    response_states.reserve(response_id)
                except _StateStoreFull:
                    return _error(
                        "too many pending brokered responses",
                        status=429,
                        code="brokered_response_state_full",
                    )
                except _StatePersistenceError as exc:
                    logger.warning("failed to persist Foundry brokered response state: %s", exc)
                    return _state_storage_error()
                try:
                    try:
                        created_at = await stream.accept(response_id) if stream is not None else None
                        model_result = await model_loop.start(run_request, call_id=call_id)
                    except AgentRunError as exc:
                        return _brokered_model_run_error(exc)
                    except Exception as exc:  # noqa: BLE001 - unknown model failures have no public diagnostic payload.
                        return _brokered_model_run_error(exc)
                    if isinstance(model_result, ModelLoopFinal):
                        payload = _responses_payload(
                            spec,
                            RunResult(text=model_result.text, usage=model_result.usage),
                            previous_response_id=previous_response_id_for_output,
                            response_id=response_id if stream is not None else None,
                            created_at=created_at,
                        )
                        if run_request.session_id:
                            completed = _HostedResponseState(
                                response_id=response_id,
                                session_id=run_request.session_id,
                                pending_calls={},
                                expires_at=time.time() + response_states.ttl_seconds,
                                status="completed",
                                final_payload=payload,
                                conversation_history=_bounded_conversation_history(
                                    model_result.messages, max_bytes=min(model_messages_limit // 2, response_states.max_bytes // 8),
                                ),
                            )
                            try:
                                response_states.add_reserved(completed)
                            except _StateStoreFull:
                                return _state_full_error()
                            except _StateSizeLimitExceeded:
                                return _model_response_too_large_error()
                            except _StatePersistenceError:
                                return _state_storage_error()
                        return JSONResponse(payload)
                    tool = {tool.name: tool for tool in brokered_tools}.get(model_result.name)
                    if tool is None:
                        return _error("model requested unknown brokered tool", status=400, code="unknown_brokered_tool")
                    try:
                        _validate_model_brokered_arguments(model_result.arguments)
                        _validate_model_arguments_for_tool(model_result.arguments, tool)
                    except AgentRunError as exc:
                        return _brokered_model_run_error(exc)
                    if len(_canonical_output_json(model_result.arguments).encode("utf-8")) > max_argument_bytes:
                        return _error(
                            "brokered function_call arguments are too large for pending state",
                            status=413,
                            code="brokered_arguments_too_large",
                        )
                    try:
                        _bounded_json_bytes(model_result.messages, max_bytes=model_messages_limit)
                    except _SerializedPayloadTooLarge:
                        return _error(
                            "model loop messages are too large for pending state",
                            status=413,
                            code="brokered_model_messages_too_large",
                        )
                    except (_InvalidUnicodeValue, TypeError, ValueError) as exc:
                        return _error(
                            f"model loop messages are invalid: {exc}",
                            status=502,
                            code="InvalidModelResponse",
                        )
                    call = _PendingCall(
                        call_id=call_id,
                        item_id=_new_function_call_id(response_id),
                        tool=tool,
                        arguments=model_result.arguments,
                    )
                    state = _HostedResponseState(
                        response_id=response_id,
                        session_id=run_request.session_id,
                        pending_calls={call_id: call},
                        expires_at=time.time() + response_states.ttl_seconds,
                        model_messages=model_result.messages,
                        model_api=model_loop.model_api,
                        initial_usage=dict(model_result.usage),
                    )
                    try:
                        response_states.add_reserved(state)
                    except _StateStoreFull:
                        return _error(
                            "too many pending brokered responses",
                            status=429,
                            code="brokered_response_state_full",
                        )
                    except _StateSizeLimitExceeded:
                        return _state_too_large_error()
                    except _StatePersistenceError as exc:
                        logger.warning("failed to persist Foundry brokered response state: %s", exc)
                        return _state_storage_error()
                    return JSONResponse(
                        _function_call_response_payload(
                            spec,
                            response_id=response_id,
                            call=call,
                            previous_response_id=previous_response_id_for_output,
                            usage=model_result.usage,
                            created_at=created_at,
                        )
                    )
                finally:
                    response_states.release_reservation(response_id)
            tool = _select_brokered_tool(brokered_tools, run_request)
            if tool is None:
                return _error(
                    "prompt must name exactly one configured brokered tool in deterministic brokered mode",
                    status=400,
                    code="brokered_tool_selection_required",
                )
            response_id = _new_response_id(previous_response_id_for_output)
            call_id = f"call_{response_id}_1"
            try:
                arguments = _deterministic_tool_arguments(tool, run_request)
                _validate_model_brokered_arguments(arguments)
                _validate_model_arguments_for_tool(arguments, tool)
            except AgentRunError as exc:
                return _error(str(exc), status=exc.status, code=exc.code)
            if len(_canonical_output_json(arguments).encode("utf-8")) > max_argument_bytes:
                return _error(
                    "brokered function_call arguments are too large for pending state",
                    status=413,
                    code="brokered_arguments_too_large",
                )
            call = _PendingCall(
                call_id=call_id,
                item_id=_new_function_call_id(response_id),
                tool=tool,
                arguments=arguments,
            )
            state = _HostedResponseState(
                response_id=response_id,
                session_id=run_request.session_id,
                pending_calls={call_id: call},
                expires_at=time.time() + response_states.ttl_seconds,
            )
            try:
                response_states.add(state)
            except _StateStoreFull:
                return _error(
                    "too many pending brokered responses",
                    status=429,
                    code="brokered_response_state_full",
                )
            except _StateSizeLimitExceeded:
                return _state_too_large_error()
            except _StatePersistenceError as exc:
                logger.warning("failed to persist Foundry brokered response state: %s", exc)
                return _state_storage_error()
            return JSONResponse(
                _function_call_response_payload(
                    spec,
                    response_id=response_id,
                    call=call,
                    previous_response_id=previous_response_id_for_output,
                )
            )

        response_id = _new_response_id() if stream is not None else None
        try:
            created_at = await stream.accept(response_id) if stream is not None else None
            result = await request.app.state.runtime.run(run_request)
        except AgentRunError as exc:
            health.record(exc)
            return _non_brokered_agent_run_error(exc)
        except Exception:  # noqa: BLE001 - deterministic protocol envelope.
            return _non_brokered_unexpected_runtime_error()

        return JSONResponse(_responses_payload(spec, result, response_id=response_id, created_at=created_at))

    return app
