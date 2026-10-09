"""Lower-level OpenAI-compatible brokered tool loop for Foundry Responses mode.

This is the Phase A4 fallback path: when high-level frameworks cannot suspend and
resume externally brokered tool calls, AgentKit can drive a minimal model loop
itself. The loop exposes only static safe brokered schemas to the model, converts
one model tool request at a time into a hosted Responses function_call, and
resumes with Orka's function_call_output until the assistant finishes.

The model is called through the OpenAI Responses API because reasoning models
can reject function tools on Chat Completions. The retained transcript keeps a
chat-shaped message list so pending continuation state stays compatible; it is
translated to Responses input items for each model request. Tool-call turns also
retain any encrypted reasoning items so the next stateless request can replay them.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from email.utils import mktime_tz, parsedate_tz
from typing import Any, Mapping, Sequence

import httpx

from .adapter_support import AgentBuildError, NO_AUTH_API_KEY, resolve_api_key, resolve_workload_identity_token
from .config import AgentSpec
from .conversation import FORWARDED_ROLES, RunRequest
from .runtime import AgentRunError, BrokeredToolDefinition
from .skills import SkillCatalog
from .tool_errors import orka_tool_error_details

_MAX_ARGUMENT_DEPTH = 128
_MAX_RATE_LIMIT_RETRIES = 2
_MAX_RETRY_AFTER_SECONDS = 60

_NORMALIZED_MODEL_ERRORS = {
    "ModelAuthMissing": (503, "model authentication is not configured"),
    "ModelAuthRejected": (503, "model service rejected configured credentials"),
    "ModelUnavailable": (503, "model service is unavailable"),
    "ModelUpstreamError": (502, "model service request failed"),
    "InvalidModelResponse": (502, "model service returned an invalid response"),
    "ModelResponseTooLarge": (502, "model response is too large to retain safely"),
}


class _ModelHTTPError(AgentRunError):
    def __init__(self, message: str, *, status: int, code: str, upstream_status: int) -> None:
        super().__init__(message, status=status, code=code)
        self.upstream_status = upstream_status


def normalized_model_error_details(exc: AgentRunError) -> tuple[int, dict[str, Any]] | None:
    """Project only runtime-owned model error definitions and bounded HTTP metadata."""
    definition = _NORMALIZED_MODEL_ERRORS.get(exc.code) if type(exc.code) is str else None
    if definition is None:
        return None
    status, message = definition
    error: dict[str, Any] = {"message": message, "code": exc.code}
    # A similarly named attribute on a framework exception is not HTTP evidence.
    if (
        isinstance(exc, _ModelHTTPError)
        and type(exc.upstream_status) is int
        and 400 <= exc.upstream_status <= 599
    ):
        error["upstream_status"] = exc.upstream_status
    return status, error


@dataclass(frozen=True)
class ModelLoopFinal:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ModelLoopToolRequest:
    name: str
    arguments: dict[str, Any]
    messages: list[dict[str, Any]]
    usage: dict[str, int] = field(default_factory=dict)


class BrokeredChatModelLoop:
    """Bounded sequential tool loop over the OpenAI Responses API."""

    def __init__(
        self,
        spec: AgentSpec,
        tools: Sequence[BrokeredToolDefinition],
        *,
        http_client: httpx.AsyncClient | None = None,
        max_argument_bytes: int = 8192,
        max_output_bytes: int = 64 * 1024,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_messages_bytes: int = 1024 * 1024,
        max_tool_calls: int = 16,
    ) -> None:
        self.spec = spec
        self.tools = list(tools)
        self.http_client = http_client
        self.max_argument_bytes = max_argument_bytes
        self.max_output_bytes = max_output_bytes
        self.max_response_bytes = max_response_bytes
        self.max_messages_bytes = max_messages_bytes
        self.max_tool_calls = max_tool_calls
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.skills = SkillCatalog.from_spec(spec)
        if self.skills and "load_skill" in self.tools_by_name:
            raise AgentBuildError("load_skill is reserved for packaged skills")

    async def start(self, request: RunRequest, *, call_id: str) -> ModelLoopFinal | ModelLoopToolRequest:
        return await self._advance(self._initial_messages(request), call_id=call_id)

    async def _advance(self, messages: list[dict[str, Any]], *, call_id: str) -> ModelLoopFinal | ModelLoopToolRequest:
        usage: dict[str, int] = {}
        while True:
            encoded_messages = json.dumps(messages, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            if len(encoded_messages) > self.max_messages_bytes:
                raise AgentRunError("model loop messages are too large", status=413, code="brokered_model_messages_too_large")
            result = await self._step(messages, call_id=call_id)
            for key, value in result.usage.items():
                usage[key] = usage.get(key, 0) + value
            if isinstance(result, ModelLoopFinal):
                return ModelLoopFinal(text=result.text, usage=usage, messages=result.messages)
            if not self.skills or result.name != "load_skill":
                return ModelLoopToolRequest(name=result.name, arguments=result.arguments, messages=result.messages, usage=usage)
            if set(result.arguments) != {"skill_name"} or not isinstance(result.arguments["skill_name"], str):
                raise AgentRunError("load_skill requires a skill_name string", status=400, code="InvalidToolArguments")
            try:
                content = self.skills.load_skill(result.arguments["skill_name"])
            except (KeyError, ValueError) as exc:
                raise AgentRunError("unknown packaged skill", status=400, code="InvalidToolArguments") from exc
            messages = result.messages
            skill_call_id = f"skill_{uuid.uuid4().hex}"
            tool_message = next(message for message in reversed(messages) if message.get("tool_calls"))
            tool_message["tool_calls"][0]["id"] = skill_call_id
            messages.append({"role": "tool", "tool_call_id": skill_call_id, "content": content})

    async def _step(self, messages: list[dict[str, Any]], *, call_id: str) -> ModelLoopFinal | ModelLoopToolRequest:
        current_turn = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), 0)
        tool_count = sum(len(message.get("tool_calls") or []) for message in messages[current_turn:])
        exhausted = tool_count >= self.max_tool_calls
        # At the limit, keep the definitions that match the replayed calls but forbid new calls.
        data = await self._create_response(messages, tools=self._tool_payloads(), tool_choice="none" if exhausted else "auto")
        message = _output_message(data)
        usage = _usage(data)
        assistant_messages = message["assistant_messages"]
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            text = _message_text(message, max_bytes=self.max_output_bytes)
            return ModelLoopFinal(text=text, usage=usage, messages=[*messages, *assistant_messages])
        if exhausted:
            raise AgentRunError("model exceeded the tool call limit", status=400, code="tool_loop_limit_exceeded")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise AgentRunError(
                "model requested multiple tools at once; brokered mode requires sequential calls",
                status=400,
                code="multiple_tool_calls_unsupported",
            )
        call = tool_calls[0]
        if not isinstance(call, Mapping) or call.get("type") != "function":
            raise AgentRunError("model returned an unsupported tool call", status=400, code="unsupported_tool_call")
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise AgentRunError("model tool call is missing function payload", status=400, code="invalid_tool_call")
        name = function.get("name")
        if not isinstance(name, str) or (name not in self.tools_by_name and not (self.skills and name == "load_skill")):
            raise AgentRunError(f"model requested unknown brokered tool {name!r}", status=400, code="unknown_brokered_tool")
        raw_arguments = function.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                if len(raw_arguments) > self.max_argument_bytes or len(raw_arguments.encode("utf-8")) > self.max_argument_bytes:
                    raise AgentRunError(
                        "model brokered tool arguments are too large",
                        status=413,
                        code="brokered_arguments_too_large",
                    )
            except UnicodeEncodeError as exc:
                raise AgentRunError("model tool arguments must contain valid Unicode", status=400, code="InvalidToolArguments") from exc
        arguments = _parse_arguments(raw_arguments)
        _validate_argument_unicode(arguments)
        argument_text = json.dumps(arguments, separators=(",", ":"), sort_keys=True)
        call["id"] = call_id
        function["arguments"] = argument_text
        return ModelLoopToolRequest(
            name=name, arguments=arguments, messages=[*messages, *assistant_messages], usage=usage,
        )

    async def resume(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        call_id: str,
        output: str,
        next_call_id: str | None = None,
    ) -> ModelLoopFinal | ModelLoopToolRequest:
        if len(output) > self.max_output_bytes:
            raise AgentRunError("brokered tool output is too large for model resume", status=413, code="brokered_output_too_large")
        try:
            output_bytes = output.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise AgentRunError(
                "brokered tool output must contain valid Unicode",
                status=400,
                code="InvalidToolOutput",
            ) from exc
        if len(output_bytes) > self.max_output_bytes:
            raise AgentRunError("brokered tool output is too large for model resume", status=413, code="brokered_output_too_large")
        try:
            payload = json.loads(output)
        except (ValueError, RecursionError):
            payload = None
        details = (
            orka_tool_error_details(payload.get("error"))
            if isinstance(payload, dict) and payload.get("approved") is False
            else None
        )
        if details is not None:
            code, message = details
            output = json.dumps({"approved": False, "error": {"code": code, "message": message}}, separators=(",", ":"))
            if len(output.encode("utf-8")) > self.max_output_bytes:
                raise AgentRunError("brokered tool output is too large for model resume", status=413, code="brokered_output_too_large")
        resumed = [dict(message) for message in messages]
        resumed.append({"role": "tool", "tool_call_id": call_id, "content": output})
        if details is not None and details[0] == "tool_outcome_unknown":
            # Another model round could repeat an action whose effect is unknown.
            # Finish this continuation and let Orka retain the execution evidence.
            text = f"{details[0]}: {details[1]}"
            return ModelLoopFinal(text=text, messages=[*resumed, {"role": "assistant", "content": text}])
        return await self._advance(resumed, call_id=next_call_id or f"call_{uuid.uuid4().hex}")

    async def validate_credentials(self) -> None:
        await self._auth_headers()

    def validate_static_credentials(self) -> None:
        if self.spec.model.auth is not None:
            return
        try:
            resolve_api_key(self.spec)
        except AgentBuildError as exc:
            raise _model_auth_missing_error() from exc

    def _initial_messages(self, request: RunRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.spec.instructions:
            messages.append({"role": "system", "content": self.spec.instructions})
        if self.skills:
            messages.append({"role": "system", "content": self.skills.instructions})
        for turn in request.history:
            if turn.role in FORWARDED_ROLES and turn.text:
                message = {"role": turn.role, "content": turn.text}
                if turn.role == "assistant" and turn.phase is not None:
                    message["phase"] = turn.phase
                messages.append(message)
        messages.append({"role": "user", "content": request.prompt})
        return messages

    def _tool_payloads(self) -> list[dict[str, Any]]:
        definitions = [self.skills.tool_schema()["function"]] if self.skills else []
        for tool in self.tools:
            description = f"Brokered class: {tool.brokered_class}. {tool.description}".strip()
            definitions.append({"name": tool.name, "description": description, "parameters": dict(tool.parameters)})
        # Brokered schemas need not satisfy strict mode; AgentKit validates arguments itself.
        return [{"type": "function", **definition, "strict": False} for definition in definitions]

    async def _create_response(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]],
        tool_choice: str,
    ) -> dict[str, Any]:
        # The full transcript is resent each round, so the provider need not retain it.
        payload: dict[str, Any] = {"model": self.spec.model.name, "input": _responses_input(messages), "store": False}
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = tool_choice
            payload["parallel_tool_calls"] = False
        headers = await self._auth_headers()
        client = self.http_client
        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=60)
            close_client = True
        try:
            for retry in range(_MAX_RATE_LIMIT_RETRIES + 1):
                retry_delay = None
                async with client.stream(
                    "POST",
                    _responses_url(self.spec.model.base_url),
                    json=payload,
                    headers=headers or None,
                ) as response:
                    # Only an explicit rate-limit rejection is safe to retry.
                    # Transport failures or accepted responses may have done work.
                    if response.status_code == 429 and retry < _MAX_RATE_LIMIT_RETRIES:
                        retry_delay = _rate_limit_retry_delay(response.headers, retry=retry)
                    if retry_delay is None:
                        response.raise_for_status()
                        response_body = await _read_response_body_bounded(
                            response,
                            max_bytes=self.max_response_bytes,
                        )
                        break
                # Release the HTTP response before waiting. Disconnect cancellation
                # must stop this wait before another model request can be submitted.
                await asyncio.sleep(retry_delay)
                headers = await self._auth_headers()
        except httpx.HTTPStatusError as exc:
            raise _normalized_model_http_error(exc.response.status_code) from exc
        except AgentRunError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize transport/model failures without leaking request URLs.
            raise AgentRunError(
                "model service request failed",
                status=502,
                code="ModelUpstreamError",
            ) from exc
        finally:
            if close_client:
                await client.aclose()
        try:
            data = json.loads(response_body)
        except Exception as exc:  # noqa: BLE001 - normalize decoder failures without exposing response internals.
            raise AgentRunError(
                "model service returned an invalid JSON response",
                status=502,
                code="InvalidModelResponse",
            ) from exc
        if not isinstance(data, dict):
            raise AgentRunError("model response must be a JSON object", status=502, code="InvalidModelResponse")
        _validate_model_response_unicode(data)
        return data

    async def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        try:
            auth = self.spec.model.auth
            if auth is not None and auth.type == "workload-identity-token":
                token = os.environ.get("AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN")
                if not token:
                    token = await asyncio.to_thread(resolve_workload_identity_token, auth.audience or "")
                headers["Authorization"] = f"Bearer {token}"
            else:
                api_key = resolve_api_key(self.spec)
                if api_key != NO_AUTH_API_KEY:
                    headers["Authorization"] = f"Bearer {api_key}"
        except AgentBuildError as exc:
            raise _model_auth_missing_error() from exc
        return headers


def _rate_limit_retry_delay(headers: httpx.Headers, *, retry: int) -> float | None:
    delay = _retry_after_seconds(headers)
    if delay is None:
        return (2.0 ** retry) * random.uniform(0.75, 1.0)
    # Never shorten a longer server delay and retry before the requested window.
    return delay if delay <= _MAX_RETRY_AFTER_SECONDS else None


def _retry_after_seconds(headers: httpx.Headers) -> float | None:
    for name, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            delay = float(value)
        except ValueError:
            continue
        if math.isfinite(delay) and delay >= 0:
            return delay / divisor
    value = headers.get("retry-after")
    if value is not None:
        try:
            date = parsedate_tz(value)
            if date is not None:
                return max(0.0, mktime_tz(date) - time.time())
        except (TypeError, ValueError, OverflowError):
            pass
    return None


async def _read_response_body_bounded(response: httpx.Response, *, max_bytes: int) -> bytearray:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            declared_length = -1
        if declared_length > max_bytes:
            raise _model_response_too_large_error()

    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > max_bytes:
            raise _model_response_too_large_error()
        body.extend(chunk)
    return body


def _model_response_too_large_error() -> AgentRunError:
    return AgentRunError(
        "model response is too large to retain safely",
        status=502,
        code="ModelResponseTooLarge",
    )


def _responses_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/responses"):
        return root
    # Keep base URLs that previously named the Chat Completions endpoint working.
    root = root.removesuffix("/chat/completions")
    return f"{root}/responses"


def _responses_input(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Translate the retained chat-shaped transcript into Responses input items."""
    items: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"]})
            continue
        items.extend(message.get("reasoning") or [])
        tool_calls = message.get("tool_calls") or []
        content = message.get("content")
        if isinstance(content, str) and (content or not tool_calls):
            item = {"type": "message", "role": message["role"], "content": content}
            if message["role"] == "assistant" and message.get("phase") is not None:
                item["phase"] = message["phase"]
            items.append(item)
        for call in tool_calls:
            function = call["function"]
            items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": function["name"],
                    "arguments": function["arguments"],
                }
            )
    return items


def _normalized_model_http_error(status_code: int) -> AgentRunError:
    if type(status_code) is not int or not 400 <= status_code <= 599:
        return AgentRunError(
            "model service request failed",
            status=502,
            code="ModelUpstreamError",
        )
    if status_code in {401, 403}:
        return _ModelHTTPError(
            "model service rejected configured credentials",
            status=503,
            code="ModelAuthRejected",
            upstream_status=status_code,
        )
    if status_code == 429 or status_code >= 500:
        return _ModelHTTPError(
            "model service is unavailable",
            status=503,
            code="ModelUnavailable",
            upstream_status=status_code,
        )
    return _ModelHTTPError(
        "model service request failed",
        status=502,
        code="ModelUpstreamError",
        upstream_status=status_code,
    )


def _validate_model_response_unicode(value: Any) -> None:
    pending = [iter((value,))]
    while pending:
        try:
            current = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        if isinstance(current, str):
            if any(0xD800 <= ord(char) <= 0xDFFF for char in current):
                raise AgentRunError(
                    "model service returned an invalid JSON response",
                    status=502,
                    code="InvalidModelResponse",
                )
        elif isinstance(current, Mapping):
            pending.append(iter(item for pair in current.items() for item in pair))
        elif isinstance(current, (list, tuple)):
            pending.append(iter(current))


def _model_auth_missing_error() -> AgentRunError:
    return AgentRunError(
        "model authentication is not configured",
        status=503,
        code="ModelAuthMissing",
    )


def _output_message(data: Mapping[str, Any]) -> dict[str, Any]:
    """Project Responses output into an ordered, chat-shaped retained transcript."""
    # Incomplete or failed generations can carry truncated text or tool arguments.
    if data.get("status") != "completed":
        raise AgentRunError("model response did not complete", status=502, code="InvalidModelResponse")
    output = data.get("output")
    if not isinstance(output, list):
        raise AgentRunError("model response did not include output items", status=502, code="InvalidModelResponse")
    texts: list[str] = []
    assistant_messages: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, Mapping):
            raise AgentRunError("model response output item must be an object", status=502, code="InvalidModelResponse")
        item_type = item.get("type")
        if item_type == "function_call":
            call = {"type": "function", "function": {"name": item.get("name"), "arguments": item.get("arguments")}}
            tool_calls.append(call)
            assistant_messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif item_type == "message":
            content = item.get("content")
            if item.get("role") != "assistant" or not isinstance(content, list):
                raise AgentRunError("model response message must be assistant content", status=502, code="InvalidModelResponse")
            phase = item.get("phase")
            if phase not in (None, "commentary", "final_answer"):
                raise AgentRunError("model response message phase is invalid", status=502, code="InvalidModelResponse")
            parts: list[str] = []
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part, Mapping) and part.get("type") == "refusal" and isinstance(part.get("refusal"), str):
                    parts.append(part["refusal"])
                else:
                    raise AgentRunError("model response message content is invalid", status=502, code="InvalidModelResponse")
            if parts:
                text = "".join(parts)
                texts.append(text)
                # Phase belongs to each message, not to the combined assistant text.
                assistant = {"role": "assistant", "content": text}
                if phase is not None:
                    assistant["phase"] = phase
                assistant_messages.append(assistant)
        elif item_type == "reasoning":
            # Keep each replayable reasoning item in its original output position.
            encrypted_content = item.get("encrypted_content")
            if isinstance(encrypted_content, str) and encrypted_content:
                summary = item.get("summary")
                retained = {"type": "reasoning", "summary": summary if isinstance(summary, list) else [], "encrypted_content": encrypted_content}
                if isinstance(item.get("id"), str):
                    retained["id"] = item["id"]
                assistant_messages.append({"role": "assistant", "content": None, "reasoning": [retained]})
        elif isinstance(item_type, str) and item_type.endswith("_call"):
            # Only function tools are offered; any other tool call is unsupported.
            tool_calls.append({"type": item_type})
        # Other non-tool items carry no assistant output.
    message: dict[str, Any] = {
        "role": "assistant", "content": "\n\n".join(texts) if texts else None,
        "assistant_messages": assistant_messages,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _message_text(message: Mapping[str, Any], *, max_bytes: int) -> str:
    content = message.get("content")
    if isinstance(content, str):
        text = content
    else:
        refusal = message.get("refusal")
        if content is not None or not isinstance(refusal, str):
            raise AgentRunError(
                "model response final assistant content must be a string",
                status=502,
                code="InvalidModelResponse",
            )
        text = refusal
    if len(text) > max_bytes:
        raise _model_response_too_large_error()
    try:
        text_bytes = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AgentRunError(
            "model response final assistant content must contain valid Unicode",
            status=502,
            code="InvalidModelResponse",
        ) from exc
    if len(text_bytes) > max_bytes:
        raise _model_response_too_large_error()
    return text


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise AgentRunError("model tool arguments must be a JSON object string", status=400, code="InvalidToolArguments")
    try:
        parsed = json.loads(
            raw or "{}",
            parse_float=_parse_json_float,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_argument_keys,
        )
    except AgentRunError:
        raise
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise AgentRunError("model tool arguments must be valid JSON", status=400, code="InvalidToolArguments") from exc
    if not isinstance(parsed, dict):
        raise AgentRunError("model tool arguments must be a JSON object", status=400, code="InvalidToolArguments")
    return parsed


def _reject_duplicate_argument_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise AgentRunError(f"model tool arguments contain duplicate key {key!r}", status=400, code="InvalidToolArguments")
        out[key] = value
    return out


def _validate_argument_unicode(value: Any, *, path: str = "arguments") -> None:
    pending: list[tuple[Any, str, int]] = [(value, path, 0)]
    while pending:
        current, current_path, depth = pending.pop()
        if depth > _MAX_ARGUMENT_DEPTH:
            raise AgentRunError("model tool arguments are nested too deeply", status=400, code="InvalidToolArguments")
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise AgentRunError(f"model tool arguments contain invalid Unicode at {current_path}", status=400, code="InvalidToolArguments") from exc
        elif isinstance(current, Mapping):
            for key, child in current.items():
                pending.append((child, f"{current_path}[{key!r}]", depth + 1))
                pending.append((key, f"{current_path}.<key>", depth + 1))
        elif isinstance(current, list):
            for index, child in enumerate(current):
                pending.append((child, f"{current_path}[{index}]", depth + 1))


def _parse_json_float(raw: str) -> float:
    try:
        decimal = Decimal(raw)
    except InvalidOperation as exc:
        raise AgentRunError("model tool arguments must contain valid JSON numbers", status=400, code="InvalidToolArguments") from exc
    parsed = float(decimal)
    if not Decimal(str(parsed)) == decimal:
        raise AgentRunError(
            "model tool arguments contain a number that cannot be represented exactly",
            status=400,
            code="InvalidToolArguments",
        )
    return parsed


def _reject_json_constant(raw: str) -> None:
    raise AgentRunError(f"model tool arguments contain non-finite number {raw}", status=400, code="InvalidToolArguments")


def _usage_token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentRunError(
            "model response usage must contain non-negative integer token counts",
            status=502,
            code="InvalidModelResponse",
        )
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise AgentRunError(
            "model response usage must contain non-negative integer token counts",
            status=502,
            code="InvalidModelResponse",
        )
    count = int(value)
    if count < 0:
        raise AgentRunError(
            "model response usage must contain non-negative integer token counts",
            status=502,
            code="InvalidModelResponse",
        )
    return count


def _usage(data: Mapping[str, Any]) -> dict[str, int]:
    usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else {}
    prompt_count = _usage_token_count(usage.get("input_tokens", 0))
    completion_count = _usage_token_count(usage.get("output_tokens", 0))
    total_count = _usage_token_count(usage.get("total_tokens", prompt_count + completion_count))
    return {"prompt_tokens": prompt_count, "completion_tokens": completion_count, "total_tokens": total_count}


__all__ = ["BrokeredChatModelLoop", "ModelLoopFinal", "ModelLoopToolRequest"]
