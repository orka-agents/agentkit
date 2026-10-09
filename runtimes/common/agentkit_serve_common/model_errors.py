"""Runtime-owned model service error definitions shared by every protocol skin.

Model SDK exceptions can embed upstream response bodies, request URLs, and any
credential a gateway echoes back. Protocol skins therefore report only these
fixed definitions plus bounded HTTP metadata, never the upstream text.
"""

from __future__ import annotations

from typing import Any

from .runtime import AgentRunError

NORMALIZED_MODEL_ERRORS = {
    "ModelAuthMissing": (503, "model authentication is not configured"),
    "ModelAuthRejected": (503, "model service rejected configured credentials"),
    "ModelUnavailable": (503, "model service is unavailable"),
    "ModelUpstreamError": (502, "model service request failed"),
    "InvalidModelResponse": (502, "model service returned an invalid response"),
    "ModelResponseTooLarge": (502, "model response is too large to retain safely"),
}


class ModelHTTPError(AgentRunError):
    def __init__(self, message: str, *, status: int, code: str, upstream_status: int) -> None:
        super().__init__(message, status=status, code=code)
        self.upstream_status = upstream_status


def normalized_model_http_error(status_code: int) -> AgentRunError:
    if type(status_code) is not int or not 400 <= status_code <= 599:
        return AgentRunError(
            "model service request failed",
            status=502,
            code="ModelUpstreamError",
        )
    if status_code in {401, 403}:
        return ModelHTTPError(
            "model service rejected configured credentials",
            status=503,
            code="ModelAuthRejected",
            upstream_status=status_code,
        )
    if status_code == 429 or status_code >= 500:
        return ModelHTTPError(
            "model service is unavailable",
            status=503,
            code="ModelUnavailable",
            upstream_status=status_code,
        )
    return ModelHTTPError(
        "model service request failed",
        status=502,
        code="ModelUpstreamError",
        upstream_status=status_code,
    )


def normalized_model_error_details(exc: AgentRunError) -> tuple[int, dict[str, Any]] | None:
    """Project only runtime-owned model error definitions and bounded HTTP metadata."""
    definition = NORMALIZED_MODEL_ERRORS.get(exc.code) if type(exc.code) is str else None
    if definition is None:
        return None
    status, message = definition
    error: dict[str, Any] = {"message": message, "code": exc.code}
    # A similarly named attribute on a framework exception is not HTTP evidence.
    if (
        isinstance(exc, ModelHTTPError)
        and type(exc.upstream_status) is int
        and 400 <= exc.upstream_status <= 599
    ):
        error["upstream_status"] = exc.upstream_status
    return status, error
