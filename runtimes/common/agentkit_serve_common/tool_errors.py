"""Fixed, non-sensitive outcomes from Orka's governed tool broker."""

from __future__ import annotations

from collections.abc import Mapping


_ORKA_TOOL_ERRORS = {
    "approval_declined": "The tool call was declined.",
    "approval_expired": "The tool approval expired.",
    "approval_cancelled": "The tool call was cancelled.",
    "approval_stale": "The tool approval is no longer valid.",
    "tool_execution_failed": "MCP tool execution failed.",
    "tool_failed_without_effect": "The tool call failed without taking effect; nothing changed.",
    "tool_outcome_unknown": "The tool execution outcome is unknown; do not retry.",
}

# Orka writes this code's message itself: a redacted, bounded summary such as
# the HTTP status a service refused the request with. Its adapter contract
# lets runtimes forward it so the model can see why nothing happened.
_CONTROLLER_SUMMARY_CODES = frozenset({"tool_failed_without_effect"})
_MAX_CONTROLLER_SUMMARY_CHARS = 1024


def orka_tool_error_details(value: object) -> tuple[str, str] | None:
    """Read an explicit broker code without forwarding tool-controlled text."""
    if not isinstance(value, Mapping):
        return None
    code = value.get("code")
    if not isinstance(code, str):
        return None
    message = _ORKA_TOOL_ERRORS.get(code)
    if message is None:
        return None
    if code in _CONTROLLER_SUMMARY_CODES:
        message = _controller_summary(value.get("message")) or message
    return code, message


def _controller_summary(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    summary = " ".join(value.split())
    if not summary or len(summary) > _MAX_CONTROLLER_SUMMARY_CHARS or not summary.isprintable():
        return None
    return summary
