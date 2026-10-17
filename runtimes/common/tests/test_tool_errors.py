"""Orka broker outcome codes reach the model as fixed or controller-written text."""

from __future__ import annotations

import pytest

from agentkit_serve_common.tool_errors import orka_tool_error_details


_NO_EFFECT = "tool_failed_without_effect"
_NO_EFFECT_FIXED = "The tool call failed without taking effect; nothing changed."


def test_no_effect_failure_forwards_the_controller_summary():
    summary = "GitHub refused the request (HTTP 403): Resource not accessible by integration"
    assert orka_tool_error_details({"code": _NO_EFFECT, "message": summary}) == (_NO_EFFECT, summary)


def test_no_effect_failure_collapses_summary_whitespace():
    details = orka_tool_error_details({"code": _NO_EFFECT, "message": "  The tool endpoint refused\n the request (HTTP 403).\t"})
    assert details == (_NO_EFFECT, "The tool endpoint refused the request (HTTP 403).")


@pytest.mark.parametrize(
    "message",
    [None, "", "   ", 403, {"text": "refused"}, "x" * 1025, "refused\x00the request", "refused\x1b[31mthe request"],
)
def test_no_effect_failure_falls_back_to_fixed_text(message):
    value = {"code": _NO_EFFECT} if message is None else {"code": _NO_EFFECT, "message": message}
    assert orka_tool_error_details(value) == (_NO_EFFECT, _NO_EFFECT_FIXED)


@pytest.mark.parametrize(
    "code",
    ["approval_declined", "approval_expired", "approval_cancelled", "approval_stale", "tool_execution_failed", "tool_outcome_unknown"],
)
def test_other_codes_never_forward_message_text(code):
    code_, message = orka_tool_error_details({"code": code, "message": "PRIVATE_UNTRUSTED_TOOL_ERROR"})
    assert code_ == code
    assert "PRIVATE_UNTRUSTED_TOOL_ERROR" not in message


@pytest.mark.parametrize("value", [None, "tool_failed_without_effect", {"code": 1}, {"code": "unknown_code"}])
def test_unrecognized_values_are_not_broker_outcomes(value):
    assert orka_tool_error_details(value) is None
