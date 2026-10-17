"""Orka broker outcome codes reach the model as fixed or controller-written text."""

from __future__ import annotations

import pytest

from agentkit_serve_common.tool_errors import orka_tool_error_details


_NO_EFFECT = "tool_failed_without_effect"
_NO_EFFECT_FIXED = "The tool call failed without taking effect; nothing changed."


def test_no_effect_failure_forwards_the_broker_result_summary():
    # The exact result shape Orka's broker returns for a refused approved call.
    summary = "GitHub refused the request (HTTP 403): Resource not accessible by integration"
    broker_result = {"isError": True, "code": _NO_EFFECT, "approvalID": "acp-tool-approval-v2:sha256:1", "error": summary}
    assert orka_tool_error_details(broker_result) == (_NO_EFFECT, summary)


def test_no_effect_failure_forwards_a_relayed_message_summary():
    summary = "The tool endpoint refused the request (HTTP 403)."
    assert orka_tool_error_details({"code": _NO_EFFECT, "message": summary}) == (_NO_EFFECT, summary)


def test_no_effect_failure_prefers_the_broker_error_field():
    details = orka_tool_error_details({"code": _NO_EFFECT, "error": "broker summary", "message": "relayed summary"})
    assert details == (_NO_EFFECT, "broker summary")


def test_no_effect_failure_collapses_summary_whitespace():
    details = orka_tool_error_details({"code": _NO_EFFECT, "message": "  The tool endpoint refused\n the request (HTTP 403).\t"})
    assert details == (_NO_EFFECT, "The tool endpoint refused the request (HTTP 403).")


@pytest.mark.parametrize(
    "message",
    [None, "", "   ", 403, {"text": "refused"}, "x" * 1025, "refused\x00the request", "refused\x1b[31mthe request"],
)
def test_no_effect_failure_falls_back_to_fixed_text(message):
    for field in ("error", "message"):
        value = {"code": _NO_EFFECT} if message is None else {"code": _NO_EFFECT, field: message}
        assert orka_tool_error_details(value) == (_NO_EFFECT, _NO_EFFECT_FIXED)


@pytest.mark.parametrize(
    "code",
    ["approval_declined", "approval_expired", "approval_cancelled", "approval_stale", "tool_execution_failed", "tool_outcome_unknown"],
)
def test_other_codes_never_forward_message_text(code):
    code_, message = orka_tool_error_details(
        {"code": code, "error": "PRIVATE_UNTRUSTED_TOOL_ERROR", "message": "PRIVATE_UNTRUSTED_TOOL_ERROR"}
    )
    assert code_ == code
    assert "PRIVATE_UNTRUSTED_TOOL_ERROR" not in message


@pytest.mark.parametrize("value", [None, "tool_failed_without_effect", {"code": 1}, {"code": "unknown_code"}])
def test_unrecognized_values_are_not_broker_outcomes(value):
    assert orka_tool_error_details(value) is None
