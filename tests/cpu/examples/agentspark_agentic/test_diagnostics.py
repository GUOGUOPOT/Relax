# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Safe replay/send failure summaries without copying upstream message text."""

import json

import pytest

from examples.agentspark_agentic.diagnostics import failure_summary, upstream_diagnostic
from examples.agentspark_agentic.protocol import AdapterError
from examples.agentspark_agentic.transport import ReplayUnavailableError


@pytest.mark.parametrize("phase", ["replay", "send"])
def test_known_sse_rejection_is_mapped_without_exposing_remote_content(phase: str) -> None:
    diagnostic = upstream_diagnostic(
        {
            "phase": phase,
            "error": {
                "message": "Eden SSE rate limiter unavailable: total",
                "request_id": "PRIVATE_ID",
                "url": "https://private.invalid",
                "authorization": "PRIVATE_KEY",
            },
            "content": {"code": "rate_limit_exceeded"},
        }
    )
    assert diagnostic == {"phase": phase, "reason_code": "eden_sse_rate_limiter_unavailable"}
    assert "PRIVATE" not in json.dumps(diagnostic)


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "Eden SSE rate limiter unavailable: total PRIVATE_KEY"},
        {"message": "Eden SSE rate limiter unavailable: unknown"},
        {"content": {"message": "Eden SSE rate limiter unavailable: total"}},
        {"phase": "PRIVATE_PHASE", "reason_code": ["PRIVATE_REASON"]},
    ],
)
def test_unknown_or_untrusted_message_content_is_not_diagnosed(payload: dict) -> None:
    assert upstream_diagnostic(payload) == {}


@pytest.mark.parametrize("code", ["replay_temporarily_unavailable", "replay_retry_exhausted"])
@pytest.mark.parametrize("phase", ["replay", "send"])
def test_failure_summary_preserves_safe_attempt_and_upstream_phase(code: str, phase: str) -> None:
    summary = failure_summary(
        AdapterError("context_audit_failure_not_recoverable"),
        audit={
            "failure_code": code,
            "failure_phase": phase,
            "failure_diagnostic": {
                "reason_code": "eden_sse_rate_limiter_unavailable",
                "message": "PRIVATE_MESSAGE",
                "request_id": "PRIVATE_ID",
            },
            "replay_attempts": 3,
        },
    )
    assert summary["audit_failure_code"] == code
    assert summary["replay_attempts"] == 3
    assert summary["upstream_errors"] == [
        {
            "origin": f"agentspark_{phase}",
            "reason_code": "eden_sse_rate_limiter_unavailable",
            "phase": phase,
        }
    ]
    assert "PRIVATE" not in json.dumps(summary)


def test_replay_unavailable_exception_diagnostic_survives_redaction() -> None:
    summary = failure_summary(ReplayUnavailableError())
    assert summary["error"] == "replay_temporarily_unavailable"
    assert summary["diagnostic"] == {
        "exception_type": "ReplayUnavailableError",
        "reason_code": "eden_sse_rate_limiter_unavailable",
        "phase": "replay",
    }
