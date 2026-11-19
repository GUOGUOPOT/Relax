# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Content-free observational parsing, independent of training control flow."""

import json

import pytest

from examples.agentspark_agentic.error_observation import observe_error_payload, sanitize_error_observation


@pytest.mark.parametrize(
    ("payload", "classification"),
    [
        (
            {"error_detail": {"cause": {"errorMessage": "Eden SSE rate limiter unavailable: total"}}},
            "limiter_unavailable",
        ),
        ({"error": json.dumps({"message": "rate limit exceeded"})}, "rate_limited"),
        ({"errors": [{"msg": "connection refused"}]}, "connection"),
        (json.dumps(json.dumps({"message": "deadline exceeded"})), "timeout"),
        ({"detail": "schema validation failed"}, "validation"),
        ({"message": "session is busy"}, "session_busy"),
        ({"cause": "session cancelled"}, "cancelled"),
        ({"errorMessage": "Redis connection reset"}, "redis_dependency"),
        ({"message": "HTTP/1.1 503 Service Unavailable"}, "http_server_error"),
        ({"message": "HTTP 429 Too Many Requests"}, "rate_limited"),
        ({"message": "session not found"}, "session_state_error"),
        ({"message": "capacity full"}, "capacity_rejected"),
        ({"message": "concurrency limit exceeded"}, "capacity_rejected"),
        ({"message": "connection limit 128 reached"}, "capacity_rejected"),
        ({"message": "too many active sessions"}, "capacity_rejected"),
        ({"message": "too many concurrent requests"}, "capacity_rejected"),
        ({"message": "InvalidArgument: bad schema"}, "validation"),
    ],
)
def test_nested_error_markers_are_observed_without_text(payload, classification):
    observed = observe_error_payload(payload, phase="send")
    assert classification in observed["classifications"]
    assert observed["phase"] == "send"
    assert "reason_code" not in observed
    assert sanitize_error_observation(observed) == observed


def test_content_identifiers_and_unknown_keys_are_never_emitted():
    secret = "PRIVATE_SECRET_https://host.invalid/path?token=DO_NOT_PRINT"
    payload = {
        "message": "connection refused " + secret,
        "session_id": secret,
        "request_id": secret,
        secret: {"token": secret},
        "error": {"authorization": secret},
    }
    observed = observe_error_payload(payload, expected_session_id=secret)
    serialized = json.dumps(observed)
    for forbidden in (secret, "PRIVATE_SECRET", "https://", "DO_NOT_PRINT", "authorization"):
        assert forbidden not in serialized
    assert observed["session_refs"] == ["matches_expected", "present"]
    assert observed["classifications"] == ["connection"]


def test_numeric_status_and_business_codes_are_separate():
    observed = observe_error_payload({"code": 500, "data": {"errorCode": -1203}}, http_status=429)
    assert observed["business_codes"] == [-1203, 500]
    assert observed["http_statuses"] == [429]
    assert observed["classifications"] == ["rate_limited"]
    assert "http_server_error" not in observed["classifications"]
    assert observe_error_payload({"status_code": 503})["classifications"] == ["http_server_error"]
    invalid = observe_error_payload({"code": True, "status": "503", "error_code": 2**60}, http_status=False)
    assert invalid["business_codes"] == invalid["http_statuses"] == []


def test_unknown_payload_and_free_text_are_not_control_classifications():
    for payload in (None, 123, "PRIVATE_UNKNOWN", {"content": "timeout"}, {"request_id": "timeout"}):
        assert observe_error_payload(payload)["classifications"] == ["unknown"]


def test_depth_nodes_text_and_json_decoding_are_bounded():
    deep = {"message": "timeout"}
    for _ in range(20):
        deep = {"cause": deep}
    observed = observe_error_payload(deep)
    assert observed["limits"]["depth_limited"]
    assert observed["classifications"] == ["unknown"]
    broad = [["x" * 5000 for _ in range(30)] for _ in range(30)]
    observed = observe_error_payload(broad)
    assert observed["limits"]["visited_nodes"] <= 64
    assert observed["limits"]["inspected_chars"] <= 16384
    assert observed["limits"]["node_limited"]
    assert observed["limits"]["children_limited"]
    assert observed["limits"]["text_limited"]
    assert len(observed["field_shapes"]) <= 24
    encoded = [json.dumps({"message": "timeout"}) for _ in range(16)]
    assert observe_error_payload(encoded)["limits"]["json_decodes"] <= 8
    cycle = {}
    cycle["cause"] = cycle
    assert observe_error_payload(cycle)["limits"]["visited_nodes"] <= 2


def test_message_prefix_limit_and_digest_are_explicit():
    observed = observe_error_payload("x" * 4096 + " timeout")
    assert observed["classifications"] == ["unknown"]
    assert observed["limits"]["text_limited"]
    assert len(observed["payload_sample_sha256"]) == 64


def test_sanitizer_discards_malicious_audit_fields_and_values():
    secret = "PRIVATE_SECRET_https://bad.invalid/token"
    observed = sanitize_error_observation(
        {
            "schema_version": secret,
            "phase": secret,
            "payload_type": secret,
            "payload_sample_sha256": secret,
            "message": secret,
            secret: secret,
            "classifications": [secret, "timeout"],
            "markers": [secret, "timeout_text"],
            "session_refs": [secret, "present"],
            "http_statuses": [True, 999, 503],
            "business_codes": [secret, True, 2**60, -1],
            "field_shapes": [
                {"field": secret, "type": "str"},
                {"field": "message", "type": "str", "length": 5, "value": secret},
            ],
            "limits": {"message": secret, "visited_nodes": 10000, "text_limited": True},
        }
    )
    assert secret not in json.dumps(observed)
    assert observed["schema_version"] == 1
    assert observed["classifications"] == ["timeout"]
    assert observed["http_statuses"] == [503]
    assert observed["business_codes"] == [-1]
    assert observed["field_shapes"] == [{"field": "message", "type": "str", "length": 5}]
    assert observed["limits"] == {"text_limited": True}
    assert sanitize_error_observation(None) == {}
