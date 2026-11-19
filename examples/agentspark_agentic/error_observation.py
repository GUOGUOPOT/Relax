# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded, content-free observations of upstream errors.

These heuristic observations must never drive retries, failure classification,
or training control flow. Raw text, arbitrary keys and identifiers are omitted.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import deque
from typing import Any


_MAX_NODES = 64
_MAX_DEPTH = 6
_MAX_CHILDREN = 16
_MAX_STRING_CHARS = 4096
_MAX_TOTAL_CHARS = 16384
_MAX_JSON_DECODES = 8
_MAX_SHAPES = 24
_TYPES = frozenset({"null", "bool", "int", "float", "str", "dict", "list", "other"})
_PHASES = frozenset({"replay", "send", "callback"})
_FIELDS = {
    "error": "error",
    "data": "data",
    "detail": "detail",
    "details": "detail",
    "error_detail": "error_detail",
    "errorDetail": "error_detail",
    "cause": "cause",
    "message": "message",
    "msg": "message",
    "error_message": "error_message",
    "errorMessage": "error_message",
    "errors": "errors",
    "exception": "exception",
    "inner_error": "inner_error",
    "innerError": "inner_error",
    "code": "code",
    "error_code": "code",
    "errorCode": "code",
    "status": "status",
    "status_code": "status_code",
    "statusCode": "status_code",
    "http_status": "http_status",
    "_http_status": "http_status",
    "type": "type",
    "event": "event",
    "session_id": "session_id",
    "sessionId": "session_id",
    "request_id": "request_id",
    "requestId": "request_id",
}
_SHAPE_FIELDS = frozenset(_FIELDS.values()) | {"root", "item"}
_SESSION_REFS = frozenset({"present", "empty", "matches_expected", "differs_expected"})
# Every emitted marker and classification below is a source-code constant.
_PATTERNS = (
    ("limiter_unavailable_text", "limiter_unavailable", r"(?:rate[ _-]?limiter|sse[ _-]?limiter).{0,40}unavailable"),
    (
        "rate_limit_exceeded_text",
        "rate_limited",
        r"rate[ _-]?limit(?:er)?(?:[ _-]?exceeded|.{0,30}(?:exceeded|reject))",
    ),
    ("too_many_requests_text", "rate_limited", r"too[ _-]?many[ _-]?requests"),
    ("admission_rejected_text", "rate_limited", r"(?:admission|concurrency[ _-]?limit).{0,30}(?:reject|exceeded)"),
    (
        "capacity_rejected_text",
        "capacity_rejected",
        r"(?:capacity|concurrency|connections?)(?:[ _-]?(?:limit|capacity))?.{0,32}(?:reached|exceeded|full|exhausted|rejected)",
    ),
    (
        "too_many_concurrent_text",
        "capacity_rejected",
        r"too[ _-]?many[ _-]?(?:connections|concurrent[ _-]?requests|active[ _-]?(?:requests|sessions))",
    ),
    ("http_429_text", "rate_limited", r"\b(?:http(?:/\d(?:\.\d)?)?|status(?:_code)?)\s*[:=]?\s*429\b"),
    ("http_5xx_text", "http_server_error", r"\b(?:http(?:/\d(?:\.\d)?)?|status(?:_code)?)\s*[:=]?\s*5\d\d\b"),
    (
        "validation_text",
        "validation",
        r"invalid[ _-]?(?:request|argument|parameter)|validation[ _-]?(?:error|fail)|schema.{0,20}(?:invalid|fail)|badrequesterror|unprocessable[ _-]?entity",
    ),
    (
        "connection_text",
        "connection",
        r"connecterror|connection[ _-]?(?:refused|reset|closed|error)|network[ _-]?unreachable|(?:failed|unable)[ _-]?to[ _-]?connect|broken[ _-]?pipe|server[ _-]?disconnected",
    ),
    (
        "timeout_text",
        "timeout",
        r"\btimeout\b|timeouterror|(?:connect|read|write|pool)timeout|timed[ _-]?out|deadline[ _-]?exceeded|etimedout",
    ),
    (
        "session_busy_text",
        "session_busy",
        r"session.{0,20}(?:busy|already[ _-]?running|concurrent)|(?:busy|concurrent).{0,20}session",
    ),
    ("cancelled_text", "cancelled", r"\bcancelled\b|\bcanceled\b|\bcancellation\b|session[ _-]?discarded"),
    (
        "session_missing_text",
        "session_state_error",
        r"session.{0,20}(?:not[ _-]?found|expired|invalid)|unknown[ _-]?session",
    ),
    ("redis_text", "redis_dependency", r"\bredis\b|rediserror|redisconnection"),
)
_COMPILED = tuple((marker, category, re.compile(pattern, re.I)) for marker, category, pattern in _PATTERNS)
_MARKERS = frozenset(marker for marker, _, _ in _PATTERNS) | {"http_429_status", "http_5xx_status"}
_CLASSES = frozenset(category for _, category, _ in _PATTERNS) | {"unknown"}


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    for cls, name in ((bool, "bool"), (str, "str"), (int, "int"), (float, "float"), (dict, "dict")):
        if isinstance(value, cls):
            return name
    return "list" if isinstance(value, (list, tuple)) else "other"


def _integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def sanitize_error_observation(value: Any) -> dict[str, Any]:
    """Rebuild the fixed observation schema; never copy arbitrary audit
    data."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {"schema_version": 1}
    for key, allowed in (("phase", _PHASES), ("payload_type", _TYPES)):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate in allowed:
            result[key] = candidate
    digest = value.get("payload_sample_sha256")
    if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
        result["payload_sample_sha256"] = digest
    for key, allowed in (("classifications", _CLASSES), ("markers", _MARKERS), ("session_refs", _SESSION_REFS)):
        items = value.get(key)
        if isinstance(items, (list, tuple)):
            result[key] = list(dict.fromkeys(x for x in items[:32] if isinstance(x, str) and x in allowed))
    for key, low, high in (("http_statuses", 100, 599), ("business_codes", -(2**31), 2**31 - 1)):
        items = value.get(key)
        if isinstance(items, (list, tuple)):
            result[key] = list(dict.fromkeys(x for x in items[:8] if _integer(x, low, high)))
    shapes = value.get("field_shapes")
    if isinstance(shapes, (list, tuple)):
        safe_shapes = []
        for item in shapes[:_MAX_SHAPES]:
            if not isinstance(item, dict):
                continue
            field, kind = item.get("field"), item.get("type")
            if (
                not isinstance(field, str)
                or field not in _SHAPE_FIELDS
                or not isinstance(kind, str)
                or kind not in _TYPES
            ):
                continue
            shape = {"field": field, "type": kind}
            if _integer(item.get("length"), 0, 2**31 - 1):
                shape["length"] = item["length"]
            safe_shapes.append(shape)
        result["field_shapes"] = safe_shapes
    limits = value.get("limits")
    if isinstance(limits, dict):
        safe_limits = {}
        for key, maximum in (
            ("visited_nodes", _MAX_NODES),
            ("inspected_chars", _MAX_TOTAL_CHARS),
            ("json_decodes", _MAX_JSON_DECODES),
        ):
            if _integer(limits.get(key), 0, maximum):
                safe_limits[key] = limits[key]
        for key in ("node_limited", "depth_limited", "children_limited", "text_limited"):
            if type(limits.get(key)) is bool:
                safe_limits[key] = limits[key]
        result["limits"] = safe_limits
    return result


def observe_error_payload(
    payload: Any,
    *,
    http_status: int | None = None,
    phase: str | None = None,
    expected_session_id: str | None = None,
) -> dict[str, Any]:
    """Observe bounded known envelope fields without retaining any text.

    The digest covers inspected values only, not the complete original payload.
    A marker is lexical evidence, not proof of the upstream failure's cause.
    """
    classes, markers, statuses, codes, session_refs = set(), set(), set(), set(), set()
    shapes = []
    limits = {
        "visited_nodes": 0,
        "inspected_chars": 0,
        "json_decodes": 0,
        "node_limited": False,
        "depth_limited": False,
        "children_limited": False,
        "text_limited": False,
    }
    digest = hashlib.sha256()
    pending = deque([(payload, 0, "root")])
    seen_containers: set[int] = set()

    def status(value: Any) -> None:
        if _integer(value, 100, 599):
            statuses.add(value)
            if value == 429:
                classes.add("rate_limited")
                markers.add("http_429_status")
            elif value >= 500:
                classes.add("http_server_error")
                markers.add("http_5xx_status")

    def enqueue(value: Any, depth: int, field: str) -> None:
        if depth > _MAX_DEPTH:
            limits["depth_limited"] = True
        elif limits["visited_nodes"] + len(pending) >= _MAX_NODES:
            limits["node_limited"] = True
        else:
            pending.append((value, depth, field))

    status(http_status)
    while pending:
        value, depth, field = pending.popleft()
        limits["visited_nodes"] += 1
        kind = _kind(value)
        digest.update((field + ":" + kind + "\0").encode())
        shape = {"field": field, "type": kind}
        if kind in {"str", "dict", "list"}:
            shape["length"] = min(len(value), 2**31 - 1)
            digest.update(str(shape["length"]).encode())
        elif kind == "bool":
            digest.update(b"true" if value else b"false")
        elif kind == "int" and value.bit_length() <= 64:
            digest.update(str(value).encode())
        elif kind == "float":
            digest.update(value.hex().encode())
        if len(shapes) < _MAX_SHAPES:
            shapes.append(shape)
        if field in {"status", "status_code", "http_status"}:
            status(value)
        if field == "code" and _integer(value, -(2**31), 2**31 - 1):
            codes.add(value)
        if field == "session_id" and isinstance(value, str):
            session_refs.add("present" if value else "empty")
            if expected_session_id is not None:
                session_refs.add("matches_expected" if value == expected_session_id else "differs_expected")
        if isinstance(value, str):
            size = min(_MAX_STRING_CHARS, _MAX_TOTAL_CHARS - limits["inspected_chars"])
            sampled = value[:size]
            limits["inspected_chars"] += len(sampled)
            limits["text_limited"] |= len(sampled) < len(value)
            digest.update(sampled.encode("utf-8", errors="replace"))
            # Identifier values are only compared; do not classify their text.
            if field not in {"session_id", "request_id"}:
                for marker, category, pattern in _COMPILED:
                    if pattern.search(sampled):
                        classes.add(category)
                        markers.add(marker)
                if len(sampled) == len(value) and sampled.lstrip().startswith(("{", "[", '"')):
                    if limits["json_decodes"] < _MAX_JSON_DECODES:
                        limits["json_decodes"] += 1
                        try:
                            decoded = json.loads(sampled)
                        except (ValueError, RecursionError):
                            pass
                        else:
                            enqueue(decoded, depth + 1, field)
            digest.update(b"\0")
        elif kind in {"dict", "list"}:
            if id(value) in seen_containers:
                continue
            seen_containers.add(id(value))
            limits["children_limited"] |= len(value) > _MAX_CHILDREN
            if isinstance(value, dict):
                for key, child in itertools.islice(value.items(), _MAX_CHILDREN):
                    if isinstance(key, str) and key in _FIELDS:
                        enqueue(child, depth + 1, _FIELDS[key])
            else:
                for child in itertools.islice(value, _MAX_CHILDREN):
                    enqueue(child, depth + 1, "item")
    result = {
        "schema_version": 1,
        "payload_type": _kind(payload),
        "phase": phase,
        "payload_sample_sha256": digest.hexdigest(),
        "classifications": sorted(classes or {"unknown"}),
        "markers": sorted(markers),
        "http_statuses": sorted(statuses)[:8],
        "business_codes": sorted(codes)[:8],
        "session_refs": sorted(session_refs),
        "field_shapes": shapes,
        "limits": limits,
    }
    return sanitize_error_observation(result)
