# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""AgentSpark Eden transport.

No automatic request retries or invented cancellation API.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .diagnostics import UpstreamDiagnosticError, sse_admission_rejection_reason
from .error_observation import observe_error_payload
from .protocol import SEND_PATH, SESSION_PATH, AdapterConfig, AdapterError, RelaxContext


class ReplayUnavailableError(AdapterError):
    """A complete replay stream explicitly rejected before any
    acknowledgement."""

    def __init__(self, reason_code: str = "eden_sse_rate_limiter_unavailable") -> None:
        super().__init__("replay_temporarily_unavailable")
        self.safe_diagnostic = {"reason_code": reason_code, "phase": "replay"}


@dataclass(frozen=True)
class SSEEvent:
    event: str | None
    payload: Any
    raw: str

    @property
    def kind(self) -> str | None:
        if isinstance(self.payload, dict) and isinstance(self.payload.get("type"), str):
            return self.payload["type"] or self.event
        return self.event

    @property
    def is_error(self) -> bool:
        return self.event in {"error", "agent.stream.error"} or self.kind in {
            "error",
            "agent.stream.error",
        }


async def iter_sse(response: httpx.Response, *, max_frame_bytes: int = 16 * 1024 * 1024) -> AsyncIterator[SSEEvent]:
    """Split only on LF bytes; Unicode line separators are valid JSON
    content."""
    pending = bytearray()
    event_name: str | None = None
    data: list[str] = []
    frame_size = 0

    def frame() -> SSEEvent | None:
        if not data and event_name is None:
            return None
        raw = "\n".join(data)
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = raw
        return SSEEvent(event_name, payload, raw)

    async for chunk in response.aiter_bytes():
        pending.extend(chunk)
        while b"\n" in pending:
            line, _, remaining = pending.partition(b"\n")
            pending = bytearray(remaining)
            try:
                text = bytes(line).removesuffix(b"\r").decode("utf-8")
            except UnicodeDecodeError as exc:
                raise AdapterError("invalid_sse_utf8") from exc
            frame_size += len(line)
            if frame_size > max_frame_bytes:
                raise AdapterError("sse_frame_too_large")
            if not text:
                decoded = frame()
                if decoded is not None:
                    yield decoded
                event_name, data, frame_size = None, [], 0
            elif not text.startswith(":"):
                field, separator, value = text.partition(":")
                if separator:
                    value = value[1:] if value.startswith(" ") else value
                    if field == "event":
                        event_name = value or None
                    elif field == "data":
                        data.append(value)
        if len(pending) + frame_size > max_frame_bytes:
            raise AdapterError("sse_frame_too_large")
    if pending:
        try:
            text = bytes(pending).removesuffix(b"\r").decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AdapterError("invalid_sse_utf8") from exc
        field, separator, value = text.partition(":")
        if separator:
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                event_name = value or None
            elif field == "data":
                data.append(value)
    decoded = frame()
    if decoded is not None:
        yield decoded


class AgentSparkClient:
    def __init__(self, config: AdapterConfig, *, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._client = client
        self._owned = client is None
        self.error_observations: list[dict[str, Any]] = []

    async def __aenter__(self) -> "AgentSparkClient":
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout_s,
                    pool=self.config.connect_timeout_s,
                    read=self.config.read_timeout_s,
                    write=self.config.read_timeout_s,
                ),
                trust_env=False,
                follow_redirects=False,
            )
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        if self._owned and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def closed(self) -> bool:
        return not self._owned or self._client is None

    async def get_session(self, context: RelaxContext) -> str:
        if self._client is None:
            raise AdapterError("client_not_open")
        try:
            response = await self._client.get(
                self.config.base_url + SESSION_PATH,
                params={"user_id": context.user_id, "source": self.config.source},
            )
        except httpx.HTTPError as exc:
            raise AdapterError("session_lookup_transport_error") from exc
        if not 200 <= response.status_code < 300:
            raise AdapterError("session_lookup_http_error")
        try:
            payload = response.json()
        except ValueError as exc:
            raise AdapterError("session_lookup_not_json") from exc
        session_id = payload.get("session_id") if isinstance(payload, dict) else None
        if not isinstance(session_id, str) or not session_id.strip():
            raise AdapterError("session_lookup_id_missing")
        return session_id

    async def events(self, body: dict[str, Any]) -> AsyncIterator[SSEEvent]:
        if self._client is None:
            raise AdapterError("client_not_open")
        started_unix_ms = time.time_ns() // 1_000_000
        started = time.monotonic()
        frames = 0
        try:
            async with self._client.stream(
                "POST",
                self.config.base_url + SEND_PATH,
                json=body,
                headers={"Accept": "text/event-stream"},
            ) as response:
                if not 200 <= response.status_code < 300:
                    # Observe only the status; do not consume an error response body.
                    try:
                        phase = "replay" if isinstance(body.get("replay_spec"), dict) else "send"
                        self.error_observations.append(
                            {
                                "request_started_unix_ms": started_unix_ms,
                                "error_unix_ms": time.time_ns() // 1_000_000,
                                "elapsed_ms": int((time.monotonic() - started) * 1000),
                                "frames_before_error": frames,
                                "transport_http_status": response.status_code,
                                "detail": observe_error_payload({}, http_status=response.status_code, phase=phase),
                            }
                        )
                        del self.error_observations[:-4]
                    except Exception:
                        pass
                    raise AdapterError("agentspark_http_error")
                async for event in iter_sse(response, max_frame_bytes=self.config.max_callback_bytes):
                    if event.is_error:
                        # Observation never changes the control-path error classification.
                        try:
                            phase = "replay" if isinstance(body.get("replay_spec"), dict) else "send"
                            self.error_observations.append(
                                {
                                    "request_started_unix_ms": started_unix_ms,
                                    "error_unix_ms": time.time_ns() // 1_000_000,
                                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                                    "frames_before_error": frames,
                                    "transport_http_status": response.status_code,
                                    "event_raw_sha256": hashlib.sha256(event.raw.encode("utf-8")).hexdigest(),
                                    "detail": observe_error_payload(
                                        event.payload, http_status=response.status_code, phase=phase
                                    ),
                                }
                            )
                            del self.error_observations[:-4]
                        except Exception:
                            pass
                    frames += 1
                    yield event
        except httpx.HTTPError as exc:
            # Never repeat a POST: the server may already have executed tools.
            raise AdapterError("agentspark_transport_error") from exc

    async def restore(self, body: dict[str, Any]) -> str:
        session_id: str | None = None
        rejection: UpstreamDiagnosticError | None = None
        rejection_reason: str | None = None
        async for event in self.events(body):
            payload = event.payload
            if rejection is not None:
                # Any further event makes the attempt ambiguous, including an ack
                # after an apparent rejection. Never permit a retry in that case.
                raise rejection
            if event.is_error:
                failure = UpstreamDiagnosticError("replay_error_event", payload)
                failure.safe_diagnostic["phase"] = "replay"
                admission_reason = sse_admission_rejection_reason(payload)
                if (
                    isinstance(body.get("replay_spec"), dict)
                    and body["replay_spec"].get("replay_type") == "restore_only"
                    and session_id is None
                    and event.event in {None, "error", "agent.stream.error"}
                    and event.kind in {"error", "agent.stream.error"}
                    and admission_reason is not None
                ):
                    # Admission may be retried by the caller only after clean EOF;
                    # errors/acks following this frame must still fail closed.
                    rejection = failure
                    rejection_reason = admission_reason
                    continue
                raise failure
            if (
                event.event not in {None, "link.up.ack"}
                or event.kind not in {None, "link.up.ack"}
                or "link.up.ack" not in {event.event, event.kind}
            ):
                raise AdapterError("replay_unexpected_event")
            if not isinstance(payload, dict) or type(payload.get("code")) is not int or payload["code"] != 0:
                raise AdapterError("replay_invalid_ack")
            data = payload.get("data")
            candidate = data.get("session_id") if isinstance(data, dict) else None
            if not isinstance(candidate, str) or not candidate.strip():
                raise AdapterError("replay_ack_session_missing")
            if session_id is not None and session_id != candidate:
                raise AdapterError("replay_ack_session_inconsistent")
            session_id = candidate
        if rejection is not None:
            raise ReplayUnavailableError(rejection_reason)
        if session_id is None:
            raise AdapterError("replay_eof_without_ack")
        return session_id
