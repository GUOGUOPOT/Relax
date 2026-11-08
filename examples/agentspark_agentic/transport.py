# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""AgentSpark Eden transport. No automatic request retries or invented cancellation API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .protocol import SEND_PATH, SESSION_PATH, AdapterConfig, AdapterError, RelaxContext


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
    """Split only on LF bytes; Unicode line separators are valid JSON content."""
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
        try:
            async with self._client.stream(
                "POST",
                self.config.base_url + SEND_PATH,
                json=body,
                headers={"Accept": "text/event-stream"},
            ) as response:
                if not 200 <= response.status_code < 300:
                    raise AdapterError("agentspark_http_error")
                async for event in iter_sse(response, max_frame_bytes=self.config.max_callback_bytes):
                    yield event
        except httpx.HTTPError as exc:
            # Never repeat a POST: the server may already have executed tools.
            raise AdapterError("agentspark_transport_error") from exc

    async def restore(self, body: dict[str, Any]) -> str:
        session_id: str | None = None
        async for event in self.events(body):
            payload = event.payload
            if event.is_error:
                raise AdapterError("replay_error_event")
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
        if session_id is None:
            raise AdapterError("replay_eof_without_ack")
        return session_id
