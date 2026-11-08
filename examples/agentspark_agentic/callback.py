# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Narrow transparent Chat callback recorder; Relax remains the token/state owner."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from aiohttp import web

from .diagnostics import UpstreamDiagnosticError, upstream_diagnostic
from .history import ReasoningHistory
from .media import ASSET_PREFIX, MEDIA_PATH, MediaPlan, canonical_bytes, strict_json_loads
from .protocol import CHAT_PATH, AdapterConfig, AdapterError, RelaxContext, json_copy
from .result import reject_credentials
from .transport import iter_sse


async def assistant_from_response(body: bytes, content_type: str) -> dict[str, Any]:
    """Read the actual Relax JSON or single-terminal-chunk Buffered SSE response."""
    if "text/event-stream" in content_type:
        payloads = []
        saw_done = False
        async for event in iter_sse(httpx.Response(200, content=body)):
            if event.raw == "[DONE]":
                if saw_done:
                    raise AdapterError("callback_duplicate_done")
                saw_done = True
                continue
            if event.payload is None:
                continue
            if event.is_error or not isinstance(event.payload, dict) or "error" in event.payload:
                raise UpstreamDiagnosticError("callback_error_response", event.payload)
            if "choices" in event.payload and event.payload["choices"]:
                if saw_done:
                    raise AdapterError("callback_content_after_done")
                payloads.append(event.payload)
        if not saw_done or len(payloads) != 1:
            raise AdapterError("callback_not_buffered_sse")
        payload, key = payloads[0], "delta"
    else:
        try:
            payload = strict_json_loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise AdapterError("callback_response_not_json") from exc
        key = "message"
    if not isinstance(payload, dict) or "error" in payload:
        raise UpstreamDiagnosticError("callback_error_response", payload)
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise AdapterError("callback_choices_invalid")
    finish_reason = choices[0].get("finish_reason")
    if not isinstance(finish_reason, str) or finish_reason not in {
        "stop",
        "length",
        "tool_calls",
    }:
        raise AdapterError("callback_finish_reason_invalid")
    assistant = choices[0].get(key)
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        raise AdapterError("callback_assistant_invalid")
    if not (assistant.get("content") or assistant.get("reasoning_content") or assistant.get("tool_calls")):
        raise AdapterError("callback_assistant_empty")
    assistant = json_copy(assistant)
    if key == "delta" and assistant.get("tool_calls"):
        # Relax adds SSE-only indexes to complete tool calls. They are not
        # part of its committed canonical assistant message.
        for tool_call in assistant["tool_calls"]:
            if not isinstance(tool_call, dict):
                raise AdapterError("callback_tool_call_invalid")
            tool_call.pop("index", None)
    return assistant


@dataclass
class CallbackCapture:
    records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    active: int = 0
    concurrent: bool = False
    started: int = 0

    def linear_candidate(self) -> dict[str, Any]:
        """Prove payload append-only shape, not committed Forest identity or reward parity."""
        if self.errors or self.active or self.concurrent or not self.records or self.started != len(self.records):
            raise AdapterError("callback_capture_incomplete_or_concurrent")
        expected: list[dict[str, Any]] | None = None
        baseline_tools: Any = None
        baseline_kwargs: Any = None
        for record in sorted(self.records, key=lambda item: item["index"]):
            request = record["request"]
            messages = request.get("messages")
            tools = request.get("tools") or []
            kwargs = request.get("chat_template_kwargs") or {}
            if (
                not isinstance(messages, list)
                or not messages
                or not isinstance(tools, list)
                or not isinstance(kwargs, dict)
            ):
                raise AdapterError("callback_context_invalid")
            if any(not isinstance(message, dict) for message in messages):
                raise AdapterError("callback_context_invalid")
            if expected is not None:
                if tools != baseline_tools or kwargs != baseline_kwargs:
                    raise AdapterError("callback_tools_or_template_changed")
                if len(messages) <= len(expected) or messages[: len(expected)] != expected:
                    raise AdapterError("callback_history_not_linear")
                if any(message.get("role") not in {"user", "tool", "system"} for message in messages[len(expected) :]):
                    raise AdapterError("callback_observation_not_linear")
            baseline_tools, baseline_kwargs = tools, kwargs
            expected = json_copy(messages) + [json_copy(record["assistant"])]
        return {
            "name": "main",
            "messages": expected,
            "tools": baseline_tools,
            "chat_template_kwargs": baseline_kwargs,
        }


class CallbackRecorder:
    """One explicitly advertised, authenticated per-invocation callback server.

    Body bytes are unchanged unless an explicit private media plan restores them.
    Both raw and actual forwarded payloads are then captured. Captures omit headers and any raw error
    response. The output is audit evidence, never a Relax training publication.
    """

    def __init__(
        self,
        config: AdapterConfig,
        context: RelaxContext,
        *,
        client: httpx.AsyncClient | None = None,
        media: MediaPlan | None = None,
        restore_reasoning: bool = False,
        pinned_template_kwargs: dict[str, Any] | None = None,
        pinned_temperature: float | None = None,
        allow_dynamic_images: bool = False,
    ) -> None:
        self.config, self.context = config, context
        self.capture = CallbackCapture()
        self.media = media
        self.allow_dynamic_images = allow_dynamic_images
        self.history = ReasoningHistory(pinned_template_kwargs, pinned_temperature) if restore_reasoning else None
        self.public_base_url = config.callback_public_base_url
        self._client = client
        self._owned = client is None
        self._runner: web.AppRunner | None = None
        self._idle = asyncio.Event()
        self._idle.set()

    async def __aenter__(self) -> "CallbackRecorder":
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
        app = web.Application(client_max_size=self.config.max_callback_bytes)
        prefix = urlsplit(self.public_base_url).path.rstrip("/")
        app.router.add_post(prefix + CHAT_PATH, self._proxy)
        if self.media is not None and self.media.assets:
            app.router.add_get(prefix + MEDIA_PATH + self.media.scope + "/{digest}", self._asset)
        self._runner = web.AppRunner(app, shutdown_timeout=self.config.cleanup_timeout_s)
        try:
            await self._runner.setup()
            site = web.TCPSite(
                self._runner,
                self.config.callback_bind_host,
                self.config.callback_bind_port,
            )
            await site.start()
            parsed = urlsplit(self.public_base_url)
            if parsed.port == 0:
                addresses = self._runner.addresses
                if len(addresses) != 1:
                    raise AdapterError("callback_address_ambiguous")
                host = f"[{parsed.hostname}]" if ":" in str(parsed.hostname) else str(parsed.hostname)
                self.public_base_url = urlunsplit((parsed.scheme, f"{host}:{addresses[0][1]}", parsed.path, "", ""))
            if self.media is not None:
                self.media.public_base_url = self.public_base_url
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *_exc: Any) -> None:
        try:
            if self._runner is not None:
                await self._runner.cleanup()
                self._runner = None
        finally:
            if self._owned and self._client is not None:
                await self._client.aclose()
                self._client = None

    @property
    def closed(self) -> bool:
        return self._runner is None and (not self._owned or self._client is None)

    async def wait_idle(self) -> None:
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=self.config.cleanup_timeout_s)
        except asyncio.TimeoutError as exc:
            raise AdapterError("callback_drain_timeout") from exc

    async def _asset(self, request: web.Request) -> web.Response:
        # The random per-invocation route is a short-lived read capability. It
        # exposes only manifest bytes, never a caller-supplied path or URL.
        ref = ASSET_PREFIX + request.match_info["digest"]
        if self.media is None or ref not in self.media.assets:
            raise web.HTTPNotFound()
        asset, raw = self.media.assets[ref]
        return web.Response(body=raw, content_type=asset["mime_type"], headers={"Cache-Control": "no-store"})

    async def _proxy(self, request: web.Request) -> web.StreamResponse:
        auth = request.headers.get("Authorization", "")
        if not hmac.compare_digest(auth.encode(), f"Bearer {self.context.session_id}".encode()):
            return web.json_response({"error": {"code": "callback_authentication_failed"}}, status=401)
        if self._client is None:
            return web.json_response({"error": {"code": "callback_not_open"}}, status=503)
        try:
            body = await request.read()
            payload = strict_json_loads(body)
            if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
                raise AdapterError("callback_request_invalid")
            # Never persist a Session credential echoed into model content or
            # body metadata. Normal bearer credentials exist only in headers.
            reject_credentials(payload, self.context.session_id)
            if "stream" in payload and type(payload["stream"]) is not bool:
                raise AdapterError("callback_stream_invalid")
        except (
            ValueError,
            UnicodeDecodeError,
            web.HTTPRequestEntityTooLarge,
            AdapterError,
        ) as exc:
            code = exc.code if isinstance(exc, AdapterError) else "callback_request_invalid"
            self.capture.errors.append(code)
            return web.json_response({"error": {"code": code}}, status=400)
        index = self.capture.started
        self.capture.started += 1
        self.capture.concurrent |= self.capture.active > 0
        self.capture.active += 1
        self._idle.clear()
        outgoing: web.StreamResponse | None = None
        upstream_status: int | None = None
        try:
            raw_payload, raw_body = json_copy(payload), body
            media_evidence = None
            candidate_media = None
            if self.media is not None:
                if self.capture.active > 1:
                    raise AdapterError("media_concurrent_callback_unsupported")
                candidate_media = self.media.fork()
                payload, media_evidence = candidate_media.restore(payload)
                reject_credentials(payload, self.context.session_id)
                body = canonical_bytes(payload)
                if len(body) > self.config.max_callback_bytes:
                    raise AdapterError("callback_forwarded_request_too_large")
            history_evidence = None
            if self.history is not None:
                if self.capture.active > 1:
                    raise AdapterError("history_concurrent_callback_unsupported")
                payload, history_evidence = self.history.restore(payload)
                if self.media is not None and self.allow_dynamic_images and history_evidence["status"] == "new_segment":
                    # A reset is transactional too: recompute from the raw callback
                    # so no media admission from the abandoned segment survives.
                    candidate_media = self.media.fork(reset_segment=True)
                    payload, media_evidence = candidate_media.restore(raw_payload)
                    payload, history_evidence = self.history.restore(payload)
                reject_credentials(payload, self.context.session_id)
                body = canonical_bytes(payload)
                if len(body) > self.config.max_callback_bytes:
                    raise AdapterError("callback_forwarded_request_too_large")
            async with self._client.stream(
                "POST",
                self.context.base_url + CHAT_PATH,
                content=body,
                headers={
                    "Authorization": auth,
                    "Content-Type": "application/json",
                    "Accept": request.headers.get("Accept", "application/json"),
                },
            ) as upstream:
                upstream_status = upstream.status_code
                content_type = upstream.headers.get("content-type", "application/json")
                outgoing = web.StreamResponse(status=upstream.status_code, headers={"Content-Type": content_type})
                await outgoing.prepare(request)
                received = bytearray()
                async for chunk in upstream.aiter_bytes():
                    received.extend(chunk)
                    if len(received) > self.config.max_callback_bytes:
                        raise AdapterError("callback_response_too_large")
                    await outgoing.write(chunk)
                await outgoing.write_eof()
                if not 200 <= upstream.status_code < 300:
                    error_body = None
                    try:
                        error_body = strict_json_loads(bytes(received))
                    except (ValueError, UnicodeError):
                        pass
                    raise UpstreamDiagnosticError("callback_upstream_http_error", error_body)
                raw = bytes(received)
                assistant = await assistant_from_response(raw, content_type)
                reject_credentials(assistant, self.context.session_id)
                if candidate_media is not None:
                    self.media.commit(candidate_media, payload, assistant)
                if self.history is not None:
                    # Media validates before mutating; history only copies the
                    # already validated JSON, so a rejected image cannot pin it.
                    self.history.commit(payload, assistant)
                self.capture.records.append(
                    {
                        "index": index,
                        "request": payload,
                        "request_identity": "actual_forwarded",
                        "forwarded_request_canonical_sha256": hashlib.sha256(canonical_bytes(payload)).hexdigest(),
                        "raw_request": raw_payload,
                        "raw_request_sha256": hashlib.sha256(raw_body).hexdigest(),
                        "media": media_evidence,
                        "history": history_evidence,
                        "assistant": assistant,
                        "request_sha256": hashlib.sha256(body).hexdigest(),
                        "response_sha256": hashlib.sha256(raw).hexdigest(),
                        "response_bytes": len(raw),
                        "status": upstream.status_code,
                    }
                )
                return outgoing
        except (Exception, asyncio.CancelledError) as exc:
            code = exc.code if isinstance(exc, AdapterError) else "callback_forward_failed"
            self.capture.errors.append(code)
            self.capture.diagnostics.append({
                "index": index,
                "code": code,
                "http_status": upstream_status,
                "upstream": upstream_diagnostic(getattr(exc, "safe_diagnostic", None)),
            })
            if isinstance(exc, asyncio.CancelledError):
                raise
            if outgoing is not None:
                outgoing.force_close()
                return outgoing
            return web.json_response({"error": {"code": code}}, status=502)
        finally:
            self.capture.active -= 1
            if self.capture.active == 0:
                self._idle.set()
