# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Standalone Stage 2 protocol audit. Success exits 3: training is not ready."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sys
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

from .callback import CallbackRecorder
from .diagnostics import persist_failure, prepare_diagnostics_dir, safe_error_code, upstream_diagnostic
from .media import ASSET_PREFIX, MediaPlan
from .protocol import AdapterConfig, AdapterError, RelaxContext, json_copy, prepare_envelope, replay_body, send_body
from .result import PENDING_GATES, reject_credentials, validate_audit_path, write_audit
from .transport import AgentSparkClient


# A deployment owner may inject an already verified cleanup operation. There is
# deliberately no guessed HTTP cancellation endpoint or dynamic import in CLI.
CleanupHook = Callable[[dict[str, str]], Awaitable[None]]
AUDIT_ONLY_EXIT = 3


async def run_audit(
    config: AdapterConfig,
    context: RelaxContext,
    session_input: dict[str, Any],
    *,
    agentspark_http: httpx.AsyncClient | None = None,
    relax_http: httpx.AsyncClient | None = None,
    cleanup_hook: CleanupHook | None = None,
    stop_event: asyncio.Event | None = None,
    restore_reasoning: bool = False,
    pinned_template_kwargs: dict[str, Any] | None = None,
    pinned_temperature: float | None = None,
    allow_dynamic_images: bool = False,
) -> dict[str, Any]:
    """Run one prepared request, drain callbacks/SSE, and return non-training evidence.

    The entire group must be admitted remotely before its first callback can
    clear Relax's group barrier. This single-invocation adapter cannot establish
    remote group capacity, so it cannot by itself make a deployment ready.
    """
    envelope = prepare_envelope(session_input)
    reject_credentials(session_input, context.session_id)
    private_media = session_input.get("metadata", {}).get("agentspark_media")
    media = (
        MediaPlan(private_media, config.media_allowed_roots, allow_dynamic_images=allow_dynamic_images)
        if private_media is not None else None
    )
    if allow_dynamic_images and media is None:
        raise AdapterError("media_plan_required")
    if media is None and ASSET_PREFIX in json.dumps(envelope, ensure_ascii=False):
        raise AdapterError("media_plan_required")
    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": 2,
        "training_ready": False,
        "audit_status": "failed",
        "pending_gates": list(PENDING_GATES),
        "invocation_sha256": context.identity_digest,
        "source_input": json_copy(session_input),
        "identity": {
            "group_id": context.group_id,
            "mode": context.mode,
            "model": config.model,
        },
        "remote_execution": "not_started",
        "send_attempts": 0,
        "replay_attempted": False,
        "stream": {"events": [], "saw_end": False, "saw_error": False, "eof": False},
        "callbacks": {"records": [], "errors": []},
        "cleanup": {
            "local": "pending",
            "remote": "not_configured_remote_state_unknown",
        },
    }
    remote_session: str | None = None
    cancelled: asyncio.CancelledError | None = None
    failure: str | None = None
    reason = "failed"
    recorder = CallbackRecorder(
        config,
        context,
        client=relax_http,
        media=media,
        restore_reasoning=restore_reasoning,
        pinned_template_kwargs=pinned_template_kwargs,
        pinned_temperature=pinned_temperature,
        allow_dynamic_images=allow_dynamic_images,
    )
    remote = AgentSparkClient(config, client=agentspark_http)

    async def execute(remote: AgentSparkClient) -> None:
        nonlocal remote_session
        wire_envelope = media.wire_envelope(envelope) if media is not None else envelope
        if wire_envelope.get("replay_spec") is not None:
            report["replay_attempted"] = True
            report["remote_execution"] = "unknown"
            remote_session = await remote.restore(replay_body(config, context, wire_envelope))
            report["session_source"] = "replay_ack"
        else:
            remote_session = await remote.get_session(context)
            report["session_source"] = "get_session"
        report["remote_session_sha256"] = hashlib.sha256(remote_session.encode()).hexdigest()
        body = send_body(config, context, wire_envelope, remote_session, recorder.public_base_url)
        report["send_attempts"] = 1
        report["remote_execution"] = "unknown"
        event_capture_failure = None
        async for event in remote.events(body):
            # Preserve business snapshots for later scoring/resource validation.
            # Refuse credential-bearing events, but continue draining to EOF.
            kind = event.kind
            known = kind if kind in {"agent.stream.end", "agent.stream.error", "error", "link.up.ack"} else "other"
            evidence = {
                "kind": known,
                "sha256": hashlib.sha256(event.raw.encode()).hexdigest(),
                "bytes": len(event.raw.encode()),
            }
            if event.is_error:
                # Error text can contain arbitrary credentials and remote paths.
                evidence["kind"] = "error"
                evidence["payload_omitted"] = "remote_error_body"
                evidence["diagnostic"] = upstream_diagnostic(event.payload)
            else:
                try:
                    reject_credentials(event.event, context.session_id)
                    reject_credentials(event.payload, context.session_id)
                    evidence["event"] = event.event
                    evidence["payload"] = json_copy(event.payload)
                except AdapterError as exc:
                    event_capture_failure = exc.code
                    evidence["payload_omitted"] = "credential_check_failed"
            report["stream"]["events"].append(evidence)
            report["stream"]["saw_end"] |= kind == "agent.stream.end" or event.event == "agent.stream.end"
            report["stream"]["saw_error"] |= event.is_error
        report["stream"]["eof"] = True
        if report["stream"]["saw_error"]:
            report["remote_execution"] = "reported_error"
            raise AdapterError("agentspark_stream_error")
        if not report["stream"]["saw_end"]:
            raise AdapterError("agentspark_stream_end_missing")
        report["remote_execution"] = "completed"
        if event_capture_failure is not None:
            raise AdapterError(event_capture_failure)
        await recorder.wait_idle()

    async def run_with_stop(remote: AgentSparkClient) -> None:
        if stop_event is None:
            await execute(remote)
            return
        if stop_event.is_set():
            raise AdapterError("audit_cancelled")
        work = asyncio.create_task(execute(remote))
        stop = asyncio.create_task(stop_event.wait())
        try:
            done, _ = await asyncio.wait({work, stop}, return_when=asyncio.FIRST_COMPLETED)
            if work in done:
                await work
            else:
                raise AdapterError("audit_cancelled")
        finally:
            for task in (work, stop):
                if not task.done():
                    task.cancel()
            await asyncio.gather(work, stop, return_exceptions=True)

    try:
        async with recorder, remote:
            try:
                await run_with_stop(remote)
                reason = "completed"
            except asyncio.CancelledError as exc:
                cancelled, failure, reason = exc, "audit_cancelled", "cancelled"
            except Exception as exc:
                failure = exc.code if isinstance(exc, AdapterError) else "audit_execution_failed"
                reason = "cancelled" if failure == "audit_cancelled" else "failed"
            finally:
                if cleanup_hook is not None and remote_session is not None:
                    try:
                        await asyncio.wait_for(
                            cleanup_hook(
                                {
                                    "session_id": remote_session,
                                    "request_id": context.request_id,
                                    "user_id": context.user_id,
                                    "reason": reason,
                                }
                            ),
                            timeout=config.cleanup_timeout_s,
                        )
                        report["cleanup"]["remote"] = "hook_completed_remote_state_unverified"
                    except asyncio.TimeoutError:
                        report["cleanup"]["remote"] = "hook_timed_out_remote_state_unknown"
                    except Exception:
                        report["cleanup"]["remote"] = "hook_failed_remote_state_unknown"
    except asyncio.CancelledError as exc:
        cancelled, failure = exc, "audit_cancelled"
    except Exception as exc:
        failure = exc.code if isinstance(exc, AdapterError) else "audit_lifecycle_failed"
    finally:
        report["cleanup"]["local"] = "closed" if recorder.closed and remote.closed else "close_failed"
        report["callbacks"] = {
            "records": recorder.capture.records,
            "errors": recorder.capture.errors,
            "diagnostics": recorder.capture.diagnostics,
            "started": recorder.capture.started,
            "concurrent": recorder.capture.concurrent,
        }
    # Finalize only after the listening socket and all handlers are closed.
    # A late callback during cleanup must not leave a stale successful candidate.
    if failure is None:
        try:
            report["linear_context_candidate"] = recorder.capture.linear_candidate()
            report["audit_status"] = "collected"
        except AdapterError as exc:
            failure = exc.code
    if failure is not None:
        report["audit_status"] = "failed"
        report["failure_code"] = failure
        report.pop("linear_context_candidate", None)
    reject_credentials(report, context.session_id)
    if cancelled is not None:
        raise cancelled
    return report


async def _run_cli(
    config: AdapterConfig, context: RelaxContext, session_input: dict[str, Any], *, allow_dynamic_images: bool = False
) -> dict[str, Any]:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed = []
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop_event.set)
            installed.append(sig)
        return await run_audit(
            config, context, session_input, stop_event=stop_event, allow_dynamic_images=allow_dynamic_images
        )
    finally:
        for sig in installed:
            loop.remove_signal_handler(sig)


def main(argv: list[str] | None = None, *, env: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="explicit AgentSpark endpoint and callback configuration JSON",
    )
    parser.add_argument(
        "--audit-output",
        required=True,
        help="new private audit file; never RELAX_OUTPUT_JSON",
    )
    parser.add_argument("--diagnostics-dir", help="private directory for content-free failure summaries")
    parser.add_argument("--allow-dynamic-images", action="store_true", help="enable bounded callback image admission")
    args = parser.parse_args(argv)
    environment = os.environ if env is None else env
    context = None
    report = None
    try:
        if args.diagnostics_dir:
            prepare_diagnostics_dir(args.diagnostics_dir)
        context = RelaxContext.from_env(environment)
        input_path, output_path = environment.get("RELAX_INPUT_JSON"), environment.get("RELAX_OUTPUT_JSON")
        if not input_path or not output_path:
            raise AdapterError("relax_input_and_output_paths_required")
        validate_audit_path(args.audit_output, output_path)
        config = AdapterConfig.from_payload(json.loads(Path(args.config).read_text(encoding="utf-8")))
        session_input = json.loads(Path(input_path).read_text(encoding="utf-8"))
        prepare_envelope(session_input)
        report = asyncio.run(_run_cli(config, context, session_input, allow_dynamic_images=args.allow_dynamic_images))
        write_audit(report, args.audit_output, output_path, context.session_id)
        if report["audit_status"] != "collected":
            persist_failure(
                args.diagnostics_dir, AdapterError(report.get("failure_code", "audit_failed")),
                context=context, audit=report, stage="collect",
            )
    except (Exception, KeyboardInterrupt) as exc:
        persist_failure(args.diagnostics_dir, exc, context=context, audit=report)
        code = safe_error_code(exc, "audit_failed")
        sys.stderr.write(f"AgentSpark audit failed: {code}\n")
        return 2
    sys.stderr.write("AgentSpark audit written; training_ready=false; RELAX_OUTPUT_JSON was not published.\n")
    return AUDIT_ONLY_EXIT if report["audit_status"] == "collected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
