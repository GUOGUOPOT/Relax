# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Production agent command: real trajectory/reward -> named native exports.

This entrypoint publishes canonical contexts, never token IDs or logprobs.
Relax validates them against the live committed Forest after process success;
the configured training converter validates unique generation ownership.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import socket
import sys
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .client import run_audit
from .contexts import analyze_audit
from .diagnostics import exception_diagnostic, persist_failure, prepare_diagnostics_dir, safe_error_code
from .media import strict_json_loads
from .protocol import AdapterConfig, AdapterError, RelaxContext, json_copy
from .result import reject_credentials, validate_audit_path, write_audit


Scorer = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def configure_callback_host(config: AdapterConfig, mapping: dict[str, str]) -> AdapterConfig:
    """Resolve only an explicit host entry for this node's actual process hostname."""
    if not isinstance(mapping, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()
    ):
        raise AdapterError("callback_host_map_invalid")
    host = mapping.get(socket.gethostname())
    if (
        not host
        or host != host.strip()
        or any(char in host for char in "/@?#")
        or any(char.isspace() for char in host)
    ):
        raise AdapterError("callback_host_map_entry_required")
    if ":" in host:
        import ipaddress

        try:
            host = "[" + str(ipaddress.IPv6Address(host.strip("[]"))) + "]"
        except ValueError as exc:
            raise AdapterError("callback_host_map_host_invalid") from exc
    parsed = urlsplit(config.callback_public_base_url)
    netloc = host + (f":{parsed.port}" if parsed.port is not None else "")
    url = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    from .protocol import http_url

    return replace(config, callback_public_base_url=http_url(url, "callback_host_map_host_invalid"))


def select_training_exports(
    audit: dict[str, Any], *, compiler_defaults: dict[str, Any], append_roles: tuple[str, ...] = ("tool",)
) -> list[dict[str, Any]]:
    """Export each consecutive segment's terminal context and own each response once.

    A native fork can share earlier response nodes. Ownership excludes those
    ancestors in the new segment; the training converter must zero their loss
    mask before computing token denominators. Canonical duplicate generations
    cannot be distinguished by Forest identity and therefore fail closed.
    """
    if (
        not append_roles
        or len(set(append_roles)) != len(append_roles)
        or set(append_roles) - {"tool", "user", "system"}
    ):
        raise AdapterError("training_append_roles_invalid")
    analysis = analyze_audit(audit, compiler_defaults=compiler_defaults)
    contexts = analysis["contexts"]
    hashes = [context["state_hash"] for context in contexts]
    if len(set(hashes)) != len(hashes):
        raise AdapterError("training_generation_identity_ambiguous")
    segments: list[list[dict[str, Any]]] = []
    previous = None
    for context in contexts:
        messages = context["messages"][:-1]
        append = (
            previous is not None
            and context["transition"] == "append"
            and not context["changes"]
            and len(messages) >= len(previous["messages"])
            and messages[: len(previous["messages"])] == previous["messages"]
            and all(message["role"] in append_roles for message in messages[len(previous["messages"]) :])
        )
        if not append:
            segments.append([])
        segments[-1].append(context)
        previous = context
    mode = audit["identity"]["mode"]
    chosen = segments if mode == "train" else segments[-1:]
    selection = [
        {
            "name": f"segment-{segment[-1]['record_index']}",
            "record_index": segment[-1]["record_index"],
            "intent": "terminal context of a committed generation segment",
        }
        for segment in chosen
    ]
    records = analyze_audit(audit, compiler_defaults=compiler_defaults, selection=selection)["records"]
    for segment, record in zip(chosen, records, strict=True):
        segment_index = segments.index(segment)
        record["metadata"]["agentspark_training"] = {
            "schema_version": 1,
            "mode": mode,
            "segment_index": segment_index,
            "segment_count": len(segments),
            "is_anchor": segment_index == len(segments) - 1,
            "record_start": segment[0]["record_index"],
            "record_stop": segment[-1]["record_index"] + 1,
            "owned_resp_state_hashes": [context["state_hash"] for context in segment],
            "all_resp_state_hashes": hashes,
            "invocation_sha256": audit["invocation_sha256"],
            "append_roles": list(append_roles),
        }
    return records


def attach_reward(records: list[dict[str, Any]], result: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach one real reward anchor; all contexts carry the scoring evidence."""
    if not isinstance(result, dict) or set(result) != {
        "base_reward",
        "metadata",
        "identity",
        "source_identity",
        "invocation_sha256",
        "prompt_bundle_fingerprint",
        "record_policy",
        "reward_record_indices",
    }:
        raise AdapterError("training_reward_result_invalid")
    reward = result["base_reward"]
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
        or not 0 <= reward <= 1
    ):
        raise AdapterError("training_reward_nonfinite")
    if any(
        not isinstance(result[key], dict) or not result[key] for key in ("metadata", "identity", "source_identity")
    ):
        raise AdapterError("training_reward_evidence_missing")
    anchor_metadata = [
        record["metadata"]["agentspark_training"]
        for record in records
        if record["metadata"]["agentspark_training"]["is_anchor"]
    ]
    if len(anchor_metadata) != 1:
        raise AdapterError("training_reward_anchor_invalid")
    anchor = anchor_metadata[0]
    fingerprint = result["prompt_bundle_fingerprint"]
    if (
        result["invocation_sha256"] != anchor["invocation_sha256"]
        or any(
            record["metadata"]["agentspark_training"]["invocation_sha256"] != result["invocation_sha256"]
            for record in records
        )
        or result["record_policy"] != "final_active_segment_v1"
        or result["reward_record_indices"] != list(range(anchor["record_start"], anchor["record_stop"]))
        or not isinstance(result["reward_record_indices"], list)
        or any(type(index) is not int for index in result["reward_record_indices"])
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 16
        or any(char not in "0123456789abcdef" for char in fingerprint)
    ):
        raise AdapterError("training_reward_identity_mismatch")
    output = json_copy(records)
    anchors = 0
    for record in output:
        record["metadata"]["agentspark_reward"] = json_copy(result)
        if record["metadata"]["agentspark_training"]["is_anchor"]:
            record["reward"] = reward
            anchors += 1
        else:
            # Sparse terminal reward from the source policy, only after RM success.
            record["reward"] = 0.0
    if anchors != 1:
        raise AdapterError("training_reward_anchor_invalid")
    return output


def publish_training_output(records: list[dict[str, Any]], output_path: str | Path, *, credential: str) -> Path:
    """Atomically create JSONL without overwriting an existing runtime output."""
    from relax.agentic.runner.ipc import SessionOutput

    path = Path(output_path)
    if path.exists() or path.is_symlink():
        raise AdapterError("training_output_already_exists")
    if not records:
        raise AdapterError("training_exports_empty")
    normalized = SessionOutput.from_records(records).records
    # reject_credentials visits JSON containers, so convert the native tuple.
    reject_credentials(list(normalized), credential)
    data = "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".agentspark-training-", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise AdapterError("training_output_already_exists") from exc
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return path


async def run_training(
    config: AdapterConfig,
    context: RelaxContext,
    session_input: dict[str, Any],
    *,
    compiler_defaults: dict[str, Any],
    scorer: Scorer,
    audit_path: str | Path,
    output_path: str | Path,
    append_roles: tuple[str, ...] = ("tool",),
    sampling_temperature: float = 0.8,
    diagnostics_dir: str | Path | None = None,
    allow_dynamic_images: bool = False,
    **execution: Any,
) -> dict[str, Any]:
    """Collect/score once, persist audit, then publish; errors never publish output."""
    validate_audit_path(audit_path, output_path)
    if Path(output_path).exists() or Path(output_path).is_symlink():
        raise AdapterError("training_output_already_exists")
    try:
        audit = await run_audit(
            config,
            context,
            session_input,
            restore_reasoning=True,
            pinned_template_kwargs=compiler_defaults,
            pinned_temperature=sampling_temperature,
            allow_dynamic_images=allow_dynamic_images,
            **execution,
        )
    except BaseException as exc:
        persist_failure(diagnostics_dir, exc, context=context, audit={"source_input": session_input}, stage="collect")
        raise
    phase = "select"
    try:
        records = select_training_exports(audit, compiler_defaults=compiler_defaults, append_roles=append_roles)
        reward_audit = json_copy(audit)
        anchor = next(
            record["metadata"]["agentspark_training"]
            for record in records
            if record["metadata"]["agentspark_training"]["is_anchor"]
        )
        reward_audit["reward_record_indices"] = list(range(anchor["record_start"], anchor["record_stop"]))
        reward_audit["reward_record_policy"] = "final_active_segment_v1"
        stop_event = execution.get("stop_event")
        if stop_event is not None and stop_event.is_set():
            raise AdapterError("training_cancelled")
        phase = "score"
        if stop_event is None:
            reward = await scorer(reward_audit)
        else:
            work, stopped = asyncio.create_task(scorer(reward_audit)), asyncio.create_task(stop_event.wait())
            try:
                done, _ = await asyncio.wait({work, stopped}, return_when=asyncio.FIRST_COMPLETED)
                if stopped in done or stop_event.is_set():
                    raise AdapterError("training_cancelled")
                reward = await work
            finally:
                for task in (work, stopped):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(work, stopped, return_exceptions=True)
        phase = "attach"
        records = attach_reward(records, reward)
        reject_credentials(records, context.session_id)
    except BaseException as exc:
        audit["production"] = {
            "status": "failed",
            "error": safe_error_code(exc, "production_scoring_or_selection_failed"),
            "diagnostic": exception_diagnostic(exc),
        }
        persist_failure(diagnostics_dir, exc, context=context, audit=audit, stage=phase)
        write_audit(audit, audit_path, output_path, context.session_id)
        raise
    audit["production"] = {
        "status": "scored_awaiting_native_validation",
        "reward_record_policy": reward_audit["reward_record_policy"],
        "reward_record_indices": reward_audit["reward_record_indices"],
        "exports": [
            {"name": record["name"], "ownership": record["metadata"]["agentspark_training"]} for record in records
        ],
        "reward_evidence": reward,
    }
    write_audit(audit, audit_path, output_path, context.session_id)
    publish_training_output(records, output_path, credential=context.session_id)
    return {"records": records, "audit": audit, "committed_validation": "performed_by_native_runtime_after_exit"}


async def _run_cli(
    config: AdapterConfig, context: RelaxContext, session_input: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    from .v21_reward import load_reward_contract, score_audit

    contract = load_reward_contract(args.reward_contract)
    defaults = strict_json_loads(Path(args.compiler_defaults).read_bytes())
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed = []
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)
            installed.append(sig)

        async def score(audit: dict[str, Any]) -> dict[str, Any]:
            return await score_audit(audit, contract=contract, reward_base_url=args.reward_base_url)

        return await run_training(
            config,
            context,
            session_input,
            compiler_defaults=defaults,
            scorer=score,
            audit_path=args.audit_output or Path(context.invocation_id) / "agentspark-audit.json",
            output_path=args.output_path,
            append_roles=tuple(args.append_roles),
            stop_event=stop_event,
            sampling_temperature=args.sampling_temperature,
            diagnostics_dir=args.diagnostics_dir,
            allow_dynamic_images=args.allow_dynamic_images,
        )
    finally:
        for sig in installed:
            loop.remove_signal_handler(sig)


def main(argv: list[str] | None = None, *, env: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="AgentSpark AdapterConfig JSON")
    parser.add_argument("--compiler-defaults", required=True, help="exact native compiler template defaults JSON")
    parser.add_argument("--reward-contract", required=True, help="validated frozen V21 reward contract JSON")
    parser.add_argument("--reward-base-url", required=True, help="explicit V21 reward proxy base URL")
    parser.add_argument(
        "--sampling-temperature", type=float, default=0.8, help="must equal native rollout temperature"
    )
    parser.add_argument("--append-roles", nargs="+", choices=("tool", "user", "system"), default=["tool"])
    parser.add_argument("--callback-host-map", help="explicit process-hostname to AgentSpark-reachable host JSON map")
    parser.add_argument("--audit-output", help="new private audit path; default is in the Session IO directory")
    parser.add_argument("--diagnostics-dir", help="private directory for content-free persistent failure summaries")
    parser.add_argument("--allow-dynamic-images", action="store_true", help="enable bounded callback image admission")
    args = parser.parse_args(argv)
    environment = os.environ if env is None else env
    context = None
    session_input = None
    try:
        if args.diagnostics_dir:
            prepare_diagnostics_dir(args.diagnostics_dir)
        context = RelaxContext.from_env(environment)
        input_path, args.output_path = environment.get("RELAX_INPUT_JSON"), environment.get("RELAX_OUTPUT_JSON")
        if not input_path or not args.output_path or Path(input_path).resolve() == Path(args.output_path).resolve():
            raise AdapterError("training_input_output_paths_invalid")
        config = AdapterConfig.from_payload(strict_json_loads(Path(args.config).read_bytes()))
        if args.callback_host_map:
            config = configure_callback_host(config, strict_json_loads(Path(args.callback_host_map).read_bytes()))
        session_input = strict_json_loads(Path(input_path).read_bytes())
        asyncio.run(_run_cli(config, context, session_input, args))
    except (Exception, KeyboardInterrupt) as exc:
        persist_failure(
            args.diagnostics_dir, exc, context=context,
            audit={"source_input": session_input} if session_input is not None else None,
        )
        code = safe_error_code(exc, "training_agent_failed")
        sys.stderr.write(f"AgentSpark training agent failed: {code}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
