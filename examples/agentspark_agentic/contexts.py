# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline context candidates and native export validation; never training publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from .media import canonical_bytes
from .protocol import AdapterError, json_copy
from .result import PENDING_GATES, reject_credentials, write_audit


TOPOLOGY_FAILURES = {
    "callback_tools_or_template_changed",
    "callback_history_not_linear",
    "callback_observation_not_linear",
}


def _complete_records(audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept complete Stage 2 evidence, including specifically identified topology failures."""
    if not isinstance(audit, dict) or type(audit.get("schema_version")) is not int or audit["schema_version"] != 1:
        raise AdapterError("context_audit_schema_invalid")
    if type(audit.get("stage")) is not int or audit["stage"] != 2 or audit.get("training_ready") is not False:
        raise AdapterError("context_audit_stage_invalid")
    status, failure = audit.get("audit_status"), audit.get("failure_code")
    if not (
        (status == "collected" and failure is None)
        or (status == "failed" and isinstance(failure, str) and failure in TOPOLOGY_FAILURES)
    ):
        raise AdapterError("context_audit_failure_not_recoverable")
    if type(audit.get("send_attempts")) is not int or audit["send_attempts"] != 1:
        raise AdapterError("context_send_evidence_invalid")
    stream, cleanup = audit.get("stream"), audit.get("cleanup")
    if (
        audit.get("remote_execution") != "completed"
        or not isinstance(stream, dict)
        or stream.get("saw_end") is not True
        or stream.get("eof") is not True
        or stream.get("saw_error") is not False
        or not isinstance(cleanup, dict)
        or cleanup.get("local") != "closed"
    ):
        raise AdapterError("context_terminal_evidence_incomplete")
    identity = audit.get("identity")
    if (
        not isinstance(identity, dict)
        or any(not isinstance(identity.get(key), str) or not identity[key] for key in ("group_id", "model", "mode"))
        or identity["mode"] not in {"train", "eval"}
        or not isinstance(audit.get("source_input"), dict)
    ):
        raise AdapterError("context_source_identity_invalid")
    capture = audit.get("callbacks")
    if not isinstance(capture, dict):
        raise AdapterError("context_capture_invalid")
    records, started = capture.get("records"), capture.get("started")
    if (
        capture.get("errors") != []
        or capture.get("concurrent") is not False
        or not isinstance(records, list)
        or not records
        or type(started) is not int
        or started != len(records)
        or any(not isinstance(record, dict) or type(record.get("index")) is not int for record in records)
    ):
        raise AdapterError("context_capture_incomplete_or_concurrent")
    if sorted(record["index"] for record in records) != list(range(started)):
        raise AdapterError("context_capture_indices_invalid")
    for record in records:
        if type(record.get("status")) is not int or not 200 <= record["status"] < 300:
            raise AdapterError("context_callback_status_invalid")
        if record.get("request_identity") is not None:
            if (
                record["request_identity"] != "actual_forwarded"
                or record.get("forwarded_request_canonical_sha256")
                != hashlib.sha256(canonical_bytes(record.get("request"))).hexdigest()
            ):
                raise AdapterError("context_forwarded_request_identity_invalid")
        elif audit["source_input"].get("metadata", {}).get("agentspark_media") is not None:
            raise AdapterError("context_forwarded_request_identity_required")
        for field in ("request_sha256", "response_sha256"):
            digest = record.get(field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise AdapterError("context_callback_digest_invalid")
    return sorted(records, key=lambda record: record["index"])


def _media_occurrences(messages: list[dict[str, Any]], record_index: int) -> list[dict[str, Any]]:
    occurrences = []
    for message_index, message in enumerate(messages):
        content = message["content"]
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if part["type"] == "image_url":
                reference = part["image_url"]["url"]
                occurrences.append(
                    {
                        "record_index": record_index,
                        "message_index": message_index,
                        "part_index": part_index,
                        "ordinal": len(occurrences),
                        "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
                        "content_verified": False,
                    }
                )
    return occurrences


def _reasoning_changed(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> bool:
    for before, after in zip(previous, current):
        if before["role"] != "assistant" or after["role"] != "assistant" or not before.get("reasoning_content"):
            continue
        if before.get("reasoning_content") == after.get("reasoning_content"):
            continue
        if {key: value for key, value in before.items() if key != "reasoning_content"} == {
            key: value for key, value in after.items() if key != "reasoning_content"
        }:
            return True
    return False


def analyze_audit(
    audit: dict[str, Any],
    *,
    compiler_defaults: dict[str, Any],
    selection: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project observed payloads and an explicit selection without creating a Forest.

    Hashes describe native canonical identity, not proof that a remote Session
    committed that identity. No generation tokens, probabilities or reward are
    inferred. Compiler defaults are an explicit caller assertion until checked
    against a live native Session by validate_native_exports.
    """
    from relax.agentic.runner.ipc import SessionOutput
    from relax.agentic.session.service import AgenticChatRequestError, _normalized_chat_request
    from relax.agentic.session.state import (
        _messages_tools_template_state_hash,
        check_messages,
        normalize_template_kwargs,
    )

    records = _complete_records(audit)
    reject_credentials(audit, "")
    if not isinstance(compiler_defaults, dict):
        raise AdapterError("context_compiler_defaults_required")
    defaults = normalize_template_kwargs(json_copy(compiler_defaults))
    reject_credentials(defaults, "")
    if selection is not None and (not isinstance(selection, list) or not selection):
        raise AdapterError("context_selection_invalid")
    observed_states: dict[str, dict[str, Any]] = {}
    leaves: set[str] = set()
    contexts = []
    previous = None
    for record in records:
        try:
            if not isinstance(record.get("request"), dict):
                raise ValueError("request must be an object")
            request = _normalized_chat_request(json_copy(record["request"]))
            assistant = check_messages([record.get("assistant")])
            if len(assistant) != 1 or assistant[0]["role"] != "assistant":
                raise ValueError("one assistant response required")
            messages, tools = request["messages"], request["tools"]
            effective_kwargs = normalize_template_kwargs({**defaults, **request["chat_template_kwargs"]})
            complete_messages = messages + assistant
        except (TypeError, ValueError, KeyError, RuntimeError, AgenticChatRequestError) as exc:
            raise AdapterError("context_native_canonicalization_failed") from exc
        parent = None
        for length in range(len(messages), 0, -1):
            state_hash = _messages_tools_template_state_hash(messages[:length], tools, effective_kwargs)
            if state_hash in observed_states:
                parent = state_hash
                break
        request_hash = _messages_tools_template_state_hash(messages, tools, effective_kwargs)
        response_hash = _messages_tools_template_state_hash(complete_messages, tools, effective_kwargs)
        changes = []
        if previous is not None:
            if tools != previous["tools"]:
                changes.append("tools_changed")
            if effective_kwargs != previous["effective_template_kwargs"]:
                changes.append("template_changed")
            if _reasoning_changed(previous["messages"], messages):
                changes.append("reasoning_removed_or_changed")
        transition = (
            "initial"
            if previous is None
            else ("reset" if parent is None else ("append" if parent == previous["state_hash"] else "fork"))
        )
        if request_hash not in observed_states:
            observed_states[request_hash] = {"kind": "observation", "record_index": record["index"]}
            leaves.add(request_hash)
            leaves.discard(parent)
        duplicate_of = observed_states.get(response_hash)
        if duplicate_of is None:
            observed_states[response_hash] = {"kind": "response", "record_index": record["index"]}
            leaves.add(response_hash)
            leaves.discard(request_hash)
        context = {
            "record_index": record["index"],
            "state_hash": response_hash,
            "request_state_hash": request_hash,
            "matched_observed_parent": parent,
            "transition": transition,
            "changes": changes,
            "duplicate_of_record": duplicate_of["record_index"] if duplicate_of else None,
            "messages": complete_messages,
            "tools": tools,
            "request_template_kwargs": request["chat_template_kwargs"],
            "effective_template_kwargs": effective_kwargs,
            "media_occurrences": _media_occurrences(complete_messages, record["index"]),
        }
        contexts.append(context)
        previous = context
    for context in contexts:
        context["observed_leaf"] = context["state_hash"] in leaves
    by_index = {context["record_index"]: context for context in contexts}
    chosen = []
    names, states = set(), set()
    for item in selection or []:
        if not isinstance(item, dict) or set(item) != {"name", "record_index", "intent"}:
            raise AdapterError("context_selection_invalid")
        name, index, intent = item["name"], item["record_index"], item["intent"]
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or type(index) is not int
            or index not in by_index
            or not isinstance(intent, str)
            or not intent.strip()
        ):
            raise AdapterError("context_selection_invalid")
        context = by_index[index]
        if name in names or context["state_hash"] in states:
            raise AdapterError("context_selection_duplicate")
        # Native Forest deduplicates equal canonical states. Different recorded
        # generations for one state cannot establish which tokens were retained.
        if sum(other["state_hash"] == context["state_hash"] for other in contexts) != 1:
            raise AdapterError("context_generation_identity_ambiguous")
        names.add(name)
        states.add(context["state_hash"])
        chosen.append(
            {
                "name": name,
                "messages": context["messages"],
                "tools": context["tools"],
                "chat_template_kwargs": context["effective_template_kwargs"],
                "metadata": {
                    "agentspark_context": {
                        "record_index": index,
                        "state_hash": context["state_hash"],
                        "intent": intent,
                    }
                },
            }
        )
    if audit["identity"]["mode"] == "eval" and len(chosen) > 1:
        raise AdapterError("context_multi_export_eval_requires_reviewed_aggregation")
    try:
        normalized = SessionOutput.from_records(chosen).records
    except (TypeError, ValueError) as exc:
        raise AdapterError("context_export_record_invalid") from exc
    report = {
        "schema_version": 1,
        "stage": 3,
        "training_ready": False,
        "source_audit": json_copy(audit),
        "compiler_defaults": defaults,
        "selection": json_copy(selection),
        "contexts": contexts,
        "records": list(normalized),
        "selection_status": "explicit" if chosen else "required",
        "committed_state_validation": "not_performed",
        "media_content_validation": "not_performed",
        "requires_custom_advantage": len(chosen) > 1,
        "requires_dynamic_batching": len(chosen) > 1,
        "pending_gates": list(PENDING_GATES),
    }
    reject_credentials(report, "")
    return report


def validate_native_exports(report: dict[str, Any], *, shard: Any, session: Any, invocation_id: str) -> Any:
    """Validate against the supplied native Session; never construct state from an audit.

    Returns the native transport for caller-owned inspection. This in-process
    check does not expose a remote API, publish a training file, supply scores,
    or mark the audit as training-ready. Test backends remain test evidence.
    """
    from relax.agentic.runner.ipc import SessionOutput
    from relax.agentic.session.service import AgenticSessionShard, _SessionRecord
    from relax.agentic.session.state import SessionForest, normalize_template_kwargs

    native_shard = AgenticSessionShard.__ray_metadata__.modified_class
    if not isinstance(shard, native_shard) or not isinstance(session, _SessionRecord):
        raise AdapterError("context_validation_native_session_required")
    if not isinstance(invocation_id, str) or not invocation_id:
        raise AdapterError("context_validation_invocation_required")
    if not isinstance(report, dict) or report.get("stage") != 3 or report.get("training_ready") is not False:
        raise AdapterError("context_validation_report_invalid")
    reject_credentials(report, session.session_id)
    expected = analyze_audit(
        report["source_audit"], compiler_defaults=report["compiler_defaults"], selection=report["selection"]
    )
    if not expected["records"] or report.get("records") != expected["records"]:
        raise AdapterError("context_validation_selection_mismatch")
    invocation_digest = hashlib.sha256(f"{session.session_id}\0{invocation_id}".encode()).hexdigest()
    if report["source_audit"].get("invocation_sha256") != invocation_digest:
        raise AdapterError("context_validation_invocation_mismatch")
    identity = report["source_audit"]["identity"]
    if identity["group_id"] != session.group.group_id or identity["mode"] != session.group.rollout_mode:
        raise AdapterError("context_validation_session_mismatch")
    if (
        normalize_template_kwargs(shard._generation_backend.compiler.apply_chat_template_kwargs)
        != expected["compiler_defaults"]
    ):
        raise AdapterError("context_validation_compiler_defaults_mismatch")
    if not isinstance(session.forest, SessionForest):
        raise AdapterError("context_validation_forest_required")
    if session.forest.session_id != session.session_id:
        raise AdapterError("context_validation_session_mismatch")
    records = SessionOutput.from_records(expected["records"]).records
    transport = shard._build_session_export_transport(
        session=session,
        reward=None,
        metadata={},
        output_records=records,
        finalize_events={},
    )
    for export in transport.exports:
        sample = export["sample_payload"]
        if sample["index"] != session.forest.index or sample["group_index"] != session.forest.group_index:
            raise AdapterError("context_validation_logical_identity_changed")
    return transport


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, help="complete private Stage 2 audit JSON")
    parser.add_argument("--compiler-defaults", required=True, help="explicit apply_chat_template_kwargs JSON object")
    parser.add_argument("--selection", help="JSON list of name, record_index and intent; omission selects nothing")
    parser.add_argument("--audit-output", required=True, help="new private Stage 3 analysis file")
    parser.add_argument(
        "--relax-output", default=os.environ.get("RELAX_OUTPUT_JSON"), help="reserved training output path"
    )
    args = parser.parse_args(argv)
    try:
        if not args.relax_output:
            raise AdapterError("context_reserved_output_path_required")
        audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
        defaults = json.loads(Path(args.compiler_defaults).read_text(encoding="utf-8"))
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8")) if args.selection else None
        report = analyze_audit(audit, compiler_defaults=defaults, selection=selection)
        write_audit(report, args.audit_output, args.relax_output, os.environ.get("RELAX_SESSION_ID", ""))
    except Exception as exc:
        code = exc.code if isinstance(exc, AdapterError) else "context_analysis_failed"
        sys.stderr.write(f"AgentSpark context analysis failed: {code}\n")
        return 2
    sys.stderr.write("AgentSpark context analysis written; committed state unverified; training_ready=false.\n")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
