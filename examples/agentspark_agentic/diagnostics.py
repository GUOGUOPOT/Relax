# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Allowlisted failure metadata; no messages, URLs, paths, credentials, or media."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .protocol import AdapterError


# Explicit codes owned by this adapter. Unknown additions stay generic until reviewed.
_SAFE_LOCAL_CODES = frozenset("""
agentspark_envelope_fields_invalid agentspark_envelope_version_invalid agentspark_http_error
agentspark_stream_end_missing agentspark_stream_error agentspark_transport_error attachment_input_invalid
audit_cancelled audit_credential_field_forbidden audit_execution_failed audit_failed audit_lifecycle_failed
audit_must_not_be_relax_output audit_path_already_exists audit_session_credential_forbidden
bound_user_id_invalid callback_address_ambiguous callback_assistant_empty callback_assistant_invalid
callback_capture_incomplete_or_concurrent callback_choices_invalid callback_content_after_done
callback_context_invalid callback_drain_timeout callback_duplicate_done callback_error_response
callback_finish_reason_invalid callback_forward_failed callback_forwarded_request_too_large
callback_history_not_linear callback_host_map_entry_required callback_host_map_host_invalid
callback_host_map_invalid callback_not_buffered_sse callback_observation_not_linear callback_port_invalid
callback_request_invalid callback_response_not_json callback_response_too_large callback_stream_invalid
callback_tool_call_invalid callback_tools_or_template_changed callback_upstream_http_error
capabilities_invalid client_not_open config_fields_invalid config_media_allowed_roots_invalid
config_not_object context_audit_failure_not_recoverable context_audit_schema_invalid
context_audit_stage_invalid context_callback_digest_invalid context_callback_status_invalid
context_capture_incomplete_or_concurrent context_capture_indices_invalid context_capture_invalid
context_compiler_defaults_required context_export_record_invalid context_forwarded_request_identity_invalid
context_forwarded_request_identity_required context_generation_identity_ambiguous
context_multi_export_eval_requires_reviewed_aggregation context_native_canonicalization_failed
context_reserved_output_path_required context_selection_duplicate context_selection_invalid
context_send_evidence_invalid context_source_identity_invalid context_terminal_evidence_incomplete
context_validation_compiler_defaults_mismatch context_validation_forest_required
context_validation_invocation_mismatch context_validation_invocation_required
context_validation_logical_identity_changed context_validation_native_session_required
context_validation_report_invalid context_validation_selection_mismatch context_validation_session_mismatch
diagnostics_directory_invalid diagnostics_directory_not_private diagnostics_write_failed
explicit_system_and_user_prompt_required history_concurrent_callback_unsupported history_messages_invalid
history_prefix_changed history_tool_calls_ambiguous history_tool_calls_invalid
history_tool_calls_semantic_mismatch history_tool_result_name_mismatch history_tool_result_unmapped
image_input_invalid input_item_invalid input_type_unsupported inputs_invalid invalid_sse_utf8
loader_catalog_incomplete loader_catalog_required media_allowed_roots_invalid media_allowed_roots_required
media_asset_identity_changed media_asset_invalid media_assistant_image_response_forbidden
media_binding_invalid media_candidate_stale_or_unrelated media_concurrent_callback_unsupported
media_content_invalid media_content_part_unsupported media_dynamic_image_block_invalid
media_dynamic_image_data_invalid media_dynamic_image_mime_mismatch media_dynamic_image_option_invalid
media_dynamic_image_role_invalid media_dynamic_image_scheme_forbidden media_dynamic_image_too_large
media_forwarded_images_changed media_image_format_unsupported media_image_limit_exceeded
media_local_source_required media_manifest_invalid media_manifest_path_duplicate
media_manifest_source_missing media_marker_count_mismatch media_marker_message_changed media_marker_missing
media_marker_order_invalid media_marker_outside_text media_marker_role_mismatch media_marker_unregistered
media_max_bytes_invalid media_messages_invalid media_plan_invalid media_plan_required
media_public_asset_unavailable media_seed_images_mismatch media_seed_message_changed
media_seed_message_missing media_seed_ref_unknown media_source_outside_roots media_source_size_invalid
media_source_unreadable_or_invalid media_tool_history_conflict media_tool_history_invalid
media_tool_history_required media_tool_history_too_large media_tool_identity_mismatch
media_tool_message_changed media_unregistered_image media_wire_asset_unresolved metadata_agentspark_required
non_json_value only_eden_protocol_supported production_k3_thinking_default_required
production_reasoning_invalid production_sampling_support_unsupported production_sampling_temperature_invalid
production_scoring_or_selection_failed production_template_kwargs_invalid
production_thinking_control_invalid production_unconfigured_template_kwargs
relax_input_and_output_paths_required replay_ack_session_inconsistent replay_ack_session_missing
replay_attributes_invalid replay_attributes_required replay_eof_without_ack replay_error_event
replay_invalid_ack replay_messages_invalid replay_spec_invalid replay_spec_required replay_unexpected_event
reward_already_postprocessed reward_audit_incomplete reward_callback_invalid reward_contract_invalid
reward_critic_execution_failed reward_digest_invalid reward_duplicate_json_key reward_endpoint_invalid
reward_fingerprint_invalid reward_followup_mask_invalid reward_formal_objective_mismatch
reward_formal_policy_mismatch reward_forwarded_identity_mismatch reward_forwarded_identity_required
reward_frozen_source_layout_missing reward_group_empty reward_http_failure reward_identity_invalid
reward_invocation_mismatch reward_logical_identity_duplicate reward_logical_identity_invalid
reward_nonfinite_json reward_number_invalid reward_physical_metadata_in_logical_sample
reward_presentation_policy_mismatch reward_record_selection_invalid reward_remote_policy_mismatch
reward_remote_scoring_failed reward_remove_sample_invalid reward_response_identity_mismatch
reward_response_invalid reward_rm_source_changed reward_runtime_context_invalid
reward_runtime_context_not_in_provider_prompt reward_score_binding_mismatch reward_score_identity_mismatch
reward_source_changed reward_source_config_changed reward_source_escape reward_source_metadata_missing
reward_source_operation_failed reward_source_request_not_captured reward_source_rubric_missing
reward_system_prompt_changed reward_terminal_assistant_required reward_timeout_invalid
reward_total_out_of_range reward_transport_or_json_failure reward_unified_execution_invalid
reward_visible_text_invalid reward_worker_operation_unknown reward_worker_protocol_invalid
reward_worker_unavailable rollout_mode_invalid session_lookup_http_error session_lookup_id_missing
session_lookup_not_json session_lookup_transport_error skills_invalid sse_frame_too_large text_input_invalid
text_requires_source_preparation tool_versions_invalid tools_duplicate tools_invalid training_agent_failed
training_append_roles_invalid training_cancelled training_exports_empty
training_generation_identity_ambiguous training_input_output_paths_invalid training_output_already_exists
training_reward_anchor_invalid training_reward_evidence_missing training_reward_identity_mismatch
training_reward_nonfinite training_reward_result_invalid
""".split())
_SAFE_TYPES = frozenset({
    "invalid_request_error", "authentication_error", "permission_error", "not_found_error",
    "rate_limit_error", "internal_error", "server_error", "api_error", "timeout_error",
    "client_disconnect", "BadRequestError", "InternalServerError", "TimeoutError",
    "ValueError", "RuntimeError",
})
_SAFE_REMOTE_CODES = _SAFE_TYPES | frozenset({
    "context_length_exceeded", "session_discarded", "invalid_request", "request_timeout",
    "rate_limit_exceeded", "no_available_workers", "model_not_found", "tool_execution_error",
    "agent_execution_error", "connection_error", "cancelled", "reward_invalid",
})
_REWARD_SIGNATURES = {
    "issue impacts must cover every verified root exactly once": "judge_issue_impact_coverage_invalid",
    "judge JSON/schema validation failed": "judge_schema_validation_failed",
    "no_available_workers": "no_available_workers",
}
_EXCEPTION_TYPES = frozenset({
    "AdapterError", "RewardError", "UpstreamDiagnosticError", "CancelledError", "KeyboardInterrupt",
    "ConnectError", "ConnectTimeout", "ReadError", "ReadTimeout", "WriteError", "WriteTimeout",
    "PoolTimeout", "RemoteProtocolError", "HTTPStatusError", "TimeoutError", "ValueError",
    "TypeError", "OSError", "FileNotFoundError", "PermissionError", "RuntimeError",
})
_STAGES = frozenset({"prepare", "collect", "select", "score", "attach", "publish", "cli"})


def safe_error_code(exc: BaseException, fallback: str) -> str:
    # The reward worker executes its module as a standalone isolated script;
    # importing RewardError lazily keeps that path independent of this helper.
    from .v21_reward import RewardError

    code = exc.code if isinstance(exc, AdapterError) else str(exc) if isinstance(exc, RewardError) else None
    return code if isinstance(code, str) and code in _SAFE_LOCAL_CODES else fallback


def _status(value: Any) -> bool:
    return type(value) is int and 100 <= value <= 599


def upstream_diagnostic(payload: Any, *, http_status: int | None = None) -> dict[str, Any]:
    """Inspect only known envelope keys; never retain arbitrary strings or recurse into content."""
    output: dict[str, Any] = {}
    pending = [payload]
    for _ in range(3):
        children = []
        for item in pending:
            if not isinstance(item, dict):
                continue
            for key, allowed in (("type", _SAFE_TYPES), ("code", _SAFE_REMOTE_CODES)):
                value = item.get(key)
                if isinstance(value, str) and value in allowed:
                    output.setdefault(key, value)
            error = item.get("error")
            if isinstance(error, str) and error in _SAFE_REMOTE_CODES:
                output.setdefault("code", error)
            for key in ("status", "status_code", "_http_status"):
                if _status(item.get(key)):
                    output.setdefault("status", item[key])
            if item.get("reason_code") in _REWARD_SIGNATURES.values():
                output.setdefault("reason_code", item["reason_code"])
            if _status(item.get("http_status")):
                output.setdefault("http_status", item["http_status"])
            children.extend(item.get(key) for key in ("error", "data", "detail") if isinstance(item.get(key), dict))
        pending = children
    if _status(http_status):
        output["http_status"] = http_status
    return output


def reward_http_diagnostic(payload: Any, http_status: int) -> dict[str, Any]:
    """Map known proxy failure signatures to constants, without copying the proxy's detail text."""
    output = upstream_diagnostic(payload, http_status=http_status)
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        for signature, code in _REWARD_SIGNATURES.items():
            if signature in detail:
                output["reason_code"] = code
                break
    return output


class UpstreamDiagnosticError(AdapterError):
    def __init__(self, code: str, payload: Any) -> None:
        super().__init__(code)
        self.safe_diagnostic = upstream_diagnostic(payload)


def exception_diagnostic(exc: BaseException) -> dict[str, Any]:
    output = upstream_diagnostic(getattr(exc, "safe_diagnostic", None))
    name = type(exc).__name__
    output["exception_type"] = name if name in _EXCEPTION_TYPES else "Exception"
    if exc.__cause__ is not None:
        cause = type(exc.__cause__).__name__
        output["cause_type"] = cause if cause in _EXCEPTION_TYPES else "Exception"
    return output


def failure_summary(
    exc: BaseException, *, context: Any = None, audit: dict[str, Any] | None = None, stage: str = "cli"
) -> dict[str, Any]:
    audit = audit if isinstance(audit, dict) else {}
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at_unix_ms": time.time_ns() // 1_000_000,
        "stage": stage if stage in _STAGES else "cli",
        "error": safe_error_code(exc, "training_agent_failed"),
        "diagnostic": exception_diagnostic(exc),
    }
    failure = audit.get("failure_code")
    if isinstance(failure, str) and failure in _SAFE_LOCAL_CODES:
        result["audit_failure_code"] = failure
    invocation = audit.get("invocation_sha256", getattr(context, "identity_digest", None))
    if isinstance(invocation, str) and re.fullmatch(r"[0-9a-f]{64}", invocation):
        result["invocation_sha256"] = invocation
    group = getattr(context, "group_id", None)
    if isinstance(group, str):
        if re.fullmatch(r"(?:train|eval):prepare_group_[0-9]+", group):
            result["group_id"] = group
        else:
            result["group_id_sha256"] = hashlib.sha256(group.encode()).hexdigest()
    source = audit.get("source_input")
    metadata = source.get("metadata") if isinstance(source, dict) else None
    prepared = metadata.get("agentspark_prepare") if isinstance(metadata, dict) else None
    source_line = prepared.get("source_line") if isinstance(prepared, dict) else None
    if type(source_line) is int and source_line > 0:
        result["source_line"] = source_line
    callbacks = audit.get("callbacks")
    callbacks = callbacks if isinstance(callbacks, dict) else {}
    count = callbacks.get("started")
    if type(count) is int and count >= 0:
        result["callbacks_started"] = count
    errors = callbacks.get("errors")
    errors = errors if isinstance(errors, list) else []
    result["callback_errors"] = [x for x in errors if isinstance(x, str) and x in _SAFE_LOCAL_CODES][-16:]
    upstream = []
    diagnostics = callbacks.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, list) else []
    for item in diagnostics[-16:]:
        if isinstance(item, dict):
            record = {"origin": "callback", **upstream_diagnostic(item.get("upstream"), http_status=item.get("http_status"))}
            code = item.get("code")
            if isinstance(code, str) and code in _SAFE_LOCAL_CODES:
                record["adapter_code"] = code
            if type(item.get("index")) is int and item["index"] >= 0:
                record["index"] = item["index"]
            upstream.append(record)
    stream = audit.get("stream")
    events = stream.get("events") if isinstance(stream, dict) else None
    events = events if isinstance(events, list) else []
    for item in events:
        if isinstance(item, dict) and item.get("kind") == "error":
            record = {"origin": "agentspark_stream", **upstream_diagnostic(item.get("diagnostic"))}
            digest = item.get("sha256")
            if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
                record["sha256"] = digest
            upstream.append(record)
    result["upstream_errors"] = upstream[-16:]
    return result


def prepare_diagnostics_dir(directory: str | Path) -> Path:
    path = Path(directory).expanduser()
    if path.is_symlink():
        raise AdapterError("diagnostics_directory_invalid")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077 or info.st_uid != os.geteuid():
        raise AdapterError("diagnostics_directory_not_private")
    return path


def persist_failure(
    directory: str | Path | None, exc: BaseException, *, context: Any = None,
    audit: dict[str, Any] | None = None, stage: str = "cli",
) -> Path | None:
    if directory is None or getattr(exc, "_agentspark_diagnostic_attempted", False):
        return None
    exc._agentspark_diagnostic_attempted = True
    temporary = None
    try:
        path = prepare_diagnostics_dir(directory)
        payload = failure_summary(exc, context=context, audit=audit, stage=stage)
        descriptor, temporary = tempfile.mkstemp(prefix=".failure-", dir=path)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        target = path / ("failure-" + uuid.uuid4().hex + ".json")
        os.link(temporary, target)
        return target
    except Exception:
        # Diagnostic publication must never replace the original training error.
        sys.stderr.write("AgentSpark diagnostics unavailable: diagnostics_write_failed\n")
        return None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass
