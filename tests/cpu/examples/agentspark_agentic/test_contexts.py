# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline payload identity tests; no backend generation or model claims."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from examples.agentspark_agentic.contexts import analyze_audit, main
from examples.agentspark_agentic.protocol import AdapterError


TOOLS = [{"type": "function", "function": {"name": "Search", "parameters": {"type": "object"}}}]


def _record(index, messages, *, assistant=None, tools=None, kwargs=None):
    request = {"messages": copy.deepcopy(messages), "tools": copy.deepcopy(TOOLS if tools is None else tools)}
    if kwargs is not None:
        request["chat_template_kwargs"] = kwargs
    assistant = assistant or {
        "role": "assistant",
        "content": f"answer-{index}",
        "reasoning_content": f"reason-{index}",
    }
    return {
        "index": index,
        "request": request,
        "assistant": copy.deepcopy(assistant),
        "status": 200,
        "request_sha256": hashlib.sha256(json.dumps(request).encode()).hexdigest(),
        "response_sha256": hashlib.sha256(json.dumps(assistant).encode()).hexdigest(),
    }


def _audit(records, *, failure=None, mode="train"):
    result = {
        "schema_version": 1,
        "stage": 2,
        "training_ready": False,
        "audit_status": "failed" if failure else "collected",
        "send_attempts": 1,
        "remote_execution": "completed",
        "stream": {"saw_end": True, "saw_error": False, "eof": True},
        "cleanup": {"local": "closed", "remote": "not_configured_remote_state_unknown"},
        "identity": {"group_id": "group", "model": "configured-label-only", "mode": mode},
        "source_input": {"metadata": {"rubric": "private rubric"}},
        "invocation_sha256": "e" * 64,
        "callbacks": {"records": records, "started": len(records), "errors": [], "concurrent": False},
    }
    if failure:
        result["failure_code"] = failure
    return result


def _select(index, name="main", intent="Keep this observed context"):
    return {"name": name, "record_index": index, "intent": intent}


def test_native_canonicalization_defaults_and_original_evidence_preserved():
    request = [
        {"role": "system", "content": [{"type": "text", "text": "first"}]},
        {"role": "system", "content": "second"},
        {"role": "user", "content": [{"type": "text", "text": "question"}]},
    ]
    first = _record(0, request, kwargs={"thinking": False})
    first["request"]["tools"][0]["function"]["strict"] = True
    first["request"]["temperature"] = 0.01  # Evidence retained; native API does not apply it.
    first["assistant"] = {
        "role": "assistant",
        "reasoning_content": "reason",
        "tool_calls": [
            {"id": "tool-1", "type": "function", "function": {"name": "Search", "arguments": '{"z":2, "a":1}'}}
        ],
    }
    second = _record(
        1,
        request + [first["assistant"], {"role": "tool", "tool_call_id": "tool-1", "content": "found"}],
        kwargs={"thinking": False},
    )
    source = _audit([first, second], failure="callback_history_not_linear")
    before = copy.deepcopy(source)
    report = analyze_audit(
        source, compiler_defaults={"thinking": True, "nested": {"b": 2, "a": 1}}, selection=[_select(1)]
    )
    candidate = report["records"][0]
    assert candidate["messages"][0] == {"role": "system", "content": "first\n\nsecond"}
    assert candidate["messages"][1]["content"] == "question"
    assert candidate["tools"] == TOOLS
    assert json.loads(candidate["messages"][2]["tool_calls"][0]["function"]["arguments"]) == {"a": 1, "z": 2}
    assert report["contexts"][1]["transition"] == "append"
    assert candidate["chat_template_kwargs"] == {"nested": {"a": 1, "b": 2}, "thinking": False}
    assert report["compiler_defaults"]["thinking"] is True
    assert source == before == report["source_audit"]
    assert report["source_audit"]["audit_status"] == "failed"
    assert report["source_audit"]["failure_code"] == "callback_history_not_linear"
    assert report["training_ready"] is False and report["committed_state_validation"] == "not_performed"
    assert "reward" not in candidate and set(candidate["metadata"]) == {"agentspark_context"}
    assert report["source_audit"]["callbacks"]["records"][0]["request"]["temperature"] == 0.01


def test_sequential_forks_resets_and_deleted_reasoning_need_explicit_selection():
    base = [{"role": "user", "content": "task"}]
    first = _record(0, base)
    full = base + [first["assistant"]]
    second = _record(1, full + [{"role": "user", "content": "left"}])
    third = _record(2, full + [{"role": "user", "content": "right"}])
    fourth = _record(3, base, tools=[], kwargs={"thinking": False})
    edited = copy.deepcopy(fourth["assistant"])
    edited.pop("reasoning_content")
    fifth = _record(4, base + [edited, {"role": "user", "content": "continue"}], tools=[], kwargs={"thinking": False})
    source = _audit([first, second, third, fourth, fifth], failure="callback_history_not_linear")
    pending = analyze_audit(source, compiler_defaults={"thinking": True})
    assert pending["selection_status"] == "required" and pending["records"] == []
    assert [context["transition"] for context in pending["contexts"]] == ["initial", "append", "fork", "reset", "fork"]
    assert pending["contexts"][3]["changes"] == ["tools_changed", "template_changed"]
    assert pending["contexts"][4]["changes"] == ["reasoning_removed_or_changed"]
    report = analyze_audit(
        source,
        compiler_defaults={"thinking": True},
        selection=[
            _select(1, "left", "Keep intentional left branch"),
            _select(2, "right", "Keep intentional right branch"),
        ],
    )
    assert [record["name"] for record in report["records"]] == ["left", "right"]
    assert report["requires_custom_advantage"] and report["requires_dynamic_batching"]
    assert report["contexts"][0]["observed_leaf"] is False
    assert report["training_ready"] is False


def test_media_occurrences_follow_current_positions_without_restoring_deleted_images():
    image = lambda url: {"type": "image_url", "image_url": {"url": url}}
    first = _record(
        0,
        [
            {
                "role": "user",
                "content": [
                    image("https://fixture.invalid/a"),
                    {"type": "text", "text": "middle"},
                    image("https://fixture.invalid/a"),
                    image("https://fixture.invalid/b"),
                ],
            }
        ],
    )
    second = _record(
        1, [{"role": "user", "content": [image("https://fixture.invalid/b"), image("https://fixture.invalid/a")]}]
    )
    report = analyze_audit(
        _audit([first, second], failure="callback_history_not_linear"), compiler_defaults={}, selection=[_select(1)]
    )
    before, after = [context["media_occurrences"] for context in report["contexts"]]
    assert [item["part_index"] for item in before] == [0, 2, 3]
    assert [item["part_index"] for item in after] == [0, 1]
    assert before[0]["reference_sha256"] == before[1]["reference_sha256"] == after[1]["reference_sha256"]
    assert before[2]["reference_sha256"] == after[0]["reference_sha256"]
    assert all(item["content_verified"] is False for item in before + after)
    assert len(report["records"][0]["messages"][0]["content"]) == 2
    assert report["media_content_validation"] == "not_performed"


def test_equal_native_states_are_aliases_not_additional_contexts():
    first = _record(0, [{"role": "user", "content": "task"}])
    second = _record(
        1, [{"role": "user", "content": [{"type": "text", "text": "task"}]}], assistant=first["assistant"]
    )
    second["request"]["tools"][0]["function"]["strict"] = False
    source = _audit([first, second], failure="callback_history_not_linear")
    report = analyze_audit(source, compiler_defaults={})
    assert report["contexts"][0]["state_hash"] == report["contexts"][1]["state_hash"]
    assert report["contexts"][1]["duplicate_of_record"] == 0
    with pytest.raises(AdapterError, match="generation_identity_ambiguous"):
        analyze_audit(source, compiler_defaults={}, selection=[_select(1)])


@pytest.mark.parametrize(
    "change",
    [
        "transport_failure",
        "combined_capture_failure",
        "error_event",
        "no_eof",
        "no_end",
        "cleanup_failure",
        "not_completed",
        "two_sends",
        "callback_error",
        "concurrent",
        "wrong_count",
        "bool_count",
        "bool_index",
        "gap_index",
        "duplicate_index",
        "bad_status",
        "bad_digest",
        "training_ready",
    ],
)
def test_incomplete_or_failed_audit_cannot_be_reclassified(change):
    source = _audit([_record(0, [{"role": "user", "content": "task"}])])
    capture = source["callbacks"]
    if change == "transport_failure":
        source.update(audit_status="failed", failure_code="agentspark_transport_error")
    elif change == "combined_capture_failure":
        source.update(audit_status="failed", failure_code="callback_capture_incomplete_or_concurrent")
    elif change in {"error_event", "no_eof", "no_end"}:
        key = {"error_event": "saw_error", "no_eof": "eof", "no_end": "saw_end"}[change]
        source["stream"][key] = change == "error_event"
    elif change == "cleanup_failure":
        source["cleanup"]["local"] = "close_failed"
    elif change == "not_completed":
        source["remote_execution"] = "unknown"
    elif change == "two_sends":
        source["send_attempts"] = 2
    elif change == "callback_error":
        capture["errors"] = ["callback_finish_reason_invalid"]
    elif change == "concurrent":
        capture["concurrent"] = True
    elif change == "wrong_count":
        capture["started"] = 2
    elif change == "bool_count":
        capture["started"] = True
    elif change == "bool_index":
        capture["records"][0]["index"] = False
    elif change == "gap_index":
        capture["records"][0]["index"] = 2
    elif change == "duplicate_index":
        capture["records"].append(copy.deepcopy(capture["records"][0]))
        capture["started"] = 2
    elif change == "bad_status":
        capture["records"][0]["status"] = 500
    elif change == "bad_digest":
        capture["records"][0]["response_sha256"] = "missing"
    else:
        source["training_ready"] = True
    with pytest.raises(AdapterError):
        analyze_audit(source, compiler_defaults={})


@pytest.mark.parametrize(
    "selection",
    [
        [],
        [_select(True)],
        [_select(3)],
        [_select(0, "")],
        [{"name": "main", "record_index": 0}],
        [_select(0), _select(0, "alias")],
    ],
)
def test_selection_must_be_explicit_unique_and_in_capture(selection):
    source = _audit([_record(0, [{"role": "user", "content": "task"}])])
    with pytest.raises(AdapterError, match="selection"):
        analyze_audit(source, compiler_defaults={}, selection=selection)


def test_invalid_native_payload_and_missing_defaults_fail_closed():
    source = _audit([_record(0, [{"role": "developer", "content": "must not silently rewrite"}])])
    with pytest.raises(AdapterError, match="canonicalization"):
        analyze_audit(source, compiler_defaults={})
    with pytest.raises(AdapterError, match="defaults_required"):
        analyze_audit(_audit([_record(0, [{"role": "user", "content": "task"}])]), compiler_defaults=None)


def test_eval_requires_one_representative_until_aggregation_is_reviewed():
    records = [_record(0, [{"role": "user", "content": "first"}]), _record(1, [{"role": "user", "content": "second"}])]
    source = _audit(records, failure="callback_history_not_linear", mode="eval")
    with pytest.raises(AdapterError, match="reviewed_aggregation"):
        analyze_audit(source, compiler_defaults={}, selection=[_select(0), _select(1, "other")])


def test_offline_cli_only_writes_separate_private_analysis_and_exits_three(tmp_path, monkeypatch):
    monkeypatch.delenv("RELAX_SESSION_ID", raising=False)
    audit_path, defaults_path, selection_path, output_path, reserved = [
        tmp_path / name for name in ("audit.json", "defaults.json", "selection.json", "analysis.json", "training.json")
    ]
    source = _audit([_record(0, [{"role": "user", "content": "task"}])])
    audit_path.write_text(json.dumps(source))
    defaults_path.write_text("{}")
    selection_path.write_text(json.dumps([_select(0)]))
    args = [
        "--audit",
        str(audit_path),
        "--compiler-defaults",
        str(defaults_path),
        "--selection",
        str(selection_path),
        "--audit-output",
        str(output_path),
        "--relax-output",
        str(reserved),
    ]
    assert main(args) == 3 and not reserved.exists()
    result = json.loads(output_path.read_text())
    assert result["training_ready"] is False and result["committed_state_validation"] == "not_performed"
    assert output_path.stat().st_mode & 0o777 == 0o600
    assert json.loads(audit_path.read_text()) == source
    args[args.index("--audit-output") + 1] = str(reserved)
    assert main(args) == 2 and not reserved.exists()
