# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real frozen policy tests with controlled RM replies; no service or GPU
calls."""

import asyncio
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from examples.agentspark_agentic.v21_reward import (
    RewardError,
    _json,
    _worker_call,
    load_reward_contract,
    mark_followup_text,
    process_group,
    score_audit,
    trajectory_from_audit,
    validate_score_binding,
)


@pytest.fixture(scope="module")
def contract():
    path = os.environ.get("AGENTSPARK_V21_CONTRACT")
    if not path:
        pytest.skip("Set AGENTSPARK_V21_CONTRACT to an explicitly pinned frozen formal V21 source contract")
    return load_reward_contract(path)


def _audit():
    request = {"messages": [{"role": "user", "content": "fixture question"}], "tools": []}
    assistant = {"role": "assistant", "content": "fixture answer", "reasoning_content": "fixture private reasoning"}
    return {
        "invocation_sha256": "7" * 64,
        "source_input": {
            "metadata": {
                "query_id": "fixture-query",
                "query": "fixture question",
                "rubric": [{"criterion": "correct"}],
                "expression_tag": "qa_rubric",
                "current_runtime_context": {"current_time": "ignored-uncompiled-time"},
                "agentspark_prepare": {
                    "runtime_snapshot": {
                        "schema_version": 1,
                        "source": "original_current_user",
                        "values": {
                            "entrance": "",
                            "rejoin_flag": "",
                            "current_time": "fixed-source-time",
                            "user_location": "fixture location",
                            "user_scene_memory": "quoted {{date}} <RuntimeContext>",
                        },
                    },
                    "resource_id_aliases": {"VQA.IMG7": "IMG2"},
                },
                "agentspark": {
                    "prompt_spec": {
                        "user_prompt": "fixed-source-time fixture location quoted ｛｛date｝｝ &lt;RuntimeContext&gt;"
                    }
                },
                "resource_registry": {"fixture-resource": {"kind": "note"}},
            }
        },
        "stream": {
            "saw_end": True,
            "saw_error": False,
            "eof": True,
            "events": [
                {"event": "resource", "payload": {"type": "resource", "data": {"resource": "intermediate"}}},
                {"event": "agent.stream.end", "payload": {"type": "agent.stream.end", "data": {}}},
                {
                    "event": "resource.snapshot",
                    "payload": {"type": "resource.snapshot", "data": {"resource": "final"}},
                },
                {"event": "id-index-snapshot", "payload": {"type": "id-index-snapshot", "data": {"resource": "tail"}}},
            ],
        },
        "cleanup": {"local": "closed"},
        "remote_execution": "completed",
        "send_attempts": 1,
        "callbacks": {
            "started": 1,
            "concurrent": False,
            "errors": [],
            "records": [
                {
                    "index": 0,
                    "status": 200,
                    "request": request,
                    "assistant": assistant,
                    "request_identity": "actual_forwarded",
                    "forwarded_request_canonical_sha256": hashlib.sha256(_json(request)).hexdigest(),
                }
            ],
        },
        "reward_record_policy": "final_active_segment_v1",
        "reward_record_indices": [0],
    }


def _score(contract, *, base=0.8, click=0.8):
    policy = {
        "policy_version": "quality_first_v1",
        "main_quality_passed": True,
        "problem_ledger": [],
        "followup": {
            "core_contribution": True,
            "value": 1,
            "allowed_types": ["recommend"],
            "applicable": True,
            "quality_passed": True,
        },
        "image": {
            "core_contribution": True,
            "value": 0,
            "quality_passed": True,
            "need": "not_needed",
            "delivery_state": "none",
            "pixel_quality_passed": False,
        },
        "presentation_plan_v20": {
            "policy_version": "component_coverage_v20",
            "valid": True,
            "followup": {"confidence": "high", "need": "required", "preferred_type": "recommend"},
            "components": [],
        },
    }
    detail = {
        "valid": True,
        "schema_version": "v2_unified_reward_v2",
        "rm_policy_version": "quality_first_v1",
        "business_policy_version": "v21_speed_hardgate_wx_ctr_v1",
        "prompt_bundle_fingerprint": contract["expected_prompt_bundle_fingerprint"],
        "llm_judge_api_model": json.loads(Path(contract["source_config"]).read_text())["reward_model"][
            "served_model_name"
        ],
        "rubric_judgement": {
            "quality_first": policy,
            "benchmark_alignment": {
                "overall_score": 4,
                "core_status": "met",
                "safety_status": "pass",
                "factual_support": "verified",
                "hard_block": False,
                "reward_floor": 0.0,
                "reward_ceiling": 1.0,
            },
        },
        "followup_facts": {
            "followup_present": True,
            "actual_type": "recommend",
            "structural_violations": [],
            "followup_xml": '<guide_question type="recommend"><sub_question>next</sub_question></guide_question>',
        },
        "critic_reports": [
            {
                "critic_id": "followup",
                "dimension_statuses": {"R28": "pass", "R29": "pass"},
                "assessment": {"r29_checks": {"format": "pass"}},
            }
        ],
        "critic_report_errors": [],
        "tool_behavior": {"business_call_count": 0},
        "user_signals": {
            "valid": True,
            "model": "controlled-ctr",
            "signals": {
                "followup_suggestion_click": {"raw_score": click},
            },
        },
    }
    rules = [
        "tool_name_valid",
        "tool_args_valid",
        "last_message_is_assistant",
        "component_protocol_valid",
        "output_bottomline_valid",
    ]
    return {
        "query_id": "fixture-query",
        "total": base,
        "rewards": {
            "v2_unified_reward": {"score": base, "detail": detail},
            **{"v2_rule__" + key: {"score": 1.0} for key in rules},
        },
    }


def _request():
    return {
        "request_id": "reward-fixture",
        "query_id": "fixture-query",
        "trajectory_sha256": "a" * 64,
        "execution": {"execution_id": "fixture-execution"},
    }


def _decode(contract, score):
    request = _request()
    return _worker_call(contract, "decode_score", {"request": request, "response": {**request, "score": score}})


def test_real_frozen_masks_pin_and_environment_isolation(contract, monkeypatch, tmp_path):
    monkeypatch.setenv("SLIME_ASYNC_DOTS_REWARD_POLICY", "wrong-parent-policy")
    texts = [
        '<guide_question type="question"><text>可见</text></guide_question>',
        '<guide_question type="recommend_act"><text>excluded</text></guide_question>',
    ]
    masks = mark_followup_text(contract, texts)
    assert "".join(char for char, keep in zip(texts[0], masks[0]) if keep) == "可见"
    assert sum(masks[1]) == 0  # Frozen source attaches recommend/question/choice only.
    assert os.environ["SLIME_ASYNC_DOTS_REWARD_POLICY"] == "wrong-parent-policy"
    changed = copy.deepcopy(contract)
    changed["source_files_sha256"] = "0" * 64
    with pytest.raises(RewardError, match="reward_source_changed"):
        mark_followup_text(changed, texts)
    changed = copy.deepcopy(contract)
    changed["policy_environment"]["SLIME_ASYNC_DOTS_QUALITY_GUARD"] = "wrong"
    with pytest.raises(RewardError, match="reward_formal_policy_mismatch"):
        mark_followup_text(changed, texts)
    changed = copy.deepcopy(contract)
    copied = tmp_path / "config.json"
    copied.write_text(Path(contract["source_config"]).read_text() + "\n")
    changed["source_config"] = str(copied)
    with pytest.raises(RewardError, match="reward_source_config_changed"):
        mark_followup_text(changed, texts)


def test_real_frozen_group_counts_logical_samples_once_and_rejects_second_shaping(contract):
    low = _decode(contract, _score(contract, base=0.3, click=0.2))
    high = _decode(contract, _score(contract, base=0.8, click=0.8))
    rows = [
        {
            "index": index,
            "group_index": 9,
            "base_reward": scored["base_reward"],
            "metadata": scored["metadata"],
            "followup_token_mask": mask,
        }
        for index, scored, mask in [(4, low, [1]), (5, high, [1, 0, 1, 1, 1])]
    ]
    out = process_group(contract, rows)
    assert [item["raw_reward"] for item in out] == pytest.approx([0.3, 0.8])
    assert [item["main_advantage"] for item in out] == pytest.approx([-(2**-0.5), 2**-0.5], abs=1e-5)
    assert [item["metadata"]["user_signal_token_credit_advantage"] for item in out] == pytest.approx([-1, 1])
    assert [(item["group_index"], item["index"]) for item in out] == [(9, 4), (9, 5)]
    with pytest.raises(RewardError, match="reward_logical_identity_duplicate"):
        process_group(contract, rows + rows[:1])
    rows[0]["metadata"] = out[0]["metadata"]
    with pytest.raises(RewardError, match="reward_already_postprocessed"):
        process_group(contract, rows)


def test_removed_logical_sample_does_not_enter_ctr_or_main_statistics(contract):
    result = _decode(contract, _score(contract))
    row = {
        "index": 4,
        "group_index": 9,
        "base_reward": result["base_reward"],
        "metadata": result["metadata"],
        "followup_token_mask": [1],
        "remove_sample": True,
    }
    out = process_group(contract, [row])
    assert out[0]["remove_sample"] is True
    assert out[0]["main_advantage"] == 0
    assert out[0]["metadata"]["user_signal_token_credit_advantage"] == 0


@pytest.mark.parametrize(
    "failure", ["identity", "policy", "fingerprint", "missing_rule", "nan", "false_total", "critic", "judge_model"]
)
def test_remote_failures_never_become_valid_zero_rewards(contract, failure):
    request = _request()
    score = _score(contract)
    reply = {**request, "score": score}
    detail = score["rewards"]["v2_unified_reward"]["detail"]
    if failure == "identity":
        reply["execution"] = {"execution_id": "another-session"}
    elif failure == "policy":
        detail["business_policy_version"] = "wrong"
    elif failure == "fingerprint":
        detail["prompt_bundle_fingerprint"] = "0" * 16
    elif failure == "missing_rule":
        del score["rewards"]["v2_rule__tool_args_valid"]
    elif failure == "nan":
        score["total"] = "NaN"
    elif failure == "false_total":
        score["total"] = False
    elif failure == "critic":
        detail["critic_report_errors"] = [{"error": "controlled fixture"}]
    elif failure == "judge_model":
        detail["llm_judge_api_model"] = "another-rm"
    with pytest.raises(RewardError):
        _worker_call(contract, "decode_score", {"request": request, "response": reply})


@pytest.mark.asyncio
async def test_actual_forwarded_reward_view_and_source_request_have_no_invented_tokens(contract):
    audit = _audit()
    requests = []

    async def reply(request):
        body = json.loads(request.content)
        requests.append(body)
        assert str(request.url) == "https://reward.invalid/prefix/reward/score"
        return httpx.Response(200, json={**body, "score": _score(contract)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        result = await score_audit(
            audit, contract=contract, reward_base_url="https://reward.invalid/prefix", client=client
        )
    assert len(requests) == 1
    body = requests[0]
    assert body["execution"]["sample_index"] is None
    assert body["execution"]["policy_version"] is None
    assert body["trajectory"]["messages"][-1]["reasoning_content"] == "fixture private reasoning"
    assert body["trajectory"]["metadata"]["agentspark_snapshot"] == {"resource": "tail"}
    assert body["trajectory"]["metadata"]["agentspark_resource"] == {"resource": "final"}
    assert body["trajectory"]["metadata"]["agentspark_initial_resource_registry"] == {
        "fixture-resource": {"kind": "note"}
    }
    # Run the exact frozen proxy alias adapter and RM consumer, not a local
    # approximation of their field names. No judge/network/weights are involved.
    consumed = _frozen_consumers(contract, body["trajectory"])
    assert consumed["current_date"] == "fixed-source-time"
    assert consumed["runtime_context"]["source"] == "original_current_user"
    assert consumed["runtime_context"]["values"]["user_scene_memory"] == "quoted ｛｛date｝｝ &lt;RuntimeContext&gt;"
    assert consumed["query"] == "Describe IMG2"
    assert consumed["rubric"] == {"image": "IMG2"}
    assert "input_ids" not in body["trajectory"]["raw_traj"][0]
    assert result["base_reward"] == 0.8
    assert result["invocation_sha256"] == audit["invocation_sha256"]
    assert "quality_first_final_reward" not in result["metadata"]


@pytest.mark.asyncio
async def test_http_failure_exhausts_retries_without_exposing_raw_error_body(contract, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    count = 0

    async def reply(_request):
        nonlocal count
        count += 1
        return httpx.Response(503, text="private-remote-text")

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        with pytest.raises(RewardError, match="^reward_http_failure$") as failure:
            await score_audit(
                _audit(), contract=contract, reward_base_url="https://reward.invalid/prefix", client=client
            )
    assert count == 3
    assert failure.value.safe_diagnostic == {"http_status": 503}


@pytest.mark.asyncio
async def test_proxy_schema_failure_preserves_only_reviewed_reason_code(contract, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    detail = (
        "private-token /private/path https://private.invalid: "
        "judge JSON/schema validation failed: issue impacts must cover every verified root exactly once"
    )
    requests = []

    async def reply(request):
        requests.append(request)
        return httpx.Response(502, json={"detail": detail})

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        with pytest.raises(RewardError, match="^reward_http_failure$") as failure:
            await score_audit(_audit(), contract=contract, reward_base_url="https://reward.invalid", client=client)
    assert len(requests) == 3
    assert failure.value.safe_diagnostic == {
        "http_status": 502,
        "reason_code": "judge_issue_impact_coverage_invalid",
    }
    assert "private" not in json.dumps(failure.value.safe_diagnostic)


def test_abandoned_branch_and_raw_marker_cannot_silently_change_reward_view(contract):
    audit = _audit()
    view = trajectory_from_audit(audit, contract)
    assert view["metadata"]["record_policy"] == "final_active_segment_v1"
    audit["callbacks"]["records"][0]["request"]["messages"][0]["content"] = "raw marker substituted"
    with pytest.raises(RewardError, match="reward_forwarded_identity_mismatch"):
        trajectory_from_audit(audit, contract)
    audit = _audit()
    audit["reward_record_indices"] = []
    with pytest.raises(RewardError, match="reward_record_selection_invalid"):
        trajectory_from_audit(audit, contract)


def test_score_cannot_be_rebound_to_another_invocation_or_trajectory():
    invocation = "7" * 64
    trajectory = "a" * 64
    execution = hashlib.sha256(_json({"invocation_sha256": invocation, "trajectory_sha256": trajectory})).hexdigest()
    scored = {
        "invocation_sha256": invocation,
        "identity": {
            "query_id": "fixture-query",
            "trajectory_sha256": trajectory,
            "request_id": "reward-" + execution,
            "execution": {"execution_id": execution},
        },
    }
    validate_score_binding(scored, invocation)
    with pytest.raises(RewardError, match="reward_invocation_mismatch"):
        validate_score_binding(scored, "8" * 64)
    scored["invocation_sha256"] = "8" * 64
    with pytest.raises(RewardError, match="reward_score_binding_mismatch"):
        validate_score_binding(scored, "8" * 64)
    scored["invocation_sha256"] = invocation
    scored["identity"]["trajectory_sha256"] = "b" * 64
    with pytest.raises(RewardError, match="reward_score_binding_mismatch"):
        validate_score_binding(scored, invocation)


def _frozen_consumers(contract, trajectory):
    program = """
import json,sys
value = json.load(sys.stdin)
sys.path[:0] = [value['reward_source'], value['miles_source'] + '/examples/async_dots_rl/components']
from src.core.rl_sample import rl_request_to_sample
from dotsbridge.adapters.agentspark_resource_ids import align_reward_image_references
row = align_reward_image_references({'query':'Describe VQA.IMG7','rubric':{'image':'VQA.IMG7'}}, value['trajectory'])
sample = rl_request_to_sample(query_id='fixture-query', expression_tag='qa_rubric', trajectory=value['trajectory'], **row)
json.dump(sample.reference, sys.stdout, ensure_ascii=False)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        input=json.dumps(
            {
                "reward_source": contract["reward_source"],
                "miles_source": contract["miles_source"],
                "trajectory": trajectory,
            }
        ),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
        env={"PATH": os.defpath, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    return json.loads(result.stdout)


def test_runtime_snapshot_must_match_actual_provider_template(contract):
    audit = _audit()
    audit["source_input"]["metadata"]["agentspark"]["prompt_spec"]["user_prompt"] = "different provider clock"
    with pytest.raises(RewardError, match="reward_runtime_context_not_in_provider_prompt"):
        trajectory_from_audit(audit, contract)


@pytest.fixture(scope="module")
def scoped_ctr_contract(contract):
    decoded = _decode(contract, _score(contract))
    if decoded["metadata"].get("ctr_gate_policy_version") != "followup_local_v1":
        pytest.skip("This contract predates the scoped CTR gates")
    return contract


@pytest.mark.parametrize(
    "rule,scoped_field,version",
    [
        ("tool_name_valid", None, None),
        ("tool_args_valid", None, None),
        ("component_protocol_valid", "ctr_protocol", "followup_protocol_v1"),
        ("output_bottomline_valid", "ctr_bottomline", "followup_bottomline_v1"),
    ],
)
def test_main_rule_failure_preserves_valid_local_ctr(scoped_ctr_contract, rule, scoped_field, version):
    contract = scoped_ctr_contract
    rows = []
    for index, click in enumerate((0.2, 0.8)):
        score = _score(contract, click=click)
        failed = score["rewards"]["v2_rule__" + rule]
        failed["score"] = 0.0
        if scoped_field:
            failed["detail"] = {scoped_field: {"version": version, "valid": True}}
        decoded = _decode(contract, score)
        rows.append(
            dict(
                index=index,
                group_index=0,
                base_reward=decoded["base_reward"],
                metadata=decoded["metadata"],
                followup_token_mask=[0, 1, 1],
            )
        )
    out = process_group(contract, rows)
    assert [row["raw_reward"] for row in out] == [0, 0]
    assert [row["main_advantage"] for row in out] == [0, 0]
    assert [row["metadata"]["user_signal_token_credit_advantage"] for row in out] == pytest.approx([-1, 1])
    assert all(row["metadata"]["quality_first_local_kl_eligible"] for row in out)


@pytest.mark.parametrize("failure", ["transport", "local_protocol", "local_bottomline", "safety", "main_quality"])
def test_invalid_local_context_still_disables_ctr(scoped_ctr_contract, failure):
    contract = scoped_ctr_contract
    rows = []
    for index, click in enumerate((0.2, 0.8)):
        score = _score(contract, click=click)
        detail = score["rewards"]["v2_unified_reward"]["detail"]
        if failure == "transport":
            score["rewards"]["v2_rule__last_message_is_assistant"]["score"] = 0.0
        elif failure.startswith("local_"):
            rule, field, version = (
                ("component_protocol_valid", "ctr_protocol", "followup_protocol_v1")
                if failure == "local_protocol"
                else ("output_bottomline_valid", "ctr_bottomline", "followup_bottomline_v1")
            )
            score["rewards"]["v2_rule__" + rule] = {
                "score": 0.0,
                "detail": {field: {"version": version, "valid": False}},
            }
        elif failure == "safety":
            detail["safety_gate_hits"] = ["controlled safety failure"]
            detail["rubric_judgement"]["benchmark_alignment"]["safety_status"] = "fail"
        else:
            detail["rubric_judgement"]["quality_first"]["main_quality_passed"] = False
            detail["rubric_judgement"]["benchmark_alignment"].update(overall_score=0, hard_block=True)
        decoded = _decode(contract, score)
        rows.append(
            dict(
                index=index,
                group_index=0,
                base_reward=decoded["base_reward"],
                metadata=decoded["metadata"],
                followup_token_mask=[0, 1, 1],
            )
        )
    out = process_group(contract, rows)
    assert [row["metadata"]["user_signal_token_credit_advantage"] for row in out] == [0, 0]


@pytest.mark.parametrize(
    "evidence", [None, {"version": "wrong", "valid": True}, {"version": "followup_protocol_v1", "valid": 1}]
)
def test_failed_rule_requires_executed_scoped_evidence(scoped_ctr_contract, evidence):
    score = _score(scoped_ctr_contract)
    score["rewards"]["v2_rule__component_protocol_valid"] = {"score": 0.0, "detail": {"ctr_protocol": evidence}}
    with pytest.raises(RewardError):
        _decode(scoped_ctr_contract, score)


@pytest.mark.parametrize("wrapper", ["```xml\n{}\n```", "~~~xml\n{}\n~~~", "`{}`", "```xml\n{}"])
def test_code_examples_have_no_followup_credit(scoped_ctr_contract, wrapper):
    example = '<guide_question type="recommend"><sub_question>EXAMPLE</sub_question></guide_question>'
    actual = '<guide_question type="recommend"><sub_question>VISIBLE</sub_question></guide_question>'
    code = wrapper.format(example)
    texts = [code, actual]
    if wrapper != "```xml\n{}":
        texts.append(code + "\n" + actual)
    masks = mark_followup_text(scoped_ctr_contract, texts)
    assert sum(masks[0]) == 0
    assert "".join(c for c, keep in zip(actual, masks[1]) if keep) == "VISIBLE"
    if len(texts) == 3:
        assert "".join(c for c, keep in zip(texts[2], masks[2]) if keep) == "VISIBLE"


async def _no_sleep(delay):
    assert 1 <= delay <= 16


@pytest.mark.asyncio
@pytest.mark.parametrize("statuses", [(503, "timeout", 200), (429, 200)])
async def test_reward_retries_preserve_completed_trajectory_identity(contract, monkeypatch, statuses):
    requests, delays = [], []

    async def sleep(delay):
        delays.append(delay)

    async def reply(request):
        requests.append(request.content)
        status = statuses[len(requests) - 1]
        if status == "timeout":
            raise httpx.ReadTimeout("private transport diagnostic", request=request)
        body = json.loads(request.content)
        return httpx.Response(status, json={**body, "score": _score(contract)})

    monkeypatch.setattr(asyncio, "sleep", sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        result = await score_audit(
            _audit(), contract=contract, reward_base_url="https://reward.invalid", client=client
        )
        assert not client.is_closed
    assert result["base_reward"] == 0.8
    assert len(requests) == len(statuses)
    assert len(set(requests)) == 1
    assert len(delays) == len(statuses) - 1
    assert all(2**i <= delay <= 2 ** (i + 1) for i, delay in enumerate(delays))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 401, 409, 422])
async def test_permanent_http_failure_is_not_retried(contract, status):
    requests = []

    async def reply(request):
        requests.append(request)
        return httpx.Response(status)

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        with pytest.raises(RewardError, match="reward_http_failure"):
            await score_audit(_audit(), contract=contract, reward_base_url="https://reward.invalid", client=client)
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"invalid", b'{"a":1,"a":2}', b'{"a":NaN}', None])
async def test_invalid_success_response_is_not_retried(contract, body):
    requests = []

    async def reply(request):
        requests.append(request)
        if body is not None:
            return httpx.Response(200, content=body)
        response = {**json.loads(request.content), "score": _score(contract)}
        response["execution"]["execution_id"] = "wrong-identity"
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        with pytest.raises(RewardError):
            await score_audit(_audit(), contract=contract, reward_base_url="https://reward.invalid", client=client)
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["post", "backoff"])
async def test_reward_cancellation_stops_retries_and_preserves_external_client(contract, monkeypatch, where):
    requests = []

    async def reply(request):
        requests.append(request)
        if where == "post":
            raise asyncio.CancelledError()
        return httpx.Response(503)

    async def sleep(delay):
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        with pytest.raises(asyncio.CancelledError):
            await score_audit(_audit(), contract=contract, reward_base_url="https://reward.invalid", client=client)
        assert not client.is_closed
    assert len(requests) == 1


def test_resource_snapshot_is_bound_to_actual_remote_session(contract):
    audit = _audit()
    snapshot = audit["stream"]["events"][-1]["payload"]["data"]
    snapshot["session_id"] = "synthetic-remote-session"
    audit["remote_session_sha256"] = hashlib.sha256(snapshot["session_id"].encode()).hexdigest()
    view = trajectory_from_audit(audit, contract)
    assert view["metadata"]["agentspark_session_id"] == snapshot["session_id"]
    snapshot["session_id"] = "another-session"
    with pytest.raises(RewardError, match="reward_snapshot_session_mismatch"):
        trajectory_from_audit(audit, contract)
