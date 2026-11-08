# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pinned formal V21 policy, isolated from Relax imports and process environment.

The worker executes the explicitly pinned, historical Miles Python files. It
never contacts a service. The async client scores one completed invocation;
``process_group`` alone applies scalar shaping and group-relative credit.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit


class RewardError(ValueError):
    """A fixed diagnostic code; remote bodies and business content stay private."""

    def __init__(self, code: str, *, safe_diagnostic: dict[str, Any] | None = None) -> None:
        super().__init__(code)
        self.safe_diagnostic = safe_diagnostic or {}


_ENTRY_MODULES = (
    "examples.async_dots_rl.reward_scoring",
    "examples.async_dots_rl.trajectory_envelope",
    "examples.async_dots_rl.reward_policy_metadata",
    "examples.async_dots_rl.dots_reward_post_process",
    "examples.async_dots_rl.followup_token_mask",
    "examples.async_dots_rl.components.dotsbridge.reward_proxy",
    "examples.async_dots_rl.components.dotsbridge.adapters.agentspark_trajectory_adapter",
    "examples.async_dots_rl.components.dotsbridge.adapters.runtime_context",
)
_POLICY_ENV = {
    "SLIME_ASYNC_DOTS_REWARD_POLICY": "quality_first_v1",
    "SLIME_ASYNC_DOTS_QUALITY_GUARD": "quality_guard_v19",
    "SLIME_ASYNC_DOTS_PRESENTATION_POLICY": "component_coverage_v20",
    "SLIME_ASYNC_DOTS_RM_POLICY": "v21_speed_hardgate_wx_ctr_v1",
    "SLIME_ASYNC_DOTS_BENCHMARK_ALIGNMENT_V14": "1",
}


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json(value: str | bytes) -> Any:
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise RewardError("reward_duplicate_json_key")
            result[key] = item
        return result

    def constant(_value):
        raise RewardError("reward_nonfinite_json")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise RewardError("reward_number_invalid")
    return float(value)


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise RewardError("reward_digest_invalid")
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RewardError("reward_identity_invalid")
    return value


def validate_score_binding(scored: dict[str, Any], invocation_sha256: str) -> None:
    """Reject a valid score copied from another logical invocation or trajectory."""
    invocation = _digest(invocation_sha256)
    if not isinstance(scored, dict) or scored.get("invocation_sha256") != invocation:
        raise RewardError("reward_invocation_mismatch")
    identity = scored.get("identity")
    if not isinstance(identity, dict):
        raise RewardError("reward_score_identity_mismatch")
    trajectory = _digest(identity.get("trajectory_sha256"))
    execution_id = _sha(_json({"invocation_sha256": invocation, "trajectory_sha256": trajectory}))
    execution = identity.get("execution")
    if (
        not isinstance(execution, dict)
        or execution.get("execution_id") != execution_id
        or identity.get("request_id") != "reward-" + execution_id
    ):
        raise RewardError("reward_score_binding_mismatch")
    _text(identity.get("query_id"))


def _source_files(root: Path) -> dict[str, str]:
    """Hash the static local import closure, including lazy policy imports."""
    pending = list(_ENTRY_MODULES)
    seen: set[str] = set()
    files = {}
    while pending:
        name = pending.pop()
        if name in seen or not name.startswith(("miles.", "examples.async_dots_rl")):
            continue
        seen.add(name)
        path = root.joinpath(*name.split(".")).with_suffix(".py")
        if not path.is_file():
            path = root.joinpath(*name.split("."), "__init__.py")
        if not path.is_file():
            if name in _ENTRY_MODULES:
                raise RewardError("reward_frozen_source_layout_missing")
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise RewardError("reward_source_escape")
        raw = path.read_bytes()
        files[str(path.relative_to(root))] = _sha(raw)
        for parent in path.parents:
            if parent == root:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file():
                files[str(initializer.relative_to(root))] = _sha(initializer.read_bytes())
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(ast.parse(raw)):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = importlib.util.resolve_name("." * node.level + base, package)
                pending.append(base)
                pending.extend(base + "." + alias.name for alias in node.names if alias.name != "*")
    return dict(sorted(files.items()))


def _policy_environment(root: Path, files: dict[str, str], config: dict[str, Any]) -> dict[str, str]:
    names = set()
    for filename in files:
        for node in ast.walk(ast.parse((root / filename).read_bytes())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = node.value
                if re.fullmatch(r"(?:SLIME|DOTSREWARD)_[A-Z0-9_]+", name) and not re.search(
                    r"(?:URL|ENDPOINT|FILE|PATH|HOST|PORT|SECRET|PASSWORD|AUTH|API_KEY|COOKIE|CREDENTIAL)", name
                ):
                    names.add(name)
    env = {key: str(value) for key, value in config["runtime_env"].items() if key in names}
    if any(env.get(key) != value for key, value in _POLICY_ENV.items()):
        raise RewardError("reward_formal_policy_mismatch")
    return dict(sorted(env.items()))


def _fingerprint(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{16}", value):
        raise RewardError("reward_fingerprint_invalid")
    return value


def load_reward_contract(path: str | Path) -> dict[str, Any]:
    contract = _strict_json(Path(path).read_bytes())
    _verify_contract(contract)
    return contract


def pin_reward_contract(
    *, miles_source: str | Path, source_config: str | Path, expected_prompt_bundle_fingerprint: str
) -> dict[str, Any]:
    """Explicit offline provisioning: inspect and return the identity to approve."""
    root = Path(miles_source).resolve(strict=True)
    config = Path(source_config).resolve(strict=True)
    values = _strict_json(config.read_bytes())
    env = values["runtime_env"]
    if any(env.get(key) != value for key, value in _POLICY_ENV.items()):
        raise RewardError("reward_formal_policy_mismatch")
    if (
        values["train"].get("advantage-estimator") != "grpo"
        or values["train"].get("followup-token-credit") is not True
    ):
        raise RewardError("reward_formal_objective_mismatch")
    files = _source_files(root)
    return {
        "schema_version": 1,
        "source_layout": "formal_v21_legacy",
        "miles_source": str(root),
        "source_config": str(config),
        "source_config_sha256": _sha(config.read_bytes()),
        "source_files": files,
        "source_files_sha256": _sha(_json(files)),
        "policy_environment": _policy_environment(root, files, values),
        "expected_prompt_bundle_fingerprint": _fingerprint(expected_prompt_bundle_fingerprint),
    }


def _verify_contract(contract: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if contract.get("schema_version") != 1 or contract.get("source_layout") != "formal_v21_legacy":
        raise RewardError("reward_contract_invalid")
    root = Path(contract["miles_source"]).resolve(strict=True)
    raw = Path(contract["source_config"]).read_bytes()
    if _sha(raw) != contract["source_config_sha256"]:
        raise RewardError("reward_source_config_changed")
    files = _source_files(root)
    if files != contract["source_files"] or _sha(_json(files)) != contract["source_files_sha256"]:
        raise RewardError("reward_source_changed")
    config = _strict_json(raw)
    if contract.get("policy_environment") != _policy_environment(root, files, config) or any(
        config["runtime_env"].get(key) != value for key, value in _POLICY_ENV.items()
    ):
        raise RewardError("reward_formal_policy_mismatch")
    _fingerprint(contract["expected_prompt_bundle_fingerprint"])
    if "reward_source" in contract and _rm_files(Path(contract["reward_source"])) != contract.get(
        "reward_source_files"
    ):
        raise RewardError("reward_rm_source_changed")
    if "system_prompt" in contract:
        prompt = contract["system_prompt"]
        if _sha(Path(prompt["path"]).read_bytes()) != prompt["sha256"]:
            raise RewardError("reward_system_prompt_changed")
    return root, config


def _source_identity(contract: dict[str, Any]) -> dict[str, Any]:
    return {key: contract[key] for key in ("source_layout", "source_config_sha256", "source_files_sha256")}


def _worker_call(contract: dict[str, Any], operation: str, payload: Any) -> Any:
    # No inherited policy flags, PYTHONPATH, credentials, or visible CUDA devices.
    env = {
        "PATH": os.defpath,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker"],
            input=_json({"contract": contract, "operation": operation, "payload": payload}),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RewardError("reward_worker_unavailable") from exc
    try:
        reply = _strict_json(result.stdout)
    except (ValueError, UnicodeError) as exc:
        raise RewardError("reward_worker_protocol_invalid") from exc
    if result.returncode != 0 or not reply.get("ok"):
        code = reply.get("code")
        if not isinstance(code, str) or not re.fullmatch(r"reward_[a-z_]+", code):
            code = "reward_source_operation_failed"
        raise RewardError(code)
    return reply["result"]


def mark_followup_text(contract: dict[str, Any], texts: list[str]) -> list[list[int]]:
    """Mark already isolated visible K3 text using the frozen Miles XML rules."""
    return _worker_call(contract, "mark_followup_text", texts)


def process_group(contract: dict[str, Any], logical_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the source objective once per logical index, before physical fanout."""
    return _worker_call(contract, "process_group", logical_samples)


def _validated_score(
    contract: dict[str, Any], request: dict[str, Any], reply: Any, config: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(reply, dict):
        raise RewardError("reward_response_invalid")
    for key in ("request_id", "query_id", "trajectory_sha256", "execution"):
        if reply.get(key) != request[key]:
            raise RewardError("reward_response_identity_mismatch")
    score = reply.get("score")
    if not isinstance(score, dict) or score.get("query_id") != request["query_id"]:
        raise RewardError("reward_score_identity_mismatch")
    if score.get("combo_error") or score.get("error") or score.get("success") is False:
        raise RewardError("reward_remote_scoring_failed")
    total = _number(score.get("total"))
    if not 0 <= total <= 1:
        raise RewardError("reward_total_out_of_range")
    rewards = score.get("rewards")
    unified = rewards.get("v2_unified_reward") if isinstance(rewards, dict) else None
    detail = unified.get("detail") if isinstance(unified, dict) else None
    if not isinstance(detail, dict) or detail.get("valid") is not True:
        raise RewardError("reward_unified_execution_invalid")
    if not 0 <= _number(unified.get("score")) <= 1:
        raise RewardError("reward_unified_execution_invalid")
    expected = {
        "schema_version": "v2_unified_reward_v2",
        "rm_policy_version": contract["policy_environment"]["SLIME_ASYNC_DOTS_REWARD_POLICY"],
        "business_policy_version": contract["policy_environment"]["SLIME_ASYNC_DOTS_RM_POLICY"],
        "prompt_bundle_fingerprint": contract["expected_prompt_bundle_fingerprint"],
        "llm_judge_api_model": _text(config["reward_model"]["served_model_name"]),
    }
    if any(detail.get(key) != value for key, value in expected.items()):
        raise RewardError("reward_remote_policy_mismatch")
    if detail.get("critic_report_errors"):
        raise RewardError("reward_critic_execution_failed")
    policy = importlib.import_module("examples.async_dots_rl.quality_first_policy")
    gates = importlib.import_module("miles.utils.rl_policy_v21")
    metadata = gates.attach_hard_gates(policy.metadata_from_detail(detail), score)
    if (
        metadata.get("presentation_policy_version")
        != contract["policy_environment"]["SLIME_ASYNC_DOTS_PRESENTATION_POLICY"]
    ):
        raise RewardError("reward_presentation_policy_mismatch")
    # Do not finalize here: the exact source group objective owns shaping once.
    return {
        "base_reward": total,
        "metadata": metadata,
        "identity": {
            key: copy.deepcopy(request[key]) for key in ("request_id", "query_id", "trajectory_sha256", "execution")
        },
        "source_identity": _source_identity(contract),
        "prompt_bundle_fingerprint": detail["prompt_bundle_fingerprint"],
    }


async def _source_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the source request builder with a capture-only transport, never HTTP."""
    import httpx

    source = importlib.import_module("examples.async_dots_rl.reward_scoring")
    proxy = importlib.import_module("examples.async_dots_rl.components.dotsbridge.reward_proxy")
    captured = []

    class Capture:
        async def post(self, _url, *, json, timeout):
            captured.append(copy.deepcopy(json))
            return httpx.Response(200, json={**json, "score": {}})

    envelope = SimpleNamespace(**payload)
    cfg = SimpleNamespace(url="http://capture.invalid", pipeline_reward_max_attempts=1, pipeline_reward_timeout_sec=1)
    await source._score_committed_trajectory(client=Capture(), cfg=cfg, envelope=envelope, trace_handle=None)
    if len(captured) != 1:
        raise RewardError("reward_source_request_not_captured")
    request = proxy.RewardScoreRequest.model_validate(captured[0]).model_dump(mode="json", exclude_none=True)
    # None fields remain explicit in execution because the legacy server echoes
    # its full execution schema; no native indices or policy version are invented.
    request["execution"] = proxy.RewardExecutionMetadata.model_validate(captured[0]["execution"]).model_dump(
        mode="json"
    )
    return request


def _source_group(config: dict[str, Any], logical: Any) -> list[dict[str, Any]]:
    if not isinstance(logical, list) or not logical:
        raise RewardError("reward_group_empty")
    seen = set()
    samples = []
    for item in logical:
        index, group = item.get("index"), item.get("group_index")
        if type(index) is not int or type(group) is not int or min(index, group) < 0:
            raise RewardError("reward_logical_identity_invalid")
        if (group, index) in seen:
            raise RewardError("reward_logical_identity_duplicate")
        seen.add((group, index))
        mask = item.get("followup_token_mask")
        if not isinstance(mask, list) or any(type(value) is not int or value not in (0, 1) for value in mask):
            raise RewardError("reward_followup_mask_invalid")
        base = _number(item.get("base_reward"))
        if not 0 <= base <= 1:
            raise RewardError("reward_total_out_of_range")
        metadata = copy.deepcopy(item["metadata"])
        if "quality_first_final_reward" in metadata:
            raise RewardError("reward_already_postprocessed")
        if any(key.startswith("tito_segment_") or key == "multi_segment_bundle_member" for key in metadata):
            raise RewardError("reward_physical_metadata_in_logical_sample")
        removed = item.get("remove_sample", False)
        if type(removed) is not bool:
            raise RewardError("reward_remove_sample_invalid")
        samples.append(
            SimpleNamespace(
                index=index,
                group_index=group,
                rollout_id=index,
                remove_sample=removed,
                metadata=metadata,
                followup_token_mask=mask,
                get_reward_value=lambda _args, value=base: value,
            )
        )
    train = config["train"]
    args = SimpleNamespace(
        advantage_estimator=train["advantage-estimator"],
        followup_token_credit=train["followup-token-credit"],
        rewards_normalization=not train.get("disable-rewards-normalization", False),
        grpo_std_normalization=not train.get("disable-grpo-std-normalization", False),
    )
    source = importlib.import_module("examples.async_dots_rl.dots_reward_post_process")
    raw, advantages = source.post_process_rewards(args, samples)
    return [
        {
            "index": sample.index,
            "group_index": sample.group_index,
            "raw_reward": _number(reward),
            "main_advantage": _number(advantage),
            "metadata": sample.metadata,
            "remove_sample": sample.remove_sample,
        }
        for sample, reward, advantage in zip(samples, raw, advantages, strict=True)
    ]


def _run_worker(request: dict[str, Any]) -> Any:
    if request["operation"] == "fingerprint":
        return _compute_fingerprint(request["payload"])
    contract = request["contract"]
    root, config = _verify_contract(contract)
    sys.path.insert(0, str(root))
    os.environ.update(contract["policy_environment"])
    operation, payload = request["operation"], request["payload"]
    if operation == "reward_view":
        return _source_reward_view(**payload)
    if operation == "build_request":
        return asyncio.run(_source_request(payload))
    if operation == "decode_score":
        return _validated_score(contract, payload["request"], payload["response"], config)
    if operation == "process_group":
        return _source_group(config, payload)
    if operation == "mark_followup_text":
        if not isinstance(payload, list) or any(not isinstance(text, str) for text in payload):
            raise RewardError("reward_visible_text_invalid")
        module = importlib.import_module("examples.async_dots_rl.followup_token_mask")
        types = (
            ("recommend", "choice", "question")
            if module.quality_first_enabled() or module.clarification_enabled()
            else ("recommend",)
        )
        masks = []
        for text in payload:
            mask = [0] * len(text)
            module._mark_recommend_visible_text(
                text, content_start=0, content_end=len(text), char_mask=mask, followup_types=types
            )
            masks.append(mask)
        return masks
    raise RewardError("reward_worker_operation_unknown")


def trajectory_from_audit(audit: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Build the explicit Relax reward view; never synthesize Miles token records."""
    from .result import reject_credentials

    reject_credentials(audit, "")
    stream, callbacks = audit.get("stream", {}), audit.get("callbacks", {})
    if (
        stream.get("saw_end") is not True
        or stream.get("saw_error") is not False
        or stream.get("eof") is not True
        or audit.get("cleanup", {}).get("local") != "closed"
        or audit.get("remote_execution") != "completed"
        or audit.get("send_attempts") != 1
        or callbacks.get("errors")
        or callbacks.get("concurrent") is not False
    ):
        raise RewardError("reward_audit_incomplete")
    records = callbacks.get("records")
    indices = audit.get("reward_record_indices")
    if (
        not isinstance(records, list)
        or not records
        or callbacks.get("started") != len(records)
        or audit.get("reward_record_policy") != "final_active_segment_v1"
        or not isinstance(indices, list)
        or not indices
        or any(type(index) is not int for index in indices)
        or indices != list(range(indices[0], len(records)))
        or indices[0] < 0
    ):
        raise RewardError("reward_record_selection_invalid")
    for index, record in enumerate(records):
        if record.get("index") != index or record.get("status") != 200:
            raise RewardError("reward_callback_invalid")
        request, assistant = record.get("request"), record.get("assistant")
        if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
            raise RewardError("reward_callback_invalid")
        if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
            raise RewardError("reward_callback_invalid")
        if record.get("request_identity") != "actual_forwarded":
            raise RewardError("reward_forwarded_identity_required")
        if _sha(_json(request)) != record.get("forwarded_request_canonical_sha256"):
            raise RewardError("reward_forwarded_identity_mismatch")
    if records[-1]["assistant"].get("tool_calls"):
        raise RewardError("reward_terminal_assistant_required")
    source = audit.get("source_input")
    if not isinstance(source, dict) or not isinstance(source.get("metadata"), dict):
        raise RewardError("reward_source_metadata_missing")
    metadata = source["metadata"]
    query_id = _text(metadata.get("query_id", source.get("query_id")))
    if not metadata.get("rubric") or not isinstance(metadata.get("expression_tag"), str):
        raise RewardError("reward_source_rubric_missing")
    selected = [records[index] for index in indices]
    last = selected[-1]
    # The source RM uses messages for current-turn content/media, and raw_traj's
    # call_llm/tools/tool_calls for tool-rule facts. Preserve reasoning fields.
    trajectory = {
        "query_id": query_id,
        "user_id": "relax-" + _digest(audit.get("invocation_sha256")),
        "messages": copy.deepcopy(last["request"]["messages"]) + [copy.deepcopy(last["assistant"])],
        "tools": copy.deepcopy(last["request"].get("tools") or []),
        "raw_traj": [
            {
                "turn_index": position,
                "call_llm": {
                    "messages": copy.deepcopy(record["request"]["messages"]),
                    "tools": copy.deepcopy(record["request"].get("tools") or []),
                },
                "role": "assistant",
                "content": copy.deepcopy(record["assistant"].get("content")),
                "reasoning_content": copy.deepcopy(record["assistant"].get("reasoning_content")),
                "tool_calls": copy.deepcopy(record["assistant"].get("tool_calls") or []),
                "status_code": record["status"],
            }
            for position, record in enumerate(selected)
        ],
        "metadata": {
            "trajectory_source": "relax_actual_forwarded_callbacks",
            "record_policy": audit["reward_record_policy"],
            "reward_record_indices": list(indices),
            "invocation_sha256": audit["invocation_sha256"],
            "agentspark_events": copy.deepcopy(stream["events"]),
            "agentspark_initial_resource_registry": copy.deepcopy(metadata.get("resource_registry")),
            "agentspark_runtime_context": copy.deepcopy(
                metadata.get("agentspark_prepare", {}).get("runtime_snapshot")
            ),
            "agentspark_legacy_image_id_aliases": copy.deepcopy(
                metadata.get("agentspark_prepare", {}).get("resource_id_aliases") or {}
            ),
            "token_provenance": "Native Relax Forest only; reward view has no invented token IDs or logprobs.",
        },
    }
    return _worker_call(
        contract,
        "reward_view",
        {
            "trajectory": trajectory,
            "runtime_prompt": metadata.get("agentspark", {}).get("prompt_spec", {}).get("user_prompt"),
        },
    )


async def score_audit(
    audit: dict[str, Any], *, contract: dict[str, Any], reward_base_url: str, client: Any = None
) -> dict[str, Any]:
    """One immutable scoring request; failures raise and publish no reward."""
    import httpx

    from .diagnostics import reward_http_diagnostic

    endpoint = urlsplit(reward_base_url)
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.hostname
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.query
        or endpoint.fragment
    ):
        raise RewardError("reward_endpoint_invalid")
    trajectory = await asyncio.to_thread(trajectory_from_audit, audit, contract)
    metadata = audit["source_input"]["metadata"]
    trajectory_hash = _sha(_json(trajectory))
    invocation = audit["invocation_sha256"]
    payload = {
        "query_id": trajectory["query_id"],
        "task_id": _text(metadata.get("task_id", "default")),
        "trajectory": trajectory,
        "trajectory_sha256": trajectory_hash,
        "execution_id": _sha(_json({"invocation_sha256": invocation, "trajectory_sha256": trajectory_hash})),
        "rollout_id": None,
        "policy_version": None,
        "group_index": None,
        "sample_index": None,
        "attempt_index": 1,
        "attempt_tag": "relax-completed-invocation",
        "csv_row": copy.deepcopy(metadata),
        "expression_tag": _text(metadata["expression_tag"]),
    }
    request = await asyncio.to_thread(_worker_call, contract, "build_request", payload)
    owned = client is None
    if owned:
        client = httpx.AsyncClient(trust_env=False, follow_redirects=False)
    try:
        # No retry re-enters AgentSpark or changes request identity. The Bridge
        # proxy owns idempotent scoring of this immutable execution identifier.
        timeout = float(contract["policy_environment"]["SLIME_ASYNC_DOTS_REWARD_TIMEOUT_SEC"])
        if not math.isfinite(timeout) or timeout <= 0:
            raise RewardError("reward_timeout_invalid")
        response = await client.post(reward_base_url.rstrip("/") + "/reward/score", json=request, timeout=timeout)
        if response.status_code != 200:
            error_body = None
            if len(response.content) <= 65536:
                try:
                    error_body = _strict_json(response.content)
                except (ValueError, UnicodeError):
                    pass
            raise RewardError(
                "reward_http_failure",
                safe_diagnostic=reward_http_diagnostic(error_body, response.status_code),
            )
        reply = _strict_json(response.content)
    except (httpx.HTTPError, ValueError, UnicodeError) as exc:
        if isinstance(exc, RewardError):
            raise
        raise RewardError("reward_transport_or_json_failure") from exc
    finally:
        if owned:
            await client.aclose()
    result = await asyncio.to_thread(_worker_call, contract, "decode_score", {"request": request, "response": reply})
    result["invocation_sha256"] = invocation
    result["record_policy"] = audit["reward_record_policy"]
    result["reward_record_indices"] = audit["reward_record_indices"]
    validate_score_binding(result, invocation)
    return result


def _source_reward_view(trajectory: dict[str, Any], runtime_prompt: str | None) -> dict[str, Any]:
    runtime = trajectory["metadata"]["agentspark_runtime_context"]
    if runtime is not None:
        source_runtime = importlib.import_module(
            "examples.async_dots_rl.components.dotsbridge.adapters.runtime_context"
        )
        values = runtime.get("values") if isinstance(runtime, dict) else None
        if (
            not isinstance(values, dict)
            or set(values) != set(source_runtime.FIELDS)
            or any(not isinstance(value, str) for value in values.values())
            or not isinstance(runtime_prompt, str)
        ):
            raise RewardError("reward_runtime_context_invalid")
        effective = {key: source_runtime.escape_context_value(value) for key, value in values.items()}
        # Existing prepared artifacts retained the selected snapshot, but not
        # the environment's escaped values. Prove every effective nonempty value
        # survived into the actual Send template; private-media substitutions
        # cannot silently become a different judge context.
        if any(value and value not in runtime_prompt for value in effective.values()):
            raise RewardError("reward_runtime_context_not_in_provider_prompt")
        runtime["values"] = effective
        trajectory["current_date"] = effective["current_time"]
    module = importlib.import_module(
        "examples.async_dots_rl.components.dotsbridge.adapters.agentspark_trajectory_adapter"
    )
    for record in trajectory["raw_traj"]:
        record["call_llm"] = module.summarize_data_images(record["call_llm"])
    events = []
    for evidence in trajectory["metadata"]["agentspark_events"]:
        payload = evidence.get("payload")
        kind = evidence.get("event") or evidence.get("kind")
        if isinstance(payload, dict):
            kind = payload.get("type") or kind
            data = payload.get("data", payload)
        else:
            data = payload
        events.append({"type": kind, "data": data})
    events = module._normalize_events(events)
    trajectory["metadata"]["agentspark_events"] = module.summarize_data_images(events)
    trajectory["metadata"]["agentspark_resource"] = next(
        (
            module.summarize_data_images(event.get("data"))
            for event in reversed(events)
            if event.get("type") in {"resource", "resource-snapshot", "resource.snapshot"}
        ),
        None,
    )
    trajectory["metadata"]["agentspark_snapshot"] = next(
        (
            module.summarize_data_images(event.get("data"))
            for event in reversed(events)
            if event.get("type") == "id-index-snapshot"
        ),
        None,
    )
    return trajectory


def _rm_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha(path.read_bytes())
        for path in sorted((root / "src").rglob("*"))
        if path.is_file() and path.suffix in {".py", ".jinja2"}
    }


def _compute_fingerprint(spec: dict[str, Any]) -> str:
    source = Path(spec["miles_source"]).resolve(strict=True)
    reward = Path(spec["reward_source"]).resolve(strict=True)
    config = _strict_json(Path(spec["source_config"]).read_bytes())
    sys.path[:0] = [str(source), str(reward)]
    for key, value in config["runtime_env"].items():
        if key.startswith("SLIME_ASYNC_DOTS_") and isinstance(value, str):
            os.environ[key] = value
    os.environ["DOTSBRIDGE_AGENTSPARK_SYSTEM_PROMPT_FILE"] = config["runtime_env"][
        "DOTSBRIDGE_AGENTSPARK_SYSTEM_PROMPT_FILE"
    ]
    module = importlib.import_module("src.rewards.v2.unified_reward")
    return _fingerprint(module._prompt_bundle_fingerprint())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pin the frozen formal V21 sources; no service is contacted.")
    commands = parser.add_subparsers(dest="command", required=True)
    pin = commands.add_parser("pin")
    pin.add_argument("--frozen-source", required=True)
    pin.add_argument("--run-config", required=True)
    pin.add_argument("--reward-source", required=True)
    pin.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    spec = {"miles_source": args.frozen_source, "source_config": args.run_config, "reward_source": args.reward_source}
    fingerprint = _worker_call({}, "fingerprint", spec)
    contract = pin_reward_contract(
        miles_source=args.frozen_source, source_config=args.run_config, expected_prompt_bundle_fingerprint=fingerprint
    )
    reward_root = Path(args.reward_source).resolve(strict=True)
    contract["reward_source"] = str(reward_root)
    contract["reward_source_files"] = _rm_files(reward_root)
    values = _strict_json(Path(args.run_config).read_bytes())
    prompt = Path(values["runtime_env"]["DOTSBRIDGE_AGENTSPARK_SYSTEM_PROMPT_FILE"]).resolve(strict=True)
    contract["system_prompt"] = {"path": str(prompt), "sha256": _sha(prompt.read_bytes())}
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_json(contract) + b"\n")
    return 0


def _worker_main() -> int:
    try:
        result = _run_worker(_strict_json(sys.stdin.buffer.read()))
        output = {"ok": True, "result": result}
        status = 0
    except Exception as exc:
        output = {"ok": False, "code": str(exc) if isinstance(exc, RewardError) else "reward_source_operation_failed"}
        status = 1
    sys.stdout.buffer.write(_json(output))
    return status


if __name__ == "__main__":
    raise SystemExit(_worker_main() if sys.argv[1:] == ["--worker"] else main())
