# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Convert complete logical AgentSpark groups to native Relax training rows."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

from examples.agentspark_agentic.k3_token_credit import attach_k3_followup_mask
from examples.agentspark_agentic.v21_loss import DEFAULTS, FIELDS, MAIN_DEFAULTS, validate


class CreditError(ValueError):
    """Fixed diagnostic codes; no business payload or credentials."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise CreditError(code)


def apply_ownership(samples: list[Any]) -> dict[int, list[Any]]:
    """Count each committed response once across physical context lineages."""
    grouped = defaultdict(list)
    group_for_index = {}
    for sample in samples:
        _require(type(sample.index) is int and type(sample.group_index) is int, "v21_logical_identity_invalid")
        _require(
            sample.index not in group_for_index or group_for_index[sample.index] == sample.group_index,
            "v21_logical_index_reused",
        )
        group_for_index[sample.index] = sample.group_index
        grouped[sample.index].append(sample)
    for siblings in grouped.values():
        owned_seen, expected = set(), None
        segment_indices = set()
        sessions = {sample.session_id for sample in siblings}
        _require(len(sessions) == 1 and None not in sessions, "v21_session_identity_mismatch")
        anchors = 0
        removed = any(sample.remove_sample for sample in siblings)
        invocation = None
        for sample in siblings:
            metadata = sample.metadata.get("agentspark_training", {})
            _require(metadata.get("schema_version") == 1, "v21_ownership_required")
            segment = metadata.get("segment_index")
            _require(
                type(segment) is int and 0 <= segment < len(siblings) and segment not in segment_indices,
                "v21_segment_index_invalid",
            )
            segment_indices.add(segment)
            _require(metadata.get("segment_count") == len(siblings), "v21_segment_count_mismatch")
            _require(type(metadata.get("is_anchor")) is bool, "v21_anchor_invalid")
            anchors += int(metadata["is_anchor"])
            _require(not metadata["is_anchor"] or segment == len(siblings) - 1, "v21_anchor_not_terminal")
            ident = metadata.get("invocation_sha256")
            _require(
                isinstance(ident, str) and re.fullmatch("[0-9a-f]{64}", ident) is not None, "v21_invocation_required"
            )
            invocation = ident if invocation is None else invocation
            _require(ident == invocation, "v21_invocation_mismatch")
            all_hashes, owned = metadata.get("all_resp_state_hashes"), metadata.get("owned_resp_state_hashes")
            _require(
                isinstance(all_hashes, list)
                and bool(all_hashes)
                and all(isinstance(h, str) and h for h in all_hashes),
                "v21_response_identity_invalid",
            )
            _require(len(set(all_hashes)) == len(all_hashes), "v21_response_identity_duplicate")
            expected = set(all_hashes) if expected is None else expected
            _require(set(all_hashes) == expected, "v21_response_identity_mismatch")
            _require(
                isinstance(owned, list) and len(set(owned)) == len(owned) and set(owned) <= expected,
                "v21_owned_response_invalid",
            )
            _require(not owned_seen.intersection(owned), "v21_response_owned_twice")
            owned_seen.update(owned)
            turns = sample.metadata.get("agentic_trace", {}).get("turns")
            spans = sample.custom_advantage
            _require(
                isinstance(turns, list) and isinstance(spans, list) and len(turns) == len(spans),
                "v21_native_spans_required",
            )
            _require(
                isinstance(sample.loss_mask, list)
                and len(sample.loss_mask) == sample.response_length
                and all(type(x) is int and x in (0, 1) for x in sample.loss_mask),
                "v21_native_loss_mask_invalid",
            )
            original_mask = list(sample.loss_mask)
            span_mask, retained_mask = [0] * sample.response_length, [0] * sample.response_length
            native_seen, previous = set(), 0
            for turn, span in zip(turns, spans, strict=True):
                _require(
                    isinstance(turn, dict)
                    and isinstance(span, list)
                    and len(span) == 2
                    and all(type(v) is int for v in span),
                    "v21_native_span_invalid",
                )
                start, stop = span
                _require(
                    previous <= start < stop <= sample.response_length and turn.get("response_length") == stop - start,
                    "v21_native_span_mismatch",
                )
                previous = stop
                state = turn.get("resp_state_hash")
                _require(
                    isinstance(state, str) and state not in native_seen and state in expected,
                    "v21_native_response_mismatch",
                )
                native_seen.add(state)
                span_mask[start:stop] = [1] * (stop - start)
                if state in owned:
                    retained_mask[start:stop] = original_mask[start:stop]
            _require(set(owned) <= native_seen, "v21_owned_response_missing")
            _require(
                all(not value or span for value, span in zip(original_mask, span_mask, strict=True)),
                "v21_observation_has_loss",
            )
            sample.remove_sample = removed
            sample.loss_mask = [0] * sample.response_length if removed else retained_mask
        _require(anchors == 1 and owned_seen == expected, "v21_incomplete_logical_rollout")
    return dict(grouped)


def convert_credit(samples: list[Any]) -> dict[str, list[Any]]:
    """Miles response-mask transport with whole logical Sample.index denominators."""
    result = {name: [] for name in FIELDS}
    totals = defaultdict(int)
    for sample in samples:
        mask = sample.metadata.get("agentspark_v21", {}).get("followup_token_mask")
        _require(
            isinstance(mask, list)
            and len(mask) == sample.response_length
            and all(type(v) is int and v in (0, 1) for v in mask),
            "v21_followup_mask_invalid",
        )
        _require(
            all(not a or b or sample.remove_sample for a, b in zip(mask, sample.loss_mask, strict=True)),
            "v21_followup_outside_loss",
        )
        quality = [0] * sample.response_length if sample.remove_sample else list(mask)
        meta = sample.metadata
        if sample.remove_sample or (
            meta.get("reward_policy_version") == "quality_first_v1" and not meta.get("quality_first_local_kl_eligible")
        ):
            mask = [0] * sample.response_length
        scalar = meta.get("user_signal_token_credit_advantage", 0.0)
        _require(type(scalar) in (int, float) and math.isfinite(scalar), "v21_credit_scalar_invalid")
        if sample.remove_sample:
            _require(scalar == 0, "v21_removed_sample_has_credit")
        violation = (
            meta.get("followup_policy_version") in {"v2", "v3-type-routing", "v5-scenario-routing", "quality_first_v1"}
            and meta.get("contract_followup_local_quality_violation")
        ) or meta.get("contract_followup_clarification_local_quality_violation")
        result["followup_token_masks"].append(list(mask))
        result["followup_quality_masks"].append(quality)
        result["followup_token_advantage_scalars"].append(float(scalar))
        result["followup_quality_violation_flags"].append(bool(violation))
        result["followup_clarification_credit_flags"].append(
            meta.get("followup_token_credit_source") == "clarification_quality"
        )
        totals[sample.index] += sum(mask)
    for sample, scalar in zip(samples, result["followup_token_advantage_scalars"], strict=True):
        _require(scalar == 0 or totals[sample.index] > 0, "v21_nonzero_credit_without_tokens")
        result["followup_token_mask_sums"].append(totals[sample.index])
    return result


@lru_cache(maxsize=2)
def _tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)


def training_overrides(contract: dict) -> dict:
    """Generate source-backed YAML fields; this never selects a server or model."""
    from examples.agentspark_agentic.v21_reward import _verify_contract

    root, config = _verify_contract(contract)
    train = config["train"]
    options = {name: train.get(name.replace("_", "-"), default) for name, default in DEFAULTS.items()}
    options["followup_token_credit"] = train["followup-token-credit"]
    identity = {name: contract[name] for name in ("source_layout", "source_config_sha256", "source_files_sha256")}
    main = {name: train.get(name.replace("_", "-"), default) for name, default in MAIN_DEFAULTS.items()}
    if train.get("skip-actor-forward-only", False):
        raise CreditError("v21_source_inline_old_policy_unvalidated")
    math_files = (
        "miles/utils/arguments.py",
        "miles/ray/rollout/followup_credit.py",
        "miles/backends/training_utils/followup_credit.py",
        "miles/backends/training_utils/loss_hub/losses.py",
        "miles/backends/training_utils/loss_hub/math_utils.py",
    )
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in math_files}
    return {
        **options,
        **main,
        "agentspark_v21_source_identity": identity,
        "agentspark_v21_loss_options": dict(options),
        "agentspark_v21_main_options": main,
        "agentspark_v21_math_source_files": hashes,
    }


def validate_training_binding(args: Any, contract: dict) -> None:
    expected = training_overrides(contract)
    for key, value in expected.items():
        _require(getattr(args, key, None) == value, "v21_training_contract_mismatch")


def processed_rewards(args: Any, samples: list[Any]) -> list[float]:
    return [sample.metadata["agentspark_v21"]["main_advantage"] for sample in samples]


def prepare_samples(args: Any, samples: list[Any], *, tokenizer: Any, contract: dict) -> list[Any]:
    from examples.agentspark_agentic.v21_reward import mark_followup_text, process_group, validate_score_binding

    validate(args)
    _require(not getattr(args, "agentic_custom_advantage_path", None), "v21_native_spans_must_not_be_overwritten")
    _require(
        not (getattr(args, "partial_rollout", False) and getattr(args, "use_dynamic_global_batch_size", False)),
        "v21_partial_dynamic_global_unvalidated",
    )
    # Copy Sample shells and mutable training metadata; media tensors remain shared read-only.
    working = [copy.copy(sample) for sample in samples]
    for sample in working:
        sample.metadata = copy.deepcopy(sample.metadata)
        _require(not sample.metadata.get("agentspark_v21", {}).get("processed"), "v21_already_processed")
    logical = apply_ownership(working)
    group_counts = defaultdict(int)
    for siblings in logical.values():
        group_counts[siblings[0].group_index] += 1
    _require(all(count == args.n_samples_per_prompt for count in group_counts.values()), "v21_incomplete_prompt_group")
    visible = {}

    def collect(text):
        visible.setdefault(text, None)
        return [0] * len(text)

    for sample in working:
        attach_k3_followup_mask(sample, tokenizer, collect)
    if visible:
        marked = mark_followup_text(contract, list(visible))
        _require(len(marked) == len(visible), "v21_marker_count_mismatch")
        visible = dict(zip(visible, marked, strict=True))
    for sample in working:
        attach_k3_followup_mask(sample, tokenizer, visible.__getitem__)
    rows = []
    for index, siblings in logical.items():
        anchor = next(sample for sample in siblings if sample.metadata["agentspark_training"]["is_anchor"])
        reward = anchor.metadata.get("agentspark_reward", {})
        _require(isinstance(reward.get("metadata"), dict), "v21_verified_reward_required")
        validate_score_binding(reward, anchor.metadata["agentspark_training"]["invocation_sha256"])
        expected_source = {
            name: contract[name] for name in ("source_layout", "source_config_sha256", "source_files_sha256")
        }
        _require(
            reward.get("source_identity") == expected_source
            and reward.get("prompt_bundle_fingerprint") == contract["expected_prompt_bundle_fingerprint"],
            "v21_reward_contract_mismatch",
        )
        _require(
            type(reward.get("base_reward")) in (int, float) and math.isfinite(reward["base_reward"]),
            "v21_base_reward_invalid",
        )
        for sample in siblings:
            _require(sample.metadata.get("agentspark_reward") == reward, "v21_score_evidence_mismatch")
            expected_reward = reward["base_reward"] if sample.metadata["agentspark_training"]["is_anchor"] else 0.0
            _require(sample.get_reward_value(args) == expected_reward, "v21_sparse_reward_mismatch")
        rows.append(
            {
                "index": index,
                "group_index": anchor.group_index,
                "base_reward": anchor.get_reward_value(args),
                "metadata": copy.deepcopy(reward["metadata"]),
                "remove_sample": anchor.remove_sample,
                "followup_token_mask": [
                    v for sample in siblings for v in sample.metadata["agentspark_v21"]["followup_token_mask"]
                ],
            }
        )
    finalized = process_group(contract, rows)
    _require(len(finalized) == len(rows), "v21_group_result_count_mismatch")
    for original, result in zip(rows, finalized, strict=True):
        _require(
            (original["index"], original["group_index"]) == (result["index"], result["group_index"]),
            "v21_group_result_identity_mismatch",
        )
        for sample in logical[result["index"]]:
            for key, value in result["metadata"].items():
                _require(
                    key not in {"agentic_trace", "tools", "chat_template_kwargs"}
                    and not key.startswith("agentspark_"),
                    "v21_policy_metadata_collision",
                )
                sample.metadata[key] = copy.deepcopy(value)
            sample.metadata["agentspark_v21"].update(
                processed=True, main_advantage=result["main_advantage"], raw_reward=result["raw_reward"]
            )
            if not any(sample.metadata["agentspark_v21"]["followup_token_mask"]) or sample.remove_sample:
                sample.metadata["user_signal_token_credit_advantage"] = 0.0
            sample.reward = result["raw_reward"]
            if result.get("remove_sample", False):
                sample.remove_sample = True
                sample.loss_mask = [0] * sample.response_length
    return working


def convert_samples(args: Any, samples: list[Any]):
    """Native custom converter hook; the only full-group policy invocation."""
    from examples.agentspark_agentic.v21_reward import load_reward_contract
    from relax.utils.utils import convert_samples_to_train_data, dict_to_tensordict

    path = os.environ.get("AGENTSPARK_V21_CONTRACT")
    _require(bool(path), "v21_contract_path_required")
    contract = load_reward_contract(path)
    validate_training_binding(args, contract)
    prepared = prepare_samples(args, samples, tokenizer=_tokenizer(args.hf_checkpoint), contract=contract)
    native_args = copy.copy(args)
    native_args.custom_convert_samples_to_train_data_path = None
    native_args.custom_reward_post_process_path = "examples.agentspark_agentic.v21_training.processed_rewards"
    native_args.debug_train_only = True
    data = convert_samples_to_train_data(native_args, prepared)
    data.update(convert_credit(prepared))
    return data if args.debug_train_only else dict_to_tensordict(data, len(prepared))
