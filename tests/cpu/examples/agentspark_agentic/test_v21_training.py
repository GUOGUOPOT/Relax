# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks for logical ownership and frozen formal V21 numerical parity.

The oracle executes the frozen source functions themselves. Only model scoring
and parallel hardware adapters are injected; no checkpoint weights are loaded.
"""

import ast
import copy
import json
import math
import os
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.agentspark_agentic import v21_loss, v21_training
from relax.utils.types import Sample
from tests.cpu.backends.megatron.test_score_centering_loss import _backend


_source_path = os.environ.get("AGENTSPARK_V21_ORACLE_SOURCE")
if not _source_path and os.environ.get("AGENTSPARK_V21_CONTRACT"):
    _source_path = json.loads(Path(os.environ["AGENTSPARK_V21_CONTRACT"]).read_text())["miles_source"]
FROZEN = Path(_source_path) if _source_path else None


def _source_functions(relative, names, namespace):
    if FROZEN is None:
        pytest.skip("Explicit frozen V21 oracle source or contract is required")
    path = FROZEN / relative
    if not path.is_file():
        pytest.skip("Frozen formal V21 source is not configured on this CPU host")
    tree = ast.parse(path.read_text())
    nodes = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), namespace)


def _args(**overrides):
    options = dict(v21_loss.DEFAULTS)
    options.update(
        followup_token_credit=True,
        followup_reference_kl_loss_coef=0.01,
        followup_clarification_reference_kl_loss_coef=0.02,
        followup_token_credit_soft_limit=2,
        followup_token_credit_hard_limit=5,
        followup_token_credit_per_token_weight_cap=0.1,
        followup_clarification_credit_loss_weight=0.4,
    )
    values = dict(
        advantage_estimator="grpo",
        loss_type="custom_loss",
        qkv_format="thd",
        calculate_per_token_loss=False,
        use_opd=False,
        opd_token_selection="sampled",
        use_kl_loss=True,
        kl_loss_coef=0.003,
        kl_loss_type="low_var_kl",
        use_unbiased_kl=False,
        entropy_coef=0.02,
        observe_training_entropy=True,
        eps_clip=0.2,
        eps_clip_high=0.28,
        eps_clip_c=3.0,
        use_rollout_logprobs=False,
        true_on_policy_mode=False,
        skip_actor_forward_only=False,
        use_opsm=False,
        use_tis=False,
        get_mismatch_metrics=False,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        rollout_temperature=1.0,
        allgather_cp=False,
        dump_details=None,
        n_samples_per_prompt=2,
        agentic_custom_advantage_path=None,
        custom_loss_function_path="examples.agentspark_agentic.v21_loss.loss",
        **options,
    )
    for name, value in v21_loss.MAIN_DEFAULTS.items():
        values.setdefault(name, value)
    values.update(overrides)
    values["agentspark_v21_main_options"] = {name: values[name] for name in v21_loss.MAIN_DEFAULTS}
    values["agentspark_v21_loss_options"] = {
        name: values[name] for name in (*v21_loss.DEFAULTS, "followup_token_credit")
    }
    return Namespace(**values)


def _batch():
    lengths = [5, 7, 3]
    return {
        "total_lengths": [8, 12, 8],
        "response_lengths": lengths,
        "unconcat_tokens": [torch.arange(n) for n in [8, 12, 8]],
        "loss_masks": [torch.tensor(v) for v in [[1, 1, 1, 0, 0], [1, 0, 1, 1, 1, 0, 0], [1, 1, 1]]],
        "sample_index_mask_sums": [7, 7, 3],
        "followup_token_masks": [[0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 1, 0]],
        "followup_quality_masks": [[0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 1, 0]],
        "followup_token_mask_sums": [4, 4, 1],
        "followup_token_advantage_scalars": [0.9, 0.9, -0.2],
        "followup_quality_violation_flags": [True, True, False],
        "followup_clarification_credit_flags": [False, False, True],
        "advantages": [torch.full((n,), a) for n, a in zip(lengths, [1.2, 1.2, -0.4])],
        "log_probs": list(torch.linspace(-0.1, -2, 15).split(lengths)),
        "rollout_log_probs": list(torch.linspace(-0.2, -1.6, 15).split(lengths)),
        "ref_log_probs": list(torch.linspace(-0.3, -2.4, 15).split(lengths)),
    }


def _oracle(cp):
    def reduce(total, response, masks, token=False, qkv="thd", maxima=None, *, denominators=None):
        return cp.get_sum_of_sample_mean(total, response, masks, token, qkv, maxima, sample_denoms=denominators)

    ns = dict(
        torch=torch,
        math=math,
        Namespace=Namespace,
        Callable=Callable,
        Any=Any,
        RolloutBatch=dict,
        _LOG_RATIO_EXP_CLAMP=20.0,
        get_parallel_state=lambda: SimpleNamespace(cp=SimpleNamespace(size=1)),
        get_local_response_loss_masks=lambda total, response, masks, *rest: masks,
        get_sum_of_sample_mean=reduce,
        compute_ess_ratio_contribution=lambda **kwargs: torch.tensor(0.0),
    )
    _source_functions(
        "miles/backends/training_utils/loss_hub/math_utils.py",
        ["_safe_clamp_log_ratio", "_safe_exp_neg_ppo_kl", "compute_policy_loss", "compute_approx_kl"],
        ns,
    )
    _source_functions(
        "miles/backends/training_utils/followup_credit.py",
        [
            "suppress_invalid_followup_positive_advantages",
            "cap_followup_token_normalization",
            "prepare_followup_advantages",
        ],
        ns,
    )
    _source_functions("miles/backends/training_utils/loss_hub/corrections.py", ["vanilla_tis_function"], ns)
    _source_functions(
        "miles/backends/training_utils/loss_hub/losses.py",
        ["_followup_coefficient", "_followup_clarification_source", "policy_loss_function"],
        ns,
    )
    return ns


@pytest.mark.parametrize("tis", [False, True])
@pytest.mark.parametrize("same_weights", [False, True])
def test_complete_loss_and_gradient_match_frozen_formal_source(monkeypatch, tis, same_weights):
    native, cp = _backend(monkeypatch)
    args = _args(use_tis=tis, tis_clip=2.0, tis_clip_low=0.0)
    if same_weights:
        args.followup_clarification_credit_loss_weight = args.followup_token_credit_loss_weight
        args.followup_clarification_reference_kl_loss_coef = args.followup_reference_kl_loss_coef
        args.agentspark_v21_loss_options = {
            name: getattr(args, name) for name in (*v21_loss.DEFAULTS, "followup_token_credit")
        }
    batch = _batch()
    current = torch.linspace(-0.2, -2.2, 15).requires_grad_()
    expected_current = current.detach().clone().requires_grad_()

    def scores(value):
        return {
            "log_probs": list(value.split(batch["response_lengths"])),
            "entropy": list((value.square() * 0.01).split(batch["response_lengths"])),
        }

    monkeypatch.setattr(native, "get_log_probs_and_entropy", lambda logits, **kwargs: (None, scores(logits)))
    reducer = cp.get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        sample_denoms=batch["sample_index_mask_sums"],
    )
    actual, metrics = v21_loss.loss(args, batch, current, reducer)
    oracle = _oracle(cp)
    reference_batch = copy.deepcopy(batch)
    reference_batch["rollout_mask_sums"] = reference_batch["sample_index_mask_sums"]
    for field in ("followup_token_masks", "followup_quality_masks"):
        reference_batch[field] = [torch.tensor(v) for v in reference_batch[field]]
    reference_batch["advantages"] = oracle["prepare_followup_advantages"](
        args, reference_batch, reference_batch["advantages"]
    )
    oracle["get_log_probs_and_entropy"] = lambda logits, **kwargs: scores(logits)
    expected, reference_metrics = oracle["policy_loss_function"](args, reference_batch, expected_current, reducer)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    for key in reference_metrics.keys() & metrics.keys():
        torch.testing.assert_close(metrics[key], reference_metrics[key], atol=2e-7, rtol=2e-6)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(current.grad, expected_current.grad, atol=2e-7, rtol=2e-6)
    assert not torch.any(current.grad[torch.cat(batch["loss_masks"]) == 0])


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("geometry", ["thd", "padded", "bshd"])
def test_cp_partitions_sum_to_full_loss_and_gradient(monkeypatch, dynamic, geometry):
    native, cp = _backend(monkeypatch)
    args = _args()
    batch = _batch()
    args.qkv_format = "bshd" if geometry == "bshd" else "thd"
    if geometry == "bshd":
        batch["max_seq_lens"] = [16] * 3
    if geometry == "padded":
        batch["padded_total_lengths"] = [16] * 3
    full = torch.linspace(-0.2, -2.2, 15).requires_grad_()
    split = full.detach().clone().requires_grad_()
    lengths = batch["response_lengths"]

    def execute(value, local_batch):
        chunks = list(value.split(lengths))
        local = [v21_loss._local(v, local_batch, i, args) for i, v in enumerate(chunks)]
        monkeypatch.setattr(
            native,
            "get_log_probs_and_entropy",
            lambda logits, **kwargs: (None, {"log_probs": local, "entropy": [x.square() * 0.01 for x in local]}),
        )
        reducer = cp.get_sum_of_sample_mean(
            local_batch["total_lengths"],
            lengths,
            local_batch["loss_masks"],
            False,
            args.qkv_format,
            local_batch.get("max_seq_lens"),
            local_batch.get("padded_total_lengths"),
            dynamic_cp_size=local_batch.get("dynamic_cp_size"),
            dynamic_cp_rank=local_batch.get("dynamic_cp_rank"),
            sample_denoms=local_batch["sample_index_mask_sums"],
        )
        return v21_loss.loss(args, local_batch, value, reducer)[0]

    reference = execute(full, batch)
    actual = split.sum() * 0
    for rank in range(2):
        if dynamic:
            partition = {**batch, "dynamic_cp_size": 2, "dynamic_cp_rank": rank}
        else:
            monkeypatch.setattr(cp.mpu, "get_context_parallel_world_size", lambda: 2)
            monkeypatch.setattr(cp.mpu, "get_context_parallel_rank", lambda rank=rank: rank)
            partition = dict(batch)
        for field in ("log_probs", "rollout_log_probs", "ref_log_probs", "advantages"):
            partition[field] = [v21_loss._local(v, partition, i, args) for i, v in enumerate(batch[field])]
        actual = actual + execute(split, partition)
    torch.testing.assert_close(actual, reference, atol=2e-7, rtol=2e-6)
    actual.backward()
    reference.backward()
    torch.testing.assert_close(split.grad, full.grad, atol=2e-7, rtol=2e-6)


def _sample(index, segment, owned, turns, spans, *, count=2, all_hashes=None):
    length = spans[-1][1]
    mask = [0] * length
    for start, end in spans:
        mask[start:end] = [1] * (end - start)
    return Sample(
        index=index,
        group_index=0,
        session_id=f"session-{index}",
        response_length=length,
        tokens=list(range(length + 3)),
        loss_mask=mask,
        custom_advantage=spans,
        metadata={
            "agentspark_training": {
                "schema_version": 1,
                "segment_index": segment,
                "segment_count": count,
                "is_anchor": segment == count - 1,
                "owned_resp_state_hashes": owned,
                "all_resp_state_hashes": all_hashes or ["r0", "r1", "r2"],
                "invocation_sha256": "a" * 64,
            },
            "agentic_trace": {
                "turns": [
                    {"resp_state_hash": h, "response_length": end - start}
                    for h, (start, end) in zip(turns, spans, strict=True)
                ]
            },
        },
    )


def test_branch_ownership_masks_shared_ancestor_once_without_changing_tokens():
    first = _sample(3, 0, ["r0", "r1"], ["r0", "r1"], [[0, 2], [3, 5]])
    second = _sample(3, 1, ["r2"], ["r0", "r2"], [[0, 2], [3, 6]])
    tokens = copy.deepcopy([first.tokens, second.tokens])
    v21_training.apply_ownership([second, first])
    assert first.loss_mask == [1, 1, 0, 1, 1]
    assert second.loss_mask == [0, 0, 0, 1, 1, 1]
    assert [first.tokens, second.tokens] == tokens
    assert second.custom_advantage == [[0, 2], [3, 6]]


@pytest.mark.parametrize("failure", ["duplicate", "missing", "span", "observation", "identity"])
def test_ownership_rejects_unproven_or_duplicated_training_tokens(failure):
    first = _sample(3, 0, ["r0", "r1"], ["r0", "r1"], [[0, 2], [3, 5]])
    second = _sample(3, 1, ["r2"], ["r0", "r2"], [[0, 2], [3, 6]])
    if failure == "duplicate":
        second.metadata["agentspark_training"]["owned_resp_state_hashes"] += ["r0"]
    if failure == "missing":
        second.metadata["agentspark_training"]["owned_resp_state_hashes"] = []
    if failure == "span":
        second.custom_advantage[1][1] = 5
    if failure == "observation":
        second.loss_mask[2] = 1
    if failure == "identity":
        second.session_id = "other"
    with pytest.raises(v21_training.CreditError):
        v21_training.apply_ownership([first, second])


def test_transport_matches_frozen_source_and_retains_quality_mask_after_eligibility_gate():
    samples = [
        _sample(3, 0, ["r0", "r1"], ["r0", "r1"], [[0, 2], [3, 5]]),
        _sample(3, 1, ["r2"], ["r0", "r2"], [[0, 2], [3, 6]]),
    ]
    v21_training.apply_ownership(samples)
    for sample in samples:
        sample.metadata.update(
            reward_policy_version="quality_first_v1",
            quality_first_local_kl_eligible=False,
            user_signal_token_credit_advantage=0.0,
            followup_policy_version="quality_first_v1",
            contract_followup_local_quality_violation=True,
        )
        sample.metadata["agentspark_v21"] = {"followup_token_mask": list(sample.loss_mask)}
        sample.followup_token_mask = list(sample.loss_mask)
        sample.rollout_id = sample.index
    ns = dict(
        math=math,
        _FOLLOWUP_QUALITY_POLICY_VERSIONS=frozenset(
            {"v2", "v3-type-routing", "v5-scenario-routing", "quality_first_v1"}
        ),
    )
    _source_functions(
        "miles/ray/rollout/followup_credit.py", ["_followup_quality_violation", "convert_followup_credit"], ns
    )
    actual = v21_training.convert_credit(samples)
    expected = ns["convert_followup_credit"](samples, [s.loss_mask for s in samples], [s.index for s in samples])
    assert actual == expected
    assert sum(map(sum, actual["followup_token_masks"])) == 0
    assert sum(map(sum, actual["followup_quality_masks"])) == 7


def test_generic_extension_fields_are_opt_in_and_survive_tensordict_selection(monkeypatch):
    from relax.utils.training.data_fields import build_data_fields
    from relax.utils.training.loss_extensions import custom_loss_fields
    from relax.utils.utils import dict_to_tensordict

    args = _args(multimodal_keys=None, fully_async=False, use_critic=False, use_rollout_routing_replay=False)
    fields = build_data_fields(args)
    assert all(name in fields for name in v21_loss.FIELDS)
    args.loss_type = "policy_loss"
    assert custom_loss_fields(args) == ()
    assert all(name not in build_data_fields(args) for name in v21_loss.FIELDS)
    batch = _batch()
    rows = {name: batch[name] for name in v21_loss.FIELDS}
    tensor = dict_to_tensordict(rows, 3)
    from transfer_queue.utils.serial_utils import decode, encode

    from tests.cpu.backends.megatron.test_data_vpp import _load_data_module

    restored = decode(encode(tensor))
    unpacked = {name: list(values.unbind()) for name, values in restored.items()}
    data = _load_data_module(monkeypatch)
    selected = data.DataIterator(unpacked, micro_batch_indices=[[2, 0, 1]]).get_next(v21_loss.FIELDS)
    assert [row.tolist() for row in selected["followup_token_masks"]] == [
        rows["followup_token_masks"][i] for i in [2, 0, 1]
    ]
    assert [int(v) for v in selected["followup_token_mask_sums"]] == [1, 4, 4]


def test_loss_rejects_unsupported_normalization_and_changed_source_options(monkeypatch):
    _backend(monkeypatch)
    with pytest.raises(ValueError, match="logical-rollout"):
        v21_loss.prepare_batch(_args(calculate_per_token_loss=True), _batch())
    args = _args()
    args.followup_reference_kl_loss_coef = 0
    with pytest.raises(ValueError, match="changed after source binding"):
        v21_loss.loss(args, _batch(), torch.zeros(15), lambda v: v.sum())


def test_masked_nonfinite_scores_match_frozen_loss_without_zero_times_nan(monkeypatch):
    native, cp = _backend(monkeypatch)
    args = _args(entropy_coef=0.0, observe_training_entropy=False)
    batch = _batch()
    current = torch.linspace(-0.2, -2.2, 15)
    inactive = torch.cat(batch["loss_masks"]) == 0
    current[inactive] = float("nan")
    current.requires_grad_()
    reference = current.detach().clone().requires_grad_()

    def score(values):
        chunks = list(values.split(batch["response_lengths"]))
        return {"log_probs": chunks, "entropy": chunks}

    monkeypatch.setattr(native, "get_log_probs_and_entropy", lambda logits, **kwargs: (None, score(logits)))
    reducer = cp.get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        sample_denoms=batch["sample_index_mask_sums"],
    )
    actual, _ = v21_loss.loss(args, batch, current, reducer)
    oracle = _oracle(cp)
    oracle["get_log_probs_and_entropy"] = lambda logits, **kwargs: score(logits)
    expected_batch = copy.deepcopy(batch)
    expected_batch["rollout_mask_sums"] = batch["sample_index_mask_sums"]
    for name in ("followup_token_masks", "followup_quality_masks"):
        expected_batch[name] = [torch.tensor(v) for v in batch[name]]
    expected_batch["advantages"] = oracle["prepare_followup_advantages"](
        args, expected_batch, expected_batch["advantages"]
    )
    expected, _ = oracle["policy_loss_function"](args, expected_batch, reference, reducer)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual)
    actual.backward()
    expected.backward()
    assert torch.isfinite(current.grad).all()
    assert not torch.any(current.grad[inactive])
    torch.testing.assert_close(current.grad, reference.grad)


def test_quality_gate_survives_zero_auxiliary_eligibility(monkeypatch):
    _backend(monkeypatch)
    batch = _batch()
    batch["followup_token_masks"] = [[0] * n for n in batch["response_lengths"]]
    batch["followup_token_advantage_scalars"] = [0.0] * 3
    batch["followup_token_mask_sums"] = [0] * 3
    prepared = v21_loss.prepare_batch(_args(), batch)
    for before, after, quality, violation in zip(
        batch["advantages"],
        prepared["advantages"],
        batch["followup_quality_masks"],
        batch["followup_quality_violation_flags"],
        strict=True,
    ):
        expected = torch.where(torch.tensor(quality).bool(), before.clamp(max=0), before) if violation else before
        torch.testing.assert_close(after, expected)
    assert all(not torch.any(value) for value in prepared["followup_token_advantages"])


def test_static_cp_sample_mean_requires_explicit_synchronous_capability(monkeypatch):
    from relax.utils.training.loss_extensions import custom_loss_supports_static_cp_sample_mean

    values = dict(context_parallel_size=2, colocate=True, fully_async=False, dynamic_context_parallel=False)
    assert custom_loss_supports_static_cp_sample_mean(_args(**values))
    for key, value in (
        ("loss_type", "policy_loss"),
        ("calculate_per_token_loss", True),
        ("context_parallel_size", 1),
        ("dynamic_context_parallel", True),
        ("fully_async", True),
        ("colocate", False),
    ):
        assert not custom_loss_supports_static_cp_sample_mean(_args(**{**values, key: value}))
    monkeypatch.setattr(v21_loss.loss, "supports_static_cp_sample_mean", "true")
    assert not custom_loss_supports_static_cp_sample_mean(_args(**values))
    monkeypatch.delattr(v21_loss.loss, "supports_static_cp_sample_mean")
    assert not custom_loss_supports_static_cp_sample_mean(_args(**values))
