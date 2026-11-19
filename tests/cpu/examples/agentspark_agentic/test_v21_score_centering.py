# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""V21 SC loss/gradient parity with the frozen Miles independent dense
oracle."""

import ast
import os
from pathlib import Path

import pytest
import torch

from examples.agentspark_agentic import v21_loss
from examples.agentspark_agentic import v21_score_centering_loss as sc
from relax.utils.score_centering import score_centering_enabled
from tests.cpu.backends.megatron.test_score_centering_loss import _backend
from tests.cpu.examples.agentspark_agentic.test_v21_training import _args as grpo_args


def _args(**overrides):
    args = grpo_args(
        custom_loss_function_path=sc.LOSS_PATH,
        followup_token_credit_loss_weight=0.15,
        followup_clarification_credit_loss_weight=0.1,
        followup_token_credit_soft_limit=0,
        followup_token_credit_hard_limit=0,
        followup_token_credit_per_token_weight_cap=0.0,
        entropy_coef=0.0,
        observe_training_entropy=False,
        rollout_temperature=0.8,
        score_centering_vocab_size=5,
        log_probs_chunk_size=-1,
        **{**sc.SC_DEFAULTS, "rollout_top_logprobs_num": 3, **overrides},
    )
    args.agentspark_v21_score_centering_options = {name: getattr(args, name) for name in sc.SC_DEFAULTS}
    return args


def _fixture(*, clarification=False, main_zero=False, auxiliary_zero=False):
    generator = torch.Generator().manual_seed(93)
    logits = torch.randn(1, 16, 8, generator=generator)
    logits[..., 5:] = 30  # Real scoring must exclude vocabulary padding.
    logits.requires_grad_()
    q = torch.randn(7, 5, generator=generator).softmax(-1)
    ids = torch.tensor([4, 1, 2, 3, 0, 4, 1])
    tokens = torch.cat((torch.tensor([0, 1, 2, 3]), ids))
    heads = q.topk(3, dim=-1).indices
    batch = dict(
        total_lengths=[11],
        response_lengths=[7],
        padded_total_lengths=[16],
        unconcat_tokens=[tokens],
        loss_masks=[torch.tensor([1, 1, 0, 1, 1, 1, 1])],
        sample_index_mask_sums=[9],
        followup_token_masks=[[0, 1, 0, 1, 0, 1, 0]],
        followup_quality_masks=[[0] * 7],
        followup_token_mask_sums=[5],
        followup_token_advantage_scalars=[0.0 if auxiliary_zero else 0.7],
        followup_quality_violation_flags=[False],
        followup_clarification_credit_flags=[clarification],
        advantages=[torch.zeros(7) if main_zero else torch.tensor([0.7, 0.0, float("nan"), -0.4, 0.8, -0.2, 0.4])],
        log_probs=[torch.full((7,), -2.0)],
        rollout_log_probs=[q.log().gather(1, ids[:, None]).squeeze(1).requires_grad_()],
        rollout_topk_token_ids=[heads],
        rollout_topk_log_probs=[q.log().gather(1, heads).requires_grad_()],
        ref_log_probs=[torch.full((7,), -1.7)],
    )
    return batch, logits, q, ids


def _oracle():
    root = os.environ.get("AGENTSPARK_V21_SC_ORACLE_SOURCE")
    if not root:
        pytest.skip("Frozen Miles Score Centering oracle source is required")
    path = Path(root) / "tests/fast/backends/training_utils/test_score_centering_followup.py"
    nodes = [
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_dense_oracle"
    ]
    assert len(nodes) == 1
    namespace = {"torch": torch}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_dense_oracle"]


def _reduce(cp, args, batch):
    return cp.get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        False,
        args.qkv_format,
        batch.get("max_seq_lens"),
        batch.get("padded_total_lengths"),
        dynamic_cp_size=batch.get("dynamic_cp_size"),
        dynamic_cp_rank=batch.get("dynamic_cp_rank"),
        sample_denoms=batch["sample_index_mask_sums"],
    )


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("clarification", [False, True])
def test_packed_loss_and_gradient_match_miles_oracle(monkeypatch, mode, centered, clarification):
    _, cp = _backend(monkeypatch)
    args = _args(score_centering_is=mode, score_centering_correction=centered)
    batch, logits, q, ids = _fixture(clarification=clarification)
    actual, metrics = sc.loss(args, batch, logits, _reduce(cp, args, batch))
    expected_logits = logits.detach()[0, 3:10, :5].clone().requires_grad_()
    reference_batch = v21_loss.prepare_batch(args, batch)
    expected = _oracle()(args, reference_batch, expected_logits, q, ids)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=3e-6)
    gradient, q_gradient, head_gradient = torch.autograd.grad(
        actual, (logits, batch["rollout_log_probs"][0], batch["rollout_topk_log_probs"][0]), allow_unused=True
    )
    assert q_gradient is head_gradient is None
    torch.testing.assert_close(
        gradient[0, 3:10, :5], torch.autograd.grad(expected, expected_logits)[0], atol=2e-7, rtol=3e-6
    )
    assert gradient[..., 5:].count_nonzero() == 0
    assert gradient[0, 5].count_nonzero() == 0  # Masked observation with poisoned main advantage.
    assert all(not value.requires_grad for value in metrics.values())
    if not centered:
        assert metrics["followup_sc_applied_correction_loss"] == metrics["sc_applied_correction_loss"] == 0


@pytest.mark.parametrize("cp_size,dynamic", [(2, False), (4, False), (8, False), (8, True)])
def test_cp_partitions_sum_to_full_loss_and_gradient(monkeypatch, cp_size, dynamic):
    native, cp = _backend(monkeypatch)
    args = _args(score_centering_is="tis")
    batch, full, _, ids = _fixture()
    partitioned = full.detach().clone().requires_grad_()
    expected = sc.loss(args, batch, full, _reduce(cp, args, batch))[0]
    full_logp = (partitioned[0, 3:10, :5] / args.rollout_temperature).log_softmax(-1)
    sampled = full_logp.gather(1, ids[:, None]).squeeze(1)
    heads = full_logp.gather(1, batch["rollout_topk_token_ids"][0])
    actual = partitioned.sum() * 0
    for rank in range(cp_size):
        local = dict(batch)
        if dynamic:
            local.update(dynamic_cp_size=cp_size, dynamic_cp_rank=rank)
        else:
            monkeypatch.setattr(cp.mpu, "get_context_parallel_world_size", lambda: cp_size)
            monkeypatch.setattr(cp.mpu, "get_context_parallel_rank", lambda rank=rank: rank)
        for name in (
            "log_probs",
            "rollout_log_probs",
            "ref_log_probs",
            "advantages",
            "rollout_topk_token_ids",
            "rollout_topk_log_probs",
        ):
            local[name] = [v21_loss._local(batch[name][0], local, 0, args)]
        scored = {
            "log_probs": [v21_loss._local(sampled, local, 0, args)],
            "topk_log_probs": [v21_loss._local(heads, local, 0, args)],
        }
        monkeypatch.setattr(native, "get_log_probs_and_entropy", lambda *a, scored=scored, **k: (None, scored))
        actual = actual + sc.loss(args, local, partitioned, _reduce(cp, args, local))[0]
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=3e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual, partitioned)[0], torch.autograd.grad(expected, full)[0], atol=2e-7, rtol=3e-6
    )


@pytest.mark.parametrize("auxiliary_zero", [False, True])
def test_zero_main_credit_keeps_ctr_or_local_reference_gradient(monkeypatch, auxiliary_zero):
    _, cp = _backend(monkeypatch)
    args = _args()
    args.kl_loss_coef = 0
    if not auxiliary_zero:
        args.followup_reference_kl_loss_coef = args.followup_clarification_reference_kl_loss_coef = 0
    args.agentspark_v21_main_options = {name: getattr(args, name) for name in v21_loss.MAIN_DEFAULTS}
    args.agentspark_v21_loss_options = {
        name: getattr(args, name) for name in (*v21_loss.DEFAULTS, "followup_token_credit")
    }
    batch, logits, q, ids = _fixture(main_zero=True, auxiliary_zero=auxiliary_zero)
    actual, metrics = sc.loss(args, batch, logits, _reduce(cp, args, batch))
    expected = _oracle()(args, v21_loss.prepare_batch(args, batch), logits[0, 3:10, :5], q, ids)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=3e-6)
    gradient = torch.autograd.grad(actual, logits)[0][0, 3:10, :5]
    mask = torch.tensor(batch["followup_token_masks"][0]).bool()
    assert gradient[mask].abs().sum() > 0
    assert gradient[~mask].count_nonzero() == 0
    assert metrics["pg_loss"] == 0


def test_nonfinite_reference_is_masked_before_exp(monkeypatch):
    _, cp = _backend(monkeypatch)
    args = _args()
    batch, logits, _, _ = _fixture(main_zero=True, auxiliary_zero=True)
    batch["ref_log_probs"][0][1:5] = torch.tensor([float("inf"), float("nan"), -1000, 1000])
    actual, metrics = sc.loss(args, batch, logits, _reduce(cp, args, batch))
    assert torch.isfinite(actual)
    assert torch.isfinite(torch.autograd.grad(actual, logits)[0]).all()
    assert metrics["kl_invalid_fraction"] > 0


def test_custom_sc_opt_in_preserves_grpo_and_detects_changed_options():
    args = _args()
    assert score_centering_enabled(args)
    assert args.advantage_estimator == "grpo"
    sc.validate_binding(args)
    args.score_centering_is = "tis"
    with pytest.raises(ValueError, match="changed after source binding"):
        sc.validate_binding(args)
    args.custom_loss_function_path = "examples.agentspark_agentic.v21_loss.loss"
    assert not score_centering_enabled(args)


def test_source_binding_rejects_mutated_frozen_math(tmp_path):
    files = {}
    for name in sc.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen")
        import hashlib

        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    config = {
        "score_centering_source": {"root": str(tmp_path), "files": files},
        "train": {
            "advantage-estimator": "grpo",
            "use-tis": False,
            "rollout-top-logprobs-num": 16,
        },
    }
    assert sc.training_overrides(config)["custom_loss_function_path"] == sc.LOSS_PATH
    (tmp_path / sc.SOURCE_FILES[0]).write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        sc.training_overrides(config)


def test_custom_sc_captures_candidates_and_transports_with_credit(monkeypatch):
    from relax.utils.score_centering import configure_score_centering_request
    from relax.utils.training.data_fields import build_data_fields
    from relax.utils.utils import convert_samples_to_train_data
    from tests.cpu.utils.test_score_centering_capture import _sample

    args = _args(rollout_top_logprobs_num=2)
    args.score_centering_vocab_size = 8
    args.custom_convert_samples_to_train_data_path = None
    args.custom_reward_post_process_path = None
    args.rewards_normalization = False
    args.reward_key = None
    args.debug_train_only = True
    args.rollout_function_path = None
    args.multimodal_keys = {"image": "images"}
    args.use_agentic_rollout = True
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "0")
    request = {"sampling_params": {}}
    configure_score_centering_request(args, request)
    assert request["top_logprobs_num"] == 2 and request["return_logprob"]
    assert request["sampling_params"]["temperature"] == 0.8
    assert request["sampling_params"]["top_p"] == 1 and request["sampling_params"]["top_k"] == -1
    sample = _sample()
    sample.reward = 0.5
    sample.multimodal_train_inputs = {
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "image_grid_thw": torch.tensor([[1, 1, 1]]),
    }
    sample.loss_mask = [1, 1]
    data = convert_samples_to_train_data(args, [sample])
    assert "rollout_topk_token_ids" in data and "rollout_topk_log_probs" in data
    assert set(sc.loss.extra_training_fields) <= set(build_data_fields(args))
    assert {"rollout_topk_token_ids", "rollout_topk_log_probs", "multimodal_train_inputs"} <= set(
        build_data_fields(args)
    )
    sample.rollout_topk_log_probs = None
    with pytest.raises(ValueError, match="Score Centering"):
        convert_samples_to_train_data(args, [sample])


@pytest.mark.parametrize("sampling", [{"top_k": 4}, {"top_p": 0.9}, {"min_p": 0.1}])
def test_v21_sc_rejects_per_call_filtered_sampling(sampling, monkeypatch):
    from relax.utils.score_centering import configure_score_centering_request

    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "0")
    with pytest.raises(ValueError, match="requires unfiltered sampling"):
        configure_score_centering_request(_args(), {"sampling_params": sampling})
