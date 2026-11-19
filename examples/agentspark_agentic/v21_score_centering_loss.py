# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""V21 logical-rollout credit with a shared Score Centering token objective."""

from __future__ import annotations

from typing import Any

import torch

from examples.agentspark_agentic import v21_loss
from relax.algorithms.score_centering import compute_score_centering_loss


SC_DEFAULTS = {
    "rollout_top_logprobs_num": None,
    "score_centering_is": "none",
    "score_centering_tis_clip": 2.0,
    "score_centering_mis_low": 0.5,
    "score_centering_mis_high": 5.0,
    "score_centering_correction": True,
}
LOSS_PATH = "examples.agentspark_agentic.v21_score_centering_loss.loss"

# Frozen separately from the legacy reward source; this profile changes the loss only.
SOURCE_FILES = (
    "examples/async_dots_rl/configs/profiles/v21_score_centering_ctr.json",
    "miles_plugins/algorithms/score_centering_v21/extension.py",
    "miles_plugins/algorithms/score_centering_v21/reductions.py",
    "miles/backends/training_utils/loss/hub/score_centering_loss.py",
    "miles/backends/training_utils/loss/hub/score_centering/estimator.py",
    "miles/backends/training_utils/loss/hub/score_centering/importance_sampling.py",
    "miles/backends/training_utils/loss/hub/score_centering/masks.py",
    "miles/backends/training_utils/loss/hub/math_utils.py",
    "tests/fast/backends/training_utils/test_score_centering_followup.py",
)


def training_overrides(config: dict) -> dict:
    import hashlib
    from pathlib import Path

    source = config.get("score_centering_source", {})
    hashes = source.get("files", {})
    if not source.get("root") or set(hashes) != set(SOURCE_FILES):
        raise ValueError("V21 Score Centering requires a frozen SC source manifest")
    if any(
        hashlib.sha256((Path(source["root"]) / name).read_bytes()).hexdigest() != digest
        for name, digest in hashes.items()
    ):
        raise ValueError("V21 Score Centering source changed after binding")
    train = config["train"]
    options = {name: train.get(name.replace("_", "-"), default) for name, default in SC_DEFAULTS.items()}
    if type(options["rollout_top_logprobs_num"]) is not int or options["rollout_top_logprobs_num"] < 1:
        raise ValueError("V21 Score Centering requires an explicit positive candidate count")
    if train.get("advantage-estimator") != "grpo" or train.get("use-tis") is not False:
        raise ValueError("V21 Score Centering requires GRPO reward normalization without PPO TIS")
    return {
        **options,
        "custom_loss_function_path": LOSS_PATH,
        "agentspark_v21_score_centering_options": dict(options),
        "agentspark_v21_score_centering_source_files": hashes,
    }


def validate_binding(args: Any) -> None:
    v21_loss.validate_binding(args)
    bound = getattr(args, "agentspark_v21_score_centering_options", None)
    if not isinstance(bound, dict) or set(bound) != set(SC_DEFAULTS):
        raise ValueError("V21 Score Centering requires source-bound SC options")
    if any(getattr(args, key, None) != value for key, value in bound.items()):
        raise ValueError("V21 Score Centering options changed after source binding")
    if args.custom_loss_function_path != LOSS_PATH or args.advantage_estimator != "grpo":
        raise ValueError("V21 Score Centering requires its custom loss and GRPO reward normalization")
    if type(args.score_centering_correction) is not bool:
        raise ValueError("Score Centering correction must be boolean")
    if any(
        getattr(args, key, False)
        for key in (
            "use_tis",
            "use_opsm",
            "use_unbiased_kl",
            "custom_tis_function_path",
            "use_sampling_support_replay",
        )
    ):
        raise ValueError("V21 Score Centering excludes PPO corrections and sampling-support replay")


def _source(batch: dict, advantages: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.full_like(value, float(flag))
            for value, flag in zip(
                batch["followup_token_advantages"], batch["followup_clarification_credit_flags"], strict=True
            )
        ]
    ).to(device=advantages.device)


def _followup_policy(args: Any, batch: dict, term: torch.Tensor, advantages: torch.Tensor, metrics: dict):
    reduce = v21_loss._reducer(args, batch, batch["followup_token_masks"])
    source = _source(batch, advantages)
    token_loss = -advantages * term
    total, ctr, clarification = reduce(token_loss), reduce(token_loss * (1 - source)), reduce(token_loss * source)
    weight = v21_loss.option(args, "followup_token_credit_loss_weight")
    clarification_weight = v21_loss.option(args, "followup_clarification_credit_loss_weight", weight)
    loss = weight * total if weight == clarification_weight else weight * ctr + clarification_weight * clarification
    log = {
        "followup_auxiliary_loss": total.detach(),
        "followup_user_signal_loss": ctr.detach(),
        "followup_clarification_quality_loss": clarification.detach(),
        "followup_user_signal_weighted_loss": (weight * ctr).detach(),
        "followup_clarification_quality_weighted_loss": (clarification_weight * clarification).detach(),
        "followup_user_signal_positive_advantage": reduce(advantages.clamp_min(0) * (1 - source)).detach(),
        "followup_user_signal_negative_advantage": reduce((-advantages).clamp_min(0) * (1 - source)).detach(),
        **{f"followup_{key}": reduce(value).detach() for key, value in metrics.items()},
    }
    correction = metrics["sc_correction"] if args.score_centering_correction else torch.zeros_like(term)
    log["followup_sc_applied_correction_loss"] = reduce(advantages * correction).detach()
    return loss, log


def _reference(args: Any, batch: dict, log_probs: torch.Tensor, active: torch.Tensor, reduce):
    reference = torch.cat(batch["ref_log_probs"]).detach()
    valid = active & torch.isfinite(log_probs) & torch.isfinite(reference) & ((log_probs - reference).abs() <= 40)
    # Match Miles SC: sanitize before exp, including when main credit is zero.
    actor = torch.where(valid, log_probs, 0.0)
    reference = torch.where(valid, reference, 0.0)
    kl = v21_loss.approximate_kl(actor, reference, args.kl_loss_type)
    kl = torch.where(valid, torch.nan_to_num(kl, nan=0, posinf=0, neginf=0), 0)
    main = reduce(kl)
    result = main * args.kl_loss_coef
    log = {"kl_loss": main.detach(), "kl_invalid_fraction": reduce((active & ~valid).float()).detach()}
    coefficient = v21_loss.option(args, "followup_reference_kl_loss_coef")
    clarify_coefficient = v21_loss.option(args, "followup_clarification_reference_kl_loss_coef", coefficient)
    if coefficient or clarify_coefficient:
        local = v21_loss._reducer(args, batch, batch["followup_token_masks"])
        source = _source(batch, kl)
        total, ctr, clarify = local(kl), local(kl * (1 - source)), local(kl * source)
        weighted = (
            coefficient * total
            if coefficient == clarify_coefficient
            else coefficient * ctr + clarify_coefficient * clarify
        )
        result = result + weighted
        log.update(
            {
                "followup_reference_kl_loss": total.detach(),
                "followup_user_signal_reference_kl_loss": ctr.detach(),
                "followup_clarification_reference_kl_loss": clarify.detach(),
                "followup_reference_kl_weighted_loss": weighted.detach(),
            }
        )
    return result, log


def loss(args: Any, batch: dict, logits: torch.Tensor, reducer):
    from relax.backends.megatron.loss import get_log_probs_and_entropy
    from relax.backends.megatron.score_centering import build_score_centering_context

    validate_binding(args)
    batch = v21_loss.prepare_batch(args, batch)
    with_entropy = args.entropy_coef != 0 or getattr(args, "observe_training_entropy", False)
    _, probabilities = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=with_entropy,
        gather_topk_token_ids=batch["rollout_topk_token_ids"],
        max_seq_lens=batch.get("max_seq_lens"),
        padded_total_lengths=batch.get("padded_total_lengths"),
        cp_output_is_zigzag=batch.get("cp_output_is_zigzag", False),
        dynamic_cp_size=batch.get("dynamic_cp_size"),
        dynamic_cp_rank=batch.get("dynamic_cp_rank"),
    )
    context = build_score_centering_context(args, batch, probabilities)
    log_probs = torch.cat(probabilities["log_probs"])
    active = context["loss_mask"].bool()
    main_advantage = torch.where(active, torch.cat(batch["advantages"]).detach(), 0.0)
    local_mask = torch.cat(
        [v21_loss._local(mask, batch, i, args) for i, mask in enumerate(batch["followup_token_masks"])]
    ).bool()
    auxiliary_advantage = torch.where(active & local_mask, torch.cat(batch["followup_token_advantages"]).detach(), 0.0)
    credit_active = active & ((main_advantage != 0) | (auxiliary_advantage != 0))
    # Unit credit exposes one score term for both independently reduced objectives.
    negative_term, metrics = compute_score_centering_loss(
        log_probs=log_probs,
        advantages=torch.ones_like(log_probs),
        rollout_log_probs=context["rollout_log_probs"].detach(),
        train_topk_log_probs=context["train_topk_log_probs"],
        rollout_topk_log_probs=context["rollout_topk_log_probs"].detach(),
        loss_mask=credit_active,
        mode=args.score_centering_is,
        tis_clip=args.score_centering_tis_clip,
        mis_low=args.score_centering_mis_low,
        mis_high=args.score_centering_mis_high,
        center_scores=args.score_centering_correction,
    )
    term = -negative_term
    main = reducer(-main_advantage * term)
    auxiliary, log = _followup_policy(args, batch, term, auxiliary_advantage, metrics)
    total = main + auxiliary
    if with_entropy:
        entropy = reducer(torch.where(active, torch.cat(probabilities["entropy"]), 0.0))
        total = total - args.entropy_coef * entropy
        log["entropy_loss"] = entropy.detach()
    if args.use_kl_loss:
        reference, reference_log = _reference(args, batch, log_probs, active, reducer)
        total = total + reference
        log.update(reference_log)
    correction = metrics["sc_correction"] if args.score_centering_correction else torch.zeros_like(term)
    log.update({key: reducer(value).detach() for key, value in metrics.items()})
    log.update(
        sc_correction_enabled=reducer(torch.full_like(term, float(args.score_centering_correction))).detach(),
        sc_applied_correction_loss=reducer(main_advantage * correction).detach(),
        pg_loss=main.detach(),
    )
    if log_probs.numel() == 0:
        total = total + 0 * logits.sum()
    log["loss"] = total.detach()
    return total, log


loss.extra_training_fields = v21_loss.FIELDS
loss.supports_static_cp_sample_mean = True
loss.supports_score_centering = True
# Native Agentic Forest keeps image observations outside response-token candidate rows.
loss.supports_score_centering_multimodal = True
