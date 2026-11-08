# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""V21 GRPO token credit, preserving the pinned formal V21 objective.

Authority: frozen miles/backends/training_utils/followup_credit.py and
loss_hub/{losses,math_utils}.py; CPU tests execute those source functions.

Masks stay full response length in TransferQueue. This loss owns their one
CP slice using the actual packed microbatch geometry, including dynamic CP.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from relax.utils.training.loss_extensions import PolicyLossAddon, PolicyLossContext, PolicyLossExtension


FIELDS = (
    "followup_token_masks",
    "followup_quality_masks",
    "followup_token_advantage_scalars",
    "followup_quality_violation_flags",
    "followup_clarification_credit_flags",
    "followup_token_mask_sums",
)
DEFAULTS = {
    "followup_token_credit_loss_weight": 0.25,
    "followup_clarification_credit_loss_weight": None,
    "followup_token_credit_soft_limit": 40,
    "followup_token_credit_hard_limit": 64,
    "followup_token_credit_tail_penalty": 0.0,
    "followup_token_credit_per_token_weight_cap": 0.0,
    "followup_reference_kl_loss_coef": 0.0,
    "followup_clarification_reference_kl_loss_coef": None,
}

# Explicit formal recipe values plus undeclared options verified against its
# frozen miles/utils/arguments.py; the generator reads declared values again.
MAIN_DEFAULTS = {
    "eps_clip": 0.2,
    "eps_clip_high": 0.28,
    "eps_clip_c": None,
    "entropy_coef": 0.0,
    "use_kl_loss": True,
    "kl_loss_coef": 0.003,
    "kl_loss_type": "low_var_kl",
    "kl_coef": 0.0,
    "use_unbiased_kl": False,
    "normalize_advantages": False,
    "use_tis": True,
    "get_mismatch_metrics": False,
    "tis_clip": 2.0,
    "tis_clip_low": 0.0,
    "use_rollout_logprobs": False,
    "rollout_temperature": 0.8,
    "calculate_per_token_loss": False,
    "use_opsm": False,
    "custom_tis_function_path": None,
    "true_on_policy_mode": False,
}


def option(args: Any, name: str, fallback: float | None = None) -> Any:
    value = getattr(args, name, DEFAULTS[name])
    return fallback if value is None else value


def validate(args: Any) -> None:
    if getattr(args, "advantage_estimator", None) != "grpo" or getattr(args, "loss_type", None) != "custom_loss":
        raise ValueError("V21 requires GRPO with its explicit custom_loss")
    if not getattr(args, "followup_token_credit", False):
        raise ValueError("V21 requires explicit followup_token_credit=true")
    if getattr(args, "calculate_per_token_loss", False):
        raise ValueError("V21 uses logical-rollout denominators, not the per-token training normalizer")
    if getattr(args, "use_opd", False) or getattr(args, "custom_pg_loss_reducer_function_path", None):
        raise ValueError("V21 OPD/custom policy reducer composition has not been validated")
    soft = option(args, "followup_token_credit_soft_limit")
    hard = option(args, "followup_token_credit_hard_limit")
    if type(soft) is not int or type(hard) is not int or soft < 0 or hard < 0 or (soft and hard <= soft):
        raise ValueError("V21 length limits must be nonnegative integers, with hard > soft")
    for name in DEFAULTS:
        value = option(args, name)
        if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
            raise ValueError(f"V21 coefficient must be finite and nonnegative: {name}")
    if (
        any(
            option(args, name, 0)
            for name in ("followup_reference_kl_loss_coef", "followup_clarification_reference_kl_loss_coef")
        )
        and not args.use_kl_loss
    ):
        raise ValueError("V21 local reference KL requires use_kl_loss")


def _local(values: torch.Tensor, batch: dict, index: int, args: Any) -> torch.Tensor:
    from relax.backends.megatron.cp_utils import slice_log_prob_with_cp

    padded = batch.get("padded_total_lengths")
    maxima = batch.get("max_seq_lens")
    return slice_log_prob_with_cp(
        values,
        batch["total_lengths"][index],
        batch["response_lengths"][index],
        args.qkv_format,
        maxima[index] if maxima is not None else None,
        padded_total_length=padded[index] if padded is not None else None,
        dynamic_cp_size=batch.get("dynamic_cp_size"),
        dynamic_cp_rank=batch.get("dynamic_cp_rank"),
    )


def prepare_batch(args: Any, batch: dict) -> dict:
    """Apply quality correction after native whitening, without mutating shared data."""
    from megatron.core import mpu

    validate(args)
    if any(batch.get(name) is None for name in FIELDS):
        raise ValueError("V21 transfer fields are missing")
    result = dict(batch)
    advantages = batch["advantages"]
    if not isinstance(advantages, list):
        raise ValueError("V21 requires per-sample native advantages")
    count = len(advantages)
    if any(len(batch[name]) != count for name in FIELDS):
        raise ValueError("V21 transfer fields are not sample aligned")
    soft, hard = option(args, "followup_token_credit_soft_limit"), option(args, "followup_token_credit_hard_limit")
    tail = option(args, "followup_token_credit_tail_penalty")
    cp = batch.get("dynamic_cp_size", mpu.get_context_parallel_world_size())
    if tail and cp > 1:
        raise ValueError("V21 followup tail penalty is unsupported with CP>1")
    device = advantages[0].device
    for name in ("followup_token_masks", "followup_quality_masks"):
        result[name] = [torch.as_tensor(value, device=device, dtype=torch.int32) for value in batch[name]]
        if any(
            mask.ndim != 1 or mask.numel() != length
            for mask, length in zip(result[name], batch["response_lengths"], strict=True)
        ):
            raise ValueError("V21 masks must retain full response alignment before CP slicing")
    for name in ("log_probs", "ref_log_probs", "rollout_log_probs"):
        if batch.get(name) is not None:
            result[name] = [value.detach() for value in batch[name]]
    corrected, auxiliary = [], []
    weight = option(args, "followup_token_credit_loss_weight")
    cap = option(args, "followup_token_credit_per_token_weight_cap")
    for i, advantage in enumerate(advantages):
        mask = _local(result["followup_token_masks"][i], batch, i, args).bool()
        quality = _local(result["followup_quality_masks"][i], batch, i, args).bool()
        if mask.shape != advantage.shape or quality.shape != advantage.shape:
            raise ValueError("V21 CP mask/advantage shape mismatch")
        violation = batch["followup_quality_violation_flags"][i]
        corrected.append(
            (torch.where(quality, advantage.clamp(max=0), advantage) if violation else advantage).detach()
        )
        scalar = torch.as_tensor(batch["followup_token_advantage_scalars"][i], device=device, dtype=torch.float32)
        total = torch.as_tensor(batch["followup_token_mask_sums"][i], device=device, dtype=torch.float32)
        clarification = batch["followup_clarification_credit_flags"][i]
        if soft and not clarification:
            scalar = torch.where(scalar > 0, scalar * ((hard - total) / (hard - soft)).clamp(0, 1), scalar)
        values = torch.ones_like(advantage, dtype=torch.float32) * scalar
        if soft and tail and not clarification:
            values = values - (mask & (mask.to(torch.int32).cumsum(0) > soft)).to(torch.float32) * tail
        if cap and weight:
            main_total = torch.as_tensor(batch["sample_index_mask_sums"][i], device=device, dtype=torch.float32)
            values = values * (cap * total / (weight * main_total.clamp_min(1))).clamp(0, 1)
        auxiliary.append(values)
    result["advantages"] = corrected
    result["followup_token_advantages"] = auxiliary
    return result


def policy_tokens(args: Any, ppo_kl: torch.Tensor, advantages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Miles safe PPO arithmetic, including asymmetric and dual clipping."""
    ratio = torch.nan_to_num(-ppo_kl.float(), nan=0, posinf=20, neginf=-20).clamp(-20, 20).exp()
    first = -ratio * advantages
    second = -ratio.clamp(1 - args.eps_clip, 1 + args.eps_clip_high) * advantages
    clipped = torch.maximum(first, second)
    dual = getattr(args, "eps_clip_c", None)
    if dual is not None:
        if dual <= 1:
            raise ValueError("V21 dual clip must exceed one")
        clipped = torch.where(advantages < 0, torch.minimum(-dual * advantages, clipped), clipped)
    return clipped, (second > first).float()


def approximate_kl(
    log_probs: torch.Tensor, reference: torch.Tensor, kl_loss_type: str, importance_ratio: torch.Tensor | None = None
) -> torch.Tensor:
    log_ratio = log_probs.float() - reference.float()
    if kl_loss_type == "k1":
        result = log_ratio
    elif kl_loss_type == "k2":
        result = log_ratio.square() / 2
    elif kl_loss_type in ("k3", "low_var_kl"):
        log_ratio = -log_ratio
        if kl_loss_type == "low_var_kl":
            log_ratio = torch.nan_to_num(log_ratio, nan=0, posinf=20, neginf=-20).clamp(-20, 20)
        result = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError("Unknown V21 reference KL type")
    if importance_ratio is not None:
        result = result * importance_ratio
    return result.clamp(-10, 10) if kl_loss_type == "low_var_kl" else result


def _reducer(args: Any, batch: dict, masks: list[torch.Tensor]):
    from relax.backends.megatron.cp_utils import get_sum_of_sample_mean

    return get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        masks,
        False,
        args.qkv_format,
        batch.get("max_seq_lens"),
        batch.get("padded_total_lengths"),
        dynamic_cp_size=batch.get("dynamic_cp_size"),
        dynamic_cp_rank=batch.get("dynamic_cp_rank"),
        sample_denoms=batch["followup_token_mask_sums"],
    )


def auxiliary_loss(args: Any, batch: dict, context: PolicyLossContext) -> PolicyLossAddon:
    advantages = torch.cat(batch["followup_token_advantages"]).detach()
    source = torch.cat(
        [
            torch.full_like(a, float(flag))
            for a, flag in zip(
                batch["followup_token_advantages"], batch["followup_clarification_credit_flags"], strict=True
            )
        ]
    )
    pg, clipped = policy_tokens(args, context.ppo_kl, advantages)
    masks = batch["followup_token_masks"]
    if context.opsm_mask is not None:
        pg = pg * context.opsm_mask
    if context.tis_function is not None:
        pg, masks, _ = context.tis_function(**{**context.tis_kwargs, "pg_loss": pg, "loss_masks": masks})
    reduce = _reducer(args, batch, masks)
    whole, ctr, clarification = reduce(pg), reduce(pg * (1 - source)), reduce(pg * source)
    weight = option(args, "followup_token_credit_loss_weight")
    clarification_weight = option(args, "followup_clarification_credit_loss_weight", weight)
    terms = (
        (weight * whole,) if weight == clarification_weight else (weight * ctr, clarification_weight * clarification)
    )
    metrics = {
        "followup_auxiliary_loss": whole.detach(),
        "followup_user_signal_loss": ctr.detach(),
        "followup_clarification_quality_loss": clarification.detach(),
        "followup_user_signal_weighted_loss": (weight * ctr).detach(),
        "followup_clarification_quality_weighted_loss": (clarification_weight * clarification).detach(),
        "followup_user_signal_positive_advantage": reduce(advantages.clamp_min(0) * (1 - source)).detach(),
        "followup_user_signal_negative_advantage": reduce((-advantages).clamp_min(0) * (1 - source)).detach(),
        "followup_pg_clipfrac": reduce(clipped).detach(),
    }
    ref_terms = ()
    coefficient = option(args, "followup_reference_kl_loss_coef")
    clarification_coefficient = option(args, "followup_clarification_reference_kl_loss_coef", coefficient)
    if coefficient or clarification_coefficient:
        if not args.use_kl_loss or context.reference_log_probs is None:
            raise ValueError("V21 local KL requires reference scores")
        ratio = (context.log_probs - context.old_log_probs).exp() if args.use_unbiased_kl else None
        kl = approximate_kl(context.log_probs, torch.cat(context.reference_log_probs), args.kl_loss_type, ratio)
        kl = torch.where(context.active_tokens, torch.nan_to_num(kl, nan=0, posinf=0, neginf=0), 0)
        reduce_ref = _reducer(args, batch, batch["followup_token_masks"])
        all_ref, ctr_ref, clarify_ref = reduce_ref(kl), reduce_ref(kl * (1 - source)), reduce_ref(kl * source)
        weighted = (
            coefficient * all_ref
            if coefficient == clarification_coefficient
            else coefficient * ctr_ref + clarification_coefficient * clarify_ref
        )
        ref_terms = (weighted,)
        metrics.update(
            {
                "followup_reference_kl_loss": all_ref.detach(),
                "followup_user_signal_reference_kl_loss": ctr_ref.detach(),
                "followup_clarification_reference_kl_loss": clarify_ref.detach(),
                "followup_reference_kl_weighted_loss": weighted.detach(),
            }
        )
    return PolicyLossAddon(terms, ref_terms, metrics)


def loss(args: Any, batch: dict, logits: torch.Tensor, reducer):
    from relax.backends.megatron.loss import policy_loss_function

    bound_options = getattr(args, "agentspark_v21_loss_options", None)
    if not isinstance(bound_options, dict) or set(bound_options) != {*DEFAULTS, "followup_token_credit"}:
        raise ValueError("V21 requires generated source-backed training overrides")
    if any(getattr(args, name, None) != value for name, value in bound_options.items()):
        raise ValueError("V21 loss options changed after source binding")
    bound_main = getattr(args, "agentspark_v21_main_options", None)
    if not isinstance(bound_main, dict) or set(bound_main) != set(MAIN_DEFAULTS):
        raise ValueError("V21 requires generated main-objective overrides")
    if any(getattr(args, name, None) != value for name, value in bound_main.items()):
        raise ValueError("V21 main objective changed after source binding")
    return policy_loss_function(
        args,
        prepare_batch(args, batch),
        logits,
        reducer,
        extension=PolicyLossExtension(auxiliary_loss, policy_tokens, approximate_kl),
    )


loss.extra_training_fields = FIELDS

# Native outer loss and pipeline/DDP scaling preserve the logical Session mean.
loss.supports_static_cp_sample_mean = True
