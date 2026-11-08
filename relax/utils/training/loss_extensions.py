# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Explicit custom-loss columns and optional native policy composition.

Columns remain per-sample values through transfer and packing. An extension
owns response alignment; the generic path never guesses which vectors to CP-slice.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from relax.utils.misc import load_function


def custom_loss_fields(args: Any) -> tuple[str, ...]:
    if getattr(args, "loss_type", None) != "custom_loss":
        return ()
    path = getattr(args, "custom_loss_function_path", None)
    if not path:
        raise ValueError("custom_loss requires custom_loss_function_path")
    fields = getattr(load_function(path), "extra_training_fields", ())
    if not isinstance(fields, tuple) or any(not isinstance(name, str) or not name.isidentifier() for name in fields):
        raise ValueError("custom loss extra_training_fields must be a tuple of field names")
    if len(set(fields)) != len(fields):
        raise ValueError("custom loss extra_training_fields contains duplicates")
    return fields


def custom_loss_supports_static_cp_sample_mean(args: Any) -> bool:
    """Opt into the verified synchronous, colocated static-CP sample-mean path.

    The loss must return CP-local contributions with whole logical-sample
    denominators; native loss/schedule/DDP scaling supplies the global mean.
    """
    cp_size = getattr(args, "context_parallel_size", 1)
    if (
        getattr(args, "loss_type", None) != "custom_loss"
        or getattr(args, "calculate_per_token_loss", False) is not False
        or type(cp_size) is not int
        or cp_size <= 1
        or getattr(args, "dynamic_context_parallel", False)
        or getattr(args, "colocate", False) is not True
        or getattr(args, "fully_async", False)
    ):
        return False
    path = getattr(args, "custom_loss_function_path", None)
    return bool(path) and getattr(load_function(path), "supports_static_cp_sample_mean", False) is True


@dataclass(frozen=True)
class PolicyLossContext:
    ppo_kl: torch.Tensor
    log_probs: torch.Tensor
    old_log_probs: torch.Tensor
    reference_log_probs: list[torch.Tensor] | None
    active_tokens: torch.Tensor
    opsm_mask: torch.Tensor | None
    tis_function: Callable | None
    tis_kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class PolicyLossAddon:
    policy_terms: tuple[torch.Tensor, ...] = ()
    reference_terms: tuple[torch.Tensor, ...] = ()
    metrics: Mapping[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyLossExtension:
    """Opt-in score arithmetic and auxiliary terms; absent means native behavior."""

    auxiliary_loss: Callable
    policy_loss: Callable
    approximate_kl: Callable


def validate_addon(addon: PolicyLossAddon) -> None:
    if not isinstance(addon, PolicyLossAddon):
        raise TypeError("Auxiliary policy loss must return PolicyLossAddon")
    if any(
        not isinstance(v, torch.Tensor) or v.ndim != 0
        for v in (*addon.policy_terms, *addon.reference_terms, *addon.metrics.values())
    ):
        raise ValueError("Auxiliary loss terms and metrics must be scalar tensors")
