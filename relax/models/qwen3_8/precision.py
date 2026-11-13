# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MoE FP8 compute with BF16 parameters for Qwen3.8 Flash Next.

The GDN input projection contains small gate sections which make its TP shards
incompatible with block-scaled FP8. Keep attention, vision, PLE and MTP in
their original precision while quantizing the backbone's aligned expert GEMMs.
The global recipe stays enabled for expert padding and activation
recomputation.
"""

from argparse import Namespace

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def configure_fp8(model: torch.nn.Module, args: Namespace) -> dict[str, str]:
    """Configure existing TE layers without replacing parameters or changing
    keys."""
    if not getattr(args, "fp8", None):
        return {}
    if not getattr(args, "bf16", False) or getattr(args, "fp8_param_gather", False):
        raise ValueError("Qwen3.8 FP8 compute requires BF16 primary parameters without --fp8-param-gather")
    if getattr(args, "use_transformer_engine_op_fuser", False):
        raise ValueError("Qwen3.8 module precision overrides require the TE Linear/GroupedLinear forward path")
    if getattr(args, "te_precision_config_file", None):
        raise ValueError("Qwen3.8 MoE FP8 cannot be combined with a separate TE precision config")

    from megatron.core.quantization.quant_config import MatchContext, QuantizationConfig
    from transformer_engine import pytorch as te

    recipe = getattr(args.fp8_recipe, "value", args.fp8_recipe)
    alignment = {"mxfp8": 32, "blockwise": 128, "tensorwise": 16}.get(recipe)
    if alignment is None:
        raise ValueError(f"Qwen3.8 MoE FP8 does not support recipe {recipe!r}")

    plan: dict[str, str] = {}
    pending = []
    for name, module in model.named_modules():
        if not isinstance(module, (te.Linear, te.LayerNormLinear, te.GroupedLinear)):
            continue
        if not callable(getattr(module, "finish_init", None)):
            raise RuntimeError(f"Qwen3.8 FP8 requires MCore per-module precision support for {name}")
        parts = name.split(".")
        expert = (
            bool({"experts", "shared_experts"}.intersection(parts))
            and parts[-1] in ("linear_fc1", "linear_fc2")
            and "mtp" not in parts
            and "vision_model" not in parts
        )
        if expert:
            for param_name, param in module.named_parameters(recurse=False):
                if param_name.startswith("weight") and param.ndim >= 2:
                    if any(size % alignment for size in param.shape[-2:]):
                        raise ValueError(
                            f"{name}.{param_name} shape {tuple(param.shape)} is not aligned to {alignment} for {recipe}"
                        )
            quant_config = None  # Inherit the global FP8 context and its recompute state.
        else:
            quant_config = QuantizationConfig(
                {
                    "transformer_engine_config_type": "TEQuantizationParams",
                    "training_recipe": {},
                },
                match_input=MatchContext(module_path=name, layer_number=None),
                config_key="qwen38_bf16",
            )
        plan[name] = recipe if expert else "bf16"
        pending.append((module, quant_config))

    # Validate the whole stage before applying any overrides.
    for module, quant_config in pending:
        module.finish_init(quant_config)
    logger.info(
        "Qwen3.8 precision: %d backbone expert GEMMs use %s; %d TE modules retain BF16; primary parameters remain BF16",
        sum(value == recipe for value in plan.values()),
        recipe,
        sum(value == "bf16" for value in plan.values()),
    )
    logger.debug("Qwen3.8 module precision: %s", plan)
    return plan
