# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in, bounded gradient staging for Megatron's overlapping CPU AdamW.

Keep native masters, moments, checkpoint mappings and parameter-copy hooks. Only
replace HDO's eager allocation of one permanent CPU gradient per parameter with
one FP32 workspace per HDO. CPU updates are serialized, trading transfer overlap
for host memory. CUDA validation is required before production acceptance.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
import textwrap
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

from relax.utils.logging_utils import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)
_ENV = "RELAX_HDO_BOUNDED_CPU_GRADS"
_MARKER = "_relax_bounded_cpu_grads"
# Fingerprints exclude formatting/comments/docstrings. Fail closed on a changed
# native algorithm: the copy hooks and child ownership are part of this contract.
_NATIVE_CONTRACT = {
    "step": "27b34f1f53c9b2fd74a61f043efd2f46c5040376dcb1fed0337553c25a3849ee",
    "_set_sub_optimizer_grads": "dbf65bcbcc14002d2696b020bd36547412553d4c66852ec2afa0889c4b7caf3d",
    "_register_param_copy_back_gpu_hook": "877649e152728d6788493f4b187aa0dd07bf3a07d47e7118167bcb42124c99e0",
    "_init_sub_optimizers": "ee832379cba3519c26f8ec189e4fedad65a48b95d1fff175acb2fb2a7bc42d54",
    "build_cpu_optimizer_list": "68eea6da0e856fc9698663ce016def14c4ee447c7d8917d6c61b0d9a7cf98c3f",
    "_sync_hdo_param_groups_to_sub_optimizers": "27cf234b2556a345cab76523a6931ee1f51626117d05482a1230634712cc8226",
}


def _source_fingerprint(method: Callable[..., Any]) -> str:
    tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(method))))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                body.pop(0)
    return hashlib.sha256(ast.dump(tree.body[0], include_attributes=False).encode()).hexdigest()


def _validate_native_contract(optimizer_cls: type) -> None:
    for name, expected in _NATIVE_CONTRACT.items():
        try:
            actual = _source_fingerprint(getattr(optimizer_cls, name))
        except (AttributeError, OSError, TypeError, SyntaxError) as error:
            raise RuntimeError(f"{_ENV}: cannot inspect native HDO.{name}") from error
        if actual != expected:
            raise RuntimeError(f"{_ENV}: unsupported native HDO.{name} contract ({actual})")


def _gradient(parameter: torch.Tensor) -> torch.Tensor | None:
    # A present-but-None decoupled_grad intentionally overrides .grad, as in HDO.
    return getattr(parameter, "decoupled_grad", parameter.grad)


def _validate_children(hdo: Any) -> tuple[list[tuple[Any, Any, Any]], list[tuple[Any, Any]]]:
    import torch

    if hdo.param_update_in_fp32 is not True or hdo.overlap_cpu_optimizer_d2h_h2d is not True:
        raise RuntimeError(f"{_ENV}: requires FP32 masters and overlapping one-parameter CPU optimizers")
    if hdo.cpu_copy_map_grad or hdo._cpu_optimizer_map_data_event:
        raise RuntimeError(f"{_ENV}: unexpected native persistent gradients or pending transfer events")
    if not isinstance(hdo.pin_cpu_grads, bool):
        raise RuntimeError(f"{_ENV}: pin_cpu_grads must be boolean")
    cpu_entries = []
    gpu_pairs = []
    seen = set()
    for child in hdo.cpu_optimizers:
        if type(child) is not torch.optim.AdamW or len(child.param_groups) != 1:
            raise RuntimeError(f"{_ENV}: requires native torch AdamW with one CPU parameter")
        group = child.param_groups[0]
        if len(group["params"]) != 1 or group.get("fused") is not True:
            raise RuntimeError(f"{_ENV}: requires one-parameter fused CPU AdamW")
        if any(group.get(key, False) for key in ("capturable", "differentiable", "amsgrad", "maximize")):
            raise RuntimeError(f"{_ENV}: unsupported CPU AdamW mode")
        inner = group["params"][0]
        original = hdo.cpu_copys_map_gpu_param.get(inner)
        if original is None or hdo.gpu_params_map_cpu_copy.get(original) is not inner:
            raise RuntimeError(f"{_ENV}: invalid CPU master ownership")
        if inner.device.type != "cpu" or inner.dtype != torch.float32 or not inner.is_contiguous():
            raise RuntimeError(f"{_ENV}: requires contiguous FP32 CPU masters")
        cpu_entries.append((child, original, inner))
        seen.add(inner)
    child = hdo.gpu_optimizer
    if child is not None:
        if (type(child).__module__, type(child).__name__) != (
            "transformer_engine.pytorch.optimizers.fused_adam",
            "FusedAdam",
        ):
            raise RuntimeError(f"{_ENV}: GPU child must be Transformer Engine FusedAdam")
        if any(getattr(child, key, False) for key in ("use_decoupled_grad", "capturable", "master_weights")):
            raise RuntimeError(f"{_ENV}: GPU child requires ordinary .grad and HDO-owned FP32 masters")
        if child.exp_avg_dtype != torch.float32 or child.exp_avg_sq_dtype != torch.float32:
            raise RuntimeError(f"{_ENV}: GPU Adam moments must be FP32")
        for group in child.param_groups:
            for inner in group["params"]:
                original = hdo.inner_param_to_orig_param.get(inner)
                if original is None or inner.device.type != "cuda" or inner.dtype != torch.float32:
                    raise RuntimeError(f"{_ENV}: invalid GPU master ownership or dtype")
                gpu_pairs.append((original, inner))
                if inner in seen:
                    raise RuntimeError(f"{_ENV}: duplicate inner parameter ownership")
                seen.add(inner)
    all_pairs = [(original, inner) for _, original, inner in cpu_entries] + gpu_pairs
    if len(all_pairs) != len(seen) or seen != set(hdo.param_to_inner_param.values()):
        raise RuntimeError(f"{_ENV}: incomplete or duplicate inner parameter ownership")
    for original, inner in all_pairs:
        if hdo.param_to_inner_param.get(original) is not inner or original.device.type != "cuda":
            raise RuntimeError(f"{_ENV}: unsupported outer-to-inner parameter mapping")
        grad = _gradient(original)
        if grad is not None and (
            grad.layout != torch.strided
            or grad.shape != inner.shape
            or grad.device != original.device
            or grad.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        ):
            raise RuntimeError(f"{_ENV}: unsupported gradient layout, shape, device or dtype")
    return cpu_entries, gpu_pairs


def _workspace(hdo: Any, cpu_entries: list[tuple[Any, Any, Any]]) -> torch.Tensor:
    import torch

    numel = max((inner.numel() for _, _, inner in cpu_entries), default=0)
    workspace = getattr(hdo, "_relax_cpu_grad_workspace", None)
    if workspace is None or workspace.numel() != numel or workspace.is_pinned() != hdo.pin_cpu_grads:
        workspace = torch.empty(numel, dtype=torch.float32, device="cpu", pin_memory=hdo.pin_cpu_grads)
        hdo._relax_cpu_grad_workspace = workspace
    return workspace


def _release_gpu_grad(original: torch.Tensor, inner: torch.Tensor) -> None:
    # Native FP32 parameters can be their own inner parameter. Preserve their
    # original .grad (no extra cast allocation); decoupled gradients stay outer.
    if original is not inner or hasattr(original, "decoupled_grad"):
        inner.grad = None


def _step_with_workspace(
    hdo: Any,
    cpu_entries: list[tuple[Any, Any, Any]],
    gpu_pairs: list[tuple[Any, Any]],
    workspace: torch.Tensor,
    copy_gradient: Callable[[torch.Tensor, torch.Tensor], None],
) -> None:
    """Run native child updates; copy_gradient must complete before returning.

    This CPU-testable core neither changes Adam's state nor caches parameter
    objects. Native load hooks may rebuild every child between calls.
    """
    import torch

    with torch.no_grad():
        try:
            # Casts and native GPU step run on the producer's current stream.
            for original, inner in gpu_pairs:
                grad = _gradient(original)
                inner.grad = None if grad is None else grad.to(dtype=inner.dtype)
            try:
                if hdo.gpu_optimizer is not None:
                    hdo.gpu_optimizer.step()
            finally:
                for original, inner in gpu_pairs:
                    _release_gpu_grad(original, inner)
            for child, original, inner in cpu_entries:
                inner.grad = None
                grad = _gradient(original)
                if grad is None:
                    # Setting requires_grad=False alone does not stop Adam from
                    # using a stale .grad or applying weight decay.
                    continue
                try:
                    view = workspace[: inner.numel()].view_as(inner)
                    copy_gradient(view, grad)
                    inner.grad = view
                    child.step()  # Preserve native H2D parameter-copy posthook.
                finally:
                    inner.grad = None
            hdo._sync_sub_optimizers_state_to_hdo()
        finally:
            for _, _, inner in cpu_entries:
                inner.grad = None
            for original, inner in gpu_pairs:
                _release_gpu_grad(original, inner)


def _bounded_step(hdo: Any, closure: Any = None) -> None:
    import torch

    if closure is not None:
        raise RuntimeError(f"{_ENV}: closures are unsupported")
    hdo._sync_hdo_param_groups_to_sub_optimizers()
    cpu_entries, gpu_pairs = _validate_children(hdo)
    workspace = _workspace(hdo, cpu_entries)
    if not getattr(hdo, "_relax_bounded_memory_logged", False):
        cpu_numel = sum(inner.numel() for _, _, inner in cpu_entries)
        gpu_numel = sum(inner.numel() for _, inner in gpu_pairs)
        logger.info(
            "[HDO bounded gradients] cpu_inner_numel=%d gpu_inner_numel=%d workspace_bytes=%d "
            "cpu_master_moments_bytes=%d gpu_master_moments_bytes=%d cpu_grad_bytes_avoided=%d",
            cpu_numel,
            gpu_numel,
            workspace.numel() * workspace.element_size(),
            cpu_numel * 12,
            gpu_numel * 12,
            cpu_numel * 4 - workspace.numel() * 4,
        )
        hdo._relax_bounded_memory_logged = True

    def copy_gradient(destination: torch.Tensor, source: torch.Tensor) -> None:
        hdo._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(hdo._d2h_stream):
            destination.copy_(source, non_blocking=True)
            event = hdo._d2h_stream.record_event()
        # CPU Adam reads the host buffer. Keep both source and workspace alive
        # until DMA completes, including asynchronous transfer exceptions.
        event.synchronize()

    _step_with_workspace(hdo, cpu_entries, gpu_pairs, workspace, copy_gradient)


def patch_hdo_bounded_cpu_grads() -> bool:
    """Install only under explicit opt-in, before HDO construction/loading."""
    enabled = os.environ.get(_ENV, "0")
    if enabled not in {"0", "1"}:
        raise ValueError(f"{_ENV} must be 0 or 1")
    if enabled == "0":
        return False
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    original = HybridDeviceOptimizer.step
    if getattr(original, _MARKER, False):
        return False
    _validate_native_contract(HybridDeviceOptimizer)
    original_init = HybridDeviceOptimizer._init_sub_optimizers

    @wraps(original_init)
    def init_children(self: Any) -> None:
        original_init(self)
        # Validate at construction and after checkpoint-driven rebuilds, before
        # paying for a rollout. Step also checks the current gradients/groups.
        _validate_children(self)

    @wraps(original)
    def step(self: Any, closure: Any = None) -> None:
        _bounded_step(self, closure)

    setattr(step, _MARKER, True)
    HybridDeviceOptimizer._init_sub_optimizers = init_children
    HybridDeviceOptimizer.step = step
    logger.info("Enabled bounded HDO CPU gradients; native Adam state and checkpoint mappings are preserved")
    return True
