# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU numerical/state tests; CUDA DMA and TE Adam are tested separately."""

from __future__ import annotations

import copy
from collections import defaultdict
from types import MethodType
from typing import Any

import pytest
import torch

from relax.backends.megatron import bounded_cpu_optimizer as module

hybrid_module = pytest.importorskip("megatron.core.optimizer.cpu_offloading.hybrid_optimizer")
HybridDeviceOptimizer = hybrid_module.HybridDeviceOptimizer


def _rebuild_cpu_children(hdo: Any) -> None:
    """Rebuild real AdamW children without invoking any CUDA APIs."""
    hdo.param_to_inner_param = {}
    cpu_groups, gpu_groups = [], []
    for index, group in enumerate(hdo.param_groups):
        original = group["params"][0]
        inner = (
            original if index in hdo.test_gpu_indices and original.dtype == torch.float32 else original.float().clone()
        )
        hdo.param_to_inner_param[original] = inner
        destination = gpu_groups if index in hdo.test_gpu_indices else cpu_groups
        destination.append({**group, "params": [inner]})
    hdo.inner_param_to_orig_param = {inner: original for original, inner in hdo.param_to_inner_param.items()}
    hdo.param_to_fp32_param = {
        original: inner for original, inner in hdo.param_to_inner_param.items() if original.dtype != torch.float32
    }
    hdo.fp32_param_to_orig_param = {inner: original for original, inner in hdo.param_to_fp32_param.items()}
    hdo.gpu_params_map_cpu_copy = {
        group["params"][0]: hdo.param_to_inner_param[group["params"][0]]
        for index, group in enumerate(hdo.param_groups)
        if index not in hdo.test_gpu_indices
    }
    hdo.cpu_copys_map_gpu_param = {inner: original for original, inner in hdo.gpu_params_map_cpu_copy.items()}
    hdo.cpu_optimizers = hdo.build_cpu_optimizer_list(torch.optim.AdamW, cpu_groups)
    hdo.gpu_optimizer = torch.optim.AdamW(gpu_groups) if gpu_groups else None
    hdo.cpu_copy_map_grad = defaultdict(torch.Tensor)
    hdo._cpu_optimizer_map_data_event = {}

    def copy_back(child, args, kwargs):
        with torch.no_grad():
            for group in child.param_groups:
                inner = group["params"][0]
                original = hdo.inner_param_to_orig_param[inner]
                original.copy_(inner)

    for child in hdo.sub_optimizers:
        child.register_step_post_hook(copy_back)


def _cpu_hdo(*, gpu_indices: tuple[int, ...] = ()) -> Any:
    originals = [
        torch.tensor([0.375, -0.625, 0.125], dtype=torch.bfloat16),
        torch.tensor([1.001, -2.003, 0.7, -0.5, 0.125], dtype=torch.float32),
        torch.tensor([0.25, 0.375], dtype=torch.bfloat16),
    ]
    groups = [
        {
            "params": [parameter],
            "lr": 0.01 * (index + 1),
            "betas": (0.8, 0.95),
            "eps": 1e-7,
            "weight_decay": 0.1,
            "fused": True,
        }
        for index, parameter in enumerate(originals)
    ]
    hdo = object.__new__(HybridDeviceOptimizer)
    torch.optim.Optimizer.__init__(hdo, groups, {"cpu_optimizer_cls": torch.optim.AdamW})
    hdo.param_update_in_fp32 = True
    hdo.overlap_cpu_optimizer_d2h_h2d = True
    hdo.pin_cpu_grads = False
    hdo.test_gpu_indices = gpu_indices
    hdo._init_sub_optimizers = MethodType(_rebuild_cpu_children, hdo)
    hdo._init_sub_optimizers()
    hdo._register_load_state_dict_hooks()
    return hdo


def _entries(hdo: Any) -> tuple[list, list]:
    cpu = [
        (child, hdo.cpu_copys_map_gpu_param[child.param_groups[0]["params"][0]], child.param_groups[0]["params"][0])
        for child in hdo.cpu_optimizers
    ]
    gpu = [
        (hdo.inner_param_to_orig_param[inner], inner)
        for group in (hdo.gpu_optimizer.param_groups if hdo.gpu_optimizer else [])
        for inner in group["params"]
    ]
    return cpu, gpu


def _step(hdo: Any) -> None:
    hdo._sync_hdo_param_groups_to_sub_optimizers()
    cpu, gpu = _entries(hdo)
    workspace = module._workspace(hdo, cpu)
    module._step_with_workspace(hdo, cpu, gpu, workspace, lambda destination, source: destination.copy_(source))


def _set_grads(hdo: Any, iteration: int) -> None:
    for index, group in enumerate(hdo.param_groups):
        parameter = group["params"][0]
        # Both first-use None and non-None -> None -> non-None must skip decay
        # and counter advancement. A stale .grad must not override this None.
        missing = (iteration == 0 and index == 0) or (iteration == 2 and index == 1)
        parameter.grad = torch.ones_like(parameter) * 7
        parameter.decoupled_grad = (
            None
            if missing
            else torch.linspace(-0.125, 0.25 * (iteration + 1), parameter.numel(), dtype=torch.float32).reshape_as(
                parameter
            )
        )
        if iteration == 3 and parameter.decoupled_grad is not None:
            parameter.decoupled_grad = parameter.decoupled_grad.bfloat16()
        group["lr"] = 0.01 * (index + 1) / (iteration + 1)
        group["weight_decay"] = 0.03 * (iteration + 1)


def _assert_state_matches(hdo: Any, oracle: torch.optim.AdamW) -> None:
    for group, expected_group in zip(hdo.param_groups, oracle.param_groups, strict=True):
        original, expected = group["params"][0], expected_group["params"][0]
        inner = hdo.param_to_inner_param[original]
        torch.testing.assert_close(inner, expected, rtol=0, atol=0)
        actual_state, expected_state = hdo.state.get(original, {}), oracle.state.get(expected, {})
        for name in ("step", "exp_avg", "exp_avg_sq"):
            if name in expected_state:
                torch.testing.assert_close(actual_state[name], expected_state[name], rtol=0, atol=0)
            else:
                assert name not in actual_state
        if expected_state:
            assert actual_state["master_param"] is inner


@pytest.mark.parametrize("gpu_indices", [(), (1, 2)], ids=["cpu_workspace", "inner_gradient_routing"])
def test_bounded_updates_match_real_adamw_with_none_decay_and_lr_changes(gpu_indices) -> None:
    hdo = _cpu_hdo(gpu_indices=gpu_indices)
    oracle = torch.optim.AdamW(
        [{**group, "params": [group["params"][0].float().clone()]} for group in hdo.param_groups]
    )
    borrowed_pointers = []
    for child in hdo.cpu_optimizers:
        child.register_step_pre_hook(
            lambda optimizer, args, kwargs: borrowed_pointers.append(
                optimizer.param_groups[0]["params"][0].grad.untyped_storage().data_ptr()
            )
        )
    for iteration in range(4):
        _set_grads(hdo, iteration)
        for group, expected_group in zip(hdo.param_groups, oracle.param_groups, strict=True):
            expected_group["lr"], expected_group["weight_decay"] = group["lr"], group["weight_decay"]
            gradient = group["params"][0].decoupled_grad
            expected_group["params"][0].grad = None if gradient is None else gradient.float().clone()
        # Simulate leftover child gradients from an earlier path, including None.
        for child in hdo.cpu_optimizers:
            inner = child.param_groups[0]["params"][0]
            inner.grad = torch.ones_like(inner) * 17
        _step(hdo)
        oracle.step()
        _assert_state_matches(hdo, oracle)
        assert all(inner.grad is None for inner in hdo.param_to_inner_param.values())
        assert not hdo.cpu_copy_map_grad
    assert len(set(borrowed_pointers)) == 1
    assert hdo._relax_cpu_grad_workspace.numel() == max(inner.numel() for _, _, inner in _entries(hdo)[0])


def test_bounded_real_hdo_checkpoint_rebuild_then_continue() -> None:
    reference = _cpu_hdo()
    for iteration in range(2):
        _set_grads(reference, iteration)
        _step(reference)
    checkpoint = copy.deepcopy(reference.state_dict())
    restored = _cpu_hdo()
    old_children = tuple(restored.cpu_optimizers)
    restored.load_state_dict(checkpoint)  # Real native pre/post hooks and state mapping.
    assert all(old is not new for old, new in zip(old_children, restored.cpu_optimizers, strict=True))
    for iteration in range(2, 5):
        _set_grads(reference, iteration)
        _set_grads(restored, iteration)
        _step(reference)
        _step(restored)
    for original, restored_original in zip(reference.param_to_inner_param, restored.param_to_inner_param, strict=True):
        torch.testing.assert_close(
            reference.param_to_inner_param[original], restored.param_to_inner_param[restored_original], rtol=0, atol=0
        )
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                reference.state[original][name], restored.state[restored_original][name], rtol=0, atol=0
            )
    assert not restored.cpu_copy_map_grad


def test_bounded_child_failure_releases_all_borrowed_gradients() -> None:
    hdo = _cpu_hdo(gpu_indices=(1, 2))
    _set_grads(hdo, 1)

    def fail_after_update(child, args, kwargs):
        raise RuntimeError("deliberate CPU post-hook failure")

    hdo.cpu_optimizers[0].register_step_post_hook(fail_after_update)
    with pytest.raises(RuntimeError, match="deliberate CPU post-hook"):
        _step(hdo)
    assert all(inner.grad is None for inner in hdo.param_to_inner_param.values())
    assert not hdo.cpu_copy_map_grad


def test_bounded_transfer_failure_releases_workspace_view() -> None:
    hdo = _cpu_hdo()
    _set_grads(hdo, 1)
    cpu, gpu = _entries(hdo)

    def fail_copy(destination, source):
        destination.copy_(source)
        raise RuntimeError("deliberate transfer failure")

    with pytest.raises(RuntimeError, match="deliberate transfer"):
        module._step_with_workspace(hdo, cpu, gpu, module._workspace(hdo, cpu), fail_copy)
    assert all(inner.grad is None for inner in hdo.param_to_inner_param.values())
    assert not hdo.state


def test_bounded_native_fp32_without_decoupled_gradient_preserves_original_grad() -> None:
    hdo = _cpu_hdo(gpu_indices=(1,))
    for original in hdo.param_to_inner_param:
        original.grad = torch.ones_like(original)
    native = hdo.param_groups[1]["params"][0]
    gradient = native.grad
    _step(hdo)
    assert native.grad is gradient
    assert hdo.param_to_inner_param[native] is native


def test_bounded_opt_in_contract_is_idempotent_and_preserves_group_step_patch(monkeypatch) -> None:
    from relax.backends.megatron.compat import _patch_hdo_group_steps

    for name in ("step", "_init_sub_optimizers", "_sync_sub_optimizers_state_to_hdo"):
        monkeypatch.setattr(HybridDeviceOptimizer, name, getattr(HybridDeviceOptimizer, name))
    monkeypatch.delenv(module._ENV, raising=False)
    before = HybridDeviceOptimizer.step
    assert module.patch_hdo_bounded_cpu_grads() is False
    assert HybridDeviceOptimizer.step is before
    _patch_hdo_group_steps(HybridDeviceOptimizer)
    monkeypatch.setenv(module._ENV, "1")
    assert module.patch_hdo_bounded_cpu_grads() is True
    assert module.patch_hdo_bounded_cpu_grads() is False
    module._validate_native_contract(HybridDeviceOptimizer)
    assert getattr(HybridDeviceOptimizer._sync_sub_optimizers_state_to_hdo, "_relax_group_steps")


def test_bounded_unknown_native_contract_and_unsupported_modes_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv(module._ENV, "invalid")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        module.patch_hdo_bounded_cpu_grads()
    monkeypatch.setattr(HybridDeviceOptimizer, "step", _step)
    monkeypatch.setenv(module._ENV, "1")
    with pytest.raises(RuntimeError, match="native HDO.step"):
        module.patch_hdo_bounded_cpu_grads()
    hdo = _cpu_hdo()
    with pytest.raises(RuntimeError, match="closures"):
        module._bounded_step(hdo, lambda: None)
    hdo.overlap_cpu_optimizer_d2h_h2d = False
    with pytest.raises(RuntimeError, match="overlapping"):
        module._validate_children(hdo)
    hdo.overlap_cpu_optimizer_d2h_h2d = True
    hdo.cpu_optimizers[0].param_groups[0]["fused"] = False
    with pytest.raises(RuntimeError, match="fused CPU AdamW"):
        module._validate_children(hdo)
