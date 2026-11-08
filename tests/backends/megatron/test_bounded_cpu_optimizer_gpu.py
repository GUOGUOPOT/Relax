# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real native HDO/TE/streams; never part of the CPU-only acceptance suite."""

from __future__ import annotations

import copy

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native HDO DMA, TE Adam and stream ordering require CUDA")
def test_bounded_native_hdo_async_gradients_updates_and_resume(monkeypatch) -> None:
    te = pytest.importorskip("transformer_engine.pytorch.optimizers")
    hybrid = pytest.importorskip("megatron.core.optimizer.cpu_offloading.hybrid_optimizer")
    from relax.backends.megatron.bounded_cpu_optimizer import patch_hdo_bounded_cpu_grads
    from relax.backends.megatron.compat import _patch_hdo_group_steps

    cls = hybrid.HybridDeviceOptimizer
    for name in ("step", "_init_sub_optimizers", "_sync_sub_optimizers_state_to_hdo"):
        monkeypatch.setattr(cls, name, getattr(cls, name))
    _patch_hdo_group_steps(cls)
    native_step = cls.step

    def create():
        params = [
            torch.linspace(-0.5, 0.5, 2**16, device="cuda", dtype=dtype)
            for dtype in (torch.bfloat16, torch.float32, torch.bfloat16, torch.float32)
        ]
        return cls(
            [{"params": [parameter]} for parameter in params],
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=te.FusedAdam,
            offload_fraction=0.5,
            param_update_in_fp32=True,
            overlap_cpu_optimizer_d2h_h2d=True,
            pin_cpu_grads=True,
            pin_cpu_params=True,
            fused=True,
            lr=0.003,
            betas=(0.8, 0.95),
            weight_decay=0.1,
        )

    reference = create()
    monkeypatch.setenv("RELAX_HDO_BOUNDED_CPU_GRADS", "1")
    assert patch_hdo_bounded_cpu_grads()
    actual = create()
    producer = torch.cuda.Stream()
    for iteration in range(5):
        if iteration == 2:
            torch.cuda.synchronize()
            state = copy.deepcopy(actual.state_dict())
            old_children = tuple(actual.sub_optimizers)
            actual.load_state_dict(state)
            assert all(old is not new for old, new in zip(old_children, actual.sub_optimizers, strict=True))
        for optimizer in (actual, reference):
            for group in optimizer.param_groups:
                group["lr"] = 0.003 / (iteration + 1)
                group["weight_decay"] = 0.025 * (iteration + 1)
        # The reference gets ready FP32 gradients. The adapted HDO receives
        # asynchronously produced BF16 gradients and must convert on this stream.
        with torch.cuda.stream(producer):
            torch.cuda._sleep(1_000_000)
            for index, (group, expected_group) in enumerate(
                zip(actual.param_groups, reference.param_groups, strict=True)
            ):
                original, expected = group["params"][0], expected_group["params"][0]
                missing = (iteration == 0 and index == 0) or (iteration == 3 and index in (0, 2))
                grad = torch.full_like(original, 0.125 * (index + iteration + 1), dtype=torch.bfloat16)
                original.decoupled_grad = None if missing else grad
                expected.decoupled_grad = None if missing else grad.float()
            actual.step()
            # Allocate/rewrite storage after inner.grad release, before a global
            # synchronize, to expose allocator or stream lifetime mistakes.
            churn = [torch.empty(2**16, device="cuda").fill_(11) for _ in range(4)]
            del churn
        torch.cuda.synchronize()
        # Native HDO retains old inner.grad for None inputs. Explicitly clear
        # these in the control so it represents Adam's documented skip semantics.
        for inner in reference.param_to_inner_param.values():
            inner.grad = None
        # Native HDO binds CPU .grad only when allocating its permanent cache.
        # Recreate this tiny control cache so cleared gradients get rebound.
        reference.cpu_copy_map_grad.clear()
        for original, inner in reference.param_to_inner_param.items():
            if original is inner:
                # Native HDO's FP32 shadow loop excludes already-FP32 params;
                # provide their direct .grad as Megatron's outer wrapper does.
                inner.grad = original.decoupled_grad
        native_step(reference)
        torch.cuda.synchronize()
        for original, expected in zip(actual.param_to_inner_param, reference.param_to_inner_param, strict=True):
            torch.testing.assert_close(
                actual.param_to_inner_param[original], reference.param_to_inner_param[expected], rtol=0, atol=0
            )
            for name in ("step", "exp_avg", "exp_avg_sq"):
                if name in reference.state.get(expected, {}):
                    torch.testing.assert_close(
                        actual.state[original][name], reference.state[expected][name], rtol=0, atol=0
                    )
            assert actual.param_to_inner_param[original].grad is None
        for group, expected_group in zip(
            actual.gpu_optimizer.param_groups, reference.gpu_optimizer.param_groups, strict=True
        ):
            assert group["step"] == expected_group["step"] == iteration + 1
        assert not actual.cpu_copy_map_grad
        assert actual._relax_cpu_grad_workspace.numel() == 2**16
        assert actual._relax_cpu_grad_workspace.is_pinned()
