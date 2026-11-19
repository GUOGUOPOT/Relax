# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real native HDO/TE/streams; never part of the CPU-only acceptance suite."""

from __future__ import annotations

import copy

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native HDO DMA, TE Adam and stream ordering require CUDA")
@pytest.mark.parametrize("pin_cpu_grads", [False, True], ids=["pageable", "pinned"])
@pytest.mark.parametrize("offload_fraction", [0.5, 0.9])
def test_bounded_native_hdo_async_gradients_updates_and_resume(monkeypatch, pin_cpu_grads, offload_fraction) -> None:
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
            for dtype in (torch.bfloat16, torch.float32) * 10
        ]
        return cls(
            [{"params": [parameter]} for parameter in params],
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=te.FusedAdam,
            offload_fraction=offload_fraction,
            param_update_in_fp32=True,
            overlap_cpu_optimizer_d2h_h2d=True,
            pin_cpu_grads=pin_cpu_grads,
            pin_cpu_params=pin_cpu_grads,
            fused=True,
            lr=0.003,
            betas=(0.8, 0.95),
            weight_decay=0.1,
        )

    reference = create()
    monkeypatch.setenv("RELAX_HDO_BOUNDED_CPU_GRADS", "1")
    assert patch_hdo_bounded_cpu_grads()
    actual = create()
    assert actual.cpu_optimizers and actual.gpu_optimizer is not None
    callback_calls = []

    def install_callback(optimizer):
        if not hasattr(optimizer, "_param_copy_back_gpu_callbacks"):
            return
        # Delay the last CPU copy-back to expose a missing H2D -> producer
        # fence when the caller immediately consumes the updated GPU weights.
        inner = optimizer.cpu_optimizers[-1].param_groups[0]["params"][0]
        original = optimizer.cpu_copys_map_gpu_param[inner]

        def copy_back(destination, source):
            callback_calls.append(source)
            torch.cuda._sleep(20_000_000)
            destination.copy_(source, non_blocking=True)

        optimizer._param_copy_back_gpu_callbacks[original] = copy_back

    install_callback(actual)
    producer = torch.cuda.Stream()
    for iteration in range(5):
        if iteration == 2:
            torch.cuda.synchronize()
            state = copy.deepcopy(actual.state_dict())
            old_children = tuple(actual.sub_optimizers)
            actual.load_state_dict(state)
            assert all(old is not new for old, new in zip(old_children, actual.sub_optimizers, strict=True))
        if iteration == 4:
            state = copy.deepcopy(actual.state_dict())
            model_state = [group["params"][0].clone() for group in actual.param_groups]
            actual = create()
            for group, saved in zip(actual.param_groups, model_state, strict=True):
                group["params"][0].copy_(saved)
            install_callback(actual)
            actual.load_state_dict(state)
        for optimizer in (actual, reference):
            for group in optimizer.param_groups:
                group["lr"] = 0.003 / (iteration + 1)
                group["weight_decay"] = 0.025 * (iteration + 1)
        # Initialization and checkpoint restore run on the default stream.
        producer.wait_stream(torch.cuda.current_stream())
        # The reference gets ready FP32 gradients. The adapted HDO receives
        # asynchronously produced BF16 gradients and must convert on this stream.
        with torch.cuda.stream(producer):
            torch.cuda._sleep(1_000_000)
            for index, (group, expected_group) in enumerate(
                zip(actual.param_groups, reference.param_groups, strict=True)
            ):
                original, expected = group["params"][0], expected_group["params"][0]
                missing = (iteration == 0 and index == 0) or (iteration == 3 and index in (0, 18))
                grad = torch.full_like(original, 0.125 * (index + iteration + 1), dtype=torch.bfloat16)
                original.decoupled_grad = None if missing else grad
                expected.decoupled_grad = None if missing else grad.float()
            actual.step()
            # Capture on the caller's stream BEFORE any global synchronize.
            observed_weights = [group["params"][0].clone() for group in actual.param_groups]
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
            torch.testing.assert_close(original, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                actual.param_to_inner_param[original], reference.param_to_inner_param[expected], rtol=0, atol=0
            )
            for name in ("step", "exp_avg", "exp_avg_sq"):
                if name in reference.state.get(expected, {}):
                    torch.testing.assert_close(
                        actual.state[original][name], reference.state[expected][name], rtol=0, atol=0
                    )
            assert actual.param_to_inner_param[original].grad is None
        for observed, group in zip(observed_weights, reference.param_groups, strict=True):
            torch.testing.assert_close(observed, group["params"][0], rtol=0, atol=0)
        if hasattr(actual, "_param_copy_back_gpu_callbacks"):
            assert len(callback_calls) == iteration + 1
        for group, expected_group in zip(
            actual.gpu_optimizer.param_groups, reference.gpu_optimizer.param_groups, strict=True
        ):
            assert group["step"] == expected_group["step"] == iteration + 1
        assert not actual.cpu_copy_map_grad
        assert actual._relax_cpu_grad_workspace.numel() == 2**16
        assert actual._relax_cpu_grad_workspace.is_pinned() is pin_cpu_grads
