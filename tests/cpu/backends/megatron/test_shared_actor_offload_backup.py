# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise real flat storage and snapshots without importing the GPU backend.

These regressions cover aliasing and weight recovery, not CUDA DMA or
collectives.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from relax.utils.training import tensor_backper


_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def offload_module(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "shared_actor_offload_test", _ROOT / "relax/backends/megatron/weight_update/train_offload.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)
    for target in (module, tensor_backper):
        monkeypatch.setattr(target, "_PIN_MEMORY", False)
        monkeypatch.setattr(target, "_NON_BLOCKING", False)
        monkeypatch.setattr(target, "device_module", device)
    monkeypatch.setattr(module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    return module


def _model():
    flat = torch.arange(14, dtype=torch.bfloat16)
    first = torch.nn.Parameter(flat[2:6].view(2, 2))
    second = torch.nn.Parameter(flat[8:11])
    buffer = SimpleNamespace(
        param_data=flat,
        grad_data=torch.ones(14, dtype=torch.float32),
        param_index_map={first: (2, 6, 0), second: (8, 11, 0)},
    )
    expert_flat = torch.arange(6, dtype=torch.bfloat16)
    expert = torch.nn.Parameter(expert_flat[:4])
    expert_buffer = SimpleNamespace(
        param_data=expert_flat,
        grad_data=torch.ones(6, dtype=torch.float32),
        param_to_index={expert: (0, 4, 0)},
    )
    model = [SimpleNamespace(buffers=[buffer], expert_parallel_buffers=[expert_buffer])]
    # A persistent buffer and a frozen parameter are outside DDP flat buffers.
    named = {"first": first, "second": second, "expert": expert, "bias": torch.tensor([2.0])}
    named["frozen"] = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
    return model, [buffer, expert_buffer], named


def _offloader(module, model, **kwargs):
    args = SimpleNamespace(offload_train=True, selective_offload=True, optimizer_cpu_offload=True, optimizer="adam")
    args.__dict__.update(kwargs)
    return module.MegatronTrainStateOffloader(model, None, args)


def test_actor_and_sleep_share_storage_without_changing_reference_or_parameter_views(offload_module):
    model, buffers, named = _model()
    backuper = tensor_backper.TensorBackuper.create(lambda: named.items(), single_tag=None)
    offloader = _offloader(offload_module, model)
    offloader.initialize_actor_backup(named.items(), backuper.get("actor"))
    assert offloader.shares_actor_backup
    backuper.backup("actor")
    backuper.backup("ref")
    reference = {name: tensor.clone() for name, tensor in backuper.get("ref").items()}
    parameter_storages = {name: tensor.untyped_storage() for name, tensor in named.items()}
    cpu_storages = [buffer.param_data.cpu_data.untyped_storage() for buffer in buffers]
    assert sum(storage.nbytes() for storage in cpu_storages) == (14 + 6) * 2
    for name in ("first", "second", "expert"):
        assert backuper.get("actor")[name].untyped_storage() in cpu_storages
        assert backuper.get("ref")[name].untyped_storage() not in cpu_storages
    for name in ("bias", "frozen"):
        assert backuper.get("actor")[name].untyped_storage() not in cpu_storages

    with torch.no_grad():
        for step in (10, 20):
            # Simulate an optimizer update after the last actor snapshot.
            for buffer in buffers:
                buffer.param_data.add_(step)
            live = {name: tensor.clone() for name, tensor in named.items()}
            flats = [buffer.param_data.clone() for buffer in buffers]
            assert not torch.equal(backuper.get("actor")["first"], live["first"])
            offloader.offload()
            offloader.offload()  # Repeated sleep must not read freed storage.
            for buffer in buffers:
                assert buffer.param_data.untyped_storage().nbytes() == 0
                assert buffer.grad_data.untyped_storage().nbytes() == 0
            for name in ("first", "second", "expert"):
                # Weight update reads this snapshot while the GPU model sleeps.
                assert torch.equal(backuper.get("actor")[name], live[name])
                assert named[name]._relax_cpu_offload_data.data_ptr() == backuper.get("actor")[name].data_ptr()
            offloader.reload()
            offloader.reload()
            for buffer, expected, cpu_storage in zip(buffers, flats, cpu_storages, strict=True):
                assert torch.equal(buffer.param_data, expected)  # Includes padding.
                assert torch.count_nonzero(buffer.grad_data) == 0
                assert buffer.param_data.cpu_data.untyped_storage() == cpu_storage
            for name, tensor in named.items():
                assert tensor.untyped_storage() == parameter_storages[name]
                assert torch.equal(tensor, live[name])
                assert not hasattr(tensor, "_relax_cpu_offload_data")
                assert torch.equal(backuper.get("ref")[name], reference[name])
            # KL forward uses the independent reference, then restores the actor.
            backuper.restore("ref")
            assert all(torch.equal(named[name], value) for name, value in reference.items())
            backuper.restore("actor")
            assert all(torch.equal(named[name], value) for name, value in live.items())


@pytest.mark.parametrize("unsupported", ["shared_grad", "missing_name", "offset", "wrapper"])
def test_unsupported_layout_keeps_separate_snapshot(offload_module, unsupported):
    model, buffers, named = _model()
    buffer = buffers[0]
    if unsupported == "shared_grad":
        buffer.grad_data = buffer.param_data
    elif unsupported == "missing_name":
        del named["second"]
    elif unsupported == "offset":
        buffer.param_index_map[named["first"]] = (0, 4, 0)
    else:

        class WrappedTensor(torch.Tensor):
            pass

        wrapped = named["first"].as_subclass(WrappedTensor)
        del buffer.param_index_map[named["first"]]
        buffer.param_index_map[wrapped] = (2, 6, 0)
        named["first"] = wrapped
    before = buffer.param_data.clone()
    backuper = tensor_backper.TensorBackuper.create(lambda: named.items(), single_tag=None)
    offloader = _offloader(offload_module, model)
    offloader.initialize_actor_backup(named.items(), backuper.get("actor"))
    backuper.backup("actor")
    assert not hasattr(buffer.param_data, "cpu_data")
    assert torch.equal(buffer.param_data, before)
    assert torch.equal(backuper.get("actor")["first"], named["first"])
    assert backuper.get("actor")["expert"].untyped_storage() == buffers[1].param_data.cpu_data.untyped_storage()


@pytest.mark.parametrize("mode", ["disabled", "tms"])
def test_other_offload_modes_do_not_allocate_shared_cpu_storage(offload_module, monkeypatch, mode):
    monkeypatch.setattr(offload_module, "torch_memory_saver_preloaded", lambda: True)
    monkeypatch.setattr(offload_module.MegatronTrainStateOffloader, "_init_torch_memory_saver", lambda *_: True)
    model, buffers, named = _model()
    offloader = _offloader(offload_module, model, offload_train=mode != "disabled", selective_offload=False)
    actor_backup = {}
    offloader.initialize_actor_backup(named.items(), actor_backup)
    assert not offloader.shares_actor_backup
    assert not actor_backup
    assert all(not hasattr(buffer.param_data, "cpu_data") for buffer in buffers)


@pytest.mark.parametrize("role,enable_backuper", [("actor", True), ("actor", False), ("critic", True)])
def test_actor_initial_backup_binds_only_normal_actor_snapshots(offload_module, role, enable_backuper):
    path = _ROOT / "relax/backends/megatron/actor.py"
    block = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "use_tensor_backuper"
    )
    # Run production initialization up to and including the first backup(actor).
    nodes = []
    for node in block.body:
        nodes.append(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and node.value.func.attr == "backup":
            break
    model, buffers, named = _model()
    offloader = _offloader(offload_module, model)
    args = SimpleNamespace(enable_weights_backuper=enable_backuper, megatron_to_hf_mode="bridge")
    actor = SimpleNamespace(args=args, model=model, role=role, _train_state_offloader=offloader)
    namespace = {
        "self": actor,
        "args": args,
        "TensorBackuper": tensor_backper.TensorBackuper,
        "named_params_and_buffers": lambda *_args, **_kwargs: named.items(),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    shared = role == "actor" and enable_backuper
    assert offloader.shares_actor_backup == shared
    assert all(hasattr(buffer.param_data, "cpu_data") == shared for buffer in buffers)
    assert torch.equal(actor.weights_backuper.get("actor")["first"], named["first"])


def test_sleep_rejects_reference_before_overwriting_shared_actor_snapshot(offload_module):
    # Execute the real actor method without its GPU imports and timer wrapper.
    path = _ROOT / "relax/backends/megatron/actor.py"
    actor_class = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
    )
    sleep = next(node for node in actor_class.body if isinstance(node, ast.FunctionDef) and node.name == "sleep")
    sleep.decorator_list = []
    namespace = {"clear_memory": Mock(), "print_memory": Mock(), "destroy_process_groups": Mock()}
    exec(compile(ast.Module(body=[sleep], type_ignores=[]), str(path), "exec"), namespace)
    model, buffers, named = _model()
    backuper = tensor_backper.TensorBackuper.create(lambda: named.items(), single_tag=None)
    offloader = _offloader(offload_module, model)
    offloader.initialize_actor_backup(named.items(), backuper.get("actor"))
    backuper.backup("actor")
    actor_weights = backuper.get("actor")["first"].clone()
    with torch.no_grad():
        named["first"].add_(100)
    actor = SimpleNamespace(
        args=SimpleNamespace(offload_train=True, use_critic=False),
        role="actor",
        _active_model_tag="ref",
        _train_state_offloader=offloader,
        _flush_resident_sft_parameters=Mock(),
    )
    with pytest.raises(RuntimeError, match="Restore actor weights before sleep"):
        namespace["sleep"](actor)
    namespace["destroy_process_groups"].assert_not_called()
    assert torch.equal(backuper.get("actor")["first"], actor_weights)
    assert buffers[0].param_data.untyped_storage().nbytes() > 0
    backuper.restore("actor")
    actor._active_model_tag = "actor"
    namespace["sleep"](actor)
    assert buffers[0].param_data.untyped_storage().nbytes() == 0
    offloader.reload()
    assert torch.equal(named["first"], actor_weights)
