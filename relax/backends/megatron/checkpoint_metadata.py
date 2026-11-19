# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Metadata checks used when selecting a checkpoint resume path."""

import hashlib
import re
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any

from relax.utils.distributed_utils import get_gloo_group


def _collective_checkpoint_probe(probe: Callable[[], Any]) -> Any:
    """Read local checkpoint metadata, then agree before collective loading.

    All ranks enter this function even when a mount or metadata read fails.
    Inconsistent views fail together instead of selecting different loaders.
    """
    import torch.distributed as dist

    try:
        result = (None, probe())
    except Exception as exc:
        result = (f"{type(exc).__name__}: {exc}", None)
    results = [result]
    if dist.is_initialized():
        group = get_gloo_group()
        results = [None] * dist.get_world_size(group)
        dist.all_gather_object(results, result, group=group)
    errors = [error for error, _ in results if error is not None]
    if errors:
        raise RuntimeError(f"Checkpoint metadata probe failed: {errors}")
    if any(value != results[0][1] for _, value in results[1:]):
        raise RuntimeError("Checkpoint metadata differs across ranks; check shared storage before resuming.")
    return results[0][1]


def _checkpoint_has_optimizer_state(checkpoint_dir: Path, args: Namespace) -> bool:
    """Whether a DCP checkpoint directory carries optimizer state.

    Megatron omits the ``optimizer`` / ``opt_param_scheduler`` entries when
    saving with ``--no-save-optim``. For DCP-based formats the shard index
    (``.metadata``) records the stored shard keys, including the
    ``chained_<index>.optimizer`` namespaces used for dense/expert optimizers.
    Inspect those namespaces without reading any tensor data. Legacy ``torch``
    checkpoints have no ``.metadata`` and are assumed loadable, keeping the
    previous fail-loudly behavior.
    """
    if getattr(args, "ckpt_format", None) not in ("torch_dist", "torch_dcp", "fsdp_dtensor"):
        return True
    if not (Path(checkpoint_dir) / ".metadata").is_file():
        return True
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(str(checkpoint_dir)).read_metadata()
    return any(
        re.match(r"^(?:chained_\d+\.)*optimizer(?:[./]|$)", str(key)) is not None
        for key in metadata.state_dict_metadata
    )


def _validate_strict_resume_metadata(args: Namespace, checkpoint_dir: Path) -> dict:
    """Fail before collective tensor load if a full synchronous resume is
    incomplete."""
    import math

    import torch
    from megatron.core import dist_checkpointing
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    from relax.utils.checkpoint_resume import validate_dataset_state, validate_strict_rl_checkpoint_mode

    validate_strict_rl_checkpoint_mode(args)
    flags = ("no_load_optim", "no_load_rng", "finetune", "reset_optimizer_states", "save_lora_only")
    if any(getattr(args, name, False) for name in flags):
        raise RuntimeError("Strict resume forbids optimizer/RNG reset, finetuning, and LoRA-only checkpoints")
    if args.ckpt_format != "torch_dist" or not re.fullmatch(r"iter_\d{7}", checkpoint_dir.name):
        raise RuntimeError("Strict resume requires a numbered torch_dist checkpoint")
    common = dist_checkpointing.load_common_state_dict(checkpoint_dir)
    iteration = int(checkpoint_dir.name[5:])
    if type(common.get("iteration")) is not int or common["iteration"] != iteration:
        raise RuntimeError("Strict resume checkpoint iteration mismatch")
    saved = common.get("args")
    value = lambda key: saved.get(key) if isinstance(saved, dict) else getattr(saved, key, None)
    if saved is None or value("no_save_optim") or value("no_save_rng") or value("save_lora_only"):
        raise RuntimeError("Strict resume requires saved args, optimizer and RNG")
    layout = (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "data_parallel_size",
        "world_size",
    )
    if any(value(key) is None or value(key) != getattr(args, key, None) for key in layout):
        raise RuntimeError("Strict resume parallel layout mismatch; RNG continuity cannot be guaranteed")
    scheduler = common.get("opt_param_scheduler", common.get("lr_scheduler"))
    steps = scheduler.get("num_steps") if isinstance(scheduler, dict) else None
    if isinstance(steps, bool) or not isinstance(steps, (int, float)) or not math.isfinite(steps) or steps < 0:
        raise RuntimeError("Strict resume requires valid optimizer scheduler progress")
    metadata = FileSystemReader(str(checkpoint_dir)).read_metadata()
    keys = metadata.state_dict_metadata
    # MCore stores ShardedTensor.key, dropping the outer "model" dict key;
    # Kimi shards are language_model.*, vision_tower.* and mm_projector.*.
    optimizer_key = re.compile(r"^(?:chained_\d+\.)*optimizer[./]")
    if not any(
        isinstance(entry, TensorStorageMetadata) and not optimizer_key.match(key) for key, entry in keys.items()
    ):
        raise RuntimeError("Strict resume checkpoint has no model tensors")
    if not any(re.match(r"^(?:chained_\d+\.)*optimizer[./]", key) for key in keys):
        raise RuntimeError("Strict resume checkpoint has no optimizer tensors")
    if not common.get("rng_state") and not any(re.match(r"^rng_state[./]", key) for key in keys):
        raise RuntimeError("Strict resume checkpoint has no RNG state")
    storage = metadata.storage_data or {}
    stored = {(index.fqn, index.offset) for index in storage}
    for key, entry in keys.items():
        chunks = getattr(entry, "chunks", None)
        offsets = [chunk.offsets for chunk in chunks] if chunks is not None else [None]
        for offset in offsets:
            if (key, offset) not in stored:
                raise RuntimeError(f"Strict resume missing shard storage for {key}")
    sizes = {}
    for info in storage.values():
        relative = Path(info.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Strict resume invalid shard path")
        if relative not in sizes:
            sizes[relative] = (checkpoint_dir / relative).stat().st_size
        if info.offset < 0 or info.length <= 0 or info.offset + info.length > sizes[relative]:
            raise RuntimeError("Strict resume truncated checkpoint shard")
    cursor = checkpoint_dir.parent / "dataset" / f"global_dataset_state_dict_{iteration}.pt"
    state = torch.load(cursor, map_location="cpu", weights_only=False)
    validate_dataset_state(state, args, iteration)
    return {
        "iteration": iteration,
        "scheduler_num_steps": steps,
        "shard_count": len(sizes),
        "dataset_sha256": hashlib.sha256(cursor.read_bytes()).hexdigest(),
    }
