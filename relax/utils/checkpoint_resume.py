# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Durable cursor state for synchronous, full RL checkpoint recovery."""

import hashlib
import os
import tempfile
from pathlib import Path


def validate_strict_rl_checkpoint_mode(args) -> None:
    if (
        not getattr(args, "colocate", False)
        or not getattr(args, "rollout_global_dataset", False)
        or any(
            getattr(args, name, False)
            for name in ("fully_async", "partial_rollout", "async_save", "agentic_prelaunch", "use_streaming_dataset")
        )
    ):
        raise RuntimeError("Strict checkpoint requires synchronous colocated eager-dataset RL without prelaunch")


def dataset_identity(args) -> dict:
    # The strict path currently supports a single local, eager dataset only.
    path = Path(args.prompt_data)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "prompt_sha256": digest.hexdigest(),
        "n_samples_per_prompt": args.n_samples_per_prompt,
        "rollout_seed": args.rollout_seed,
        "rollout_shuffle": args.rollout_shuffle,
        "preprocessing": {
            name: getattr(args, name, None)
            for name in (
                "input_key",
                "label_key",
                "metadata_key",
                "multimodal_keys",
                "system_prompt",
                "tool_key",
                "apply_chat_template",
                "apply_chat_template_kwargs",
                "custom_prompt_path",
                "rollout_max_prompt_len",
            )
        },
    }


def validate_dataset_state(state: dict, args, rollout_id: int) -> None:
    if not isinstance(state, dict):
        raise RuntimeError("Strict resume requires a dataset state dictionary")
    contract = state.get("resume_contract")
    size = contract.get("dataset_size") if isinstance(contract, dict) else None
    if type(size) is not int or size <= 0:
        raise RuntimeError("Strict resume requires a positive effective dataset size")
    expected = {"version": 1, "rollout_id": rollout_id, "dataset_size": size, **dataset_identity(args)}
    if contract != expected:
        raise RuntimeError("Strict resume dataset identity or rollout ID mismatch")
    for key in ("sample_offset", "epoch_id", "sample_group_index", "sample_index"):
        if type(state.get(key)) is not int or state[key] < 0:
            raise RuntimeError(f"Strict resume requires valid dataset {key}")
    if (
        state["sample_offset"] > size
        or state["epoch_id"] * size + state["sample_offset"] != state["sample_group_index"]
    ):
        raise RuntimeError("Strict resume dataset epoch/offset/group counters disagree")
    if state["sample_index"] != state["sample_group_index"] * args.n_samples_per_prompt:
        raise RuntimeError("Strict resume dataset sample/group counters disagree")
    if not isinstance(state.get("metadata"), dict) or state.get("use_streaming") is not False:
        raise RuntimeError("Strict resume currently requires an eager dataset with metadata")


def atomic_save_dataset_state(state: dict, path: str) -> None:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
