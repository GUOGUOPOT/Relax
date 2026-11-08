# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check an explicitly selected existing cluster; never start Ray or launch training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


MODEL_FILES = (
    "config.json",
    "configuration_kimi_k3.py",
    "encoding_k3.py",
    "generation_config.json",
    "kimi_k3_processor.py",
    "kimi_k3_vision_processing.py",
    "media_utils.py",
    "modeling_kimi_k3.py",
    "modeling_kimi_linear.py",
    "preprocessor_config.json",
    "tiktoken.model",
    "tokenization_kimi.py",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)
SAMPLE_BYTES = 65536


class PreflightError(RuntimeError):
    pass


def require(condition: Any, code: str) -> None:
    if not condition:
        raise PreflightError(code)


def file_identity(path: str | Path, *, full: bool = True) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(before.st_size > 0, "empty_file")
        head = stream.read(SAMPLE_BYTES)
        stream.seek(max(0, before.st_size - SAMPLE_BYTES))
        tail = stream.read(SAMPLE_BYTES)
        result = {"bytes": before.st_size, "head_tail_sha256": hashlib.sha256(head + tail).hexdigest()}
        if full:
            stream.seek(0)
            digest = hashlib.sha256()
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
            result["sha256"] = digest.hexdigest()
        after = os.fstat(stream.fileno())
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "file_changed_during_read")
        return result


def model_identity(directory: str, shards: int) -> dict[str, Any]:
    root = Path(directory).resolve(strict=True)
    hashes = {name: file_identity(root / name)["sha256"] for name in MODEL_FILES}
    config = json.loads((root / "config.json").read_text())
    require(config.get("model_type") == "kimi_k3", "unexpected_model_type")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    require(len(names) == shards, "model_shard_count_mismatch")
    sizes = {}
    for name in names:
        path = (root / name).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file(), "model_shard_path_invalid")
        sizes[name] = path.stat().st_size
        require(sizes[name] > 0, "model_shard_empty")
    return {
        "source_sha256": hashes,
        "shard_sizes": sizes,
        "shards": len(sizes),
        "weight_content_sha256_verified": False,
    }


def media_catalog(path: str, roots: list[str]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {root: [] for root in roots}
    seen = set()
    with Path(path).open() as stream:
        for line in stream:
            item = json.loads(line)
            require(item.get("status") == "verified", "media_manifest_entry_unverified")
            source = Path(item["path"])
            require(source.is_absolute() and ".." not in source.parts, "media_manifest_path_invalid")
            require(str(source) not in seen, "media_manifest_path_duplicate")
            seen.add(str(source))
            matches = [root for root in roots if source.is_relative_to(root)]
            require(matches, "media_manifest_root_not_authorized")
            groups[max(matches, key=len)].append(item)
    require(all(groups.values()), "media_root_has_no_representative")
    for entries in groups.values():
        entries.sort(key=lambda item: item["path"])
    return file_identity(path), groups


def check_files(spec: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    actual_model = model_identity(spec["model_dir"], spec["shards"])
    require(actual_model == spec["model_identity"], "node_model_identity_mismatch")
    actual_data = file_identity(spec["prepared_data"], full=spec["node_full_sha256"])
    require(all(spec["data_identity"].get(k) == v for k, v in actual_data.items()), "node_data_identity_mismatch")
    require(
        file_identity(spec["media_manifest"])["sha256"] == spec["manifest_identity"]["sha256"],
        "node_manifest_identity_mismatch",
    )
    roots = [Path(root).resolve(strict=True) for root in spec["media_roots"]]
    checked = []
    for sample in samples:
        path = Path(sample["path"]).resolve(strict=True)
        require(any(path.is_relative_to(root) for root in roots), "node_media_outside_roots")
        identity = file_identity(path)
        require(
            identity["sha256"] == sample["sha256"] and identity["bytes"] == sample["bytes"],
            "node_media_identity_mismatch",
        )
        checked.append({"path": sample["path"], "sha256": identity["sha256"], "bytes": identity["bytes"]})
    return {
        "hostname": socket.gethostname(),
        "model_sources_and_shard_sizes_match": True,
        "weight_content_sha256_verified": False,
        "prepared_identity": actual_data,
        "prepared_full_sha256_verified": spec["node_full_sha256"],
        "media_checked": checked,
        "media_scope": "explicit_representatives_per_root_only",
        "manifest_sha256": spec["manifest_identity"]["sha256"],
    }


def topology(nodes: list[dict[str, Any]], expected_nodes: int, gpus: int) -> list[dict[str, Any]]:
    alive = [node for node in nodes if node.get("Alive") and node.get("Resources", {}).get("GPU", 0) > 0]
    require(len(alive) == expected_nodes, "gpu_node_count_mismatch")
    require(len({node["NodeID"] for node in alive}) == expected_nodes, "node_identity_duplicate")
    require(len({node["NodeManagerAddress"] for node in alive}) == expected_nodes, "node_address_duplicate")
    require(all(node["Resources"]["GPU"] == gpus for node in alive), "node_gpu_resource_count_mismatch")
    return sorted(alive, key=lambda node: node["NodeID"])


def check_gpu_results(results: list[dict[str, Any]], node_ids: list[str], gpus: int) -> None:
    require(len(results) == len(node_ids) * gpus, "gpu_result_count_mismatch")
    require(all(item.get("status") == "PASS" for item in results), "gpu_probe_failed")
    uuids = [item.get("uuid") for item in results]
    require(
        all(isinstance(value, str) and value and not value.upper().startswith("MIG-") for value in uuids),
        "physical_gpu_uuid_required",
    )
    require(len(set(uuids)) == len(results), "physical_gpu_uuid_duplicate")
    require(len({(item["torch"], item["cuda"]) for item in results}) == 1, "gpu_software_version_mismatch")
    for node_id in node_ids:
        local = [item for item in results if item["node_id"] == node_id]
        require(len(local) == gpus, "gpu_node_coverage_mismatch")
        require(len({tuple(item["ray_gpu_ids"]) for item in local}) == gpus, "gpu_assignment_duplicate")

    require(
        all("B300" in item["name"].upper() and "MIG" not in item["name"].upper() for item in results),
        "gpu_type_mismatch",
    )
    require(all(item["tensor_sum"] == 32640 for item in results), "gpu_tensor_value_mismatch")
    require(len({tuple(item["capability"]) for item in results}) == 1, "gpu_capability_mismatch")


class Probe:
    def __init__(self, rank: int = -1) -> None:
        self.rank, self.store, self.group_ready = rank, None, False

    def files(self, spec: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
        import ray

        try:
            result = {"status": "PASS", **check_files(spec, samples)}
        except Exception as exc:
            result = {
                "status": "FAILED",
                "error": str(exc) if isinstance(exc, PreflightError) else "file_probe_failed",
                "error_type": type(exc).__name__,
            }
        return {"ray_node_id": ray.get_runtime_context().get_node_id(), **result}

    def gpu(self, driver_version: bool) -> dict[str, Any]:
        import ray

        try:
            return {"status": "PASS", **self._gpu(driver_version)}
        except Exception as exc:
            return {
                "status": "FAILED",
                "rank": self.rank,
                "node_id": ray.get_runtime_context().get_node_id(),
                "error": str(exc) if isinstance(exc, PreflightError) else "gpu_probe_failed",
                "error_type": type(exc).__name__,
            }

    def _gpu(self, driver_version: bool) -> dict[str, Any]:
        import ray
        import torch

        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "assigned_cuda_device_not_isolated")
        ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
        require(len(ids) == 1, "ray_gpu_assignment_invalid")
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        value = torch.arange(256, device="cuda", dtype=torch.float32).sum()
        torch.cuda.synchronize()
        result = {
            "rank": self.rank,
            "node_id": ray.get_runtime_context().get_node_id(),
            "ray_gpu_ids": ids,
            "name": props.name,
            "uuid": str(props.uuid) if getattr(props, "uuid", None) is not None else None,
            "capability": [props.major, props.minor],
            "total_memory": props.total_memory,
            "tensor_sum": int(value.item()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        if driver_version:
            import subprocess

            process = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            result["driver_versions"] = sorted(set(process.stdout.splitlines()))
        return result

    def rendezvous(self, world: int, timeout: int) -> tuple[str, int]:
        import ray
        import torch.distributed as dist

        host = ray.util.get_node_ip_address()
        self.store = dist.TCPStore(host, 0, world, True, timedelta(seconds=timeout), wait_for_workers=False)
        return host, self.store.port

    def nccl(self, host: str, port: int, world: int, timeout: int) -> dict[str, Any]:
        import torch
        import torch.distributed as dist

        if self.store is None:
            self.store = dist.TCPStore(host, port, world, False, timedelta(seconds=timeout))
        dist.init_process_group(
            "nccl",
            store=self.store,
            rank=self.rank,
            world_size=world,
            timeout=timedelta(seconds=timeout),
            device_id=torch.device("cuda", 0),
        )
        self.group_ready = True
        reduced = torch.tensor([self.rank], device="cuda", dtype=torch.int64)
        dist.all_reduce(reduced)
        value = torch.tensor([self.rank], device="cuda", dtype=torch.int64)
        gathered = [torch.empty_like(value) for _ in range(world)]
        dist.all_gather(gathered, value)
        torch.cuda.synchronize()
        require(reduced.item() == world * (world - 1) // 2, "nccl_all_reduce_value_mismatch")
        require([item.item() for item in gathered] == list(range(world)), "nccl_all_gather_value_mismatch")
        return {"rank": self.rank, "all_reduce": "PASS", "all_gather": "PASS"}

    def close(self) -> None:
        if self.group_ready:
            import torch.distributed as dist

            dist.destroy_process_group()
            self.group_ready = False
        self.store = None


def cluster_checks(args: argparse.Namespace, spec: dict[str, Any], groups: dict[str, list], report: dict) -> None:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    require(not ray.is_initialized(), "existing_ray_driver_not_owned")
    owned = []
    pending = None
    try:
        report["stage"] = "connect"
        ray.init(address=args.ray_address, ignore_reinit_error=False, log_to_driver=False)
        report["stage"] = "topology"
        nodes = topology(ray.nodes(), args.nodes, args.gpus_per_node)
        node_ids = [node["NodeID"] for node in nodes]
        remote = ray.remote(max_restarts=0, max_task_retries=0)(Probe)
        report["stage"] = "node_files"
        cpu = []
        for node in node_ids:
            actor = remote.options(
                num_cpus=1, num_gpus=0, scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False)
            ).remote()
            owned.append(actor)
            cpu.append(actor)
        samples = [
            [
                entries[(i * args.samples_per_root + j) % len(entries)]
                for entries in groups.values()
                for j in range(min(args.samples_per_root, len(entries)))
            ]
            for i in range(len(nodes))
        ]
        report["node_files"] = dict(
            zip(
                node_ids,
                ray.get(
                    [actor.files.remote(spec, sample) for actor, sample in zip(cpu, samples)], timeout=args.timeout_s
                ),
                strict=True,
            )
        )
        require(
            all(
                result["status"] == "PASS" and result["ray_node_id"] == node
                for node, result in report["node_files"].items()
            ),
            "node_file_checks_failed",
        )
        for actor in cpu:
            ray.kill(actor, no_restart=True)
            owned.remove(actor)
        # File checks complete before any GPU reservation.
        report["stage"] = "gpus"
        gpu = []
        for node in node_ids:
            for _ in range(args.gpus_per_node):
                actor = remote.options(
                    num_cpus=1,
                    num_gpus=1,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False),
                ).remote(len(gpu))
                gpu.append(actor)
                owned.append(actor)
        report["gpus"] = ray.get([actor.gpu.remote(args.driver_version) for actor in gpu], timeout=args.timeout_s)
        check_gpu_results(report["gpus"], node_ids, args.gpus_per_node)
        if args.driver_version:
            require(
                len({tuple(item["driver_versions"]) for item in report["gpus"]}) == 1, "gpu_driver_version_mismatch"
            )
        if args.nccl:
            report["stage"] = "nccl"
            host, port = ray.get(gpu[0].rendezvous.remote(len(gpu), args.timeout_s), timeout=args.timeout_s)
            result = ray.get(
                [actor.nccl.remote(host, port, len(gpu), args.timeout_s) for actor in gpu], timeout=args.timeout_s
            )
            require(sorted(item["rank"] for item in result) == list(range(len(gpu))), "nccl_rank_coverage_mismatch")
            report["nccl"] = {"status": "PASS", "ranks": len(result), "tests": ["all_reduce", "all_gather"]}
    except Exception:
        pending = sys.exc_info()[1]
        raise
    finally:
        cleanup_ok = True
        if ray.is_initialized():
            live = [actor for actor in owned if actor not in locals().get("cpu", [])]
            try:
                ray.get([actor.close.remote() for actor in live], timeout=min(args.timeout_s, 15))
            except Exception:
                cleanup_ok = False
            for actor in owned:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    cleanup_ok = False
            ray.shutdown()  # Disconnect this driver; never shuts down the cluster.
        report["owned_actor_cleanup"] = "PASS" if cleanup_ok else "FAILED"
        if not cleanup_ok and pending is None:
            raise PreflightError("owned_actor_cleanup_failed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ray-address", help="Existing Ray Core host:port or ray://host:port; not an HTTP dashboard URL"
    )
    for name in ("model-dir", "prepared-data", "media-manifest", "report"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--media-root", action="append", required=True)
    for name, default in (
        ("nodes", 16),
        ("gpus-per-node", 8),
        ("expected-shards", 96),
        ("timeout-s", 180),
        ("samples-per-root", 1),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    for name in ("expected-prepared-sha256", "expected-media-manifest-sha256", "expected-model-encoder-sha256"):
        parser.add_argument("--" + name)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    for name in ("nccl", "driver-version", "node-data-sha256"):
        parser.add_argument("--" + name, action="store_true")
    args = parser.parse_args(argv)
    require(
        all(
            getattr(args, name) > 0
            for name in ("nodes", "gpus_per_node", "expected_shards", "timeout_s", "samples_per_root")
        ),
        "positive_parameter_required",
    )
    require(
        args.local or args.dry_run or args.ray_address not in (None, "", "auto", "local"),
        "explicit_ray_address_required",
    )
    require(not (args.local and args.nccl), "local_cannot_run_nccl")
    if args.ray_address is not None:
        try:
            parsed = urlsplit(args.ray_address if "://" in args.ray_address else "//" + args.ray_address)
            port = parsed.port
        except ValueError as exc:
            raise PreflightError("ray_address_invalid") from exc
        require(
            parsed.scheme in ("", "ray")
            and parsed.hostname
            and port
            and not parsed.username
            and not parsed.password
            and not parsed.path
            and not parsed.query
            and not parsed.fragment,
            "ray_address_invalid",
        )
    require(len(set(args.media_root)) == len(args.media_root), "media_root_duplicate")
    require(
        all(str(Path(root)) == root and ".." not in Path(root).parts for root in args.media_root),
        "media_root_not_canonical",
    )
    require(
        all(
            Path(path).is_absolute()
            for path in [args.model_dir, args.prepared_data, args.media_manifest, args.report, *args.media_root]
        ),
        "absolute_paths_required",
    )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = {
        "nodes": args.nodes,
        "gpus_per_node": args.gpus_per_node,
        "expected_gpu_slots": args.nodes * args.gpus_per_node,
        "model_dir": args.model_dir,
        "prepared_data": args.prepared_data,
        "media_manifest": args.media_manifest,
        "media_roots": args.media_root,
        "ray_address": args.ray_address,
        "nccl_requested": args.nccl,
        "node_prepared_verification": "full_sha256" if args.node_data_sha256 else "size_and_head_tail_sha256",
    }
    if args.dry_run:
        sys.stdout.write(json.dumps({"status": "NOT_RUN", "training_ready": False, "plan": plan}) + "\n")
        return 0
    output = Path(args.report)
    require(not output.exists() and not output.is_symlink(), "report_already_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "FAILED",
        "training_ready": False,
        "plan": plan,
        "nccl": {"status": "UNVERIFIED"},
    }
    code = 2
    # Reserve the report before connecting; collisions cannot start a cluster probe.
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        try:
            data = file_identity(args.prepared_data)
            manifest, groups = media_catalog(args.media_manifest, args.media_root)
            model = model_identity(args.model_dir, args.expected_shards)
            expected = [
                args.expected_prepared_sha256,
                args.expected_media_manifest_sha256,
                args.expected_model_encoder_sha256,
            ]
            actual = [data["sha256"], manifest["sha256"], model["source_sha256"]["encoding_k3.py"]]
            require(
                all(want is None or want == got for want, got in zip(expected, actual)), "expected_identity_mismatch"
            )
            report["driver_identities"] = {"prepared": data, "manifest": manifest, "model": model}
            report["caller_identity_expectations_verified"] = all(expected)
            spec = {
                "model_dir": args.model_dir,
                "shards": args.expected_shards,
                "model_identity": model,
                "prepared_data": args.prepared_data,
                "data_identity": data,
                "node_full_sha256": args.node_data_sha256,
                "media_manifest": args.media_manifest,
                "manifest_identity": manifest,
                "media_roots": args.media_root,
            }
            if args.local:
                report["local_files"] = check_files(
                    spec, [item for entries in groups.values() for item in entries[: args.samples_per_root]]
                )
            else:
                cluster_checks(args, spec, groups, report)
            report["status"] = "PASS" if not args.local and all(expected) else "UNVERIFIED"
            code = 0 if report["status"] == "PASS" else 3
        except Exception as exc:
            report["error"] = str(exc) if isinstance(exc, PreflightError) else "probe_failed"
            report["error_type"] = type(exc).__name__
        finally:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.write(json.dumps({"status": report["status"], "report": str(output), "training_ready": False}) + "\n")
    return code


if __name__ == "__main__":

    def interrupted(_signum: int, _frame: Any) -> None:
        raise PreflightError("preflight_interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except (PreflightError, OSError) as error:
        sys.stderr.write(
            json.dumps({"error": str(error) if isinstance(error, PreflightError) else "preflight_io_failed"}) + "\n"
        )
        raise SystemExit(2)
