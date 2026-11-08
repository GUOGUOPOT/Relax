"""Contract tests for file/topology gates; never allocate GPUs or connect Ray."""

from __future__ import annotations

import builtins
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from examples.agentspark_agentic import cluster_preflight as probe


def _fixture(tmp_path: Path) -> list[str]:
    model = tmp_path / "model"
    model.mkdir()
    for name in probe.MODEL_FILES:
        (model / name).write_text(name)
    (model / "config.json").write_text('{"model_type":"kimi_k3"}')
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "first.safetensors", "b": "second.safetensors"}})
    )
    for name in ("first.safetensors", "second.safetensors"):
        (model / name).write_bytes(b"nonempty fixture: not a verified weight tensor")
    data = tmp_path / "prepared.jsonl"
    data.write_bytes(b'{"messages": [{"role": "user", "content": "test"}]}\n')
    media = tmp_path / "images"
    media.mkdir()
    picture = media / "fixture.bin"
    picture.write_bytes(b"representative bytes: preflight verifies hash, not image decoding")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "status": "verified",
                "path": str(picture),
                "bytes": picture.stat().st_size,
                "sha256": hashlib.sha256(picture.read_bytes()).hexdigest(),
            }
        )
        + "\n"
    )
    return [
        "--model-dir",
        str(model),
        "--prepared-data",
        str(data),
        "--media-manifest",
        str(manifest),
        "--media-root",
        str(media),
        "--report",
        str(tmp_path / "report.json"),
        "--expected-shards",
        "2",
    ]


def _forbid_accelerator_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"ray", "torch"}:
            raise AssertionError("local/dry-run must not import Ray or Torch")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_dry_run_has_no_filesystem_gpu_or_ray_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    _forbid_accelerator_imports(monkeypatch)
    args = [
        "--model-dir",
        str(tmp_path / "absent-model"),
        "--prepared-data",
        str(tmp_path / "absent-data"),
        "--media-manifest",
        str(tmp_path / "absent-manifest"),
        "--media-root",
        str(tmp_path / "absent-root"),
        "--report",
        str(tmp_path / "absent-report"),
        "--dry-run",
    ]
    assert probe.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "NOT_RUN" and result["training_ready"] is False
    assert result["plan"]["expected_gpu_slots"] == 128
    assert list(tmp_path.iterdir()) == []


def test_local_files_are_verified_without_claiming_cluster_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path)
    _forbid_accelerator_imports(monkeypatch)
    assert probe.main(args + ["--local"]) == 3
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "UNVERIFIED" and report["training_ready"] is False
    assert report["nccl"]["status"] == "UNVERIFIED" and "gpus" not in report
    assert report["local_files"]["prepared_full_sha256_verified"] is False
    assert report["local_files"]["weight_content_sha256_verified"] is False
    assert len(report["local_files"]["media_checked"]) == 1
    assert (tmp_path / "report.json").stat().st_mode & 0o777 == 0o600


def test_wrong_expected_identity_fails_before_ray_or_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _fixture(tmp_path)
    _forbid_accelerator_imports(monkeypatch)
    assert probe.main(args + ["--ray-address", "explicit-head:6379", "--expected-prepared-sha256", "0" * 64]) == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["error"] == "expected_identity_mismatch"
    assert "node_files" not in report


def test_existing_report_fails_before_cluster_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _fixture(tmp_path)
    path = tmp_path / "report.json"
    path.write_text("do not replace")
    _forbid_accelerator_imports(monkeypatch)
    with pytest.raises(probe.PreflightError, match="^report_already_exists$"):
        probe.main(args + ["--ray-address", "explicit-head:6379"])
    assert path.read_text() == "do not replace"


@pytest.mark.parametrize(
    "address", ["auto", "local", "http://head:8265", "https://head:8265", "ray://user:secret@head:10001"]
)
def test_implicit_or_dashboard_addresses_are_rejected(tmp_path: Path, address: str) -> None:
    with pytest.raises(probe.PreflightError):
        probe.parse_args(_fixture(tmp_path) + ["--ray-address", address])


def test_duplicate_and_traversing_media_roots_are_rejected(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    with pytest.raises(probe.PreflightError, match="^media_root_duplicate$"):
        probe.parse_args(args + ["--local", "--media-root", str(tmp_path / "images")])
    with pytest.raises(probe.PreflightError, match="^media_root_not_canonical$"):
        probe.parse_args(args + ["--local", "--media-root", str(tmp_path) + "/other/../images"])


def test_size_and_edge_fingerprint_does_not_claim_full_content_identity(tmp_path: Path) -> None:
    path = tmp_path / "data"
    content = b"a" * probe.SAMPLE_BYTES + b"middle-A" + b"z" * probe.SAMPLE_BYTES
    path.write_bytes(content)
    original = probe.file_identity(path)
    path.write_bytes(content.replace(b"middle-A", b"middle-B"))
    partial = probe.file_identity(path, full=False)
    assert "sha256" not in partial
    assert partial["head_tail_sha256"] == original["head_tail_sha256"]
    assert probe.file_identity(path)["sha256"] != original["sha256"]


def test_shard_index_controls_count_and_rejects_escape(tmp_path: Path) -> None:
    args = probe.parse_args(_fixture(tmp_path) + ["--local"])
    model = Path(args.model_dir)
    (model / "first.safetensors_download_tmp").write_bytes(b"not a referenced shard")
    assert probe.model_identity(str(model), 2)["shards"] == 2
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"nonempty")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "../outside.safetensors", "b": "second.safetensors"}})
    )
    with pytest.raises(probe.PreflightError, match="^model_shard_path_invalid$"):
        probe.model_identity(str(model), 2)


def test_sample_hash_and_authorized_root_are_actual_file_checks(tmp_path: Path) -> None:
    args = probe.parse_args(_fixture(tmp_path) + ["--local"])
    manifest, groups = probe.media_catalog(args.media_manifest, args.media_root)
    spec = {
        "model_dir": args.model_dir,
        "shards": 2,
        "model_identity": probe.model_identity(args.model_dir, 2),
        "prepared_data": args.prepared_data,
        "data_identity": probe.file_identity(args.prepared_data),
        "node_full_sha256": False,
        "media_manifest": args.media_manifest,
        "manifest_identity": manifest,
        "media_roots": args.media_root,
    }
    samples = next(iter(groups.values()))
    Path(samples[0]["path"]).write_bytes(b"changed pixels")
    with pytest.raises(probe.PreflightError, match="^node_media_identity_mismatch$"):
        probe.check_files(spec, samples)


def _gpu(node: str, slot: int) -> dict[str, Any]:
    return {
        "status": "PASS",
        "node_id": node,
        "ray_gpu_ids": [str(slot)],
        "uuid": f"GPU-{node}-{slot}",
        "name": "NVIDIA B300",
        "tensor_sum": 32640,
        "capability": [10, 3],
        "torch": "fixture",
        "cuda": "fixture",
    }


def test_physical_gpu_coverage_is_global_not_logical_ray_slots() -> None:
    nodes = [
        {"Alive": True, "NodeID": name, "NodeManagerAddress": f"address-{name}", "Resources": {"GPU": 8}}
        for name in ("node-A", "node-B")
    ]
    probe.topology(nodes, 2, 8)
    results = [_gpu(name, slot) for name in ("node-A", "node-B") for slot in range(8)]
    probe.check_gpu_results(results, ["node-A", "node-B"], 8)
    for slot in range(8):
        results[8 + slot]["uuid"] = results[slot]["uuid"]
    with pytest.raises(probe.PreflightError, match="^physical_gpu_uuid_duplicate$"):
        probe.check_gpu_results(results, ["node-A", "node-B"], 8)
    nodes[1]["NodeManagerAddress"] = nodes[0]["NodeManagerAddress"]
    with pytest.raises(probe.PreflightError, match="^node_address_duplicate$"):
        probe.topology(nodes, 2, 8)


@pytest.mark.parametrize(
    ("key", "bad", "code"),
    [
        ("uuid", None, "physical_gpu_uuid_required"),
        ("torch", "different", "gpu_software_version_mismatch"),
        ("name", "NVIDIA A100", "gpu_type_mismatch"),
        ("tensor_sum", 0, "gpu_tensor_value_mismatch"),
    ],
)
def test_gpu_probe_failures_cannot_be_reported_as_pass(key: str, bad: Any, code: str) -> None:
    results = [_gpu("node-A", 0), _gpu("node-A", 1)]
    results[1][key] = bad
    with pytest.raises(probe.PreflightError, match="^" + code + "$"):
        probe.check_gpu_results(results, ["node-A"], 2)


def test_existing_ray_driver_is_never_disconnected(monkeypatch: pytest.MonkeyPatch) -> None:
    ray = ModuleType("ray")
    ray.is_initialized = lambda: True
    calls = []
    ray.shutdown = lambda: calls.append("shutdown")
    ray.init = lambda **kwargs: calls.append("init")
    strategies = ModuleType("ray.util.scheduling_strategies")
    strategies.NodeAffinitySchedulingStrategy = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", ModuleType("ray.util"))
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", strategies)
    with pytest.raises(probe.PreflightError, match="^existing_ray_driver_not_owned$"):
        probe.cluster_checks(SimpleNamespace(), {}, {}, {})
    assert calls == []


@pytest.mark.parametrize("construction_failure", [False, True])
def test_file_failures_never_reserve_gpus_and_partial_construction_is_owned(
    monkeypatch: pytest.MonkeyPatch, construction_failure: bool
) -> None:
    state = {"connected": False, "created": [], "killed": [], "shutdown": 0, "attempts": 0}
    ray = ModuleType("ray")
    ray.is_initialized = lambda: state["connected"]
    ray.init = lambda **kwargs: state.update(connected=True)
    ray.nodes = lambda: [
        {"Alive": True, "NodeID": node, "NodeManagerAddress": node + "-address", "Resources": {"GPU": 1}}
        for node in ("a", "b")
    ]

    def shutdown() -> None:
        state["shutdown"] += 1
        state["connected"] = False

    ray.shutdown = shutdown
    ray.get = lambda values, **kwargs: values
    ray.kill = lambda actor, **kwargs: state["killed"].append(actor)

    def create(options: dict[str, Any], *args: Any) -> Any:
        state["attempts"] += 1
        if construction_failure and state["attempts"] == 2:
            raise probe.PreflightError("injected_construction_failure")
        assert options["num_gpus"] == 0, "file failure must precede every GPU reservation"
        node = options["scheduling_strategy"]["node_id"]
        actor = SimpleNamespace(
            files=SimpleNamespace(
                remote=lambda *args: {"status": "FAILED", "error": "missing_mount", "ray_node_id": node}
            )
        )
        state["created"].append(actor)
        return actor

    factory = SimpleNamespace(options=lambda **options: SimpleNamespace(remote=lambda *args: create(options, *args)))
    ray.remote = lambda **kwargs: lambda cls: factory
    strategies = ModuleType("ray.util.scheduling_strategies")
    strategies.NodeAffinitySchedulingStrategy = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", ModuleType("ray.util"))
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", strategies)
    args = SimpleNamespace(ray_address="explicit:6379", nodes=2, gpus_per_node=1, samples_per_root=1, timeout_s=1)
    report = {}
    code = "injected_construction_failure" if construction_failure else "node_file_checks_failed"
    with pytest.raises(probe.PreflightError, match="^" + code + "$"):
        probe.cluster_checks(args, {}, {"root": [{"path": "fixture"}]}, report)
    assert state["killed"] == state["created"]
    assert state["shutdown"] == 1
    if not construction_failure:
        assert set(report["node_files"]) == {"a", "b"}
        assert all(value["error"] == "missing_mount" for value in report["node_files"].values())
