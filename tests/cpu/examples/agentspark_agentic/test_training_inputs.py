"""Prevent full-data recipe drift, accidental dry-run submission and wrong
inputs."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from examples.agentspark_agentic import training_inputs, v21_training
from examples.agentspark_agentic.training_inputs import validate_data, write_new_json


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts/training/agentic/run-kimi-k3-agentspark-v21-128b300.sh"


def _environment(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "MODEL_DIR": str(tmp_path / "model with spaces"),
            "PROMPT_SET": str(tmp_path / "prepared.jsonl"),
            "AGENTSPARK_CONFIG": str(tmp_path / "agent.json"),
            "AGENTSPARK_CALLBACK_HOST_MAP": str(tmp_path / "hosts.json"),
            "AGENTSPARK_V21_CONTRACT": str(tmp_path / "reward.json"),
            "AGENTSPARK_REWARD_URL": "http://reward.invalid:8080",
            "V21_TRAINING_OVERRIDES": str(tmp_path / "training.json"),
            "K3_COMPILER_DEFAULTS_JSON": str(tmp_path / "compiler.json"),
            "AGENTIC_CONCURRENCY": "2",
            "EXPECTED_PREPARED_SHA256": "a" * 64,
            "SAVE_DIR": str(tmp_path / "checkpoints"),
            "ACCEPTED_ROWS": "21258",
            "ROLLOUT_BATCH_SIZE": "18",
            "EPOCHS": "1",
        }
    )
    # Any accidental execution is observable; DRY_RUN must not contact Ray or
    # execute Python input checking, even though the paths do not exist.
    for name in ("python", "python3", "ray"):
        fake = tmp_path / name
        fake.write_text("#!/bin/sh\necho dry_run_executed_external_tool >&2\nexit 99\n")
        fake.chmod(0o700)
    env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
    return env


def test_128b300_recipe_dry_run_preserves_full_data_and_logical_objective(tmp_path: Path) -> None:
    env = _environment(tmp_path)
    result = subprocess.run(["/bin/bash", str(SCRIPT)], env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    assert argv[:3] == ["python3", "-m", "relax.entrypoints.train"]
    for key, value in {
        "--hf-checkpoint": env["MODEL_DIR"],
        "--ref-load": env["MODEL_DIR"],
        "--num-rollout": "1181",
        "--global-batch-size": "72",
        "--n-samples-per-prompt": "4",
        "--num-gpus-per-node": "8",
        "--rollout-num-gpus-per-engine": "16",
        "--loss-type": "custom_loss",
        "--rollout-temperature": "0.8",
        "--kl-loss-coef": "0.003",
        "--kl-loss-type": "low_var_kl",
    }.items():
        assert argv[argv.index(key) + 1] == value
    assert json.loads(argv[argv.index("--resource") + 1]) == {"actor": [1, 128], "rollout": [1, 128]}
    assert "--calculate-per-token-loss" not in argv
    assert "--no-save-optim" not in argv
    assert "--use-kl-loss" in argv
    assert "--disable-weights-backuper" not in argv
    cp_size = int(argv[argv.index("--context-parallel-size") + 1])
    max_context = int(argv[argv.index("--rollout-max-context-len") + 1])
    for flag in ("--max-tokens-per-gpu", "--log-probs-max-tokens-per-gpu"):
        assert int(argv[argv.index(flag) + 1]) * cp_size >= max_context
    assert "--agentic-custom-advantage-path" not in argv
    assert "--custom-rm-path" not in argv
    command = shlex.split(argv[argv.index("--agent-command") + 1])
    assert command[:3] == ["python3", "-m", "examples.agentspark_agentic.production"]
    assert "--callback-host-map" in command
    assert not (tmp_path / "checkpoints").exists()


def test_full_data_recipe_rejects_tail_wrap_before_any_submission(tmp_path: Path) -> None:
    env = _environment(tmp_path)
    env["ROLLOUT_BATCH_SIZE"] = "64"
    result = subprocess.run(["/bin/bash", str(SCRIPT)], env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 2
    assert "tail wrapping" in result.stderr
    assert "dry_run_executed_external_tool" not in result.stderr


def test_prepared_input_identity_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    item = {
        "messages": [{"role": "user", "content": "query"}],
        "metadata": {"agentspark": {}, "agentspark_prepare": {}},
    }
    raw = (json.dumps(item) + "\n").encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert validate_data(path, expected_sha256=digest, expected_rows=1) == {"rows": 1, "sha256": digest}
    for expected_digest, rows in (("b" * 64, 1), (digest, 2)):
        with pytest.raises(ValueError, match="identity_mismatch"):
            validate_data(path, expected_sha256=expected_digest, expected_rows=rows)
    with pytest.raises(FileExistsError):
        write_new_json(path, {})
    assert path.read_bytes() == raw
    path.write_text(json.dumps({"query": "raw unprepared"}) + "\n")
    with pytest.raises(ValueError, match="not_prepared"):
        validate_data(path, expected_sha256=digest, expected_rows=1)


@pytest.fixture
def check_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    # The frozen Miles source is external; exercise the real file/schema checks
    # using a fixed expected contract without requiring that checkout in CI.
    monkeypatch.setattr(training_inputs, "load_reward_contract", lambda path: {})
    monkeypatch.setattr(v21_training, "training_overrides", lambda contract: {"frozen": True})
    values = {
        "overrides": {"frozen": True},
        "compiler-defaults": {"thinking": True},
        "agent-config": {
            "base_url": "http://agent.invalid",
            "model": "Kimi-K3",
            "agent": "diandian",
            "source": "eden",
            "max_tokens": 8192,
            "callback_bind_host": "0.0.0.0",
            "callback_bind_port": 0,
            "callback_public_base_url": "http://callback.invalid:0",
        },
        "callback-host-map": {"worker": "callback.invalid"},
        "data": {
            "messages": [{"role": "user", "content": "query"}],
            "metadata": {"agentspark": {}, "agentspark_prepare": {}},
        },
    }
    argv = ["check", "--contract", str(tmp_path / "contract.json"), "--expected-rows", "1"]
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        write_new_json(path, value)
        argv.extend(["--" + name, str(path)])
    data = tmp_path / "data.json"
    data.write_text(json.dumps(values["data"]) + "\n")
    argv.extend(["--expected-sha256", hashlib.sha256(data.read_bytes()).hexdigest()])
    # Initialize logging before pytest installs the per-test capture handler.
    training_inputs.logger.info("Training input diagnostic test setup")
    return argv


def test_training_inputs_check_reports_unprepared_source(
    check_args: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    assert training_inputs.main(check_args) == 0
    data = Path(check_args[check_args.index("--data") + 1])
    data.write_text(json.dumps({"query": "private-source-text", "metadata": {}}) + "\n")
    assert training_inputs.main(check_args) == 2
    assert "failed at data: training_data_not_prepared" in caplog.text
    assert "private-source-text" not in caplog.text


@pytest.mark.parametrize(
    ("option", "content", "diagnostic"),
    [
        ("overrides", "secret: [private-input\n", "overrides: training_overrides_invalid_yaml"),
        ("agent-config", '{"secret": "private-input"}', "agent_config: config_fields_invalid"),
        ("compiler-defaults", '{"private-input":', "compiler_defaults: invalid_input (JSONDecodeError)"),
    ],
)
def test_training_inputs_check_redacts_invalid_inputs(
    check_args: list[str], option: str, content: str, diagnostic: str, caplog: pytest.LogCaptureFixture
) -> None:
    Path(check_args[check_args.index("--" + option) + 1]).write_text(content)
    assert training_inputs.main(check_args) == 2
    assert diagnostic in caplog.text
    assert "private-input" not in caplog.text
    assert "Traceback" not in caplog.text
