# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline preparation and exact-input checks for the AgentSpark V21 recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from relax.utils.logging_utils import get_logger

from .protocol import AdapterConfig, AdapterError
from .v21_reward import RewardError, load_reward_contract


logger = get_logger(__name__)


class TrainingInputError(ValueError):
    """A fixed diagnostic code that contains no input data or credentials."""


def write_new_json(path: str | Path, value: Any) -> None:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(raw)


def validate_data(path: str | Path, *, expected_sha256: str, expected_rows: int) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or type(expected_rows) is not int or expected_rows <= 0:
        raise TrainingInputError("training_data_expectation_invalid")
    digest, rows = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for line in stream:
            digest.update(line)
            item = json.loads(line)
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("messages"), list)
                or not item["messages"]
                or not isinstance(item.get("metadata"), dict)
                or not isinstance(item["metadata"].get("agentspark"), dict)
                or not isinstance(item["metadata"].get("agentspark_prepare"), dict)
            ):
                raise TrainingInputError("training_data_not_prepared")
            rows += 1
    if rows != expected_rows or digest.hexdigest() != expected_sha256:
        raise TrainingInputError("training_data_identity_mismatch")
    return {"rows": rows, "sha256": digest.hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("configure")
    prepare.add_argument("--contract", required=True)
    prepare.add_argument("--overrides-output", required=True)
    prepare.add_argument("--compiler-defaults-output", required=True)
    check = modes.add_parser("check")
    for name in (
        "data",
        "expected-sha256",
        "contract",
        "overrides",
        "compiler-defaults",
        "agent-config",
        "callback-host-map",
    ):
        check.add_argument("--" + name, required=True)
    check.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args(argv)
    stage = "contract"
    try:
        from .v21_training import CreditError, training_overrides

        contract = load_reward_contract(args.contract)
        expected = training_overrides(contract)
        if args.mode == "configure":
            stage = "configuration_outputs"
            # JSON is also valid YAML, accepted by native --custom-config-path.
            # Refuse both existing destinations before writing either file.
            destinations = [Path(args.overrides_output), Path(args.compiler_defaults_output)]
            if destinations[0].absolute() == destinations[1].absolute() or any(
                p.exists() or p.is_symlink() for p in destinations
            ):
                raise TrainingInputError("training_configuration_output_exists")
            write_new_json(destinations[0], expected)
            write_new_json(destinations[1], {"thinking": True})
            logger.info("Created frozen V21 training overrides and K3 compiler defaults")
            return 0
        import yaml

        stage = "overrides"
        try:
            supplied = yaml.safe_load(Path(args.overrides).read_text())
        except yaml.YAMLError as exc:
            raise TrainingInputError("training_overrides_invalid_yaml") from exc
        if supplied != expected:
            raise TrainingInputError("training_overrides_not_frozen_contract")
        stage = "compiler_defaults"
        if json.loads(Path(args.compiler_defaults).read_text()) != {"thinking": True}:
            raise TrainingInputError("training_compiler_defaults_mismatch")
        stage = "agent_config"
        config = AdapterConfig.from_payload(json.loads(Path(args.agent_config).read_text()))
        if config.max_tokens != 8192:
            raise TrainingInputError("training_max_tokens_not_frozen_contract")
        stage = "callback_host_map"
        hosts = json.loads(Path(args.callback_host_map).read_text())
        if (
            not isinstance(hosts, dict)
            or not hosts
            or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in hosts.items())
        ):
            raise TrainingInputError("training_callback_host_map_invalid")
        stage = "agent_config"
        if config.callback_bind_port != 0:
            raise TrainingInputError("training_concurrent_callbacks_require_ephemeral_port")
        from urllib.parse import urlsplit

        if urlsplit(config.callback_public_base_url).port != 0:
            raise TrainingInputError("training_callback_public_url_requires_port_zero")
        stage = "data"
        identity = validate_data(args.data, expected_sha256=args.expected_sha256, expected_rows=args.expected_rows)
        logger.info("Offline training input validation passed: %s", identity)
        return 0
    except (OSError, ValueError, KeyError, TypeError, AdapterError) as exc:
        # Only our fixed-code errors are safe to show; parser/OS errors can
        # include data, URLs, credentials or private paths in their messages.
        code = (
            str(exc)
            if isinstance(exc, (TrainingInputError, AdapterError, RewardError, CreditError))
            else "invalid_input"
        )
        logger.error("Training input validation failed at %s: %s (%s)", stage, code, type(exc).__name__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
