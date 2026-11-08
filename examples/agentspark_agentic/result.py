# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-closed audit publication, separate from Relax's training output path."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .protocol import AdapterError


SENSITIVE_KEYS = {
    "authorization",
    "proxyauthorization",
    "apikey",
    "accesskey",
    "secretkey",
    "accesstoken",
    "refreshtoken",
    "password",
    "clientsecret",
    "llmconfig",
}
PENDING_GATES = [
    "real_kimi_model_and_template_validation",
    "committed_session_forest_export_and_media_identity",
    "v21_reward_and_token_credit",
    "remote_cleanup_contract_and_capacity_validation",
]


def reject_credentials(value: Any, session_credential: str) -> None:
    """Reject credential-bearing audit inputs instead of silently changing identity.

    Credentials are permitted in transport headers only. Keeping a candidate
    whose content was redacted would no longer describe the actual request.
    """

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = "".join(char for char in str(key).lower() if char.isalnum())
                if normalized in SENSITIVE_KEYS:
                    raise AdapterError("audit_credential_field_forbidden")
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            if session_credential and session_credential in item:
                raise AdapterError("audit_session_credential_forbidden")
            # Chat tool arguments/observations often encode a JSON object inside
            # a string. Check those keys too without changing the recorded text.
            if item.lstrip().startswith(("{", "[")):
                try:
                    nested = json.loads(item)
                except ValueError:
                    return
                if isinstance(nested, (dict, list)):
                    visit(nested)

    visit(value)


def validate_audit_path(audit_path: str | Path, relax_output_path: str | Path) -> Path:
    """Resolve symlinks before comparison and refuse to replace any existing file."""
    path = Path(audit_path).expanduser()
    output = Path(relax_output_path).expanduser()
    if path.resolve() == output.resolve():
        raise AdapterError("audit_must_not_be_relax_output")
    if path.exists() or path.is_symlink():
        raise AdapterError("audit_path_already_exists")
    return path


def write_audit(
    report: dict[str, Any],
    audit_path: str | Path,
    relax_output_path: str | Path,
    session_credential: str,
) -> Path:
    """Create a private new audit file atomically; never publish RELAX_OUTPUT_JSON."""
    path = validate_audit_path(audit_path, relax_output_path)
    reject_credentials(report, session_credential)
    data = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".agentspark-audit-", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # link is atomic and fails if another writer created the target meanwhile.
        validate_audit_path(path, relax_output_path)
        os.link(temporary, path)
    except FileExistsError as exc:
        raise AdapterError("audit_path_already_exists") from exc
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return path
