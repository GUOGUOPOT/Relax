# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline Miles prepared-row conversion; never imports Miles in the training path.

The explicit source checkout supplies the authoritative input compiler. Only its
local-image publisher is replaced, within a single-process context, by a verified
local asset store. No gateway, model, AgentSpark, or reward service is contacted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from urllib.parse import urlsplit

from .protocol import AdapterError, prepare_envelope
from .result import reject_credentials


class PreparationError(AdapterError):
    """A stable code; rejected business content is never interpolated into it."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreparationError(
                "duplicate_json_key_equal"
                if _encoded(result[key]) == _encoded(value)
                else "duplicate_json_key_conflicting"
            )
        result[key] = value
    return result


def strict_json(value: str | bytes) -> Any:
    def nonfinite(_: str) -> None:
        raise PreparationError("nonfinite_json_value")

    try:
        return json.loads(value, object_pairs_hook=_pairs, parse_constant=nonfinite)
    except PreparationError:
        raise
    except (ValueError, UnicodeDecodeError) as exc:
        raise PreparationError("invalid_json") from exc


def _encoded(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError) as exc:
        raise PreparationError("non_json_value") from exc


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreparationError(code)
    return value


@dataclass(frozen=True)
class PreparationConfig:
    miles_source: Path
    system_prompt: str
    user_prompt: str
    tools: list[str]
    tool_versions: dict[str, str]
    image_allowed_hosts: list[str]
    runtime_from_history: bool
    max_images: int
    environment_max_chars: int
    history_max_chars: int
    resource_max_chars: int
    identity: dict[str, Any]

    @classmethod
    def from_source_config(
        cls, source_config: str | Path, *, miles_source: str | Path, profile_dir: str | Path
    ) -> "PreparationConfig":
        """Read only the recipe's prompt/tool/input settings; never copy runtime_env."""
        config_path = Path(source_config).resolve(strict=True)
        raw = config_path.read_bytes()
        config = strict_json(raw)
        if not isinstance(config, dict) or not isinstance(config.get("runtime_env"), dict):
            raise PreparationError("source_config_invalid")
        env = config["runtime_env"]
        agent = config.get("agent")
        if not isinstance(agent, dict):
            raise PreparationError("source_agent_config_invalid")
        profile = Path(profile_dir).resolve(strict=True)
        files = {name: profile / name for name in ("system_prompt.md", "user_prompt.md", "tools.json")}
        for key, filename in (
            ("DOTSBRIDGE_AGENTSPARK_SYSTEM_PROMPT_FILE", "system_prompt.md"),
            ("DOTSBRIDGE_AGENTSPARK_USER_PROMPT_FILE", "user_prompt.md"),
        ):
            declared = Path(_text(env.get(key), "source_prompt_path_missing")).resolve(strict=True)
            if declared != files[filename].resolve(strict=True):
                raise PreparationError("source_profile_path_mismatch")
        blobs = {name: path.read_bytes() for name, path in files.items()}
        tools = strict_json(blobs["tools.json"])
        if strict_json(_text(env.get("DOTSBRIDGE_AGENTSPARK_TOOLS_JSON"), "source_tools_missing")) != tools:
            raise PreparationError("source_tools_mismatch")
        system = _text(blobs["system_prompt.md"].decode(), "source_system_prompt_invalid")
        user = _text(blobs["user_prompt.md"].decode(), "source_user_prompt_invalid")
        if user.count("{{user_query}}") != 1:
            raise PreparationError("source_query_slot_invalid")
        runtime = env.get("DOTSBRIDGE_AGENTSPARK_RUNTIME_CONTEXT_FROM_HISTORY")
        if runtime not in ("0", "1"):
            raise PreparationError("source_runtime_policy_missing")
        # This converter deliberately covers the chosen no-Loader task only.
        if env.get("SLIME_AGENTSPARK_SKILL_LOADER_ENABLE") != "0":
            raise PreparationError("source_skill_loader_not_supported")
        if (
            agent.get("image_transport") != "agentspark_remote_url"
            or agent.get("image_selection_policy") != "pinned_compat"
        ):
            raise PreparationError("source_image_policy_not_supported")
        versions = strict_json(env.get("DOTSBRIDGE_AGENTSPARK_TOOL_VERSIONS_JSON", "{}"))
        hosts = agent.get("image_allowed_hosts")
        if (
            not isinstance(hosts, list)
            or not hosts
            or any(not isinstance(host, str) or not host.strip() for host in hosts)
        ):
            raise PreparationError("source_image_hosts_missing")
        hosts = [host.strip().lower() for host in hosts]
        limits = {}
        for name in ("max_images", "environment_max_chars", "history_max_chars", "resource_max_chars"):
            value = agent.get(name)
            if type(value) is not int or value <= 0:
                raise PreparationError("source_input_limit_invalid")
            limits[name] = value
        probe = {
            "schema_version": 1,
            "inputs": [{"type": "TEXT", "info": {"text": "profile-validation"}}],
            "tools": tools,
            "tool_versions": versions,
            "prompt_spec": {"system_prompt": system, "user_prompt": user},
        }
        prepare_envelope({"metadata": {"agentspark": probe}})
        identity = {
            "source_config_sha256": _digest(raw),
            "profile_files": {name: _digest(blob) for name, blob in blobs.items()},
            "profile_name": env.get("SLIME_AGENTSPARK_PROMPT_PROFILE"),
            "runtime_context_from_history": runtime == "1",
            "skill_loader_enabled": False,
            "limits": limits,
            "image_selection_policy": "pinned_compat",
            "image_allowed_hosts": hosts,
        }
        identity["profile_sha256"] = _digest(_encoded(identity))
        return cls(
            miles_source=Path(miles_source).resolve(strict=True),
            system_prompt=system,
            user_prompt=user,
            tools=tools,
            tool_versions=versions,
            image_allowed_hosts=hosts,
            runtime_from_history=runtime == "1",
            identity=identity,
            **limits,
        )


def _validate_history(history: Any) -> list[dict[str, Any]]:
    if not isinstance(history, list) or any(not isinstance(message, dict) for message in history):
        raise PreparationError("history_not_message_list")
    for message in history:
        role = message.get("role")
        if not isinstance(role, str) or role.strip().lower() not in {"system", "user", "assistant", "tool"}:
            raise PreparationError("history_role_invalid")
        calls = message.get("tool_calls")
        if calls is not None and (not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls)):
            raise PreparationError("history_tool_calls_invalid")
        content = message.get("content")
        empty = content is None or content == "" or content == []
        if isinstance(content, list):
            empty = not any(
                isinstance(part, dict) and (part.get("type") != "text" or bool(part.get("text"))) for part in content
            )
        if (
            empty
            and not (role.strip().lower() == "assistant" and calls)
            and not (role.strip().lower() == "tool" and content == "")
        ):
            raise PreparationError("history_empty_content")
        for call in calls or []:
            function = call.get("function")
            if not isinstance(function, dict):
                raise PreparationError("history_tool_function_invalid")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = strict_json(arguments)
                except PreparationError as exc:
                    raise PreparationError("history_tool_arguments_" + exc.code) from exc
            else:
                parsed = arguments
            if not isinstance(parsed, dict):
                raise PreparationError("history_tool_arguments_not_object")
    return copy.deepcopy(history)


def _attachments(metadata: dict[str, Any], row: dict[str, Any]) -> list[Any]:
    attachments = metadata.get("attachments", [])
    images = metadata.get("image_urls", row.get("image_urls", []))
    if not isinstance(attachments, list) or any(not isinstance(item, dict) for item in attachments):
        raise PreparationError("attachments_invalid")
    if not isinstance(images, list) or any(not isinstance(item, str) or not item for item in images):
        raise PreparationError("current_images_invalid")
    if "image_urls" in metadata and "image_urls" in row and metadata["image_urls"] != row["image_urls"]:
        raise PreparationError("current_image_fields_disagree")
    result = copy.deepcopy(attachments)
    existing = set()
    for item in attachments:
        if str(item.get("type", "")).strip() == "image":
            source = item.get("imageUrl") or item.get("image_url") or item.get("url")
            source = source.get("url") if isinstance(source, dict) else source
            if not isinstance(source, str):
                raise PreparationError("attachment_image_source_invalid")
            existing.add(source)
    result.extend(
        {"type": "image", "source": "upload", "imageUrl": image} for image in images if image not in existing
    )
    return result


def load_verified_media_manifest(path: str | Path | None) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Reuse explicitly supplied byte/hash/decode evidence, never locator hashes."""
    if path is None:
        return None, {"source": "local_read_hash_and_decode", "runtime_revalidation_required": True}
    mime_types = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "BMP": "image/bmp",
    }
    records = []
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for line in stream:
            digest.update(line)
            entry = strict_json(line)
            if not isinstance(entry, dict) or entry.get("status") != "verified":
                raise PreparationError("media_manifest_unverified_entry")
            if not isinstance(entry.get("format"), str) or entry["format"] not in mime_types:
                raise PreparationError("media_manifest_format_unsupported")
            required = ("path", "sha256", "bytes", "width", "height", "mtime_ns")
            if any(key not in entry for key in required):
                raise PreparationError("media_manifest_entry_incomplete")
            if (
                not isinstance(entry["path"], str)
                or not entry["path"].startswith("/")
                or not isinstance(entry["sha256"], str)
                or len(entry["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in entry["sha256"])
                or any(
                    type(entry[key]) is not int or entry[key] <= 0 for key in ("bytes", "width", "height", "mtime_ns")
                )
            ):
                raise PreparationError("media_manifest_entry_invalid")
            records.append(
                {
                    "ref": "agentspark-asset:" + str(entry["sha256"]),
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                    "byte_length": entry["bytes"],
                    "width": entry["width"],
                    "height": entry["height"],
                    "mime_type": mime_types[entry["format"]],
                }
            )
    return records, {
        "source": "verified_read_hash_decode_manifest",
        "sha256": digest.hexdigest(),
        "entries": len(records),
        "runtime_revalidation_required": True,
    }


class MilesPreparedConverter:
    """Single-process offline context, with explicit source and asset authority."""

    def __init__(
        self,
        config: PreparationConfig,
        *,
        allowed_roots: list[str | Path],
        max_asset_bytes: int = 16 * 1024 * 1024,
        verified_media_manifest: str | Path | None = None,
    ):
        self.config = config
        self.allowed_roots = [str(Path(root).resolve(strict=True)) for root in allowed_roots]
        if not self.allowed_roots:
            raise PreparationError("asset_roots_required")
        self.max_asset_bytes = max_asset_bytes
        from .media import AssetStore

        manifest, self.media_evidence = load_verified_media_manifest(verified_media_manifest)
        self._asset_store = AssetStore(self.allowed_roots, max_bytes=max_asset_bytes, verified_manifest=manifest)
        self.source_identity: dict[str, Any] = {}
        self._compiler: Any = None
        self._modules: tuple[Any, Any] | None = None

    @contextmanager
    def source(self) -> Iterator["MilesPreparedConverter"]:
        """Import only the explicitly selected checkout, without changing its files."""
        component_dir = self.config.miles_source / "examples/async_dots_rl/components"
        if not (component_dir / "dotsbridge/adapters/input_compiler.py").is_file():
            raise PreparationError("miles_input_compiler_missing")
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if name == "dotsbridge" or name.startswith("dotsbridge."):
                if filename and not Path(filename).resolve().is_relative_to(component_dir):
                    raise PreparationError("different_miles_source_already_loaded")
        sys.path.insert(0, str(component_dir))
        old_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            compiler = importlib.import_module("dotsbridge.adapters.input_compiler")
            environment = importlib.import_module("dotsbridge.adapters.agentspark_environment_adapter")
            replay = importlib.import_module("dotsbridge.adapters.replay_image_plan")
            files = {}
            for name, module in tuple(sys.modules.items()):
                filename = getattr(module, "__file__", None)
                if filename and (name == "dotsbridge" or name.startswith("dotsbridge.")):
                    path = Path(filename).resolve()
                    if path.is_relative_to(self.config.miles_source) and path.suffix == ".py":
                        files[str(path.relative_to(self.config.miles_source))] = _digest(path.read_bytes())
            converter_files = {
                name: _digest(Path(__file__).with_name(name).read_bytes()) for name in ("prepare.py", "media.py")
            }
            self.source_identity = {
                "files": files,
                "converter_files": converter_files,
                "sha256": _digest(_encoded({"miles_files": files, "converter_files": converter_files})),
            }
            self._compiler, self._modules = compiler.compile_agent_input, (environment, replay)
            yield self
        finally:
            self._compiler = None
            self._modules = None
            sys.dont_write_bytecode = old_bytecode
            sys.path.remove(str(component_dir))

    def convert(self, row: Any, *, source_line: int, source_sha256: str | None = None) -> dict[str, Any]:
        if self._compiler is None or self._modules is None:
            raise PreparationError("source_context_required")
        if type(source_line) is not int or source_line <= 0:
            raise PreparationError("source_line_invalid")
        if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
            raise PreparationError("prepared_row_invalid")
        reject_credentials(row, "")
        metadata = row["metadata"]
        if any(key in metadata for key in ("agentspark", "agentspark_media", "agentspark_prepare")):
            raise PreparationError("already_prepared_or_reserved_metadata")
        query = _text(metadata.get("query", row.get("query")), "current_query_missing")
        for name in ("query", "raw_query"):
            if name in metadata and metadata[name] != query:
                raise PreparationError("current_query_fields_disagree")
        if row.get("query") != query:
            raise PreparationError("current_query_fields_disagree")
        _text(metadata.get("query_id"), "query_id_missing")
        user_id = _text(metadata.get("user_id"), "user_id_missing")
        _text(metadata.get("expression_tag"), "expression_tag_missing")
        rubric = metadata.get("rubric")
        if not isinstance(rubric, (dict, str)) or not rubric:
            raise PreparationError("rubric_missing")
        if isinstance(rubric, str):
            strict_json(rubric)
        selected = "replay_history" if isinstance(metadata.get("replay_history"), list) else "history"
        history = _validate_history(metadata.get(selected, []))
        source_hash = source_sha256 or _digest(_encoded(row))
        namespace = _digest((source_hash + self.config.identity["profile_sha256"]).encode())[:32]
        assets = self._asset_store.new_scope()
        csv_row = {
            key: copy.deepcopy(metadata[key])
            for key in (
                "expression_tag",
                "rubric",
                "query",
                "query_id",
                "current_runtime_context",
                "current_attachment_snapshot",
                "current_user_profile",
            )
            if key in metadata
        }
        values = {
            "query": query,
            "history": history,
            "resource_registry": copy.deepcopy(metadata.get("resource_registry")),
            "csv_row": csv_row,
            "attachments": _attachments(metadata, row),
        }
        # Only explicit model-visible occurrences require frozen pixels here.
        # Registry URLs (including note covers/avatars) are references restored
        # by Replay, not instructions to download every image in a resource.
        # The authoritative compiler validates/preserves those remote URLs using
        # the selected recipe's allowlist; no network fetch occurs.
        placeholders = importlib.import_module("dotsbridge.adapters.agentspark_image_placeholders")
        visual_sources = [placeholders.attachment_image_source(item) for item in values["attachments"]]
        contents = [message.get("content") for message in history]
        snapshot = metadata.get("current_attachment_snapshot")
        if isinstance(snapshot, dict):
            contents.append(snapshot.get("parts"))
        for content in contents:
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"image", "image_url"}:
                    source = part.get("image_url") or part.get("image") or part.get("url")
                    visual_sources.append(source.get("url") if isinstance(source, dict) else source)
        for source in visual_sources:
            if isinstance(source, str) and urlsplit(source).scheme in {"http", "https"}:
                raise PreparationError("unresolved_remote_media")
        settings = SimpleNamespace(
            agentspark_runtime_context_from_history=self.config.runtime_from_history,
            agentspark_image_selection_policy="pinned_compat",
            agentspark_environment_max_chars=self.config.environment_max_chars,
            agentspark_history_max_chars=self.config.history_max_chars,
            agentspark_resource_max_chars=self.config.resource_max_chars,
            agentspark_max_images=self.config.max_images,
            agentspark_image_transport="agentspark_remote_url",
            # These sentinels satisfy source checks only. The scoped publisher
            # below consumes all local paths and never creates a gateway URL.
            agentspark_image_allowed_hosts=self.config.image_allowed_hosts,
            image_gateway_public_base_url="offline:asset-store",
            image_gateway_registry_dir="offline:asset-store",
            image_gateway_allowed_roots=self.allowed_roots,
            image_gateway_max_bytes=self.max_asset_bytes,
            agentspark_user_prompt=self.config.user_prompt,
            agentspark_system_prompt=self.config.system_prompt,
        )
        originals = tuple(module.publish_local_image for module in self._modules)

        def publisher(source: str, **_: Any) -> str:
            return assets.freeze(source)

        try:
            for module in self._modules:
                module.publish_local_image = publisher
            compiled = self._compiler(values, settings, image_marker_namespace=namespace)
        except AdapterError:
            raise
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise PreparationError("miles_source_compilation_failed") from exc
        finally:
            for module, original in zip(self._modules, originals, strict=True):
                module.publish_local_image = original
        environment = compiled["environment"]
        envelope: dict[str, Any] = {
            "schema_version": 1,
            "inputs": environment.inputs,
            "tools": copy.deepcopy(self.config.tools),
            "tool_versions": copy.deepcopy(self.config.tool_versions),
            "bound_user_id": user_id,
            "prompt_spec": {"system_prompt": self.config.system_prompt, **environment.prompt_spec},
        }
        if compiled["replay_messages"]:
            envelope["replay_spec"] = {
                "messages": compiled["replay_messages"],
                "attributes": compiled["replay_attributes"],
            }
        output = copy.deepcopy(row)
        output["messages"] = [{"role": "user", "content": query}]
        output["metadata"]["agentspark"] = envelope
        output["metadata"]["agentspark_media"] = assets.build(
            compiled["replay_image_plan"], environment.provider_seed_image_sources
        )
        output["metadata"]["agentspark_prepare"] = {
            "schema_version": 1,
            "source_line": source_line,
            "source_row_sha256": source_hash,
            "profile_sha256": self.config.identity["profile_sha256"],
            "compiler_sha256": self.source_identity["sha256"],
            "selected_history": selected,
            "normalizations": (
                ["miles_replay_roles_content_and_positions", "selected_system_prompt"] if history else []
            ),
            "resource_id_aliases": compiled["resource_id_aliases"],
            "compiled_query": compiled["values"]["query"],
            "runtime_snapshot": compiled["runtime_snapshot"],
            "input_stats": environment.stats,
            "remote_registry_urls_are_references_not_verified_pixels": True,
            "media_evidence": self.media_evidence,
            "training_ready": False,
        }
        prepare_envelope(output)
        reject_credentials(output, "")
        return output


def prepare_jsonl(
    source: str | Path,
    output: str | Path,
    *,
    converter: MilesPreparedConverter,
    rejection_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Create distinct private artifacts. Rejections are explicit and never repaired silently."""
    source_path = Path(source).resolve(strict=True)
    destinations = [Path(path).resolve() for path in (output, rejection_path, report_path)]
    if source_path in destinations or len(set(destinations)) != len(destinations):
        raise PreparationError("preparation_paths_overlap")
    if any(path.exists() for path in destinations):
        raise PreparationError("preparation_output_exists")
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    total = accepted = 0
    failures: dict[str, int] = {}
    digest = hashlib.sha256()
    with converter.source(), source_path.open("rb") as input_stream:
        handles = []
        try:
            for path in destinations[:2]:
                handles.append(os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"))
            prepared_stream, rejected_stream = handles
            for total, line in enumerate(input_stream, 1):
                digest.update(line)
                row_hash = _digest(line)
                row = None
                try:
                    row = strict_json(line)
                    converted = converter.convert(row, source_line=total, source_sha256=row_hash)
                except AdapterError as exc:
                    failures[exc.code] = failures.get(exc.code, 0) + 1
                    query_id = (
                        (row.get("metadata") or {}).get("query_id")
                        if isinstance(row, dict) and isinstance(row.get("metadata"), dict)
                        else None
                    )
                    rejected_stream.write(
                        json.dumps(
                            {
                                "source_line": total,
                                "source_row_sha256": row_hash,
                                "query_id_sha256": _digest(query_id.encode()) if isinstance(query_id, str) else None,
                                "code": exc.code,
                                "action": "rejected_without_repair",
                            }
                        )
                        + "\n"
                    )
                    continue
                prepared_stream.write(json.dumps(converted, ensure_ascii=False, allow_nan=False) + "\n")
                accepted += 1
        finally:
            for handle in handles:
                handle.close()
    report = {
        "schema_version": 1,
        "source_sha256": digest.hexdigest(),
        "rows": total,
        "accepted": accepted,
        "rejected": total - accepted,
        "rejection_codes": failures,
        "profile": converter.config.identity,
        "compiler": converter.source_identity,
        "media_evidence": converter.media_evidence,
        "training_ready": False,
        "remote_services_contacted": False,
    }
    with os.fdopen(os.open(destinations[2], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "source-config", "miles-source", "profile-dir", "rejections", "report"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--allowed-root", action="append", required=True)
    parser.add_argument("--verified-media-manifest")
    args = parser.parse_args()
    try:
        config = PreparationConfig.from_source_config(
            args.source_config, miles_source=args.miles_source, profile_dir=args.profile_dir
        )
        report = prepare_jsonl(
            args.input,
            args.output,
            converter=MilesPreparedConverter(
                config, allowed_roots=args.allowed_root, verified_media_manifest=args.verified_media_manifest
            ),
            rejection_path=args.rejections,
            report_path=args.report,
        )
    except (AdapterError, OSError) as exc:
        code = exc.code if isinstance(exc, AdapterError) else "preparation_io_failed"
        sys.stderr.write(json.dumps({"error": code}) + "\n")
        return 2
    sys.stdout.write(
        json.dumps({key: report[key] for key in ("rows", "accepted", "rejected", "training_ready")}) + "\n"
    )
    return 3 if report["rejected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
