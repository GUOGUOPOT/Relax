# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Private, byte-pinned local images and request-scoped Replay restoration.

No Miles runtime import, filesystem authority from callback content, remote fetch,
training output, or inferred image gateway is involved.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import json
import re
import secrets
from pathlib import Path
from typing import Any

from .protocol import AdapterError, json_copy


ASSET_PREFIX = "agentspark-asset:"
MEDIA_PATH = "/agentspark-media/"
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGES = 30
MARKER = re.compile(r"\[\[MILES_REPLAY_IMAGE:([a-f0-9]{32}):([0-9]{4})\]\]")
_DYNAMIC_IMAGE = re.compile(r"^data:image/(jpeg|png|webp|gif|bmp);base64,([A-Za-z0-9+/]*={0,2})$", re.IGNORECASE)
_IMAGE_OMISSION = "[图片因视觉上限省略]"


def _dynamic_image(source: Any) -> tuple[str, dict[str, Any]]:
    """Validate bounded inline pixels without fetching URLs or changing image bytes."""
    from PIL import Image

    if not isinstance(source, str) or not source.startswith("data:image/"):
        raise AdapterError("media_dynamic_image_scheme_forbidden")
    match = _DYNAMIC_IMAGE.fullmatch(source)
    if match is None:
        raise AdapterError("media_dynamic_image_data_invalid")
    encoded = match[2]
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise AdapterError("media_dynamic_image_too_large")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if not 0 < len(raw) <= MAX_IMAGE_BYTES:
            raise AdapterError("media_dynamic_image_too_large")
        with Image.open(io.BytesIO(raw)) as image:
            mime, size = Image.MIME.get(image.format), image.size
            image.verify()
        if mime != "image/" + match[1].lower():
            raise AdapterError("media_dynamic_image_mime_mismatch")
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
    except AdapterError:
        raise
    except (binascii.Error, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise AdapterError("media_dynamic_image_data_invalid") from exc
    return "data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii"), {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_length": len(raw),
        "mime_type": mime,
        "width": size[0],
        "height": size[1],
    }


def _image_value(part: dict[str, Any]) -> str | None:
    """Validate the image shapes accepted by the frozen harness before K3 sees them."""
    kind = part.get("type")
    if kind not in ("image_url", "image"):
        if any(key in part for key in ("image_url", "image", "video", "video_url")) or kind in ("video", "video_url"):
            raise AdapterError("media_dynamic_image_block_invalid")
        return None
    if set(part) != {"type", kind}:
        raise AdapterError("media_dynamic_image_block_invalid")
    value = part[kind]
    if isinstance(value, dict):
        if set(value) not in ({"url"}, {"url", "detail"}) or value.get("detail", "auto") not in ("auto", "low", "high"):
            raise AdapterError("media_dynamic_image_block_invalid")
        value = value.get("url")
    if not isinstance(value, str) or not value.strip():
        raise AdapterError("media_dynamic_image_block_invalid")
    return value.strip()


def _visual_signature(messages: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    return [
        (i, j, value)
        for i, message in enumerate(messages)
        if isinstance(message.get("content"), list)
        for j, part in enumerate(message["content"])
        if (value := _image_value(part)) is not None
    ]


def _seed_nonvisual(message: dict[str, Any]) -> str:
    copied = json_copy(message)
    copied["content"] = [
        part for part in _content_parts(copied.get("content"))
        if _image_value(part) is None and part.get("text") not in ("[图片]", _IMAGE_OMISSION)
    ]
    return _digest(copied)


def strict_json_loads(value: str | bytes) -> Any:
    def object_pairs(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = item
        return result

    def invalid_constant(_value):
        raise ValueError("non-finite JSON number")

    return json.loads(value, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _roots(values: Any) -> tuple[Path, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise AdapterError("media_allowed_roots_required")
    try:
        roots = tuple(Path(value).resolve(strict=True) for value in values)
        if any(not root.is_dir() for root in roots):
            raise ValueError
        return roots
    except (OSError, TypeError, ValueError) as exc:
        raise AdapterError("media_allowed_roots_invalid") from exc


def _read_asset(source: str, roots: tuple[Path, ...], max_bytes: int) -> tuple[dict[str, Any], bytes]:
    from PIL import Image

    if not isinstance(source, str) or not source.startswith("/"):
        raise AdapterError("media_local_source_required")
    try:
        path = Path(source).resolve(strict=True)
        if not any(path.is_relative_to(root) for root in roots):
            raise AdapterError("media_source_outside_roots")
        if not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
            raise AdapterError("media_source_size_invalid")
        with path.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
        if not 0 < len(raw) <= max_bytes:
            raise AdapterError("media_source_size_invalid")
        with Image.open(io.BytesIO(raw)) as image:
            mime = Image.MIME.get(image.format)
            width, height = image.size
            image.verify()
        if mime not in {"image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"}:
            raise AdapterError("media_image_format_unsupported")
        # Decode as well as checking headers; keep original bytes unchanged.
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
        digest = hashlib.sha256(raw).hexdigest()
        return {
            "ref": ASSET_PREFIX + digest,
            "path": str(path),
            "sha256": digest,
            "mime_type": mime,
            "byte_length": len(raw),
            "width": width,
            "height": height,
        }, raw
    except AdapterError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise AdapterError("media_source_unreadable_or_invalid") from exc


class AssetStore:
    """Offline compiler helper. Pin source identity; runtime rechecks then freezes bytes."""

    def __init__(
        self,
        allowed_roots: list[str | Path],
        max_bytes: int = MAX_IMAGE_BYTES,
        *,
        verified_manifest: list[dict[str, Any]] | None = None,
    ) -> None:
        self.roots = _roots(allowed_roots)
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_IMAGE_BYTES:
            raise AdapterError("media_max_bytes_invalid")
        self.max_bytes = max_bytes
        self._assets: dict[str, dict[str, Any]] = {}
        self._catalog: dict[str, dict[str, Any]] | None = None
        if verified_manifest is not None:
            # Explicit offline evidence reuse. Runtime still reads/decodes and
            # hashes each used image; this catalog never grants filesystem access.
            if not isinstance(verified_manifest, list):
                raise AdapterError("media_manifest_invalid")
            for asset in verified_manifest:
                _validate_plan({"schema_version": 1, "assets": [asset], "replay_bindings": [], "seed_refs": []})
            self._catalog = {asset["path"]: json_copy(asset) for asset in verified_manifest}
            if len(self._catalog) != len(verified_manifest):
                raise AdapterError("media_manifest_path_duplicate")

    def new_scope(self) -> "AssetStore":
        """Share a validated read-only catalog; collect only this row's used assets."""
        scoped = object.__new__(AssetStore)
        scoped.roots, scoped.max_bytes = self.roots, self.max_bytes
        scoped._catalog = self._catalog
        scoped._assets = {}
        return scoped

    def freeze(self, source: str) -> str:
        if self._catalog is None:
            asset, _ = _read_asset(source, self.roots, self.max_bytes)
        else:
            if not isinstance(source, str) or not source.startswith("/"):
                raise AdapterError("media_local_source_required")
            # Catalog paths are absolute, canonical paths produced by the
            # verifier. Reject traversal/symlink aliases; do not silently scan.
            path = Path(source)
            if str(path) != source or ".." in path.parts or not any(path.is_relative_to(root) for root in self.roots):
                raise AdapterError("media_source_outside_roots")
            if source not in self._catalog:
                raise AdapterError("media_manifest_source_missing")
            asset = self._catalog[source]
            if asset["byte_length"] > self.max_bytes:
                raise AdapterError("media_source_size_invalid")
        self._assets.setdefault(asset["ref"], asset)
        return asset["ref"]

    def manifest(self) -> list[dict[str, Any]]:
        return json_copy(list(self._assets.values()))

    def build(self, replay_bindings: list[dict[str, Any]], seed_refs: list[str]) -> dict[str, Any]:
        plan = {
            "schema_version": 1,
            "assets": self.manifest(),
            "replay_bindings": json_copy(replay_bindings),
            "seed_refs": json_copy(seed_refs),
        }
        _validate_plan(plan)
        return plan


def _validate_plan(plan: Any) -> None:
    if (
        not isinstance(plan, dict)
        or set(plan) != {"schema_version", "assets", "replay_bindings", "seed_refs"}
        or type(plan.get("schema_version")) is not int
        or plan["schema_version"] != 1
    ):
        raise AdapterError("media_plan_invalid")
    assets, bindings, seeds = plan["assets"], plan["replay_bindings"], plan["seed_refs"]
    if not all(isinstance(value, list) for value in (assets, bindings, seeds)):
        raise AdapterError("media_plan_invalid")
    if len(bindings) + len(seeds) > MAX_IMAGES:
        raise AdapterError("media_image_limit_exceeded")
    refs = set()
    for asset in assets:
        if not isinstance(asset, dict) or set(asset) != {
            "ref",
            "path",
            "sha256",
            "mime_type",
            "byte_length",
            "width",
            "height",
        }:
            raise AdapterError("media_asset_invalid")
        path, mime = asset["path"], asset["mime_type"]
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or not isinstance(mime, str)
            or mime not in {"image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"}
        ):
            raise AdapterError("media_asset_invalid")
        digest = asset["sha256"]
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            or asset["ref"] != ASSET_PREFIX + digest
            or asset["ref"] in refs
        ):
            raise AdapterError("media_asset_invalid")
        if any(type(asset[key]) is not int or asset[key] <= 0 for key in ("byte_length", "width", "height")):
            raise AdapterError("media_asset_invalid")
        if asset["byte_length"] > MAX_IMAGE_BYTES:
            raise AdapterError("media_asset_invalid")
        refs.add(asset["ref"])
    if any(not isinstance(ref, str) or ref not in refs for ref in seeds):
        raise AdapterError("media_seed_ref_unknown")
    namespace, histories = None, {}
    for ordinal, binding in enumerate(bindings):
        if (
            not isinstance(binding, dict)
            or not {"marker", "source", "role"} <= set(binding)
            or set(binding) - {"marker", "source", "role", "tool_history"}
        ):
            raise AdapterError("media_binding_invalid")
        marker = binding["marker"]
        match = MARKER.fullmatch(marker) if isinstance(marker, str) else None
        if match is None or int(match[2]) != ordinal or (namespace is not None and match[1] != namespace):
            raise AdapterError("media_marker_order_invalid")
        namespace = match[1]
        if (
            not isinstance(binding["role"], str)
            or binding["role"] not in {"user", "tool"}
            or not isinstance(binding["source"], str)
            or binding["source"] not in refs
        ):
            raise AdapterError("media_binding_invalid")
        history = binding.get("tool_history")
        if history is None:
            if binding["role"] == "tool":
                raise AdapterError("media_tool_history_required")
            continue
        if (
            binding["role"] != "tool"
            or not isinstance(history, dict)
            or set(history) != {"tool_call_id", "name", "arguments_sha256", "content"}
        ):
            raise AdapterError("media_tool_history_invalid")
        call_id, name, digest, content = (
            history[key] for key in ("tool_call_id", "name", "arguments_sha256", "content")
        )
        if (
            not isinstance(call_id, str)
            or not 0 < len(call_id) <= 256
            or not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9_]{1,128}", name) is None
            or not isinstance(digest, str)
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            or not isinstance(content, str)
            or len(content.encode()) > 512 * 1024
            or content.count(marker) != 1
        ):
            raise AdapterError("media_tool_history_invalid")
        if call_id in histories and histories[call_id] != history:
            raise AdapterError("media_tool_history_conflict")
        histories[call_id] = history
    if sum(len(history["content"].encode()) for history in histories.values()) > 1024 * 1024:
        raise AdapterError("media_tool_history_too_large")


def _argument_digest(value: Any) -> str:
    try:
        value = strict_json_loads(value) if isinstance(value, str) else value
        if not isinstance(value, dict):
            raise ValueError
        return _digest(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError("media_tool_identity_mismatch") from exc


def _check_tool_call(messages: list[dict[str, Any]], index: int, history: dict[str, Any]) -> None:
    message = messages[index]
    calls = [
        call
        for previous in messages[:index]
        if previous.get("role") == "assistant"
        for call in previous.get("tool_calls") or []
        if isinstance(call, dict) and call.get("id") == history["tool_call_id"]
    ]
    if len(calls) != 1 or message.get("name", history["name"]) != history["name"]:
        raise AdapterError("media_tool_identity_mismatch")
    function = calls[0].get("function")
    if (
        not isinstance(function, dict)
        or function.get("name") != history["name"]
        or _argument_digest(function.get("arguments")) != history["arguments_sha256"]
    ):
        raise AdapterError("media_tool_identity_mismatch")


def _content_parts(content: Any) -> list[dict[str, Any]]:
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if not isinstance(parts, list) or any(not isinstance(part, dict) for part in parts):
        raise AdapterError("media_content_invalid")
    return parts


def _message_anchor(message: dict[str, Any], markers: set[str]) -> str:
    """Permit only occurrence edits, retaining every non-image message field and text."""
    copy = json_copy(message)
    parts, text = [], ""
    for part in _content_parts(copy.get("content")):
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            chunk = part["text"]
            for marker in markers:
                chunk = chunk.replace(marker, "")
            text += chunk
        else:
            if text:
                parts.append({"type": "text", "text": text})
                text = ""
            parts.append(part)
    if text:
        parts.append({"type": "text", "text": text})
    copy["content"] = parts
    return _digest(copy)


class MediaPlan:
    """Per-invocation immutable bytes plus first-callback message authorization."""

    def __init__(
        self,
        plan: dict[str, Any],
        allowed_roots: list[str | Path] | tuple[str, ...],
        *,
        allow_dynamic_images: bool = False,
    ) -> None:
        if type(allow_dynamic_images) is not bool:
            raise AdapterError("media_dynamic_image_option_invalid")
        _validate_plan(plan)
        roots = _roots(allowed_roots) if plan["assets"] else ()
        self.plan = json_copy(plan)
        self.assets: dict[str, tuple[dict[str, Any], bytes]] = {}
        for expected in plan["assets"]:
            actual, raw = _read_asset(expected["path"], roots, MAX_IMAGE_BYTES)
            if actual != expected:
                raise AdapterError("media_asset_identity_changed")
            self.assets[actual["ref"]] = (actual, raw)
        self.bindings = {binding["marker"]: binding for binding in plan["replay_bindings"]}
        self.scope = secrets.token_hex(16)
        self.public_base_url: str | None = None
        self.anchors: dict[str, str] = {}
        self.pinned_messages: dict[str, dict[str, Any]] = {}
        self.initialized = False
        self.allow_dynamic_images = allow_dynamic_images
        self._dynamic_assets: dict[str, dict[str, Any]] = {}
        self._admitted_dynamic_images: list[str] = []
        self._seed_message_index: int | None = None
        self._seed_message_identity: str | None = None
        self._committed_messages: list[dict[str, Any]] = []
        self._prepared_signature: list[tuple[int, int, str]] | None = None
        self._revision = 0
        self._parent: MediaPlan | None = None
        self._reset_segment = False

    def fork(self, *, reset_segment: bool = False) -> "MediaPlan":
        """Stage one request; a failed provider response cannot admit images or pin history."""
        candidate = copy.copy(self)
        candidate.anchors = dict(self.anchors)
        candidate.pinned_messages = json_copy(self.pinned_messages)
        candidate._dynamic_assets = {} if reset_segment else dict(self._dynamic_assets)
        candidate._admitted_dynamic_images = [] if reset_segment else list(self._admitted_dynamic_images)
        candidate._committed_messages = [] if reset_segment else self._committed_messages
        candidate._prepared_signature = None
        candidate._parent = self
        candidate._reset_segment = reset_segment
        return candidate

    def commit(self, candidate: "MediaPlan", payload: dict[str, Any], assistant: dict[str, Any]) -> None:
        """Commit only the successful, actually forwarded visual context and response."""
        if candidate._parent is not self or candidate._revision != self._revision:
            raise AdapterError("media_candidate_stale_or_unrelated")
        if self.allow_dynamic_images:
            if candidate._prepared_signature != _visual_signature(payload["messages"]):
                raise AdapterError("media_forwarded_images_changed")
            if _visual_signature([assistant]):
                raise AdapterError("media_assistant_image_response_forbidden")
        self.anchors, self.pinned_messages, self.initialized = (
            candidate.anchors,
            candidate.pinned_messages,
            candidate.initialized,
        )
        self._dynamic_assets = candidate._dynamic_assets
        self._admitted_dynamic_images = candidate._admitted_dynamic_images
        self._seed_message_index = candidate._seed_message_index
        self._seed_message_identity = candidate._seed_message_identity
        self._committed_messages = json_copy(payload["messages"] + [assistant]) if self.allow_dynamic_images else []
        self._revision += 1

    def _restore_dynamic_history(self, messages: list[dict[str, Any]]) -> tuple[bool, set[tuple[int, int]]]:
        """Restore only positional replay of a committed image or overflow omission.

        A changed prefix is left untouched. The caller's history reconciler may
        then request an explicit new segment and rerun a reset candidate from
        the original provider payload, as the frozen Miles core does.
        """
        stored = self._committed_messages
        if not stored:
            return True, set()
        if len(messages) < len(stored):
            return False, set()
        restored, positions = json_copy(messages), set()
        for i, before in enumerate(stored):
            after = messages[i]
            if before.get("role") != after.get("role"):
                return False, set()
            old, new = before.get("content"), after.get("content")
            if isinstance(old, list) != isinstance(new, list):
                return False, set()
            if not isinstance(old, list):
                continue
            if len(old) != len(new):
                return False, set()
            for j, (old_part, new_part) in enumerate(zip(old, new, strict=True)):
                if not isinstance(old_part, dict) or not isinstance(new_part, dict):
                    raise AdapterError("media_dynamic_image_block_invalid")
                old_image, new_image = _image_value(old_part), _image_value(new_part)
                placeholder = new_part.get("type") == "text" and new_part.get("text") == "[图片]"
                omission = old_part.get("type") == "text" and old_part.get("text") == _IMAGE_OMISSION
                if old_image is not None:
                    if old_image != new_image and not placeholder:
                        return False, set()
                    restored[i]["content"][j] = json_copy(old_part)
                    positions.add((i, j))
                elif omission and (new_image is not None or placeholder or new_part == old_part):
                    restored[i]["content"][j] = json_copy(old_part)
                elif new_image is not None or placeholder or new_part.get("text") == _IMAGE_OMISSION:
                    return False, set()
        messages[:] = restored
        return True, positions

    def public_url(self, ref: str) -> str:
        if ref not in self.assets or self.public_base_url is None:
            raise AdapterError("media_public_asset_unavailable")
        return self.public_base_url + MEDIA_PATH + self.scope + "/" + self.assets[ref][0]["sha256"]

    def data_uri(self, ref: str) -> str:
        asset, raw = self.assets[ref]
        return "data:" + asset["mime_type"] + ";base64," + base64.b64encode(raw).decode("ascii")

    def wire_envelope(self, envelope: dict[str, Any]) -> dict[str, Any]:
        def replace(value: Any) -> Any:
            if isinstance(value, str) and value.startswith(ASSET_PREFIX):
                return self.public_url(value)
            if isinstance(value, list):
                return [replace(child) for child in value]
            if isinstance(value, dict):
                return {key: replace(child) for key, child in value.items()}
            return value

        result = replace(json_copy(envelope))
        if ASSET_PREFIX in canonical_bytes(result).decode():
            raise AdapterError("media_wire_asset_unresolved")
        return result

    def _restore_stubs(self, messages: list[dict[str, Any]]) -> None:
        histories = {
            binding["tool_history"]["tool_call_id"]: binding["tool_history"]
            for binding in self.bindings.values()
            if binding.get("tool_history")
        }
        for call_id, history in histories.items():
            indexes = [
                i
                for i, message in enumerate(messages)
                if message.get("role") == "tool" and message.get("tool_call_id") == call_id
            ]
            if not indexes and self.initialized:
                continue
            if len(indexes) != 1:
                raise AdapterError("media_tool_identity_mismatch")
            index = indexes[0]
            content = messages[index].get("content")
            relevant = content == f"[调用了 {history['name']} 工具]" or any(
                marker in text for text in _strings(content) for marker in self.bindings
            )
            if self.initialized and not relevant:
                continue
            _check_tool_call(messages, index, history)
            if messages[index].get("content") == f"[调用了 {history['name']} 工具]":
                restored = json_copy(messages[index])
                restored["content"] = history["content"]
                if self.initialized:
                    marker = next(
                        marker for marker, binding in self.bindings.items() if binding.get("tool_history") == history
                    )
                    if _digest(restored) != _digest(self.pinned_messages[marker]):
                        raise AdapterError("media_tool_message_changed")
                messages[index] = restored

    def restore(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        result = json_copy(payload)
        messages = result.get("messages")
        if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
            raise AdapterError("media_messages_invalid")
        self._restore_stubs(messages)
        if not self.initialized and self.plan["seed_refs"]:
            users = [message for message in messages if message.get("role") == "user"]
            if not users:
                raise AdapterError("media_seed_message_missing")
            aliases = {self.public_url(ref): ref for ref in self.assets}
            aliases.update({self.data_uri(ref): ref for ref in self.assets})
            current_refs = []
            for part in _content_parts(users[-1].get("content")):
                if part.get("type") == "image_url":
                    value = part.get("image_url")
                    current_refs.append(aliases.get(value.get("url") if isinstance(value, dict) else value))
            if current_refs != self.plan["seed_refs"]:
                raise AdapterError("media_seed_images_mismatch")
            if self.allow_dynamic_images:
                self._seed_message_index = next(
                    i for i in reversed(range(len(messages))) if messages[i].get("role") == "user"
                )
                self._seed_message_identity = _seed_nonvisual(users[-1])
        if self.allow_dynamic_images and self.plan["seed_refs"]:
            index = self._seed_message_index
            if index is None or index >= len(messages) or _seed_nonvisual(messages[index]) != self._seed_message_identity:
                raise AdapterError("media_seed_message_changed")
            # The seed is a pinned dataset snapshot, also across segment resets.
            # Only this message's ordered seed occurrences receive reserved slots.
            parts, insert_at = [], None
            for part in _content_parts(messages[index].get("content")):
                if _image_value(part) is not None or part.get("text") in ("[图片]", _IMAGE_OMISSION):
                    if insert_at is None:
                        insert_at = len(parts)
                else:
                    parts.append(part)
            insert_at = len(parts) if insert_at is None else insert_at
            parts[insert_at:insert_at] = [
                {"type": "image_url", "image_url": {"url": self.data_uri(ref)}} for ref in self.plan["seed_refs"]
            ]
            messages[index]["content"] = parts
        markers = set(self.bindings)
        anchors, pinned = dict(self.anchors), dict(self.pinned_messages)
        if not self.initialized:
            located = set()
            for message in messages:
                for marker, binding in self.bindings.items():
                    total = sum(text.count(marker) for text in _strings(message))
                    if not total:
                        continue
                    if (
                        total != 1
                        or sum(text.count(marker) for text in _strings(message.get("content"))) != 1
                        or marker in located
                    ):
                        raise AdapterError("media_marker_count_mismatch")
                    if message.get("role") != binding["role"]:
                        raise AdapterError("media_marker_role_mismatch")
                    anchors[marker] = _message_anchor(message, markers)
                    pinned[marker] = json_copy(message)
                    located.add(marker)
            if located != markers:
                raise AdapterError("media_marker_missing")
        public_refs = {self.public_url(ref): ref for ref in self.assets}
        data_refs = {self.data_uri(ref): ref for ref in self.assets}
        for message_index, message in enumerate(messages):
            content = message.get("content")
            if content is None:
                continue
            parts = _content_parts(content)
            expanded = []
            for part in parts:
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    text, cursor = part["text"], 0
                    if message.get("role") not in {"user", "tool"}:
                        expanded.append(part)
                        continue
                    matches = list(MARKER.finditer(text))
                    for match in matches:
                        marker = match[0]
                        if marker not in self.bindings:
                            raise AdapterError("media_marker_unregistered")
                        if _message_anchor(message, markers) != anchors[marker]:
                            raise AdapterError("media_marker_message_changed")
                        if match.start() > cursor:
                            expanded.append({"type": "text", "text": text[cursor : match.start()]})
                        expanded.append(
                            {"type": "image_url", "image_url": {"url": self.data_uri(self.bindings[marker]["source"])}}
                        )
                        cursor = match.end()
                    if cursor < len(text) or not matches:
                        expanded.append({"type": "text", "text": text[cursor:]})
                elif part.get("type") == "image_url" or (self.allow_dynamic_images and part.get("type") == "image"):
                    value = part.get("image_url")
                    source = value.get("url") if isinstance(value, dict) else value
                    if self.allow_dynamic_images:
                        source = _image_value(part)
                        if message.get("role") not in {"user", "tool", "assistant"}:
                            raise AdapterError("media_dynamic_image_role_invalid")
                    ref = public_refs.get(source) or data_refs.get(source)
                    if ref is None:
                        if not self.allow_dynamic_images:
                            raise AdapterError("media_unregistered_image")
                        canonical = source
                    else:
                        canonical = self.data_uri(ref)
                    expanded.append({"type": "image_url", "image_url": {"url": canonical}})
                else:
                    if any(marker in text for text in _strings(part) for marker in markers):
                        raise AdapterError("media_marker_outside_text")
                    raise AdapterError("media_content_part_unsupported")
            # Keep all-text payload shape unchanged, but materialize visual occurrences.
            if expanded != parts or isinstance(content, list):
                message["content"] = expanded
        # Expand prepared markers and compressed tool stubs before comparing the
        # historical prefix: the committed view already contains canonical pixels.
        history_aligned, replay_positions = (
            self._restore_dynamic_history(messages) if self.allow_dynamic_images else (True, set())
        )
        occurrences, omitted, newly_admitted, replayed = [], 0, 0, 0
        seed_remaining = [self.data_uri(ref) for ref in self.plan["seed_refs"]]
        for message_index, message in enumerate(messages):
            if not isinstance(message.get("content"), list):
                continue
            for part_index, part in enumerate(message["content"]):
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    ref = data_refs.get(url)
                    if self.allow_dynamic_images:
                        seed = message_index == self._seed_message_index and seed_remaining and url == seed_remaining[0]
                        if seed:
                            seed_remaining.pop(0)
                            replayed += 1
                        elif (message_index, part_index) in replay_positions:
                            replayed += 1
                        elif (
                            (not history_aligned and self._committed_messages)
                            or len(self.plan["seed_refs"]) + len(self._admitted_dynamic_images) >= MAX_IMAGES
                        ):
                            message["content"][part_index] = {"type": "text", "text": _IMAGE_OMISSION}
                            omitted += 1
                            continue
                        else:
                            # Validate/cache only admitted occurrences. Overflow,
                            # including invalid or oversized input, stays text.
                            if ref is None and url not in self._dynamic_assets:
                                url, identity = _dynamic_image(url)
                                self._dynamic_assets[url] = identity
                                part["image_url"]["url"] = url
                            self._admitted_dynamic_images.append(hashlib.sha256(url.encode()).hexdigest())
                            newly_admitted += 1
                    asset = self.assets[ref][0] if ref else self._dynamic_assets.get(url)
                    if asset is None:
                        raise AdapterError("media_unregistered_image")
                    occurrences.append(
                        {
                            "message_index": message_index,
                            "part_index": part_index,
                            "sha256": asset["sha256"],
                            "byte_length": asset["byte_length"],
                            "mime_type": asset["mime_type"],
                            **({"origin": "dynamic" if ref is None else "prepared"} if self.allow_dynamic_images else {}),
                        }
                    )
        if self.allow_dynamic_images:
            if seed_remaining:
                raise AdapterError("media_seed_images_mismatch")
            self._prepared_signature = _visual_signature(messages)
        elif len(occurrences) > MAX_IMAGES:
            raise AdapterError("media_image_limit_exceeded")
        self.anchors, self.pinned_messages, self.initialized = anchors, pinned, True
        return result, {
            "schema_version": 1,
            "byte_identity_verified": True,
            "occurrences": occurrences,
            "first_request_message_sha256": {marker: _digest(message) for marker, message in pinned.items()},
            **(
                {
                    "dynamic_images": {
                        "policy": "pinned_compat",
                        "history_aligned": history_aligned,
                        "segment_reset": self._reset_segment,
                        "admitted_occurrences": newly_admitted,
                        "admitted_total": len(self.plan["seed_refs"]) + len(self._admitted_dynamic_images),
                        "replayed_occurrences": replayed,
                        "omitted_occurrences": omitted,
                    }
                }
                if self.allow_dynamic_images else {}
            ),
        }
