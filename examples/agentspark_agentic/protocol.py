# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Explicit, prepared AgentSpark envelopes; no dataset or history reconstruction."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlsplit


SESSION_PATH = "/diandian/session"
SEND_PATH = "/diandian/chat/sse-send-protocol"
CHAT_PATH = "/v1/chat/completions"
MAX_PREPARED_TEXT_BYTES = 1900
REPLAY_ATTRIBUTE_KEYS = {
    "ark.idindex.mapping",
    "ark.idindex.image_urls",
    "ark.idindex.reverse_mapping",
}


class AdapterError(RuntimeError):
    """A fixed, non-sensitive failure code safe to include in audit metadata."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise AdapterError("non_json_value") from exc


def required_text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(code)
    return value


def object_value(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdapterError(code)
    return json_copy(value)


def http_url(value: Any, code: str) -> str:
    value = required_text(value, code)
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as exc:
        raise AdapterError(code) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(char.isspace() for char in value)
    ):
        raise AdapterError(code)
    return value.rstrip("/")


def positive_seconds(value: Any, code: str, *, nullable: bool = False) -> float | None:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise AdapterError(code)
    return float(value)


@dataclass(frozen=True)
class AdapterConfig:
    base_url: str
    model: str
    agent: str
    source: str
    max_tokens: int
    callback_bind_host: str
    callback_bind_port: int
    callback_public_base_url: str
    connect_timeout_s: float = 10.0
    read_timeout_s: float | None = None
    cleanup_timeout_s: float = 2.0
    max_callback_bytes: int = 16 * 1024 * 1024
    media_allowed_roots: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AdapterConfig":
        if not isinstance(payload, dict):
            raise AdapterError("config_not_object")
        required = {
            "base_url",
            "model",
            "agent",
            "source",
            "max_tokens",
            "callback_bind_host",
            "callback_bind_port",
            "callback_public_base_url",
        }
        optional = {
            "connect_timeout_s",
            "read_timeout_s",
            "cleanup_timeout_s",
            "max_callback_bytes",
            "media_allowed_roots",
        }
        if not required <= payload.keys() or set(payload) - required - optional:
            raise AdapterError("config_fields_invalid")
        config = dict(payload)
        config["base_url"] = http_url(config["base_url"], "agentspark_url_invalid")
        config["callback_public_base_url"] = http_url(config["callback_public_base_url"], "callback_url_invalid")
        for key in ("model", "agent", "source", "callback_bind_host"):
            required_text(config[key], f"config_{key}_invalid")
        if config["source"] != "eden":
            raise AdapterError("only_eden_protocol_supported")
        for key in ("max_tokens", "max_callback_bytes"):
            value = config.get(key, cls.max_callback_bytes)
            if type(value) is not int or value <= 0:
                raise AdapterError(f"config_{key}_invalid")
        port = config["callback_bind_port"]
        if type(port) is not int or not 0 <= port <= 65535:
            raise AdapterError("callback_port_invalid")
        for key in ("connect_timeout_s", "cleanup_timeout_s"):
            config[key] = positive_seconds(config.get(key, getattr(cls, key)), f"config_{key}_invalid")
        config["read_timeout_s"] = positive_seconds(
            config.get("read_timeout_s"),
            "config_read_timeout_s_invalid",
            nullable=True,
        )
        roots = config.get("media_allowed_roots", [])
        if not isinstance(roots, (list, tuple)) or any(
            not isinstance(root, str) or not root.startswith("/") for root in roots
        ):
            raise AdapterError("config_media_allowed_roots_invalid")
        config["media_allowed_roots"] = tuple(roots)
        return cls(**config)


@dataclass(frozen=True)
class RelaxContext:
    base_url: str
    session_id: str = field(repr=False)
    group_id: str
    mode: str
    invocation_id: str = field(repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "RelaxContext":
        values = {}
        for key in (
            "RELAX_BASE_URL",
            "RELAX_SESSION_ID",
            "RELAX_GROUP_ID",
            "RELAX_ROLLOUT_MODE",
            "RELAX_SESSION_IO_DIR",
        ):
            values[key] = required_text(env.get(key), f"missing_{key.lower()}")
        if values["RELAX_ROLLOUT_MODE"] not in {"train", "eval"}:
            raise AdapterError("rollout_mode_invalid")
        return cls(
            http_url(values["RELAX_BASE_URL"], "relax_base_url_invalid"),
            values["RELAX_SESSION_ID"],
            values["RELAX_GROUP_ID"],
            values["RELAX_ROLLOUT_MODE"],
            values["RELAX_SESSION_IO_DIR"],
        )

    @property
    def identity_digest(self) -> str:
        return hashlib.sha256(f"{self.session_id}\0{self.invocation_id}".encode()).hexdigest()

    @property
    def user_id(self) -> str:
        return f"eden_relax_{self.identity_digest[:32]}"

    @property
    def request_id(self) -> str:
        return f"relax-{self.identity_digest[:32]}"


def prepare_envelope(session_input: Any) -> dict[str, Any]:
    """Require an explicit prepared envelope instead of guessing from messages.

    Native prepared inputs, replay messages/resource mappings and prompt overrides
    must come from the source dataset converter. This layer only validates them.
    """
    if not isinstance(session_input, dict) or not isinstance(session_input.get("metadata"), dict):
        raise AdapterError("metadata_agentspark_required")
    source = session_input["metadata"].get("agentspark")
    envelope = object_value(source, "metadata_agentspark_required")
    required = {"schema_version", "inputs", "tools", "prompt_spec"}
    optional = {
        "bound_user_id",
        "tool_versions",
        "capabilities",
        "skills_name_list",
        "replay_spec",
    }
    if not required <= envelope.keys() or set(envelope) - required - optional:
        raise AdapterError("agentspark_envelope_fields_invalid")
    if type(envelope["schema_version"]) is not int or envelope["schema_version"] != 1:
        raise AdapterError("agentspark_envelope_version_invalid")
    tools = envelope["tools"]
    if (
        not isinstance(tools, list)
        or not tools
        or any(not isinstance(tool, str) or not tool or tool != tool.strip() for tool in tools)
    ):
        raise AdapterError("tools_invalid")
    if len(set(tools)) != len(tools):
        raise AdapterError("tools_duplicate")
    inputs = envelope["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise AdapterError("inputs_invalid")
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"type", "info"}:
            raise AdapterError("input_item_invalid")
        kind, info = item["type"], item["info"]
        if kind == "TEXT":
            if not isinstance(info, dict) or set(info) != {"text"}:
                raise AdapterError("text_input_invalid")
            required_text(info["text"], "text_input_invalid")
        elif kind in {"NOTE", "POI"}:
            key = "note_id" if kind == "NOTE" else "id"
            if not isinstance(info, dict) or set(info) != {key}:
                raise AdapterError("attachment_input_invalid")
            required_text(info[key], "attachment_input_invalid")
        elif kind in {"IMAGE", "IMAGES"}:
            # Prepared inputs must already use the known wire list shape.
            images = info
            if not isinstance(images, list) or not images:
                raise AdapterError("image_input_invalid")
            for image in images:
                if not isinstance(image, dict) or set(image) - {
                    "image_url",
                    "file_id",
                    "width",
                    "height",
                }:
                    raise AdapterError("image_input_invalid")
                if ("image_url" in image) == ("file_id" in image):
                    raise AdapterError("image_input_invalid")
                required_text(image.get("image_url", image.get("file_id")), "image_input_invalid")
                for dimension in ("width", "height"):
                    if dimension in image and (type(image[dimension]) is not int or image[dimension] <= 0):
                        raise AdapterError("image_input_invalid")
        else:
            raise AdapterError("input_type_unsupported")
    if (
        sum(len(item["info"]["text"].encode("utf-8")) for item in inputs if item["type"] == "TEXT")
        > MAX_PREPARED_TEXT_BYTES
    ):
        raise AdapterError("text_requires_source_preparation")
    prompt = envelope["prompt_spec"]
    if not isinstance(prompt, dict) or set(prompt) != {"system_prompt", "user_prompt"}:
        raise AdapterError("explicit_system_and_user_prompt_required")
    for value in prompt.values():
        required_text(value, "prompt_invalid")
    if "bound_user_id" in envelope:
        bound = required_text(envelope["bound_user_id"], "bound_user_id_invalid")
        if bound != bound.strip():
            raise AdapterError("bound_user_id_invalid")
    versions = envelope.get("tool_versions")
    if versions is not None and (
        not isinstance(versions, dict)
        or any(key not in tools or not isinstance(value, str) or not value.strip() for key, value in versions.items())
    ):
        raise AdapterError("tool_versions_invalid")
    capabilities = envelope.get("capabilities")
    if capabilities is not None and (
        not isinstance(capabilities, dict)
        or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in capabilities.items()
        )
    ):
        raise AdapterError("capabilities_invalid")
    skills = envelope.get("skills_name_list")
    if skills is not None and (
        not isinstance(skills, list)
        or any(not isinstance(name, str) or not name for name in skills)
        or len(set(skills)) != len(skills)
    ):
        raise AdapterError("skills_invalid")
    if capabilities and capabilities.get("skill_loader") == "1" and skills:
        system = prompt["system_prompt"]
        if "<AvailableSkills>" not in system or "</AvailableSkills>" not in system:
            raise AdapterError("loader_catalog_required")
        catalog = system.split("<AvailableSkills>", 1)[1].split("</AvailableSkills>", 1)[0]
        if any(name not in catalog for name in skills):
            raise AdapterError("loader_catalog_incomplete")
    replay = envelope.get("replay_spec")
    if replay is not None:
        if not isinstance(replay, dict) or set(replay) != {"messages", "attributes"}:
            raise AdapterError("replay_spec_invalid")
        messages = replay["messages"]
        if not isinstance(messages, list) or not messages:
            raise AdapterError("replay_messages_invalid")
        for message in messages:
            if not isinstance(message, dict):
                raise AdapterError("replay_messages_invalid")
            required_text(message.get("role"), "replay_messages_invalid")
        attributes = replay["attributes"]
        if not isinstance(attributes, dict) or set(attributes) != REPLAY_ATTRIBUTE_KEYS:
            raise AdapterError("replay_attributes_required")
        if any(not isinstance(value, dict) for value in attributes.values()):
            raise AdapterError("replay_attributes_invalid")
    return envelope


def user_payload(context: RelaxContext, envelope: dict[str, Any]) -> dict[str, str]:
    user = {"user_id": context.user_id}
    if "bound_user_id" in envelope:
        user["bound_user_id"] = envelope["bound_user_id"]
    return user


def replay_body(config: AdapterConfig, context: RelaxContext, envelope: dict[str, Any]) -> dict[str, Any]:
    replay = envelope.get("replay_spec")
    if replay is None:
        raise AdapterError("replay_spec_required")
    return {
        "request_id": context.request_id + "-restore",
        "agent": config.agent,
        "source": config.source,
        "type": "send",
        "user": user_payload(context, envelope),
        "replay_spec": {"replay_type": "restore_only", **copy.deepcopy(replay)},
    }


def llm_config(config: AdapterConfig, context: RelaxContext, callback_base_url: str) -> dict[str, Any]:
    endpoint = http_url(callback_base_url, "callback_url_invalid") + CHAT_PATH
    parsed = urlsplit(endpoint)
    # AgentSpark resolves endpoint_path independently; keep the service prefix
    # in that path rather than relying on URL-join behavior for a base path.
    return {
        "provider": "openai",
        "base_url": f"{parsed.scheme}://{parsed.netloc}",
        "endpoint_path": parsed.path,
        "auth_mode": "bearer",
        "api_key": context.session_id,
        "model": config.model,
        "max_tokens": config.max_tokens,
    }


def send_body(
    config: AdapterConfig,
    context: RelaxContext,
    envelope: dict[str, Any],
    session_id: str,
    callback_base_url: str,
) -> dict[str, Any]:
    body = {
        "request_id": context.request_id,
        "session_id": required_text(session_id, "remote_session_id_invalid"),
        "agent": config.agent,
        "source": config.source,
        "type": "send",
        "user": user_payload(context, envelope),
        "llm_config": llm_config(config, context, callback_base_url),
    }
    body.update(
        {
            key: copy.deepcopy(value)
            for key, value in envelope.items()
            if key not in {"schema_version", "replay_spec", "bound_user_id"}
        }
    )
    return body
