# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Production-only restoration of omitted, previously observed tool reasoning.

The reconciliation follows the formal V21 frozen
miles/agentspark_session/chat_template_utils/template.py (SHA-256 below),
not the newer cross-history reasoning search. Only the fixed stored prefix
may be restored; appended tool-result IDs follow the validated call mapping.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .media import canonical_bytes, strict_json_loads
from .protocol import AdapterError, json_copy


FROZEN_RECONCILIATION_SHA256 = "6272ae85844f92a6e1d41380b865b2b2b44d31c111b5cbfcf8377ca322959545"


def _calls(calls: Any) -> list[tuple[str, str, bytes]]:
    if not isinstance(calls, list) or not calls:
        raise AdapterError("history_tool_calls_invalid")
    result = []
    for call in calls:
        if not isinstance(call, dict) or call.get("type") not in (None, "function"):
            raise AdapterError("history_tool_calls_invalid")
        function = call.get("function")
        if function is None and "name" in call and "arguments" in call:
            function = call
        if not isinstance(function, dict):
            raise AdapterError("history_tool_calls_invalid")
        call_id, name, arguments = call.get("id"), function.get("name"), function.get("arguments")
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
            raise AdapterError("history_tool_calls_invalid")
        try:
            if isinstance(arguments, str):
                arguments = strict_json_loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be an object")
            encoded = canonical_bytes(arguments)
        except (TypeError, ValueError) as exc:
            raise AdapterError("history_tool_calls_invalid") from exc
        result.append((call_id, name, encoded))
    if len({item[0] for item in result}) != len(result) or len({item[1:] for item in result}) != len(result):
        raise AdapterError("history_tool_calls_ambiguous")
    return result


def _omitted_content(stored: dict[str, Any], incoming: dict[str, Any]) -> bool:
    if (
        stored.get("role") != "assistant"
        or incoming.get("role") != "assistant"
        or not stored.get("tool_calls")
        or not incoming.get("tool_calls")
    ):
        return False
    content = stored.get("content")
    if incoming.get("content") not in (None, "") or not isinstance(content, str):
        return False
    empty_think = re.fullmatch(r"\s*(?:<think>\s*)?</think>\s*", content) is not None
    narration = (
        incoming.get("reasoning_content") in (None, "")
        and "reasoning_content" in stored
        and (stored["reasoning_content"] is None or isinstance(stored["reasoning_content"], str))
        and (not content.strip() or re.fullmatch(r"\s*<short_content>[^<>]*</short_content>\s*", content) is not None)
    )
    return empty_think or narration


def pin_training_request(payload: dict[str, Any], defaults: dict[str, Any], temperature: float) -> dict[str, Any]:
    """Apply the frozen pin policy using K3's native ``thinking`` template key."""
    if not isinstance(defaults, dict) or type(defaults.get("thinking")) is not bool:
        raise AdapterError("production_k3_thinking_default_required")
    controls = {"enable_thinking", "preserve_thinking", "clear_thinking", "thinking", "separate_reasoning"}

    def validate(values: dict[str, Any]) -> None:
        if any(type(values[key]) is not bool for key in controls & values.keys()):
            raise AdapterError("production_thinking_control_invalid")
        if "reasoning_effort" in values and (
            not isinstance(values["reasoning_effort"], str) or not values["reasoning_effort"]
        ):
            raise AdapterError("production_thinking_control_invalid")

    client = payload.get("chat_template_kwargs")
    client = {} if client is None else client
    if not isinstance(client, dict):
        raise AdapterError("production_template_kwargs_invalid")
    validate(client)
    validate({key: value for key, value in payload.items() if value is not None})
    if client.keys() - defaults.keys() - controls - {"reasoning_effort"}:
        raise AdapterError("production_unconfigured_template_kwargs")
    reasoning = payload.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, dict):
        raise AdapterError("production_reasoning_invalid")
    aliases = {
        "effort": "reasoning_effort",
        "reasoning_effort": "reasoning_effort",
        "enabled": "enable_thinking",
        "enable": "enable_thinking",
    }
    for alias, key in aliases.items():
        if reasoning and reasoning.get(alias) is not None:
            validate({key: reasoning[alias]})
    if temperature is None:
        raise AdapterError("production_sampling_temperature_invalid")
    for value in (temperature, payload.get("temperature", temperature)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
        ):
            raise AdapterError("production_sampling_temperature_invalid")
    if (
        payload.get("top_p") not in (None, 1.0)
        or isinstance(payload.get("top_p"), bool)
        or (payload.get("top_k") is not None and (type(payload["top_k"]) is not int or payload["top_k"] != -1))
    ):
        raise AdapterError("production_sampling_support_unsupported")
    result = json_copy(payload)
    result["chat_template_kwargs"] = json_copy(defaults)
    result["enable_thinking"] = defaults["thinking"]
    result["separate_reasoning"] = True
    result["reasoning_effort"] = defaults.get("reasoning_effort") if defaults["thinking"] else None
    for key in ("thinking", "preserve_thinking", "clear_thinking"):
        result.pop(key, None)
    if reasoning is not None:
        result["reasoning"] = {key: value for key, value in reasoning.items() if key not in aliases} or None
    result["temperature"], result["top_p"], result["top_k"] = temperature, 1.0, -1
    return result


class ReasoningHistory:
    """Restore from the last successfully observed context, never synthesize text."""

    def __init__(self, defaults: dict[str, Any] | None = None, temperature: float | None = None) -> None:
        self.defaults, self.temperature = defaults, temperature
        if defaults is not None:
            pin_training_request({}, defaults, temperature)
        self.messages: list[dict[str, Any]] = []
        self.tools: Any = None

    def restore(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        result = (
            json_copy(payload)
            if self.defaults is None
            else pin_training_request(payload, self.defaults, self.temperature)
        )
        messages = result.get("messages")
        if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
            raise AdapterError("history_messages_invalid")
        evidence = {
            "policy": "formal_v21_fixed_prefix_reconcile_v1",
            "source_sha256": FROZEN_RECONCILIATION_SHA256,
            "restored": [],
            "mapped_tool_results": 0,
            "template_pinned": self.defaults is not None,
            "temperature_pinned": self.temperature,
        }
        if not self.messages:
            evidence["status"] = "initial"
            return result, evidence
        if self.tools != (payload.get("tools") or None):
            evidence.update(status="new_segment", reason="tool_schema_mismatch")
            return result, evidence
        try:
            actual, restored, mapped = self._reconcile(messages)
        except AdapterError as exc:
            # The frozen core treats these semantic continuity failures as
            # segment restarts. Never publish a partially restored prefix.
            evidence.update(status="new_segment", reason=exc.code)
            return result, evidence
        result["messages"] = actual
        evidence.update(status="reconciled", restored=restored, mapped_tool_results=mapped)
        return result, evidence

    def _reconcile(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        def normalized(value: Any) -> Any:
            return None if value is None or value == "" or value == [] else value

        if len(messages) < len(self.messages):
            raise AdapterError("history_prefix_changed")
        restored = []
        last_map: dict[str, str] = {}
        last_names: dict[str, str] = {}
        for index, (stored, incoming) in enumerate(zip(self.messages, messages[: len(self.messages)], strict=True)):
            fields = []
            for key in ("role", "content", "reasoning_content"):
                if normalized(stored.get(key)) == normalized(incoming.get(key)):
                    continue
                if key == "content" and _omitted_content(stored, incoming):
                    fields.append(key)
                    continue
                if (
                    key == "reasoning_content"
                    and stored.get("role") == incoming.get("role") == "assistant"
                    and stored.get("tool_calls")
                    and incoming.get("tool_calls")
                    and isinstance(stored.get(key), str)
                    and incoming.get(key) in (None, "")
                ):
                    fields.append(key)
                    continue
                raise AdapterError("history_prefix_changed")
            if stored.get("role") != "assistant":
                if normalized(stored.get("tool_calls")) != normalized(incoming.get("tool_calls")):
                    raise AdapterError("history_prefix_changed")
                continue
            before = [] if normalized(stored.get("tool_calls")) is None else _calls(stored["tool_calls"])
            after = [] if normalized(incoming.get("tool_calls")) is None else _calls(incoming["tool_calls"])
            if [call[1:] for call in before] != [call[1:] for call in after]:
                raise AdapterError("history_tool_calls_semantic_mismatch")
            last_map = {
                incoming_call[0]: stored_call[0] for stored_call, incoming_call in zip(before, after, strict=True)
            }
            last_names = {call[0]: call[1] for call in after}
            if normalized(stored.get("tool_calls")) != normalized(incoming.get("tool_calls")):
                fields.append("tool_calls")
            if fields:
                restored.append({"message_index": index, "previous_message_index": index, "fields": fields})
        actual = json_copy(self.messages) + json_copy(messages[len(self.messages) :])
        seen: set[str] = set()
        mapped = 0
        for message in actual[len(self.messages) :]:
            if message.get("role") != "tool":
                continue
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in last_map or call_id in seen:
                raise AdapterError("history_tool_result_unmapped")
            seen.add(call_id)
            if message.get("name") not in (None, "", last_names[call_id]):
                raise AdapterError("history_tool_result_name_mismatch")
            message["tool_call_id"] = last_map[call_id]
            mapped += int(call_id != last_map[call_id])
        if last_map and seen != set(last_map):
            raise AdapterError("history_tool_result_unmapped")
        return actual, restored, mapped

    def commit(self, payload: dict[str, Any], assistant: dict[str, Any]) -> None:
        self.messages = json_copy(payload["messages"]) + [json_copy(assistant)]
        self.tools = json_copy(payload.get("tools") or None)
