"""Offline source conversion regressions; no inference, network, or GPU calls."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from examples.agentspark_agentic.prepare import (
    MilesPreparedConverter,
    PreparationConfig,
    PreparationError,
    _validate_history,
    load_verified_media_manifest,
    prepare_jsonl,
    strict_json,
)


USER_TEMPLATE = (
    "<RuntimeContext>{{entrance}}|{{rejoin_flag}}|{{current_time}}"
    "<UserBasicInfo>{{user_location}}</UserBasicInfo>"
    "<SceneMemory>{{user_scene_memory}}</SceneMemory></RuntimeContext>"
    "<Attachment>{{Attachment}}</Attachment><UserInput>{{user_query}}</UserInput>"
)


def _write_profile(tmp_path: Path, *, source: Path | None = None) -> tuple[Path, Path, Path]:
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "system_prompt.md").write_text("explicit test system", encoding="utf-8")
    (profile / "user_prompt.md").write_text(USER_TEMPLATE, encoding="utf-8")
    tools = ["search_request", "note_read"]
    (profile / "tools.json").write_text(json.dumps(tools))
    config = {
        "runtime_env": {
            "SLIME_AGENTSPARK_PROMPT_PROFILE": "offline-fixture",
            "DOTSBRIDGE_AGENTSPARK_SYSTEM_PROMPT_FILE": str(profile / "system_prompt.md"),
            "DOTSBRIDGE_AGENTSPARK_USER_PROMPT_FILE": str(profile / "user_prompt.md"),
            "DOTSBRIDGE_AGENTSPARK_TOOLS_JSON": json.dumps(tools),
            "DOTSBRIDGE_AGENTSPARK_TOOL_VERSIONS_JSON": '{"search_request":"SearchRequestV2Tool"}',
            "DOTSBRIDGE_AGENTSPARK_RUNTIME_CONTEXT_FROM_HISTORY": "1",
            "SLIME_AGENTSPARK_SKILL_LOADER_ENABLE": "0",
            "UNRELATED_SECRET": "must-not-enter-artifact",
        },
        "agent": {
            "max_images": 30,
            "environment_max_chars": 65536,
            "history_max_chars": 12000,
            "resource_max_chars": 6000,
            "image_transport": "agentspark_remote_url",
            "image_selection_policy": "pinned_compat",
            "image_allowed_hosts": ["ci.xiaohongshu.com", "images.example.test"],
        },
    }
    path = tmp_path / "source-config.json"
    path.write_text(json.dumps(config))
    return path, profile, source or tmp_path


def _config(paths: tuple[Path, Path, Path]) -> PreparationConfig:
    config, profile, source = paths
    return PreparationConfig.from_source_config(config, miles_source=source, profile_dir=profile)


def _row(query: str = "current request") -> dict[str, Any]:
    return {
        "query": query,
        "label": "",
        "image_urls": [],
        "metadata": {
            "query": query,
            "raw_query": query,
            "query_id": "query-fixture",
            "user_id": "business-user-fixture",
            "expression_tag": "test-tag",
            "rubric": {"response_rubric": {"must_have": [{"id": "criterion-1"}]}},
            "history": [],
            "replay_history": [],
            "attachments": [],
            "image_urls": [],
            "no_new_turn": True,
            "input_repair": {"version": "source-repair"},
        },
    }


@pytest.fixture
def converter(tmp_path: Path) -> Any:
    source = os.environ.get("MILES_SOURCE_DIR")
    if source is None:
        pytest.skip("set MILES_SOURCE_DIR to explicitly validate the authoritative offline Miles compiler")
    config = _config(_write_profile(tmp_path, source=Path(source)))
    instance = MilesPreparedConverter(config, allowed_roots=[tmp_path])
    with instance.source():
        yield instance


def _image(path: Path) -> str:
    from PIL import Image

    Image.new("RGB", (2, 3), color="red").save(path)
    return str(path)


def test_source_profile_is_explicit_and_only_whitelisted_identity_is_retained(tmp_path: Path) -> None:
    paths = _write_profile(tmp_path)
    config = _config(paths)
    assert config.tools == ["search_request", "note_read"]
    assert config.tool_versions == {"search_request": "SearchRequestV2Tool"}
    assert config.runtime_from_history is True
    assert config.max_images == 30
    serialized = json.dumps(config.identity)
    assert "must-not-enter-artifact" not in serialized
    assert "UNRELATED_SECRET" not in serialized
    assert config.identity["profile_files"]["user_prompt.md"] == hashlib.sha256(USER_TEMPLATE.encode()).hexdigest()


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ("tools", "source_tools_mismatch"),
        ("runtime", "source_runtime_policy_missing"),
        ("loader", "source_skill_loader_not_supported"),
        ("slot", "source_query_slot_invalid"),
    ],
)
def test_recipe_changes_require_explicit_supported_contract(tmp_path: Path, change: str, code: str) -> None:
    paths = _write_profile(tmp_path)
    config = json.loads(paths[0].read_text())
    if change == "tools":
        config["runtime_env"]["DOTSBRIDGE_AGENTSPARK_TOOLS_JSON"] = '["different"]'
    elif change == "runtime":
        del config["runtime_env"]["DOTSBRIDGE_AGENTSPARK_RUNTIME_CONTEXT_FROM_HISTORY"]
    elif change == "loader":
        config["runtime_env"]["SLIME_AGENTSPARK_SKILL_LOADER_ENABLE"] = "1"
    else:
        (paths[1] / "user_prompt.md").write_text("missing required query slot")
    paths[0].write_text(json.dumps(config))
    with pytest.raises(PreparationError, match="^" + code + "$"):
        _config(paths)


def test_duplicate_json_keys_are_rejected_before_last_value_can_replace_data() -> None:
    with pytest.raises(PreparationError, match="^duplicate_json_key_conflicting$"):
        strict_json('{"metadata":{"query":"first","query":"second"}}')
    with pytest.raises(PreparationError, match="^history_tool_arguments_duplicate_json_key_conflicting$"):
        _validate_history(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call",
                            "function": {"name": "search_request", "arguments": '{"queries":1,"queries":2}'},
                        }
                    ],
                }
            ]
        )


@pytest.mark.parametrize("content", [None, "", [], [{"type": "text", "text": ""}]])
def test_empty_history_is_rejected_explicitly_instead_of_silently_dropped(content: Any) -> None:
    with pytest.raises(PreparationError, match="^history_empty_content$"):
        _validate_history([{"role": "user", "content": content}])


def test_current_query_is_not_reextracted_and_empty_replay_is_authoritative(converter: MilesPreparedConverter) -> None:
    row = _row()
    row["metadata"]["history"] = [{"role": "user", "content": "old question"}]
    original = copy.deepcopy(row)
    output = converter.convert(row, source_line=7)
    assert row == original
    assert output["messages"] == [{"role": "user", "content": "current request"}]
    assert "replay_spec" not in output["metadata"]["agentspark"]
    assert output["metadata"]["agentspark_prepare"]["selected_history"] == "replay_history"
    assert output["metadata"]["agentspark_prepare"]["training_ready"] is False
    assert output["metadata"]["rubric"] == row["metadata"]["rubric"]
    assert output["metadata"]["input_repair"] == row["metadata"]["input_repair"]
    assert output["metadata"]["agentspark"]["bound_user_id"] == row["metadata"]["user_id"]


def test_prior_tool_ids_and_argument_spelling_survive_replay(converter: MilesPreparedConverter) -> None:
    row = _row()
    history = [
        {"role": "user", "content": "previous request"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "saved reasoning",
            "tool_calls": [
                {
                    "id": "opaque-call-A",
                    "type": "function",
                    "function": {"name": "search_request", "arguments": '{ "n":1e2 }'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "opaque-call-A", "content": "saved result"},
    ]
    row["metadata"]["replay_history"] = history
    result = converter.convert(row, source_line=1)
    replay = result["metadata"]["agentspark"]["replay_spec"]
    assert [m["role"] for m in replay["messages"]] == ["SYSTEM", "USER", "ASSISTANT", "TOOL"]
    assert replay["messages"][2]["tool_calls"] == history[1]["tool_calls"]
    assert replay["messages"][2]["reasoning_content"] == "saved reasoning"
    assert replay["messages"][3]["tool_call_id"] == "opaque-call-A"
    assert set(replay["attributes"]) == {
        "ark.idindex.mapping",
        "ark.idindex.image_urls",
        "ark.idindex.reverse_mapping",
    }


def test_long_utf8_query_roundtrips_through_template_prefix_and_send_suffix(converter: MilesPreparedConverter) -> None:
    query = "甲🛰️" * 1400 + " final request"
    result = converter.convert(_row(query), source_line=1)
    envelope = result["metadata"]["agentspark"]
    suffix = envelope["inputs"][0]["info"]["text"]
    assert len(suffix.encode()) <= 1900
    rendered = envelope["prompt_spec"]["user_prompt"].replace("{{user_query}}", suffix)
    assert rendered.split("<UserInput>")[1].split("</UserInput>")[0] == query
    assert result["messages"][0]["content"] == query


def test_query_bound_runtime_and_profile_are_rendered_and_original_metadata_preserved(
    converter: MilesPreparedConverter,
) -> None:
    row = _row()
    query_hash = hashlib.sha256(row["query"].encode()).hexdigest()
    row["metadata"]["current_user_profile"] = {
        "schema_version": 1,
        "query_sha256": query_hash,
        "values": {"gender": "fixture-gender", "age_range": "fixture-age", "frequent_city": "fixture-city"},
    }
    row["metadata"]["current_runtime_context"] = {
        "schema_version": 1,
        "source": "original_current_user",
        "query_sha256": query_hash,
        "values": {
            "entrance": "fixture-entrance",
            "rejoin_flag": "0",
            "current_time": "fixture-time",
            "user_location": "fixture-location",
            "user_scene_memory": "fixture-memory",
        },
    }
    result = converter.convert(row, source_line=1)
    prompt = result["metadata"]["agentspark"]["prompt_spec"]["user_prompt"]
    for value in (
        "fixture-time",
        "fixture-location",
        "fixture-memory",
        "fixture-gender",
        "fixture-age",
        "fixture-city",
    ):
        assert value in prompt
    assert result["metadata"]["current_user_profile"] == row["metadata"]["current_user_profile"]
    row["metadata"]["current_runtime_context"]["query_sha256"] = "0" * 64
    with pytest.raises(PreparationError, match="^miles_source_compilation_failed$"):
        converter.convert(row, source_line=1)


def test_snapshot_images_become_private_occurrences_not_plain_text_or_global_image_column(
    converter: MilesPreparedConverter, tmp_path: Path
) -> None:
    row = _row()
    path = _image(tmp_path / "snapshot.png")
    row["metadata"]["current_attachment_snapshot"] = {
        "schema_version": 1,
        "query_sha256": hashlib.sha256(row["query"].encode()).hexdigest(),
        "parts": [
            {"type": "text", "text": "before "},
            {"type": "image_url", "image_url": {"url": path}},
            {"type": "text", "text": " after"},
        ],
        "text_sha256": hashlib.sha256("before <image> after".encode()).hexdigest(),
    }
    row["metadata"]["resource_registry"] = {
        "ark.idindex.mapping": {"IMG1": "image:natural-image"},
        "ark.idindex.image_urls": {"IMG1": path},
        "ark.idindex.reverse_mapping": {"image:natural-image": "IMG1"},
    }
    result = converter.convert(row, source_line=1)
    envelope = result["metadata"]["agentspark"]
    plan = result["metadata"]["agentspark_media"]
    assert len(plan["assets"]) == len(plan["replay_bindings"]) == 1
    assert plan["seed_refs"] == []
    marker = plan["replay_bindings"][0]["marker"]
    assert "before " + marker + " after" in envelope["prompt_spec"]["user_prompt"]
    assert plan["replay_bindings"][0]["source"] == plan["assets"][0]["ref"]
    assert envelope["replay_spec"]["messages"] == [{"role": "SYSTEM", "content": "explicit test system"}]
    assert path not in json.dumps(envelope)
    assert result["metadata"]["current_attachment_snapshot"] == row["metadata"]["current_attachment_snapshot"]


def test_current_and_history_repeated_image_occurrences_keep_distinct_bindings(
    converter: MilesPreparedConverter, tmp_path: Path
) -> None:
    row = _row()
    path = _image(tmp_path / "repeated.png")
    row["image_urls"] = row["metadata"]["image_urls"] = [path, path]
    row["metadata"]["replay_history"] = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": path}},
                {"type": "image_url", "image_url": {"url": path}},
            ],
        }
    ]
    result = converter.convert(row, source_line=1)
    plan = result["metadata"]["agentspark_media"]
    assert len(plan["assets"]) == 1
    assert len(plan["seed_refs"]) == len(plan["replay_bindings"]) == 2
    assert plan["seed_refs"][0] == plan["seed_refs"][1]
    assert plan["replay_bindings"][0]["marker"] != plan["replay_bindings"][1]["marker"]
    assert result["metadata"]["agentspark"]["inputs"][1]["info"] == [
        {"image_url": plan["seed_refs"][0]},
        {"image_url": plan["seed_refs"][1]},
    ]


def test_history_tool_image_keeps_tool_identity_for_safe_stub_restoration(
    converter: MilesPreparedConverter, tmp_path: Path
) -> None:
    row = _row()
    image = _image(tmp_path / "tool.png")
    row["metadata"]["replay_history"] = [
        {"role": "user", "content": "old question"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "opaque-B", "type": "function", "function": {"name": "note_read", "arguments": '{"id":"N1"}'}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "opaque-B",
            "name": "note_read",
            "content": [{"type": "image_url", "image_url": {"url": image}}],
        },
    ]
    result = converter.convert(row, source_line=1)
    binding = result["metadata"]["agentspark_media"]["replay_bindings"][0]
    assert binding["role"] == "tool"
    assert binding["tool_history"]["tool_call_id"] == "opaque-B"
    assert binding["tool_history"]["name"] == "note_read"
    assert binding["marker"] in binding["tool_history"]["content"]


def test_jsonl_stream_has_private_outputs_explicit_rejections_and_never_overwrites(
    converter: MilesPreparedConverter, tmp_path: Path
) -> None:
    # Use a fresh context because prepare_jsonl owns the compiler lifecycle.
    converter = MilesPreparedConverter(converter.config, allowed_roots=[tmp_path])
    source, output = tmp_path / "input.jsonl", tmp_path / "converted.jsonl"
    rejected, report_path = tmp_path / "rejected.jsonl", tmp_path / "report.json"
    good, bad = _row(), _row()
    bad["metadata"]["replay_history"] = [{"role": "user", "content": ""}]
    original = (json.dumps(good) + "\n" + json.dumps(bad) + "\n").encode()
    source.write_bytes(original)
    report = prepare_jsonl(source, output, converter=converter, rejection_path=rejected, report_path=report_path)
    assert (report["rows"], report["accepted"], report["rejected"]) == (2, 1, 1)
    assert source.read_bytes() == original
    entry = json.loads(rejected.read_text())
    assert entry["source_line"] == 2 and entry["code"] == "history_empty_content"
    assert "current request" not in rejected.read_text()
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in (output, rejected, report_path))
    assert "must-not-enter-artifact" not in report_path.read_text()
    with pytest.raises(PreparationError, match="^preparation_output_exists$"):
        prepare_jsonl(source, output, converter=converter, rejection_path=rejected, report_path=report_path)


def test_equal_duplicate_arguments_remain_an_explicit_source_error() -> None:
    with pytest.raises(PreparationError, match="^history_tool_arguments_duplicate_json_key_equal$"):
        _validate_history(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call",
                            "function": {
                                "name": "search_request",
                                "arguments": '{"queries":["same"],"queries":["same"]}',
                            },
                        }
                    ],
                }
            ]
        )
    assert _validate_history([{"role": "tool", "tool_call_id": "call", "content": ""}])[0]["content"] == ""


def test_remote_pixels_are_rejected_without_network_or_neighbor_image_substitution(
    converter: MilesPreparedConverter,
) -> None:
    row = _row()
    row["metadata"]["replay_history"] = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://images.example.test/missing.jpg"}}],
        }
    ]
    with pytest.raises(PreparationError, match="^unresolved_remote_media$"):
        converter.convert(row, source_line=7431)


def test_manifest_reuses_actual_hash_decode_evidence_and_missing_assets_fail_closed(
    converter: MilesPreparedConverter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.agentspark_agentic import media

    path = Path(_image(tmp_path / "verified.png"))
    raw = path.read_bytes()
    manifest = tmp_path / "verified.jsonl"
    entry = {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mtime_ns": path.stat().st_mtime_ns,
        "width": 2,
        "height": 3,
        "format": "PNG",
        "status": "verified",
    }
    manifest.write_text(json.dumps(entry) + "\n")
    instance = MilesPreparedConverter(converter.config, allowed_roots=[tmp_path], verified_media_manifest=manifest)

    def unexpected_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("explicit verified manifest must avoid another offline NFS read")

    monkeypatch.setattr(media, "_read_asset", unexpected_read)
    row = _row()
    row["metadata"]["image_urls"] = row["image_urls"] = [str(path)]
    with instance.source():
        first = instance.convert(row, source_line=1)
        second = instance.convert(_row(), source_line=2)
        assert len(first["metadata"]["agentspark_media"]["assets"]) == 1
        assert second["metadata"]["agentspark_media"]["assets"] == []
        row["metadata"]["image_urls"] = row["image_urls"] = [str(tmp_path / "not-in-catalog.png")]
        with pytest.raises(media.AdapterError, match="^media_manifest_source_missing$"):
            instance.convert(row, source_line=3)
    assert instance.media_evidence["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert instance.media_evidence["runtime_revalidation_required"] is True


def test_unverified_manifest_or_locator_hash_is_not_accepted(tmp_path: Path) -> None:
    path = tmp_path / "bad-manifest.jsonl"
    path.write_text(json.dumps({"status": "readable", "path": "/unverified.png"}) + "\n")
    with pytest.raises(PreparationError, match="^media_manifest_unverified_entry$"):
        load_verified_media_manifest(path)


def test_registry_image_references_and_avatar_do_not_trigger_offline_pixel_downloads(
    converter: MilesPreparedConverter, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.agentspark_agentic import media

    def no_pixels(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("registry references must not be mistaken for visible pixels")

    monkeypatch.setattr(media, "_read_asset", no_pixels)
    row = _row()
    row["metadata"]["replay_history"] = [{"role": "user", "content": "prior question"}]
    row["metadata"]["resource_registry"] = {
        "resources": {
            "N1": {
                "id": "N1",
                "type": "note",
                "raw_ids": {"note_id": "natural-note"},
                "image": [
                    {
                        "id": "N1.IMG1",
                        "type": "image",
                        "raw_ids": {"id": "natural-image"},
                        "url": "http://ci.xiaohongshu.com/registered-cover.jpg",
                    }
                ],
            },
            "U2": {
                "id": "U2",
                "type": "user",
                "raw_ids": {"id": "natural-user"},
                "avatar": "https://unlisted-avatar.example.test/avatar.jpg",
            },
        }
    }
    result = converter.convert(row, source_line=108)
    replay = result["metadata"]["agentspark"]["replay_spec"]["attributes"]
    assert replay["ark.idindex.image_urls"] == {"N1.IMG1": "https://ci.xiaohongshu.com/registered-cover.jpg"}
    assert replay["ark.idindex.mapping"]["N1.IMG1"] == "image:natural-image"
    assert result["metadata"]["agentspark_media"]["assets"] == []
    assert result["metadata"]["resource_registry"] == row["metadata"]["resource_registry"]
    assert result["metadata"]["agentspark_prepare"]["remote_registry_urls_are_references_not_verified_pixels"] is True
    row["metadata"]["resource_registry"]["resources"]["N1"]["image"][0][
        "url"
    ] = "https://not-allowed.example.test/image.jpg"
    with pytest.raises(PreparationError, match="^miles_source_compilation_failed$"):
        converter.convert(row, source_line=1)
