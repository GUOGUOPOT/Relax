# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local media byte identity, replay authority, and real callback transport tests."""

import base64
import copy
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from PIL import Image

from examples.agentspark_agentic.callback import CallbackRecorder
from examples.agentspark_agentic.client import run_audit
from examples.agentspark_agentic.contexts import analyze_audit
from examples.agentspark_agentic.media import ASSET_PREFIX, AssetStore, MediaPlan, canonical_bytes
from examples.agentspark_agentic.protocol import AdapterConfig, AdapterError, RelaxContext
from tests.cpu.models.test_kimi_k3_hf import official_k3  # noqa: F401


@pytest.fixture
def assets(tmp_path):
    paths = [tmp_path / "red.png", tmp_path / "blue.png"]
    for path, color in zip(paths, ("red", "blue")):
        Image.new("RGB", (3, 2), color).save(path)
    store = AssetStore([tmp_path])
    refs = [store.freeze(str(path)) for path in paths]
    return store, refs, paths


def marker(index):
    return "[[MILES_REPLAY_IMAGE:" + "a" * 32 + f":{index:04d}]]"


def history_plan(assets):
    store, refs, paths = assets
    bindings = [{"marker": marker(i), "source": ref, "role": "user"} for i, ref in enumerate(refs)]
    media = MediaPlan(store.build(bindings, []), [paths[0].parent])
    media.public_base_url = "http://127.0.0.1:12345/media-prefix"
    return media


def images(payload):
    return [
        part["image_url"]["url"]
        for message in payload["messages"]
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]


def test_freeze_identity_roots_catalog_and_runtime_change(assets, tmp_path):
    store, refs, paths = assets
    manifest = store.manifest()
    for key, invalid in (("path", []), ("mime_type", "image/svg+xml")):
        bad = copy.deepcopy(manifest)
        bad[0][key] = invalid
        with pytest.raises(AdapterError, match="media_asset_invalid"):
            AssetStore([tmp_path], verified_manifest=bad)
    catalog = AssetStore([tmp_path], verified_manifest=manifest)
    assert catalog.freeze(str(paths[0])) == refs[0]
    assert catalog.manifest() == manifest[:1]
    with pytest.raises(AdapterError, match="media_manifest_source_missing"):
        catalog.freeze(str(tmp_path / "missing.png"))
    with pytest.raises(AdapterError, match="media_local_source_required"):
        store.freeze("https://example.invalid/image.png")
    outside = tmp_path / "separate"
    outside.mkdir()
    with pytest.raises(AdapterError, match="media_source_outside_roots"):
        AssetStore([outside]).freeze(str(paths[0]))
    link = outside / "escape.png"
    link.symlink_to(paths[0])
    with pytest.raises(AdapterError, match="media_source_outside_roots"):
        AssetStore([outside]).freeze(str(link))
    media = MediaPlan(store.build([], refs), [tmp_path])
    before = media.data_uri(refs[0])
    Image.new("RGB", (3, 2), "green").save(paths[0])
    assert media.data_uri(refs[0]) == before  # bytes frozen for this invocation
    with pytest.raises(AdapterError, match="media_asset_identity_changed"):
        MediaPlan(store.build([], refs), [tmp_path])


def test_occurrence_moves_duplicates_deletions_and_branches_preserve_byte_order(assets):
    media = history_plan(assets)
    messages = [
        {"role": "user", "content": "history:" + marker(0) + marker(1)},
        {"role": "assistant", "content": "prior answer"},
    ]
    original = copy.deepcopy(messages)
    result, evidence = media.restore({"messages": messages})
    assert messages == original
    expected = [media.data_uri(ref) for ref in assets[1]]
    assert images(result) == expected
    assert [item["sha256"] for item in evidence["occurrences"]] == [ref[len(ASSET_PREFIX) :] for ref in assets[1]]
    changed = [{"role": "user", "content": marker(1) + "history:" + marker(0) + marker(1)}]
    result, _ = media.restore({"messages": changed})
    assert images(result) == [expected[1], expected[0], expected[1]]
    result, _ = media.restore({"messages": [{"role": "user", "content": "history:"}]})
    assert not images(result)  # no restoration from an earlier segment
    result, _ = media.restore({"messages": [{"role": "system", "content": "new branch"}] + original})
    assert images(result) == expected
    # Assistant text and tool arguments cannot authorize an image occurrence.
    result, _ = media.restore(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": marker(0),
                    "tool_calls": [
                        {
                            "id": "new",
                            "type": "function",
                            "function": {"name": "Search", "arguments": json.dumps({"q": marker(1)})},
                        }
                    ],
                }
            ]
        }
    )
    assert not images(result)
    with pytest.raises(AdapterError, match="media_marker_message_changed"):
        media.restore({"messages": [{"role": "user", "content": "new request:" + marker(0)}]})


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("duplicate", "media_marker_count_mismatch"),
        ("role", "media_marker_role_mismatch"),
        ("missing", "media_marker_missing"),
        ("outside", "media_marker_count_mismatch"),
    ],
)
def test_first_marker_authority_is_complete_unique_and_role_bound(assets, mutation, code):
    media = history_plan(assets)
    message = {"role": "user", "content": marker(0) + marker(1)}
    if mutation == "duplicate":
        message["content"] += marker(0)
    elif mutation == "role":
        message["role"] = "assistant"
    elif mutation == "missing":
        message["content"] = marker(0)
    else:
        message["tool_calls"] = [{"function": {"arguments": marker(0)}}]
    with pytest.raises(AdapterError, match=code):
        media.restore({"messages": [message]})
    assert not media.initialized


def test_tool_compression_requires_original_id_name_arguments_and_message(assets):
    store, refs, paths = assets
    history = {
        "tool_call_id": "call",
        "name": "Search",
        "arguments_sha256": hashlib.sha256(canonical_bytes({"q": 1})).hexdigest(),
        "content": "found " + marker(0),
    }
    plan = store.build([{"marker": marker(0), "source": refs[0], "role": "tool", "tool_history": history}], [])
    media = MediaPlan(plan, [paths[0].parent])
    media.public_base_url = "http://127.0.0.1:12345"
    call = {
        "role": "assistant",
        "tool_calls": [{"id": "call", "type": "function", "function": {"name": "Search", "arguments": '{"q": 1}'}}],
    }
    stub = {"role": "tool", "tool_call_id": "call", "content": "[调用了 Search 工具]"}
    payload = {"messages": [call, stub]}
    result, _ = media.restore(payload)
    assert images(result) == [media.data_uri(refs[0])]
    assert payload["messages"][1] == stub
    result, _ = media.restore(payload)
    assert images(result) == [media.data_uri(refs[0])]
    wrong = copy.deepcopy(payload)
    wrong["messages"][0]["tool_calls"][0]["function"]["arguments"] = '{"q": 2}'
    with pytest.raises(AdapterError, match="media_tool_identity_mismatch"):
        media.restore(wrong)
    wrong = copy.deepcopy(payload)
    wrong["messages"][1]["name"] = "Search"
    with pytest.raises(AdapterError, match="media_tool_message_changed"):
        media.restore(wrong)
    # Removing the old image-bearing tool segment is a legitimate new context.
    result, _ = media.restore({"messages": [{"role": "user", "content": "reset"}]})
    assert not images(result)


def config_for(roots):
    return AdapterConfig(
        "http://agentspark.invalid/prefix",
        "fixture-model",
        "diandian",
        "eden",
        32,
        "127.0.0.1",
        0,
        "http://127.0.0.1:0/callback",
        media_allowed_roots=tuple(map(str, roots)),
    )


def reply():
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]}
    )


@pytest.mark.asyncio
async def test_callback_asset_route_frozen_bytes_forwarded_identity_and_no_private_send(assets):
    store, refs, paths = assets
    context = RelaxContext("http://relax.invalid/prefix", "private-session", "group", "train", "/run/1")
    config = config_for([paths[0].parent])
    plan = store.build([{"marker": marker(0), "source": refs[0], "role": "user"}], [refs[1]])
    source = {
        "metadata": {
            "agentspark_media": plan,
            "agentspark": {
                "schema_version": 1,
                "inputs": [
                    {"type": "TEXT", "info": {"text": "question"}},
                    {"type": "IMAGE", "info": [{"image_url": refs[1]}]},
                ],
                "tools": ["Search"],
                "prompt_spec": {"system_prompt": "system", "user_prompt": "user"},
            },
        }
    }
    forwarded, sends, raw_requests, asset_urls = [], [], [], []

    async def backend(request):
        forwarded.append(json.loads(request.content))
        return reply()

    async def remote(request):
        if request.method == "GET":
            return httpx.Response(200, json={"session_id": "remote"})
        body = json.loads(request.content)
        sends.append(body)
        assert "agentspark_media" not in body and "replay_bindings" not in body
        assert ASSET_PREFIX not in json.dumps(body)
        url = body["inputs"][1]["info"][0]["image_url"]
        asset_urls.append(url)
        callback = body["llm_config"]["base_url"] + body["llm_config"]["endpoint_path"]
        payload = {
            "messages": [
                {"role": "user", "content": "history " + marker(0)},
                {"role": "assistant", "content": "previous"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "question"},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                },
            ]
        }
        raw_requests.append(copy.deepcopy(payload))
        async with httpx.AsyncClient(trust_env=False) as caller:
            image = await caller.get(url)
            assert image.content == paths[1].read_bytes()
            assert image.headers["content-type"] == "image/png"
            assert (await caller.get(url.rsplit("/", 1)[0] + "/unknown")).status_code == 404
            response = await caller.post(
                callback, json=payload, headers={"Authorization": "Bearer " + context.session_id}
            )
            assert response.status_code == 200
        return httpx.Response(200, content=b'data: {"type":"agent.stream.end"}\n\n')

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as relax_http:
        async with httpx.AsyncClient(transport=httpx.MockTransport(remote)) as remote_http:
            report = await run_audit(config, context, source, relax_http=relax_http, agentspark_http=remote_http)
    assert report["audit_status"] == "collected" and report["training_ready"] is False, (
        report.get("failure_code"),
        report["callbacks"]["errors"],
    )
    record = report["callbacks"]["records"][0]
    assert record["raw_request"] == raw_requests[0]
    assert record["request"] == forwarded[0] and record["request"] != record["raw_request"]
    assert record["request_sha256"] == hashlib.sha256(canonical_bytes(forwarded[0])).hexdigest()
    assert record["forwarded_request_canonical_sha256"] == record["request_sha256"]
    assert [base64.b64decode(url.split(",", 1)[1]) for url in images(forwarded[0])] == [
        path.read_bytes() for path in paths
    ]
    assert len(record["media"]["occurrences"]) == 2
    contexts = analyze_audit(
        report, compiler_defaults={}, selection=[{"name": "main", "record_index": 0, "intent": "actual forwarded"}]
    )
    assert images({"messages": contexts["records"][0]["messages"]}) == images(forwarded[0])
    tampered = copy.deepcopy(report)
    tampered["callbacks"]["records"][0]["request"] = raw_requests[0]
    with pytest.raises(AdapterError, match="context_forwarded_request_identity_invalid"):
        analyze_audit(tampered, compiler_defaults={})
    async with httpx.AsyncClient(trust_env=False) as caller:
        with pytest.raises(httpx.ConnectError):
            await caller.get(asset_urls[0])


@pytest.mark.asyncio
async def test_wrong_seed_order_and_unknown_images_never_reach_backend(assets):
    store, refs, paths = assets
    media = MediaPlan(store.build([], refs), [paths[0].parent])
    context = RelaxContext("http://relax.invalid", "private-session", "group", "train", "/run/2")
    forwarded = []

    async def backend(request):
        forwarded.append(request)
        return reply()

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
        async with CallbackRecorder(config_for([paths[0].parent]), context, client=client, media=media) as recorder:
            async with httpx.AsyncClient(trust_env=False) as caller:
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": media.public_url(ref)}}
                                for ref in reversed(refs)
                            ],
                        }
                    ]
                }
                response = await caller.post(
                    recorder.public_base_url + "/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": "Bearer " + context.session_id},
                )
                assert response.status_code == 502
    assert not forwarded and recorder.capture.errors == ["media_seed_images_mismatch"]


@pytest.mark.asyncio
async def test_forwarded_base64_budget_and_duplicate_json_are_rejected(assets):
    store, refs, paths = assets
    media = history_plan(assets)
    context = RelaxContext("http://relax.invalid", "private-session", "group", "train", "/run/3")
    forwarded = []

    async def backend(request):
        forwarded.append(request)
        return reply()

    payload = {"messages": [{"role": "user", "content": marker(0) + marker(1)}]}
    raw = canonical_bytes(payload)
    config = replace(config_for([paths[0].parent]), max_callback_bytes=len(raw) + 30)
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
        async with CallbackRecorder(config, context, client=client, media=media) as recorder:
            async with httpx.AsyncClient(trust_env=False) as caller:
                headers = {"Authorization": "Bearer " + context.session_id}
                response = await caller.post(
                    recorder.public_base_url + "/v1/chat/completions", content=raw, headers=headers
                )
                assert response.status_code == 502
                response = await caller.post(
                    recorder.public_base_url + "/v1/chat/completions",
                    content=b'{"messages":[],"messages":[]}',
                    headers=headers,
                )
                assert response.status_code == 400
    assert not forwarded
    assert recorder.capture.errors == ["callback_forwarded_request_too_large", "callback_request_invalid"]


def test_tool_hash_matches_source_algorithm_and_rejects_duplicate_keys(assets):
    store, refs, paths = assets
    arguments = {"n": 100.0}
    history = {
        "tool_call_id": "call",
        "name": "Search",
        "arguments_sha256": hashlib.sha256(canonical_bytes(arguments)).hexdigest(),
        "content": marker(0),
    }
    media = MediaPlan(
        store.build([{"marker": marker(0), "source": refs[0], "role": "tool", "tool_history": history}], []),
        [paths[0].parent],
    )
    media.public_base_url = "http://127.0.0.1:12345"
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call", "type": "function", "function": {"name": "Search", "arguments": '{"n":1e2}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call", "content": "[调用了 Search 工具]"},
        ]
    }
    actual, _ = media.restore(payload)
    assert len(images(actual)) == 1
    payload["messages"][0]["tool_calls"][0]["function"]["arguments"] = '{"n":0,"n":1e2}'
    with pytest.raises(AdapterError, match="media_tool_identity_mismatch"):
        media.restore(payload)
    # This is the source binding hash, intentionally separate from Relax's
    # native integer/float canonical context identity.
    payload["messages"][0]["tool_calls"][0]["function"]["arguments"] = '{"n":100}'
    with pytest.raises(AdapterError, match="media_tool_identity_mismatch"):
        media.restore(payload)


def dynamic_plan():
    return MediaPlan(
        {"schema_version": 1, "assets": [], "replay_bindings": [], "seed_refs": []},
        [],
        allow_dynamic_images=True,
    )


def image_part(uri):
    return {"type": "image_url", "image_url": {"url": uri}}


def image_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def test_dynamic_admission_is_opt_in_and_commits_only_successful_candidate(assets):
    uri = image_uri(assets[2][0])
    payload = {"messages": [{"role": "tool", "tool_call_id": "call", "content": [image_part(uri)]}]}
    disabled = MediaPlan({"schema_version": 1, "assets": [], "replay_bindings": [], "seed_refs": []}, [])
    with pytest.raises(AdapterError, match="media_unregistered_image"):
        disabled.restore(payload)
    media = dynamic_plan()
    failed = media.fork()
    failed.restore(payload)
    assert not media.initialized and not media._dynamic_assets
    candidate = media.fork()
    actual, evidence = candidate.restore(payload)
    assert actual == payload and images(actual) == [uri]
    assert evidence["occurrences"][0]["sha256"] == hashlib.sha256(assets[2][0].read_bytes()).hexdigest()
    assert uri not in json.dumps(evidence) and "data:image" not in json.dumps(evidence)
    media.commit(candidate, actual, {"role": "assistant", "content": "done"})
    assert media.initialized
    with pytest.raises(AdapterError, match="media_candidate_stale_or_unrelated"):
        media.commit(failed, payload, {"role": "assistant", "content": "failed"})


def test_dynamic_repeated_occurrences_overflow_and_placeholder_history(assets):
    red, blue = map(image_uri, assets[2])
    media = dynamic_plan()
    payload = {"messages": [{"role": "tool", "tool_call_id": "call", "content": [image_part(v) for v in [red, blue] * 16]}]}
    candidate = media.fork()
    actual, evidence = candidate.restore(payload)
    assert images(actual) == [red, blue] * 15
    assert actual["messages"][0]["content"][-2:] == [{"type": "text", "text": "[图片因视觉上限省略]"}] * 2
    assert evidence["dynamic_images"]["omitted_occurrences"] == 2
    assistant = {"role": "assistant", "content": "done"}
    media.commit(candidate, actual, assistant)
    replay = copy.deepcopy(payload)
    replay["messages"][0]["content"] = [{"type": "text", "text": "[图片]"}] * 32
    replay["messages"] += [assistant, {"role": "user", "content": [image_part(blue)]}]
    candidate = media.fork()
    restored, evidence = candidate.restore(replay)
    assert restored["messages"][:2] == actual["messages"] + [assistant]
    assert images(restored) == [red, blue] * 15
    assert restored["messages"][-1]["content"] == [{"type": "text", "text": "[图片因视觉上限省略]"}]
    assert evidence["dynamic_images"]["replayed_occurrences"] == 30


def test_dynamic_new_segment_recomputes_budget_from_raw_context(assets):
    from examples.agentspark_agentic.history import ReasoningHistory

    red, blue = map(image_uri, assets[2])
    media, history = dynamic_plan(), ReasoningHistory()
    first = {"messages": [{"role": "user", "content": [image_part(red)] * 31}]}
    candidate = media.fork()
    actual, _ = candidate.restore(first)
    actual, _ = history.restore(actual)
    answer = {"role": "assistant", "content": "done"}
    history.commit(actual, answer)
    media.commit(candidate, actual, answer)
    raw = {"messages": [{"role": "user", "content": [{"type": "text", "text": "new branch"}, image_part(blue)]}]}
    candidate = media.fork()
    unaligned, evidence = candidate.restore(raw)
    assert evidence["dynamic_images"]["history_aligned"] is False
    _, history_evidence = history.restore(unaligned)
    assert history_evidence["status"] == "new_segment"
    reset = media.fork(reset_segment=True)
    forwarded, evidence = reset.restore(raw)
    forwarded, history_evidence = history.restore(forwarded)
    assert history_evidence["status"] == "new_segment"
    assert images(forwarded) == [blue] and evidence["dynamic_images"]["segment_reset"]
    history.commit(forwarded, answer)
    media.commit(reset, forwarded, answer)
    replay = {"messages": [*forwarded["messages"], answer, {"role": "user", "content": [image_part(red)] * 30}]}
    final, _ = media.fork().restore(replay)
    assert images(final) == [blue] + [red] * 29


@pytest.mark.parametrize("source", ["https://example.invalid/new.png", "http://127.0.0.1/new.png", "file:///tmp/new.png", "/tmp/new.png"])
def test_dynamic_remote_and_local_paths_never_gain_authority(source):
    with pytest.raises(AdapterError, match="media_dynamic_image_scheme_forbidden"):
        dynamic_plan().fork().restore({"messages": [{"role": "tool", "content": [image_part(source)]}]})


def test_dynamic_decode_limits_shapes_and_forwarded_identity(assets, monkeypatch):
    from examples.agentspark_agentic import media as module

    uri = image_uri(assets[2][0])
    monkeypatch.setattr(module, "MAX_IMAGE_BYTES", 8)
    with pytest.raises(AdapterError, match="media_dynamic_image_too_large"):
        dynamic_plan().fork().restore({"messages": [{"role": "tool", "content": [image_part(uri)]}]})
    monkeypatch.setattr(module, "MAX_IMAGE_BYTES", 16 * 1024 * 1024)
    for invalid in ("data:image/png;base64,%%%", "data:image/png;base64," + base64.b64encode(b"not pixels").decode()):
        with pytest.raises(AdapterError, match="media_dynamic_image_data_invalid"):
            dynamic_plan().fork().restore({"messages": [{"role": "tool", "content": [image_part(invalid)]}]})
    with pytest.raises(AdapterError, match="media_dynamic_image_mime_mismatch"):
        dynamic_plan().fork().restore({"messages": [{"role": "tool", "content": [image_part(uri.replace("image/png", "image/jpeg"))]}]})
    with pytest.raises(AdapterError, match="media_dynamic_image_block_invalid"):
        dynamic_plan().fork().restore({"messages": [{"role": "tool", "content": [{**image_part(uri), "image": uri}]}]})
    media = dynamic_plan()
    candidate = media.fork()
    actual, _ = candidate.restore({"messages": [{"role": "user", "content": [image_part(uri)]}]})
    with pytest.raises(AdapterError, match="media_forwarded_images_changed"):
        media.commit(candidate, {"messages": []}, {"role": "assistant", "content": "changed"})
    with pytest.raises(AdapterError, match="media_assistant_image_response_forbidden"):
        media.commit(candidate, actual, {"role": "assistant", "content": [image_part(uri)]})


def test_prepared_images_keep_reserved_slots_when_dynamic_images_overflow(assets):
    store, refs, paths = assets
    media = MediaPlan(store.build([], refs), [paths[0].parent], allow_dynamic_images=True)
    media.public_base_url = "http://127.0.0.1:12345"
    from PIL import Image

    green = paths[0].parent / "green.png"
    Image.new("RGB", (3, 2), "green").save(green)
    dynamic = image_uri(green)
    payload = {"messages": [
        {"role": "tool", "content": [image_part(dynamic)] * 30},
        {"role": "user", "content": [image_part(media.public_url(ref)) for ref in refs]},
    ]}
    actual, evidence = media.fork().restore(payload)
    assert images(actual) == [dynamic] * 28 + [media.data_uri(ref) for ref in refs]
    assert evidence["dynamic_images"]["omitted_occurrences"] == 2


def test_same_pixels_as_seed_still_consume_new_occurrence_slots(assets):
    store, refs, paths = assets
    media = MediaPlan(store.build([], refs[:1]), [paths[0].parent], allow_dynamic_images=True)
    media.public_base_url = "http://127.0.0.1:12345"
    uri = media.data_uri(refs[0])
    actual, evidence = media.fork().restore({"messages": [
        {"role": "user", "content": [image_part(media.public_url(refs[0]))]},
        {"role": "tool", "content": [image_part(uri)] * 30},
    ]})
    assert images(actual) == [uri] * 30
    assert actual["messages"][-1]["content"][-1] == {"type": "text", "text": "[图片因视觉上限省略]"}
    assert evidence["dynamic_images"]["admitted_total"] == 30
    assert evidence["dynamic_images"]["admitted_occurrences"] == 29


def test_overflow_images_are_not_decoded_or_cached(assets, monkeypatch):
    from examples.agentspark_agentic import media as module

    red, blue = map(image_uri, assets[2])
    decode = module._dynamic_image
    decoded = []

    def counted(source):
        decoded.append(hashlib.sha256(source.encode()).hexdigest())
        return decode(source)

    monkeypatch.setattr(module, "_dynamic_image", counted)
    monkeypatch.setattr(module, "MAX_IMAGE_BYTES", 100)
    overflow = [blue, "data:image/png;base64,%%%", "data:image/png;base64," + "A" * 1000]
    candidate = dynamic_plan().fork()
    actual, evidence = candidate.restore({"messages": [
        {"role": "tool", "content": [image_part(red)] * 30 + [image_part(value) for value in overflow]}
    ]})
    assert images(actual) == [red] * 30
    assert decoded == [hashlib.sha256(red.encode()).hexdigest()]
    assert list(candidate._dynamic_assets) == [red]
    assert evidence["dynamic_images"]["omitted_occurrences"] == 3


@pytest.mark.parametrize("compressed_tool", [False, True])
def test_prepared_replay_is_expanded_before_dynamic_prefix_restoration(assets, compressed_tool):
    from examples.agentspark_agentic.history import ReasoningHistory

    store, refs, paths = assets
    binding = {"marker": marker(0), "source": refs[0], "role": "user"}
    prefix = [{"role": "user", "content": "question " + marker(0)}]
    if compressed_tool:
        binding.update(role="tool", tool_history={
            "tool_call_id": "call", "name": "Search",
            "arguments_sha256": hashlib.sha256(canonical_bytes({})).hexdigest(),
            "content": "found " + marker(0),
        })
        prefix = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call", "type": "function", "function": {"name": "Search", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call", "content": "[调用了 Search 工具]"},
        ]
    media = MediaPlan(store.build([binding], []), [paths[0].parent], allow_dynamic_images=True)
    media.public_base_url = "http://127.0.0.1:12345"
    history = ReasoningHistory()
    candidate = media.fork()
    first, _ = candidate.restore({"messages": prefix})
    first, _ = history.restore(first)
    answer = {"role": "assistant", "content": "inspect", "tool_calls": [
        {"id": "new-call", "type": "function", "function": {"name": "ViewImage", "arguments": "{}"}}
    ]}
    history.commit(first, answer)
    media.commit(candidate, first, answer)
    # AgentSpark repeats its original marker/stub, not the image-expanded payload.
    candidate = media.fork()
    actual, evidence = candidate.restore({"messages": [
        *prefix, answer, {"role": "tool", "tool_call_id": "new-call", "content": [image_part(image_uri(paths[1]))]}
    ]})
    actual, history_evidence = history.restore(actual)
    assert history_evidence["status"] == "reconciled"
    assert evidence["dynamic_images"]["history_aligned"] is True
    assert images(actual) == [media.data_uri(refs[0]), image_uri(paths[1])]


def test_seed_snapshot_survives_segment_reset_and_cannot_change_message_identity(assets):
    store, refs, paths = assets
    media = MediaPlan(store.build([], refs), [paths[0].parent], allow_dynamic_images=True)
    media.public_base_url = "http://127.0.0.1:12345"
    seed = {"role": "user", "content": [
        {"type": "text", "text": "seed question"}, *[image_part(media.public_url(ref)) for ref in refs]
    ]}
    candidate = media.fork()
    first, _ = candidate.restore({"messages": [seed]})
    media.commit(candidate, first, {"role": "assistant", "content": "done"})
    raw = {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "seed question"}, {"type": "text", "text": "[图片]"}]},
        {"role": "tool", "content": [image_part(image_uri(paths[0]))] * 30},
    ]}
    actual, evidence = media.fork(reset_segment=True).restore(raw)
    assert images(actual) == [media.data_uri(ref) for ref in refs] + [image_uri(paths[0])] * 28
    assert evidence["dynamic_images"]["admitted_total"] == 30
    for changed in (
        {"messages": raw["messages"][1:]},
        {"messages": [{"role": "system", "content": "new"}] + raw["messages"]},
    ):
        with pytest.raises(AdapterError, match="media_seed_message_changed"):
            media.fork(reset_segment=True).restore(changed)


def test_dynamic_occurrences_and_replay_match_frozen_harness(assets, monkeypatch):
    source = os.environ.get("AGENTSPARK_V21_ORACLE_SOURCE")
    if not source and os.environ.get("AGENTSPARK_V21_CONTRACT"):
        source = json.loads(Path(os.environ["AGENTSPARK_V21_CONTRACT"]).read_text())["miles_source"]
    if not source:
        pytest.skip("Explicit frozen V21 source is required for the image admission oracle")
    monkeypatch.syspath_prepend(source)
    path = Path(source) / "miles/agentspark_session/image_bindings.py"
    spec = importlib.util.spec_from_file_location("_agentspark_frozen_image_admission", path)
    oracle = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, oracle)
    spec.loader.exec_module(oracle)
    red, blue = map(image_uri, assets[2])
    media, state = dynamic_plan(), oracle.ImageBindingState(bindings=())
    raw = [{"role": "user", "content": "question"}, {"role": "tool", "content": [image_part(v) for v in [red, blue] * 16]}]
    expected, stats = oracle.apply_image_admission(raw, state, stored_messages=())
    candidate = media.fork()
    actual, evidence = candidate.restore({"messages": raw})
    assert actual["messages"] == expected
    assert evidence["dynamic_images"]["omitted_occurrences"] == stats.omitted_images == 2
    answer = {"role": "assistant", "content": "done"}
    media.commit(candidate, actual, answer)
    stored = expected + [answer]
    replay = copy.deepcopy(raw) + [answer, {"role": "tool", "content": [image_part(red)]}]
    replay[1]["content"] = [{"type": "text", "text": "[图片]"}] * 32
    expected, _ = oracle.apply_image_admission(replay, state, stored_messages=stored)
    actual, _ = media.fork().restore({"messages": replay})
    assert actual["messages"] == expected
    changed = [{"role": "user", "content": "fresh branch"}, {"role": "tool", "content": [image_part(blue)] * 2}]
    expected, _ = oracle.apply_image_admission(changed, state, stored_messages=stored)
    actual, _ = media.fork().restore({"messages": changed})
    assert actual["messages"] == expected
    oracle.reset_segment_image_admission(state)
    expected, _ = oracle.apply_image_admission(changed, state, stored_messages=())
    actual, _ = media.fork(reset_segment=True).restore({"messages": changed})
    assert actual["messages"] == expected


def test_real_k3_processor_accepts_admitted_tool_image_with_exact_occurrence_order(assets, official_k3):
    import io

    from relax.utils.data.kimi_k3 import compile_kimi_k3_agentic_messages

    red, blue = map(image_uri, assets[2])
    payload = {"messages": [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call", "type": "function", "function": {"name": "ViewImage", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "call", "content": [image_part(red), image_part(blue), image_part(red)]},
    ]}
    media = dynamic_plan()
    actual, _ = media.fork().restore(payload)
    pixels = [Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGB") for value in images(actual)]
    train_ids, backend_ids, features = compile_kimi_k3_agentic_messages(
        official_k3.tokenizer,
        official_k3.processor,
        actual["messages"],
        tools=None,
        chat_template_kwargs={"thinking": True},
        multimodal_inputs={"images": pixels},
    )
    assert len(features["image_grid_thw"]) == 3
    assert backend_ids.count(official_k3.tokenizer.convert_tokens_to_ids("<|media_pad|>")) == 3
    assert len(train_ids) > len(backend_ids)
    assert pixels[0].tobytes() == pixels[2].tobytes() != pixels[1].tobytes()
