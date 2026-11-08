# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Native state/export acceptance with real K3 encoding and a controlled backend.

The backend emits fixed template-produced tokens/logprobs. These tests validate
committed state and training export, not real model generation or GPU execution.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from examples.agentspark_agentic.contexts import analyze_audit, validate_native_exports
from examples.agentspark_agentic.media import AssetStore, MediaPlan, canonical_bytes
from examples.agentspark_agentic.production import select_training_exports
from examples.agentspark_agentic.protocol import AdapterError
from relax.agentic.pipeline import TrainingFieldArtifact
from relax.agentic.pipeline.runtime import BackendGenerateResult, SGLangMessageCompiler
from relax.agentic.profile import TRACE_KEY
from relax.agentic.runner import SessionOutput
from relax.agentic.runner.ipc import _session_output_from_text
from relax.agentic.session.service import (
    AgenticSessionShard,
    ResidentGroup,
    _NonFinalizableExportError,
    _normalized_chat_request,
    _SessionRecord,
    _SessionResultCell,
)
from relax.agentic.session.state import SessionForest
from tests.cpu.models.test_kimi_k3_hf import official_k3  # noqa: F401


def _image_url(size: tuple[int, int], color: tuple[int, ...]) -> str:
    image = Image.new("RGBA" if len(color) == 4 else "RGB", size, color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _user(text: str, *images: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}]
        + [{"type": "image_url", "image_url": {"url": image}} for image in images],
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class _NativeDriver:
    """Only backend scheduling is controlled; all commits/export use native code."""

    invocation_id = "native-export-fixture-invocation"
    compiler_defaults = {"thinking": False}

    def __init__(
        self, case: Any, executor: ThreadPoolExecutor, *, session_id: str = "native-fixture-session", index: int = 17
    ) -> None:
        self.case = case
        self.records: list[dict[str, Any]] = []
        self.response_hashes: list[str] = []
        self.generated: dict[str, tuple[list[int], list[float]]] = {}
        self.shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
        self.shard.args = Namespace(
            agentic_reasoning_parser="kimi_k3",
            agentic_tool_call_parser="kimi_k3",
            rollout_max_context_len=4096,
            eval_max_context_len=4096,
            partial_rollout=False,
            use_rollout_routing_replay=False,
        )
        compiler = SGLangMessageCompiler(
            tokenizer=case.tokenizer,
            processor=case.processor,
            cpu_executor=executor,
            apply_chat_template_kwargs=dict(self.compiler_defaults),
        )
        self.shard._generation_backend = SimpleNamespace(compiler=compiler, tokenizer=case.tokenizer)
        self.shard._state_revision = 0
        self.shard._state_changed = asyncio.Event()
        group = ResidentGroup(rollout_mode="train", group_id="native-fixture-group", result_cells={}, sessions=[])
        self.session = _SessionRecord(
            group=group,
            session_id=session_id,
            session_sampling_params={"max_new_tokens": 64},
            result_cell=_SessionResultCell(),
        )
        group.sessions.append(self.session)
        group.result_cells[self.session.session_id] = self.session.result_cell
        self.session.forest = SessionForest.create_empty(
            session_id=self.session.session_id,
            index=index,
            group_index=3,
            train_metadata={"loss_owner": "source-policy"},
            metadata={"source_tag": "native-fixture"},
        )
        self.shard._dispatch_queued_irs_locked = self._controlled_backend
        seed = [{"role": "user", "content": "fixture prompt"}]
        prompt = case.tokenizer.apply_chat_template(seed, tokenize=True, thinking=False, add_generation_prompt=True)
        complete = case.tokenizer.apply_chat_template(
            seed + [{"role": "assistant", "content": "fixed backend answer"}],
            tokenize=True,
            thinking=False,
            add_generation_prompt=False,
        )
        assert complete[: len(prompt)] == prompt
        self.response_ids = list(complete[len(prompt) :])
        self.response_logprobs = [-0.25 - index / 1000 for index in range(len(self.response_ids))]

    def _controlled_backend(self, session: _SessionRecord) -> None:
        while session.queued_irs:
            ir = session.queued_irs.popleft()
            result = BackendGenerateResult(
                new_tokens=list(self.response_ids),
                new_log_probs=list(self.response_logprobs),
                finish_type="stop",
                meta_info={"prompt_tokens": len(ir.history_rollout_token_prefix)},
                elapsed=0.001,
            )
            self.shard._apply_generate_result(ir, result)
            ir.pending_status = "completed"
            payload = self.shard._terminal_response_locked(session=session, ir=ir, finish_type="stop")
            self.generated[ir.request_id] = (list(ir.pending_token_delta), list(ir.pending_logprob_delta))
            session.live_irs.remove(ir)
            ir.waiter.set_result(payload)

    async def commit(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        request = {"model": "local-k3-fixture", "messages": copy.deepcopy(messages)}
        payload = await self.shard._chat(session=self.session, **_normalized_chat_request(request))
        assert "error" not in payload
        assistant = payload["message"]
        wire_response = {"choices": [{"message": assistant, "finish_reason": payload["finish_reason"]}]}
        self.records.append(
            {
                "index": len(self.records),
                "request": request,
                "assistant": copy.deepcopy(assistant),
                "status": 200,
                "request_sha256": _digest(request),
                "response_sha256": _digest(wire_response),
            }
        )
        self.response_hashes.append(self.session.resp_state_hash_by_request_id[payload["request_id"]])
        return assistant

    def report(self, indices: list[int]) -> dict[str, Any]:
        audit = {
            "schema_version": 1,
            "stage": 2,
            "training_ready": False,
            "audit_status": "failed" if len(self.records) > 1 else "collected",
            "failure_code": "callback_history_not_linear" if len(self.records) > 1 else None,
            "identity": {"group_id": self.session.group.group_id, "mode": "train", "model": "local-k3-fixture"},
            "invocation_sha256": hashlib.sha256(
                f"{self.session.session_id}\0{self.invocation_id}".encode()
            ).hexdigest(),
            "source_input": {
                "metadata": {
                    "source_tag": "native-fixture",
                    "agentspark": {
                        "schema_version": 1,
                        "inputs": [{"type": "TEXT", "info": {"text": "Inspect the tool images."}}],
                        "tools": ["inspect"],
                        "prompt_spec": {"system_prompt": "Inspect images.", "user_prompt": "Describe their colors."},
                    },
                }
            },
            "send_attempts": 1,
            "remote_execution": "completed",
            "stream": {"saw_end": True, "saw_error": False, "eof": True},
            "cleanup": {"local": "closed"},
            "callbacks": {"started": len(self.records), "concurrent": False, "errors": [], "records": self.records},
        }
        return analyze_audit(
            audit,
            compiler_defaults=dict(self.compiler_defaults),
            selection=[
                {"name": f"context-{index}", "record_index": index, "intent": "inspect committed branch"}
                for index in indices
            ],
        )

    def native_export(self, records: list[dict[str, Any]]):
        output = SessionOutput.from_records(records)
        return self.shard._build_session_export_transport(
            session=self.session,
            reward=output.reward,
            metadata=output.metadata,
            output_records=output.records,
            finalize_events={},
        )


@pytest.fixture
def native(official_k3):
    with ThreadPoolExecutor(max_workers=2) as executor:
        yield _NativeDriver(official_k3, executor)


async def _branches(native: _NativeDriver) -> None:
    red = _image_url((43, 29), (255, 0, 0, 128))
    blue = _image_url((84, 56), (0, 0, 255))
    green = _image_url((56, 28), (0, 255, 0))
    initial = _user("Inspect these images.", red)
    first = await native.commit([initial])
    await native.commit([initial, first, _user("Add this image.", blue)])
    await native.commit([_user("Inspect these images.")])
    await native.commit([_user("Inspect these images.", red, red)])
    await native.commit([_user("Inspect these images.", blue, red)])
    await native.commit([initial, first, _user("Independent branch image.", green)])


async def test_native_context_exports_preserve_logical_identity_and_forest_training_fields(native):
    await _branches(native)
    report = native.report([1, 2, 3, 4, 5])
    assert [context["transition"] for context in report["contexts"]] == [
        "initial",
        "append",
        "reset",
        "reset",
        "reset",
        "fork",
    ]
    assert [context["state_hash"] for context in report["contexts"]] == native.response_hashes
    explicit_output = SessionOutput.from_records(report["records"])
    assert len(explicit_output.records) == 5
    jsonl = "\n".join(json.dumps(record, ensure_ascii=False) for record in report["records"])
    assert _session_output_from_text(jsonl).records == explicit_output.records
    before = copy.deepcopy(report)
    transport = validate_native_exports(
        report,
        shard=native.shard,
        session=native.session,
        invocation_id=native.invocation_id,
    )
    samples = [TrainingFieldArtifact(export["sample_payload"]).to_sample() for export in transport.exports]
    assert len(samples) == 5
    assert [export["name"] for export in transport.exports] == [f"context-{index}" for index in (1, 2, 3, 4, 5)]
    assert {(sample.index, sample.group_index, sample.session_id) for sample in samples} == {
        (17, 3, native.session.session_id)
    }
    assert all(sample.train_metadata == {"loss_owner": "source-policy"} for sample in samples)
    assert all(sample.metadata["source_tag"] == "native-fixture" and sample.reward is None for sample in samples)
    assert report == before and report["training_ready"] is False
    forest = native.session.forest
    generation_length = len(native.response_ids)
    for sample, index in zip(samples, (1, 2, 3, 4, 5)):
        lineage = forest.lineage(native.response_hashes[index])
        assert sample.tokens == [token for node in lineage for token in node.train_token_delta]
        assert sample.rollout_tokens == [token for node in lineage for token in node.rollout_token_delta]
        assert sample.metadata["agentspark_context"]["state_hash"] == native.response_hashes[index]
        assert sample.metadata["tools"] == []
        assert sample.metadata["chat_template_kwargs"] == native.compiler_defaults
        trace = sample.metadata[TRACE_KEY]
        assert trace["leaf_state_hash"] == native.response_hashes[index]
        assert trace["turn_count"] == len(trace["turns"]) == (2 if index in (1, 5) else 1)
        if index in (1, 5):
            observation_length = len(lineage[-2].train_token_delta)
            assert sample.loss_mask == [1] * generation_length + [0] * observation_length + [1] * generation_length
            assert sample.custom_advantage == [
                [0, generation_length],
                [generation_length + observation_length, sample.response_length],
            ]
            assert (
                sample.rollout_log_probs
                == native.response_logprobs + [0.0] * observation_length + native.response_logprobs
            )
        else:
            assert sample.loss_mask == [1] * generation_length
            assert sample.custom_advantage == [[0, generation_length]]
            assert sample.rollout_log_probs == native.response_logprobs


async def test_native_image_add_delete_duplicate_move_and_fork_do_not_mix_features(native):
    await _branches(native)
    transport = validate_native_exports(
        native.report([1, 2, 3, 4, 5]),
        shard=native.shard,
        session=native.session,
        invocation_id=native.invocation_id,
    )
    samples = [TrainingFieldArtifact(export["sample_payload"]).to_sample() for export in transport.exports]
    expected_grids = [
        [[1, 4, 4], [1, 4, 6]],
        None,
        [[1, 4, 4], [1, 4, 4]],
        [[1, 4, 6], [1, 4, 4]],
        [[1, 4, 4], [1, 2, 4]],
    ]
    expected_sizes = [[(43, 29), (84, 56)], [], [(43, 29), (43, 29)], [(84, 56), (43, 29)], [(43, 29), (56, 28)]]
    expected_colors = [
        [(255, 0, 0, 128), (0, 0, 255)],
        [],
        [(255, 0, 0, 128)] * 2,
        [(0, 0, 255), (255, 0, 0, 128)],
        [(255, 0, 0, 128), (0, 255, 0)],
    ]
    media_id = native.case.tokenizer.convert_tokens_to_ids("<|media_pad|>")
    for sample, index, grids, sizes, colors in zip(
        samples, (1, 2, 3, 4, 5), expected_grids, expected_sizes, expected_colors
    ):
        prefix = native.session.forest.build_execution_prefix(native.response_hashes[index])
        images = [Image.open(BytesIO(base64.b64decode(data))) for data in prefix.backend_image_data]
        assert [image.size for image in images] == sizes
        assert [image.getpixel((0, 0)) for image in images] == colors
        assert sample.rollout_tokens.count(media_id) == len(images)
        if grids is None:
            assert sample.multimodal_train_inputs is None
            assert sample.tokens.count(media_id) == 0
            continue
        assert sample.multimodal_train_inputs["image_grid_thw"].tolist() == grids
        assert sample.tokens.count(media_id) == sum(t * h * w // 4 for t, h, w in grids)
        native_pixels = native.case.processor.media_processor.preprocess(
            [{"type": "image", "image": image} for image in images],
            return_tensors="pt",
        )["pixel_values"].to(torch.bfloat16)
        torch.testing.assert_close(sample.multimodal_train_inputs["pixel_values"], native_pixels)
        assert sample.multimodal_train_inputs["pixel_values"].device.type == "cpu"


async def test_explicit_metadata_does_not_replace_source_training_metadata(native):
    await native.commit([_user("A small committed context.")])
    record = native.report([0])["records"][0]
    record["metadata"]["train_metadata"] = {"loss_owner": "export-metadata-only"}
    sample = TrainingFieldArtifact(native.native_export([record]).exports[0]["sample_payload"]).to_sample()
    assert sample.metadata["train_metadata"] == {"loss_owner": "export-metadata-only"}
    assert sample.train_metadata == {"loss_owner": "source-policy"}


@pytest.mark.parametrize("damage", ["messages", "tools", "template", "no_response"])
async def test_native_explicit_records_reject_uncommitted_or_response_free_state(native, damage):
    await native.commit([_user("A small committed context.")])
    record = native.report([0])["records"][0]
    if damage == "messages":
        record["messages"][-1]["content"] = "never committed"
    elif damage == "tools":
        record["tools"] = [{"type": "function", "function": {"name": "uncommitted_tool"}}]
    elif damage == "template":
        record["chat_template_kwargs"] = {"thinking": True}
    else:
        record["messages"] = record["messages"][:1]
    with pytest.raises(_NonFinalizableExportError, match="committed Forest state|no committed response"):
        native.native_export([record])


async def test_context_validator_rejects_incomplete_records_wrong_defaults_and_impostors(native):
    await native.commit([_user("A small committed context.")])
    report = native.report([0])
    with pytest.raises(TypeError, match="must include 'messages'"):
        SessionOutput.from_records([{"name": "incomplete"}])
    tampered = copy.deepcopy(report)
    tampered["records"][0]["messages"].pop()
    with pytest.raises(AdapterError, match="context_validation_selection_mismatch"):
        validate_native_exports(
            tampered, shard=native.shard, session=native.session, invocation_id=native.invocation_id
        )
    wrong_defaults = analyze_audit(
        report["source_audit"], compiler_defaults={"thinking": True}, selection=report["selection"]
    )
    with pytest.raises(AdapterError, match="context_validation_compiler_defaults_mismatch"):
        validate_native_exports(
            wrong_defaults, shard=native.shard, session=native.session, invocation_id=native.invocation_id
        )
    with pytest.raises(AdapterError, match="context_validation_invocation_mismatch"):
        validate_native_exports(
            report, shard=native.shard, session=native.session, invocation_id="different invocation"
        )
    impostor = SimpleNamespace(_generation_backend=native.shard._generation_backend)
    with pytest.raises(AdapterError, match="context_validation_native_session_required"):
        validate_native_exports(report, shard=impostor, session=native.session, invocation_id=native.invocation_id)
    forged_audit = copy.deepcopy(report["source_audit"])
    forged_audit["callbacks"]["records"][0]["assistant"]["content"] = "never committed by the native backend"
    forged = analyze_audit(forged_audit, compiler_defaults=native.compiler_defaults, selection=report["selection"])
    with pytest.raises(_NonFinalizableExportError, match="committed Forest state"):
        validate_native_exports(forged, shard=native.shard, session=native.session, invocation_id=native.invocation_id)


async def test_native_export_identity_rejects_grpo_sibling_and_credential_intent(native):
    messages = [_user("Same canonical context for sibling rollouts.")]
    await native.commit(messages)
    sibling = _NativeDriver(
        native.case,
        native.shard._generation_backend.compiler.cpu_executor,
        session_id="other-native-fixture-session",
        index=18,
    )
    await sibling.commit(messages)
    assert sibling.response_hashes == native.response_hashes
    assert sibling.session.group.group_id == native.session.group.group_id
    assert sibling.session.group.rollout_mode == native.session.group.rollout_mode
    assert sibling.session.session_id != native.session.session_id
    assert sibling.session.forest.index != native.session.forest.index
    report = native.report([0])
    with pytest.raises(AdapterError, match="context_validation_invocation_mismatch"):
        validate_native_exports(
            report, shard=sibling.shard, session=sibling.session, invocation_id=sibling.invocation_id
        )
    credential_intent = analyze_audit(
        report["source_audit"],
        compiler_defaults=native.compiler_defaults,
        selection=[{"name": "credential-leak", "record_index": 0, "intent": native.session.session_id}],
    )
    with pytest.raises(AdapterError, match="audit_session_credential_forbidden"):
        validate_native_exports(
            credential_intent, shard=native.shard, session=native.session, invocation_id=native.invocation_id
        )


async def test_restored_tool_images_export_only_actual_forwarded_native_states(native, tmp_path):
    """Fixed backend tokens exercise state/export, not real model generation."""
    images = [Image.new("RGBA", (43, 29), (255, 0, 0, 128)), Image.new("RGB", (84, 56), (0, 0, 255))]
    store = AssetStore([tmp_path])
    refs = []
    for index, image in enumerate(images):
        path = tmp_path / f"native-media-{index}.png"
        image.save(path)
        refs.append(store.freeze(str(path)))
    markers = ["[[MILES_REPLAY_IMAGE:" + "a" * 32 + f":{index:04d}]]" for index in range(2)]
    arguments = '{"n":1e2}'
    # Replay restoration checks the source argument hash before native Session
    # canonicalization; source 1e2/100.0 become canonical 100 inside the Forest.
    tool_history = {
        "tool_call_id": "known-image-call",
        "name": "inspect",
        "arguments_sha256": hashlib.sha256(canonical_bytes(json.loads(arguments))).hexdigest(),
        "content": "Tool result " + markers[1],
    }
    plan = store.build(
        [
            {"marker": markers[0], "source": refs[0], "role": "user"},
            {"marker": markers[1], "source": refs[1], "role": "tool", "tool_history": tool_history},
        ],
        [],
    )
    media = MediaPlan(plan, [tmp_path])
    media.public_base_url = "http://fixture.invalid/prefix"
    raw_requests = [
        {
            "model": "local-k3-fixture",
            "messages": [
                {"role": "user", "content": "Known input " + markers[0]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "known-image-call",
                            "type": "function",
                            "function": {
                                "name": "inspect",
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "known-image-call", "content": "[调用了 inspect 工具]"},
            ],
        },
        # Remove both image occurrences and the old tool segment. This becomes
        # a native reset context; no previous branch's pixels may be restored.
        {"model": "local-k3-fixture", "messages": [{"role": "user", "content": "Known input "}]},
    ]
    raw_before = copy.deepcopy(raw_requests)
    for raw in raw_requests:
        forwarded, evidence = media.restore(raw)
        await native.commit(forwarded["messages"])
        record = native.records[-1]
        assert record["request"] == forwarded
        forwarded_hash = hashlib.sha256(canonical_bytes(forwarded)).hexdigest()
        record.update(
            request_identity="actual_forwarded",
            forwarded_request_canonical_sha256=forwarded_hash,
            raw_request=copy.deepcopy(raw),
            raw_request_sha256=hashlib.sha256(canonical_bytes(raw)).hexdigest(),
            request_sha256=forwarded_hash,
            media=evidence,
        )
    assert raw_requests == raw_before
    assert [len(record["media"]["occurrences"]) for record in native.records] == [2, 0]
    assert native.records[0]["request"] != native.records[0]["raw_request"]
    assert all(marker not in json.dumps(native.records[0]["request"]) for marker in markers)

    base = native.report([0, 1])
    audit = base["source_audit"]
    audit["source_input"]["metadata"]["agentspark_media"] = plan
    report = analyze_audit(audit, compiler_defaults=native.compiler_defaults, selection=base["selection"])
    assert [context["state_hash"] for context in report["contexts"]] == native.response_hashes
    assert [context["transition"] for context in report["contexts"]] == ["initial", "reset"]
    output = validate_native_exports(
        report, shard=native.shard, session=native.session, invocation_id=native.invocation_id
    )
    samples = [TrainingFieldArtifact(export["sample_payload"]).to_sample() for export in output.exports]
    assert {(sample.index, sample.group_index) for sample in samples} == {(17, 3)}
    media_id = native.case.tokenizer.convert_tokens_to_ids("<|media_pad|>")
    assert samples[0].multimodal_train_inputs["image_grid_thw"].tolist() == [[1, 4, 4], [1, 4, 6]]
    assert samples[0].tokens.count(media_id) == 10
    assert samples[0].rollout_tokens.count(media_id) == 2
    pixels = native.case.processor.media_processor.preprocess(
        [{"type": "image", "image": image} for image in images], return_tensors="pt"
    )["pixel_values"].to(torch.bfloat16)
    torch.testing.assert_close(samples[0].multimodal_train_inputs["pixel_values"], pixels)
    assert samples[1].multimodal_train_inputs is None
    assert samples[1].tokens.count(media_id) == samples[1].rollout_tokens.count(media_id) == 0
    for sample, state_hash, image_count in zip(samples, native.response_hashes, (2, 0)):
        node = native.session.forest.nodes_by_hash[state_hash]
        assert node.train_token_delta == node.rollout_token_delta == native.response_ids
        assert node.logprob_delta == native.response_logprobs
        assert (
            sample.tokens[-len(native.response_ids) :]
            == sample.rollout_tokens[-len(native.response_ids) :]
            == native.response_ids
        )
        assert sample.rollout_log_probs == native.response_logprobs
        assert sample.loss_mask == [1] * len(native.response_ids)
        prefix = native.session.forest.build_execution_prefix(state_hash)
        assert len(prefix.backend_image_data or []) == image_count
        decoded = [Image.open(BytesIO(base64.b64decode(data))) for data in prefix.backend_image_data or []]
        assert [image.size for image in decoded] == [image.size for image in images[:image_count]]
        assert [image.getpixel((0, 0)) for image in decoded] == [
            image.getpixel((0, 0)) for image in images[:image_count]
        ]

    raw_audit = copy.deepcopy(audit)
    forged = raw_audit["callbacks"]["records"][0]
    forged["request"] = copy.deepcopy(forged["raw_request"])
    with pytest.raises(AdapterError, match="context_forwarded_request_identity_invalid"):
        analyze_audit(raw_audit, compiler_defaults=native.compiler_defaults, selection=base["selection"])
    # Even recomputing a self-consistent audit hash cannot manufacture the
    # raw-marker context as a committed native response state.
    digest = hashlib.sha256(canonical_bytes(forged["request"])).hexdigest()
    forged["forwarded_request_canonical_sha256"] = forged["request_sha256"] = digest
    raw_report = analyze_audit(raw_audit, compiler_defaults=native.compiler_defaults, selection=base["selection"])
    with pytest.raises(_NonFinalizableExportError, match="committed Forest state"):
        validate_native_exports(
            raw_report, shard=native.shard, session=native.session, invocation_id=native.invocation_id
        )


async def test_dynamic_images_and_placeholders_preserve_native_export_identity(native):
    """Real K3/Forest/export with controlled tokens; user appends isolate image identity."""
    plan = {"schema_version": 1, "assets": [], "replay_bindings": [], "seed_refs": []}
    media = MediaPlan(plan, [], allow_dynamic_images=True)
    red = _image_url((43, 29), (255, 0, 0, 128))
    blue = _image_url((84, 56), (0, 0, 255))

    async def forward(messages):
        raw = {"model": "local-k3-fixture", "messages": copy.deepcopy(messages)}
        candidate = media.fork()
        forwarded, evidence = candidate.restore(raw)
        answer = await native.commit(forwarded["messages"])
        media.commit(candidate, forwarded, answer)
        record = native.records[-1]
        assert record["request"] == forwarded
        digest = hashlib.sha256(canonical_bytes(forwarded)).hexdigest()
        record.update(
            request_identity="actual_forwarded",
            forwarded_request_canonical_sha256=digest,
            request_sha256=digest,
            raw_request=raw,
            raw_request_sha256=hashlib.sha256(canonical_bytes(raw)).hexdigest(),
            media=evidence,
        )
        return forwarded["messages"] + [answer]

    first = await forward([_user("Inspect the next images.")])
    second = await forward(first + [_user("New images", red, blue, red)])
    raw_replay = copy.deepcopy(second)
    for message in raw_replay:
        if isinstance(message.get("content"), list):
            message["content"] = [
                {"type": "text", "text": "[图片]"} if part.get("type") == "image_url" else part
                for part in message["content"]
            ]
    final = await forward(raw_replay + [_user("One more image", blue)])
    assert native.records[-1]["request"] != native.records[-1]["raw_request"]
    assert native.records[-1]["media"]["dynamic_images"]["replayed_occurrences"] == 3
    assert [record["media"]["dynamic_images"]["admitted_occurrences"] for record in native.records] == [0, 3, 1]

    report = native.report([2])
    assert [context["transition"] for context in report["contexts"]] == ["initial", "append", "append"]
    assert [context["state_hash"] for context in report["contexts"]] == native.response_hashes
    validated = validate_native_exports(
        report, shard=native.shard, session=native.session, invocation_id=native.invocation_id
    )
    records = select_training_exports(
        report["source_audit"], compiler_defaults=native.compiler_defaults, append_roles=("user",)
    )
    output = SessionOutput.from_records(records)
    assert len(output.records) == 1
    # Native canonicalization collapses text-only content lists to strings.
    assert output.records[0]["messages"] == report["contexts"][-1]["messages"]
    assert output.records[0]["metadata"]["agentspark_training"]["owned_resp_state_hashes"] == native.response_hashes
    sample = TrainingFieldArtifact(native.native_export(output.records).exports[0]["sample_payload"]).to_sample()
    checked = TrainingFieldArtifact(validated.exports[0]["sample_payload"]).to_sample()
    assert sample.tokens == checked.tokens and sample.loss_mask == checked.loss_mask

    expected = [red, blue, red, blue]
    exported = [
        part["image_url"]["url"]
        for message in output.records[0]["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"] if part.get("type") == "image_url"
    ]
    assert exported == expected
    assert [
        part["image_url"]["url"]
        for message in final if isinstance(message.get("content"), list)
        for part in message["content"] if part.get("type") == "image_url"
    ] == expected
    expected_bytes = [base64.b64decode(uri.split(",", 1)[1]) for uri in expected]
    prefix = native.session.forest.build_execution_prefix(native.response_hashes[-1])
    assert [base64.b64decode(data) for data in prefix.backend_image_data] == expected_bytes
    pixels = native.case.processor.media_processor.preprocess(
        [{"type": "image", "image": Image.open(BytesIO(raw))} for raw in expected_bytes], return_tensors="pt"
    )["pixel_values"].to(torch.bfloat16)
    torch.testing.assert_close(sample.multimodal_train_inputs["pixel_values"], pixels)
    assert sample.multimodal_train_inputs["image_grid_thw"].tolist() == [[1, 4, 4], [1, 4, 6]] * 2
    # All three actual responses are trained once; image observations carry no response loss.
    assert [token for token, mask in zip(sample.tokens[-sample.response_length:], sample.loss_mask) if mask] == native.response_ids * 3
    assert [value for value, mask in zip(sample.rollout_log_probs, sample.loss_mask) if mask] == native.response_logprobs * 3
