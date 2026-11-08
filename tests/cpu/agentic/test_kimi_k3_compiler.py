# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""K3 Agentic regressions plus opt-in, local-checkpoint contract validation.

The small segment encoder is a protocol fixture, not a copied K3 template.
Set KIMI_K3_SOURCE_DIR to exercise the checkpoint's own tokenizer/processor;
those optional checks load no model weights and make no network requests.
"""

from __future__ import annotations

import base64
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import pytest
import torch
from PIL import Image

from relax.agentic.pipeline import runtime
from relax.agentic.pipeline.runtime import SGLangMessageCompiler
from relax.utils.data.kimi_k3 import compile_kimi_k3_agentic_messages
from tests.cpu.models.test_kimi_k3_hf import official_k3  # noqa: F401


@dataclass
class _Segment:
    text: str
    allow_special: bool = False


def build_chat_segments(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    image_prompts: list[str] | None = None,
    add_generation_prompt: bool = True,
    unstable_prefix: bool = False,
) -> list[_Segment]:
    pieces = []
    if tools:
        pieces.append(_Segment(json.dumps(tools)))
    if unstable_prefix:
        pieces.append(_Segment(str(len(messages))))
    image_iter = iter(image_prompts or [])
    for message in messages:
        pieces.extend([_Segment("<|open|>", True), _Segment(message["role"]), _Segment("<|sep|>", True)])
        if message.get("reasoning_content"):
            pieces.append(_Segment(message["reasoning_content"]))
        for call in message.get("tool_calls", []):
            # The fixture makes an extra adapter-side JSON parse visible;
            # the selected checkpoint encoder owns numeric rendering.
            assert isinstance(call["function"]["arguments"], str)
            pieces.append(_Segment(call["function"]["arguments"]))
        content = message.get("content") or ""
        if isinstance(content, list):
            for part in content:
                pieces.append(_Segment(next(image_iter), True) if part["type"] == "image" else _Segment(part["text"]))
        else:
            pieces.append(_Segment(content))
        pieces.append(_Segment("<|end_of_msg|>", True))
    if add_generation_prompt:
        pieces.extend([_Segment("<|open|>", True), _Segment("assistant"), _Segment("<|sep|>", True)])
    return pieces


class _Tokenizer:
    specials = {"<|open|>": 1000, "<|sep|>": 1001, "<|end_of_msg|>": 1002, "<|media_pad|>": 1003}

    def __init__(self, *, long_calibration: bool = False) -> None:
        self.long_calibration = long_calibration

    def _encode_text_piece(self, text: str, *, allow_special_tokens: bool) -> list[int]:
        tokens = []
        if self.long_calibration and text == runtime._DUMMY_MESSAGES[0]["content"]:
            # Deliberately make training/backend calibration lengths differ.
            # Sharing either trim length must corrupt the resulting delta.
            text = text * 2
        while text:
            special = next((key for key in self.specials if text.startswith(key)), None)
            if special is not None and allow_special_tokens:
                tokens.append(self.specials[special])
                text = text[len(special) :]
            else:
                tokens.append(ord(text[0]))
                text = text[1:]
        return tokens

    def apply_chat_template(self, messages: list[dict[str, Any]], *, tokenize: bool, **kwargs: Any) -> list[int]:
        assert tokenize is True, "K3 must not render to text and re-tokenize"
        segments = build_chat_segments(messages, **kwargs)
        return [
            token
            for segment in segments
            for token in self._encode_text_piece(segment.text, allow_special_tokens=segment.allow_special)
        ]

    def encode(self, *args: Any, **kwargs: Any) -> list[int]:
        raise AssertionError("K3 must preserve the official encoder's structural/text distinction")

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.specials[token]

    def decode(self, tokens: list[int]) -> str:
        inverse = {value: key for key, value in self.specials.items()}
        return "".join(inverse[token] if token in inverse else chr(token) for token in tokens)


class _Processor:
    media_proc_cfg = {"merge_kernel_size": 2}

    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.media_processor = self

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("K3 must not pass rendered text back to the processor")

    def preprocess_medias(self, medias: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        return medias, [f"image {media['image'].width}x{media['image'].height}<|media_pad|>" for media in medias]

    def preprocess(self, medias: list[dict[str, Any]], *, return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        widths = [media["image"].width for media in medias]
        return {
            "pixel_values": torch.zeros(sum(widths) * 4, 3),
            "grid_thws": torch.tensor([[1, 2, width * 2] for width in widths]),
        }


@pytest.fixture
def executor():
    with ThreadPoolExecutor(max_workers=2) as pool:
        yield pool


async def test_k3_first_prompt_preserves_literals_reasoning_and_tool_json(executor):
    tokenizer = _Tokenizer()
    compiler = SGLangMessageCompiler(tokenizer=tokenizer, processor=None, cpu_executor=executor)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    messages = [
        {"role": "user", "content": "literal <|open|> <|media_pad|>"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "consider",
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": '{"x":1e2}'}}],
        },
        {"role": "tool", "content": "literal <|end_of_msg|>"},
    ]
    before = copy.deepcopy(messages)
    encoded = await compiler.encode_messages(messages, tools=tools)
    expected = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True)
    assert encoded.backend_prompt_ids == expected == encoded.train_prompt_ids
    assert encoded.backend_prompt_ids.count(tokenizer.specials["<|media_pad|>"]) == 0
    assert encoded.backend_prompt_ids.count(tokenizer.specials["<|open|>"]) == len(messages) + 1
    assert '{"x":1e2}' in tokenizer.decode(encoded.train_prompt_ids)
    assert "consider" in tokenizer.decode(encoded.train_prompt_ids)
    assert messages == before


@pytest.mark.parametrize("observation", [False, True])
async def test_k3_images_keep_raw_backend_slots_and_expand_training_spans(executor, monkeypatch, observation):
    tokenizer = _Tokenizer()
    train_tokenizer = _Tokenizer(long_calibration=True)
    processor = _Processor(train_tokenizer)
    compiler = SGLangMessageCompiler(tokenizer=tokenizer, processor=processor, cpu_executor=executor)
    images = [Image.new("RGB", (2, 4)), Image.new("RGB", (3, 4))]
    # The media loader is not under test. Keep the real compiler, feature
    # preprocessing, image encoding, and separate-token export paths.
    monkeypatch.setattr(runtime, "_normalize_multimodal_inputs_sync", lambda media, *_: (media, {}))
    messages = [
        {
            "role": "tool" if observation else "user",
            "content": [
                {"type": "text", "text": "literal <|media_pad|>"},
                {"type": "image_url", "image_url": {"url": "first"}},
                {"type": "text", "text": "between"},
                {"type": "image_url", "image_url": {"url": "second"}},
            ],
        }
    ]
    encode = compiler.encode_observation_delta if observation else compiler.encode_messages
    encoded = await encode(messages, tools=None, multimodal_inputs={"images": images})
    media_id = tokenizer.specials["<|media_pad|>"]
    assert encoded.backend_prompt_ids.count(media_id) == 2
    assert encoded.train_prompt_ids.count(media_id) == 5
    train_text = train_tokenizer.decode(encoded.train_prompt_ids)
    backend_text = tokenizer.decode(encoded.backend_prompt_ids)
    assert "literal <|media_pad|>" in train_text and "literal <|media_pad|>" in backend_text
    assert "image 2x4" in train_text and "image 3x4" in train_text
    assert train_text.index("image 2x4") < train_text.index("between") < train_text.index("image 3x4")
    assert "image 2x4" not in backend_text
    assert runtime._DUMMY_MESSAGES[0]["content"] not in train_text
    assert runtime._DUMMY_MESSAGES[1]["content"] not in train_text
    assert encoded.train_prompt_ids[0] == encoded.backend_prompt_ids[0] == tokenizer.specials["<|open|>"]
    assert encoded.multimodal_train_inputs["image_grid_thw"].tolist() == [[1, 2, 4], [1, 2, 6]]
    assert encoded.multimodal_train_inputs["pixel_values"].dtype == torch.bfloat16
    assert len(encoded.backend_image_data) == 2
    assert encoded.backend_audio_data == encoded.backend_video_data == []
    if observation:
        options = dict(tokenize=True, add_generation_prompt=False)
        assert len(tokenizer.apply_chat_template(runtime._DUMMY_MESSAGES, **options)) != len(
            train_tokenizer.apply_chat_template(runtime._DUMMY_MESSAGES, **options)
        )


async def test_k3_observation_rejects_nonprefix_template_before_slicing(executor):
    compiler = SGLangMessageCompiler(tokenizer=_Tokenizer(), processor=None, cpu_executor=executor)
    with pytest.raises(ValueError, match="calibration token prefix"):
        await compiler.encode_observation_delta(
            [{"role": "tool", "content": "result"}], tools=None, chat_template_kwargs={"unstable_prefix": True}
        )


@pytest.mark.parametrize(
    "messages,media,kwargs,match",
    [
        ([{"role": "user", "content": [{"type": "image"}]}], None, None, "image count"),
        ([{"role": "user", "content": "question"}], {"audio": [object()]}, None, "images only"),
        ([{"role": "user", "content": [{"type": "video"}]}], None, None, "text and images only"),
        ([{"role": "user", "content": "question"}], None, {"image_prompts": []}, "owns template options"),
    ],
)
def test_k3_rejects_mismatched_media_and_overridden_encoding(messages, media, kwargs, match):
    with pytest.raises(ValueError, match=match):
        compile_kimi_k3_agentic_messages(
            _Tokenizer(), None, messages, tools=None, chat_template_kwargs=kwargs, multimodal_inputs=media
        )


async def test_non_k3_keeps_existing_tool_json_rendering(executor):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            arguments = messages[0]["tool_calls"][0]["function"]["arguments"]
            assert arguments == {"x": 100.0}
            return json.dumps(arguments)

        def encode(self, text, **kwargs):
            return list(text.encode())

    compiler = SGLangMessageCompiler(tokenizer=Tokenizer(), processor=None, cpu_executor=executor)
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": '{"x":1e2}'}}]}
    ]
    encoded = await compiler.encode_messages(messages, tools=None)
    assert encoded.train_prompt_ids == encoded.backend_prompt_ids == list(b'{"x": 100.0}')
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"x":1e2}'


@pytest.mark.parametrize("observation", [False, True])
async def test_k3_official_checkpoint_compiler_contract(official_k3, executor, observation):
    """Optional: actual checkpoint encoder/processor, entirely local and CPU."""
    case = official_k3
    compiler = SGLangMessageCompiler(tokenizer=case.tokenizer, processor=case.processor, cpu_executor=executor)
    messages = [
        {
            "role": "tool" if observation else "user",
            "content": [{"type": "text", "text": "literal <|open|>"}, {"type": "image"}],
        }
    ]
    history = [{"role": "assistant", "content": "", "tool_calls": [_official_call("call-image", "inspect")]}]
    if observation:
        messages[0]["tool_call_id"] = "call-image"
    method = compiler.encode_observation_delta if observation else compiler.encode_messages
    context = {"history_messages": history} if observation else {}
    encoded = await method(messages, tools=None, multimodal_inputs={"images": [case.image]}, **context)
    prefix = runtime._DUMMY_MESSAGES + history if observation else []
    backend = case.tokenizer.apply_chat_template(
        prefix + messages, tokenize=True, add_generation_prompt=True, image_prompts=["<|media_pad|>"]
    )
    prefix_ids = (
        case.tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=False) if prefix else []
    )
    assert list(backend[: len(prefix_ids)]) == list(prefix_ids)
    assert encoded.backend_prompt_ids == list(backend[len(prefix_ids) :])
    media_id = case.tokenizer.convert_tokens_to_ids("<|media_pad|>")
    assert encoded.backend_prompt_ids.count(media_id) == 1
    from relax.utils.data.processing_utils import expand_kimi_k25_placeholders, remap_mm_train_inputs

    native = case.processor(
        messages=prefix + messages,
        tools=None,
        medias=[{"type": "image", "image": case.image}],
        add_generation_prompt=True,
    )
    train_ids = expand_kimi_k25_placeholders(
        case.processor,
        native["input_ids"][0].tolist(),
        remap_mm_train_inputs(case.processor, {"grid_thws": native["grid_thws"]}),
    )
    train_prefix = (
        case.processor.tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=False)
        if prefix
        else []
    )
    assert train_ids[: len(train_prefix)] == list(train_prefix)
    assert encoded.train_prompt_ids == train_ids[len(train_prefix) :]
    torch.testing.assert_close(encoded.multimodal_train_inputs["image_grid_thw"], native["grid_thws"])


def _official_call(call_id: str, name: str, arguments: str = "{}") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.mark.parametrize("thinking", [False, True])
async def test_k3_real_tokenizer_tools_reasoning_and_literal_arguments(official_k3, executor, thinking):
    case = official_k3
    compiler = SGLangMessageCompiler(tokenizer=case.tokenizer, processor=case.processor, cpu_executor=executor)
    tools = [{"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}}]
    messages = [
        {"role": "user", "content": "literal <|open|> <|media_pad|>"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "consider <|close|>",
            "tool_calls": [_official_call("opaque-call", "inspect", '{"n":1e2,"literal":"<|sep|>"}')],
        },
        {"role": "tool", "tool_call_id": "opaque-call", "content": "result <|end_of_msg|>"},
    ]
    before = copy.deepcopy(messages)
    encoded = await compiler.encode_messages(messages, tools=tools, chat_template_kwargs={"thinking": thinking})
    expected = case.tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, thinking=thinking)
    assert encoded.train_prompt_ids == encoded.backend_prompt_ids == expected
    text = case.tokenizer.decode(expected)
    # Checkpoint revisions may render 1e2 as 100.0. Native IDs above are
    # authoritative; the adapter must preserve the numeric value and literals.
    number = text.split('argument key="n" type="number"<|sep|>', 1)[1].split("<|close|>", 1)[0]
    assert json.loads(number) == 100
    assert "literal <|open|> <|media_pad|>" in text
    assert ("consider <|close|>" in text) == thinking
    assert expected.count(case.tokenizer.convert_tokens_to_ids("<|media_pad|>")) == 0
    assert messages == before


@pytest.mark.parametrize("observation", [False, True])
async def test_k3_real_tool_reordering_keeps_two_images_and_native_pixels_aligned(official_k3, executor, observation):
    from relax.utils.data.processing_utils import expand_kimi_k25_placeholders, remap_mm_train_inputs

    case = official_k3
    compiler = SGLangMessageCompiler(tokenizer=case.tokenizer, processor=case.processor, cpu_executor=executor)
    red = Image.new("RGBA", (43, 29), (255, 0, 0, 128))
    blue = Image.new("RGB", (84, 56), (0, 0, 255))
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [_official_call("opaque-a", "red_tool"), _official_call("opaque-b", "blue_tool")],
    }
    red_result = {
        "role": "tool",
        "tool_call_id": "opaque-a",
        "content": [{"type": "text", "text": "red result <|open|>"}, {"type": "image"}],
    }
    blue_result = {
        "role": "tool",
        "tool_call_id": "opaque-b",
        "content": [{"type": "text", "text": "blue result"}, {"type": "image"}],
    }
    # Ground truth is call A/red then B/blue. Native processor alone extracts
    # media before sorting tools; feeding the same wrong order is not an oracle.
    prefix = runtime._DUMMY_MESSAGES + [assistant] if observation else []
    messages = [blue_result, red_result] if observation else [assistant, blue_result, red_result]
    ordered = [red_result, blue_result] if observation else [assistant, red_result, blue_result]
    before = copy.deepcopy(messages)
    method = compiler.encode_observation_delta if observation else compiler.encode_messages
    context = {"history_messages": [assistant]} if observation else {}
    encoded = await method(messages, tools=None, multimodal_inputs={"images": [blue, red]}, **context)
    native = case.processor(
        messages=prefix + ordered,
        tools=None,
        medias=[{"type": "image", "image": red}, {"type": "image", "image": blue}],
        add_generation_prompt=True,
    )
    expected_train = expand_kimi_k25_placeholders(
        case.processor,
        native["input_ids"][0].tolist(),
        remap_mm_train_inputs(case.processor, {"grid_thws": native["grid_thws"]}),
    )
    expected_backend = case.tokenizer.apply_chat_template(
        prefix + ordered,
        tokenize=True,
        add_generation_prompt=True,
        image_prompts=["<|media_pad|>"] * 2,
    )
    trim = case.tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=False) if prefix else []
    assert expected_train[: len(trim)] == expected_backend[: len(trim)] == trim
    assert encoded.train_prompt_ids == expected_train[len(trim) :]
    assert encoded.backend_prompt_ids == expected_backend[len(trim) :]
    media_id = case.tokenizer.convert_tokens_to_ids("<|media_pad|>")
    assert encoded.backend_prompt_ids.count(media_id) == 2
    assert encoded.train_prompt_ids.count(media_id) == 10
    assert encoded.multimodal_train_inputs["image_grid_thw"].tolist() == [[1, 4, 4], [1, 4, 6]]
    torch.testing.assert_close(
        encoded.multimodal_train_inputs["pixel_values"], native["pixel_values"].to(torch.bfloat16)
    )
    decoded = [Image.open(BytesIO(base64.b64decode(value))) for value in encoded.backend_image_data]
    assert [image.size for image in decoded] == [red.size, blue.size]
    assert [image.mode for image in decoded] == ["RGBA", "RGB"]
    assert decoded[0].getpixel((0, 0)) == (255, 0, 0, 128)
    assert decoded[1].getpixel((0, 0)) == (0, 0, 255)
    assert messages == before
    assert all(value.device.type == "cpu" for value in encoded.multimodal_train_inputs.values())


async def test_k3_real_incremental_tool_indices_continue_and_cross_prefix_reordering_fails(official_k3, executor):
    case = official_k3
    compiler = SGLangMessageCompiler(tokenizer=case.tokenizer, processor=case.processor, cpu_executor=executor)
    assistant = {
        "role": "assistant",
        "content": "original generated content",
        "reasoning_content": "original reasoning",
        "tool_calls": [_official_call("opaque-a", "first"), _official_call("opaque-b", "second")],
    }
    first = {"role": "tool", "tool_call_id": "opaque-a", "content": "first result"}
    second = {"role": "tool", "tool_call_id": "opaque-b", "content": "second result"}
    encoded = await compiler.encode_observation_delta([second], tools=None, history_messages=[assistant, first])
    text = case.tokenizer.decode(encoded.backend_prompt_ids)
    assert 'tool="second" index="2"' in text
    assert "original generated content" not in text and "original reasoning" not in text and "first result" not in text
    assert encoded.train_prompt_ids == encoded.backend_prompt_ids
    with pytest.raises(ValueError, match="previously committed observation prefix"):
        await compiler.encode_observation_delta([first], tools=None, history_messages=[assistant, second])


@pytest.mark.parametrize(
    "override", [{"image_min_token_num": 8}, {"image_max_token_num": 128}, {"image_resize_scale_factor": 0}]
)
async def test_k3_rejects_foreign_image_resize_overrides(executor, override):
    from relax.utils.multimodal.config import MultimodalConfig

    compiler = SGLangMessageCompiler(
        tokenizer=_Tokenizer(),
        processor=_Processor(_Tokenizer()),
        cpu_executor=executor,
        multimodal_config=MultimodalConfig(**override),
    )
    with pytest.raises(ValueError, match="unsupported override"):
        await compiler.encode_messages(
            [{"role": "user", "content": [{"type": "image"}]}],
            tools=None,
            multimodal_inputs={"images": [Image.new("RGB", (8, 8))]},
        )


async def test_k3_real_increment_after_previously_reordered_tool_results(official_k3, executor):
    case = official_k3
    compiler = SGLangMessageCompiler(tokenizer=case.tokenizer, processor=case.processor, cpu_executor=executor)
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [_official_call(f"opaque-{name}", name) for name in ("first", "second", "third")],
    }
    first, second, third = [
        {"role": "tool", "tool_call_id": f"opaque-{name}", "content": f"{name} result"}
        for name in ("first", "second", "third")
    ]
    # Forest retains the original message order, while the preceding encoding
    # has already emitted the native canonical order (first, second).
    await compiler.encode_observation_delta([second, first], tools=None, history_messages=[assistant])
    encoded = await compiler.encode_observation_delta([third], tools=None, history_messages=[assistant, second, first])
    assert 'tool="third" index="3"' in case.tokenizer.decode(encoded.backend_prompt_ids)


async def test_k3_service_passes_committed_tool_history_to_real_compiler(official_k3, executor):
    """Exercise actual Service/Forest wiring, with no hand-supplied compiler history."""
    import asyncio
    import time
    from argparse import Namespace
    from types import SimpleNamespace

    from relax.agentic.session.service import AgenticSessionShard
    from relax.agentic.session.state import InflightRequest, RequestKind, SessionForest, check_messages

    case = official_k3
    compiler = SGLangMessageCompiler(tokenizer=case.tokenizer, processor=case.processor, cpu_executor=executor)
    shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = Namespace(
        agentic_reasoning_parser="kimi_k3",
        agentic_tool_call_parser="kimi_k3",
        rollout_max_context_len=4096,
        eval_max_context_len=4096,
    )
    shard._generation_backend = SimpleNamespace(compiler=compiler, tokenizer=case.tokenizer)
    forest = SessionForest.create_empty(session_id="k3-history-wiring")
    session = SimpleNamespace(
        forest=forest,
        session_id=forest.session_id,
        rollout_id=0,
        group=SimpleNamespace(rollout_mode="train"),
        resp_state_hash_by_request_id={},
    )
    tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object", "properties": {"n": {"type": "number"}}}},
        }
        for name in ("red_tool", "blue_tool")
    ]
    template_kwargs = {"thinking": True}
    history = [{"role": "user", "content": "Inspect both tool images."}]
    initial, _ = await shard._append_observation(
        session=session,
        forest=forest,
        parent=forest.nodes_by_hash[forest.root_state_hash],
        messages_delta=check_messages(history),
        tools=tools,
        chat_template_kwargs=template_kwargs,
    )
    assistant = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Inspect both tools first.",
        "tool_calls": [
            _official_call("source-red", "red_tool", '{"n":1e2}'),
            _official_call("source-blue", "blue_tool", '{"n":2}'),
        ],
    }
    completed = case.tokenizer.apply_chat_template(
        history + [assistant], tools=tools, tokenize=True, add_generation_prompt=False, **template_kwargs
    )
    assert completed[: len(initial.rollout_token_delta)] == initial.rollout_token_delta
    generated = completed[len(initial.rollout_token_delta) :]
    original_ids = list(generated)
    original_logprobs = [-0.125 - index / 1000 for index in range(len(generated))]
    ir = InflightRequest(
        request_id="native-tools",
        parent_state_hash=initial.state_hash,
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=0,
        waiter=asyncio.get_running_loop().create_future(),
        wall_started_at=time.monotonic(),
        pending_token_delta=generated,
        pending_logprob_delta=list(original_logprobs),
        pending_status="completed",
    )
    terminal = shard._terminal_response_locked(session=session, ir=ir, finish_type="stop")
    assert terminal["finish_reason"] == "tool_calls"
    calls = terminal["message"]["tool_calls"]
    assert [call["function"]["name"] for call in calls] == ["red_tool", "blue_tool"]
    # Forest canonicalizes JSON, but the selected source's actual generated
    # token sequence must survive subsequent image observations byte-for-byte.
    assert calls[0]["function"]["arguments"] == '{"n":100}'
    number = case.tokenizer.decode(generated).split('argument key="n" type="number"<|sep|>', 1)[1]
    assert json.loads(number.split("<|close|>", 1)[0]) == 100
    response = forest.nodes_by_hash[session.resp_state_hash_by_request_id[ir.request_id]]
    parent = response
    images = [Image.new("RGBA", (43, 29), (255, 0, 0, 128)), Image.new("RGB", (84, 56), (0, 0, 255))]
    observations = []
    for index, (call, image) in enumerate(zip(calls, images), start=1):
        png = BytesIO()
        image.save(png, format="PNG")
        image_url = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode("ascii")
        delta = check_messages(
            [
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": [{"type": "image_url", "image_url": {"url": image_url}}],
                }
            ]
        )
        assert "name" not in delta[0] and "tool" not in delta[0]
        parent, _ = await shard._append_observation(
            session=session,
            forest=forest,
            parent=parent,
            messages_delta=delta,
            tools=tools,
            chat_template_kwargs=template_kwargs,
        )
        observations.append(parent)
        backend_text = case.tokenizer.decode(parent.rollout_token_delta)
        assert f'tool="{call["function"]["name"]}" index="{index}"' in backend_text
        assert response.train_token_delta == response.rollout_token_delta == original_ids
        assert response.logprob_delta == original_logprobs
    assert [node.multimodal_train_inputs_delta["image_grid_thw"].tolist() for node in observations] == [
        [[1, 4, 4]],
        [[1, 4, 6]],
    ]
    media_id = case.tokenizer.convert_tokens_to_ids("<|media_pad|>")
    assert [node.rollout_token_delta.count(media_id) for node in observations] == [1, 1]
    assert [node.train_token_delta.count(media_id) for node in observations] == [4, 6]
    prefix = forest.build_execution_prefix(parent.state_hash)
    assert prefix.train_token_prefix == initial.train_token_delta + original_ids + [
        token for node in observations for token in node.train_token_delta
    ]
    decoded = [Image.open(BytesIO(base64.b64decode(data))) for data in prefix.backend_image_data]
    assert [image.size for image in decoded] == [(43, 29), (84, 56)]
    assert [image.getpixel((0, 0)) for image in decoded] == [(255, 0, 0, 128), (0, 0, 255)]


@pytest.mark.parametrize("reasoning", [" \t\n", ""])
async def test_k3_real_canonical_json_and_whitespace_reasoning_prefix(official_k3, executor, reasoning):
    from relax.agentic.session.state import check_messages

    case = official_k3
    compiler = SGLangMessageCompiler(tokenizer=case.tokenizer, processor=case.processor, cpu_executor=executor)
    assistant = {
        "role": "assistant",
        "content": "",
        "reasoning_content": reasoning,
        "tool_calls": [_official_call("canonical-call", "inspect", '{"n":100,"nested":{"x":[1,2]}}')],
    }
    history = [{"role": "user", "content": "Inspect the result."}, assistant]
    normalized = check_messages(history)
    assert (
        normalized[-1]["tool_calls"][0]["function"]["arguments"] == assistant["tool_calls"][0]["function"]["arguments"]
    )
    options = dict(tokenize=True, add_generation_prompt=True)
    assert case.tokenizer.apply_chat_template(history, **options) == case.tokenizer.apply_chat_template(
        normalized, **options
    )
    initial = await compiler.encode_messages(normalized, tools=None)
    assert (
        initial.train_prompt_ids
        == initial.backend_prompt_ids
        == case.tokenizer.apply_chat_template(normalized, **options)
    )
    delta = [{"role": "tool", "tool_call_id": "canonical-call", "content": "canonical result"}]
    observed = await compiler.encode_observation_delta(delta, tools=None, history_messages=normalized)
    # This direct native prefix includes the actual canonical arguments and
    # whitespace reasoning; compiler calibration is synthetic. Their next
    # observation suffix must still be identical without touching old tokens.
    prefix = runtime._DUMMY_MESSAGES + normalized
    before = case.tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=False)
    complete = case.tokenizer.apply_chat_template(prefix + delta, **options)
    assert complete[: len(before)] == before
    assert observed.train_prompt_ids == observed.backend_prompt_ids == complete[len(before) :]
    assert 'tool="inspect" index="1"' in case.tokenizer.decode(observed.backend_prompt_ids)
