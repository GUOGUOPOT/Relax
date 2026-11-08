# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from relax.utils.data.data_utils import check_sample_length
from relax.utils.types import Sample
from tests.cpu.models.test_kimi_k3_hf import official_k3  # noqa: F401


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    model_path = os.environ.get("RELAX_TEST_QWEN3_4B")
    if not model_path or not Path(model_path).is_dir():
        pytest.skip("Qwen3-4B is unavailable; set RELAX_TEST_QWEN3_4B to its local directory")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        pytest.skip(f"transformers AutoTokenizer unavailable: {exc}")
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)


@pytest.fixture(scope="module")
def qwen3_vl_processor():
    model_path = os.environ.get("RELAX_TEST_QWEN3_VL_4B")
    if not model_path or not Path(model_path).is_dir():
        pytest.skip("Qwen3-VL-4B-Instruct is unavailable; set RELAX_TEST_QWEN3_VL_4B to its local directory")
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        pytest.skip(f"transformers AutoProcessor unavailable: {exc}")
    return AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)


def _render_image_prompt(processor: Any, image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the image."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert isinstance(prompt, str)
    return prompt


def _multimodal_token_count(processor: Any, prompt: str, image: Image.Image) -> int:
    return len(processor(text=prompt, images=[image])["input_ids"][0])


def test_multimodal_processor_expands_image_tokens(qwen3_vl_processor):
    image = Image.new("RGB", (224, 224), (32, 96, 160))
    prompt = _render_image_prompt(qwen3_vl_processor, image)

    text_token_count = len(qwen3_vl_processor.tokenizer(prompt, add_special_tokens=False)["input_ids"])
    multimodal_token_count = _multimodal_token_count(qwen3_vl_processor, prompt, image)

    assert multimodal_token_count > text_token_count


def test_multimodal_length_boundary_is_inclusive(qwen3_vl_processor):
    image = Image.new("RGB", (224, 224), (32, 96, 160))
    prompt = _render_image_prompt(qwen3_vl_processor, image)
    sample = Sample(prompt=prompt, multimodal_inputs={"images": [image]})
    actual_length = _multimodal_token_count(qwen3_vl_processor, prompt, image)

    assert check_sample_length(
        sample,
        qwen3_vl_processor.tokenizer,
        qwen3_vl_processor,
        max_length=actual_length,
    )
    assert not check_sample_length(
        sample,
        qwen3_vl_processor.tokenizer,
        qwen3_vl_processor,
        max_length=actual_length - 1,
    )


def test_larger_image_resolution_produces_more_tokens(qwen3_vl_processor):
    small_image = Image.new("RGB", (224, 224), (32, 96, 160))
    large_image = Image.new("RGB", (448, 448), (32, 96, 160))
    prompt = _render_image_prompt(qwen3_vl_processor, small_image)

    small_length = _multimodal_token_count(qwen3_vl_processor, prompt, small_image)
    large_length = _multimodal_token_count(qwen3_vl_processor, prompt, large_image)

    assert large_length > small_length

    sample = Sample(prompt=prompt, multimodal_inputs={"images": [large_image]})
    assert not check_sample_length(
        sample,
        qwen3_vl_processor.tokenizer,
        qwen3_vl_processor,
        max_length=small_length,
    )


def test_text_prompt_length_boundary_without_processor(qwen3_tokenizer):
    prompt = "Compare two integers and explain the answer. " * 8
    sample = Sample(prompt=prompt)
    actual_length = len(qwen3_tokenizer(prompt, add_special_tokens=False)["input_ids"])

    assert check_sample_length(sample, qwen3_tokenizer, processor=None, max_length=actual_length)
    assert not check_sample_length(sample, qwen3_tokenizer, processor=None, max_length=actual_length - 1)


@pytest.mark.parametrize("as_messages", [False, True])
@pytest.mark.parametrize("with_image", [False, True])
def test_kimi_k3_length_matches_native_training_tokens(official_k3, as_messages, with_image):
    from relax.utils.data.processing_utils import expand_kimi_k25_placeholders, remap_mm_train_inputs

    case = official_k3
    image = Image.new("RGBA", (84, 56), (32, 96, 160, 128))
    content = [{"type": "text", "text": "Describe this literal <|open|>."}]
    if with_image:
        content.append({"type": "image"})
    messages = [{"role": "user", "content": content}]
    tools = [{"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}}]
    options = {"thinking": False}
    prompt = messages if as_messages else case.tokenizer.apply_chat_template(messages, tools=tools, **options)
    sample = Sample(
        prompt=prompt,
        metadata={"tools": tools, "apply_chat_template_kwargs": options},
        # Dataset populates this dictionary even when there is no actual media.
        multimodal_inputs={"images": [image] if with_image else None, "videos": None, "audio": None},
    )
    if with_image:
        native_kwargs = {"messages": messages, "tools": tools, **options} if as_messages else {"text": prompt}
        native = case.processor(medias=[{"type": "image", "image": image}], **native_kwargs)
        native_ids = native["input_ids"][0].tolist()
        expected = expand_kimi_k25_placeholders(
            case.processor, native_ids, remap_mm_train_inputs(case.processor, {"grid_thws": native["grid_thws"]})
        )
        assert len(expected) > len(native_ids)
    elif as_messages:
        expected = case.tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, **options)
    else:
        expected = case.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    assert check_sample_length(
        sample, case.tokenizer, case.processor, len(expected), apply_chat_template_kwargs={"thinking": True}
    )
    assert not check_sample_length(
        sample, case.tokenizer, case.processor, len(expected) - 1, apply_chat_template_kwargs={"thinking": True}
    )


@pytest.mark.parametrize("failure", ["missing_image", "missing_processor", "unsupported_media"])
def test_kimi_k3_length_errors_do_not_admit_sample(official_k3, failure):
    case = official_k3
    media = {"images": [case.image]}
    processor = case.processor
    if failure == "missing_image":
        media = {}
    elif failure == "missing_processor":
        processor = None
    else:
        media["videos"] = ["not-opened.mp4"]
    sample = Sample(prompt=[{"role": "user", "content": [{"type": "image"}]}], multimodal_inputs=media)
    assert not check_sample_length(sample, case.tokenizer, processor, max_length=100_000)


def test_non_k3_length_boundary_and_existing_error_policy():
    def tokenizer(text, **kwargs):
        return {"input_ids": list(range(len(text)))}

    class BrokenProcessor:
        def __call__(self, **kwargs):
            raise ValueError("unsupported image")

    sample = Sample(prompt="three")
    assert check_sample_length(sample, tokenizer, None, 5)
    assert not check_sample_length(sample, tokenizer, None, 4)
    sample.multimodal_inputs = {"images": [object()]}
    assert check_sample_length(sample, tokenizer, BrokenProcessor(), 4)


@pytest.mark.parametrize("kimi_k3", [False, True])
def test_length_worker_failure_does_not_reintroduce_kimi_k3_fail_open(official_k3, monkeypatch, kimi_k3):
    from concurrent.futures import ThreadPoolExecutor

    from relax.utils.data import data_utils

    def broken_worker(sample, max_length):
        raise RuntimeError("length worker failed")

    monkeypatch.setattr(data_utils, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(data_utils, "_check_sample_length_worker", broken_worker)
    tokenizer = official_k3.tokenizer if kimi_k3 else object()
    samples = [Sample(prompt="first"), Sample(prompt="second")]
    result = data_utils.filter_long_prompts(samples, tokenizer, object(), max_length=100, num_workers=2)
    assert result == ([] if kimi_k3 else samples)


def test_kimi_k3_batch_fast_paths_do_not_admit_invalid_samples(official_k3):
    from relax.utils.data.data_utils import filter_long_prompts

    tokenizer = official_k3.tokenizer
    valid = Sample(prompt="Valid text prompt.")
    invalid_slot = Sample(prompt="Unbound <|media_pad|> slot.")
    missing_processor = Sample(
        prompt="Image <|kimi_image_placeholder|>", multimodal_inputs={"images": [official_k3.image]}
    )
    unsupported = Sample(prompt={"invalid": "not a text or message-list prompt"})
    string_samples = [valid, invalid_slot, missing_processor]
    assert filter_long_prompts(string_samples, tokenizer, None, max_length=10_000) == [valid]
    assert filter_long_prompts([unsupported, *string_samples], tokenizer, None, max_length=10_000) == [valid]
