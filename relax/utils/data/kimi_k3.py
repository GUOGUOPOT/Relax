# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Kimi K3 training encoding through the checkpoint's Python XTML encoder.

K3 encodes structural segments as special tokens and message text as ordinary
text. Rendering to a string and tokenizing that string loses this distinction;
the slow TikToken tokenizer also cannot supply character offset mappings.
"""

from typing import TYPE_CHECKING, Any, Callable


if TYPE_CHECKING:
    from relax.engine.sft.dataset.sample import CanonicalSample


KIMI_K3_SFT_REQUEST = "_relax_kimi_k3_sft"
KIMI_K3_SFT_LOSS_MASK = "_relax_kimi_k3_sft_loss_mask"
_OPEN = "<|open|>"
_CLOSE = "<|close|>"
_SEP = "<|sep|>"
_END = "<|end_of_msg|>"


def _segment_builder(tokenizer: Any) -> Callable | None:
    # K2.x also calls its tokenizer TikTokenTokenizer. Detect the K3 Python
    # chat encoder, including inherited methods, rather than the class name.
    method = getattr(type(tokenizer), "apply_chat_template", None)
    namespace = getattr(method, "__globals__", {})
    builder = namespace.get("build_chat_segments") if isinstance(namespace, dict) else None
    if callable(builder) and callable(getattr(tokenizer, "_encode_text_piece", None)):
        return builder
    return None


def is_kimi_k3_tokenizer(tokenizer: Any) -> bool:
    return _segment_builder(tokenizer) is not None


def encode_kimi_k3_rollout_prompt(tokenizer: Any, prompt: str, image_count: int) -> list[int]:
    """Encode one raw SGLang media slot per image, without HF image framing.

    Keep the original prompt for the training processor: it replaces the HF
    placeholder with dimensions and expands the feature span. SGLang performs
    that expansion itself on the single media token sent here.
    """
    placeholder = "<|kimi_image_placeholder|>"
    found = prompt.count(placeholder)
    if found != image_count:
        raise ValueError(f"Kimi K3 rollout expects {image_count} image placeholders, found {found}.")
    # Use the HF call path directly: K3's encode(**kwargs) delegates to it
    # after emitting a warning for every request.
    ids = tokenizer(prompt.replace(placeholder, "<|media_pad|>"), add_special_tokens=False)["input_ids"]
    media_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
    if ids.count(media_id) != image_count:
        raise ValueError(
            "Kimi K3 rollout requires exactly one media token per image; ambiguous media tokens in prompt."
        )
    return ids


def _kimi_k3_agentic_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    rendered = []
    image_count = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                kind = part["type"]
                if kind in {"image", "image_url"}:
                    parts.append({"type": "image"})
                    image_count += 1
                elif kind == "text":
                    parts.append(part)
                else:
                    raise ValueError("Kimi K3 Agentic encoding currently supports text and images only.")
            content = parts
        # K3's encoder owns tool JSON rendering. In particular, do not decode
        # argument strings into dictionaries: that changes literal numbers.
        rendered.append({**message, "content": content})
    return rendered, image_count


def kimi_k3_observation_prefix(
    history_messages: list[dict[str, Any]], calibration_messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep only metadata needed for native tool association and running indices.

    These synthetic messages are calibration scaffolding, never replacement
    tokens for the already generated assistant response or previous observations.
    """
    start = next(
        (
            index
            for index in range(len(history_messages) - 1, -1, -1)
            if history_messages[index].get("role") == "assistant"
        ),
        len(history_messages),
    )
    prefix = list(calibration_messages)
    for message in history_messages[start:]:
        item = {"role": message["role"], "content": ""}
        for key in ("tool_call_id", "tool", "name"):
            if key in message:
                item[key] = message[key]
        if message.get("tool_calls"):
            item["tool_calls"] = [
                {
                    **call,
                    "function": {**call["function"], "arguments": "{}"},
                }
                for call in message["tool_calls"]
            ]
        prefix.append(item)
    return prefix


def order_kimi_k3_agentic_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    multimodal_inputs: dict[str, Any] | None,
    prefix_messages: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    """Apply native tool ordering to messages and their image spans together."""
    prefix = list(prefix_messages or [])
    builder = _segment_builder(tokenizer)
    normalize = getattr(builder, "__globals__", {}).get("normalize_xtml_tool_result_messages")
    if callable(normalize):
        # Forest keeps HTTP message order; previously encoded tool results are
        # already in native order. Compare new sorting against that same prefix.
        prefix = normalize(prefix)
    combined = prefix + messages
    media = dict(multimodal_inputs or {})
    images = list(media.get("images") or [])
    spans = []
    offset = 0
    for message in combined:
        content = message.get("content")
        count = sum(part.get("type") in {"image", "image_url"} for part in content) if isinstance(content, list) else 0
        spans.append(images[offset : offset + count])
        offset += count
    if offset != len(images):
        raise ValueError("Kimi K3 message image count does not match the supplied images.")
    if not callable(normalize):
        return messages, multimodal_inputs, prefix
    marker = "_relax_k3_message_order"
    ordered = normalize([{**message, marker: index} for index, message in enumerate(combined)])
    order = [message[marker] for message in ordered]
    if order[: len(prefix)] != list(range(len(prefix))):
        raise ValueError("Kimi K3 tool ordering would rewrite a previously committed observation prefix.")
    cleaned = [{key: value for key, value in message.items() if key != marker} for message in ordered]
    if images:
        media["images"] = [image for index in order for image in spans[index]]
    return cleaned[len(prefix) :], media or None, cleaned[: len(prefix)]


def _encode_kimi_k3_agentic_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    template_kwargs: dict[str, Any],
    image_prompts: list[str],
    prefix_messages: list[dict[str, Any]],
) -> list[int]:
    if not is_kimi_k3_tokenizer(tokenizer):
        raise ValueError("Kimi K3 Agentic encoding requires the checkpoint's Python segment tokenizer.")

    def encode(conversation: list[dict[str, Any]], *, generation: bool, images: list[str]) -> list[int]:
        # tokenize=True delegates both default options and structural/text
        # segment encoding to the checkpoint, without a rendered-text round trip.
        return list(
            tokenizer.apply_chat_template(
                conversation,
                tools=tools,
                tokenize=True,
                add_generation_prompt=generation,
                image_prompts=images,
                **template_kwargs,
            )
        )

    token_ids = encode(prefix_messages + messages, generation=True, images=image_prompts)
    if prefix_messages:
        prefix_ids = encode(prefix_messages, generation=False, images=[])
        if token_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("Kimi K3 observation encoding changed its calibration token prefix.")
        token_ids = token_ids[len(prefix_ids) :]
    return token_ids


def compile_kimi_k3_agentic_messages(
    tokenizer: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any] | None,
    multimodal_inputs: dict[str, Any] | None,
    prefix_messages: list[dict[str, Any]] | None = None,
) -> tuple[list[int], list[int], dict[str, Any] | None]:
    """Compile one prompt/observation into separate train and SGLang tokens.

    Existing generated tokens are owned by SessionForest and never enter this
    compiler. Images must already be normalized by the Agentic media loader.
    """
    kwargs = dict(chat_template_kwargs or {})
    reserved = {
        "chat_template",
        "add_non_thinking_prefix",
        "image_prompts",
        "tokenize",
        "add_generation_prompt",
        "return_tensors",
        "return_dict",
        "padding",
        "truncation",
        "max_length",
    }
    if unsupported := sorted(reserved.intersection(kwargs)):
        raise ValueError(f"Kimi K3 Agentic encoding owns template options: {', '.join(unsupported)}")
    media = multimodal_inputs or {}
    if any(media.get(key) for key in ("videos", "audio", "audios")):
        raise ValueError("Kimi K3 Agentic encoding currently supports images only.")
    rendered, image_count = _kimi_k3_agentic_messages(messages)
    prefix, prefix_image_count = _kimi_k3_agentic_messages(prefix_messages or [])
    if prefix_image_count:
        raise ValueError("Kimi K3 calibration messages must not contain images.")
    images = list(media.get("images") or [])
    if image_count != len(images):
        raise ValueError("Kimi K3 message image count does not match the supplied images.")
    encode_kwargs = dict(tools=tools, template_kwargs=kwargs, prefix_messages=prefix)
    backend_ids = _encode_kimi_k3_agentic_prompt(
        tokenizer, rendered, image_prompts=["<|media_pad|>"] * image_count, **encode_kwargs
    )
    if backend_ids.count(tokenizer.convert_tokens_to_ids("<|media_pad|>")) != image_count:
        raise ValueError("Kimi K3 rollout requires exactly one media token per image.")
    if not images:
        return backend_ids, backend_ids, None
    if processor is None:
        raise ValueError("Kimi K3 image encoding requires the checkpoint's processor.")
    features = build_kimi_k3_image_features(processor, images)
    if len(features["image_prompts"]) != image_count:
        raise ValueError("Kimi K3 processor image prompt count does not match the supplied images.")
    # Compute and verify the training prefix separately: processor text/image
    # encoding is not the SGLang raw-media token representation.
    train_ids = _encode_kimi_k3_agentic_prompt(
        processor.tokenizer, rendered, image_prompts=features["image_prompts"], **encode_kwargs
    )
    train_ids, _ = _expand_kimi_k3_image_tokens(
        processor, train_ids, [0] * len(train_ids), features["image_grid_thw"], image_count
    )
    return (
        train_ids,
        backend_ids,
        {
            "pixel_values": features["pixel_values"],
            "image_grid_thw": features["image_grid_thw"],
        },
    )


def make_kimi_k3_sft_request(
    sample: "CanonicalSample",
    apply_chat_template_kwargs: dict | None = None,
    *,
    last_turn_only: bool = False,
    ignore_empty_think: bool = False,
) -> dict[str, Any]:
    """Make a small, pickleable request shared by text and image SFT paths."""
    if sample.videos or sample.audios:
        raise ValueError("Kimi K3 SFT currently supports images only; video and audio inputs are unsupported.")
    kwargs = {
        **(apply_chat_template_kwargs or {}),
        **(sample.metadata.get("apply_chat_template_kwargs") or {}),
    }
    if kwargs.get("chat_template") or kwargs.pop("add_non_thinking_prefix", False):
        raise ValueError("Kimi K3 uses its Python XTML encoder; Jinja overrides and ChatML prefixes are unsupported.")
    kwargs.pop("chat_template", None)
    if kwargs.get("add_generation_prompt"):
        raise ValueError("Kimi K3 SFT requires add_generation_prompt=False.")
    kwargs["add_generation_prompt"] = False
    kwargs.setdefault("thinking_effort", "max")
    # The dataset owns length handling and the image worker owns image prompts.
    for key in ("image_prompts", "padding", "truncation", "max_length", "tokenize", "return_tensors", "return_dict"):
        if kwargs.get(key):
            raise ValueError(f"Kimi K3 SFT does not accept chat-template option {key!r}.")
        kwargs.pop(key, None)

    messages: list[dict[str, Any]] = []
    for message in sample.messages:
        content = message.content
        if isinstance(content, list):
            # Media bytes travel through the existing IPC media channel. The
            # encoder only needs their ordered positions in the conversation.
            content = [{"type": "image"} if part["type"] in ("image", "image_url") else part for part in content]
            if any(part["type"] in ("video", "video_url", "audio", "audio_url", "input_audio") for part in content):
                raise ValueError("Kimi K3 SFT currently supports images only; video and audio inputs are unsupported.")
        rendered: dict[str, Any] = {
            "role": "assistant" if message.role == "function_call" else message.role,
            "content": content,
        }
        if message.reasoning_content is not None:
            rendered["reasoning_content"] = message.reasoning_content
        if message.tool_calls is not None:
            rendered["tool_calls"] = message.tool_calls
        messages.append(rendered)
    last_user = max((i for i, message in enumerate(messages) if message["role"] == "user"), default=-1)
    return {
        "messages": messages,
        "tools": sample.tools,
        "learn": [
            message.learn and (not last_turn_only or i > last_user) for i, message in enumerate(sample.messages)
        ],
        "kwargs": kwargs,
        "ignore_empty_think": ignore_empty_think,
    }


def _read_tag(segments: list[Any], index: int) -> tuple[str, str, int] | None:
    """Read only encoder-owned XTML tags; literal text cannot open a tag."""
    segment = segments[index]
    if not segment.allow_special or segment.text not in (_OPEN, _CLOSE):
        return None
    end = index + 1
    header: list[str] = []
    while end < len(segments):
        piece = segments[end]
        if piece.allow_special:
            if piece.text != _SEP:
                raise ValueError("Kimi K3 encoder emitted an unexpected structural token in an XTML tag.")
            return segment.text, "".join(header), end + 1
        header.append(piece.text)
        end += 1
    raise ValueError("Kimi K3 encoder emitted an unterminated XTML tag.")


def encode_kimi_k3_sft(
    tokenizer: Any,
    request: dict[str, Any],
    *,
    image_prompts: list[str] | None = None,
) -> tuple[list[int], list[int]]:
    """Encode official segments once and project SFT spans without offsets."""
    builder = _segment_builder(tokenizer)
    if builder is None:
        raise ValueError("Kimi K3 SFT requires the checkpoint's Python segment tokenizer.")
    segments = builder(request["messages"], tools=request["tools"], image_prompts=image_prompts, **request["kwargs"])
    segment_mask = [0] * len(segments)
    message_index = 0
    cursor = 0
    while cursor < len(segments):
        tag = _read_tag(segments, cursor)
        if tag is None or tag[0] != _OPEN or not tag[1].startswith("message "):
            cursor += 1
            continue
        _, header, body_start = tag
        end = body_start
        while end < len(segments):
            closing = _read_tag(segments, end)
            if closing is not None and closing[:2] == (_CLOSE, "message"):
                end = closing[2]
                break
            end += 1
        if end >= len(segments) or not segments[end].allow_special or segments[end].text != _END:
            raise ValueError("Kimi K3 encoder emitted a message without its closing end_of_msg token.")
        end += 1
        # Tool declarations, thinking effort, and output constraints are
        # additional system messages inserted by the official encoder.
        if ' type="' in header:
            cursor = end
            continue
        if message_index >= len(request["messages"]):
            raise ValueError("Kimi K3 encoder emitted more conversation messages than the SFT sample.")
        role = request["messages"][message_index]["role"]
        if not header.startswith(f'message role="{role}"'):
            raise ValueError(f"Kimi K3 encoder changed the SFT message order at index {message_index}.")
        learn_start = body_start
        scaffold = _read_tag(segments, body_start)
        if role == "assistant" and scaffold is not None and scaffold[0] == _OPEN:
            if scaffold[1] not in ("think", "response"):
                raise ValueError("Kimi K3 assistant message has an unknown leading channel.")
            learn_start = scaffold[2]
            if scaffold[1] == "think" and request["ignore_empty_think"]:
                think_end = learn_start
                while think_end < end:
                    closing = _read_tag(segments, think_end)
                    if closing is not None and closing[:2] == (_CLOSE, "think"):
                        if not "".join(piece.text for piece in segments[learn_start:think_end]).strip():
                            learn_start = closing[2]
                        break
                    think_end += 1
        if request["learn"][message_index]:
            segment_mask[learn_start:end] = [1] * (end - learn_start)
        message_index += 1
        cursor = end
    if message_index != len(request["messages"]):
        raise ValueError("Kimi K3 encoder omitted one or more SFT conversation messages.")

    token_ids: list[int] = []
    loss_mask: list[int] = []
    for segment, learn in zip(segments, segment_mask):
        piece_ids = tokenizer._encode_text_piece(segment.text, allow_special_tokens=segment.allow_special)
        token_ids.extend(piece_ids)
        loss_mask.extend([learn] * len(piece_ids))
    return token_ids, loss_mask


def build_kimi_k3_image_features(processor: Any, images: list[Any]) -> dict[str, Any]:
    """Run the K3 pixel pipeline only: preprocess medias, build pixels and
    grid.

    Shared by the producer's :func:`process_kimi_k3_sft_images` and the rank-
    side rebuild worker (``--sft-image-preprocess-on-rank``), so both sides
    apply the identical processor, resize/crop config and BF16 conversion rule.
    The encoder-side ``image_prompts`` are returned for the producer; the rank-
    side rebuild must ignore them — it never re-tokenizes.
    """
    import numpy as np
    import torch

    medias = [{"type": "image", "image": image} for image in images]
    medias, image_prompts = processor.preprocess_medias(medias)
    features = processor.media_processor.preprocess(medias, return_tensors="pt")
    grid = features["grid_thws"]
    pixel_values = features["pixel_values"]
    if isinstance(pixel_values, np.ndarray):
        pixel_values = torch.from_numpy(pixel_values)
    if pixel_values.dtype == torch.float32:
        # Same rule as processor_pool._BF16_DOWNCAST_KEYS: the pixel tensor is
        # re-cast to the vision tower's weight dtype downstream, so the bf16
        # transfer form is lossless. Producer and consumer must agree on it.
        pixel_values = pixel_values.to(torch.bfloat16)
    return {
        "pixel_values": pixel_values.contiguous(),
        "image_grid_thw": grid,
        "image_prompts": image_prompts,
    }


def process_kimi_k3_sft_images(
    processor: Any,
    multimodal_inputs: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    """Prepare images, then encode dimension prompts and expand visual
    slots."""
    import torch

    if multimodal_inputs.get("videos") or multimodal_inputs.get("audios") or multimodal_inputs.get("audio"):
        raise ValueError("Kimi K3 SFT currently supports images only; video and audio inputs are unsupported.")
    images = multimodal_inputs.get("images", [])
    features = build_kimi_k3_image_features(processor, images)
    pixel_values = features["pixel_values"]
    grid = features["image_grid_thw"]
    token_ids, loss_mask = encode_kimi_k3_sft(processor.tokenizer, request, image_prompts=features["image_prompts"])
    expanded_ids, expanded_mask = _expand_kimi_k3_image_tokens(processor, token_ids, loss_mask, grid, len(images))
    return {
        "input_ids": [expanded_ids],
        "pixel_values": pixel_values,
        "image_grid_thw": grid,
        KIMI_K3_SFT_LOSS_MASK: torch.tensor(expanded_mask, dtype=torch.long),
    }


def _expand_kimi_k3_image_tokens(
    processor: Any,
    token_ids: list[int],
    loss_mask: list[int],
    grid: Any,
    image_count: int,
) -> tuple[list[int], list[int]]:
    merge = processor.media_processor.media_proc_cfg["merge_kernel_size"]
    merge_h, merge_w = (int(merge), int(merge)) if isinstance(merge, (int, float)) else map(int, merge)
    if merge_h <= 0 or merge_w <= 0:
        raise ValueError("Kimi K3 image merge dimensions must be positive.")
    lengths: list[int] = []
    for temporal, height, width in grid.tolist():
        if temporal != 1 or height <= 0 or width <= 0 or height % merge_h or width % merge_w:
            raise ValueError(f"Kimi K3 SFT received an invalid image grid: {(temporal, height, width)}.")
        lengths.append((height // merge_h) * (width // merge_w))
    if len(lengths) != image_count:
        raise ValueError("Kimi K3 image grid count does not match the input images.")
    placeholder_id = processor.tokenizer.convert_tokens_to_ids("<|media_pad|>")
    expanded_ids: list[int] = []
    expanded_mask: list[int] = []
    image_index = 0
    for token_id, learn in zip(token_ids, loss_mask):
        repeats = 1
        if token_id == placeholder_id:
            if image_index >= len(lengths):
                raise ValueError("Kimi K3 prompt has more media_pad placeholders than image grids.")
            repeats = lengths[image_index]
            image_index += 1
            # Visual inputs are conditioning, including images in assistant turns.
            learn = 0
        expanded_ids.extend([token_id] * repeats)
        expanded_mask.extend([learn] * repeats)
    if image_index != len(lengths):
        raise ValueError("Kimi K3 prompt has fewer media_pad placeholders than image grids.")
    return expanded_ids, expanded_mask
