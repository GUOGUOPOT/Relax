# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Map V21 visible-wording credit onto original K3 response token IDs.

The caller supplies the pinned Miles character policy. K3's structural channel
IDs select the answer before that policy runs; literal spellings of those IDs
in text do not change channels. Training IDs and log probabilities are untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


CharacterMarker = Callable[[str], list[int]]


def _decode(tokenizer: Any, tokens: list[int]) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def extract_k3_response_spans(tokenizer: Any, token_ids: list[int], *, thinking: bool) -> list[tuple[int, int]]:
    """Return answer-body intervals from a native K3 generation continuation.

    The generation prefix already opens either ``think`` or ``response``.
    An unfinished thinking channel has no visible answer. An incomplete header
    is never treated as answer text. The subsequent tools channel is excluded.
    """
    if type(thinking) is not bool:
        raise ValueError("K3 token credit requires explicit effective thinking")
    names = ("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>")
    ids = [tokenizer.convert_tokens_to_ids(name) for name in names]
    if any(type(value) is not int or value < 0 for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("K3 structural token vocabulary is missing")
    opening, closing, separator, terminal = ids
    structural = set(ids)

    def header(position: int, marker: int, expected: str) -> int | None:
        if position >= len(token_ids) or token_ids[position] != marker:
            return None
        end = position + 1
        while end < len(token_ids) and token_ids[end] not in structural:
            end += 1
        if end == len(token_ids) or token_ids[end] != separator:
            return None
        return end + 1 if _decode(tokenizer, token_ids[position + 1 : end]) == expected else None

    start = 0
    if thinking:
        # Reasoning may contain ordinary text spelling a structural tag, but
        # the real close is a dedicated ID emitted by the model.
        while start < len(token_ids) and token_ids[start] not in structural:
            start += 1
        after_think = header(start, closing, "think")
        if after_think is None:
            return []
        after_open = header(after_think, opening, "response")
        if after_open is None:
            return []
        start = after_open
    end = start
    while end < len(token_ids) and token_ids[end] not in structural:
        end += 1
    return [(start, end)] if start < end else []


def map_char_mask_to_tokens(tokenizer: Any, token_ids: list[int], char_mask: list[int]) -> list[int]:
    """Use decoded original prefixes, including noncanonical byte tokenization."""
    text = _decode(tokenizer, token_ids)
    if len(char_mask) != len(text) or any(type(value) is not int or value not in (0, 1) for value in char_mask):
        raise ValueError("V21 character mask must be a binary vector matching decoded text")
    cache = {0: 0, len(token_ids): len(text)}

    def prefix_chars(count: int) -> int:
        if count not in cache:
            decoded = _decode(tokenizer, token_ids[:count])
            matched = 0
            for actual, expected in zip(decoded, text):
                if actual != expected:
                    break
                matched += 1
            cache[count] = matched
        return cache[count]

    def first_prefix_at_least(position: int) -> int:
        left, right = 0, len(token_ids)
        while left < right:
            middle = (left + right) // 2
            if prefix_chars(middle) < position:
                left = middle + 1
            else:
                right = middle
        return left

    mask = [0] * len(token_ids)
    cursor = 0
    while cursor < len(char_mask):
        if not char_mask[cursor]:
            cursor += 1
            continue
        end = cursor + 1
        while end < len(char_mask) and char_mask[end]:
            end += 1
        first = first_prefix_at_least(cursor)
        if prefix_chars(first) > cursor:
            first -= 1
        last = first_prefix_at_least(end)
        if first < 0 or last <= first:
            raise ValueError("V21 wording cannot be located on original K3 tokens")
        mask[first:last] = [1] * (last - first)
        cursor = end
    return mask


def attach_k3_followup_mask(sample: Any, tokenizer: Any, mark_visible_text: CharacterMarker) -> None:
    """Attach response-aligned credit using native per-generation spans.

    Call after applying export ownership to ``loss_mask`` and before replacing
    native ``custom_advantage`` span records with numerical advantages.
    """
    length = sample.response_length
    if type(length) is not int or length <= 0 or len(sample.tokens) < length:
        raise ValueError("K3 credit requires a nonempty native response")
    losses = sample.loss_mask
    if not isinstance(losses, list) or len(losses) != length or any(value not in (0, 1) for value in losses):
        raise ValueError("K3 credit requires response-aligned binary ownership loss_mask")
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        raise ValueError("K3 credit requires native export metadata")
    template = metadata.get("chat_template_kwargs", {})
    thinking = template.get("thinking")
    if type(thinking) is not bool:
        raise ValueError("K3 credit requires the effective chat_template_kwargs.thinking")
    turns = metadata.get("agentic_trace", {}).get("turns")
    spans = sample.custom_advantage
    if not isinstance(turns, list) or not isinstance(spans, list) or len(spans) != len(turns) or not turns:
        raise ValueError("K3 credit requires native response-node spans and trace turns")
    tokens = sample.tokens[-length:]
    mask = [0] * length
    last_end = 0
    for span, turn in zip(spans, turns, strict=True):
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or not last_end <= span[0] < span[1] <= length
            or not isinstance(turn, dict)
            or turn.get("response_length") != span[1] - span[0]
        ):
            raise ValueError("K3 native response spans do not match trace")
        start, end = span
        last_end = end
        if not any(losses[start:end]):
            continue
        generated = tokens[start:end]
        for left, right in extract_k3_response_spans(tokenizer, generated, thinking=thinking):
            body = generated[left:right]
            body_mask = map_char_mask_to_tokens(tokenizer, body, mark_visible_text(_decode(tokenizer, body)))
            for offset, value in enumerate(body_mask, start + left):
                mask[offset] = value * losses[offset]
    metadata.setdefault("agentspark_v21", {})["followup_token_mask"] = mask
    metadata["n_followup_credit_tokens"] = sum(mask)
