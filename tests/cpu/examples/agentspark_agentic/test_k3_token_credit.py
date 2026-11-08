"""Real local K3 IDs: visible credit cannot leak into reasoning, tools or ancestors."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from examples.agentspark_agentic.k3_token_credit import (
    attach_k3_followup_mask,
    extract_k3_response_spans,
    map_char_mask_to_tokens,
)
from tests.cpu.models.test_kimi_k3_hf import official_k3  # noqa: F401


WORD = "可见追问🙂"


def _marker(text: str) -> list[int]:
    mask = [0] * len(text)
    start = 0
    while (start := text.find(WORD, start)) >= 0:
        mask[start : start + len(WORD)] = [1] * len(WORD)
        start += len(WORD)
    return mask


def _continuation(tokenizer, *, thinking: bool) -> list[int]:
    history = [{"role": "user", "content": "Question"}]
    assistant = {
        "role": "assistant",
        "content": f'Literal <|close|>think<|sep|>. <guide_question type="recommend">{WORD}</guide_question>',
        "reasoning_content": f"Reason privately about {WORD}.",
        "tool_calls": [{"id": "x", "type": "function", "function": {"name": "lookup", "arguments": {"query": WORD}}}],
    }
    options = {"tokenize": True, "thinking": thinking}
    prefix = tokenizer.apply_chat_template(history, add_generation_prompt=True, **options)
    full = tokenizer.apply_chat_template(history + [assistant], add_generation_prompt=False, **options)
    assert full[: len(prefix)] == prefix
    return list(full[len(prefix) :])


@pytest.mark.parametrize("thinking", [True, False])
def test_real_k3_visible_channel_excludes_reasoning_and_tool_arguments(official_k3, thinking: bool) -> None:
    tokenizer = official_k3.tokenizer
    tokens = _continuation(tokenizer, thinking=thinking)
    spans = extract_k3_response_spans(tokenizer, tokens, thinking=thinking)
    assert len(spans) == 1
    left, right = spans[0]
    text = tokenizer.decode(tokens[left:right])
    assert text.startswith("Literal <|close|>think<|sep|>.")
    assert text.count(WORD) == 1
    native_spans = [[0, len(tokens)]]
    sample = SimpleNamespace(
        response_length=len(tokens),
        tokens=tokens.copy(),
        rollout_log_probs=[-0.2] * len(tokens),
        loss_mask=[1] * len(tokens),
        custom_advantage=native_spans,
        metadata={
            "chat_template_kwargs": {"thinking": thinking},
            "agentic_trace": {"turns": [{"response_length": len(tokens)}]},
        },
    )
    before = copy.deepcopy((sample.tokens, sample.rollout_log_probs, sample.custom_advantage))
    attach_k3_followup_mask(sample, tokenizer, _marker)
    mask = sample.metadata["agentspark_v21"]["followup_token_mask"]
    assert sum(mask) > 0
    assert not any(mask[:left]) and not any(mask[right:])
    assert WORD in tokenizer.decode([token for token, keep in zip(tokens, mask, strict=True) if keep])
    assert (sample.tokens, sample.rollout_log_probs, sample.custom_advantage) == before


def test_noncanonical_utf8_tokens_and_piece_straddling_word_boundary(official_k3) -> None:
    tokenizer = official_k3.tokenizer
    text = f"before {WORD} after"
    ranks = tokenizer.model._mergeable_ranks
    byte_tokens = [ranks[bytes([value])] for value in text.encode()]
    assert tokenizer.decode(byte_tokens) == text
    assert byte_tokens != tokenizer.encode(text, add_special_tokens=False)
    mask = map_char_mask_to_tokens(tokenizer, byte_tokens, _marker(text))
    selected = [token for token, keep in zip(byte_tokens, mask, strict=True) if keep]
    assert tokenizer.decode(selected) == WORD
    tokens = tokenizer.encode(" unexpected", add_special_tokens=False)
    char_mask = [0] + [1] * (len(" unexpected") - 1)
    assert map_char_mask_to_tokens(tokenizer, tokens, char_mask)[0] == 1


def test_truncated_reasoning_and_unowned_ancestor_receive_zero_credit(official_k3) -> None:
    tokenizer = official_k3.tokenizer
    first = _continuation(tokenizer, thinking=True)
    close = tokenizer.convert_tokens_to_ids("<|close|>")
    assert extract_k3_response_spans(tokenizer, first[: first.index(close)], thinking=True) == []
    # Two distinct native generations separated by an observation. Ownership
    # removes the first generation; the observation contains target wording too.
    observation = tokenizer.encode(WORD, add_special_tokens=False)
    second = _continuation(tokenizer, thinking=True)
    start = len(first) + len(observation)
    tokens = first + observation + second
    sample = SimpleNamespace(
        tokens=tokens,
        response_length=len(tokens),
        loss_mask=[0] * start + [1] * len(second),
        custom_advantage=[[0, len(first)], [start, len(tokens)]],
        metadata={
            "chat_template_kwargs": {"thinking": True},
            "agentic_trace": {"turns": [{"response_length": len(first)}, {"response_length": len(second)}]},
        },
    )
    attach_k3_followup_mask(sample, tokenizer, _marker)
    mask = sample.metadata["agentspark_v21"]["followup_token_mask"]
    assert not any(mask[:start]) and any(mask[start:])
    sample.custom_advantage[1][0] -= 1
    with pytest.raises(ValueError, match="spans"):
        attach_k3_followup_mask(sample, tokenizer, _marker)


def test_k3_credit_rejects_missing_effective_thinking_and_bad_policy_mask(official_k3) -> None:
    tokenizer = official_k3.tokenizer
    with pytest.raises(ValueError, match="thinking"):
        extract_k3_response_spans(tokenizer, [], thinking=None)
    tokens = tokenizer.encode(WORD, add_special_tokens=False)
    with pytest.raises(ValueError, match="binary"):
        map_char_mask_to_tokens(tokenizer, tokens, [2] * len(WORD))
