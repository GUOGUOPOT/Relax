# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local checkpoint format acceptance through Relax's actual response parsers.

These are template-generated token continuations, not model generation results.
The local source fixture loads no model weight shards and makes no network calls.
"""

from __future__ import annotations

import asyncio
import json
import time
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import pytest

from relax.agentic.session.service import AgenticSessionShard, _decode_response_payload
from relax.agentic.session.state import InflightRequest, RequestKind, SessionForest
from tests.cpu.models.test_kimi_k3_hf import official_k3  # noqa: F401


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("tool_turn", [False, True])
def test_local_k3_template_continuation_matches_relax_parsers(official_k3, thinking: bool, tool_turn: bool) -> None:
    tokenizer = official_k3.tokenizer
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                },
            },
        }
    ]
    history = [{"role": "user", "content": "Find the requested record."}]
    assistant = {"role": "assistant", "content": "Searching now." if tool_turn else "The record is ready."}
    if thinking:
        assistant["reasoning_content"] = "Check the requested record first."
    if tool_turn:
        assistant["tool_calls"] = [
            {
                "id": "source-call-id",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query":"A & B","limit":2}'},
            }
        ]
    options = {"tokenize": True, "thinking": thinking, "tools": tools}
    prompt = list(tokenizer.apply_chat_template(history, add_generation_prompt=True, **options))
    completed = list(tokenizer.apply_chat_template(history + [assistant], add_generation_prompt=False, **options))
    assert completed[: len(prompt)] == prompt
    continuation = completed[len(prompt) :]
    assert continuation
    args = Namespace(agentic_reasoning_parser="kimi_k3", agentic_tool_call_parser="kimi_k3")
    result, has_tools = _decode_response_payload(
        args=args, tokenizer=tokenizer, token_ids=continuation, tools=tools, parent_state_hash="parent-a"
    )
    assert has_tools is tool_turn
    parsed = result[0]
    assert parsed["role"] == "assistant"
    assert parsed["content"] == assistant["content"]
    assert parsed.get("reasoning_content") == assistant.get("reasoning_content")
    if tool_turn:
        call = parsed["tool_calls"][0]
        assert call["function"]["name"] == "lookup"
        assert json.loads(call["function"]["arguments"]) == {"query": "A & B", "limit": 2}
        same, _ = _decode_response_payload(
            args=args, tokenizer=tokenizer, token_ids=continuation, tools=tools, parent_state_hash="parent-a"
        )
        other, _ = _decode_response_payload(
            args=args, tokenizer=tokenizer, token_ids=continuation, tools=tools, parent_state_hash="parent-b"
        )
        assert same == result
        assert other[0]["tool_calls"][0]["id"] != call["id"]
    else:
        assert "tool_calls" not in parsed


async def _commit_local_continuation(
    tokenizer: Any, *, thinking: bool, content: str, reasoning: str | None = None, truncated: bool = False
) -> tuple[dict[str, Any], list[int]]:
    history = [{"role": "user", "content": "Return a short answer."}]
    assistant = {"role": "assistant", "content": content}
    if reasoning is not None:
        assistant["reasoning_content"] = reasoning
    options = {"tokenize": True, "thinking": thinking}
    prompt = list(tokenizer.apply_chat_template(history, add_generation_prompt=True, **options))
    completed = list(tokenizer.apply_chat_template(history + [assistant], add_generation_prompt=False, **options))
    assert completed[: len(prompt)] == prompt
    continuation = completed[len(prompt) :]
    if truncated:
        # Cut at the first structural close token; ordinary text spelling that
        # marker remains a different BPE sequence in the checkpoint vocabulary.
        continuation = continuation[: continuation.index(tokenizer.convert_tokens_to_ids("<|close|>"))]
    original_ids = list(continuation)
    logprobs = [-0.125 - index / 1000 for index in range(len(continuation))]
    forest = SessionForest.create_empty(session_id="parser-local-session")
    observation = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=history,
        train_token_delta=prompt,
        rollout_token_delta=prompt,
        tools=[],
        chat_template_kwargs={"thinking": thinking},
    )
    shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = Namespace(agentic_reasoning_parser="kimi_k3", agentic_tool_call_parser="kimi_k3")
    shard._generation_backend = SimpleNamespace(
        tokenizer=tokenizer,
        # The committed subtree must win over an opposite global default.
        compiler=SimpleNamespace(apply_chat_template_kwargs={"thinking": not thinking}),
    )
    session = SimpleNamespace(forest=forest, resp_state_hash_by_request_id={})
    ir = InflightRequest(
        request_id="parser-request",
        parent_state_hash=observation.state_hash,
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=0,
        waiter=asyncio.get_running_loop().create_future(),
        wall_started_at=time.monotonic(),
        pending_token_delta=continuation,
        pending_logprob_delta=list(logprobs),
        pending_status="truncated" if truncated else "completed",
        logprobs=True,
    )
    payload = shard._terminal_response_locked(session=session, ir=ir, finish_type="length" if truncated else "stop")
    response = forest.nodes_by_hash[session.resp_state_hash_by_request_id[ir.request_id]]
    assert ir.pending_token_delta == response.train_token_delta == response.rollout_token_delta == original_ids
    assert ir.pending_logprob_delta == response.logprob_delta == logprobs
    assert payload["usage"]["completion_tokens"] == len(original_ids)
    assert len(payload["logprobs"]["content"]) == len(original_ids)
    assert [entry["logprob"] for entry in payload["logprobs"]["content"]] == logprobs
    return payload["message"], original_ids


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
async def test_k3_terminal_parser_uses_effective_subtree_thinking(
    official_k3, thinking: bool, truncated: bool
) -> None:
    content = "A partial answer."
    reasoning = "A partial analysis." if thinking else None
    message, _ = await _commit_local_continuation(
        official_k3.tokenizer, thinking=thinking, content=content, reasoning=reasoning, truncated=truncated
    )
    if thinking and truncated:
        assert message["content"] == ""
        assert message["reasoning_content"] == reasoning
    else:
        assert message["content"] == content
        assert message.get("reasoning_content") == reasoning


@pytest.mark.parametrize("truncated", [False, True])
async def test_k3_parser_keeps_literal_eom_text_and_raw_terminal_token(official_k3, truncated: bool) -> None:
    tokenizer = official_k3.tokenizer
    content = "This spelling is ordinary text: <|end_of_msg|>"
    message, ids = await _commit_local_continuation(tokenizer, thinking=False, content=content, truncated=truncated)
    eom_id = tokenizer.convert_tokens_to_ids("<|end_of_msg|>")
    assert ids.count(eom_id) == (0 if truncated else 1)
    assert message["content"] == content
    assert "reasoning_content" not in message
