"""Offline regressions for prepared inputs and AgentSpark's one-shot SSE protocol."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from examples.agentspark_agentic.protocol import (
    AdapterConfig,
    AdapterError,
    RelaxContext,
    llm_config,
    prepare_envelope,
    replay_body,
    send_body,
)
from examples.agentspark_agentic.transport import AgentSparkClient, SSEEvent, iter_sse


def _config() -> AdapterConfig:
    return AdapterConfig.from_payload(
        {
            "base_url": "https://agentspark.test/service/eden/",
            "model": "fixture-model",
            "agent": "fixture-agent",
            "source": "eden",
            "max_tokens": 512,
            "callback_bind_host": "127.0.0.1",
            "callback_bind_port": 0,
            "callback_public_base_url": "https://callback.test/service/bridge/",
        }
    )


def _context() -> RelaxContext:
    return RelaxContext.from_env(
        {
            "RELAX_BASE_URL": "https://relax.test/service/session/",
            "RELAX_SESSION_ID": "fixture-session-token",
            "RELAX_GROUP_ID": "fixture-group",
            "RELAX_ROLLOUT_MODE": "train",
            "RELAX_SESSION_IO_DIR": "fixture-invocation",
        }
    )


def _envelope() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "inputs": [{"type": "TEXT", "info": {"text": "保留原文和顺序"}}],
        "tools": ["search", "read"],
        "prompt_spec": {"system_prompt": "explicit system", "user_prompt": "explicit user"},
    }


def _prepare(envelope: dict[str, Any]) -> dict[str, Any]:
    return prepare_envelope({"metadata": {"agentspark": envelope}})


def _replay() -> dict[str, Any]:
    return {
        "messages": [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-7", "arguments": '{"n":1e2}'}]},
            {"role": "tool", "tool_call_id": "call-7", "content": "原始结果", "native_attachment": {"index": 7}},
        ],
        "attributes": {
            "ark.idindex.mapping": {"7": "native-ref"},
            "ark.idindex.image_urls": {"7": "https://images.test/seventh.png"},
            "ark.idindex.reverse_mapping": {"native-ref": 7},
        },
    }


def test_prepared_send_and_replay_preserve_fields_order_and_independent_copies() -> None:
    envelope = _envelope()
    envelope["inputs"] += [
        {"type": "IMAGE", "info": [{"file_id": "second", "width": 17, "height": 19}]},
        {"type": "IMAGES", "info": [{"image_url": "https://images.test/third.png"}, {"file_id": "fourth"}]},
        {"type": "NOTE", "info": {"note_id": "note-five"}},
        {"type": "POI", "info": {"id": "poi-six"}},
    ]
    envelope.update(
        bound_user_id="bound-user",
        tool_versions={"search": "v7"},
        capabilities={"native": "enabled"},
        skills_name_list=["skill-z", "skill-a"],
        replay_spec=_replay(),
    )
    original = copy.deepcopy(envelope)
    prepared = _prepare(envelope)
    context, config = _context(), _config()
    send = send_body(config, context, prepared, "remote-session", config.callback_public_base_url)
    restore = replay_body(config, context, prepared)
    assert envelope == prepared == original
    assert {key: send[key] for key in original if key not in {"schema_version", "replay_spec", "bound_user_id"}} == {
        key: value for key, value in original.items() if key not in {"schema_version", "replay_spec", "bound_user_id"}
    }
    assert send["user"] == restore["user"] == {"user_id": context.user_id, "bound_user_id": "bound-user"}
    assert restore["replay_spec"] == {"replay_type": "restore_only", **original["replay_spec"]}
    assert "replay_spec" not in send
    assert "inputs" not in restore
    send["inputs"][1]["info"][0]["file_id"] = "changed"
    restore["replay_spec"]["attributes"]["ark.idindex.mapping"]["7"] = "changed"
    prepared["prompt_spec"]["system_prompt"] = "changed"
    assert envelope == original
    assert prepared["inputs"] == original["inputs"]
    assert prepared["replay_spec"] == original["replay_spec"]


def test_text_budget_is_total_utf8_bytes_and_never_truncates() -> None:
    envelope = _envelope()
    half = "界" * 316 + "ab"
    envelope["inputs"] = [{"type": "TEXT", "info": {"text": half}} for _ in range(2)]
    assert sum(len(item["info"]["text"].encode("utf-8")) for item in envelope["inputs"]) == 1900
    assert _prepare(envelope) == envelope
    envelope["inputs"][1]["info"]["text"] += "!"
    original = copy.deepcopy(envelope)
    with pytest.raises(AdapterError, match="^text_requires_source_preparation$"):
        _prepare(envelope)
    assert envelope == original


@pytest.mark.parametrize(
    "info",
    [
        {"image_url": "https://images.test/a.png"},
        "https://images.test/a.png",
        ["https://images.test/a.png"],
        [{"image_url": "https://images.test/a.png", "file_id": "ambiguous"}],
        [{"file_id": "image-a", "width": True}],
        [],
    ],
)
def test_image_wire_info_requires_explicit_standard_list(info: Any) -> None:
    envelope = _envelope()
    envelope["inputs"] = [{"type": "IMAGE", "info": info}]
    with pytest.raises(AdapterError, match="^image_input_invalid$"):
        _prepare(envelope)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("tools", [" search"], "tools_invalid"),
        ("tools", ["read\n"], "tools_invalid"),
        ("tools", ["search", "search"], "tools_duplicate"),
        ("bound_user_id", " bound-user", "bound_user_id_invalid"),
        ("bound_user_id", "bound-user\t", "bound_user_id_invalid"),
        ("schema_version", True, "agentspark_envelope_version_invalid"),
        ("unexpected_native_field", "must-not-be-dropped", "agentspark_envelope_fields_invalid"),
    ],
)
def test_prepared_envelope_rejects_ambiguous_or_unknown_fields(field: str, value: Any, code: str) -> None:
    envelope = _envelope()
    envelope[field] = value
    with pytest.raises(AdapterError, match=f"^{code}$"):
        _prepare(envelope)


def test_history_is_never_inferred_from_ambient_messages_or_metadata() -> None:
    envelope = _envelope()
    ambient = {"messages": _replay()["messages"], "metadata": {"history": _replay(), "agentspark": envelope}}
    prepared = prepare_envelope(ambient)
    assert prepared == envelope
    assert "replay_spec" not in prepared
    with pytest.raises(AdapterError, match="^replay_spec_required$"):
        replay_body(_config(), _context(), prepared)
    with pytest.raises(AdapterError, match="^metadata_agentspark_required$"):
        prepare_envelope({"messages": ambient["messages"], "metadata": {"history": _replay()}})


def test_replay_requires_explicit_complete_resource_mappings() -> None:
    envelope = _envelope()
    envelope["replay_spec"] = _replay()
    del envelope["replay_spec"]["attributes"]["ark.idindex.reverse_mapping"]
    with pytest.raises(AdapterError, match="^replay_attributes_required$"):
        _prepare(envelope)


def test_llm_config_keeps_callback_prefix_and_scoped_bearer_auth() -> None:
    config, context = _config(), _context()
    actual = llm_config(config, context, config.callback_public_base_url)
    assert actual == {
        "provider": "openai",
        "base_url": "https://callback.test",
        "endpoint_path": "/service/bridge/v1/chat/completions",
        "auth_mode": "bearer",
        "api_key": context.session_id,
        "model": config.model,
        "max_tokens": 512,
    }
    assert actual["base_url"] + actual["endpoint_path"] == "https://callback.test/service/bridge/v1/chat/completions"


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, fail_after_data: bool = False) -> None:
        self.chunks = chunks
        self.fail_after_data = fail_after_data
        self.finished = False
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk
        if self.fail_after_data:
            raise httpx.ReadError("fixture stream interruption")
        self.finished = True

    async def aclose(self) -> None:
        self.closed = True


def _frame(payload: Any, event: str | None = None, *, ending: str = "\n\n") -> bytes:
    prefix = f"event: {event}\n" if event is not None else ""
    return (prefix + "data: " + json.dumps(payload, ensure_ascii=False) + ending).encode("utf-8")


def _ack(session_id: str = "restored-session", code: Any = 0) -> dict[str, Any]:
    return {"type": "link.up.ack", "code": code, "data": {"session_id": session_id}}


async def _decode(chunks: list[bytes], *, max_frame_bytes: int = 4096) -> list[SSEEvent]:
    response = httpx.Response(200, stream=_Chunks(chunks))
    try:
        return [event async for event in iter_sse(response, max_frame_bytes=max_frame_bytes)]
    finally:
        await response.aclose()


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_sse_handles_split_utf8_and_preserves_unicode_line_separators(newline: str) -> None:
    content = "中文\u0085保留\u2028分隔\u2029末尾"
    wire = newline.join(
        [
            ": keepalive",
            "event: snapshot",
            'data: {"type": "snapshot",',
            "data: " + json.dumps({"content": content}, ensure_ascii=False)[1:],
            "",
            "",
        ]
    ).encode("utf-8")
    events = asyncio.run(_decode([wire[index : index + 1] for index in range(len(wire))]))
    assert len(events) == 1
    assert events[0].event == events[0].kind == "snapshot"
    assert events[0].payload == {"type": "snapshot", "content": content}
    assert "\n" in events[0].raw


def test_sse_consumes_trailing_snapshot_after_end_without_final_newline() -> None:
    end = {"type": "agent.stream.end", "data": {"done": True}}
    snapshot = {"type": "snapshot", "data": {"messages": ["最后快照"]}}
    events = asyncio.run(_decode([_frame(end), _frame(snapshot, ending="")]))
    assert [event.payload for event in events] == [end, snapshot]


@pytest.mark.parametrize("wire", [b"data: \xff\n\n", b"data: \xe4\xb8"])
def test_sse_rejects_invalid_or_truncated_utf8(wire: bytes) -> None:
    with pytest.raises(AdapterError, match="^invalid_sse_utf8$"):
        asyncio.run(_decode([wire]))


@pytest.mark.parametrize("wire", [b"data: 123456789", b"data: 123\ndata: 456\n\n"])
def test_sse_bounds_pending_and_assembled_frame_bytes(wire: bytes) -> None:
    with pytest.raises(AdapterError, match="^sse_frame_too_large$"):
        asyncio.run(_decode([wire], max_frame_bytes=12))


async def _restore(wire: bytes, *, fail_after_data: bool = False) -> tuple[str, _Chunks, list[httpx.Request]]:
    stream = _Chunks([wire] if wire else [], fail_after_data=fail_after_data)
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        async with AgentSparkClient(_config(), client=client) as agentspark:
            session_id = await agentspark.restore({"replay_spec": {"replay_type": "restore_only"}})
    return session_id, stream, requests


def test_replay_accepts_matching_acks_only_after_complete_eof() -> None:
    payload_without_type = {"code": 0, "data": {"session_id": "restored-session"}}
    wire = _frame(_ack()) + _frame(payload_without_type, "link.up.ack", ending="")
    session_id, stream, requests = asyncio.run(_restore(wire))
    assert session_id == "restored-session"
    assert stream.finished and stream.closed
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "https://agentspark.test/service/eden/diandian/chat/sse-send-protocol"
    assert requests[0].headers["accept"] == "text/event-stream"


@pytest.mark.parametrize(
    ("wire", "code"),
    [
        (b"", "replay_eof_without_ack"),
        (b": keepalive\n\n", "replay_eof_without_ack"),
        (_frame({"code": 0, "data": {"session_id": "unlabeled"}}), "replay_unexpected_event"),
        (_frame(_ack(code=False)), "replay_invalid_ack"),
        (_frame(_ack(code=0.0)), "replay_invalid_ack"),
        (_frame(_ack(code="0")), "replay_invalid_ack"),
        (_frame(_ack(code=1)), "replay_invalid_ack"),
        (_frame({"type": "link.up.ack", "data": {"session_id": "missing-code"}}), "replay_invalid_ack"),
        (_frame(_ack(session_id="  ")), "replay_ack_session_missing"),
        (_frame({"type": "link.up.ack", "code": 0}), "replay_ack_session_missing"),
        (_frame(_ack("first")) + _frame(_ack("second")), "replay_ack_session_inconsistent"),
        (_frame(_ack(), "agent.stream.error"), "replay_error_event"),
        (_frame({"type": "error"}, "link.up.ack"), "replay_error_event"),
        (_frame(_ack(), "snapshot"), "replay_unexpected_event"),
        (_frame({"type": "snapshot"}, "link.up.ack"), "replay_unexpected_event"),
        (_frame(_ack()) + _frame({"type": "agent.stream.end"}), "replay_unexpected_event"),
    ],
)
def test_replay_fails_closed_for_ambiguous_ack_error_and_non_ack_frames(wire: bytes, code: str) -> None:
    with pytest.raises(AdapterError, match=f"^{code}$"):
        asyncio.run(_restore(wire))


def test_post_stream_failure_after_ack_fails_without_retry() -> None:
    requests: list[httpx.Request] = []
    stream = _Chunks([_frame(_ack())], fail_after_data=True)

    async def run() -> None:
        async def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, stream=stream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            async with AgentSparkClient(_config(), client=client) as agentspark:
                await agentspark.restore({"replay_spec": {"replay_type": "restore_only"}})

    with pytest.raises(AdapterError, match="^agentspark_transport_error$"):
        asyncio.run(run())
    assert len(requests) == 1
    assert not stream.finished
    assert stream.closed


@pytest.mark.parametrize("failure", ["http", "transport"])
def test_post_failure_is_attempted_once(failure: str) -> None:
    requests: list[httpx.Request] = []

    async def run() -> None:
        async def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if failure == "transport":
                raise httpx.ConnectError("fixture failure", request=request)
            return httpx.Response(503, content=b"fixture unavailable")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            async with AgentSparkClient(_config(), client=client) as agentspark:
                async for _event in agentspark.events({"request_id": "must-not-be-repeated"}):
                    pytest.fail("a failed POST must not produce events")

    code = "agentspark_transport_error" if failure == "transport" else "agentspark_http_error"
    with pytest.raises(AdapterError, match=f"^{code}$"):
        asyncio.run(run())
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert json.loads(requests[0].content) == {"request_id": "must-not-be-repeated"}


def test_session_lookup_preserves_service_prefix_and_scoped_user() -> None:
    requests: list[httpx.Request] = []
    context = _context()

    async def run() -> str:
        async def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"session_id": "remote-session"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            async with AgentSparkClient(_config(), client=client) as agentspark:
                return await agentspark.get_session(context)

    assert asyncio.run(run()) == "remote-session"
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "GET"
    assert request.url.path == "/service/eden/diandian/session"
    assert dict(request.url.params) == {"user_id": context.user_id, "source": "eden"}
