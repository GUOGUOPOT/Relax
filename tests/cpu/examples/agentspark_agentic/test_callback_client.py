# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import copy
import hashlib
import json
from dataclasses import asdict

import httpx
import pytest

from examples.agentspark_agentic.callback import CallbackCapture, CallbackRecorder, assistant_from_response
from examples.agentspark_agentic.client import AUDIT_ONLY_EXIT, main, run_audit
from examples.agentspark_agentic.protocol import AdapterConfig, AdapterError, RelaxContext
from examples.agentspark_agentic.result import write_audit


@pytest.fixture
def config():
    return AdapterConfig.from_payload(
        {
            "base_url": "http://agentspark.invalid/prefix",
            "model": "fixture-model",
            "agent": "diandian",
            "source": "eden",
            "max_tokens": 32,
            "callback_bind_host": "127.0.0.1",
            "callback_bind_port": 0,
            "callback_public_base_url": "http://127.0.0.1:0/callback",
            "cleanup_timeout_s": 0.2,
        }
    )


@pytest.fixture
def context():
    return RelaxContext(
        "http://relax.invalid/serving-prefix",
        "private-session-credential",
        "group",
        "train",
        "/invocation/1",
    )


@pytest.fixture
def session_input():
    return {
        "metadata": {
            "agentspark": {
                "schema_version": 1,
                "inputs": [{"type": "TEXT", "info": {"text": "task"}}],
                "tools": ["Search"],
                "prompt_spec": {"system_prompt": "system", "user_prompt": "user"},
            }
        }
    }


def completion(message=None, *, sse=False, finish="stop"):
    message = message or {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "reason",
    }
    payload = {"choices": [{"delta" if sse else "message": message, "finish_reason": finish}]}
    if sse:
        raw = ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode()
        return httpx.Response(200, content=raw, headers={"Content-Type": "text/event-stream"})
    return httpx.Response(200, json=payload)


def event(kind, **extra):
    return ("data: " + json.dumps({"type": kind, **extra}, ensure_ascii=False) + "\n\n").encode()


@pytest.mark.asyncio
async def test_callback_preserves_raw_bytes_auth_prefix_and_sse_tool_history(config, context):
    calls = []
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "Search", "arguments": "{}"},
    }
    canonical = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "think",
        "tool_calls": [tool_call],
    }
    wire = copy.deepcopy(canonical)
    wire["tool_calls"][0]["index"] = 0
    responses = [completion(wire, sse=True, finish="tool_calls"), completion(sse=True)]

    async def upstream(request):
        calls.append(request)
        return responses[len(calls) - 1]

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as backend:
        async with CallbackRecorder(config, context, client=backend) as recorder:
            async with httpx.AsyncClient(trust_env=False) as caller:
                url = recorder.public_base_url + "/v1/chat/completions"
                first = {
                    "messages": [{"role": "user", "content": "line\u2028separator"}],
                    "stream": True,
                    "tools": [{"type": "function", "function": {"name": "Search"}}],
                    "chat_template_kwargs": {"thinking": True},
                }
                raw = json.dumps(first, indent=2, ensure_ascii=False).encode()
                denied = await caller.post(url, content=raw, headers={"Authorization": "Bearer wrong"})
                assert denied.status_code == 401 and not calls
                response = await caller.post(
                    url,
                    content=raw,
                    headers={"Authorization": "Bearer " + context.session_id},
                )
                assert response.content == responses[0].content
                await recorder.wait_idle()
                second = copy.deepcopy(first)
                second["messages"] += [
                    canonical,
                    {"role": "tool", "tool_call_id": "call_1", "content": "found"},
                ]
                response = await caller.post(
                    url,
                    json=second,
                    headers={"Authorization": "Bearer " + context.session_id},
                )
                assert response.content == responses[1].content
                await recorder.wait_idle()
            candidate = recorder.capture.linear_candidate()
            assert candidate["messages"][-3] == canonical
            assert calls[0].content == raw
            assert str(calls[0].url) == context.base_url + "/v1/chat/completions"
            assert calls[0].headers["authorization"] == "Bearer " + context.session_id
            assert recorder.capture.records[0]["request_sha256"] == hashlib.sha256(raw).hexdigest()
            assert recorder.capture.records[0]["response_sha256"] == hashlib.sha256(responses[0].content).hexdigest()
            assert context.session_id not in json.dumps(recorder.capture.records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,code",
    [
        (b'data: {"choices": []}\n\n', "callback_not_buffered_sse"),
        (b"data: [DONE]\n\ndata: [DONE]\n\n", "callback_duplicate_done"),
        (b'data: {"error":"secret"}\n\ndata: [DONE]\n\n', "callback_error_response"),
    ],
)
async def test_callback_rejects_incomplete_or_error_sse(body, code):
    with pytest.raises(AdapterError, match=code):
        await assistant_from_response(body, "text/event-stream")


@pytest.mark.asyncio
async def test_malformed_second_callback_cannot_produce_partial_candidate(config, context):
    count = 0

    async def upstream(request):
        nonlocal count
        count += 1
        return completion(finish=[] if count == 2 else "stop")

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as backend:
        async with CallbackRecorder(config, context, client=backend) as recorder:
            async with httpx.AsyncClient(trust_env=False) as caller:
                for _ in range(2):
                    await caller.post(
                        recorder.public_base_url + "/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "task"}]},
                        headers={"Authorization": "Bearer " + context.session_id},
                    )
                await recorder.wait_idle()
            assert recorder.capture.started == 2 and len(recorder.capture.records) == 1
            assert recorder.capture.errors == ["callback_finish_reason_invalid"]
            with pytest.raises(AdapterError, match="capture_incomplete"):
                recorder.capture.linear_candidate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"api_key": "other-secret"},
        {"metadata": {"Authorization": "Bearer other"}},
        {"messages": [{"role": "tool", "content": '{"api_key":"other-secret"}'}]},
    ],
)
async def test_credentials_in_body_rejected_without_forwarding(config, context, extra):
    forwarded = []

    async def upstream(request):
        forwarded.append(request)
        return completion()

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as backend:
        async with CallbackRecorder(config, context, client=backend) as recorder:
            async with httpx.AsyncClient(trust_env=False) as caller:
                response = await caller.post(
                    recorder.public_base_url + "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "task"}], **extra},
                    headers={"Authorization": "Bearer " + context.session_id},
                )
            assert response.status_code == 400 and not forwarded and not recorder.capture.records
            assert "other-secret" not in json.dumps(recorder.capture.errors)


@pytest.mark.parametrize(
    "change,code",
    [
        ("history", "callback_history_not_linear"),
        ("tools", "callback_tools_or_template_changed"),
        ("template", "callback_tools_or_template_changed"),
        ("concurrent", "capture_incomplete"),
        ("missing_record", "capture_incomplete"),
    ],
)
def test_complex_contexts_fail_closed(change, code):
    assistant = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "reason",
    }
    first = {"messages": [{"role": "user", "content": "task"}], "tools": []}
    second = {
        "messages": first["messages"] + [assistant, {"role": "user", "content": "next"}],
        "tools": [],
    }
    capture = CallbackCapture(
        records=[
            {"index": 0, "request": first, "assistant": assistant},
            {"index": 1, "request": second, "assistant": assistant},
        ],
        started=2,
    )
    if change == "history":
        second["messages"] = [{"role": "user", "content": "different fork"}]
    elif change == "tools":
        second["tools"] = [{"type": "function"}]
    elif change == "template":
        second["chat_template_kwargs"] = {"thinking": False}
    elif change == "concurrent":
        capture.concurrent = True
    else:
        capture.started = 3
    with pytest.raises(AdapterError, match=code):
        capture.linear_candidate()


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", [False, True])
async def test_offline_end_to_end_drains_tail_and_never_claims_training(config, context, session_input, replay):
    posts = []
    cleanup_calls = []
    snapshot_raw = json.dumps({"data": {"private_business_data": "retained-private"}})
    snapshot = ("event: id.snapshot\ndata: " + snapshot_raw + "\n\n").encode()
    if replay:
        session_input["metadata"]["agentspark"]["replay_spec"] = {
            "messages": [{"role": "user", "content": "prior"}],
            "attributes": {
                key: {}
                for key in (
                    "ark.idindex.mapping",
                    "ark.idindex.image_urls",
                    "ark.idindex.reverse_mapping",
                )
            },
        }

    async def remote(request):
        if request.method == "GET":
            assert not replay
            return httpx.Response(200, json={"session_id": "remote-session"})
        body = json.loads(request.content)
        posts.append(body)
        if "replay_spec" in body:
            return httpx.Response(
                200,
                content=event("link.up.ack", code=0, data={"session_id": "remote-session"}),
            )
        llm = body["llm_config"]
        async with httpx.AsyncClient(trust_env=False) as caller:
            response = await caller.post(
                llm["base_url"] + llm["endpoint_path"],
                json={"messages": [{"role": "user", "content": "actual model prompt"}]},
                headers={"Authorization": "Bearer " + llm["api_key"]},
            )
            assert response.status_code == 200
        return httpx.Response(200, content=event("agent.stream.end") + snapshot.rstrip(b"\n"))

    async def cleanup(info):
        cleanup_calls.append(info)

    async with httpx.AsyncClient(transport=httpx.MockTransport(remote)) as spark:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: completion())) as backend:
            report = await run_audit(
                config,
                context,
                session_input,
                agentspark_http=spark,
                relax_http=backend,
                cleanup_hook=cleanup,
            )
    assert report["audit_status"] == "collected" and report["training_ready"] is False
    assert report["session_source"] == ("replay_ack" if replay else "get_session")
    assert report["send_attempts"] == 1 and len(posts) == 1 + int(replay)
    assert report["stream"]["eof"] and len(report["stream"]["events"]) == 2
    assert (
        report["stream"]["events"][-1]["sha256"]
        == hashlib.sha256(snapshot.decode().split("data: ")[1].strip().encode()).hexdigest()
    )
    assert report["linear_context_candidate"]["messages"][0]["content"] == "actual model prompt"
    assert len(cleanup_calls) == 1 and cleanup_calls[0]["reason"] == "completed"
    assert report["cleanup"]["remote"] == "hook_completed_remote_state_unverified"
    assert "v21_reward_and_token_credit" in report["pending_gates"]
    assert "reward" not in report and "exports" not in report
    serialized = json.dumps(report)
    assert context.session_id not in serialized and "retained-private" in serialized and "api_key" not in serialized
    assert report["stream"]["events"][-1]["event"] == "id.snapshot"
    assert report["source_input"] == session_input
    assert report["identity"] == {
        "group_id": "group",
        "mode": "train",
        "model": "fixture-model",
    }


class BrokenStream(httpx.AsyncByteStream):
    def __init__(self, *, wait=None):
        self.closed = False
        self.wait = wait

    async def __aiter__(self):
        yield event("agent.stream.end")
        if self.wait is not None:
            self.wait.set()
            await asyncio.Future()
        raise httpx.ReadError("credential-hidden-transport-detail")

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["disconnect", "late_error", "no_end", "cancel", "cleanup_fail", "cleanup_timeout"],
)
async def test_failures_no_retry_and_cleanup(config, context, session_input, mode):
    calls = []
    cleanup_calls = []
    stop = asyncio.Event()
    stream = BrokenStream(wait=stop if mode == "cancel" else None)

    async def remote(request):
        if request.method == "GET":
            return httpx.Response(200, json={"session_id": "remote"})
        calls.append(request)
        if mode in {"disconnect", "cancel"}:
            return httpx.Response(200, stream=stream)
        body = event("agent.stream.end") + event("error") if mode == "late_error" else b""
        return httpx.Response(200, content=body)

    async def cleanup(info):
        cleanup_calls.append(info)
        if mode == "cleanup_fail":
            raise RuntimeError("secret detail")
        if mode == "cleanup_timeout":
            await asyncio.Future()

    async with httpx.AsyncClient(transport=httpx.MockTransport(remote)) as spark:
        report = await run_audit(
            config,
            context,
            session_input,
            agentspark_http=spark,
            cleanup_hook=cleanup,
            stop_event=stop if mode == "cancel" else None,
        )
    assert report["audit_status"] == "failed" and report["send_attempts"] == 1 and len(calls) == 1
    assert len(cleanup_calls) == 1 and report["cleanup"]["local"] == "closed"
    assert "linear_context_candidate" not in report and "secret" not in json.dumps(report)
    if mode in {"disconnect", "cancel"}:
        assert stream.closed
    if mode == "cancel":
        assert cleanup_calls[0]["reason"] == "cancelled"
    if mode == "late_error":
        assert report["stream"]["saw_end"] and report["stream"]["saw_error"] and report["stream"]["eof"]
    if mode == "cleanup_timeout":
        assert report["cleanup"]["remote"] == "hook_timed_out_remote_state_unknown"
    if mode == "cleanup_fail":
        assert report["cleanup"]["remote"] == "hook_failed_remote_state_unknown"


def test_private_atomic_audit_never_overwrites_relax_output(tmp_path, context):
    output = tmp_path / "output.json"
    output.write_text("existing training output")
    alias = tmp_path / "alias.json"
    alias.symlink_to(output)
    for path in (output, alias):
        with pytest.raises(AdapterError, match="audit_must_not_be_relax_output"):
            write_audit({}, path, output, context.session_id)
    audit = tmp_path / "audit.json"
    write_audit({"training_ready": False}, audit, output, context.session_id)
    assert audit.stat().st_mode & 0o777 == 0o600
    with pytest.raises(AdapterError, match="already_exists"):
        write_audit({}, audit, output, context.session_id)
    assert output.read_text() == "existing training output"
    with pytest.raises(AdapterError, match="credential"):
        write_audit(
            {"data": context.session_id},
            tmp_path / "leak.json",
            output,
            context.session_id,
        )
    assert not (tmp_path / "leak.json").exists()


def test_cli_success_nonzero_prevents_implicit_export(tmp_path, monkeypatch, config, context, session_input):
    from examples.agentspark_agentic import client

    config_path, input_path, output, audit = [
        tmp_path / name for name in ("config.json", "input.json", "output.json", "audit.json")
    ]
    config_path.write_text(json.dumps(asdict(config)))
    input_path.write_text(json.dumps(session_input))
    env = {
        "RELAX_BASE_URL": context.base_url,
        "RELAX_SESSION_ID": context.session_id,
        "RELAX_GROUP_ID": context.group_id,
        "RELAX_ROLLOUT_MODE": context.mode,
        "RELAX_SESSION_IO_DIR": context.invocation_id,
        "RELAX_INPUT_JSON": str(input_path),
        "RELAX_OUTPUT_JSON": str(output),
    }

    async def collected(*args, **kwargs):
        assert kwargs == {"allow_dynamic_images": False}
        return {"audit_status": "collected", "training_ready": False}

    monkeypatch.setattr(client, "_run_cli", collected)
    assert main(["--config", str(config_path), "--audit-output", str(audit)], env=env) == AUDIT_ONLY_EXIT == 3
    assert audit.exists() and not output.exists()
    assert main(["--config", str(config_path), "--audit-output", str(output)], env=env) == 2
    assert not output.exists()


@pytest.mark.asyncio
async def test_late_callback_during_cleanup_invalidates_candidate(config, context, session_input):
    callback = {}

    async def remote(request):
        if request.method == "GET":
            return httpx.Response(200, json={"session_id": "remote"})
        callback.update(json.loads(request.content)["llm_config"])
        async with httpx.AsyncClient(trust_env=False) as caller:
            await caller.post(
                callback["base_url"] + callback["endpoint_path"],
                json={"messages": [{"role": "user", "content": "valid"}]},
                headers={"Authorization": "Bearer " + context.session_id},
            )
        return httpx.Response(200, content=event("agent.stream.end"))

    async def cleanup(info):
        async with httpx.AsyncClient(trust_env=False) as caller:
            response = await caller.post(
                callback["base_url"] + callback["endpoint_path"],
                json={"messages": "invalid"},
                headers={"Authorization": "Bearer " + context.session_id},
            )
            assert response.status_code == 400

    async with httpx.AsyncClient(transport=httpx.MockTransport(remote)) as spark:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: completion())) as backend:
            report = await run_audit(
                config,
                context,
                session_input,
                agentspark_http=spark,
                relax_http=backend,
                cleanup_hook=cleanup,
            )
    assert report["audit_status"] == "failed" and report["callbacks"]["errors"]
    assert "linear_context_candidate" not in report


@pytest.mark.asyncio
async def test_replay_bad_ack_is_unknown_remote_state(config, context, session_input):
    session_input["metadata"]["agentspark"]["replay_spec"] = {
        "messages": [{"role": "user", "content": "prior"}],
        "attributes": {
            key: {}
            for key in (
                "ark.idindex.mapping",
                "ark.idindex.image_urls",
                "ark.idindex.reverse_mapping",
            )
        },
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b""))) as spark:
        report = await run_audit(config, context, session_input, agentspark_http=spark)
    assert report["replay_attempted"] and report["send_attempts"] == 0
    assert report["remote_execution"] == "unknown"
    assert report["cleanup"]["remote"] == "not_configured_remote_state_unknown"


@pytest.mark.asyncio
async def test_local_close_failure_not_claimed_closed(config, context, session_input, monkeypatch):
    original = CallbackRecorder.__aexit__

    async def fail_close(self, *args):
        await original(self, *args)
        self._runner = object()  # A failed resource teardown is visible to the state predicate.
        raise RuntimeError("secret cleanup detail")

    monkeypatch.setattr(CallbackRecorder, "__aexit__", fail_close)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as spark:
        report = await run_audit(config, context, session_input, agentspark_http=spark)
    assert report["cleanup"]["local"] == "close_failed"
    assert report["audit_status"] == "failed"
    assert "secret cleanup detail" not in json.dumps(report)


@pytest.mark.asyncio
async def test_stream_credential_frame_is_omitted_but_tail_drained(config, context, session_input):
    async def remote(request):
        if request.method == "GET":
            return httpx.Response(200, json={"session_id": "remote"})
        return httpx.Response(
            200,
            content=event("agent.stream.end") + event("secret", api_key="hidden") + event("tail", data="kept"),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(remote)) as spark:
        report = await run_audit(config, context, session_input, agentspark_http=spark)
    assert report["audit_status"] == "failed" and report["stream"]["eof"]
    assert report["stream"]["events"][1]["payload_omitted"] == "credential_check_failed"
    assert report["stream"]["events"][-1]["payload"]["data"] == "kept"
    assert "hidden" not in json.dumps(report)


@pytest.mark.asyncio
async def test_remote_error_body_never_persisted(config, context, session_input):
    async def remote(request):
        if request.method == "GET":
            return httpx.Response(200, json={"session_id": "remote"})
        body = event("agent.stream.end") + event("error", message="Authorization: Bearer OTHER_SECRET /private/path")
        return httpx.Response(200, content=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(remote)) as spark:
        report = await run_audit(config, context, session_input, agentspark_http=spark)
    assert report["failure_code"] == "agentspark_stream_error"
    assert report["stream"]["events"][-1]["payload_omitted"] == "remote_error_body"
    assert "OTHER_SECRET" not in json.dumps(report) and "/private/path" not in json.dumps(report)


@pytest.mark.asyncio
async def test_remote_error_keeps_only_known_structured_diagnostics(config, context, session_input):
    async def remote(request):
        if request.method == "GET":
            return httpx.Response(200, json={"session_id": "remote"})
        return httpx.Response(200, content=event("error", data={"error": {
            "code": "context_length_exceeded", "type": "invalid_request_error", "status_code": 400,
            "message": context.session_id + " https://private.invalid /private/file data:image/png;base64,SECRET",
            "request_id": "SECRET", "authorization": "SECRET",
        }}))

    async with httpx.AsyncClient(transport=httpx.MockTransport(remote)) as spark:
        report = await run_audit(config, context, session_input, agentspark_http=spark)
    assert report["stream"]["events"][0]["diagnostic"] == {
        "code": "context_length_exceeded", "type": "invalid_request_error", "status": 400,
    }
    rendered = json.dumps(report)
    assert all(secret not in rendered for secret in ("SECRET", "private.invalid", "/private/file", context.session_id))


@pytest.mark.asyncio
@pytest.mark.parametrize("sse,status", [(False, 200), (True, 200), (False, 503)])
async def test_callback_error_response_retains_safe_upstream_fields(config, context, sse, status):
    payload = {"error": {
        "code": "context_length_exceeded", "type": "invalid_request_error",
        "message": "TOP_SECRET https://private.invalid /private/file", "param": "TOP_SECRET",
    }}
    body = ("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n").encode() if sse else json.dumps(payload).encode()
    response = httpx.Response(status, content=body, headers={
        "content-type": "text/event-stream" if sse else "application/json",
    })
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as backend:
        async with CallbackRecorder(config, context, client=backend) as recorder:
            async with httpx.AsyncClient(trust_env=False) as caller:
                observed = await caller.post(
                    recorder.public_base_url + "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "task"}]},
                    headers={"Authorization": "Bearer " + context.session_id},
                )
            await recorder.wait_idle()
            code = "callback_error_response" if status == 200 else "callback_upstream_http_error"
            assert observed.content == body and observed.status_code == status
            assert recorder.capture.errors == [code] and recorder.capture.records == []
            assert recorder.capture.diagnostics == [{
                "index": 0, "code": code, "http_status": status,
                "upstream": {"code": "context_length_exceeded", "type": "invalid_request_error"},
            }]
            assert "TOP_SECRET" not in json.dumps(recorder.capture.diagnostics)


@pytest.mark.asyncio
async def test_dynamic_images_cannot_bypass_media_plan(config, context, session_input):
    with pytest.raises(AdapterError, match="^media_plan_required$"):
        await run_audit(config, context, session_input, allow_dynamic_images=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "upstream", "assistant_image"])
async def test_dynamic_callback_segment_commit_is_transactional(config, context, tmp_path, failure):
    from PIL import Image

    from tests.cpu.examples.agentspark_agentic.test_media import dynamic_plan, image_part, image_uri, images

    paths = [tmp_path / "red.png", tmp_path / "blue.png"]
    for path, color in zip(paths, ["red", "blue"], strict=True):
        Image.new("RGB", (2, 2), color).save(path)
    red, blue = [image_uri(path) for path in paths]
    media = dynamic_plan()
    responses = [completion()]
    if failure == "upstream":
        responses.append(httpx.Response(503, json={"error": {"code": "no_available_workers"}}))
    elif failure == "assistant_image":
        responses.append(completion({"role": "assistant", "content": [image_part(blue)]}))
    else:
        responses.append(completion())
    calls = []

    def backend(request):
        calls.append(json.loads(request.content))
        return responses[len(calls) - 1]

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        async with CallbackRecorder(
            config, context, client=http, media=media, restore_reasoning=True, allow_dynamic_images=True
        ) as recorder:
            async with httpx.AsyncClient(trust_env=False) as caller:
                url = recorder.public_base_url + "/v1/chat/completions"
                headers = {"Authorization": "Bearer " + context.session_id}
                await caller.post(url, json={"messages": [{"role": "user", "content": [image_part(red)]}]}, headers=headers)
                await recorder.wait_idle()
                before_history = copy.deepcopy(recorder.history.messages)
                before_media = copy.deepcopy(media._committed_messages)
                before_revision = media._revision
                await caller.post(url, json={"messages": [{"role": "user", "content": [image_part(blue)]}]}, headers=headers)
                await recorder.wait_idle()
            assert images(calls[0]) == [red] and images(calls[1]) == [blue]
            if failure:
                assert len(recorder.capture.records) == 1 and recorder.capture.errors
                assert recorder.history.messages == before_history
                assert media._committed_messages == before_media and media._revision == before_revision
            else:
                assert len(recorder.capture.records) == 2 and not recorder.capture.errors
                assert recorder.capture.records[1]["media"]["dynamic_images"]["segment_reset"]
                assert images({"messages": media._committed_messages}) == [blue]
