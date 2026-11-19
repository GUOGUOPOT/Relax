# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Recovery preserves the original invocation and never repeats a formal
Send."""

import asyncio
import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import httpx
import pytest

from examples.agentspark_agentic import client, production
from examples.agentspark_agentic.protocol import AdapterConfig, AdapterError, RelaxContext


@pytest.fixture
def config():
    return AdapterConfig.from_payload(
        {
            "base_url": "https://agent.invalid/service",
            "agent": "diandian",
            "source": "eden",
            "model": "fixture",
            "max_tokens": 32,
            "callback_bind_host": "127.0.0.1",
            "callback_bind_port": 0,
            "callback_public_base_url": "http://127.0.0.1:0",
            "replay_max_attempts": 3,
            "replay_initial_backoff_s": 0.001,
            "replay_max_backoff_s": 0.002,
        }
    )


@pytest.fixture
def context():
    return RelaxContext(
        "http://relax.invalid", "fixture-private-credential", "train:prepare_group_9", "train", "/io/9"
    )


@pytest.fixture
def source():
    return {
        "metadata": {
            "agentspark": {
                "schema_version": 1,
                "inputs": [{"type": "TEXT", "info": {"text": "original task"}}],
                "tools": ["Search"],
                "prompt_spec": {"system_prompt": "system", "user_prompt": "user"},
                "replay_spec": {
                    "messages": [{"role": "user", "content": "history"}],
                    "attributes": {
                        "ark.idindex.mapping": {},
                        "ark.idindex.image_urls": {},
                        "ark.idindex.reverse_mapping": {},
                    },
                },
            }
        }
    }


@pytest.fixture(autouse=True)
def recorder(monkeypatch):
    class Recorder:
        def __init__(self, *args, **kwargs):
            self.public_base_url = "http://callback.invalid"
            self.closed = False
            self.first_callback_started = asyncio.Event()
            self.capture = SimpleNamespace(
                records=[],
                errors=[],
                diagnostics=[],
                started=0,
                concurrent=False,
                linear_candidate=lambda: {"messages": []},
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.closed = True

        async def wait_idle(self):
            pass

    monkeypatch.setattr(client, "CallbackRecorder", Recorder)
    return Recorder


def frame(payload, event=None):
    return ((f"event: {event}\n" if event else "") + "data: " + json.dumps(payload) + "\n\n").encode()


REJECTION = frame({"message": "Eden SSE rate limiter unavailable: total"}, "error")
ACK = frame({"type": "link.up.ack", "code": 0, "data": {"session_id": "restored"}})
END = frame({"type": "agent.stream.end"})


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_hold,refusals", [(False, 2), (True, 2)])
async def test_replay_recovers_same_input_then_sends_once(config, context, source, legacy_hold, refusals):
    requests = []
    original = copy.deepcopy(source)

    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("replay_spec"):
            return httpx.Response(200, content=REJECTION if len(requests) <= refusals else ACK)
        return httpx.Response(200, content=END)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        report = await client.run_audit(
            config,
            context,
            source,
            agentspark_http=http,
            hold_on_replay_exhaustion=legacy_hold,
        )
    assert report["audit_status"] == "collected"
    assert report["replay_attempts"] == refusals + 1
    assert report["send_attempts"] == 1
    assert len(requests) == refusals + 2
    assert all(body == requests[0] for body in requests[:-1])
    assert requests[-1]["request_id"] == context.request_id
    assert source == original == report["source_input"]
    assert report["cleanup"]["local"] == "closed"


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_hold", [False, True])
async def test_replay_exhaustion_is_explicit_and_never_sends_or_scores(config, context, source, tmp_path, legacy_hold):
    requests = []

    async def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=REJECTION)

    async def scorer(_audit):
        pytest.fail("a failed replay cannot be scored")

    output = tmp_path / "output.jsonl"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        with pytest.raises(AdapterError, match="^replay_retry_exhausted$"):
            await production.run_training(
                config,
                context,
                source,
                compiler_defaults={},
                scorer=scorer,
                audit_path=tmp_path / "audit.json",
                output_path=output,
                agentspark_http=http,
                hold_on_replay_exhaustion=legacy_hold,
            )
    assert len(requests) == config.replay_max_attempts
    assert all(body == requests[0] for body in requests)
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["send_attempts"] == 0
    assert audit["production"]["error"] == "replay_retry_exhausted"
    assert audit["failure_diagnostic"]["reason_code"] == "eden_sse_rate_limiter_unavailable"
    assert not output.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_hold", [False, True])
@pytest.mark.parametrize(
    "message,reason",
    [
        ("Eden SSE rate limiter unavailable: total", "eden_sse_rate_limiter_unavailable"),
        ("Eden SSE rate limit exceeded: total", "eden_sse_rate_limit_exceeded_total"),
        ("Eden SSE rate limit exceeded: source", "eden_sse_rate_limit_exceeded_source"),
    ],
)
async def test_send_error_with_zero_callbacks_is_never_replayed(config, context, source, legacy_hold, message, reason):
    requests = []

    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, content=ACK if body.get("replay_spec") else frame({"message": message}, "error"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        report = await client.run_audit(
            config, context, source, agentspark_http=http, hold_on_replay_exhaustion=legacy_hold
        )
    assert len(requests) == 2
    assert report["callbacks"]["started"] == 0
    assert report["failure_code"] == "agentspark_dependency_unavailable"
    assert report["send_attempts"] == report["replay_attempts"] == 1
    assert report["stream"]["events"][0]["diagnostic"]["reason_code"] == reason


@pytest.mark.asyncio
async def test_stop_interrupts_legacy_backoff_without_consuming_another_attempt(config, context, source):
    stopped = asyncio.Event()
    requests = []

    async def handle(request):
        requests.append(request)
        asyncio.get_running_loop().call_later(0.01, stopped.set)
        return httpx.Response(200, content=REJECTION)

    config = replace(config, replay_max_attempts=3, replay_initial_backoff_s=30, replay_max_backoff_s=30)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        report = await asyncio.wait_for(
            client.run_audit(
                config,
                context,
                source,
                agentspark_http=http,
                stop_event=stopped,
                hold_on_replay_exhaustion=True,
            ),
            timeout=1,
        )
    assert len(requests) == 1
    assert report["failure_code"] == "audit_cancelled"
    assert report["cleanup"]["local"] == "closed"


@pytest.mark.asyncio
async def test_ambiguous_restore_timeout_is_bounded_and_not_retried(config, context, source):
    requests = []

    async def handle(request):
        requests.append(request)
        await asyncio.sleep(60)

    config = replace(config, replay_attempt_timeout_s=0.01)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        report = await client.run_audit(config, context, source, agentspark_http=http)
    assert len(requests) == 1
    assert report["failure_code"] == "replay_attempt_timeout"
    assert report["send_attempts"] == 0


def test_independent_endpoint_pin_survives_candidate_drift_without_overriding_other_settings(config, tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(asdict(config)))
    assert AdapterConfig.from_file(path) == config
    pin = path.with_name(path.name + ".endpoint-pin.json")
    pin.write_text(json.dumps({"schema_version": 1, "base_url": config.base_url}))
    drifted = {**asdict(config), "base_url": "https://unexpected.invalid", "max_tokens": 64}
    path.write_text(json.dumps(drifted))
    actual = AdapterConfig.from_file(path)
    assert actual.base_url == config.base_url
    assert actual.max_tokens == 64
    assert json.loads(path.read_text())["base_url"] == drifted["base_url"]


@pytest.mark.parametrize(
    "pin",
    [
        {"schema_version": True, "base_url": "https://pin.invalid"},
        {"schema_version": 1, "base_url": "https://user:secret@pin.invalid"},
        {"schema_version": 1, "base_url": "https://pin.invalid", "unknown": True},
    ],
)
def test_invalid_endpoint_pin_fails_closed(config, tmp_path, pin):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(asdict(config)))
    path.with_name(path.name + ".endpoint-pin.json").write_text(json.dumps(pin))
    with pytest.raises(AdapterError, match="^endpoint_pin_invalid$"):
        AdapterConfig.from_file(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("replay_max_attempts", False),
        ("replay_max_attempts", 0),
        ("replay_initial_backoff_s", 0),
        ("replay_max_backoff_s", float("inf")),
        ("replay_attempt_timeout_s", None),
    ],
)
def test_retry_configuration_rejects_invalid_bounds(config, field, value):
    with pytest.raises(AdapterError):
        AdapterConfig.from_payload({**asdict(config), field: value})
