# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Collect upstream evidence without changing replay admission decisions."""

import hashlib
import json

import httpx
import pytest

from examples.agentspark_agentic.diagnostics import failure_summary
from examples.agentspark_agentic.protocol import AdapterConfig, AdapterError
from examples.agentspark_agentic.transport import AgentSparkClient, ReplayUnavailableError


@pytest.mark.asyncio
@pytest.mark.parametrize("known", [False, True])
async def test_error_observation_does_not_expand_replay_retry_classification(known):
    payload = (
        {"message": "Eden SSE rate limiter unavailable: total"}
        if known
        else {
            "status": 503,
            "error_detail": json.dumps({"message": "too many requests PRIVATE_CONTENT https://private.invalid"}),
        }
    )
    wire = "event: error\ndata: " + json.dumps(payload) + "\n\n"
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text=wire, headers={"content-type": "text/event-stream"})

    config = AdapterConfig.from_payload(
        {
            "base_url": "http://agentspark.invalid",
            "model": "test",
            "agent": "test",
            "source": "eden",
            "max_tokens": 10,
            "callback_bind_host": "127.0.0.1",
            "callback_bind_port": 0,
            "callback_public_base_url": "http://127.0.0.1:1",
        }
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        remote = AgentSparkClient(config, client=http)
        with pytest.raises(AdapterError) as failure:
            await remote.restore({"replay_spec": {"replay_type": "restore_only"}})
    assert isinstance(failure.value, ReplayUnavailableError) == known
    assert failure.value.code == ("replay_temporarily_unavailable" if known else "replay_error_event")
    assert len(calls) == 1
    assert len(remote.error_observations) == 1
    observation = remote.error_observations[0]
    assert observation["request_started_unix_ms"] <= observation["error_unix_ms"]
    assert observation["elapsed_ms"] >= 0
    assert observation["frames_before_error"] == 0
    expected_digest = hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()
    assert observation["transport_http_status"] == 200
    assert observation["event_raw_sha256"] == expected_digest
    if not known:
        assert observation["detail"]["http_statuses"] == [200, 503]
    record = failure_summary(failure.value, audit={"upstream_observations": remote.error_observations})
    assert len(record["upstream_observations"]) == 1
    assert record["upstream_observations"][0]["transport_http_status"] == 200
    assert record["upstream_observations"][0]["event_raw_sha256"] == expected_digest
    encoded = json.dumps(record)
    assert "PRIVATE_CONTENT" not in encoded
    assert "private.invalid" not in encoded


def test_failure_observation_rejects_untrusted_extra_fields():
    from examples.agentspark_agentic.error_observation import observe_error_payload

    detail = observe_error_payload({"message": "timeout"}, http_status=200, phase="replay")
    detail["message"] = "PRIVATE_MESSAGE"
    row = {"detail": detail, "request_started_unix_ms": True, "elapsed_ms": "PRIVATE_SECRET", "url": "PRIVATE_URL"}
    result = failure_summary(AdapterError("replay_error_event"), audit={"upstream_observations": [row] * 10})
    assert len(result["upstream_observations"]) == 4
    assert "PRIVATE" not in json.dumps(result)
    for observation in result["upstream_observations"]:
        assert "request_started_unix_ms" not in observation


@pytest.mark.parametrize(
    ("status", "digest"),
    [(99, "g" * 64), (600, "f" * 63), (True, {"token": "PRIVATE_SECRET"}), ("200", "https://private.invalid")],
)
def test_failure_observation_rejects_invalid_transport_status_and_raw_digest(status, digest):
    from examples.agentspark_agentic.error_observation import observe_error_payload

    observation = {
        "detail": observe_error_payload({"message": "timeout"}, phase="send"),
        "transport_http_status": status,
        "event_raw_sha256": digest,
    }
    result = failure_summary(AdapterError("agentspark_stream_error"), audit={"upstream_observations": [observation]})
    persisted = result["upstream_observations"][0]
    assert "transport_http_status" not in persisted
    assert "event_raw_sha256" not in persisted
    assert "PRIVATE_SECRET" not in json.dumps(result)
    assert "private.invalid" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503])
async def test_non_2xx_observes_status_without_reading_body_or_changing_failure(status):
    reads = []

    class ResponseBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            reads.append(True)
            yield b"PRIVATE_RESPONSE_BODY"

    config = AdapterConfig.from_payload(
        {
            "base_url": "http://agentspark.invalid",
            "model": "test",
            "agent": "test",
            "source": "eden",
            "max_tokens": 10,
            "callback_bind_host": "127.0.0.1",
            "callback_bind_port": 0,
            "callback_public_base_url": "http://127.0.0.1:1",
        }
    )
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, stream=ResponseBody())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        remote = AgentSparkClient(config, client=http)
        with pytest.raises(AdapterError, match="agentspark_http_error") as failure:
            await remote.restore({"replay_spec": {"replay_type": "restore_only"}})
    assert len(calls) == 1
    assert reads == []
    assert not isinstance(failure.value, ReplayUnavailableError)
    record = failure_summary(failure.value, audit={"upstream_observations": remote.error_observations})
    assert len(record["upstream_observations"]) == 1
    observation = record["upstream_observations"][0]
    assert observation["transport_http_status"] == status
    assert observation["frames_before_error"] == 0
    assert observation["detail"]["phase"] == "replay"
    assert observation["detail"]["http_statuses"] == [status]
    assert "event_raw_sha256" not in observation
    assert "PRIVATE_RESPONSE_BODY" not in json.dumps(record)
