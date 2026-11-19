# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preparation bounds must never time model/barrier waits or resubmit a
Send."""

# Imported fixtures intentionally share names with their test parameters.
# ruff: noqa: F811

import asyncio
import copy
import json
from dataclasses import asdict, replace

import httpx
import pytest

from examples.agentspark_agentic import client
from examples.agentspark_agentic.protocol import AdapterConfig, AdapterError
from tests.cpu.examples.agentspark_agentic.test_replay_resilience import (  # noqa: F401
    ACK,
    END,
    REJECTION,
    config,
    context,
    recorder,
    source,
)


@pytest.mark.parametrize("field", ["session_lookup_timeout_s", "first_callback_timeout_s"])
@pytest.mark.parametrize("value", [0, -1, True, None, "30", float("inf"), float("nan")])
def test_preparation_deadlines_require_finite_positive_seconds(config, field, value):
    with pytest.raises(AdapterError, match=f"^config_{field}_invalid$"):
        AdapterConfig.from_payload({**asdict(config), field: value})


@pytest.mark.parametrize("value", [-1, True, None, "30", float("inf"), float("nan")])
def test_start_spread_requires_finite_nonnegative_seconds(config, value):
    with pytest.raises(AdapterError, match="^config_prepare_start_spread_s_invalid$"):
        AdapterConfig.from_payload({**asdict(config), "prepare_start_spread_s": value})


def test_preparation_defaults_and_explicit_zero_spread(config):
    assert config.prepare_start_spread_s == 0.0
    assert config.session_lookup_timeout_s == 30.0
    assert config.first_callback_timeout_s == 120.0
    assert AdapterConfig.from_payload({**asdict(config), "prepare_start_spread_s": 0}).prepare_start_spread_s == 0.0
    assert AdapterConfig.from_payload({**asdict(config), "prepare_start_spread_s": 2.5}).prepare_start_spread_s == 2.5


@pytest.mark.asyncio
async def test_start_spread_happens_once_before_restore_retries(config, context, source, monkeypatch):
    choices = []
    requests = []
    config = replace(config, prepare_start_spread_s=0.004)

    def uniform(low, high):
        choices.append((low, high))
        return high

    monkeypatch.setattr(client.random, "uniform", uniform)

    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("replay_spec"):
            return httpx.Response(200, content=REJECTION if len(requests) == 1 else ACK)
        return httpx.Response(200, content=END)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        report = await client.run_audit(config, context, source, agentspark_http=http)
    assert report["audit_status"] == "collected"
    assert choices == [(0.0, 0.004), (0.75, 1.25)]
    assert report["prepare_start_delay_s"] == 0.004
    assert requests[0] == requests[1]
    assert report["replay_attempts"] == 2
    assert report["send_attempts"] == 1


@pytest.mark.asyncio
async def test_stop_interrupts_start_spread_before_any_remote_request(config, context, source, monkeypatch):
    stopped = asyncio.Event()
    config = replace(config, prepare_start_spread_s=30)
    monkeypatch.setattr(client.random, "uniform", lambda _low, high: high)
    asyncio.get_running_loop().call_later(0.01, stopped.set)

    async def handle(_request):
        pytest.fail("the cancelled startup spread must not submit a request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        report = await asyncio.wait_for(
            client.run_audit(config, context, source, agentspark_http=http, stop_event=stopped), timeout=1
        )
    assert report["failure_code"] == "audit_cancelled"
    assert report["replay_attempts"] == report["send_attempts"] == 0
    assert report["cleanup"]["local"] == "closed"


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_hold", [False, True])
async def test_session_lookup_timeout_closes_locally_without_send_or_retry(config, context, source, legacy_hold):
    source = copy.deepcopy(source)
    del source["metadata"]["agentspark"]["replay_spec"]
    config = replace(config, session_lookup_timeout_s=0.01)
    requests = []
    cancelled = asyncio.Event()

    async def handle(request):
        requests.append(request)
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        report = await asyncio.wait_for(
            client.run_audit(config, context, source, agentspark_http=http, hold_on_replay_exhaustion=legacy_hold),
            timeout=1,
        )
    assert report["failure_code"] == "session_lookup_timeout"
    assert len(requests) == 1 and requests[0].method == "GET"
    assert cancelled.is_set()
    assert report["send_attempts"] == report["replay_attempts"] == 0
    assert report["cleanup"]["local"] == "closed"


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_hold", [False, True])
async def test_first_callback_deadline_cancels_one_ambiguous_send(config, context, source, legacy_hold):
    config = replace(config, first_callback_timeout_s=0.01)
    requests = []
    cancelled = asyncio.Event()

    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("replay_spec"):
            return httpx.Response(200, content=ACK)
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        report = await asyncio.wait_for(
            client.run_audit(config, context, source, agentspark_http=http, hold_on_replay_exhaustion=legacy_hold),
            timeout=1,
        )
    assert report["failure_code"] == "first_callback_timeout"
    assert len(requests) == 2
    assert cancelled.is_set()
    assert report["send_attempts"] == report["replay_attempts"] == 1
    assert report["callbacks"]["started"] == 0
    assert report["remote_execution"] == "unknown"
    assert report["cleanup"]["local"] == "closed"


@pytest.mark.asyncio
async def test_first_callback_disables_deadline_during_long_model_and_barrier_wait(
    config, context, source, recorder, monkeypatch
):
    config = replace(config, first_callback_timeout_s=0.01)
    instances = []
    released = asyncio.Event()
    callback = asyncio.Event()
    requests = []

    class ObservedRecorder(recorder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr(client, "CallbackRecorder", ObservedRecorder)

    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("replay_spec"):
            return httpx.Response(200, content=ACK)
        instances[0].first_callback_started.set()
        callback.set()
        await released.wait()
        return httpx.Response(200, content=END)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        task = asyncio.create_task(client.run_audit(config, context, source, agentspark_http=http))
        try:
            await asyncio.wait_for(callback.wait(), timeout=1)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
            assert not task.done()
            released.set()
            report = await asyncio.wait_for(task, timeout=1)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert report["audit_status"] == "collected"
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_stop_still_cancels_send_after_first_callback(config, context, source, recorder, monkeypatch):
    stopped = asyncio.Event()
    cancelled = asyncio.Event()
    instances = []

    class ObservedRecorder(recorder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr(client, "CallbackRecorder", ObservedRecorder)

    async def handle(request):
        body = json.loads(request.content)
        if body.get("replay_spec"):
            return httpx.Response(200, content=ACK)
        instances[0].first_callback_started.set()
        asyncio.get_running_loop().call_later(0.01, stopped.set)
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        report = await asyncio.wait_for(
            client.run_audit(config, context, source, agentspark_http=http, stop_event=stopped), timeout=1
        )
    assert report["failure_code"] == "audit_cancelled"
    assert cancelled.is_set()
    assert report["send_attempts"] == 1
    assert report["cleanup"]["local"] == "closed"


@pytest.mark.parametrize(
    "code,phase", [("session_lookup_timeout", "session_lookup"), ("first_callback_timeout", "send")]
)
def test_preparation_timeout_diagnostics_keep_fixed_code_and_phase(context, code, phase):
    from examples.agentspark_agentic.diagnostics import failure_summary

    summary = failure_summary(
        AdapterError(code), context=context, audit={"failure_code": code, "failure_phase": phase}
    )
    assert summary["error"] == summary["audit_failure_code"] == code
    assert summary["upstream_errors"] == [{"origin": "agentspark_" + phase, "phase": phase}]
