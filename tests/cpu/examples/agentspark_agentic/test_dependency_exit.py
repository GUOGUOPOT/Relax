# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Runner capability handshake and bounded legacy dependency failures."""

# Imported pytest fixtures intentionally share names with their test parameters.
# ruff: noqa: F811

import asyncio
import json
from dataclasses import asdict, replace

import httpx
import pytest

from examples.agentspark_agentic import client, production, v21_reward
from examples.agentspark_agentic.protocol import AdapterError
from tests.cpu.examples.agentspark_agentic.test_replay_resilience import (  # noqa: F401
    ACK,
    REJECTION,
    config,
    context,
    recorder,
    source,
)


@pytest.mark.parametrize(
    "capability,code,expected_exit,expected_hold",
    [
        ("1", "replay_retry_exhausted", 75, False),
        ("1", "replay_attempt_timeout", 75, False),
        ("1", "agentspark_dependency_unavailable", 75, False),
        ("1", "session_lookup_timeout", 75, False),
        ("1", "first_callback_timeout", 75, False),
        (None, "session_lookup_timeout", 2, True),
        (None, "first_callback_timeout", 2, True),
        ("1", "replay_error_event", 2, False),
        ("1", "agentspark_stream_error", 2, False),
        (None, "replay_retry_exhausted", 2, True),
    ],
)
def test_main_negotiates_dependency_exit_without_publishing_output(
    config,
    context,
    source,
    tmp_path,
    monkeypatch,
    capsys,
    capability,
    code,
    expected_exit,
    expected_hold,
):
    config_path = tmp_path / "config.json"
    input_path = tmp_path / "input.json"
    defaults_path = tmp_path / "compiler-defaults.json"
    output_path = tmp_path / "output.jsonl"
    config_path.write_text(json.dumps(asdict(config)))
    input_path.write_text(json.dumps(source))
    defaults_path.write_text("{}")
    calls = []

    async def fail_training(*args, **kwargs):
        calls.append(kwargs)
        raise AdapterError(code)

    monkeypatch.setattr(production, "run_training", fail_training)
    monkeypatch.setattr(v21_reward, "load_reward_contract", lambda _path: {})
    env = {
        "RELAX_BASE_URL": context.base_url,
        "RELAX_SESSION_ID": context.session_id,
        "RELAX_GROUP_ID": context.group_id,
        "RELAX_ROLLOUT_MODE": context.mode,
        "RELAX_SESSION_IO_DIR": str(tmp_path / "invocation"),
        "RELAX_INPUT_JSON": str(input_path),
        "RELAX_OUTPUT_JSON": str(output_path),
    }
    if capability is not None:
        env["RELAX_AGENT_DEPENDENCY_EXIT_SUPPORTED"] = capability
    result = production.main(
        [
            "--config",
            str(config_path),
            "--compiler-defaults",
            str(defaults_path),
            "--reward-contract",
            str(tmp_path / "mock-contract.json"),
            "--reward-base-url",
            "https://reward.invalid",
        ],
        env=env,
    )
    assert result == expected_exit
    assert len(calls) == 1
    assert calls[0]["hold_on_replay_exhaustion"] is expected_hold
    assert calls[0]["output_path"] == str(output_path)
    assert not output_path.exists()
    stderr = capsys.readouterr().err
    assert code in stderr
    assert context.session_id not in stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["send_rejection", "restore_timeout"])
async def test_legacy_dependency_returns_without_repeating_ambiguous_operation(config, context, source, failure):
    requests = []
    config = replace(config, replay_attempt_timeout_s=0.01)

    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if failure == "restore_timeout":
            await asyncio.Future()
        return httpx.Response(200, content=ACK if body.get("replay_spec") else REJECTION)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        report = await asyncio.wait_for(
            client.run_audit(config, context, source, agentspark_http=http, hold_on_replay_exhaustion=True),
            timeout=1,
        )
    expected_code = "agentspark_dependency_unavailable" if failure == "send_rejection" else "replay_attempt_timeout"
    expected_requests = 2 if failure == "send_rejection" else 1
    assert len(requests) == expected_requests
    assert sum("replay_spec" in body for body in requests) == 1
    assert report["failure_code"] == expected_code
    assert report["audit_status"] == "failed"
    assert report["cleanup"]["local"] == "closed"
    assert report["replay_attempts"] == 1
    assert report["send_attempts"] == expected_requests - 1
    assert report["callbacks"]["started"] == 0
