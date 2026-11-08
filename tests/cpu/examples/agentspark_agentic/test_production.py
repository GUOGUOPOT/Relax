# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Production publication and generation ownership; no live model or RM claims."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import stat
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest

from examples.agentspark_agentic.history import ReasoningHistory, pin_training_request
from examples.agentspark_agentic.production import (
    attach_reward,
    configure_callback_host,
    publish_training_output,
    run_training,
    select_training_exports,
)
from examples.agentspark_agentic.protocol import AdapterError
from examples.agentspark_agentic.v21_reward import RewardError
from relax.agentic.runner.ipc import _session_output_from_text
from tests.cpu.examples.agentspark_agentic.test_callback_client import (  # noqa: F401
    completion,
    config,
    context,
    event,
    session_input,
)
from tests.cpu.examples.agentspark_agentic.test_contexts import _audit, _record
from tests.cpu.examples.agentspark_agentic.test_native_exports import _user, native, official_k3  # noqa: F401


def _score(*, records=None, audit=None):
    invocation, indices = "e" * 64, [0]
    if records:
        anchor = next(
            record["metadata"]["agentspark_training"]
            for record in records
            if record["metadata"]["agentspark_training"]["is_anchor"]
        )
        invocation, indices = anchor["invocation_sha256"], list(range(anchor["record_start"], anchor["record_stop"]))
    if audit:
        invocation, indices = audit["invocation_sha256"], audit["reward_record_indices"]
    return {
        "base_reward": 0.7,
        "invocation_sha256": invocation,
        "reward_record_indices": indices,
        "record_policy": "final_active_segment_v1",
        "prompt_bundle_fingerprint": "a" * 16,
        "metadata": {"verified_reward": "local-fixture-only"},
        "identity": {"trajectory_sha256": "a" * 64},
        "source_identity": {"source_sha256": "b" * 64},
    }


def _tool_assistant(call_id="c1", arguments='{"n":100.0}'):
    return {
        "role": "assistant",
        "content": "<short_content>checking</short_content>",
        "reasoning_content": "observed private reasoning",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "Search", "arguments": arguments}}],
    }


def test_reasoning_restores_only_the_frozen_prefix_and_remaps_tool_result_ids():
    projector = ReasoningHistory()
    assistant = _tool_assistant()
    initial = {"role": "user", "content": "question"}
    projector.commit({"messages": [initial]}, assistant)
    replay = _tool_assistant("renamed", '{"n":1e2}')
    replay.pop("reasoning_content")
    replay["content"] = ""
    tool = {"role": "tool", "tool_call_id": "renamed", "content": "found"}
    payload = {"messages": [initial, replay, tool]}
    actual, evidence = projector.restore(payload)
    assert actual["messages"] == [initial, assistant, {**tool, "tool_call_id": "c1"}]
    assert evidence["restored"] and evidence["mapped_tool_results"] == 1
    assert "reasoning_content" not in replay and tool["tool_call_id"] == "renamed"
    for arguments in ('{"n":101}', '{"n":100}', '{"n":100.0,"n":100.0}'):
        changed = copy.deepcopy(payload)
        changed["messages"][1]["tool_calls"][0]["function"]["arguments"] = arguments
        unchanged, evidence = projector.restore(changed)
        assert unchanged == changed and evidence["status"] == "new_segment" and evidence["restored"] == []
    for changed in (
        {"messages": [replay, tool]},
        {"messages": [{"role": "user", "content": "changed history"}, replay, tool]},
        {**payload, "tools": [{"type": "function", "function": {"name": "new_tool"}}]},
        {"messages": [initial, replay, {**tool, "tool_call_id": "unmapped"}]},
    ):
        unchanged, evidence = projector.restore(changed)
        assert unchanged == changed and evidence["status"] == "new_segment" and evidence["restored"] == []


def test_segment_selection_rejects_duplicate_generation_and_incomplete_capture():
    initial = [{"role": "user", "content": "q"}]
    first = _record(0, initial)
    duplicate = copy.deepcopy(first)
    duplicate["index"] = 1
    with pytest.raises(AdapterError, match="training_generation_identity_ambiguous"):
        select_training_exports(_audit([first, duplicate]), compiler_defaults={})
    audit = _audit([first])
    audit["callbacks"]["errors"] = ["forward_failed"]
    with pytest.raises(AdapterError, match="context_capture_incomplete"):
        select_training_exports(audit, compiler_defaults={})


@pytest.mark.asyncio
async def test_segment_ownership_matches_real_native_response_states_once(native):
    root = _user("initial")
    first = await native.commit([root])
    await native.commit([root, first, _user("continuation")])
    await native.commit([root, first, _user("branch")])
    await native.commit([_user("reset")])
    audit = native.report([1, 2, 3])["source_audit"]
    records = select_training_exports(audit, compiler_defaults=native.compiler_defaults, append_roles=("user", "tool"))
    assert len(records) == 3
    owned = [record["metadata"]["agentspark_training"]["owned_resp_state_hashes"] for record in records]
    assert owned == [native.response_hashes[:2], native.response_hashes[2:3], native.response_hashes[3:]]
    records = attach_reward(records, _score(records=records))
    transport = native.native_export(records)
    seen = []
    for record, export in zip(records, transport.exports, strict=True):
        sample = export["sample_payload"]
        assert sample["index"] == 17 and sample["group_index"] == 3
        turns = sample["metadata"]["agentic_trace"]["turns"]
        hashes = [turn["resp_state_hash"] for turn in turns]
        wanted = record["metadata"]["agentspark_training"]["owned_resp_state_hashes"]
        assert set(wanted) <= set(hashes)
        seen.extend(wanted)
        response_ids = sample["tokens"][-sample["response_length"] :]
        for turn, (start, end) in zip(turns, sample["custom_advantage"], strict=True):
            generated_ids, generated_logprobs = native.generated[turn["request_id"]]
            assert list(response_ids[start:end]) == generated_ids
            assert list(sample["rollout_log_probs"][start:end]) == pytest.approx(generated_logprobs)
    assert seen == native.response_hashes and len(set(seen)) == len(seen)
    assert [record["reward"] for record in records] == [0.0, 0.0, 0.7]
    # Eval exports one representative, never treats old segments as extra episodes.
    audit["identity"]["mode"] = "eval"
    representative = select_training_exports(
        audit, compiler_defaults=native.compiler_defaults, append_roles=("user", "tool")
    )
    assert len(representative) == 1 and representative[0]["metadata"]["agentspark_training"]["is_anchor"]


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), True])
def test_reward_rejects_nonfinite_or_boolean(reward):
    result = _score()
    result["base_reward"] = reward
    with pytest.raises(AdapterError, match="training_reward_nonfinite"):
        attach_reward([], result)


def test_atomic_publication_no_overwrite_or_credentials(tmp_path):
    records = select_training_exports(_audit([_record(0, [{"role": "user", "content": "q"}])]), compiler_defaults={})
    records = attach_reward(records, _score(records=records))
    path = tmp_path / "output.json"
    publish_training_output(records, path, credential="fixture-secret")
    before = path.read_bytes()
    assert len(_session_output_from_text(before.decode()).records) == 1
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(AdapterError, match="training_output_already_exists"):
        publish_training_output(records, path, credential="fixture-secret")
    assert path.read_bytes() == before
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(AdapterError, match="training_output_already_exists"):
        publish_training_output(records, alias, credential="fixture-secret")
    records[0]["metadata"]["secret"] = "fixture-secret"
    with pytest.raises(AdapterError, match="audit_session_credential_forbidden"):
        publish_training_output(records, tmp_path / "denied.json", credential="fixture-secret")
    assert not (tmp_path / "denied.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "reward", "reward_identity", "cancel", "late_error"])
async def test_production_loopback_publication_requires_complete_real_protocol_and_scoring(
    tmp_path, config, context, session_input, failure
):
    forwarded, score_calls, send_calls = [], [], []
    assistant = _tool_assistant()
    responses = [completion(assistant, sse=True, finish="tool_calls"), completion(sse=True)]
    stop_event = asyncio.Event()

    async def backend(request):
        forwarded.append(json.loads(request.content))
        return responses[len(forwarded) - 1]

    async def platform(request):
        if request.method == "GET":
            return httpx.Response(200, json={"session_id": "remote-session"})
        body = json.loads(request.content)
        send_calls.append(body)
        model = body["llm_config"]
        async with httpx.AsyncClient(trust_env=False) as caller:
            url = model["base_url"] + model["endpoint_path"]
            headers = {"Authorization": "Bearer " + model["api_key"]}
            first = {
                "messages": [{"role": "user", "content": "q"}],
                "stream": True,
                "chat_template_kwargs": {"thinking": False},
                "temperature": 0.1,
            }
            first_response = await caller.post(url, json=first, headers=headers)
            assert first_response.status_code == 200
            replay = copy.deepcopy(assistant)
            replay.pop("reasoning_content")
            replay["content"] = ""
            second = {
                "messages": first["messages"] + [replay, {"role": "tool", "tool_call_id": "c1", "content": "found"}],
                "stream": True,
            }
            second_response = await caller.post(url, json=second, headers=headers)
            assert second_response.status_code == 200
        stream = event("agent.stream.end") + event("id_snapshot", data={"done": True})
        if failure == "late_error":
            stream += event("error", message="private remote error")
        return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})

    async def scorer(audit):
        score_calls.append(audit)
        assert audit["stream"]["eof"] and audit["cleanup"]["local"] == "closed"
        if failure == "reward":
            raise AdapterError("rm_failed")
        if failure == "cancel":
            stop_event.set()
            await asyncio.sleep(10)
        score = _score(audit=audit)
        if failure == "reward_identity":
            score["invocation_sha256"] = "f" * 64
        return score

    audit_path, output_path = tmp_path / "audit.json", tmp_path / "output.json"
    async with httpx.AsyncClient(transport=httpx.MockTransport(platform)) as platform_http, httpx.AsyncClient(
        transport=httpx.MockTransport(backend)
    ) as backend_http:
        coroutine = run_training(
            config,
            context,
            session_input,
            compiler_defaults={"thinking": True},
            scorer=scorer,
            audit_path=audit_path,
            output_path=output_path,
            agentspark_http=platform_http,
            relax_http=backend_http,
            stop_event=stop_event,
        )
        if failure:
            with pytest.raises(AdapterError):
                await coroutine
            assert not output_path.exists()
        else:
            result = await coroutine
            assert result["records"][0]["reward"] == 0.7
            assert len(_session_output_from_text(output_path.read_text()).records) == 1
    assert len(send_calls) == 1 and len(forwarded) == 2
    assert len(score_calls) == (0 if failure == "late_error" else 1)
    audit = json.loads(audit_path.read_text())
    record = audit["callbacks"]["records"][1]
    assert record["raw_request"]["messages"][1].get("reasoning_content") is None
    assert record["request"]["messages"][1] == assistant
    assert record["history"]["restored"]
    assert all(
        request["chat_template_kwargs"] == {"thinking": True} and request["temperature"] == 0.8
        for request in forwarded
    )
    assert record["request_sha256"] != record["raw_request_sha256"]
    assert context.session_id not in audit_path.read_text()


def test_callback_host_mapping_is_explicit_and_keeps_per_session_port(config, monkeypatch):
    monkeypatch.setattr("examples.agentspark_agentic.production.socket.gethostname", lambda: "worker-A")
    configured = configure_callback_host(config, {"worker-A": "routable.example.invalid"})
    assert configured.callback_public_base_url == "http://routable.example.invalid:0/callback"
    assert (
        configure_callback_host(config, {"worker-A": "2001:db8::1"}).callback_public_base_url
        == "http://[2001:db8::1]:0/callback"
    )
    for mapping in ({"worker-B": "routable.example.invalid"}, {"worker-A": "http://host"}, {"worker-A": "host:8080"}):
        with pytest.raises(AdapterError, match="callback_host_map"):
            configure_callback_host(config, mapping)


@pytest.mark.parametrize(
    "payload",
    [
        {"chat_template_kwargs": {"thinking": "false"}},
        {"chat_template_kwargs": {"unconfigured": True}},
        {"chat_template_kwargs": []},
        {"reasoning": {"enabled": "true"}},
        {"temperature": 0},
        {"top_p": 0.9},
    ],
)
def test_production_pins_reject_malformed_or_unsupported_controls(payload):
    with pytest.raises(AdapterError, match="production_"):
        pin_training_request(payload, {"thinking": True}, 0.8)


@pytest.mark.asyncio
async def test_reward_failure_persists_private_summary_without_dataset_content(
    tmp_path, monkeypatch, config, context, session_input
):
    from examples.agentspark_agentic import production

    audit = _audit([_record(0, [{"role": "user", "content": "PRIVATE_QUERY"}])])
    audit["source_input"] = {"metadata": {
        "agentspark_prepare": {"source_line": 123}, "query": "PRIVATE_QUERY /private/path https://private.invalid",
    }}

    async def collected(*args, **kwargs):
        return copy.deepcopy(audit)

    async def scorer(_audit):
        raise RewardError("reward_http_failure", safe_diagnostic={
            "http_status": 502, "reason_code": "judge_issue_impact_coverage_invalid", "message": "PRIVATE_ERROR",
        })

    monkeypatch.setattr(production, "run_audit", collected)
    context = replace(context, group_id="train:prepare_group_7")
    directory = tmp_path / "diagnostics"
    for i in range(2):
        with pytest.raises(RewardError, match="^reward_http_failure$"):
            await run_training(
                config, context, session_input, compiler_defaults={}, scorer=scorer,
                audit_path=tmp_path / f"audit-{i}.json", output_path=tmp_path / f"output-{i}.json",
                diagnostics_dir=directory,
            )
        assert not (tmp_path / f"output-{i}.json").exists()
        legacy = json.loads((tmp_path / f"audit-{i}.json").read_text())
        assert legacy["production"]["error"] == "reward_http_failure"
    reports = sorted(directory.glob("failure-*.json"))
    assert len(reports) == 2
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for path in reports:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        summary = json.loads(path.read_text())
        assert summary["error"] == "reward_http_failure" and summary["stage"] == "score"
        assert summary["group_id"] == context.group_id and summary["source_line"] == 123
        assert summary["invocation_sha256"] == audit["invocation_sha256"]
        assert summary["diagnostic"] == {
            "http_status": 502, "reason_code": "judge_issue_impact_coverage_invalid", "exception_type": "RewardError",
        }
        assert all(x not in path.read_text() for x in ("PRIVATE_", "/private/path", "private.invalid", context.session_id))


def test_cli_preserves_reward_error_and_accepts_opt_in_flags(tmp_path, monkeypatch, capsys, config, context, session_input):
    from examples.agentspark_agentic import production

    config_path, input_path = tmp_path / "config.json", tmp_path / "input.json"
    config_path.write_text(json.dumps(asdict(config)))
    session_input["metadata"]["agentspark_prepare"] = {"source_line": 17}
    input_path.write_text(json.dumps(session_input))
    seen = []

    async def fail(*values):
        seen.append(values[-1])
        raise RewardError("reward_critic_execution_failed")

    monkeypatch.setattr(production, "_run_cli", fail)
    env = {
        "RELAX_BASE_URL": context.base_url, "RELAX_SESSION_ID": context.session_id,
        "RELAX_GROUP_ID": "train:prepare_group_4", "RELAX_ROLLOUT_MODE": context.mode,
        "RELAX_SESSION_IO_DIR": context.invocation_id,
        "RELAX_INPUT_JSON": str(input_path), "RELAX_OUTPUT_JSON": str(tmp_path / "out.json"),
    }
    assert production.main([
        "--config", str(config_path), "--compiler-defaults", "unused", "--reward-contract", "unused",
        "--reward-base-url", "http://unused.invalid", "--allow-dynamic-images",
        "--diagnostics-dir", str(tmp_path / "diagnostics"),
    ], env=env) == 2
    assert seen[0].allow_dynamic_images is True
    assert "reward_critic_execution_failed" in capsys.readouterr().err
    summary = json.loads(next((tmp_path / "diagnostics").glob("failure-*.json")).read_text())
    assert summary["source_line"] == 17 and summary["group_id"] == "train:prepare_group_4"


def test_diagnostics_filter_unknown_values_and_tolerate_malformed_early_input(tmp_path, context):
    from examples.agentspark_agentic.diagnostics import failure_summary, persist_failure, upstream_diagnostic

    assert upstream_diagnostic({
        "error": {"type": "PRIVATE_TYPE", "code": "PRIVATE_CODE", "status": True, "status_code": 999999},
        "message": "PRIVATE", "url": "https://private.invalid", "authorization": context.session_id,
    }) == {}
    for source in (["PRIVATE"], "PRIVATE", None):
        error = ValueError("PRIVATE /private/path")
        path = persist_failure(tmp_path / "diagnostics", error, context=context, audit={
            "source_input": source, "callbacks": ["PRIVATE"], "stream": "PRIVATE",
            "failure_code": "agentspark_stream_error",
        })
        summary = json.loads(path.read_text())
        assert summary["error"] == "training_agent_failed"
        assert summary["audit_failure_code"] == "agentspark_stream_error"
        assert "group_id" not in summary and "group_id_sha256" in summary
        assert "PRIVATE" not in path.read_text()
    assert persist_failure(None, RuntimeError("PRIVATE"), context=context) is None
    assert failure_summary(RewardError("reward_private_secret"))["error"] == "training_agent_failed"
