"""Native K3 exports -> frozen V21 -> transfer fields -> differentiable CPU loss.

Only model generation/scoring and distributed hardware are controlled. The real
local checkpoint tokenizer, native committed states, source reward policy and
production conversion/loss functions execute. This is not GPU or live RM proof.
"""

from __future__ import annotations

import copy
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
import torch

from examples.agentspark_agentic import v21_loss, v21_training
from examples.agentspark_agentic.production import attach_reward, select_training_exports
from examples.agentspark_agentic.v21_reward import _json, score_audit
from relax.agentic.pipeline import TrainingFieldArtifact
from tests.cpu.examples.agentspark_agentic.test_native_exports import _NativeDriver, _user, official_k3  # noqa: F401
from tests.cpu.examples.agentspark_agentic.test_v21_reward import _audit as reward_audit_fixture
from tests.cpu.examples.agentspark_agentic.test_v21_reward import _score, contract  # noqa: F401
from tests.cpu.examples.agentspark_agentic.test_v21_training import _args, _backend


class _ThinkingDriver(_NativeDriver):
    compiler_defaults = {"thinking": True}


@pytest.mark.asyncio
async def test_native_multicontext_to_v21_transfer_and_loss_backward(official_k3, contract, monkeypatch):
    samples = []
    tokenizer = official_k3.tokenizer
    with ThreadPoolExecutor(max_workers=2) as executor:
        for sibling in range(4):
            driver = _ThinkingDriver(official_k3, executor, session_id=f"full-pipeline-{sibling}", index=sibling)
            history = [{"role": "user", "content": "test"}]
            answer = {
                "role": "assistant",
                "reasoning_content": '<guide_question type="recommend">do not credit reasoning</guide_question>',
                "content": '<guide_question type="recommend"><sub_question>可见追问🙂</sub_question></guide_question>',
            }
            prefix = tokenizer.apply_chat_template(history, tokenize=True, thinking=True, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(
                history + [answer], tokenize=True, thinking=True, add_generation_prompt=False
            )
            assert full[: len(prefix)] == prefix
            driver.response_ids = list(full[len(prefix) :])
            driver.response_logprobs = [-0.5] * len(driver.response_ids)
            root = _user("fixture question")
            first = await driver.commit([root])
            await driver.commit([root, first, _user("continue")])
            await driver.commit([root, first, _user("branch")])
            await driver.commit([_user("reset")])
            audit = driver.report([1, 2, 3])["source_audit"]
            records = select_training_exports(
                audit, compiler_defaults=driver.compiler_defaults, append_roles=("user", "tool")
            )
            assert len(records) == 3
            anchor = records[-1]["metadata"]["agentspark_training"]
            reward_input = copy.deepcopy(audit)
            fixture = reward_audit_fixture()
            reward_input["source_input"] = fixture["source_input"]
            reward_input["stream"]["events"] = fixture["stream"]["events"]
            reward_input["reward_record_indices"] = list(range(anchor["record_start"], anchor["record_stop"]))
            reward_input["reward_record_policy"] = "final_active_segment_v1"
            for record in reward_input["callbacks"]["records"]:
                record["request_identity"] = "actual_forwarded"
                record["forwarded_request_canonical_sha256"] = hashlib.sha256(_json(record["request"])).hexdigest()

            def remote_response(request):
                import json

                payload = json.loads(request.content)
                return httpx.Response(
                    200,
                    json={**payload, "score": _score(contract, base=0.2 + sibling * 0.2, click=0.2 + sibling * 0.2)},
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(remote_response)) as client:
                scored = await score_audit(
                    reward_input, contract=contract, reward_base_url="http://fixture.invalid", client=client
                )
            output = driver.native_export(attach_reward(records, scored))
            samples.extend(TrainingFieldArtifact(export["sample_payload"]).to_sample() for export in output.exports)
    args = _args(
        n_samples_per_prompt=4,
        hf_checkpoint=os.environ["KIMI_K3_SOURCE_DIR"],
        multimodal_keys={"image": "images"},
        reward_key=None,
        debug_train_only=True,
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path="examples.agentspark_agentic.v21_training.convert_samples",
        **v21_training.training_overrides(contract),
    )
    # Use exactly the fixture's verified tokenizer instance, avoiding a second
    # dynamic-module cache; production resolves the same explicit local path.
    monkeypatch.setattr(v21_training, "_tokenizer", lambda _path: tokenizer)
    before = [
        (copy.deepcopy(sample.tokens), list(sample.rollout_log_probs), copy.deepcopy(sample.loss_mask))
        for sample in samples
    ]
    data = v21_training.convert_samples(args, samples)
    assert len(data["tokens"]) == 12 and len(set(data["sample_indices"])) == 4
    for sibling in range(4):
        positions = [i for i, index in enumerate(data["sample_indices"]) if index == sibling]
        effective = sum(sum(data["loss_masks"][i]) for i in positions)
        assert effective == 4 * len(driver.response_ids)
        assert all(data["sample_index_mask_sums"][i] == effective for i in positions)
        followup = sum(sum(data["followup_token_masks"][i]) for i in positions)
        assert followup > 0
        assert all(data["followup_token_mask_sums"][i] == followup for i in positions)
        assert len({data["rewards"][i] for i in positions}) == 1
    assert [(sample.tokens, sample.rollout_log_probs, sample.loss_mask) for sample in samples] == before
    assert data["rewards"][0] < 0 < data["rewards"][-1]
    backend, cp = _backend(monkeypatch)
    batch = dict(data)
    batch["loss_masks"] = [torch.tensor(mask) for mask in data["loss_masks"]]
    batch["unconcat_tokens"] = [torch.tensor(tokens) for tokens in data["tokens"]]
    lengths = data["response_lengths"]
    batch["advantages"] = [
        torch.full((length,), reward) for length, reward in zip(lengths, data["rewards"], strict=True)
    ]
    batch["rollout_log_probs"] = [torch.tensor(values) for values in data["rollout_log_probs"]]
    batch["log_probs"] = [values.clone() for values in batch["rollout_log_probs"]]
    batch["ref_log_probs"] = [values - 0.05 for values in batch["rollout_log_probs"]]
    current = (torch.cat(batch["rollout_log_probs"]) + 0.025).requires_grad_()
    monkeypatch.setattr(
        backend,
        "get_log_probs_and_entropy",
        lambda logits, **_kwargs: (
            None,
            {"log_probs": list(logits.split(lengths)), "entropy": list(torch.zeros_like(logits).split(lengths))},
        ),
    )
    reducer = cp.get_sum_of_sample_mean(
        data["total_lengths"], lengths, batch["loss_masks"], sample_denoms=data["sample_index_mask_sums"]
    )
    loss, metrics = v21_loss.loss(args, batch, current, reducer)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(current.grad).all() and current.grad.abs().sum() > 0
    assert not torch.any(current.grad[torch.cat(batch["loss_masks"]) == 0])
    assert any("followup" in name for name in metrics)
