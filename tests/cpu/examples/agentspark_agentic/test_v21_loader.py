# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Loader request evidence reaches the frozen reward parser without env
inference."""

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from examples.agentspark_agentic.v21_reward import RewardError, load_reward_contract, trajectory_from_audit
from tests.cpu.examples.agentspark_agentic.test_v21_reward import _audit, _json


@pytest.fixture
def loader_contract():
    path = os.environ.get("AGENTSPARK_V21_LOADER_CONTRACT")
    if not path:
        pytest.skip("Explicit loader reward contract is required")
    return load_reward_contract(path)


def _loader_audit(contract):
    audit = _audit()
    source = json.loads(Path(contract["source_config"]).read_text())["runtime_env"]
    snapshot = {
        "tools": json.loads(source["DOTSBRIDGE_AGENTSPARK_TOOLS_JSON"]),
        "capabilities": json.loads(source["DOTSBRIDGE_AGENTSPARK_CAPABILITIES_JSON"]),
        "skills_name_list": json.loads(source["DOTSBRIDGE_AGENTSPARK_SKILLS_NAME_LIST_JSON"]),
    }
    audit["source_input"]["metadata"]["agentspark"].update(copy.deepcopy(snapshot))
    audit["agentspark_request"] = snapshot
    request = audit["callbacks"]["records"][0]["request"]
    request["messages"].insert(
        0, {"role": "system", "content": Path(source["DOTSBRIDGE_AGENTSPARK_SYSTEM_PROMPT_FILE"]).read_text()}
    )
    audit["callbacks"]["records"][0]["forwarded_request_canonical_sha256"] = hashlib.sha256(_json(request)).hexdigest()
    return audit


def test_loader_request_snapshot_reaches_reward_view(loader_contract):
    audit = _loader_audit(loader_contract)
    trajectory = trajectory_from_audit(audit, loader_contract)
    assert trajectory["metadata"]["agentspark_request"] == audit["agentspark_request"]
    assert trajectory["raw_traj"][0]["call_llm"]["messages"][0]["role"] == "system"


@pytest.mark.parametrize(
    "change,code",
    [
        ("missing", "reward_loader_request_missing"),
        ("capture", "reward_tool_request_snapshot_mismatch"),
        ("config", "reward_tool_request_config_mismatch"),
    ],
)
def test_loader_request_missing_or_mismatched_evidence_rejected(loader_contract, change, code):
    audit = _loader_audit(loader_contract)
    if change == "missing":
        del audit["agentspark_request"]
    elif change == "capture":
        audit["agentspark_request"]["skills_name_list"] = []
    else:
        audit["agentspark_request"]["skills_name_list"] = []
        audit["source_input"]["metadata"]["agentspark"]["skills_name_list"] = []
    with pytest.raises(RewardError, match=code):
        trajectory_from_audit(audit, loader_contract)
