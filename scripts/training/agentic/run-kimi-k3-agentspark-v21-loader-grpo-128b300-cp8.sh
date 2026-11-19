#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Loader + V21 GRPO CTR. Requires its own prepared data and reward contract.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
: "${AGENTSPARK_V21_CONTRACT:?Source the loader GRPO training environment}"
: "${V21_TRAINING_OVERRIDES:?Set generated loader GRPO overrides}"
python3 - "$AGENTSPARK_V21_CONTRACT" "$V21_TRAINING_OVERRIDES" <<'CHECK_LOADER'
import json
import sys
from pathlib import Path
from examples.agentspark_agentic.v21_reward import load_reward_contract
from examples.agentspark_agentic.v21_training import training_overrides

contract = load_reward_contract(sys.argv[1])
source = json.loads(Path(contract["source_config"]).read_text())
if source["runtime_env"].get("SLIME_AGENTSPARK_SKILL_LOADER_ENABLE") != "1":
    raise SystemExit("Use the loader GRPO contract with this entrypoint")
if source["train"].get("loss-type") == "score_centering":
    raise SystemExit("Loader + Score Centering is not part of this recipe")
if json.loads(Path(sys.argv[2]).read_text()) != training_overrides(contract):
    raise SystemExit("Loader GRPO overrides must match the pinned source")
CHECK_LOADER
export CP8_EXP_NAME="${CP8_EXP_NAME:-kimi-k3-agentspark-v21-loader-grpo-128b300-cp8}"
exec bash "${SCRIPT_DIR}/run-kimi-k3-agentspark-v21-128b300-cp8.sh" "$@"
