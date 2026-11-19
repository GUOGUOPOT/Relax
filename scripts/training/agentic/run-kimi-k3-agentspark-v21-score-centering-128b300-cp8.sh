#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# No-loader V21 SC + CTR; source its separately pinned training environment.
# Uses the same CP8 resources and schedule as the GRPO recipe.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
: "${V21_TRAINING_OVERRIDES:?Set the generated V21 Score Centering overrides}"
python3 - "$V21_TRAINING_OVERRIDES" <<'CHECK_SC'
import json
import sys

with open(sys.argv[1]) as stream:
    config = json.load(stream)
if (
    config.get("custom_loss_function_path") != "examples.agentspark_agentic.v21_score_centering_loss.loss"
    or config.get("use_tis") is not False
    or type(config.get("rollout_top_logprobs_num")) is not int
    or config["rollout_top_logprobs_num"] < 1
):
    raise SystemExit("Use the matching V21 Score Centering contract and overrides for this script")
CHECK_SC
export CP8_EXP_NAME="${CP8_EXP_NAME:-kimi-k3-agentspark-v21-score-centering-128b300-cp8}"
# Behavior q must include the generation temperature.
export SGLANG_RETURN_ORIGINAL_LOGPROB=0
exec bash "${SCRIPT_DIR}/run-kimi-k3-agentspark-v21-128b300-cp8.sh" "$@"
