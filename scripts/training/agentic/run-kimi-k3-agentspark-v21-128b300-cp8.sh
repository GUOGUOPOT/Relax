#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# AgentSpark V21 / K3 CP8: 16 nodes x 8 B300; existing dataset/batch schedule.
# Derived from the maintained K3 128xBlackwell model recipe.
# V21 uses logical Session-mean loss, frozen rewards and response-local credit.
#
# Full-parameter Kimi K3 GRPO, 16 x 8 GPUs (280G Blackwell each), actor/rollout colocated.
# Training: TP4/PP4 (21+24+24+24)/CP8/EP32/ETP1, DP1, including vision.
# Rollout: 8 engines with attention TP8/DP2 and MoE EP16, two nodes per engine.
# Requires the current SGLang MXFP4 reload patch on EVERY node.
# CPU optimizer offload defaults to 0.7 with bounded FP32 CPU gradient staging.
# This requires the Relax HDO compatibility patch on every actor; Adam precision is unchanged.
# Validate the first update AND the next rollout: GPU Adam states remain resident.
# Failed jobs exit through cleanup by default; MAX_GLOBAL_RESTART explicitly permits retries.
# Source env.sh, then submit through scripts/entrypoint/ray-job.sh.
# DRY_RUN=1 bash <this-script> prints arguments without touching Ray.
# Source the existing training-env.wuxi-only.sh, then run this script.
# SMOKE_NUM_ROLLOUT remains inactive; the commented branch below is unchanged.
# NUM_ROLLOUT retains floor(ACCEPTED_ROWS / ROLLOUT_BATCH_SIZE) * EPOCHS.
# CP8_EXP_NAME and CP8_CHECKPOINT_DIR override this variant's output paths.
# Inherited CP2 CHECKPOINT_DIR is not reused; LOAD_DIR remains an explicit opt-in.
# Save optimizer state for recovery; real distributed save/resume still needs acceptance.
# Initialize from HF_CHECKPOINT by default; LOAD_DIR enables checkpoint recovery.

set -eo pipefail
if (( $# != 0 )); then
    echo "Use documented environment options; free CLI overrides can invalidate full-data coverage" >&2
    exit 2
fi
export RAY_NO_WAIT="${RAY_NO_WAIT-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}"
source "${MODEL_CONFIG_DIR}/kimi-k3.sh"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR:?Set MODEL_DIR to the Kimi-K3 directory itself}}"
PROMPT_SET="${PROMPT_SET:?Set PROMPT_SET to the validated prepared-relax JSONL on shared storage}"
: "${AGENTSPARK_CONFIG:?Set AGENTSPARK_CONFIG to the connection JSON}"
: "${AGENTSPARK_CALLBACK_HOST_MAP:?Set explicit worker hostname to callback hostname mapping JSON}"
: "${AGENTSPARK_V21_CONTRACT:?Set the pinned frozen V21 contract JSON}"
: "${AGENTSPARK_REWARD_URL:?Set the reachable V21 reward proxy base URL}"
: "${V21_TRAINING_OVERRIDES:?Set generated V21 training overrides YAML/JSON}"
: "${K3_COMPILER_DEFAULTS_JSON:?Set native compiler defaults JSON, including thinking=true}"
: "${AGENTIC_CONCURRENCY:?Set resident prompt groups within confirmed remote capacity}"
: "${EXPECTED_PREPARED_SHA256:?Set the validated prepared data SHA256}"
ACCEPTED_ROWS="${ACCEPTED_ROWS:-21258}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-18}"
N_SAMPLES_PER_PROMPT=4
EPOCHS="${EPOCHS:-1}"
for _value in "$ACCEPTED_ROWS" "$ROLLOUT_BATCH_SIZE" "$EPOCHS" "$AGENTIC_CONCURRENCY"; do
    [[ "$_value" =~ ^[1-9][0-9]*$ ]] || { echo "Positive integer required" >&2; exit 2; }
done
#if [ -n "${SMOKE_NUM_ROLLOUT:-}" ]; then
#    [[ "$SMOKE_NUM_ROLLOUT" =~ ^[1-9][0-9]*$ ]] || { echo "SMOKE_NUM_ROLLOUT must be a positive integer" >&2; exit 2; }
#    [ -z "${LOAD_DIR:-}" ] || { echo "Smoke validation requires a fresh run without LOAD_DIR" >&2; exit 2; }
#    (( SMOKE_NUM_ROLLOUT <= ACCEPTED_ROWS / ROLLOUT_BATCH_SIZE )) || { echo "Smoke target exceeds one dataset pass" >&2; exit 2; }
#    # Explicit partial validation overrides EPOCHS; retain the real input row/hash checks below.
#    NUM_ROLLOUT="$SMOKE_NUM_ROLLOUT"
#    echo "Smoke validation: $NUM_ROLLOUT rollouts, $ROLLOUT_BATCH_SIZE groups x $N_SAMPLES_PER_PROMPT Sessions each" >&2
#else
#    (( ACCEPTED_ROWS % ROLLOUT_BATCH_SIZE == 0 )) || { echo "Batch must divide accepted rows; tail wrapping is forbidden" >&2; exit 2; }
#    NUM_ROLLOUT=$((ACCEPTED_ROWS / ROLLOUT_BATCH_SIZE * EPOCHS))
#fi
NUM_ROLLOUT=$((ACCEPTED_ROWS / ROLLOUT_BATCH_SIZE * EPOCHS))
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
MAX_GLOBAL_RESTART="${MAX_GLOBAL_RESTART:-0}"
[[ "$MAX_GLOBAL_RESTART" =~ ^[0-9]+$ ]] || { echo "MAX_GLOBAL_RESTART must be a non-negative integer" >&2; exit 2; }
# TP4 * PP4 * CP8 uses all 128 GPUs: DP1 imposes no further batch divisibility.
now=$(date '+%Y-%m-%d-%H-%M-%S')
EXP_NAME="${CP8_EXP_NAME:-${EXP_NAME:-kimi-k3-agentspark-v21-128b300}-cp8}"
PROJECT_NAME="${PROJECT_NAME:-Relax/rl/kimi-k3-agentspark}"
CHECKPOINT_DIR="${CP8_CHECKPOINT_DIR:-${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}/${EXP_NAME}}"
# Keep runtime image admission opt-in for jobs submitted before this adapter update.
AGENT_OPTIONS=()
case "${AGENTSPARK_ALLOW_DYNAMIC_IMAGES:-0}" in
    0) ;;
    1) AGENT_OPTIONS+=(--allow-dynamic-images) ;;
    *) echo "AGENTSPARK_ALLOW_DYNAMIC_IMAGES must be 0 or 1" >&2; exit 2 ;;
esac
if [ -n "${AGENTSPARK_DIAGNOSTICS_DIR:-}" ]; then
    AGENT_OPTIONS+=(--diagnostics-dir "$AGENTSPARK_DIAGNOSTICS_DIR")
fi
# %q quoting is consumed by the native managed command's shlex parser.
printf -v AGENT_COMMAND '%q ' python3 -m examples.agentspark_agentic.production \
    --config "$AGENTSPARK_CONFIG" \
    --callback-host-map "$AGENTSPARK_CALLBACK_HOST_MAP" \
    --compiler-defaults "$K3_COMPILER_DEFAULTS_JSON" \
    --reward-contract "$AGENTSPARK_V21_CONTRACT" \
    --reward-base-url "$AGENTSPARK_REWARD_URL" \
    --sampling-temperature 0.8 "${AGENT_OPTIONS[@]}"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --ref-load "${HF_CHECKPOINT}"
    --trust-remote-code
    --megatron-to-hf-mode bridge
    --save "${CHECKPOINT_DIR}"
    --save-interval "${SAVE_INTERVAL:-100}"
    --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-1}"
)
if [ -n "${LOAD_DIR:-}" ]; then
    CKPT_ARGS+=(--load "${LOAD_DIR}")
fi
# Routed weight updates are enabled by default; set to 0 to use broadcast.
if [ "${COLOCATE_EXPERT_WEIGHT_ROUTING:-1}" = "1" ]; then
    CKPT_ARGS+=(--colocate-expert-weight-routing)
fi
if [ -n "${DUMP_DETAILS:-}" ]; then
    CKPT_ARGS+=(--dump-details "${DUMP_DETAILS}")
fi
ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key messages
    --label-key label
    --metadata-key metadata
    --multimodal-keys '{"image":"images"}'
    --apply-chat-template
    --apply-chat-template-kwargs '{"thinking":true}'
    --rollout-shuffle
    --balance-data
    --use-agentic-rollout
    --agent-command "$AGENT_COMMAND"
    --agent-cwd "$RELAX_ROOT"
    --agentic-reasoning-parser kimi_k3
    --agentic-tool-call-parser kimi_k3
    --agentic-concurrency "$AGENTIC_CONCURRENCY"
    --agent-timeout "${AGENT_TIMEOUT:-1800}"
    --agent-env "AGENTSPARK_V21_CONTRACT=$AGENTSPARK_V21_CONTRACT"
    --custom-convert-samples-to-train-data-path examples.agentspark_agentic.v21_training.convert_samples
    --loss-type custom_loss
    --custom-loss-function-path examples.agentspark_agentic.v21_loss.loss
    --custom-config-path "$V21_TRAINING_OVERRIDES"
    --disable-rewards-normalization
    --num-rollout "$NUM_ROLLOUT"
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --rollout-max-response-len 8192
    --rollout-max-context-len 131072
    --rollout-temperature 0.8
    #--image-max-token-num "${IMAGE_MAX_TOKEN_NUM:-1024}"
    --rollout-health-check-timeout 120
)

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 21
    --decoder-last-pipeline-num-layers 24
    --context-parallel-size 8
    --expert-model-parallel-size 32
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
    --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU:-16384}"
    # CP8 * 16384 covers the 131072 context; each budget is divisible by TP4 * 1024.
    --data-pad-size-multiplier 1024
    # EFA: retain the K3 alltoall path, without DeepEP/NVSHMEM.
    --moe-token-dispatcher-type alltoall
    --optimizer-cpu-offload
    --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION:-0.7}"
    --selective-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --no-pin-cpu-grads
    --no-pin-cpu-params
    --empty-unused-memory-level "${EMPTY_UNUSED_MEMORY_LEVEL:-1}"
    # Colocated KL keeps independent actor/ref CPU snapshots; bounded HDO gradients
    # remove the full FP32 CPU gradient cache without changing either snapshot.
    # Empty allocator settings override inherited expandable segments under TMS.
    --train-env-vars '{"FLA_TILELANG":"0","OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG":"1","TORCHINDUCTOR_COMPILE_THREADS":"1","PYTORCH_ALLOC_CONF":"","PYTORCH_CUDA_ALLOC_CONF":"","RELAX_KIMI_K3_FUSED_SITU":"1","RELAX_HDO_BOUNDED_CPU_GRADS":"1"}'
)
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-coef 0.0
    --kl-loss-coef 0.003
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
    --tis-clip 2
    --tis-clip-low 0
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 16
    --sglang-enable-dp-attention
    --sglang-context-length 131072
    --sglang-dp-size 2
    --sglang-ep-size 16
    --sglang-moe-runner-backend flashinfer_mxfp4
    --sglang-moe-a2a-backend none
    --use-slime-router
    # The initial full actor push supplies real weights before the first rollout.
    --sglang-load-format "${SGLANG_LOAD_FORMAT:-dummy}"
    # Leave room for GPU Adam moments allocated after the first rollout.
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.65}"
    # Validation observed 74 concurrent requests on one engine, falling back to eager at 64.
    --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS:-128}"
    --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-256}"
    --sglang-watchdog-timeout 3600
)
METRICS_ARGS=(
    --use-tensorboard
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
if [ "${USE_CLEARML:-0}" = 1 ]; then
    METRICS_ARGS+=(--use-clearml)
fi
TRAIN_ARGS=(
    --resource '{"actor": [1, 128], "rollout": [1, 128]}'
    --num-gpus-per-node 8
    --colocate
    --max-staleness 0
    --num-data-storage-units 16
    --use-health-check
    --max-global-restart "$MAX_GLOBAL_RESTART"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE:-8589934592}"
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${METRICS_ARGS[@]}"
)
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi
# Validate the exact local inputs before contacting Ray; do not load GPU weights.
python3 -m examples.agentspark_agentic.training_inputs check \
    --data "$PROMPT_SET" --expected-sha256 "$EXPECTED_PREPARED_SHA256" \
    --expected-rows "$ACCEPTED_ROWS" --contract "$AGENTSPARK_V21_CONTRACT" \
    --overrides "$V21_TRAINING_OVERRIDES" --compiler-defaults "$K3_COMPILER_DEFAULTS_JSON" \
    --agent-config "$AGENTSPARK_CONFIG" --callback-host-map "$AGENTSPARK_CALLBACK_HOST_MAP"
: "${RAY_DASHBOARD_ADDRESS:?Set explicit Ray Jobs/Dashboard URL}"
export RELAX_SKIP_CLUSTER_CLEANUP=1
export AGENTSPARK_V21_CONTRACT
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
# Also clear inherited allocator settings in the driver and rollout workers.
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os
runtime = json.loads(os.environ['RUNTIME_ENV_JSON'])
runtime.setdefault('env_vars', {}).update({
    'PYTORCH_ALLOC_CONF': '',
    'PYTORCH_CUDA_ALLOC_CONF': '',
    'TORCHINDUCTOR_COMPILE_THREADS': '1',
    # Explicitly enable bounded model-save staging for this RL recipe.
    'MEGATRON_SYNC_SAVE_BOUNDED_STAGING': os.environ.get('MEGATRON_SYNC_SAVE_BOUNDED_STAGING', '1'),
    'MEGATRON_SYNC_SAVE_STAGE_BYTES': os.environ.get('MEGATRON_SYNC_SAVE_STAGE_BYTES', '1073741824'),
    # Initial full sync exceeds Serve's user-event-loop probe timeout.
    # Keep probes responsive while synchronous service methods wait on Ray.
    'RAY_SERVE_RUN_SYNC_IN_THREADPOOL': '1',
    'AGENTSPARK_V21_CONTRACT': os.environ['AGENTSPARK_V21_CONTRACT'],
})
env = runtime['env_vars']
propagate = env.get('RELAX_PROPAGATE_ENV_VARS', '').split(',')
env['RELAX_PROPAGATE_ENV_VARS'] = ','.join(dict.fromkeys(
    name for name in [*propagate, 'MEGATRON_SYNC_SAVE_BOUNDED_STAGING', 'MEGATRON_SYNC_SAVE_STAGE_BYTES', 'AGENTSPARK_V21_CONTRACT'] if name
))
print(json.dumps(runtime))
PY
)
export RUNTIME_ENV_JSON
mkdir -p "${RELAX_ROOT}/log"
RAY_SUBMIT_ARGS=()
if [ "${RAY_NO_WAIT}" = "1" ]; then
    RAY_SUBMIT_ARGS+=(--no-wait)
fi
ray job submit "${RAY_SUBMIT_ARGS[@]}" \
    --address="${RAY_DASHBOARD_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${RELAX_ROOT}/log/${EXP_NAME}-${now}.log"
