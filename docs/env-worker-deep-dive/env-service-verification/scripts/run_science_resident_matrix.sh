#!/usr/bin/env bash
# Run one training update per arm with equal task demand and two resident pressure levels.
# Usage: configure the required environment paths, then run this script.
set -euo pipefail
experiment_repo="${PSRL_REPO:?Set PSRL_REPO}"
experiment_output="${RESULT_DIR:?Set RESULT_DIR to a fresh directory}"
source "${PSRL_ENV_SCRIPT:?Set PSRL_ENV_SCRIPT}"
export PSRL_WORKSPACE="$experiment_repo"
export PYTHONPATH="$experiment_repo:${VERIFICATION_DEPS:?Set VERIFICATION_DEPS}:$experiment_repo/docs/env-worker-deep-dive/env-service-verification/scripts"
export CUDA_VISIBLE_DEVICES=0,1,2 TMPDIR=/tmp RAY_TMPDIR=/tmp
export LOCAL_IP="${NODE_IP:?Set NODE_IP}" VLLM_HOST_IP="$NODE_IP"
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ALLOW_INSECURE_SERIALIZATION=1
unset RAY_ADDRESS
cd "$experiment_repo"
common=(--repo "$experiment_repo" --baseline-repo "${BASELINE_REPO:?Set BASELINE_REPO}"
  --model "${MODEL_PATH:?Set MODEL_PATH}" --dataset "${DATASET_PATH:?Set DATASET_PATH}"
  --node-ip "$NODE_IP" --steps 1 --rollout-n 2 --task-count 2 --turns 2
  --response-length 4096 --task-timeout 180 --environment-gpus 6
  --pressure-gib 14 --pressure-hold 15 --unit-budget-gib 90
  --initialization-timeout 900 --work-timeout 480 --timeout 1380 --prewarm)
script=docs/env-worker-deep-dive/env-service-verification/scripts/science_fixed_work_training.py
python -u "$script" "${common[@]}" --output "$experiment_output/low" --resident-gib 0 --modes off command
python -u "$script" "${common[@]}" --output "$experiment_output/high" --resident-gib 70 --modes command off
