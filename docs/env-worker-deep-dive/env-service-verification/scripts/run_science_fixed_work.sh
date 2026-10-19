#!/usr/bin/env bash
set -euo pipefail
repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
source "${PSRL_ENV_SCRIPT:?Set PSRL_ENV_SCRIPT to the environment activation script}"
export PSRL_WORKSPACE="$repo_root"
export PYTHONPATH="$PSRL_WORKSPACE:${VERIFICATION_DEPS:?Set VERIFICATION_DEPS}:$PSRL_WORKSPACE/docs/env-worker-deep-dive/env-service-verification/scripts"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" TMPDIR=/tmp RAY_TMPDIR=/tmp
export LOCAL_IP="${NODE_IP:?Set NODE_IP to the verified experiment node address}" VLLM_HOST_IP="$NODE_IP"
cd "$PSRL_WORKSPACE"
result_dir="$PSRL_WORKSPACE/docs/env-worker-deep-dive/env-service-verification/results/science-fixed-work-20261019"
mkdir -p "$result_dir/scripts-snapshot"
cp docs/env-worker-deep-dive/env-service-verification/scripts/{science_fixed_work_training.py,science_fixed_work_agent.py,science_pressure_agent.py,cuda_pressure.py,run_science_fixed_work.sh} "$result_dir/scripts-snapshot/"
python -u docs/env-worker-deep-dive/env-service-verification/scripts/science_fixed_work_training.py \
 --repo "$PSRL_WORKSPACE" \
 --baseline-repo "${BASELINE_REPO:?Set BASELINE_REPO to the isolated baseline checkout}" \
 --model "${MODEL_ROOT:?Set MODEL_ROOT}/Qwen3.5-4B" \
 --dataset "$PSRL_WORKSPACE/examples/sciaccel_rl/data/laps/repair_easy/train/L1.parquet" \
 --node-ip "$NODE_IP" --output "$result_dir" --steps 1 --timeout 3600 --task-timeout 240 --turns 16 "$@"
