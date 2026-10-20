#!/usr/bin/env bash
set -euo pipefail
repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
source "${PSRL_ENV_SCRIPT:?Set PSRL_ENV_SCRIPT}"
export PSRL_WORKSPACE="$repo_root"
export PYTHONPATH="$repo_root:${VERIFICATION_DEPS:?Set VERIFICATION_DEPS}:$repo_root/docs/env-worker-deep-dive/env-service-verification/scripts"
export CUDA_VISIBLE_DEVICES=0,1,2 TMPDIR=/tmp RAY_TMPDIR=/tmp
export LOCAL_IP="${NODE_IP:?Set NODE_IP}" VLLM_HOST_IP="$NODE_IP"
cd "$repo_root"
result_dir="${RESULT_DIR:?Set RESULT_DIR}"
mkdir -p "$result_dir/scripts-snapshot"
cp docs/env-worker-deep-dive/env-service-verification/scripts/{science_fixed_work_training.py,science_fixed_work_agent.py,science_pressure_agent.py,cuda_pressure.py,run_science_command_work.sh} "$result_dir/scripts-snapshot/"
git diff > "$result_dir/source.patch"
git rev-parse HEAD > "$result_dir/source-head.txt"
python -m pytest tests/env_service/test_position_ids_roundtrip.py -q > "$result_dir/position-ids-preflight.log" 2>&1
python -u docs/env-worker-deep-dive/env-service-verification/scripts/science_fixed_work_training.py \
 --repo "$repo_root" --baseline-repo "${BASELINE_REPO:?Set BASELINE_REPO}" \
 --model "${MODEL_PATH:?Set MODEL_PATH}" --dataset "${DATASET_PATH:?Set DATASET_PATH}" \
 --node-ip "$NODE_IP" --output "$result_dir" --steps 1 --timeout 1800 --task-timeout 360 --turns 4 \
 --modes command off "$@"
