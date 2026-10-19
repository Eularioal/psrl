#!/usr/bin/env bash
set -eo pipefail
run_root=${WORKSPACE_ROOT}
run_out="$run_root/docs/env-worker-deep-dive/env-service-verification/results/science-training-ab-20261016-live"
source ${PSRL_ENV_SCRIPT} > "$run_out/environment-activation.log" 2>&1
export PSRL_WORKSPACE="$run_root"
export PSRL_PATH="$run_root/psrl"
export PYTHONPATH="$run_root/psrl:$run_root/.verification-deps"
export TMPDIR=/tmp RAY_TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=0,1,2 CUDA_DEVICE_ORDER=PCI_BUS_ID
export LOCAL_IP=192.168.1.13 VLLM_HOST_IP=192.168.1.13
export TRANSFER_QUEUE_ENABLE=1
cd "$run_root/psrl"
date -Is > "$run_out/on-started-at.txt"
set +e
python -u "$run_out/science_training_ab.on.runtime.py" \
  --repo "$run_root/psrl" \
  --model ${MODEL_ROOT}/Qwen3.5-4B \
  --dataset ${PSRL_WORKSPACE}/examples/sciaccel_rl/data/laps/repair_easy/train/L1.parquet \
  --output "$run_out" --node-ip 192.168.1.13 \
  --steps 2 --timeout 1440 --task-timeout 240 --turns 16 --modes on
run_exit=$?
printf '%s\n' "$run_exit" > "$run_out/exit-code.txt"
date -Is > "$run_out/finished-at.txt"
exit "$run_exit"
