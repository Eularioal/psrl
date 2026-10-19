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
date -Is > "$run_out/started-at.txt"
timeout 5 docker info --format '{{.ServerVersion}}' > "$run_out/docker-preflight.txt"
python -c 'import psrl, verl, transfer_queue; print({x.__name__: x.__file__ for x in (psrl, verl, transfer_queue)})' > "$run_out/driver-imports.txt"
python -m pytest -q tests/env_service/test_position_ids_roundtrip.py > "$run_out/position-ids-regression.log" 2>&1
touch "$run_root/docs/env-worker-deep-dive/env-service-verification/results/gpu-hold-20261016/STOP"
python - <<'PY'
import json, time
from pathlib import Path
p = Path('${WORKSPACE_ROOT}/docs/env-worker-deep-dive/env-service-verification/results/gpu-hold-20261016/status.json')
for _ in range(20):
    if json.loads(p.read_text())['status'] == 'released':
        break
    time.sleep(1)
else:
    raise RuntimeError('Temporary GPU hold did not release; refusing to start training')
PY
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv > "$run_out/gpu-preflight.txt"
cp "$run_root/docs/env-worker-deep-dive/env-service-verification/scripts/science_training_ab.py" "$run_out/science_training_ab.runtime.py"
sha256sum "$run_out/science_training_ab.runtime.py" "$run_root/psrl/examples/sciaccel_rl/runner.py" "$run_root/psrl/examples/sciaccel_rl/agent_loop.py" > "$run_out/source-hashes.txt"
set +e
python -u "$run_out/science_training_ab.runtime.py" \
  --repo "$run_root/psrl" \
  --model ${MODEL_ROOT}/Qwen3.5-4B \
  --dataset ${PSRL_WORKSPACE}/examples/sciaccel_rl/data/laps/repair_easy/train/L1.parquet \
  --output "$run_out" --node-ip 192.168.1.13 \
  --steps 2 --timeout 1440 --task-timeout 240 --turns 16
run_exit=$?
printf '%s\n' "$run_exit" > "$run_out/exit-code.txt"
date -Is > "$run_out/finished-at.txt"
exit "$run_exit"
