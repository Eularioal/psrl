#!/usr/bin/env bash
set -euo pipefail
repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
source "${PSRL_ENV_SCRIPT:?Set PSRL_ENV_SCRIPT to the environment activation script}"
export PSRL_WORKSPACE="$repo_root"
export PYTHONPATH="$PSRL_WORKSPACE:${VERIFICATION_DEPS:?Set VERIFICATION_DEPS}:$PSRL_WORKSPACE/docs/env-worker-deep-dive/env-service-verification/scripts"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" TMPDIR=/tmp RAY_TMPDIR=/tmp
export LOCAL_IP="${NODE_IP:?Set NODE_IP to the verified experiment node address}" VLLM_HOST_IP="$NODE_IP"
cd "$PSRL_WORKSPACE"
result_dir="$PSRL_WORKSPACE/docs/env-worker-deep-dive/env-service-verification/results/science-pressure-ab-20261018"
python - <<'PY'
import json
from pathlib import Path
p=Path('docs/env-worker-deep-dive/env-service-verification/results/science-pressure-ab-20261018/preflight')
for mode in ('off','on'):
    assert json.loads((p/'audited-results.json').read_text())[mode]['trial_count']==4
    probes=[json.loads(f.read_text()) for f in (p/mode).rglob('gpu-pressure.json')]
    assert len(probes)==4, (mode,len(probes))
    assert all(x['status']=='ok' for x in probes) if mode=='on' else any(x['status']=='cuda_oom' for x in probes)
PY
mkdir -p "$result_dir/scripts-snapshot"
cp docs/env-worker-deep-dive/env-service-verification/scripts/{science_pressure_training.py,science_pressure_agent.py,cuda_pressure.py,science_pressure_preflight.py,run_science_pressure.sh} "$result_dir/scripts-snapshot/"
python docs/env-worker-deep-dive/env-service-verification/scripts/science_pressure_training.py \
 --repo "$PSRL_WORKSPACE" --model "${MODEL_ROOT:?Set MODEL_ROOT}/Qwen3.5-4B" \
 --dataset "$PSRL_WORKSPACE/examples/sciaccel_rl/data/laps/repair_easy/train/L1.parquet" \
 --node-ip "$NODE_IP" --output "$result_dir" --steps 4 --timeout 3300 --task-timeout 240 --turns 16 "$@"
