#!/usr/bin/env bash
set -euo pipefail

repo=$1
dataset=$2
model=$3
out=$4
node_ip=$5
mkdir -p "$out"
export VERIFY_DATASET="$dataset" VERIFY_OUT="$out"
python - <<'PY'
import os
from pathlib import Path
import pandas as pd
import yaml
p = Path(os.environ['VERIFY_OUT'])
d = pd.read_parquet(os.environ['VERIFY_DATASET'])
d = d[d['task_name'] == 'sciaccel/laps-repair-coef-2d-rktmod-l17k1']
assert len(d) == 1
d.to_parquet(p / 'one-task.parquet', index=False)
agent = yaml.safe_load(Path('examples/sciaccel_rl/config/sciaccel_agent_config.yaml').read_text())
agent[0]['harbor']['jobs_dir'] = str(p / 'harbor-jobs')
agent[0]['harbor']['max_concurrent_episodes'] = 2
agent[0]['task_timeout_sec'] = int(os.environ.get('VERIFY_TASK_TIMEOUT', '600'))
(p / 'agent.yaml').write_text(yaml.safe_dump(agent))
PY
export PSRL_PATH="$repo" HF_MODEL_PATH="$model" OUTPUT_DIR="$out"
export DATA_DIR="$(dirname "$(dirname "$dataset")")" HINT_LEVEL=L1
export TRAIN_BATCH_SIZE=1 TOTAL_TRAINING_STEPS=1 MAX_RESPONSE_LENGTH=${VERIFY_RESPONSE_LENGTH:-32768}
export MAX_TURNS=${VERIFY_MAX_TURNS:-16} TRAIN_SP=1 TRAIN_FSDP=2 STALENESS=0
export AGENT_NODE_IPS="$node_ip" AGENT_LOOP_WORKERS=1
export WANDB_MODE=disabled SAVE_FREQ=-1 TEST_FREQ=-1
export SCIACCEL_TMPDIR=/tmp
ray_options=(++ray_kwargs.ray_init.address=local "++ray_kwargs.ray_init._temp_dir=/tmp/evt-$RANDOM")
if [[ -n "${VERIFY_RAY_ADDRESS:-}" ]]; then
    export RAY_ADDRESS="$VERIFY_RAY_ADDRESS"
    ray_options=("++ray_kwargs.ray_init.address=$VERIFY_RAY_ADDRESS"
        "++ray_kwargs.ray_init.runtime_env.env_vars.RAY_ADDRESS=$VERIFY_RAY_ADDRESS"
        '++ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_ENABLE="1"')
fi
if [[ -n "${VERIFY_DEPENDENCY_PATH:-}" ]]; then
    export PYTHONPATH="$repo:$VERIFY_DEPENDENCY_PATH"
    ray_options+=("++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PYTHONPATH")
fi
timeout --signal=TERM --kill-after=30 "${VERIFY_TIMEOUT_SECONDS:-1200}" bash examples/sciaccel_rl/fsdp_qwen35_4b.sh \
    psrl.rollout_n=2 psrl.staleness_buffer_entries=1 \
    psrl.deployment.n_rollout_instances=1 \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=2 \
    psrl.deployment.train_nnodes=1 psrl.deployment.train_ngpus_per_node=2 \
    psrl.deployment.total_nnodes=1 psrl.deployment.n_validate_instances=0 \
    psrl.colocate_validate_and_train=False \
    psrl.env_service.enable=True psrl.env_service.node_policy=cpu_only \
    "psrl.env_service.units=[{unit_id:science-cpu,node_ip:$node_ip,cpu_quota:16,memory_mb:32768,max_containers:2}]" \
    gen_actor_rollout_ref.rollout.n=2 \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    gen_actor_rollout_ref.rollout.enforce_eager=True \
    ++gen_actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
    "gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=$out/agent.yaml" \
    train_actor_rollout_ref.actor.rollout_n=2 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    train_actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    "data.train_files=$out/one-task.parquet" "data.val_files=$out/one-task.parquet" \
    trainer.nnodes=1 trainer.n_gpus_per_node=8 trainer.logger='[console]' \
    trainer.resume_mode=disable trainer.total_training_steps=1 \
    "${ray_options[@]}" \
    ++ray_kwargs.ray_init.include_dashboard=False
