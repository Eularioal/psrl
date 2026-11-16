#!/bin/bash
# Train AIRS GPU tasks through the PSRL session and native sandbox recipe.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${AIRS_DATA_ROOT:?Set AIRS_DATA_ROOT to the prepared bounded suite}"
: "${MLGYM_REPO:?Set MLGYM_REPO to the pinned MLGym source}"
: "${TRAIN_FILES:?Set TRAIN_FILES to prepared training Parquet}"
: "${VAL_FILES:?Set VAL_FILES to prepared validation Parquet}"
: "${HF_MODEL_PATH:?Set HF_MODEL_PATH to local policy weights}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh experiment directory}"
: "${LOCAL_IP:?Set LOCAL_IP to the verified Ray node address}"
export PSRL_PATH="$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export AIRS_AGENT_CONFIG="${AIRS_AGENT_CONFIG:-$repo/examples/airs_bench/agent_config.yaml}"
export AGENT_LOOP_CONFIG_PATH="$repo/examples/airs_bench/config/gpu_agent_config.yaml"
export REWARD_PATH="$repo/examples/airs_bench/reward.py"
export DEFAULT_AGENT_LOOP=mlgym_agent PROJECT_NAME=psrl_airs_gpu
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1} TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
export TRAIN_SP=${TRAIN_SP:-1} TRAIN_FSDP=${TRAIN_FSDP:-2} AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-1}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-12288} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384} MAX_TURNS=${MAX_TURNS:-6}
export SAVE_FREQ=${SAVE_FREQ:--1} TEST_FREQ=${TEST_FREQ:--1}
exec bash "$repo/examples/sciaccel_rl/fsdp_qwen35_4b.sh" \
  psrl.env_service.enable=False psrl.env_worker.enable=True \
  psrl.env_worker.gpu_slots_per_worker=1 psrl.env_worker.cpu_slots_per_worker=1 \
  psrl.env_worker.cpu_budget_per_worker=6 psrl.env_worker.memory_budget_bytes_per_worker=21474836480 \
  psrl.deployment.total_nnodes=1 psrl.deployment.train_nnodes=1 psrl.deployment.train_ngpus_per_node=2 \
  psrl.deployment.n_rollout_instances=1 psrl.deployment.rollout_nnodes_per_instance=1 \
  psrl.deployment.rollout_ngpus_per_node_per_instance=1 psrl.deployment.n_validate_instances=0 \
  psrl.colocate_validate_and_train=False psrl.rollout_n=2 \
  gen_actor_rollout_ref.rollout.n=2 gen_actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.55 gen_actor_rollout_ref.rollout.enforce_eager=True \
  train_actor_rollout_ref.actor.rollout_n=2 train_actor_rollout_ref.rollout.n=2 \
  trainer.nnodes=1 trainer.n_gpus_per_node=3 trainer.resume_mode=disable \
  gen_actor_rollout_ref.rollout.agent.env.name=mlgym_env "$@"
