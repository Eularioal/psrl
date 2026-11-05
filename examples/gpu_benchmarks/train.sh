#!/bin/bash
# Launch benchmark training through the shared PSRL Harbor training recipe.
set -euo pipefail
recipe=${1:?Pass kernel_bench, mle_bench, inference_bench, autoresearch, or re_bench}
shift
case "$recipe" in
  kernel_bench|mle_bench|inference_bench|autoresearch|re_bench) ;;
  *) exit 2 ;;
esac
if [[ "$recipe" == re_bench ]]; then
  : "${REBENCH_JOBS_DIR:?Set REBENCH_JOBS_DIR to a dedicated shared job directory}"
fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${TRAIN_FILES:?Set TRAIN_FILES to prepared training Parquet}"
: "${VAL_FILES:?Set VAL_FILES to a disjoint validation Parquet}"
export PSRL_PATH="$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export AGENT_LOOP_CONFIG_PATH="$repo/examples/$recipe/agent_config.yaml"
if [[ "${GPU_RESERVATION:-episode}" == command ]]; then
  [[ "$recipe" == kernel_bench ]] || { echo "Command recipe requires kernel_bench" >&2; exit 2; }
  export AGENT_LOOP_CONFIG_PATH="$repo/examples/kernel_bench/command_agent_config.yaml"
elif [[ "${GPU_RESERVATION:-episode}" != episode ]]; then
  echo "GPU_RESERVATION must be episode or command" >&2
  exit 2
fi
export REWARD_PATH="$repo/examples/gpu_benchmarks/reward.py"
export DEFAULT_AGENT_LOOP=gpu_benchmark PROJECT_NAME="psrl_$recipe"
export OUTPUT_DIR="${OUTPUT_DIR:-$repo/outputs/$recipe}"
exec bash "$repo/examples/sciaccel_rl/fsdp_qwen35_4b.sh" psrl.env_service.enable=True "$@"
