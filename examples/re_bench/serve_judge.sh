#!/bin/bash
# Serve one dedicated Qwen replica for a bounded validation or experiment.
set -euo pipefail
: "${JUDGE_MODEL_PATH:?Set the local Qwen model directory}"
: "${JUDGE_GPU_UUIDS:?Set dedicated GPU UUIDs, disjoint from task GPUs}"
: "${JUDGE_SERVED_MODEL:?Set the model identity used by prepared tasks}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Resolve actual CUDA ordinals because some vLLM versions reject UUID visibility strings.
CUDA_VISIBLE_DEVICES=$(env -u CUDA_VISIBLE_DEVICES python - <<'PYCODE'
import os
import torch

wanted = os.environ["JUDGE_GPU_UUIDS"].split(",")
devices = {str(torch.cuda.get_device_properties(i).uuid): i for i in range(torch.cuda.device_count())}
print(",".join(str(devices[uuid.removeprefix("GPU-")]) for uuid in wanted))
PYCODE
)
export CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
exec timeout --signal=TERM --kill-after=30 "${JUDGE_LIFETIME_S:-1800}" \
  python -m vllm.entrypoints.openai.api_server \
  --model "$JUDGE_MODEL_PATH" --served-model-name "$JUDGE_SERVED_MODEL" \
  --host "${JUDGE_HOST:-127.0.0.1}" --port "${JUDGE_PORT:-18764}" \
  --tensor-parallel-size "${JUDGE_TP:-1}" --dtype bfloat16 \
  --gpu-memory-utilization "${JUDGE_MEMORY_FRACTION:-0.45}" \
  --max-model-len 8192 --max-num-seqs 8 --enforce-eager --no-enable-log-requests
