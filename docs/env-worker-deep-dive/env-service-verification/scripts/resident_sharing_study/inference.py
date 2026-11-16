"""Measure real vLLM generation while retaining production sharing validation."""

import time
import uuid

from real_gpu_lending import InferenceOnlyServer


class StudyServer(InferenceOnlyServer):
    """Keep the inference model and memory profile identical across policies."""

    async def query(self, index=0):
        from vllm import SamplingParams

        prompt = (
            "Write a Python function to train an image classifier and explain each operation. "
            "Include data preparation, optimization, and validation.\n"
        )
        started = time.monotonic_ns()
        times = []
        result = None
        async for result in self.engine.generate(
            prompt,
            SamplingParams(temperature=0, max_tokens=64, min_tokens=64, ignore_eos=True),
            request_id=f"study-{index}-{uuid.uuid4().hex}",
        ):
            times.append(time.monotonic_ns())
        if result is None or not result.finished:
            raise RuntimeError("Generation did not finish.")
        return {
            "token_ids": list(result.outputs[0].token_ids),
            "engine_started_ns": started,
            "first_output_ns": times[0],
            "engine_ended_ns": times[-1],
            "engine_duration_s": (times[-1] - started) / 1e9,
        }
