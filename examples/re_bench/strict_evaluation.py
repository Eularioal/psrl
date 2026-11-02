"""Retain the official preference evaluator and reject transport failures."""

import evaluation_official
import replicate


async def evaluate(*args, **kwargs):
    before = replicate.failures
    result = await evaluation_official.evaluate(*args, **kwargs)
    if replicate.failures != before:
        raise RuntimeError("Judge transport failed. No valid preference feedback is available.")
    return result
