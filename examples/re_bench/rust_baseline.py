"""Local Rust scaffold: CUDA language-model ranking plus public-test filtering."""

import asyncio
import json
from pathlib import Path

from rust_codecontests_utils import evaluate_rust_code


async def solve(problem, score_code):
    candidates = json.loads(Path("/data/public/candidates.json").read_text())[problem["name"]]
    ranked = sorted((score_code(problem["description"], code), code) for code in candidates)
    for _, code in ranked:
        report = await evaluate_rust_code(asyncio.Semaphore(1), problem, code, "/tmp")
        if report["succeeded"]:
            return code
    return ranked[0][1]
