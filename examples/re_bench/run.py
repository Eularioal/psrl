"""Run a native Env episode with a bounded host judge bridge when required."""

import asyncio
import json
import os
import threading

from examples.gpu_benchmarks.run import build_parser
from examples.gpu_benchmarks.run import run as run_episode
from examples.re_bench.judge import serve


async def run(args):
    provenance = json.loads((args.task / "provenance.json").read_text())
    if provenance.get("benchmark") != "re_bench":
        raise ValueError("Select a prepared RE-Bench task.")
    if args.environment_backend != "env_worker":
        raise ValueError("The RE-Bench recipe requires the native EnvWorker backend.")
    if provenance["mode"] == "smoke":
        await run_episode(args)
        return
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError("Configure REPLICATE_API_TOKEN before starting the official task.")
    if min(args.judge_max_calls, args.judge_timeout, args.judge_workers) <= 0:
        raise ValueError("Judge budgets must be positive.")
    if args.output.exists():
        raise FileExistsError(args.output)
    stop = threading.Event()
    broker = asyncio.create_task(
        asyncio.to_thread(
            serve, args.output, token, args.judge_max_calls, args.judge_timeout, args.judge_workers, stop
        )
    )
    try:
        await run_episode(args)
    finally:
        stop.set()
        counters = await broker
        if args.output.is_dir():
            (args.output / "judge.json").write_text(json.dumps(counters, indent=2) + "\n")


def main():
    parser = build_parser(__doc__)
    parser.set_defaults(environment_backend="env_worker")
    parser.add_argument("--judge-max-calls", type=int, default=2048)
    parser.add_argument("--judge-timeout", type=float, default=120)
    parser.add_argument("--judge-workers", type=int, default=20)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
