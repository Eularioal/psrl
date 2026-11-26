"""Measure bounded single GPU inference curves with owned process cleanup."""

import argparse
import asyncio
import dataclasses
import importlib.metadata
import importlib.util
import json
import logging
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

psrl_logger = logging.getLogger(__name__)


def save(path, value):
    """Publish complete JSON snapshots atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def gpu_snapshot(gpu):
    """Read the selected physical GPU without starting a CUDA context."""
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(gpu)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle, version=pynvml.nvmlMemory_v2)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return dict(
        at_s=time.monotonic(),
        uuid=gpu,
        index=pynvml.nvmlDeviceGetIndex(handle),
        minor=pynvml.nvmlDeviceGetMinorNumber(handle),
        used_mib=memory.used / 1024**2,
        reserved_mib=memory.reserved / 1024**2,
        total_mib=memory.total / 1024**2,
        gpu_util_percent=utilization.gpu,
        memory_util_percent=utilization.memory,
        pids=[p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)],
    )


def bootstrap_transformers(repo):
    """Limit package metadata discovery to the local import phase."""
    spec = importlib.util.spec_from_file_location(
        "fast_metadata", repo / "patch/transformers/fast_package_metadata.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original = importlib.metadata.packages_distributions
    importlib.metadata.packages_distributions = helper.packages_distributions
    try:
        import transformers  # noqa: F401
    finally:
        importlib.metadata.packages_distributions = original


async def worker(args):
    """Collect client timing and engine scheduling evidence for each wave."""
    started = time.monotonic()
    evidence = dict(status="initializing", config=vars(args), waves=[], warmups=[])
    target = args.output / "evidence.json"
    save(target, evidence)
    bootstrap_transformers(args.repo)
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.metrics.loggers import StatLoggerBase

    counters = dict(preemptions=0, max_running=0, max_waiting=0, max_kv_usage=0.0)
    finished = []

    class CurveLogger(StatLoggerBase):
        def __init__(self, vllm_config, engine_index=0):
            pass

        def log_engine_initialized(self):
            pass

        def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
            if scheduler_stats is not None:
                counters["max_running"] = max(counters["max_running"], scheduler_stats.num_running_reqs)
                counters["max_waiting"] = max(counters["max_waiting"], scheduler_stats.num_waiting_reqs)
                counters["max_kv_usage"] = max(counters["max_kv_usage"], scheduler_stats.kv_cache_usage)
            if iteration_stats is not None:
                counters["preemptions"] += iteration_stats.num_preempted_reqs
                finished.extend(dataclasses.asdict(item) for item in iteration_stats.finished_requests)

    engine = None
    try:
        options = dict(
            model=str(args.model), tensor_parallel_size=1, dtype="bfloat16", enforce_eager=True,
            max_model_len=2560, max_num_seqs=256, max_num_batched_tokens=2048,
            kv_cache_memory_bytes=16 * 1024**3, gpu_memory_utilization=0.6,
            enable_prefix_caching=False, enable_chunked_prefill=True, disable_log_stats=True, seed=17,
        )
        evidence["engine_options"] = options
        evidence["sources"] = {
            name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue", "vllm")
        }
        evidence["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers")}
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**options), stat_loggers=[CurveLogger])
        evidence["initialization_s"] = time.monotonic() - started
        evidence["status"] = "running"
        save(target, evidence)

        async def wave(length, count, repetition, warmup=False):
            counters.update(preemptions=0, max_running=0, max_waiting=0, max_kv_usage=0.0)
            finished.clear()
            begin = time.monotonic()

            async def query(index):
                submitted = time.monotonic()
                first = None
                intervals = []
                last = None
                previous_count = 0
                result = None
                # Fixed token IDs avoid tokenization and cache hits changing the workload.
                prompt = [1000 + ((i + index) % 200) for i in range(length)]
                async for result in engine.generate(
                    {"prompt_token_ids": prompt},
                    SamplingParams(temperature=0, max_tokens=512, min_tokens=512, ignore_eos=True, detokenize=False),
                    request_id=f"{length}-{count}-{repetition}-{warmup}-{index}",
                ):
                    now = time.monotonic()
                    size = len(result.outputs[0].token_ids)
                    if size > previous_count:
                        if first is None:
                            first = now
                        elif last is not None:
                            intervals.append((now - last) / (size - previous_count))
                        previous_count, last = size, now
                if result is None or not result.finished or previous_count != 512:
                    raise RuntimeError("Generation did not finish with the required 512 tokens.")
                return dict(
                    index=index, submitted_s=submitted - begin, latency_s=last - submitted,
                    ttft_s=first - submitted, tpot_s=(last - first) / 511,
                    interval_p95_s=sorted(intervals)[int(0.95 * (len(intervals) - 1))],
                    output_tokens=previous_count,
                    metrics=dataclasses.asdict(result.metrics) if result.metrics is not None else None,
                )

            results = await asyncio.wait_for(asyncio.gather(*(query(i) for i in range(count))), timeout=150)
            elapsed = time.monotonic() - begin
            row = dict(
                input_tokens=length, concurrency=count, repetition=repetition, elapsed_s=elapsed,
                output_tokens_per_s=count * 512 / elapsed,
                mean_latency_s=statistics.mean(r["latency_s"] for r in results),
                mean_ttft_s=statistics.mean(r["ttft_s"] for r in results),
                mean_tpot_s=statistics.mean(r["tpot_s"] for r in results),
                counters=dict(counters), engine_finished=list(finished), requests=results,
            )
            evidence["warmups" if warmup else "waves"].append(row)
            save(target, evidence)
            psrl_logger.info(f"Completed input={length}, concurrency={count}, latency={row['mean_latency_s']:.3f}s.")

        await wave(128, 16, 0, warmup=True)
        for length, counts in ((128, [1, 2, 4, 8, 16, 32, 64, 128, 256]), (2048, [1, 8, 32, 64])):
            for repetition in range(2):
                for count in counts if repetition == 0 else list(reversed(counts)):
                    await wave(length, count, repetition)
        evidence["status"] = "passed"
    except BaseException:
        evidence["status"] = "failed"
        evidence["error"] = traceback.format_exc()
        raise
    finally:
        if engine is not None:
            engine.shutdown(timeout=15)
        evidence["worker_elapsed_s"] = time.monotonic() - started
        save(target, evidence)


def supervise(args):
    """Enforce the wall budget and verify physical release of this process group."""
    args.output.mkdir(parents=True, exist_ok=False)
    initial = gpu_snapshot(args.gpu)
    if initial["pids"] or initial["used_mib"] > 256:
        save(args.output / "preflight.json", initial)
        raise RuntimeError("The selected GPU is already occupied.")
    save(args.output / "preflight.json", initial)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    environment = dict(
        os.environ, CUDA_VISIBLE_DEVICES=str(initial["index"]), CUDA_DEVICE_ORDER="PCI_BUS_ID",
        TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
    )
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
    started = time.monotonic()
    with (args.output / "worker.log").open("w") as log, (args.output / "telemetry.jsonl").open("w") as telemetry:
        process = subprocess.Popen(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        save(args.output / "process.json", dict(pid=process.pid, command=command, budget_s=args.budget_s))
        timed_out = False
        try:
            while process.poll() is None:
                if time.monotonic() - started > args.budget_s:
                    timed_out = True
                    break
                try:
                    sample = gpu_snapshot(args.gpu)
                except Exception as error:
                    sample = dict(at_s=time.monotonic(), telemetry_error=repr(error))
                telemetry.write(json.dumps(sample) + "\n")
                telemetry.flush()
                time.sleep(2)
        finally:
            # Every worker descendant belongs to the dedicated process group.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                time.sleep(3)
            process.wait(timeout=10)
            final = None
            for _ in range(10):
                final = gpu_snapshot(args.gpu)
                if not final["pids"] and final["used_mib"] <= initial["used_mib"] + 32:
                    break
                time.sleep(2)
            save(args.output / "completion.json", dict(
                exit_code=process.returncode, timed_out=timed_out, elapsed_s=time.monotonic() - started,
                final_gpu=final, physical_release=not final["pids"] and final["used_mib"] <= initial["used_mib"] + 32,
            ))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-s", type=int, default=1200)
    parser.add_argument("--worker", action="store_true")
    arguments = parser.parse_args()
    if arguments.worker:
        asyncio.run(worker(arguments))
    else:
        supervise(arguments)
