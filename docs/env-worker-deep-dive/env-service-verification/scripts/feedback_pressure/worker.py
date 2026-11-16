"""Run official candidate measurements with cooperative bounded competing GPU work."""

import json
import multiprocessing as mp
import random
import time
import traceback
from pathlib import Path
from statistics import median


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def pressure_worker(conn, pause, paused, started, done, execution_lock, cfg):
    import torch

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    n = cfg["matrix_size"]
    left = torch.randn(n, n, device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    out = torch.empty_like(left)
    source = torch.ones(cfg["vector_elements"], device="cuda")
    target = torch.empty_like(source)
    torch.mm(left, right, out=out)
    target.copy_(source)
    torch.cuda.synchronize()
    conn.send({"ready": True, "allocated": torch.cuda.memory_allocated()})
    while True:
        job = conn.recv()
        if job is None:
            break
        kind, count = job
        began = time.monotonic()
        for _ in range(count):
            while True:
                while pause.is_set():
                    paused.set()
                    time.sleep(0.0005)
                with execution_lock:
                    if pause.is_set():
                        continue
                    paused.clear()
                    for _ in range(cfg["operations_per_unit"]):
                        if kind == "memory":
                            target.copy_(source)
                        else:
                            torch.mm(left, right, out=out)
                    torch.cuda.synchronize()
                    started.set()
                    break
        result = {
            "units": count,
            "kind": kind,
            "elapsed_s": time.monotonic() - began,
            "peak_bytes": torch.cuda.max_memory_allocated(),
        }
        # No further CUDA work after the synchronized final unit.
        done.set()
        conn.send(result)


class Pressure:
    def __init__(self, cfg):
        self.workers = []
        self.active = []
        self.transitions = []
        self.held = []
        ctx = mp.get_context("spawn")
        for _ in range(8):
            parent, child = ctx.Pipe()
            pause, paused, started, done = [ctx.Event() for _ in range(4)]
            execution_lock = ctx.Lock()
            process = ctx.Process(
                target=pressure_worker, args=(child, pause, paused, started, done, execution_lock, cfg)
            )
            process.start()
            self.workers.append((process, parent, pause, paused, started, done, execution_lock))
        for process, conn, *_ in self.workers:
            if not conn.poll(120):
                raise TimeoutError("Pressure worker initialization timed out.")
            assert conn.recv()["ready"], "Pressure worker must initialize CUDA."

    def start(self, kinds, units):
        self.active = self.workers[: len(kinds)]
        self.transitions = []
        for worker, kind in zip(self.active, kinds):
            _, conn, pause, paused, started, done, _ = worker
            for event in (pause, paused, started, done):
                event.clear()
            conn.send((kind, units))
        for _, _, _, _, started, _, _ in self.active:
            if not started.wait(30):
                raise TimeoutError("Pressure did not begin real GPU work.")

    def protect(self):
        began = time.monotonic()
        for _, _, pause, *_ in self.active:
            pause.set()
        for worker in self.active:
            lock = worker[6]
            if not lock.acquire(timeout=30):
                raise TimeoutError("Competing CUDA work did not reach a synchronized boundary.")
            self.held.append(lock)
            worker[3].set()
        self.transitions.append(time.monotonic() - began)

    def release(self):
        for lock in self.held:
            lock.release()
        self.held.clear()
        for _, _, pause, *_ in self.active:
            pause.clear()

    def finish(self):
        self.release()
        result = []
        for _, conn, *_ in self.active:
            if not conn.poll(90):
                raise TimeoutError("Fixed competing workload did not complete.")
            result.append(conn.recv())
        return result

    def close(self):
        self.release()
        for process, conn, *_ in self.workers:
            if process.is_alive():
                conn.send(None)
        for process, *_ in self.workers:
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(5)


def collect(root):
    import torch
    from flashinfer_bench import BenchmarkConfig, TraceSet
    from flashinfer_bench.bench.evaluators.default import DefaultEvaluator
    from flashinfer_bench.compile import BuilderRegistry

    torch.set_num_threads(1)
    torch.manual_seed(20261112)
    protocol = json.loads((root / "protocol.json").read_text())
    traces = TraceSet.from_path(root / "subset")
    registry = BuilderRegistry.get_instance()
    config = BenchmarkConfig.default()
    report = {"scope": protocol["scope"], "portability": [], "groups": [], "references": {}, "passed": False}
    cases = []
    pool = None

    def clone(inputs):
        return [[v.clone() if isinstance(v, torch.Tensor) else v for v in row] for row in inputs]

    def correctness(case, candidate):
        inputs = clone(case["baseline"].inputs)
        correctness_value, failure = DefaultEvaluator.check_correctness(
            case["definition"], candidate["fn"], inputs, case["baseline"].outputs, case["config"], "", "cuda:0"
        )
        if failure is not None:
            raise RuntimeError(f"Correctness failure: {failure.status!r}.")
        return correctness_value.model_dump(mode="json")

    def measure(case, candidate):
        value, error = DefaultEvaluator.eval_performance(
            case["definition"],
            candidate["fn"],
            clone(case["baseline"].inputs),
            case["baseline"].mean_latency_ms,
            case["config"],
            "",
            "cuda:0",
        )
        if error is not None:
            raise RuntimeError(f"Performance failure: {error.status!r}.")
        return value.latency_ms

    def references(label):
        values = {}
        for case in cases:
            values[case["id"]] = {c["name"]: [] for c in case["candidates"]}
        for repeat in range(protocol["reference_rounds"]):
            ordered = cases if repeat % 2 == 0 else list(reversed(cases))
            for case in ordered:
                candidates = case["candidates"]
                shift = repeat % len(candidates)
                for c in candidates[shift:] + candidates[:shift]:
                    values[case["id"]][c["name"]].append(measure(case, c))
        report["references"][label] = values
        save(root / "measurements.json", report)

    try:
        began = time.monotonic()
        for entry in protocol["inventory"]:
            definition = traces.definitions[entry["definition"]]
            cfg = config.resolve_eval_config(definition)
            for trace in traces.workloads[definition.name]:
                baseline = DefaultEvaluator.build_baseline(definition, trace.workload, cfg, "cuda:0", traces.root)
                case = {
                    "id": trace.workload.uuid,
                    "definition": definition,
                    "config": cfg,
                    "axes": trace.workload.axes,
                    "baseline": baseline,
                    "candidates": [],
                }
                for solution in traces.solutions[definition.name]:
                    row = {
                        "case": case["id"],
                        "definition": definition.name,
                        "axes": case["axes"],
                        "solution": solution.name,
                        "declared_hardware": solution.spec.target_hardware,
                    }
                    try:
                        candidate = {"name": solution.name, "fn": registry.build(definition, solution)}
                        row["correctness"] = correctness(case, candidate)
                        row["warmup_ms"] = measure(case, candidate)
                        row["passed"] = True
                        case["candidates"].append(candidate)
                    except Exception as error:
                        row["passed"] = False
                        row["error"] = repr(error)
                        row["traceback"] = traceback.format_exc()
                    report["portability"].append(row)
                    save(root / "measurements.json", report)
                assert len(case["candidates"]) >= 2, "Require two correct candidates for every original workload."
                cases.append(case)
        report["preparation_s"] = time.monotonic() - began
        report["configs"] = {c["definition"].name: c["config"].model_dump(mode="json") for c in cases}
        references("before")
        pool = Pressure(protocol["background"])
        report["reference_and_pressure_startup_s"] = time.monotonic() - began - report["preparation_s"]
        blocks = [
            (pressure, size, repeat)
            for pressure in protocol["pressures"]
            for size in protocol["group_sizes"]
            for repeat in range(protocol["repeats"])
        ]
        rng = random.Random(20261112)
        rng.shuffle(blocks)
        for block, (pressure, size, repeat) in enumerate(blocks):
            selected = cases if size == 12 else [cases[3 * i + repeat] for i in range(4)]
            policies = list(protocol["policies"])
            rng.shuffle(policies)
            for policy in policies:
                start = time.monotonic()
                pool.start(protocol["pressures"][pressure], protocol["background"]["units"])
                group = {
                    "block": block,
                    "pressure": pressure,
                    "size": size,
                    "repeat": repeat,
                    "policy": policy,
                    "cases": {},
                    "background_active_at_samples": 0,
                    "evaluations": 0,
                }
                if policy == "exclusive":
                    pool.protect()
                prep = time.monotonic()
                for case in selected:
                    for c in case["candidates"]:
                        correctness(case, c)
                    group["cases"][case["id"]] = {c["name"]: [] for c in case["candidates"]}
                group["correctness_s"] = time.monotonic() - prep
                if policy in ("group5", "group2", "adaptive"):
                    pool.protect()
                max_rounds = 2 if policy == "group2" else 5
                unresolved = {case["id"] for case in selected}
                for sample_round in range(max_rounds):
                    for case in selected:
                        if policy == "adaptive" and case["id"] not in unresolved:
                            continue
                        candidates = case["candidates"]
                        shift = (sample_round + repeat) % len(candidates)
                        for c in candidates[shift:] + candidates[:shift]:
                            if policy == "per_call":
                                pool.protect()
                            group["background_active_at_samples"] += int(
                                any(not w[5].is_set() and not w[3].is_set() for w in pool.active)
                            )
                            ms = measure(case, c)
                            group["cases"][case["id"]][c["name"]].append(ms)
                            group["evaluations"] += 1
                            if policy == "per_call":
                                pool.release()
                        if policy == "adaptive" and sample_round >= 1:
                            values = group["cases"][case["id"]]
                            scores = sorted((median(v), name) for name, v in values.items())
                            best, other = scores[:2]
                            spread = max(median([abs(x - median(v)) for x in v]) for v in values.values())
                            if other[0] - best[0] > max(0.05 * best[0], 4 * spread):
                                unresolved.discard(case["id"])
                    if policy == "adaptive" and not unresolved:
                        break
                group["feedback_s"] = time.monotonic() - start
                pool.release()
                group["background"] = pool.finish()
                group["makespan_s"] = time.monotonic() - start
                group["switch_wait_s"] = sum(pool.transitions)
                group["protected_entries"] = len(pool.transitions)
                group["adaptive_unresolved"] = sorted(unresolved) if policy == "adaptive" else []
                group["gpu_allocated_bytes"] = torch.cuda.memory_allocated()
                group["gpu_reserved_bytes"] = torch.cuda.memory_reserved()
                assert all(x["units"] == protocol["background"]["units"] for x in group["background"]), (
                    "Equal background work is required."
                )
                report["groups"].append(group)
                save(root / "measurements.json", report)
                print(
                    json.dumps(
                        {
                            "completed": len(report["groups"]),
                            "pressure": pressure,
                            "size": size,
                            "policy": policy,
                            "makespan_s": group["makespan_s"],
                        }
                    ),
                    flush=True,
                )
        pool.close()
        pool = None
        references("after")
        report["passed"] = len(report["groups"]) == protocol["expected_groups"]
        save(root / "measurements.json", report)
    except BaseException as error:
        report["error"] = repr(error)
        save(root / "measurements.json", report)
        raise
    finally:
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    collect(Path("/work"))
