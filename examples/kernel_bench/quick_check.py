"""Check prepared references with the official GPU evaluator under bounded budgets."""

import ast
import contextlib
import json
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path


def bounded_source(source):
    """Bound explicit module dimensions for a separately labelled interface check."""
    overrides = {}
    names = {"n", "m", "k", "b", "c", "h", "w", "d", "l", "dim", "height", "width", "depth", "length"}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name.lower() not in names and not name.lower().endswith(
            ("_size", "_dim", "_length", "_features", "_channels", "_shape")
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        limit = 2 if "batch" in name.lower() else 32
        if isinstance(value, int) and value > limit:
            overrides[name] = limit
        elif isinstance(value, (tuple, list)) and value and all(isinstance(x, int) for x in value):
            overrides[name] = type(value)(min(x, limit) for x in value)
    source += "\n" + "\n".join(f"{name} = {value!r}" for name, value in overrides.items()) + "\n"
    return source, overrides


def check_one(item, destination):
    """Use the public reference as a correctness control, without speed measurements."""
    with destination.with_suffix(".log").open("w") as stream, contextlib.redirect_stdout(stream):
        try:
            if item.get("kind") == "flashinfer":
                from examples.flashinfer_bench.verify import main as verify

                output = destination.with_suffix("")
                output.mkdir()
                subset = Path(item["subset"])
                verify(subset / "solutions/reference.json", output, subset)
                reward = json.loads((output / "reward.json").read_text())
                record = {"status": "passed" if reward["reward"] == 1 else "failed", "result": reward}
                destination.write_text(json.dumps(record))
                return
            import torch
            from kernelbench.eval import eval_kernel_against_ref

            source = Path("/opt/upstream", item["reference"]).read_text()
            overrides = {}
            if item.get("constraint_profile"):
                from examples.kernel_bench.local_profile import evaluate_source, prepare_source

                source, overrides = prepare_source(source)
                result = evaluate_source(source, source + "\nModelNew = Model\n")
                destination.write_text(json.dumps({"status": "passed", "result": result, "overrides": overrides}))
                return
            if item.get("bounded_inputs"):
                source, overrides = bounded_source(source)
            result = eval_kernel_against_ref(
                source,
                source + "\nModelNew = Model\n",
                num_correct_trials=1,
                num_perf_trials=1,
                measure_performance=False,
                device=torch.device("cuda:0"),
                backend="cuda",
            )
            record = {"status": "passed" if result.correctness else "failed", "result": result.model_dump()}
            record.update(
                profile="bounded_module_dimensions" if item.get("bounded_inputs") else "original_inputs",
                overrides=overrides,
            )
        except Exception as exc:
            record = {"status": "error", "error": repr(exc)}
    destination.write_text(json.dumps(record, default=str))


def worker_loop(connection):
    """Retain imports and the CUDA context while isolating stalled checks by process group."""
    os.setsid()
    while True:
        item, destination = connection.recv()
        check_one(item, destination)
        connection.send(True)


def stop_worker(worker):
    if worker is not None and worker.is_alive():
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            worker.kill()
        worker.join(5)


def main():
    import torch

    torch.set_num_threads(1)
    config = json.loads(Path("/tests/check.json").read_text())
    output = Path("/logs/verifier")
    deadline = time.monotonic() + config["suite_seconds"]
    results = []
    worker = None
    for item in config["tasks"]:
        record = {"task_id": item["task_id"], "reference": item["reference"], "level": item["level"]}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            record["status"] = "not_run_budget"
        else:
            destination = output / f"{item['task_id']}.json"
            started = time.monotonic()
            if worker is None or not worker.is_alive():
                connection, child_connection = mp.get_context("fork").Pipe()
                worker = mp.get_context("fork").Process(target=worker_loop, args=(child_connection,))
                worker.start()
                budget = max(config["task_seconds"], config.get("worker_start_seconds", 30))
            else:
                budget = config["task_seconds"]
            connection.send((item, destination))
            if not connection.poll(min(budget, remaining)):
                stop_worker(worker)
                record["status"] = "timeout"
            else:
                try:
                    connection.recv()
                    record.update(json.loads(destination.read_text()))
                except (EOFError, OSError, ValueError) as exc:
                    record.update(status="worker_error", error=repr(exc))
                    stop_worker(worker)
            record["seconds"] = time.monotonic() - started
        results.append(record)
        (output / "quick-results.json").write_text(json.dumps(results, indent=2, default=str))
    stop_worker(worker)
    passed = sum(item["status"] == "passed" for item in results)
    (output / "reward.json").write_text(
        json.dumps({"reward": 0.0, "quick_check": 1, "passed_tasks": passed, "total_tasks": len(results)})
    )


if __name__ == "__main__":
    main()
