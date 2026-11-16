"""Audit fixed baseline sources and demonstrate state lifetimes using only CPU."""

import argparse
import hashlib
import json
import os
import re
import selectors
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASELINE = "0eeead861ffafed6bf31c962d259fd847775dcf9"
SOURCES = {
    "agent-loop": "examples/sciaccel_rl/agent_loop.py",
    "runner": "examples/sciaccel_rl/runner.py",
    "worker": "psrl/workers/env_worker/worker.py",
    "shell": "psrl/workers/env_worker/shell.py",
}


def stop(process):
    """Terminate only a child created by this experiment."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def demonstrate(shell_source):
    """Use baseline wrapping on local bash, then demonstrate ordinary Python imports."""
    namespace = {"__file__": "baseline_shell.py"}
    exec(compile(shell_source, "baseline_shell.py", "exec"), namespace)
    with tempfile.TemporaryDirectory(prefix="baseline-state-") as scratch:
        root = Path(scratch)
        shell = subprocess.Popen(
            ["bash", "--noprofile", "--norc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        selector = selectors.DefaultSelector()
        selector.register(shell.stdout, selectors.EVENT_READ)

        def command(text):
            shell.stdin.write(namespace["wrap_command"](text).encode())
            shell.stdin.flush()
            buffer = ""
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if selector.select(timeout=0.1):
                    chunk = os.read(shell.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError("The baseline shell exited before its sentinel.")
                    buffer += chunk.decode()
                    parsed = namespace["parse_sentinel"](buffer)
                    if parsed:
                        if parsed[1] != 0:
                            raise RuntimeError(f"Shell command failed: {parsed!r}.")
                        return parsed[0].strip()
            raise TimeoutError("The bounded CPU shell command did not complete.")

        try:
            command(f"cd {shlex.quote(str(root))}\nexport AUDIT_VALUE=kept\nprintf 'v1' > state.txt")
            before = command('printf "%s " "$AUDIT_VALUE"\ncat state.txt')
            command("printf 'v2' > state.txt")
            after = command('printf "%s " "$AUDIT_VALUE"\ncat state.txt')
        finally:
            stop(shell)
            selector.close()
            shell.stdin.close()
            shell.stdout.close()

        module = root / "candidate.py"
        module.write_text("VALUE = 1\n")
        server = root / "server.py"
        server.write_text(
            "import sys\n"
            "sys.dont_write_bytecode = True\n"
            "import candidate\n"
            "for line in sys.stdin:\n"
            "    print(candidate.VALUE, flush=True)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-u", str(server)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)

        def query():
            process.stdin.write("read\n")
            process.stdin.flush()
            if not selector.select(timeout=5):
                raise TimeoutError("The CPU state holder did not respond.")
            return int(process.stdout.readline())

        try:
            old = query()
            module.write_text("VALUE = 2\n")
            retained = query()
            fresh = int(
                subprocess.check_output(
                    [sys.executable, "-B", "-c", "import candidate; print(candidate.VALUE)"], cwd=root, timeout=5
                )
            )
        finally:
            stop(process)
            selector.close()
            process.stdin.close()
            process.stdout.close()
    if (before, after, old, retained, fresh) != ("kept v1", "kept v2", 1, 1, 2):
        raise RuntimeError("State lifetime observations differ from the expected mechanism.")
    return dict(
        scope="Baseline wrap/parse on local bash and Python import demonstration. No Harbor, Docker or GPU.",
        shell_before=before,
        shell_after=after,
        python_before_edit=old,
        python_retained_after_edit=retained,
        python_fresh_after_edit=fresh,
        child_processes_reaped=True,
    )


def profile(root):
    """Extract available stage measurements without inventing missing model timings."""
    evidence = json.loads((root / "performance/evidence.json").read_text())
    if evidence["status"] != "passed":
        raise ValueError("Only completed comparisons can be profiled.")
    rows = []
    for label, command in evidence["commands"].items():
        if "warm" in label:
            continue
        log = (root / "performance" / f"{label}.log").read_text()
        pytest = re.search(r"4 passed.*? in ([0-9.]+)s", log)
        shards = re.findall(r"100% Completed.*?([0-9.]+)s/it", log)
        prep, reservation = command["preparation"], command["reservation"]
        rows.append(
            dict(
                label=label,
                cpu_import_s=prep["ready"] - prep["started"],
                gpu_reservation_s=reservation["released_at"] - reservation["admitted_at"],
                pytest_total_s=float(pytest[1]) if pytest else None,
                shard_progress_s_per_iteration=float(shards[-1]) if shards else None,
            )
        )
    return dict(
        scope="Prior reference service logs. No new edit trajectories or historical baseline execution.",
        rows=rows,
        mean_cpu_import_s=statistics.mean(row["cpu_import_s"] for row in rows),
        missing=[
            "CPU weight parsing",
            "complete model initialization",
            "GPU compute-only duration",
            "edit invalidation trace",
        ],
        caution="Pytest includes fixture lifecycle. Shard progress is partial loading, not a disjoint stage.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-run", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, sources = {}, {}
    for name, path in SOURCES.items():
        source = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=args.repo)
        target = args.output / f"baseline-{name}.py.txt"
        target.write_bytes(source)
        sources[name] = source.decode()
        manifest[name] = dict(source=path, snapshot=target.name, sha256=hashlib.sha256(source).hexdigest())
    result = dict(baseline=BASELINE, sources=manifest, cpu_observations=demonstrate(sources["shell"]))
    if args.score_run:
        result["prior_service_profile"] = profile(args.score_run)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
