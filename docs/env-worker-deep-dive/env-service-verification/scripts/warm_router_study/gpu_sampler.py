"""Record optional GPU observations without blocking the experiment event loop."""

import csv
import json
import math
import subprocess
import threading
import time
from pathlib import Path

_FIELDS = ("utilization_gpu_pct", "memory_used_mb", "clock_sm_mhz", "power_draw_w")


class GpuSampler:
    """Sample exactly two selected GPUs in a daemon thread with bounded shutdown."""

    def __init__(self, gpus: list[int], path: Path, interval_s: float = 2):
        if len(gpus) != 2 or len(set(gpus)) != 2 or any(type(index) is not int or index < 0 for index in gpus):
            raise ValueError("Exactly two distinct nonnegative GPU indices are required.")
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("The sampling interval must be finite and positive.")
        self.gpus = tuple(gpus)
        self.path = Path(path)
        self.interval_s = interval_s
        self.last_error = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        """Start optional monitoring without creating files on the caller's thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"status": "running"}
            self._stop = threading.Event()
            self.last_error = None
            self._thread = threading.Thread(target=self._run, name="warm-gpu-sampler", daemon=True)
            try:
                self._thread.start()
            except Exception as error:
                self.last_error = {"status": "unavailable", "error_type": type(error).__name__}
                return dict(self.last_error)
        return {"status": "started"}

    def stop(self):
        """Stop scheduling observations and wait at most five seconds for the daemon."""
        with self._lock:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            try:
                thread.join(timeout=5)
            except RuntimeError:
                pass
        return {
            "status": "stop_timeout" if thread is not None and thread.is_alive() else "stopped",
            "last_error": self.last_error,
        }

    def _sample(self):
        started = time.monotonic()
        command = [
            "nvidia-smi",
            "--id=" + ",".join(map(str, self.gpus)),
            "--query-gpu=index,utilization.gpu,memory.used,clocks.sm,power.draw",
            "--format=csv,noheader,nounits",
        ]
        status, observed = "ok", {}
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
            if result.returncode:
                status = "failed"
            else:
                for row in csv.reader(result.stdout.splitlines()):
                    if len(row) != 5:
                        status = "invalid"
                        continue
                    try:
                        index = int(row[0])
                    except ValueError:
                        status = "invalid"
                        continue
                    if index not in self.gpus:
                        continue
                    if index in observed:
                        observed[index] = {"status": "invalid"}
                        continue
                    values = {}
                    for field, raw in zip(_FIELDS, row[1:], strict=True):
                        try:
                            value = float(raw.strip())
                            values[field] = value if math.isfinite(value) and value >= 0 else None
                        except ValueError:
                            values[field] = None
                    observed[index] = dict(status="partial" if None in values.values() else "ok", **values)
        except subprocess.TimeoutExpired:
            status = "timeout"
        except OSError:
            status = "unavailable"
        except Exception as error:
            status = "unavailable"
            self.last_error = {"status": status, "error_type": type(error).__name__}
        observed_at = time.time()
        elapsed = time.monotonic() - started
        return [
            dict(
                time_unix_s=observed_at,
                index=index,
                query_wall_s=elapsed,
                **observed.get(index, {"status": "missing" if status == "ok" else status}),
            )
            for index in self.gpus
        ]

    def _run(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", buffering=1) as stream:
                while not self._stop.is_set():
                    started = time.monotonic()
                    rows = self._sample()
                    for row in rows:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    self._stop.wait(max(0, self.interval_s - (time.monotonic() - started)))
        except Exception as error:
            self.last_error = {"status": "unavailable", "error_type": type(error).__name__}
