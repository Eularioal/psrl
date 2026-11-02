"""Forward bounded artifact requests to the official Replicate judge."""

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "meta/meta-llama-3-8b-instruct"


def validate_request(payload):
    if payload.get("model") != MODEL:
        raise ValueError("Only the official RE-Bench judge is allowed.")
    inputs = payload.get("input", {})
    if set(inputs) != {"prompt", "system_prompt", "temperature", "max_tokens"}:
        raise ValueError("Unexpected judge input fields.")
    if any(not isinstance(inputs[k], str) or len(inputs[k]) > 60000 for k in ("prompt", "system_prompt")):
        raise ValueError("Judge text exceeds its bounded input size.")
    if inputs["temperature"] != 0.0 or inputs["max_tokens"] != 100:
        raise ValueError("Keep the official judge sampling parameters.")
    return inputs


def validate_ranking(output):
    try:
        ranking = json.loads(output)
    except json.JSONDecodeError:
        match = re.search(r"\[\s*{.+?}\s*\]", output, re.DOTALL)
        if match is None:
            raise ValueError("Judge did not return a ranking.") from None
        ranking = json.loads(match.group())
    if (
        not isinstance(ranking, list)
        or len(ranking) != 2
        or any(not isinstance(item, dict) for item in ranking)
        or {item.get("model") for item in ranking} != {"model_1", "model_2"}
        or {item.get("rank") for item in ranking} != {1, 2}
    ):
        raise ValueError("Judge returned an invalid ranking.")
    return output


def call_judge(inputs, token, timeout):
    deadline = time.monotonic() + timeout

    def request(path, data=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("The Replicate judge exceeded its deadline.")
        req = urllib.request.Request(
            "https://api.replicate.com/v1/" + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=min(30, remaining)) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Replicate returned HTTP {exc.code}.") from None

    result = request(f"models/{MODEL}/predictions", {"input": inputs})
    prediction_id = result.get("id", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", prediction_id):
        raise RuntimeError("Replicate returned an invalid prediction identifier.")
    while result.get("status") not in ("succeeded", "failed", "canceled"):
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
        result = request(f"predictions/{prediction_id}")
    if result.get("status") != "succeeded":
        raise RuntimeError("The Replicate prediction did not succeed.")
    output = result.get("output")
    if isinstance(output, list):
        output = "".join(output)
    if not isinstance(output, str):
        raise ValueError("Replicate did not return text.")
    return validate_ranking(output)


def serve(root, token, max_calls, timeout, workers, stop):
    root = root.resolve()
    seen = set()
    pending = {}
    counters = {"submitted": 0, "succeeded": 0, "failed": 0}

    def respond(path, result):
        destination = path.parent.parent / "responses" / path.name
        destination.parent.mkdir(exist_ok=True)
        if not destination.resolve().is_relative_to(root):
            raise ValueError("Judge response escaped the run directory.")
        temporary = destination.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(result, stream)
        temporary.replace(destination)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        while not stop.is_set() or pending:
            for future in list(pending):
                if not future.done():
                    continue
                path = pending.pop(future)
                try:
                    result = {"output": future.result()}
                    counters["succeeded"] += 1
                except Exception as exc:
                    result = {"error": type(exc).__name__ + ": " + str(exc)}
                    counters["failed"] += 1
                respond(path, result)
            if not stop.is_set():
                for path in root.glob("**/rebench-judge/requests/*.json"):
                    if path in seen or len(pending) >= workers:
                        continue
                    seen.add(path)
                    if (path.parent.parent / "responses" / path.name).exists():
                        continue
                    if not path.resolve().is_relative_to(root) or path.is_symlink():
                        continue
                    try:
                        if path.stat().st_size > 150000:
                            raise ValueError("Judge request is too large.")
                        inputs = validate_request(json.loads(path.read_text()))
                        if counters["submitted"] >= max_calls:
                            raise RuntimeError("The run exhausted its judge request budget.")
                        counters["submitted"] += 1
                        pending[pool.submit(call_judge, inputs, token, timeout)] = path
                    except Exception as exc:
                        counters["failed"] += 1
                        respond(path, {"error": str(exc)})
            stop.wait(0.1) if not stop.is_set() else time.sleep(0.1)
    return counters


def main():
    import threading

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--max-calls", type=int, default=2048)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        parser.error("Configure REPLICATE_API_TOKEN in the host environment.")
    if min(args.duration, args.max_calls, args.request_timeout, args.workers) <= 0:
        parser.error("All budgets must be positive.")
    stop = threading.Event()
    timer = threading.Timer(args.duration, stop.set)
    timer.start()
    try:
        print(json.dumps(serve(args.jobs_dir, token, args.max_calls, args.request_timeout, args.workers, stop)))
    finally:
        stop.set()
        timer.cancel()


if __name__ == "__main__":
    main()
