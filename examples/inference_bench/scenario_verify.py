"""Bounded local input-heavy, output-heavy, concurrent and general service checks."""

import concurrent.futures
import json
import subprocess
import time
import urllib.request
from pathlib import Path

try:
    from .local_verify import stop_service
except ImportError:
    from local_verify import stop_service

PROFILES = {
    "a_input_heavy": {"repeat_context": 3, "max_tokens": 8, "requests": 2, "concurrency": 1},
    "b_output_heavy": {"repeat_context": 1, "max_tokens": 64, "requests": 2, "concurrency": 1},
    "c_high_load": {"repeat_context": 1, "max_tokens": 16, "requests": 4, "concurrency": 2},
    "d_general": {"repeat_context": 1, "max_tokens": 16, "requests": 2, "concurrency": 1},
}


def request(profile, index):
    samples = [json.loads(line) for line in Path("/opt/resident/samples.jsonl").read_text().splitlines()]
    messages = json.loads(json.dumps(samples[index % len(samples)]["messages"]))
    for message in messages:
        if message["role"] == "user":
            message["content"] = "\n".join([message["content"]] * profile["repeat_context"])
    payload = {
        "model": "/opt/model",
        "messages": messages,
        "temperature": 0,
        "max_tokens": profile["max_tokens"],
        "stream": False,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8001/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        body = json.load(response)
    text = body["choices"][0]["message"]["content"]
    if not text.strip():
        raise ValueError("Empty model output.")
    return {"text": text, "usage": body.get("usage")}


def main():
    config = json.loads(Path("/tests/profile.json").read_text())
    output = Path("/logs/verifier")
    service = None
    rows = []
    collect = config.get("collect_references", False)
    refs = {} if collect else json.loads(Path("/tests/references.json").read_text())
    metrics = {"reward": 0, "local_evaluation": 1, "eval_valid": 0, "quality_pass": 0}
    try:
        with (output / "service.log").open("w") as log:
            service = subprocess.Popen(
                ["bash", "/logs/artifacts/serve.sh"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            Path("/tmp/resident.pid").write_text(str(service.pid))
            link = Path("/logs/artifacts/server.log")
            link.unlink(missing_ok=True)
            link.symlink_to(output / "service.log")
            subprocess.run(["python", "/opt/resident/client.py", "--ready"], check=True, timeout=250)
            for name in config["profiles"]:
                profile = PROFILES[name]
                started = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=profile["concurrency"]) as pool:
                    answers = list(
                        pool.map(lambda i, profile=profile: request(profile, i), range(profile["requests"]))
                    )
                elapsed = time.monotonic() - started
                actual = [item["text"] for item in answers]
                valid = not collect and actual == refs[name]
                rows.append(
                    {"profile": name, **profile, "seconds": elapsed, "quality_pass": int(valid), "answers": answers}
                )
                if collect:
                    refs[name] = actual
                (output / "scenario-results.json").write_text(json.dumps(rows, indent=2))
        if collect:
            (output / "calibrated-references.json").write_text(json.dumps(refs, indent=2))
            metrics.update(quick_check=1, reference_collection=1)
        else:
            valid = all(r["quality_pass"] for r in rows)
            metrics.update(
                eval_valid=1,
                quality_pass=int(valid),
                reward=sum(1 / (1 + r["seconds"]) for r in rows) / len(rows) if valid else 0,
            )
    except Exception as exc:
        (output / "error.json").write_text(json.dumps({"error": repr(exc)}))
    finally:
        if service is not None:
            stop_service(service)
        (output / "reward.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
