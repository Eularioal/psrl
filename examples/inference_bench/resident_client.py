"""Send bounded real requests to an unchanged InferenceBench server."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", action="store_true")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.ready:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            pid = int(Path("/tmp/resident.pid").read_text())
            log = Path("/logs/artifacts/server.log").read_text()[-8192:]
            process = Path(f"/proc/{pid}/stat")
            if (
                "CUDA out of memory" in log
                or not process.exists()
                or process.read_text().rsplit(")", 1)[1].split()[0] == "Z"
            ):
                raise RuntimeError(f"Resident server failed during startup: {log}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8001/v1/models", timeout=3) as response:
                    print(response.read().decode(), flush=True)
                    return
            except (OSError, urllib.error.URLError):
                time.sleep(1)
        raise TimeoutError("Resident server did not become ready within 240 seconds.")
    sample = json.loads(Path("/opt/resident/samples.jsonl").read_text().splitlines()[args.index])
    payload = {
        "model": "/opt/model",
        "messages": sample["messages"],
        "temperature": 0,
        "max_tokens": 16,
        "stream": False,
    }
    started = time.time()
    request = urllib.request.Request(
        "http://127.0.0.1:8001/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.load(response)
    content = body["choices"][0]["message"]["content"]
    assert content.strip(), "Inference must produce a nonempty response."
    record = {
        "sample_id": sample["sample_id"],
        "started_at": started,
        "finished_at": time.time(),
        "response": body,
        "server_pid": int(Path("/tmp/resident.pid").read_text()),
    }
    Path(f"/logs/artifacts/request-{args.index}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
