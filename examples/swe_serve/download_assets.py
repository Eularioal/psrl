"""Download pinned upstream assets with a bounded process pool and durable results."""

import argparse
import json
import multiprocessing
import time
from pathlib import Path


def fetch(name, spec, cache, result, endpoint):
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import GatedRepoError

    record = {"asset": name, "repository": spec["repository"]}
    try:
        revision = spec.get("revision")
        if not revision or revision == "main":
            revision = (
                HfApi(endpoint=endpoint).repo_info(spec["repository"], repo_type=spec["repo_type"], timeout=20).sha
            )
        path = snapshot_download(
            spec["repository"],
            repo_type=spec["repo_type"],
            revision=revision,
            cache_dir=cache / "hub",
            allow_patterns=spec.get("allow_patterns"),
            ignore_patterns=spec.get("ignore_patterns"),
            max_workers=4,
            endpoint=endpoint,
        )
        record.update(status="downloaded", revision=revision, path=path)
    except GatedRepoError:
        record["status"] = "credentials_required"
    except Exception as error:
        record.update(status="failed", error=type(error).__name__, message=str(error)[:400])
    result.write_text(json.dumps(record, indent=2))


def download(repo, cache, output, assets=None, budget=1800, workers=2, endpoint="https://huggingface.co"):
    repo, cache, output = Path(repo), Path(cache).resolve(), Path(output).resolve()
    spec = json.loads((repo / "scripts/configs/hf_cache_requirements.json").read_text())["assets"]
    names = list(spec) if assets is None else list(dict.fromkeys(assets))
    if set(names) - set(spec) or not 1 <= workers <= 4 or not 1 <= budget <= 3600:
        raise ValueError("Select known assets, one to four workers, and a budget of at most 3600 seconds.")
    output.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    active, records = {}, {}
    deadline = time.monotonic() + budget
    pending = names.copy()
    try:
        while (pending or active) and time.monotonic() < deadline:
            while pending and len(active) < workers:
                name = pending.pop(0)
                process = context.Process(
                    target=fetch, args=(name, spec[name], cache, output / f"{name}.json", endpoint)
                )
                process.start()
                active[name] = process
            for name, process in list(active.items()):
                if process.is_alive():
                    continue
                process.join()
                path = output / f"{name}.json"
                records[name] = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {"asset": name, "status": "failed", "exit": process.exitcode}
                )
                del active[name]
            (output / "manifest.json").write_text(json.dumps(list(records.values()), indent=2))
            time.sleep(1)
    finally:
        for name, process in active.items():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            records[name] = {"asset": name, "status": "budget_exhausted"}
        for name in pending:
            records[name] = {"asset": name, "status": "not_started"}
        (output / "manifest.json").write_text(json.dumps(list(records.values()), indent=2))
    return list(records.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "cache", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--assets", nargs="+")
    parser.add_argument("--budget", type=int, default=1800)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    download(**vars(parser.parse_args()))
