"""Freeze official assets and a bounded GPU feedback experiment protocol."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    names = ["rmsnorm_h2048", "rmsnorm_h4096", "fused_add_rmsnorm_h2048", "gemm_n128_k2048"]
    assets = root / "subset"
    for folder in ["definitions", "workloads", "solutions"]:
        (assets / folder).mkdir(parents=True)
    inventory = []
    for name in names:
        definition = next((args.trace / "definitions").rglob(name + ".json"))
        workload = next((args.trace / "workloads").rglob(name + ".jsonl"))
        rows = [json.loads(s) for s in workload.read_text().splitlines() if s]
        rows = [r for r in rows if all(v["type"] in ("random", "scalar") for v in r["workload"]["inputs"].values())]
        rows = [r for r in rows if max(r["workload"]["axes"].values()) <= 16384]
        rows.sort(key=lambda r: tuple(r["workload"]["axes"].values()))
        chosen = [rows[i] for i in sorted({0, len(rows) // 2, len(rows) - 1})]
        shutil.copy2(definition, assets / "definitions" / definition.name)
        (assets / "workloads" / workload.name).write_text("".join(json.dumps(r) + "\n" for r in chosen))
        sources = [p for p in (args.trace / "solutions").rglob("*.json") if p.parent.name == name]
        selected = [next(p for p in sources if "baseline" in p.parts)]
        for prefix in ["gemini", "gpt-o3", "gpt-5"]:
            selected.append(next(p for p in sorted(sources) if p.stem.startswith(prefix) and "triton" in p.stem))
        for p in selected:
            shutil.copy2(p, assets / "solutions" / (name + "__" + p.name))
        inventory.append(
            {
                "definition": name,
                "workloads": [r["workload"] for r in chosen],
                "solutions": [json.loads(p.read_text())["name"] for p in selected],
            }
        )
    shutil.copytree(
        args.upstream / "flashinfer_bench",
        root / "upstream/flashinfer_bench",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    script_dir = Path(__file__).parent
    for p in script_dir.glob("*.py"):
        shutil.copy2(p, root / p.name)
    shutil.copy2(script_dir.parent / "flashinfer_env_probe.py", root / "flashinfer_env_probe.py")
    # Freeze the harness with this experiment's declared memory budget.
    harness = root / "flashinfer_env_probe.py"
    harness.write_text(
        harness.read_text().replace("EnvDemand(4, 16384, True, 8192)", "EnvDemand(4, 16384, True, 49152)")
    )
    protocol = {
        "harness_patch": "Declared GPU memory 8192 -> 49152 MiB. Production scheduling unchanged.",
        "scope": (
            "Research controller inside one real Env command. No new production Env policy or historical OFF run."
        ),
        "inventory": inventory,
        "policies": ["shared", "exclusive", "per_call", "group5", "group2", "adaptive"],
        "pressures": {
            "none": [],
            "memory4": ["memory"] * 4,
            "compute4": ["compute"] * 4,
            "mixed8": ["memory", "compute"] * 4,
        },
        "group_sizes": [4, 12],
        "repeats": 3,
        "fixed_rounds": 5,
        "reference_rounds": 7,
        "adaptive": {"min_rounds": 2, "max_rounds": 5, "relative_gap": 0.05, "mad_multiplier": 4},
        "background": {
            "units": 1200,
            "operations_per_unit": 8,
            "matrix_size": 2048,
            "vector_elements": 33554432,
            "synthetic_stress": True,
        },
        "limits": {
            "command_s": 2400,
            "env_s": 2700,
            "supervisor_s": 3000,
            "gpu_memory_mb": 49152,
            "max_gpus": 1,
            "portability_exclusions": "Record failures before comparisons, require at least two candidates per case.",
        },
        "timing": (
            "Official DefaultEvaluator and default config, CUDA events fallback if "
            "CUPTI unavailable. References before and after."
        ),
        "decision": (
            "Select fastest correct candidate. Adaptive early stopping is a heuristic, not a statistical guarantee."
        ),
        "ordering": (
            "Seeded shuffle of scenario blocks and policy order. Rotate candidate order per measurement round."
        ),
        "psrl_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "upstream_head": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        "trace_head": subprocess.check_output(["git", "-C", str(args.trace), "rev-parse", "HEAD"], text=True).strip(),
    }
    if args.profile == "sustained":
        protocol["policies"] = ["shared", "exclusive", "group5", "group2", "adaptive"]
        protocol["pressures"] = {"mixed8": ["memory", "compute"] * 4}
        protocol["group_sizes"] = [4]
        protocol["background"]["units"] = 6000
        protocol["limits"].update(command_s=1800, env_s=2000, supervisor_s=2200)
        script = harness.read_text()
        for old, new in [
            ("num_cpus=4", "num_cpus=16"),
            ('"cpu_quota": 4', '"cpu_quota": 16'),
            ('"memory_mb": 16384', '"memory_mb": 32768'),
            ("EnvDemand(4, 16384, True, 49152)", "EnvDemand(16, 32768, True, 49152)"),
            ("cpus=4, memory_mb=16384", "cpus=16, memory_mb=32768"),
        ]:
            script = script.replace(old, new)
        harness.write_text(script)
        protocol["harness_patch"] = "16 CPU quota, 32768 MiB RAM, 49152 MiB declared GPU memory."
        protocol["scope"] += " Supplemental sustained competition, separate device and increased CPU quota."
    protocol["profile"] = args.profile
    protocol["expected_groups"] = (
        len(protocol["pressures"]) * len(protocol["group_sizes"]) * protocol["repeats"] * len(protocol["policies"])
    )
    protocol["asset_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.rglob("*") if p.is_file()
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(
        json.dumps(
            {
                "root": str(root),
                "cases": sum(len(x["workloads"]) for x in inventory),
                "policy_groups": protocol["expected_groups"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["trace", "upstream", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--profile", choices=["matrix", "sustained"], default="matrix")
    main(parser.parse_args())
