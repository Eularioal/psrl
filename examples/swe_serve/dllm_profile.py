"""Prepare bounded CUDA diffusion-sampler regressions without starting a full model service."""

import argparse
import json
import shutil
from pathlib import Path

import toml
from examples.gpu_benchmarks.prepare import write_dataset
from examples.swe_serve.bounded_tests import prepare as bounded

TASK = "ss-sgl-cluster-dllm-serving-radix-graph"


def prepare(source, output, image):
    source, output = Path(source).resolve(), Path(output).resolve()
    provenance = json.loads((source / "provenance.json").read_text())
    if provenance["task"] != TASK:
        raise ValueError("The sampler profile applies only to the audited diffusion-serving task.")
    nodes = []
    for kind, names in (
        (
            "fail_to_pass",
            (
                "test_low_confidence_multiblock_independent_threshold",
                "test_low_confidence_fallback_progresses_every_active_block",
                "test_joint_threshold_m2t_t2t_preserves_prompt",
                "test_joint_threshold_mask_free_avoids_iterative_refinement",
            ),
        ),
        ("pass_to_pass", ("test_legacy_low_confidence_single_block", "test_low_confidence_mask_free_fast_path")),
    ):
        available = (source / f"tests/{kind}.txt").read_text().splitlines()
        for name in names:
            matches = [line.strip() for line in available if line.strip().endswith("::" + name)]
            if len(matches) != 1:
                raise ValueError(f"Review the changed upstream test identity {name!r}.")
            node = matches[0]
            if not node.startswith("/tests/postmerge_tests/"):
                raise ValueError("The local sampler profile requires immutable postmerge tests.")
            nodes.append({"kind": kind, "node": node, "upstream_node": node})
    bounded(source, output, image, case_limit=2, hardware_profile="h20_1")
    (output / "tests/local-cases.json").write_text(json.dumps(nodes, indent=2))
    shutil.copy2(Path(__file__).with_name("cuda_pytest.py"), output / "tests/cuda_pytest.py")
    launcher = output / "tests/test.sh"
    text = launcher.read_text()
    entry = "python3 -I /tests/bounded_verify.py\n"
    if text.count(entry) != 1:
        raise ValueError("Review the changed bounded verifier entry point.")
    launcher.write_text(text.replace(entry, "python3 -I /tests/bounded_verify.py --cuda-default-device\n"))
    config = toml.loads((output / "task.toml").read_text())
    config["verifier"]["timeout_sec"] = 180
    (output / "task.toml").write_text(toml.dumps(config))
    provenance = json.loads((output / "provenance.json").read_text())
    provenance.update(
        protocol="bounded_cuda_diffusion_sampling",
        selected_cases=nodes,
        cuda_default_device=True,
        gpu_scope="Production samplers with immutable upstream mock-runner logits. Full LLM service gates excluded.",
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    instruction = output / "instruction.md"
    _, separator, remainder = instruction.read_text().partition("\n\n")
    if not separator:
        raise ValueError("Review the changed bounded instruction structure.")
    instruction.write_text(
        "Local CUDA sampler protocol: six immutable production regressions run with default CUDA tensors. "
        "Per-case dispatch telemetry rejects tests with no CUDA tensor operations. "
        "This exercises the real diffusion sampling algorithms using upstream fixed logits. "
        "Full-model quality, serving, prefix reuse and CUDA graph gates remain in official-test.sh.\n\n" + remainder
    )
    write_dataset(output, "swe_serve_regression_local", 8192)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "output", "image"):
        parser.add_argument("--" + key, required=True)
    prepare(**vars(parser.parse_args()))
