"""Reject unsupported tasks and missing assets before starting training workers."""

from __future__ import annotations

import argparse

import pandas as pd
from examples.airs_bench.config import build_runtime_config
from examples.airs_bench.grading import build_verifier_spec
from psrl.environments.mlgym_env import assert_default_history_processor, build_sandbox_spec


def main() -> None:
    """Validate prepared task paths and the grader contract without starting containers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", nargs="+", required=True)
    args = parser.parse_args()
    runtime = build_runtime_config({"data_root": args.data_root})
    assert_default_history_processor(runtime.agent_config_path)
    for dataset in args.dataset:
        for extra in pd.read_parquet(dataset, columns=["extra_info"])["extra_info"]:
            verifier, _ = build_verifier_spec(runtime, extra["dataset_data_path"], extra["task_config_path"], {})
            build_sandbox_spec(runtime, extra["dataset_data_path"], {}, verifier_spec=verifier)


if __name__ == "__main__":
    main()
