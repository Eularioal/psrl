"""Prepare all non-Kaggle AIRS task identities with bounded real data."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_from_disk
from examples.airs_bench.prepare.build_parquet import build_row
from examples.airs_bench.prepare.prepare_airs_data import load_task_metadata

KAGGLE_TASK = "TimeSeriesForecastingKaggleWebTrafficMASE"
TARGETS = {
    "CoreferenceResolutionSuperGLUEWSCAccuracy": "label",
    "CoreferenceResolutionWinograndeAccuracy": "answer",
    "CvMolecularPropertyPredictionQm9MeanAbsoluteError": "c_v",
    "GMolecularPropertyPredictionQm9MeanAbsoluteError": "G",
    "GraphRegressionZincMae": "y",
    "MathQuestionAnsweringSVAMPAccuracy": "Answer",
    "QuestionAnsweringFinqaAccuracy": "answer",
    "R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError": "R_2_Abs",
    "SentimentAnalysisYelpReviewFullAccuracy": "label",
    "TextualClassificationSickAccuracy": "label",
    "TextualSimilaritySickSpearmanCorrelation": "relatedness_score",
    "U0MolecularPropertyPredictionQm9MeanAbsoluteError": "U_0",
}


def baseline_spec(task, train, test, corpus=None):
    """Construct baseline inputs exclusively from named public splits."""
    columns = ["prediction"]
    if task == "CodeRetrievalCodeXGlueMRR":
        return {
            "kind": "retrieval",
            "columns": ["query", "rankings"],
            "train_x": [{"query": row["code"]} for row in corpus],
            "train_y": list(corpus["id"]),
            "test_x": list(test),
        }
    keys = test.column_names
    train_x = [{k: row[k] for k in keys} for row in train]
    test_x = list(test)
    if task == "CodeGenerationAPPSPassAt5":
        outputs = [json.loads(row["solutions"] or "[]") or ["print(0)"] for row in train]
        kind, columns = "code", [f"code{i}" for i in range(1, 6)]
    elif task in ("ReadingComprehensionSquadExactMatch", "QuestionAnsweringEli5RougeL"):
        outputs = [(row["answers"]["text"] or ["unknown"])[0] for row in train]
        kind = "text"
    elif task == "QuestionAnsweringDuoRCAccuracy":
        outputs = [(row["answers"] or ["unknown"])[0] for row in train]
        kind, columns = "duorc", ["answer", "has_answer"]
    elif task.startswith("TimeSeries"):
        horizon = 48 if "Rideshare" in task else 5
        train_x, test_x, outputs = [], [], []
        for row in train:
            histories = row["target"] if "Rideshare" in task else [row["target"]]
            futures = row["label_target"] if "Rideshare" in task else [row["label_target"]]
            for history, future in zip(histories, futures, strict=True):
                target = future[len(history) : len(history) + horizon]
                if len(target) == horizon and all(pd.notna(v) for v in target):
                    train_x.append({"target": history[-48:]})
                    outputs.append(target)
        for row in test:
            histories = row["target"] if "Rideshare" in task else [row["target"]]
            test_x.extend({"target": v[-48:]} for v in histories)
        if len(outputs) < 2:
            raise ValueError(f"Insufficient finite forecast targets for {task!r}.")
        kind = "regression"
    else:
        outputs = list(train[TARGETS[task]])
        classification = "Coreference" in task or "Classification" in task or "Sentiment" in task
        kind = "classification" if classification else "regression"
        if task in ("MathQuestionAnsweringSVAMPAccuracy", "QuestionAnsweringFinqaAccuracy"):
            kind = "text"
    return {"kind": kind, "columns": columns, "train_x": train_x, "train_y": outputs, "test_x": test_x}


def save(dataset, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))


def prepare(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    configs = out / "airs_configs"
    shutil.copytree(args.data_root / "airs_configs/datasets", configs / "datasets")
    (configs / "tasks").mkdir()
    (configs / "acceptance").mkdir()
    upstream = args.repo / "airsbench/tasks/rad"
    tasks = sorted(p.name for p in upstream.iterdir() if (p / "metadata.yaml").exists())
    if len(tasks) != 20:
        raise ValueError("Review the task inventory before changing the pinned twenty-task scope.")
    rows, records = [], []
    for task in tasks:
        if task == KAGGLE_TASK:
            continue
        source = args.data_root / "airs_prepared" / task
        dest = out / "airs_prepared" / task
        labels = load_from_disk(str(source / "test_with_labels"))
        ntest = 2 if task == "CodeGenerationAPPSPassAt5" else args.test_rows
        labels = labels.select(range(min(ntest, len(labels))))
        public_path = source / ("test/queries" if task == "CodeRetrievalCodeXGlueMRR" else "test")
        public_columns = load_from_disk(str(public_path)).column_names
        test = labels.select_columns(public_columns)
        if task == "CodeGenerationAPPSPassAt5":

            def bound_cases(row):
                cases = json.loads(row["input_output"])
                for key in ("inputs", "outputs"):
                    cases[key] = cases[key][:2]
                return {"input_output": json.dumps(cases)}

            labels = labels.map(bound_cases)
        corpus = None
        if task == "CodeRetrievalCodeXGlueMRR":
            for split in ("train", "validation", "test"):
                original = load_from_disk(str(source / split / "search_corpus"))
                # Keep the entire test corpus so corpus selection does not reveal labels.
                corpus_split = (
                    original if split == "test" else original.select(range(min(args.train_rows, len(original))))
                )
                save(corpus_split, dest / split / "search_corpus")
                if split != "test":
                    queries = load_from_disk(str(source / split / "queries_with_labels"))
                    ids = set(corpus_split["id"])
                    queries = queries.select([i for i, value in enumerate(queries["id"]) if value in ids])
                    save(queries, dest / split / "queries_with_labels")
            corpus = load_from_disk(str(dest / "test/search_corpus"))
            train = load_from_disk(str(dest / "train/queries_with_labels"))
            save(test, dest / "test/queries")
        else:
            train = load_from_disk(str(source / "train"))
            train = train.select(range(min(args.train_rows, len(train))))
            save(train, dest / "train")
            if (source / "validation").is_dir():
                validation = load_from_disk(str(source / "validation"))
                save(validation.select(range(min(args.test_rows, len(validation)))), dest / "validation")
            save(test, dest / "test")
        save(labels, dest / "test_with_labels")
        spec = baseline_spec(task, train, test, corpus)
        (dest / "train/psrl-baseline.json").write_text(json.dumps(spec, allow_nan=True))
        task_assets = configs / "data" / task
        shutil.copytree(args.data_root / "airs_configs/data" / task, task_assets)
        shutil.copy2(Path(__file__).parents[1] / "gpu_baseline.py", task_assets / "gpu_baseline.py")
        config = yaml.safe_load((args.data_root / "airs_configs/tasks" / f"{task}.yaml").read_text())
        (task_assets / "original-description.md").write_text(config["description"])
        config.update(
            description=(
                f"Solve AIRS task {task} on the prepared bounded real data. "
                f"This local protocol has {len(test)} held-out inputs, not the original full split. "
                f"The submission must contain {len(spec['test_x'])} rows. "
                "Develop a solution to improve the task metric, save submission.csv, then run `submit`. "
                "A runnable CUDA starting baseline is `python gpu_baseline.py`. "
                "The baseline fits public training data or retrieves public examples on CUDA. "
                "The independent verifier alone sees held-out answers. Public baseline details are in "
                "data/train/psrl-baseline.json and original task instructions are in original-description.md."
            ),
            starter_code=[f"data/{task}/gpu_baseline.py", f"data/{task}/original-description.md"],
            dataset_configs=[],
            training_timeout=120,
            eval_timeout=120,
            use_generic_conda=True,
            requirements_path=None,
            memory_path=None,
        )
        (configs / "tasks" / f"{task}.yaml").write_text(yaml.safe_dump(config))
        acceptance = dict(config)
        acceptance["description"] = (
            f"This is a bounded PSRL integration acceptance for {task}. "
            "Run exactly `python gpu_baseline.py`, wait for completion, then run `submit`. "
            "Do not explore or optimize. The public starting solution performs real CUDA work, "
            "and the independent verifier computes the real metric on hidden local test data."
        )
        (configs / "acceptance" / f"{task}.yaml").write_text(yaml.safe_dump(acceptance))
        metadata = load_task_metadata(upstream / task)
        row = build_row(task, metadata, f"tasks/{task}.yaml", str(dest))
        row["data_source"] = "airs_bench_gpu_local"
        row["extra_info"].update(local_evaluation=True, gpus=1, protocol="bounded-real-data-v1")
        rows.append(row)
        records.append(
            {
                "task": task,
                "train_rows": len(train),
                "test_rows": len(test),
                "submission_rows": len(spec["test_x"]),
                "search_corpus_rows": len(corpus) if corpus is not None else 0,
                "kind": spec["kind"],
                "metric": metadata["metric"],
                "public_columns": public_columns,
                "evaluator_sha256": hashlib.sha256((task_assets / "evaluate.py").read_bytes()).hexdigest(),
            }
        )
    pd.DataFrame(rows).to_parquet(out / "tasks.parquet", index=False)
    split = json.loads(Path(__file__).with_name("split.json").read_text())
    for name in ("train", "val"):
        selected = [row for row in rows if row["extra_info"]["airs_task_id"] in split[name]]
        pd.DataFrame(selected).to_parquet(out / f"{name}.parquet", index=False)
    revision = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    (out / "manifest.json").write_text(
        json.dumps(
            {"revision": revision, "local_evaluation": True, "tasks": records, "deferred": [KAGGLE_TASK]}, indent=2
        )
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "data-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=64)
    parser.add_argument("--test-rows", type=int, default=8)
    args = parser.parse_args()
    if args.train_rows < 5 or args.test_rows < 2:
        parser.error("At least five training rows and two test rows are required.")
    print(prepare(args))


if __name__ == "__main__":
    main()
