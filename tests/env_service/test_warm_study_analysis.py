"""Check matrix evidence acceptance with synthetic records and no GPU dependencies."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "docs/env-worker-deep-dive/env-service-verification/scripts/warm_router_study"
)
A, B = "Qwen3-4B-Base", "Qwen3-1.7B-Base"
CASES = {"default": (2, 128), "short": (1024, 128), "boundary": (1024, 512), "long": (1024, 2048)}
PRIVATE_VALUES = ("203.0.113.9", "/private/fixture-origin", "00000000-0000-0000-0000-000000000001")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def command(model, steps):
    return dict(
        status="passed",
        model=model,
        steps=steps,
        optimizer_updates=steps,
        data_sha256=digest("synthetic public inputs"),
        norm_sha256=digest(f"norm:{model}:{steps}"),
        losses_sha256=digest(f"losses:{model}:{steps}"),
        load_s=4,
        compute_s=10,
        total_s=14,
    )


def registered_cells():
    cells = []
    for backend in ("baseline", "native"):
        for repeat in range(1, 4):
            for case, (target_steps, blocker_steps) in CASES.items():
                strategies = ("affinity",) if backend == "baseline" else ("affinity", "always_copy", "predictive")
                for strategy in strategies:
                    cells.append(
                        dict(
                            cell_id=f"{backend}-r{repeat}-{case}-{strategy}",
                            backend=backend,
                            round=repeat,
                            phase="core",
                            case_id=case,
                            strategy=strategy,
                            target_steps=target_steps,
                            blocker_steps=blocker_steps,
                            forecast_scale=1,
                            report_age_offset_s=0,
                            cold_budget_mb=20480,
                        )
                    )
    for repeat in range(1, 4):
        for case, reference in (("overpredict_short", "short"), ("stale_short", "short"), ("memory_long", "long")):
            cell = copy.deepcopy(
                next(value for value in cells if value["cell_id"] == f"native-r{repeat}-{reference}-predictive")
            )
            cell.update(
                cell_id=f"native-r{repeat}-{case}-predictive", case_id=case, reference_case=reference, phase="fault"
            )
            cell.update(
                forecast_scale=10 if case == "overpredict_short" else 1,
                report_age_offset_s=10 if case == "stale_short" else 0,
                cold_budget_mb=6144 if case == "memory_long" else 20480,
            )
            cells.append(cell)
    return cells


def measured_block(cell):
    available, required = cell["cold_budget_mb"] * 1024**2, 9 * 1024**3
    hot = (cell["round"] - 1) % 2
    forecast = dict(status="estimated", estimate_s=4, lower_s=3, upper_s=6)
    return dict(
        **cell,
        status="passed",
        started_unix_s=1,
        finished_unix_s=2,
        hot_device=hot,
        cold_device=1 - hot,
        selected_device="hot",
        target=command(A, cell["target_steps"]),
        blocker=command(B, cell["blocker_steps"]),
        primary_makespan_s=60 if cell["strategy"] == "predictive" else 100,
        closed_cycle_s=61 if cell["strategy"] == "predictive" else 101,
        target_latency_s=50,
        blocker_remaining_actual_s=5,
        progress_forecast=forecast,
        effective_forecast=forecast,
        effective_active=None
        if cell["case_id"] == "stale_short"
        else dict(point_s=4 * cell["forecast_scale"], interval=dict(lower_s=3, upper_s=60)),
        server_admission=dict(required_bytes=required, available_bytes=available, admits=available >= required),
        service_states=[dict(cache_keys=[A, B]), dict(cache_keys=[], budget_bytes=available)],
        decision=dict(
            action="wait",
            device_id="hot",
            strategy=cell["strategy"],
            mode="point",
            switch_margin_s=2,
            reason_code="affinity_retained",
            estimates=[
                dict(
                    device_id="cold",
                    capacity_fits=available >= required,
                    required_gpu_bytes=required,
                    available_gpu_bytes=available,
                )
            ],
        ),
        private_origin=PRIVATE_VALUES,
    )


def inspection(periods):
    return dict(
        status="ok",
        containers=[
            dict(
                id=PRIVATE_VALUES[2],
                host_config=dict(IpcMode="private", CpusetCpus="", CpusetMems=""),
                cpu_stat=dict(
                    status="ok",
                    source="cgroup_v1_cpu",
                    observed_unix_s=periods,
                    values=dict(nr_periods=periods, nr_throttled=0, throttled_time=0),
                ),
            )
        ],
    )


@pytest.fixture
def analyzer(monkeypatch):
    modules = {}
    for name in ("analyze", "analyze_long", "analyze_matrix"):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["analyze_matrix"]


@pytest.fixture
def study(tmp_path, analyzer):
    policies, sources = {}, {}
    for folder, names, hashes in (
        ("policy", analyzer.MODULES, policies),
        ("source-snapshot", analyzer.CORE_SOURCES, sources),
    ):
        for name in names:
            content = f"Synthetic frozen source for {name}.\n"
            destination = tmp_path / folder / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)
            hashes[name] = digest(content)
    patch = "Synthetic worktree patch.\n"
    (tmp_path / "current-worktree.patch").write_text(patch)
    protocol = dict(profile_steps=128, hot_device_by_round=[0, 1, 0], mode="point", switch_margin_s=2)
    manifest = dict(
        cells=registered_cells(),
        runs=[dict(backend=name, path=f"{name}/status.json") for name in ("baseline", "native")],
        protocol=protocol,
        image_id="sha256:" + digest("synthetic image"),
        policy_hashes=policies,
        source_hashes=sources,
        current_worktree_patch_sha256=digest(patch),
        status="passed",
    )
    runs = {}
    for name in ("baseline", "native"):
        profiles = [
            dict(**command(model, 128), round=repeat, device=device)
            for repeat in range(1, 4)
            for device, model in ((0, A), (1, A), ((repeat - 1) % 2, B))
        ]
        blocks = [measured_block(cell) for cell in manifest["cells"] if cell["backend"] == name]
        runs[name] = dict(
            status="passed",
            finished=100,
            git_commit=analyzer.BASELINE if name == "baseline" else "a" * 40,
            image_id=manifest["image_id"],
            protocol=protocol,
            source_hashes=sources,
            policy_hashes=policies,
            profiles=profiles,
            blocks=blocks,
            requests=[dict(status="passed", result=value) for value in profiles]
            + [dict(status="passed", result=block[role]) for block in blocks for role in ("target", "blocker")],
            cleanup_errors=[],
            containers_remaining=[],
            gpu_after={"0": dict(used_mb=0, pids=[]), "1": dict(used_mb=0, pids=[])},
            ledger_after_cleanup=[dict(leases_active=0)],
            container_inspection_before=inspection(1),
            container_inspection_after=inspection(11),
            error_private=PRIVATE_VALUES,
        )
        write_json(tmp_path / name / "status.json", runs[name])
    write_json(tmp_path / "manifest.json", manifest)
    return SimpleNamespace(root=tmp_path, analyzer=analyzer, manifest=manifest, runs=runs)


def analyze(study):
    for name, run in study.runs.items():
        write_json(study.root / name / "status.json", run)
    return study.analyzer.analyze(study.root)


def test_complete_plan_preserves_updates_pairing_and_forecast_faults(study):
    result = analyze(study)
    assert result["audit"]["status"] == "passed"
    assert result["planned_cells"] == result["observed_complete_cells"] == result["audited_cells"] == 57
    assert result["work_counts"]["request_derived_updates"] == 89112
    summary = result["mixed_comparisons"]["all_core_equal_case_counts_predictive_vs_affinity"]["summaries"]
    assert summary["primary_makespan_s"]["aggregate_improvement_pct"] == pytest.approx(40)
    assert (
        result["prediction_audit"]["fault_overpredict_short"]["estimates"]["effective"]["signed_error_s"]["mean"] == 35
    )
    assert result["prediction_audit"]["fault_stale_short"]["estimates"]["effective"]["unknown"] == 3


@pytest.mark.parametrize("leave_partial_numbers", [False, True])
def test_budget_exit_keeps_complete_core_and_excludes_unfinished_fault(study, leave_partial_numbers):
    native = study.runs["native"]
    native.update(status="failed", error="The next registered cell does not fit the remaining work budget.")
    native.pop("container_inspection_after")
    if leave_partial_numbers:
        native["blocks"][-1].update(status="running", target=None)
    else:
        native["blocks"].pop()
    result = analyze(study)
    assert result["audit"]["status"] == "incomplete"
    assert result["planned_cells"] == 57 and result["audited_cells"] == 56
    assert result["audit"]["checks"]["all_complete_pairs_output_and_runtime_checks_passed"]
    assert result["runs"][1]["cleanup"]["passed"]
    assert not result["runs"][1]["container_cpu_inspection"]["complete"]
    for key, group in result["comparisons"].items():
        if key.startswith("core_"):
            assert all(pair["passed"] for pair in group["pairs"])
            assert group["summaries"]["primary_makespan_s"]["audited_pairs"] == 3
    fault = result["comparisons"]["fault_memory_long_vs_affinity"]
    assert fault["summaries"]["primary_makespan_s"]["measured_pairs"] == 2
    assert fault["pairs"][-1]["metrics"] == {}
    assert result["prediction_audit"]["fault_memory_long"]["actual_residuals"] == 2


def test_running_backend_is_not_certified_before_final_cleanup(study):
    study.runs["native"].pop("finished")
    result = analyze(study)
    assert result["audit"]["status"] == "incomplete"
    assert result["observed_complete_cells"] == 57
    assert result["audited_cells"] == 12
    assert not any(cell["audit"]["passed"] for cell in result["cells"] if cell["backend"] == "native")


@pytest.mark.parametrize("corruption", ["policy", "output", "duplicate"])
def test_frozen_source_output_and_duplicate_corruption_are_rejected(study, corruption):
    if corruption == "policy":
        (study.root / "policy/warm_dispatch.py").write_text("Unregistered source.\n")
    elif corruption == "output":
        study.runs["native"]["blocks"][0]["target"]["norm_sha256"] = "f" * 64
    else:
        study.runs["native"]["blocks"].append(copy.deepcopy(study.runs["native"]["blocks"][0]))
    result = analyze(study)
    assert result["audit"]["status"] != "passed"
    assert result["planned_cells"] == 57
    if corruption == "policy":
        assert not result["policy_audit"]["passed"]
    elif corruption == "output":
        assert not result["audit"]["checks"]["all_paired_output_and_runtime_checks_passed"]
    else:
        cell = next(value for value in result["cells"] if value["cell_id"] == "native-r1-default-affinity")
        assert not cell["audit"]["passed"] and len(cell["observations"]) == 2


def test_public_export_drops_private_paths_addresses_and_container_ids(study):
    result = analyze(study)
    assert result["auxiliary_cpu_observation"]["status"] == "missing"
    rendered = json.dumps(result)
    for value in PRIVATE_VALUES:
        assert value not in rendered
    assert str(study.root) not in rendered


@pytest.mark.parametrize("reported,verified", [(36, True), (35, False)])
def test_auxiliary_cpu_snapshot_preserves_its_scope_and_checks_completed_core(study, reported, verified):
    study.runs["native"].pop("container_inspection_after")
    snapshot = inspection(11)
    snapshot.update(observed_unix_s=11, completed_core_cells=reported, scope=PRIVATE_VALUES[1], tag=PRIVATE_VALUES[0])
    write_json(study.root / "native-cpu-near-end.json", snapshot)
    result = analyze(study)
    auxiliary = result["auxiliary_cpu_observation"]
    assert auxiliary["completed_core_cells_verified"] is verified
    assert auxiliary["completed_core_cells_before_snapshot"] == 36
    assert auxiliary["status"] == ("ok" if verified else "partial")
    assert auxiliary["raw_sha256"] == digest(json.dumps(snapshot))
    assert auxiliary["scope"] == "after all native core, during fault checks, before cleanup"
    assert not auxiliary["is_final_inspection"] and not auxiliary["affects_cell_audit"]
    counter = auxiliary["cpu_observation"]["containers"][0]
    assert counter["cpu_delta"]["nr_periods"] == 10
    assert counter["cpu_delta"]["throttled_time"] == 0
    assert counter["counter_units"]["throttled_time"] == "nanoseconds"
    assert not result["runs"][1]["container_cpu_inspection"]["complete"]
    assert result["audited_cells"] == 57
    for value in PRIVATE_VALUES:
        assert value not in json.dumps(auxiliary)
