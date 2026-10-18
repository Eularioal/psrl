"""Check score semantics and data boundaries without running untrusted host code."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from examples.airs_bench.apps import judge
from examples.airs_bench.apps.comparison import matches
from psrl.workers.env_worker.isolated_code import OutputLimitError, candidate_spec, read_bounded
from psrl.workers.env_worker.sandbox import CodeExecutionSpec, SandboxSpec, build_docker_run_argv

pytestmark = pytest.mark.cpu_test


def test_judge_counts_fifth_candidate_and_never_transmits_expected_answers(monkeypatch):
    requests = []
    current = None

    def request(message):
        nonlocal current
        requests.append(message)
        if message["op"] == "begin":
            current = message["code"]
        if message["op"] == "case":
            return {"ok": True, "stdout": "secret-answer" if current == "correct" else '{"Pass@5":1}'}
        return {"ok": True}

    monkeypatch.setattr(judge, "request", request)
    sample = {"input_output": json.dumps({"inputs": ["public-input"], "outputs": ["secret-answer"]})}
    score = judge.grade([["wrong"] * 4 + ["correct"], ["wrong"] * 5], [sample, sample])
    assert score == {"Pass@5": 0.5}
    assert "secret-answer" not in json.dumps(requests)
    assert sum(r["op"] == "begin" for r in requests) == 10
    assert sum(r["op"] == "end" for r in requests) == 10


def test_bad_function_result_does_not_skip_remaining_candidates(monkeypatch):
    operations = []
    current = None

    def request(message):
        nonlocal current
        operations.append(message["op"])
        if message["op"] == "begin":
            current = message["code"]
        return {"ok": True, "stdout": "1" if current == "correct" else '{"not_an_integer": 1}'}

    monkeypatch.setattr(judge, "request", request)
    sample = {"input_output": json.dumps({"fn_name": "solve", "inputs": [[1]], "outputs": [1]})}
    score = judge.grade([["wrong", "correct", "wrong", "wrong", "wrong"]], [sample])
    assert score == {"Pass@5": 1}, "A valid later candidate must still count."
    assert operations == ["begin", "case", "end"] * 2, "Both candidates must be cleaned up."


@pytest.mark.parametrize("cases", ["not-json", "{}", '{"inputs": [1], "outputs": [1]}'])
def test_private_fixture_failures_are_not_submission_errors(cases):
    with pytest.raises(RuntimeError, match="Private APPS cases"):
        judge.grade([["pass"] * 5], [{"input_output": cases}])


def test_private_protocol_limit_is_not_a_submission_error(monkeypatch):
    monkeypatch.setattr(judge, "MAX_WIRE_BYTES", 32)
    with pytest.raises(RuntimeError, match="protocol limit"):
        judge.request({"op": "case", "input": "x" * 32})


def test_judge_does_not_relabel_unexpected_value_errors(monkeypatch, tmp_path, capsys):
    import builtins

    import datasets

    submission = tmp_path / "submission.csv"
    submission.write_text("code1,code2,code3,code4,code5\npass,pass,pass,pass,pass\n")
    original_open = builtins.open
    monkeypatch.setattr(judge, "open", lambda *args, **kwargs: original_open(submission, **kwargs), raising=False)
    monkeypatch.setattr(datasets, "load_from_disk", lambda path: [None])

    def broken_grade(*args):
        raise ValueError("Internal grader failure.")

    monkeypatch.setattr(judge, "grade", broken_grade)
    with pytest.raises(ValueError, match="Internal grader failure"):
        judge.main()
    assert "invalid_submission" not in capsys.readouterr().out, "Internal errors must discard the episode."


def test_candidate_spec_cannot_inherit_private_mounts_or_proxy_environment():
    config = CodeExecutionSpec("/python", "/opt/psrl-code/candidate.py", "/public/runner")
    verifier = SandboxSpec(
        image="fixture",
        mounts=(("/private", "/answers", "ro"),),
        env={"HTTPS_PROXY": "secret"},
        code_execution=config,
    )
    spec = candidate_spec(verifier)
    argv = build_docker_run_argv(spec, "fixture")
    assert spec.mounts == (("/public/runner", "/opt/psrl-code", "ro"),)
    assert "HTTPS_PROXY" not in spec.env and spec.network == "none"
    assert spec.user == "agent" and spec.read_only and spec.pids_limit == 64
    assert "--cap-drop=ALL" in argv and "--security-opt=no-new-privileges" in argv
    assert "--read-only" in argv and "--pids-limit" in argv and "--tmpfs" in argv


@pytest.mark.parametrize(
    ("actual", "expected", "function", "match"),
    [
        ("5\n", "5", False, True),
        ("wrong", "5", False, False),
        ("1.000001\n", "1.0", False, True),
        ("a b\nc d", "d c\nb a", False, True),
        ([1, 2], [[1, 2]], True, True),
        (3, 4, True, False),
        ({"not_an_integer": 1}, 1, True, False),
        ({"name": 1}, {"name": 1}, True, True),
        ({"1": 2}, {"1": 2}, True, True),
    ],
)
def test_apps_comparison_contract(actual, expected, function, match):
    assert matches(actual, expected, function) == match


@pytest.mark.parametrize(
    ("source", "inputs", "function", "expected"),
    [
        ("a,b=map(int,input().split());print(a+b)", "2 3\n", None, "5"),
        ("a,b=map(int,open(0).read().split());print(a+b)", "2 3\n", None, "5"),
        ("import sys; a,b=map(int,sys.stdin.buffer.read().split());print(a+b)", "2 3\n", None, "5"),
        ("class Solution:\n def add(self,a,b): return a+b", [2, 3], "add", "5"),
        ("def negate(a): return [-x for x in a]", [[1, 2]], "negate", "[-1, -2]"),
        ("def lookup(d): return d['name']", [{"name": 7}], "lookup", "7"),
        ("def lookup(d): return d[1]", [{"1": 7}], "lookup", "7"),
        ("def add(a,b): return np.int64(a+b)", [2, 3], "add", "5"),
        ("def negate(a): return -np.array(a)", [[1, 2]], "negate", "[-1, -2]"),
    ],
)
def test_candidate_adapter_with_fixed_trusted_programs(source, inputs, function, expected):
    path = Path(judge.__file__).with_name("candidate.py")
    result = subprocess.run(
        [sys.executable, "-I", str(path)],
        input=json.dumps({"code": source, "fn_name": function}) + "\n" + json.dumps({"input": inputs}) + "\n",
        text=True,
        capture_output=True,
        timeout=3,
    )
    assert result.returncode == 0
    ready, response = map(json.loads, result.stdout.splitlines())
    assert ready == {"ready": True}
    assert response["ok"] and response["stdout"].strip() == expected


async def test_output_reader_enforces_a_byte_limit():
    import asyncio

    stream = asyncio.StreamReader()
    stream.feed_data(b"x" * 20)
    stream.feed_eof()
    with pytest.raises(OutputLimitError):
        await read_bounded(stream, 10)


async def test_process_cleanup_drains_a_paused_pipe_and_reaps_the_child():
    import asyncio

    from psrl.workers.env_worker.isolated_code import stop_process

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys,time; sys.stdout.buffer.write(b'x'*3000000); time.sleep(30)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1024,
    )
    try:
        with pytest.raises(OutputLimitError):
            await read_bounded(process.stdout, 100)
    finally:
        await asyncio.wait_for(stop_process(process), 3)
    assert process.returncode is not None
