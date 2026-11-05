"""Check RE-Bench data isolation, official reward gates, and offline judge transport."""

import argparse
import asyncio
import json
import threading

import pandas as pd
import pytest
import tomllib
from examples.gpu_benchmarks.reward import compute_score
from examples.re_bench import judge, replicate, run
from examples.re_bench.download import split_prompts
from examples.re_bench.prepare import prepare


def test_official_split_excludes_answers():
    texts = [f"question {i}\n### Response:\nSECRET ANSWER" for i in range(514)]
    public, private = split_prompts(texts)
    assert len(private) == 512 and len(public) == 2, "Keep the official held-out boundary."
    assert public[0].startswith("question 512"), "Do not leak held-out prompts into training."
    assert all("SECRET" not in prompt for prompt in public + private), "Expose prompts without ground truth."


@pytest.mark.parametrize("smoke", [False, True])
def test_native_task_and_score_identity(tmp_path, monkeypatch, smoke):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "assets.json").write_text(json.dumps({"heldout_prompts": 512, "model": "gpt2-xl"}))
    monkeypatch.setattr("examples.re_bench.prepare.provenance", lambda _: {"revision": "test"})
    args = argparse.Namespace(
        repo=tmp_path,
        assets=assets,
        output=tmp_path / "task",
        image="public:local",
        verifier_image="private:local",
        timeout=120,
        gpu_memory_mb=49152,
        smoke=smoke,
        local_judge_model=None,
        feedback_smoke=False,
    )
    task = prepare(args)
    config = tomllib.loads((task / "task.toml").read_text())
    assert config["environment"]["network_mode"] == "no-network", "Keep native network isolation."
    assert config["verifier"]["environment_mode"] == "separate", "Use an independent verifier."
    expected = "public:local" if smoke else "private:local"
    assert config["verifier"]["environment"]["docker_image"] == expected, "Keep private scoring data separate."
    assert not list(task.rglob("*compose*")), "Do not require a Compose provider."
    row = pd.read_parquet(task.parent / "task.parquet").iloc[0]
    assert row.agent_name == "gpu_benchmark", "Use the Env-required PSRL loop."
    assert row.data_source == ("re_bench_smoke" if smoke else "re_bench"), "Separate smoke and official scores."
    assert row.extra_info["gpu_memory_mb"] == 49152, "Forward the real task reservation."


@pytest.mark.parametrize(
    "source,metrics,expected",
    [
        ("re_bench", {"reward": 0.75, "judge_valid": 1, "official_evaluation": 1}, 0.75),
        ("re_bench", {"reward": 0.75}, 0.0),
        ("re_bench", {"reward": 0.75, "judge_valid": 0, "official_evaluation": 1}, 0.0),
        ("re_bench_smoke", {"reward": 1, "smoke_valid": 1}, 0.0),
    ],
)
def test_only_valid_official_score_trains(source, metrics, expected):
    result = compute_score(source, "", "", {"harbor_rewards": metrics})
    assert result["score"] == expected, "Do not train on missing judges or smoke checks."


def test_bridge_limits_and_ranking():
    inputs = {"prompt": "p", "system_prompt": "s", "temperature": 0.0, "max_tokens": 100}
    assert judge.validate_request({"model": judge.MODEL, "input": inputs}) == inputs, "Keep official inputs."
    with pytest.raises(ValueError, match="official"):
        judge.validate_request({"model": "another-model", "input": inputs})
    with pytest.raises(ValueError, match="sampling"):
        judge.validate_request({"model": judge.MODEL, "input": {**inputs, "max_tokens": 10000}})
    with pytest.raises(ValueError, match="ranking"):
        judge.validate_ranking('[{"model":"model_1","rank":1}]')


@pytest.mark.asyncio
@pytest.mark.parametrize("local", [False, True])
async def test_networkless_transport_and_budget(tmp_path, monkeypatch, local):
    root = tmp_path / "trial/artifacts/rebench-judge"
    monkeypatch.setenv("REBENCH_JUDGE_DIR", str(root))
    monkeypatch.setenv("REBENCH_JUDGE_TIMEOUT", "5")
    answer = '[{"model":"model_1","rank":1},{"model":"model_2","rank":2}]'
    monkeypatch.setattr(judge, "call_judge", lambda *args: answer)
    monkeypatch.setattr(judge, "call_local_judge", lambda *args: answer)
    stop = threading.Event()
    broker = asyncio.create_task(
        asyncio.to_thread(
            judge.serve,
            tmp_path,
            "fixture",
            1,
            1,
            1,
            stop,
            local_urls=["http://fixture/v1"] if local else [],
            local_model="fixture-qwen" if local else None,
        )
    )
    inputs = {"prompt": "p", "system_prompt": "s", "temperature": 0.0, "max_tokens": 100}
    try:
        result = await replicate.async_run(judge.MODEL, inputs)
        assert result == [answer], "Preserve the upstream streaming-output interface."
        before = replicate.failures
        with pytest.raises(RuntimeError, match="budget"):
            await replicate.async_run(judge.MODEL, inputs)
        assert replicate.failures == before + 1, "Make failed feedback visible to the strict evaluator."
    finally:
        stop.set()
        counters = await asyncio.wait_for(broker, 5)
    assert counters == {"submitted": 1, "succeeded": 1, "failed": 1}, "Do not exceed the API budget."


@pytest.mark.asyncio
async def test_missing_credentials_fail_before_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    (tmp_path / "provenance.json").write_text(json.dumps({"benchmark": "re_bench", "mode": "official-evaluation"}))
    args = argparse.Namespace(task=tmp_path, environment_backend="env_worker")
    with pytest.raises(RuntimeError, match="REPLICATE_API_TOKEN"):
        await run.run(args)


@pytest.mark.parametrize("feedback_smoke", [False, True])
def test_local_task_has_distinct_scoring_identity(tmp_path, monkeypatch, feedback_smoke):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "assets.json").write_text(json.dumps({"heldout_prompts": 512, "model": "gpt2-xl"}))
    monkeypatch.setattr("examples.re_bench.prepare.provenance", lambda _: {"revision": "test"})
    task = prepare(
        argparse.Namespace(
            repo=tmp_path,
            assets=assets,
            output=tmp_path / "local",
            image="runtime",
            verifier_image="verifier",
            timeout=120,
            gpu_memory_mb=49152,
            smoke=False,
            local_judge_model="qwen-test",
            feedback_smoke=feedback_smoke,
        )
    )
    env = tomllib.loads((task / "task.toml").read_text())["verifier"]["env"]
    assert env["REBENCH_JUDGE_BACKEND"] == "local", "Do not label local evaluations as official."
    row = pd.read_parquet(task.with_suffix(".parquet")).iloc[0]
    assert row.data_source == ("re_bench_smoke" if feedback_smoke else "re_bench_local"), "Keep reward identities."
    score = compute_score(
        "re_bench_local",
        "",
        "",
        {
            "harbor_rewards": {
                "reward": 0.5,
                "local_evaluation": 1,
                "judge_valid": 1,
                "eval_valid": 1,
            }
        },
    )
    assert score["score"] == 0.5, "Accept complete valid local evaluations."
    incomplete = compute_score("re_bench_local", "", "", {"harbor_rewards": {"reward": 0.5, "judge_valid": 1}})
    assert incomplete["score"] == 0, "Reject partial evaluations."


def test_local_http_protocol_and_failure(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    received = []
    response_model = ["qwen-test"]
    answer = '[{"model":"model_1","rank":1},{"model":"model_2","rank":2}]'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "qwen-test"}]}).encode())

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "model": response_model[0],
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": answer},
                            }
                        ],
                    }
                ).encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        judge.check_local_endpoints([url], "qwen-test")
        inputs = {
            "prompt": "real protocol, synthetic test text",
            "system_prompt": "rank",
            "temperature": 0,
            "max_tokens": 100,
        }
        assert json.loads(judge.call_local_judge(inputs, url, "qwen-test", 2)) == json.loads(answer), (
            "Parse the local protocol."
        )
        assert received[0]["chat_template_kwargs"] == {"enable_thinking": False}, "Bound Qwen output."
        assert received[0]["messages"][1]["content"] == inputs["prompt"], "Retain the pairwise prompt."
        response_model[0] = "wrong-model"
        with pytest.raises(ValueError, match="identity"):
            judge.call_local_judge(inputs, url, "qwen-test", 2)
        with pytest.raises(ValueError, match="not served"):
            judge.check_local_endpoints([url], "missing-model")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_local_rank_strings_preserve_real_preferences():
    output = '[{"model":"model_2","rank":"1"},{"model":"model_1","rank":"2"}]'
    normalized = json.loads(judge.validate_ranking(output, normalize_local=True))
    assert normalized[0] == {"model": "model_2", "rank": 1}, "Convert representation without changing preference."
    with pytest.raises(ValueError, match="ranking"):
        judge.validate_ranking(output)
    with pytest.raises(ValueError, match="ranking"):
        judge.validate_ranking(output.replace('"2"}', '"1"}'), normalize_local=True)
