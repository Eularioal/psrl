"""Check bounded protocols reject invalid evidence and preserve input contracts."""

import torch
from examples.fastkernels_bench.input_builders import build, copy_strided, normalize_init
from examples.gpu_benchmarks.reward import compute_score
from examples.swe_serve.build_local_images import local_recipe


def test_recommendation_offsets_and_vocabulary_bounds():
    values = {"input_ids": torch.arange(8), "offsets": torch.tensor([15, -2, 99])}
    _, result = build("module:EmbeddingBag", {}, "cpu", {"include_last_offset": True}, lambda *args: values)
    assert result["offsets"].tolist() == [0, 4, 8], "Offsets must partition the actual flattened indices."
    config = normalize_init({"config": {"$dataclass": "Config", "fields": {"num_embeddings_per_feature": [3, 5]}}})
    values = {"sparse_indices": [torch.tensor([-1, 9]), torch.tensor([12, 17])]}
    _, result = build("module:DLRMv2", {}, "cpu", config, lambda *args: values)
    assert result["sparse_indices"][0].tolist() == [2, 0], "Indices must respect the first table vocabulary."
    assert result["sparse_indices"][1].tolist() == [2, 2], "Indices must respect the second table vocabulary."


def test_local_image_uses_selected_runtime_and_keeps_version_pin():
    source = (
        "FROM upstream:old\nRUN git clone source /code && cd /code && git checkout abc123\n"
        "COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv\n"
        "RUN uv pip install --system transformers==5.3.0\n"
    )
    result = local_recipe(source, "local:modern", wheelhouse=True)
    assert result.startswith("FROM local:modern\n"), "Image contents must use the selected runtime."
    assert "RUN rm -rf /code\nCOPY code /code" in result, "Pinned source must replace inherited source."
    assert "transformers==5.3.0" in result, "Explicit task dependency pins must remain intact."
    assert "PIP_NO_INDEX=1" in result and "ghcr.io" not in result, "Offline builds must use local dependencies."


def test_broadcast_inputs_preserve_captured_strides_without_overlapping_writes():
    target = torch.empty_strided((2, 1, 8, 8), (0, 64, 8, 1))
    source = torch.randn(2, 1, 8, 8)
    copy_strided(target, source)
    assert target.stride() == (0, 64, 8, 1), "Captured broadcast layout must remain intact."
    assert torch.equal(target[0], target[1]), "Broadcast rows must reference the same values."
    assert torch.equal(target[0], source[0]), "Materialization must initialize the underlying storage."


def test_one_hot_indices_respect_forward_class_count():
    values = {"x": torch.tensor([-1, 127, 8]), "num_classes": 8}
    _, result = build("module:OneHot", {}, "cpu", {}, lambda *args: values)
    encoded = torch.nn.functional.one_hot(result["x"], result["num_classes"])
    assert encoded.shape == (3, 8), "Generated indices must produce the captured class dimension."
    assert encoded.argmax(-1).tolist() == [7, 7, 0], "All indices must lie inside the class vocabulary."


def test_bounded_swe_reward_requires_executed_valid_tests():
    metrics = {"reward": 1, "local_evaluation": 1, "eval_valid": 1, "tests": 4}
    for key in ("local_evaluation", "eval_valid", "tests"):
        result = compute_score("swe_serve_regression_local", "", None, {"harbor_rewards": {**metrics, key: 0}})
        assert result["score"] == 0, "Missing execution evidence cannot produce a positive reward."
