"""Exercise the public multi-item scoring API on CUDA before submitting code."""

import json


def main():
    from sglang import Engine

    engine = Engine(
        model_path="/hf-cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        enable_mis=True,
        attention_backend="flashinfer",
        mem_fraction_static=0.15,
    )
    try:
        scores = engine.score(
            query="Rate each option:",
            items=["Option A", "Option B"],
            label_token_ids=[9454, 2753],
            apply_softmax=True,
        ).scores
        if len(scores) != 2 or any(len(row) != 2 or abs(sum(row) - 1) > 1e-5 for row in scores):
            raise ValueError("The scoring service returned invalid probabilities.")
        print(json.dumps({"cuda_service_probe": 1, "items": len(scores)}))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
