"""Check sampling support and coarse probabilities under an explicit local budget."""

import torch


def expected_distribution(probs, top_k=None, top_p=None):
    """Apply the public reference's sequential top-k then normalized top-p rule.

    The upstream tolerance mask is only a support acceptance envelope. It is
    not the exact probability distribution (especially for combined top-k/p).
    """
    if top_p is not None and (not torch.isfinite(top_p).all() or ((top_p < 0) | (top_p > 1)).any()):
        raise ValueError("The workload top_p must be finite and inside [0, 1].")
    result = probs.float().clone()
    for row in range(len(result)):
        if top_k is not None:
            k = int(top_k[row] if top_k.ndim else top_k)
            if 0 < k < result.shape[1]:
                indices = result[row].argsort(descending=True)[k:]
                result[row, indices] = 0
        result[row] /= result[row].sum()
        if top_p is not None:
            p = float(top_p[row] if top_p.ndim else top_p)
            if p <= 0:
                best = result[row].argmax()
                result[row].zero_()
                result[row, best] = 1
            elif p < 1:
                values, indices = result[row].sort(descending=True)
                remove = values.cumsum(0) > p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                result[row, indices[remove]] = 0
                result[row] /= result[row].sum()
    return result


def check(definition, workload, solution, root):
    from flashinfer_bench.bench.evaluators.sampling import _compute_valid_sampling_mask, _detect_thresholding_method
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.compile import BuilderRegistry

    tensors = load_safetensors(definition, workload, root)
    inputs = gen_inputs(definition, workload, device="cuda:0", safe_tensors=tensors)
    names = list(definition.inputs)
    named = dict(zip(names, inputs, strict=True))
    probs = named["probs"]
    expected = expected_distribution(probs, named.get("top_k"), named.get("top_p"))
    mask = _compute_valid_sampling_mask(
        probs, _detect_thresholding_method(definition), {k: v for k, v in named.items() if k != "probs"}
    )
    if not torch.isfinite(expected).all():
        raise ValueError("Expected a finite sampling distribution.")
    runnable = BuilderRegistry.get_instance().build(definition, solution)
    batch, vocab = probs.shape
    max_error = 0.0
    for row in range(batch):
        expanded = [
            value[row : row + 1].repeat(128, *([1] * (value.ndim - 1)))
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch
            else value
            for value in inputs
        ]
        draws = []
        for _ in range(16):
            with torch.no_grad():
                samples = runnable(*expanded)
            if isinstance(samples, (tuple, list)):
                samples = samples[0]
            if samples.shape != (128,) or samples.dtype not in (torch.int32, torch.int64):
                raise ValueError("Return one integer token per input row.")
            if not samples.is_cuda or ((samples < 0) | (samples >= vocab)).any():
                raise ValueError("Return CUDA samples inside the vocabulary.")
            if not mask[row, samples.long()].all():
                raise ValueError("Samples violate the top-k or top-p support.")
            draws.append(samples.long())
        observed = torch.bincount(torch.cat(draws) % 32, minlength=32).float() / 2048
        target = torch.zeros(32, device=probs.device)
        target.scatter_add_(0, torch.arange(vocab, device=probs.device) % 32, expected[row])
        error = (observed - target).abs().max().item()
        max_error = max(max_error, error)
        if error > 0.07:
            raise ValueError(f"Coarse probability error exceeds the local bound: {error!r}.")
    torch.cuda.synchronize()
    return {
        "reward": 1,
        "correctness": 1,
        "local_evaluation": 1,
        "eval_valid": 1,
        "sampling_quick": 1,
        "samples_per_row": 2048,
        "max_bin_error": max_error,
    }


def main():
    import argparse
    import json
    from pathlib import Path

    from flashinfer_bench import TraceSet

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    traces = TraceSet.from_path(args.root)
    rows = []
    for name, definition in traces.definitions.items():
        try:
            result = check(definition, traces.workloads[name][0].workload, traces.solutions[name][0], args.root)
            rows.append({"definition": name, "status": "passed", "result": result})
        except Exception as exc:
            rows.append({"definition": name, "status": "error", "error": repr(exc)})
        (args.output / "sampling-results.json").write_text(json.dumps(rows, indent=2))
    (args.output / "reward.json").write_text(json.dumps({"reward": 0, "quick_check": 1, "evaluations": len(rows)}))


if __name__ == "__main__":
    main()
