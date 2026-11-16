"""Materialize an isolated runtime with lazy FP8 dependency initialization."""

import hashlib
import shutil
from pathlib import Path


def materialize(repo, output):
    repo, output = Path(repo), Path(output)
    source = repo / "fastkernels/bench.py"
    text = source.read_text()
    old = (
        "    from fastkernels.tasks.baseline.L1.fp8_linear import Fp8Linear\n"
        "    from fastkernels.infra.weight_loader import _postprocess_moe_fp8_weights\n"
    )
    if text.count(old) != 1:
        raise ValueError("Review the changed upstream FP8 initialization contract.")
    new = "    if not any(p.dtype == torch.float8_e4m3fn for p in module.parameters()):\n        return\n" + old
    shutil.copytree(
        repo / "fastkernels", output / "fastkernels", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    patched = text.replace(old, new)
    copy_line = "            strided.copy_(t)"
    if patched.count(copy_line) != 1:
        raise ValueError("Review changed upstream strided input materialization.")
    patched = patched.replace(
        copy_line,
        "            from psrl_input_builders import copy_strided\n            copy_strided(strided, t)",
    )
    scalar_line = "    return {k: v for k, v in call.items() if isinstance(v, (int, float, bool, str))}"
    if patched.count(scalar_line) != 1:
        raise ValueError("Review changed upstream constructor metadata handling.")
    patched = patched.replace(
        scalar_line, "    from psrl_input_builders import normalize_init\n    return normalize_init(call)"
    )
    dispatch = "    builder = _INPUT_BUILDERS.get(op.qualname)"
    if patched.count(dispatch) != 1:
        raise ValueError("Review changed upstream input builder dispatch.")
    patched = patched.replace(
        dispatch,
        '    if op.class_name in ("OneHot", "EmbeddingBag", "EmbeddingBagCollection", "DLRMv2", "LightGCN"):\n'
        "        from psrl_input_builders import build\n"
        "        return build(op.qualname, fwd_args, device, init_args, _materialize_all)\n" + dispatch,
    )
    shutil.copy2(Path(__file__).with_name("input_builders.py"), output / "psrl_input_builders.py")
    shutil.copy2(Path(__file__).with_name("psrl_bench_config.py"), output / "psrl_bench_config.py")
    (output / "fastkernels/bench.py").write_text(patched)
    return {
        "patch": "Lazy FP8 initialization, valid integer inputs, and broadcast stride handling.",
        "original_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "runtime_sha256": hashlib.sha256(patched.encode()).hexdigest(),
    }
