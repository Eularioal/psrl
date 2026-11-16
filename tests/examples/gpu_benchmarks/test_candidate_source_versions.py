"""
Reject stale artifact bytecode at real RE candidate loading boundaries without a GPU.
"""

import asyncio
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.re_bench import cumsum_verify, restricted_mlm_verify

ROOT = Path(__file__).resolve().parents[3]


def stale_source(
    directory,
    invalidation,
):
    """
    Preserve valid old bytecode while replacing source with equal sized new bytes.
    """
    source = directory / "candidate.py"
    source.write_text("raise RuntimeError('old-source')\n")
    os.utime(source, (1_700_000_000, 1_700_000_000))
    cache = Path(py_compile.compile(str(source), doraise=True, invalidation_mode=invalidation))
    source.write_text("raise RuntimeError('new-source')\n")
    os.utime(source, (1_700_000_000, 1_700_000_000))
    return source, cache


def artifact_paths(
    source,
    output,
):
    """
    Map only fixed container paths while keeping the real verifier loading statements.
    """

    def resolve(value):
        if str(value).startswith("/logs/artifacts/"):
            return source
        return output if str(value) == "/logs/verifier" else Path(value)

    return resolve


@pytest.mark.parametrize(
    "invalidation", [py_compile.PycInvalidationMode.TIMESTAMP, py_compile.PycInvalidationMode.UNCHECKED_HASH]
)
@pytest.mark.parametrize("verifier", ["restricted", "cumsum", "rust"])
def test_real_candidate_entry_ignores_stale_bytecode(tmp_path, monkeypatch, invalidation, verifier):
    source, cache = stale_source(tmp_path, invalidation)
    cached = cache.read_bytes()
    output = tmp_path / "verifier"
    output.mkdir()
    resolve = artifact_paths(source, output)
    if verifier == "restricted":
        with pytest.raises(RuntimeError, match="^new-source$"):
            restricted_mlm_verify.load("candidate_model", source)
    elif verifier == "cumsum":
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "reference", SimpleNamespace(prefix_sum_torch=None))
        monkeypatch.setattr(cumsum_verify, "Path", resolve)
        cumsum_verify.main()
        assert json.loads((output / "error.json").read_text())["error"] == "RuntimeError('new-source')", (
            "The real prefix sum verifier must execute the submitted source."
        )
    else:
        model = SimpleNamespace()
        model.cuda = lambda: model
        model.eval = lambda: model
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None))
        monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(get_encoding=lambda _: None))
        monkeypatch.setitem(sys.modules, "rust_codecontests_utils", SimpleNamespace(evaluate_rust_code=None))
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)),
        )
        path = ROOT / "examples/re_bench/rust_verify.py"
        spec = importlib.util.spec_from_file_location("actual_rust_verifier", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module, "Path", resolve)
        with pytest.raises(RuntimeError, match="^new-source$"):
            asyncio.run(module.evaluate())
    assert cache.read_bytes() == cached, "Loading source must not delete or rewrite unrelated bytecode."


@pytest.mark.parametrize("flags", [[], ["-B"]])
def test_fresh_verifier_process_loads_current_source(tmp_path, flags):
    source = tmp_path / "model.py"
    source.write_text("VALUE = 1\n")
    os.utime(source, (1_700_000_000, 1_700_000_000))
    py_compile.compile(str(source), doraise=True)
    source.write_text("VALUE = 2\n")
    os.utime(source, (1_700_000_000, 1_700_000_000))
    code = (
        "from examples.re_bench.restricted_mlm_verify import load\n"
        "import sys\n"
        "print(load('candidate_model', sys.argv[1]).VALUE)\n"
    )
    result = subprocess.check_output(
        [sys.executable, *flags, "-c", code, str(source)], cwd=ROOT, text=True, timeout=10
    )
    assert result.strip() == "2", "A fresh verifier must read current source even when bytecode writing is disabled."


def test_explicit_source_preserves_encoding_and_module_identity(tmp_path):
    source = tmp_path / "model.py"
    source.write_bytes("# coding: latin-1\nVALUE = 'caf\xe9'\n".encode("latin-1"))
    module = restricted_mlm_verify.load("candidate_model", source)
    assert module.VALUE == "caf\xe9", "Source compilation must honor the declared Python encoding."
    assert module.__file__ == str(source), (
        "Keep source paths for code that resolves checkpoint files relative to itself."
    )
    assert module.__name__ == "candidate_model" and module.__spec__.origin == str(source), (
        "Explicit source execution must preserve module identity."
    )


def test_explicit_source_does_not_turn_syntax_errors_into_cached_success(tmp_path):
    source = tmp_path / "model.py"
    source.write_text("VALUE = 1\n")
    os.utime(source, (1_700_000_000, 1_700_000_000))
    py_compile.compile(str(source), doraise=True)
    source.write_text("VALUE = !\n")
    os.utime(source, (1_700_000_000, 1_700_000_000))
    with pytest.raises(SyntaxError):
        restricted_mlm_verify.load("candidate_model", source)
