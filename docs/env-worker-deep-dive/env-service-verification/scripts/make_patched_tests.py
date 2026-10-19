"""Copy tests/env_service to /tmp and adapt only the dict-style access to Harbor's environment config."""

import shutil
from pathlib import Path

SRC = (
    next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    / "tests/env_service"
)
DST = Path("/tmp/env_verify/patched_env_service_tests")

if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)

target = DST / "test_runner_env_service.py"
text = target.read_text()

helper = '''

def _env_dict(env):
    """Real Harbor gives a pydantic EnvironmentConfig; the stub gave a dict."""
    if env is None:
        return {}
    if hasattr(env, "model_dump"):
        return env.model_dump(exclude_unset=True)
    return env

'''
text = text.replace('NODE = "10.0.0.1"', helper + 'NODE = "10.0.0.1"', 1)

replacements = [
    ("env = job_config.environment or {}", "env = _env_dict(job_config.environment)"),
    (
        'Path(job_config.environment["extra_docker_compose"][0])',
        'Path(_env_dict(job_config.environment)["extra_docker_compose"][0])',
    ),
    (
        "(job_cls.create.await_args.args[0].environment or {})",
        "_env_dict(job_cls.create.await_args.args[0].environment)",
    ),
]
for old, new in replacements:
    assert old in text, old
    text = text.replace(old, new)

target.write_text(text)
print("patched", target, "with", len(replacements), "replacements")
