"""
Make heavyweight modules importable on a CPU-only box, without ever hiding a real one.

`stub_if_missing` installs a stand-in only when the real import fails, so in a full
environment (Ray, aiohttp, FastAPI, Harbor installed) every test here runs against the
genuine modules and nothing is shadowed. On a bare box the stand-ins are enough to import
`session_router` and the SciAccel runner and exercise their own logic.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

REPO_ROOT = Path(__file__).resolve().parents[2]
_STUB_FLAG = "_light_stub"


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def stub_if_missing(name: str, **attrs: object) -> types.ModuleType:
    """
    Return the real module when it imports, else install and return a stand-in.

    Calling it again for a name that is already one of our stand-ins adds any attributes it
    does not have yet, so independent loaders can share a stub without clobbering each other.
    """
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, _STUB_FLAG, False):
        for key, value in attrs.items():
            existing.__dict__.setdefault(key, value)
        return existing
    if _importable(name):
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    module.__path__ = []  # type: ignore[attr-defined]  # a package, so dotted children resolve
    setattr(module, _STUB_FLAG, True)
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)
    return module


def stub_chain(name: str, **attrs: object) -> types.ModuleType:
    """Stub every dotted prefix of `name` that is missing, putting `attrs` on the leaf."""
    parts = name.split(".")
    for i in range(1, len(parts)):
        stub_if_missing(".".join(parts[:i]))
    return stub_if_missing(name, **attrs)


def stub_package_keeping_children(name: str, directory: Path) -> None:
    """
    Replace a package whose `__init__` will not import with an empty one.

    Its real submodules still load from `directory`, which lets a stdlib-only module such
    as `docker_utils` import for real while skipping a package `__init__` that pulls in Ray
    or aiohttp.
    """
    if _importable(name):
        return
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]  # type: ignore[attr-defined]
    sys.modules[name] = package


class _Anything:
    """Accepts any construction, call or attribute, for stubbing web frameworks."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, _name):
        return self


def load_session_router() -> types.ModuleType:
    """Import `psrl.workers.gen.session_router`, stubbing only what the box lacks."""
    stub_if_missing("aiohttp")
    stub_if_missing("fastapi", FastAPI=_Anything, Request=_Anything)
    stub_chain("fastapi.responses", JSONResponse=_Anything, Response=_Anything)
    stub_package_keeping_children("psrl.utils.common", REPO_ROOT / "psrl" / "utils" / "common")
    stub_if_missing(
        "psrl.utils.common.http_utils",
        HttpResponse=_Anything,
        create_aiohttp_client=MagicMock(),
        filter_http_headers=MagicMock(),
        request_raw=MagicMock(),
    )
    stub_package_keeping_children("psrl.utils.rollout", REPO_ROOT / "psrl" / "utils" / "rollout")
    stub_if_missing("psrl.utils.rollout.turn_output_writer", TurnOutputWriter=_Anything)
    stub_chain(
        "psrl.workers.gen.smg_adapter",
        TITO_SESSIONS_PATH="tito/sessions",
        TRAJECTORY_ID_STRATEGIES={"manual", "auto"},
    )
    return importlib.import_module("psrl.workers.gen.session_router")


class _Cfg:
    """Keyword container standing in for a Harbor pydantic config."""

    _defaults: dict = {}

    def __init__(self, **kwargs):
        self.__dict__.update({k: (list(v) if isinstance(v, list) else v) for k, v in self._defaults.items()})
        self.__dict__.update(kwargs)


class _JobConfig(_Cfg):
    _defaults = {"extra_instructions": [], "environment": None}


def load_sciaccel_runner() -> types.ModuleType:
    """Import `examples.sciaccel_rl.runner`, stubbing Harbor and friends only when absent."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    stub_if_missing("omegaconf", OmegaConf=MagicMock(), DictConfig=object)
    stub_if_missing("aiohttp")
    stub_package_keeping_children("psrl.utils.common", REPO_ROOT / "psrl" / "utils" / "common")
    stub_if_missing("psrl.utils.common.http_utils", post=AsyncMock(return_value=None))
    stub_chain("harbor.job", Job=MagicMock())
    stub_chain("harbor.models.job.config", AgentConfig=_Cfg, JobConfig=_JobConfig, SourceJobConfig=_Cfg)
    stub_chain("harbor.models.trial.config", TaskConfig=_Cfg)
    return importlib.import_module("examples.sciaccel_rl.runner")
