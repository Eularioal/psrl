import importlib

MODULES = [
    "psrl.workers.env_service",
    "psrl.workers.env_service.router",
    "psrl.workers.env_service.client",
    "psrl.workers.agent_loop.manager",
    "psrl.trainer.ppo.ray_trainer",
    "examples.sciaccel_rl.runner",
    "examples.sciaccel_rl.agent_loop",
    "examples.sciaccel_rl.env_demand",
    "psrl.workers.gen.session_router",
]

for name in MODULES:
    try:
        importlib.import_module(name)
        print("OK  ", name)
    except Exception as exc:  # noqa: BLE001
        print("FAIL", name, type(exc).__name__, str(exc)[:200])
