"""Trainer-side integration of the env service on a REAL (isolated) Ray cluster."""

import time
from types import SimpleNamespace

import ray
from omegaconf import OmegaConf
from psrl.trainer.ppo.ray_trainer import PSRL_RayPPOTrainer as RayPPOTrainer
from psrl.workers.env_service import GRANTED, REJECT, EnvDemand
from psrl.workers.env_service.config_check import validate_env_service_config
from psrl.workers.env_service.node_policy import restrict_to_unit_nodes, select_worker_nodes

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


ray.init(
    address="local",
    num_cpus=8,
    num_gpus=0,
    include_dashboard=False,
    _temp_dir="/tmp/env_verify/ray",
    logging_level="ERROR",
    namespace="envverify",
    log_to_driver=False,
)
node_ip = ray.util.get_node_ip_address()
stamp = int(time.time())

# --- real placement group, exactly the call _init_env_service makes ------------------------------
pg = ray.util.placement_group([{"CPU": 1}], strategy="PACK")
ray.get(pg.ready())
table = ray._private.state.state.placement_group_table(pg.id)
check(
    "ray._private placement_group_table(pg.id) works on this Ray",
    "bundles_to_node_id" in table,
    f"ray={ray.__version__}",
)

all_wg = {"actor_rollout": SimpleNamespace(resource_pool=SimpleNamespace(pgs=[pg]))}


def base_cfg(**extra):
    cfg = {
        "enable": True,
        "router_name": f"trainer_router_{stamp}_{len(results)}",
        "node_policy": "cpu_only",
        "default_gpu_memory_mb": 24000,
        "slot_ttl_s": 1800,
        "heartbeat_interval_s": 60,
        "train_rollout_node_ips": [],
        "allow_node_ips": [],
        "deny_node_ips": [],
        "units": [
            {"unit_id": "cpu0", "node_ip": node_ip, "cpu_quota": 32, "memory_mb": 100000, "max_containers": 4},
            {
                "unit_id": "gpu0",
                "node_ip": node_ip,
                "cpu_quota": 16,
                "memory_mb": 50000,
                "gpu_indices": [0],
                "max_containers": 4,
                "gpu_memory_mb": 97871,
            },
        ],
    }
    cfg.update(extra)
    return cfg


def trainer_with(cfg):
    return SimpleNamespace(config=OmegaConf.create({"psrl": {"env_service": cfg}}), env_router_handle=None)


# --- 1. disabled -> legacy path, no actor ---------------------------------------------------------
t = trainer_with(base_cfg(enable=False))
check(
    "disabled: returns None and creates no router",
    RayPPOTrainer._init_env_service(t, all_wg) is None and t.env_router_handle is None,
)

# --- 2. enabled, train/rollout nodes DERIVED from the real placement group ------------------------
t = trainer_with(base_cfg())
rules = RayPPOTrainer._init_env_service(t, all_wg)
check(
    "enabled: derives train/rollout nodes from the placement group",
    sorted(rules.train_rollout_node_ips) == [node_ip],
    f"{sorted(rules.train_rollout_node_ips)}",
)
check("enabled: router actor exists and answers", ray.get(t.env_router_handle.ping.remote()) is True)
check("enabled: env_unit_node_ips recorded", t.env_unit_node_ips == [node_ip])
warns = ray.get(t.env_router_handle.warnings.remote())
check(
    "GPU unit declared on a train node raises a lint warning",
    any("gpu0" in w for w in warns),
    (warns[0][:90] if warns else "no warnings"),
)
gpu = ray.get(t.env_router_handle.try_admit.remote("g", EnvDemand(needs_gpu=True), node_ip))
cpu = ray.get(t.env_router_handle.try_admit.remote("c", EnvDemand(), node_ip))
check(
    "cpu_only on a train node: GPU episode REJECT, CPU episode GRANTED",
    gpu.status == REJECT and cpu.status == GRANTED,
    f"gpu={gpu.status} cpu={cpu.status}",
)
ray.kill(t.env_router_handle)

# --- 3. explicit train_rollout_node_ips beats derivation; node_policy any lets GPU through --------
t = trainer_with(base_cfg(train_rollout_node_ips=["10.255.255.1"], node_policy="any"))
rules = RayPPOTrainer._init_env_service(t, all_wg)
gpu = ray.get(t.env_router_handle.try_admit.remote("g", EnvDemand(needs_gpu=True), node_ip))
check("explicit train_rollout_node_ips overrides derivation", sorted(rules.train_rollout_node_ips) == ["10.255.255.1"])
check(
    "env node (not a train node): GPU episode GRANTED on card 0",
    gpu.status == GRANTED and gpu.lease.gpu_indices == (0,),
    f"lease cards={gpu.lease.gpu_indices}",
)
ray.kill(t.env_router_handle)

# --- 4. worker-node selection against the real node table -----------------------------------------
alive = [(n["NodeID"], n["NodeManagerAddress"]) for n in ray.nodes() if n["Alive"]]
train_rules = RayPPOTrainer._init_env_service(trainer_with(base_cfg(router_name=f"tmp_{stamp}")), all_wg)
for name in ("cpu_only", "any"):
    rules = type(train_rules).build(name, train_rollout_node_ips=[node_ip])
    kept, _ = select_worker_nodes(alive, rules)
    check(f"policy {name}: the train node may host agent workers", len(kept) == 1)
rules = type(train_rules).build("avoid", train_rollout_node_ips=[node_ip])
try:
    select_worker_nodes(alive, rules)
    check("policy avoid with only a train node: startup raises instead of running broken", False, "no error")
except ValueError as exc:
    check("policy avoid with only a train node: startup raises instead of running broken", True, str(exc)[:80])
try:
    restrict_to_unit_nodes(alive, ["10.255.255.9"])
    check("no node declares a unit: startup raises", False, "no error")
except ValueError as exc:
    check("no node declares a unit: startup raises", True, str(exc)[:80])

# --- 5. startup validation of bad configs (pure function; same one validate_config calls) ----------
bad = {
    "no units": dict(base_cfg(), units=[]),
    "unit missing a key": dict(base_cfg(), units=[{"unit_id": "x", "node_ip": "1.1.1.1"}]),
    "two units share a card": dict(
        base_cfg(),
        units=[
            {
                "unit_id": "a",
                "node_ip": "1.1.1.1",
                "cpu_quota": 1,
                "memory_mb": 1,
                "gpu_indices": [0],
                "gpu_memory_mb": 90000,
                "max_containers": 2,
            },
            {
                "unit_id": "b",
                "node_ip": "1.1.1.1",
                "cpu_quota": 1,
                "memory_mb": 1,
                "gpu_indices": [0],
                "gpu_memory_mb": 90000,
                "max_containers": 2,
            },
        ],
    ),
    "heartbeat > ttl/2": dict(base_cfg(), slot_ttl_s=100, heartbeat_interval_s=60),
    "gpu unit smaller than one task": dict(
        base_cfg(),
        units=[
            {
                "unit_id": "s",
                "node_ip": "1.1.1.1",
                "cpu_quota": 1,
                "memory_mb": 1,
                "gpu_indices": [0],
                "gpu_memory_mb": 1000,
                "max_containers": 2,
            }
        ],
    ),
    "unknown node_policy": dict(base_cfg(), node_policy="whatever"),
}
for label, cfg in bad.items():
    try:
        validate_env_service_config(cfg)
        check(f"validate rejects: {label}", False, "accepted")
    except ValueError as exc:
        check(f"validate rejects: {label}", True, str(exc)[:70])
check("validate accepts the good config", len(validate_env_service_config(base_cfg())) == 2)

# --- 6. the shipped default yaml loads, is disabled, and composes ---------------------------------
shipped = OmegaConf.load("psrl/trainer/config/psrl/env_service.yaml")
check(
    "shipped env_service.yaml defaults to enable=False with no units",
    shipped.enable is False and len(shipped.units) == 0,
)

ray.shutdown()
print(f"\n{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
