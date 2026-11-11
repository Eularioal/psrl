"""Demonstrate startup contracts and physical recovery gates on CPU only."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("router_contract", REPO / "psrl/workers/env_service/router_contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def main():
    # Diagram 1 and 2 use the production contract comparison.
    actual = contract.router_contract({"default_gpu_memory_mb": 80000})
    contract.verify_router_contract(actual, contract.router_contract({"default_gpu_memory_mb": 80000}))
    print("same_configuration: accepted")
    try:
        contract.verify_router_contract(actual, contract.router_contract({"default_gpu_memory_mb": 40000}))
    except ValueError:
        print("changed_configuration: rejected")
    else:
        raise AssertionError("Changed configuration must be rejected.")

    # Diagram 3 to 6 simulate physical probes without Ray, Docker or CUDA.
    failures, reserved, healthy = 3, 60000, False
    print(f"exhausted: failures={failures}, reserved_mb={reserved}, healthy={healthy}")
    failures = 0
    docker_ready, cleanup_confirmed = True, False
    healthy = docker_ready and cleanup_confirmed
    assert not healthy and reserved == 60000, "A responsive service cannot prove physical return."
    print(f"probe_without_cleanup: healthy={healthy}, reserved_mb={reserved}")
    cleanup_confirmed = True
    if cleanup_confirmed:
        reserved = 0
    healthy = docker_ready and cleanup_confirmed
    assert healthy and reserved == 0, "Confirmed cleanup must permit safe reuse in this model."
    print(f"confirmed_cleanup: healthy={healthy}, reserved_mb={reserved}")


if __name__ == "__main__":
    main()
