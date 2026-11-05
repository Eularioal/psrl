"""Run Harbor verification through explicit managed GPU calls."""

from harbor.verifier.verifier import Verifier

VERIFIER_PATH = "examples.kernel_bench.verifier:KernelCommandVerifier"


def validate_units(units):
    """Restrict performance evaluation to a dedicated serial direct device."""
    if not units or any(
        u.get("gpu_reservation", "episode") != "command"
        or u.get("gpu_execution_backend", "direct") != "direct"
        or len(u.get("gpu_indices", [])) != 1
        or u.get("max_containers", 1) != 1
        or u.get("rollout_replica_rank") is not None
        or u.get("shared_rollout_replica_rank") is not None
        for u in units
    ):
        raise ValueError("Kernel command evaluation requires dedicated single-GPU direct units with one session.")


class ManagedVerifierEnvironment:
    """Delegate file operations and reserve GPU access for verifier shell calls."""

    def __init__(self, environment, timeout):
        binding = environment._gpu_execution
        if not binding or binding["mode"] != "command" or binding.get("backend", "direct") != "direct":
            raise ValueError("Kernel verifier requires a direct command lease.")
        self.environment = environment
        self.timeout = timeout

    def __getattr__(self, name):
        return getattr(self.environment, name)

    async def exec(self, command, **kwargs):
        timeout = kwargs.pop("timeout_sec", None) or self.timeout
        result = await self.environment.exec_gpu(
            command, timeout_sec=timeout, wait_timeout_sec=self.timeout, overlap_return=False, **kwargs
        )
        return result


class KernelCommandVerifier(Verifier):
    """Preserve Harbor upload, logging and scoring while managing device access."""

    def __init__(self, *args, **kwargs):
        kwargs["skip_tests_upload"] = True
        super().__init__(*args, **kwargs)
        if self.task.config.verifier.environment_mode != "separate":
            raise ValueError("Kernel command verification requires a separate preprovisioned verifier.")
        self.environment = ManagedVerifierEnvironment(self.environment, self.task.config.verifier.timeout_sec)
