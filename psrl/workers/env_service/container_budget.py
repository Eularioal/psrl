"""Reserve and enforce the complete Harbor container collection's host resources."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContainerBudget:
    """Bound one main service and its sidecars in each admitted environment."""

    main_cpu: float
    main_memory_mb: int
    max_sidecars: int
    environments: int = 1
    sidecar_cpu: float = 0.25
    sidecar_memory_mb: int = 128

    def __post_init__(self):
        if min(self.main_cpu, self.main_memory_mb, self.sidecar_cpu, self.sidecar_memory_mb) <= 0:
            raise ValueError("Container CPU and memory limits must be positive.")
        if self.environments not in (1, 2) or self.max_sidecars < 0:
            raise ValueError("Invalid Harbor environment or sidecar count.")

    @property
    def cpu(self):
        return self.environments * (self.main_cpu + self.max_sidecars * self.sidecar_cpu)

    @property
    def memory_mb(self):
        return self.environments * (self.main_memory_mb + self.max_sidecars * self.sidecar_memory_mb)

    def service_limits(self, services):
        """Refuse unbudgeted services and pin each service to one bounded container."""
        if "main" not in services or len(services) - 1 > self.max_sidecars:
            raise ValueError("Compose services exceed the admitted container budget.")
        limits = {}
        for name in services:
            cpu = self.main_cpu if name == "main" else self.sidecar_cpu
            memory = f"{self.main_memory_mb if name == 'main' else self.sidecar_memory_mb}M"
            limits[name] = {
                "cpus": cpu,
                "mem_limit": memory,
                "memswap_limit": memory,
                "scale": 1,
                "deploy": {"replicas": 1, "resources": {"limits": {"cpus": str(cpu), "memory": memory}}},
            }
        return limits
