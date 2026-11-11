"""Compare immutable startup contracts before reusing a named router."""

import hashlib
import json
from collections.abc import Mapping

ROUTER_PROTOCOL_VERSION = 1


def router_contract(config, train_rollout_node_ips=()):
    """Snapshot declared configuration and node identities without retaining caller references."""

    def plain(value):
        if isinstance(value, Mapping):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    payload = {
        "protocol_version": ROUTER_PROTOCOL_VERSION,
        "config": plain(config),
        "train_rollout_node_ips": sorted(set(train_rollout_node_ips)),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"fingerprint": hashlib.sha256(serialized.encode()).hexdigest(), "configuration": json.loads(serialized)}


def verify_router_contract(expected, actual):
    """Reject incompatible reuse without exposing or modifying live service configuration."""
    if expected != actual:
        raise ValueError(
            "EnvRouter startup contract mismatch. Use the same declared configuration and protocol, "
            "or stop the existing router after all leases and cleanup complete before replacing it."
        )
