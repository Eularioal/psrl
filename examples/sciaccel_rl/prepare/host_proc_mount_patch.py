"""Render a host procfs deployment patch without accessing or modifying a cluster."""

import argparse
import json
import re


def mount_patch(container, target):
    """Attach host procfs only to the selected trusted node runtime container."""
    if len(container) > 63 or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", container):
        raise ValueError("Container name must be a Kubernetes DNS label.")
    spec = {
        "volumes": [{"name": "psrl-host-proc", "hostPath": {"path": "/proc", "type": "Directory"}}],
        "containers": [
            {
                "name": container,
                "volumeMounts": [{"name": "psrl-host-proc", "mountPath": "/host/proc", "readOnly": True}],
                "env": [{"name": "PSRL_GPU_HOST_PROC", "value": "/host/proc"}],
            }
        ],
    }
    if target == "pod":
        return {"spec": spec}
    if target == "controller":
        return {"spec": {"template": {"spec": spec}}}
    raise ValueError("Target must be pod or controller.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--container", required=True, help="Existing trusted node container name in the outer template."
    )
    parser.add_argument("--target", choices=("pod", "controller"), default="pod")
    args = parser.parse_args()
    try:
        document = mount_patch(args.container, args.target)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
