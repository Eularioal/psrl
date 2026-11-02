"""Read real provider inheritance and illustrate reservation scopes without services."""

import ast
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[4]
    providers = {
        "compose": ("psrl/workers/env_service/docker.py", "ManagedDockerEnvironment"),
        "native": ("psrl/workers/env_worker/harbor.py", "EnvWorkerEnvironment"),
    }
    print("Actual source inheritance, no imports of Ray or Docker:")
    for provider, (relative, name) in providers.items():
        tree = ast.parse((root / relative).read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        print(f"{provider}: {name}({', '.join(ast.unparse(base) for base in cls.bases)})")

    print("\nTeaching simulation only: 8 GiB pool, 3 GiB per environment, no physical GPU.")
    for provider in providers:
        for mode in ("episode", "command"):
            # Diagram 2 and 3: admission precedes creation for either provider.
            reserved = 3 if mode == "episode" else 0
            print(f"\n{provider}/{mode}: session admitted, container alive, reserved={reserved} GiB")
            state = {"source": "edited"}
            for job in ("compile-and-test", "retest"):
                # Diagram 4 and 5: command borrows again, episode retains its grant.
                if mode == "command":
                    reserved += 3
                print(f"  {job}: running, reserved={reserved} GiB, free={8 - reserved} GiB")
                print(f"  output ready: reserved={reserved} GiB")
                # Diagram 5: assume physical return succeeded, never free at output alone.
                if mode == "command":
                    reserved -= 3
                print(f"  settled: reserved={reserved} GiB, source={state['source']}")
            # Diagram 6: successful container removal precedes session release.
            state.clear()
            reserved = 0
            print(f"  environment removed, session released, reserved={reserved} GiB")


if __name__ == "__main__":
    main()
