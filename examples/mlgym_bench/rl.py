"""Train and independently evaluate bounded MLGym policies on real Gymnax CUDA environments."""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

ENVIRONMENTS = {
    "rlBreakoutMinAtar": "Breakout-MinAtar",
    "rlMetaMazeMisc": "MetaMaze-misc",
    "rlMountainCarContinuous": "MountainCarContinuous-v0",
    "rlMountainCarContinuousReinforce": "MountainCarContinuous-v0",
}


def execute(source, task, checkpoint, training):
    import gymnax
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from dotmap import DotMap
    from flax import serialization

    if not jax.devices() or any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("This task requires the JAX CUDA backend.")
    sys.path.insert(0, str(source / "src"))
    from networks import get_model_ready

    config = DotMap(yaml.safe_load((source / "src/config.yaml").read_text())["train_config"])
    config.network_config.num_hidden_units = 32
    config.network_config.num_hidden_layers = 2
    model, initial = get_model_ready(jax.random.PRNGKey(42), config)
    env, env_params = gymnax.make(ENVIRONMENTS[task])
    batch_size, frames = 16, 64

    @jax.jit
    def collect(params, key):
        key, reset_key = jax.random.split(key)
        obs, state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(reset_key, batch_size), env_params)

        def step(carry, unused):
            observation, env_state, rng, alive = carry
            rng, action_rng, env_rng = jax.random.split(rng, 3)
            value, distribution = model.apply(params, observation, rng=None)
            action = distribution.sample(seed=action_rng)
            next_obs, next_state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(env_rng, batch_size), env_state, action, env_params
            )
            data = (observation, action, reward * alive, value.squeeze(-1), alive)
            return (next_obs, next_state, rng, alive * (1 - done)), data

        _, trajectory = jax.lax.scan(step, (obs, state, key, jnp.ones(batch_size)), None, length=frames)
        return trajectory

    params = initial
    updates = 0
    if training:
        optimizer = optax.adam(3e-4)
        state = optimizer.init(params)

        @jax.jit
        def update(parameters, optimizer_state, trajectory):
            obs, actions, rewards, values, alive = trajectory

            def accumulate(total, reward):
                total = reward + 0.99 * total
                return total, total

            _, returns = jax.lax.scan(accumulate, jnp.zeros(batch_size), rewards, reverse=True)
            advantage = jax.lax.stop_gradient(returns - values)

            def objective(candidate):
                flat_obs = obs.reshape((-1, *obs.shape[2:]))
                flat_actions = actions.reshape((-1, *actions.shape[2:]))
                predicted, distribution = model.apply(candidate, flat_obs, rng=None)
                log_prob = distribution.log_prob(flat_actions).reshape((frames, batch_size))
                value_loss = (predicted.reshape((frames, batch_size)) - returns) ** 2
                return jnp.mean(alive * (-log_prob * advantage + 0.5 * value_loss))

            loss, grads = jax.value_and_grad(objective)(parameters)
            delta, optimizer_state = optimizer.update(grads, optimizer_state, parameters)
            return optax.apply_updates(parameters, delta), optimizer_state, loss

        for seed in (42, 43):
            trajectory = collect(params, jax.random.PRNGKey(seed))
            params, state, loss = update(params, state, trajectory)
            if not math.isfinite(float(loss)):
                raise ValueError("Training produced a nonfinite loss.")
            updates += 1
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(serialization.to_bytes(params))
    else:
        if not checkpoint.is_file() or checkpoint.stat().st_size > 16 * 1024**2:
            raise ValueError("Missing or oversized policy checkpoint.")
        params = serialization.from_bytes(initial, checkpoint.read_bytes())

    returns = []
    for seed in (44,) if training else (109, 211):
        trajectory = collect(params, jax.random.PRNGKey(seed))
        returns.extend(np.asarray(trajectory[2].sum(axis=0)).tolist())
    raw = float(np.mean(returns))
    if not math.isfinite(raw):
        raise ValueError("Evaluation produced a nonfinite return.")
    reward = math.exp(min(raw, 0) / 100) if "MountainCar" in task else 1 - math.exp(-max(raw, 0) / 10)
    return {
        "reward": reward,
        "raw_return": raw,
        "local_evaluation": 1,
        "eval_valid": 1,
        "cuda": 1,
        "evaluation_episodes": len(returns),
        "frames_per_episode": frames,
        "optimizer_updates": updates,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    parser.add_argument("--source", type=Path, default=Path("/opt/task"))
    parser.add_argument("--task", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/logs/artifacts/policy.msgpack"))
    args = parser.parse_args()
    metrics = execute(args.source, args.task, args.checkpoint, args.mode == "train")
    if args.mode == "verify":
        output = Path("/logs/verifier/reward.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
