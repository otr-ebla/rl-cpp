"""
Measures the speed of the multi-agent environment and of a full MAPPO update
(rollout + training) on the default JAX device, to choose num_envs for the
available GPU.

Examples:
  python benchmark_marl.py
  python benchmark_marl.py --num_envs 256 1024 4096 --num_agents 4
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from marl.config import EnvConfig, TrainConfig
from marl.env import EnvParams
from marl.maps import concat_banks
from marl.mappo import MAPPO, make_training_maps


def timed(f, *args, repeats=3):
    out = f(*args)
    jax.block_until_ready(out)
    t = time.perf_counter()
    for _ in range(repeats):
        out = f(*args)
        jax.block_until_ready(out)
    return out, (time.perf_counter() - t) / repeats


def memory_gb(key='peak_bytes_in_use'):
    stats = jax.devices()[0].memory_stats() or {}
    return stats.get(key, 0) / 1e9


def benchmark(env_cfg, E, num_steps, skip_update, rng):
    train_cfg = TrainConfig(num_envs=E, num_steps=num_steps, num_random_maps=64)
    fixed, random, _, _, _ = make_training_maps(env_cfg, train_cfg, rng)
    bank = concat_banks([fixed, random])
    params = EnvParams(bank=bank, map_logits=jnp.zeros(bank.num_maps), goal_coverage=jnp.float32(0.99))
    algo = MAPPO(env_cfg, train_cfg)
    runner = algo.init(jax.random.PRNGKey(0), params)

    # Environment only
    step = jax.jit(jax.vmap(algo.env.step, in_axes=(0, 0, 0, None)))
    actions = jnp.zeros((E, env_cfg.num_agents, env_cfg.action_dim))
    keys = jax.random.split(jax.random.PRNGKey(1), E)
    _, dt = timed(step, keys, runner.env_state, actions, params, repeats=10)
    print(f'num_envs {E:6d}  env step {1000 * dt:8.2f} ms  '
          f'{E / dt:12,.0f} env steps/s  {E * env_cfg.num_agents / dt:12,.0f} agent steps/s', flush=True)
    if skip_update:
        return

    # Full MAPPO update (rollout of num_steps steps + training)
    t = time.perf_counter()
    runner, metrics, _ = algo.update(runner, params)
    jax.block_until_ready(metrics)
    compile_time = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(3):
        runner, metrics, _ = algo.update(runner, params)
    jax.block_until_ready(metrics)
    dt = (time.perf_counter() - t) / 3
    print(f'num_envs {E:6d}  update {dt:8.2f} s (first call incl. compilation {compile_time:.0f} s)  '
          f'{E * num_steps / dt:12,.0f} env steps/s with training  peak memory {memory_gb():.1f} GB', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num_envs', default=[64, 256, 512, 1024], type=int, nargs='+')
    parser.add_argument('--num_agents', default=4, type=int)
    parser.add_argument('--num_steps', default=128, type=int)
    parser.add_argument('--skip_update', action='store_true', help='only benchmark the environment')
    args = parser.parse_args()

    print('JAX', jax.__version__, 'devices:', jax.devices(), flush=True)
    env_cfg = EnvConfig(num_agents=args.num_agents)
    rng = np.random.default_rng(0)

    print(f'GPU memory available to JAX: {memory_gb("bytes_limit"):.1f} GB', flush=True)
    for E in args.num_envs:
        try:
            benchmark(env_cfg, E, args.num_steps, args.skip_update, rng)
        except jax.errors.JaxRuntimeError as e:
            if 'RESOURCE_EXHAUSTED' not in str(e) and 'Out of memory' not in str(e):
                raise
            print(f'num_envs {E:6d}  out of GPU memory, use fewer environments (or a smaller --num_steps)', flush=True)
            break


if __name__ == '__main__':
    main()
