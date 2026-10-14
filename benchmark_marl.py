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


def memory_gb():
    stats = jax.devices()[0].memory_stats() or {}
    return stats.get('peak_bytes_in_use', 0) / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num_envs', default=[64, 256, 1024], type=int, nargs='+')
    parser.add_argument('--num_agents', default=4, type=int)
    parser.add_argument('--num_steps', default=128, type=int)
    parser.add_argument('--skip_update', action='store_true', help='only benchmark the environment')
    args = parser.parse_args()

    print('JAX', jax.__version__, 'devices:', jax.devices(), flush=True)
    env_cfg = EnvConfig(num_agents=args.num_agents)
    rng = np.random.default_rng(0)

    for E in args.num_envs:
        train_cfg = TrainConfig(num_envs=E, num_steps=args.num_steps, num_random_maps=64)
        fixed, random, _, _, _ = make_training_maps(env_cfg, train_cfg, rng)
        bank = concat_banks([fixed, random])
        params = EnvParams(bank=bank, map_logits=jnp.zeros(bank.num_maps), goal_coverage=jnp.float32(0.99))
        algo = MAPPO(env_cfg, train_cfg)
        runner = algo.init(jax.random.PRNGKey(0), params)

        # Environment only
        step = jax.jit(jax.vmap(algo.env.step, in_axes=(0, 0, 0, None)))
        actions = jnp.zeros((E, args.num_agents, env_cfg.action_dim))
        keys = jax.random.split(jax.random.PRNGKey(1), E)
        _, dt = timed(step, keys, runner.env_state, actions, params, repeats=10)
        print(f'num_envs {E:6d}  env step {1000 * dt:8.2f} ms  '
              f'{E / dt:12,.0f} env steps/s  {E * args.num_agents / dt:12,.0f} agent steps/s', flush=True)

        # Full MAPPO update
        if not args.skip_update:
            t = time.perf_counter()
            runner, metrics, _ = algo.update(runner, params)
            jax.block_until_ready(metrics)
            compile_time = time.perf_counter() - t
            t = time.perf_counter()
            for _ in range(3):
                runner, metrics, _ = algo.update(runner, params)
            jax.block_until_ready(metrics)
            dt = (time.perf_counter() - t) / 3
            steps = E * args.num_steps
            print(f'num_envs {E:6d}  update {dt:8.2f} s (first call incl. compilation {compile_time:.0f} s)  '
                  f'{steps / dt:12,.0f} env steps/s with training  peak memory {memory_gb():.1f} GB', flush=True)


if __name__ == '__main__':
    main()
