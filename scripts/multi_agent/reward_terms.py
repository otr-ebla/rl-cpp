"""
Size of each term of the reward for a trained policy, to choose the weights
of new reward terms before training with them.

Runs the policy of a checkpoint (stochastic actions, as in training) on the
maps of the last training level (fixed and random maps, goal coverage 0.99)
with the team overlap and workload terms switched on at weight 1, without
training: these terms do not change the actions, so the episodes are the
ones the policy would collect anyway. Prints, for each communication range,
every term of the reward (robot mean) per episode and per step; the other
terms with their weights of the checkpoint. A term at weight w is w times
the value printed at weight 1.

Examples:
  python -m scripts.multi_agent.reward_terms --load experiments/mappo_sb3_v2/checkpoint_best
  python -m scripts.multi_agent.reward_terms --load experiments/mappo_sb3_v2/checkpoint_best --comm_range 0 5 2
"""
import argparse
import csv
import os

import jax
import jax.numpy as jnp
import numpy as np

from marl.config import EnvConfig
from marl.curriculum import Curriculum
from marl.env import REWARD_TERMS, EnvParams, MultiMowerEnv
from marl.maps import concat_banks
from marl.mappo import actor_apply, load_checkpoint, make_networks, make_training_maps

NEW_TERMS = ('team_overlap', 'workload')


def collect(env, actor, actor_params, env_params, num_envs, num_steps, key):
    """Episode statistics (only where done) of num_steps steps of num_envs environments"""
    N = env.cfg.num_agents

    @jax.jit
    def rollout(key):
        k_reset, k_steps = jax.random.split(key)
        obs, state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_reset, num_envs), env_params)

        def body(carry, k):
            obs, state = carry
            k_act, k_env = jax.random.split(k)
            mean, log_std = actor_apply(actor, actor_params, obs)
            action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
            obs, state, _, done, info = env.step_batch(
                jax.random.split(k_env, num_envs), state, action.reshape(num_envs, N, -1), env_params)
            return (obs, state), dict(done=done, terms=info['terms'], length=info['length'],
                                      success=info['success'], coverage=info['coverage'])

        return jax.lax.scan(body, (obs, state), jax.random.split(k_steps, num_steps))[1]

    out = jax.device_get(rollout(key))
    done = out['done'].astype(bool)
    return {k: v[done] for k, v in out.items() if k != 'done'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--load', required=True, type=str, help='checkpoint directory')
    parser.add_argument('--comm_range', default=[0.0, 5.0], type=float, nargs='+',
                        help='communication ranges [m], 0: unlimited')
    parser.add_argument('--num_agents', default=None, type=int, help='default: as in training')
    parser.add_argument('--workload_size', default=64, type=int)
    parser.add_argument('--num_envs', default=256, type=int)
    parser.add_argument('--num_steps', default=4000, type=int, help='steps of each environment')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out', default=None, type=str, help='CSV (default: <load>/reward_terms.csv)')
    args = parser.parse_args()

    params, env_cfg, train_cfg = load_checkpoint(args.load)
    rows = []
    for comm_range in args.comm_range:
        cfg = env_cfg.to_dict()
        cfg.update(comm_range=comm_range, workload_size=args.workload_size, team_overlap_reward_scale=1.0,
                   workload_reward_scale=1.0, shaping_gamma=train_cfg.gamma)
        if args.num_agents is not None:
            cfg['num_agents'] = args.num_agents
        cfg = EnvConfig.from_dict(cfg)
        rng = np.random.default_rng(args.seed)
        fixed, random, groups, floor_plan, obstacles = make_training_maps(cfg, train_cfg, rng)
        bank = concat_banks([b for b in (fixed, random) if b is not None])
        curriculum = Curriculum(groups, floor_plan, obstacles, start_level=1, enabled=False)   # last level
        env_params = EnvParams(bank=bank, map_logits=jnp.asarray(curriculum.map_logits()),
                               goal_coverage=jnp.float32(curriculum.goal_coverage))
        env = MultiMowerEnv(cfg)
        actor, _ = make_networks(cfg, train_cfg)
        ep = collect(env, actor, params['actor'], env_params, args.num_envs, args.num_steps,
                     jax.random.PRNGKey(args.seed))
        n = len(ep['length'])
        steps = ep['length'].sum()
        comm = 'unlimited' if comm_range == 0 else f'{comm_range:g} m'
        print(f'\n=== {cfg.num_agents} robots, communication {comm}: {n} episodes, success {ep["success"].mean():.2f}, '
              f'length {ep["length"].mean():.0f} steps, coverage {ep["coverage"].mean():.3f}')
        print(f'{"term":14s} {"weight":>8s} {"per episode":>12s} {"per step":>10s}')
        for k, name in enumerate(REWARD_TERMS):
            per_episode = ep['terms'][:, k].mean()
            per_step = ep['terms'][:, k].sum() / steps
            weight = '1 (new)' if name in NEW_TERMS else ''
            print(f'{name:14s} {weight:>8s} {per_episode:12.2f} {per_step:10.4f}')
            rows.append(dict(comm_range=comm_range, num_agents=cfg.num_agents, episodes=n, term=name,
                             new=name in NEW_TERMS, per_episode=per_episode, per_step=per_step))
        old = sum(ep['terms'][:, k] for k, name in enumerate(REWARD_TERMS) if name not in NEW_TERMS)
        print(f'{"total (old)":14s} {"":>8s} {old.mean():12.2f} {old.sum() / steps:10.4f}')

    out = args.out or os.path.join(args.load, 'reward_terms.csv')
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nwritten {out}')


if __name__ == '__main__':
    main()
