"""
Evaluate a MAPPO checkpoint on the evaluation maps.

All (map, episode) pairs run in parallel. Writes a CSV with the metrics of
each episode and an image of the paths of the robots for each episode.

Examples:
  python eval_marl.py --load experiments/marl_4robots/checkpoint
  python eval_marl.py --load experiments/marl_4robots/checkpoint --num_agents 8 --episodes 3
  python eval_marl.py --load experiments/marl_4robots/checkpoint --maps "maps/giant_mowing_*.png"

The policy is shared by all robots and only uses each robot's own
observation, so it can be evaluated with a different number of robots than
it was trained with.
"""
import argparse
import csv
import glob
import os
from argparse import BooleanOptionalAction

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from marl.config import EnvConfig
from marl.env import EnvParams, MultiMowerEnv
from marl.maps import build_bank, canvas_size_for, eval_map_files, load_map
from marl.mappo import actor_apply, load_checkpoint, make_networks

COLORS = [(228, 26, 28), (55, 126, 184), (255, 127, 0), (152, 78, 163),
          (166, 86, 40), (247, 129, 191), (77, 175, 74), (0, 160, 160)]


def run_episodes(env, actor, actor_params, env_params, map_ids, key, max_steps, deterministic):
    """
    Runs one episode per entry of map_ids in parallel.
    Returns per step positions (T, E, N, 2), coverage (T, E), done (T, E) and infos.
    """
    E = len(map_ids)
    K = env_params.bank.num_maps
    logits = jnp.where(jnp.arange(K)[None, :] == jnp.asarray(map_ids)[:, None], 0.0, -jnp.inf)
    per_env = EnvParams(bank=None, map_logits=0, goal_coverage=None)
    reset = jax.vmap(env.reset, in_axes=(0, per_env))
    step = jax.vmap(env.step, in_axes=(0, 0, 0, per_env))
    params = env_params.replace(map_logits=logits)
    N = env.cfg.num_agents

    @jax.jit
    def rollout(key):
        k_reset, k_steps = jax.random.split(key)
        obs, state = reset(jax.random.split(k_reset, E), params)

        def body(carry, k):
            obs, state = carry
            k_act, k_env = jax.random.split(k)
            mean, log_std = actor_apply(actor, actor_params, obs)
            action = mean if deterministic else mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
            pos = state.pos
            obs, state, _, done, info = step(jax.random.split(k_env, E), state, action.reshape(E, N, -1), params)
            coverage = info['coverage']
            return (obs, state), (pos, coverage, done, info)

        _, out = jax.lax.scan(body, (obs, state), jax.random.split(k_steps, max_steps))
        return out

    pos, coverage, done, info = jax.device_get(rollout(key))
    return pos, coverage, done, info


def draw_paths(filename, known, unknown, paths, radius_p, scale=2):
    """paths: list of (T, 2) positions in pixels, one per robot"""
    size = known.shape[0]
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[unknown] = 128
    img[known] = 0
    # display with x to the right and y up, as in MowerEnv
    img = np.ascontiguousarray(np.flip(img.transpose(1, 0, 2), axis=0))
    image = Image.fromarray(img).resize((size * scale, size * scale), Image.NEAREST).convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for n, path in enumerate(paths):
        if len(path) < 2:
            continue
        pts = [(scale * p[0], scale * (size - p[1])) for p in path]
        color = COLORS[n % len(COLORS)]
        draw.line(pts, fill=color + (70,), width=max(1, int(2 * radius_p * scale)))
        draw.line(pts, fill=color + (255,), width=1)
        x, y = pts[0]
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(0, 0, 0, 255), fill=color + (255,))
    Image.alpha_composite(image, overlay).convert('RGB').save(filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--load', required=True, type=str, help='checkpoint directory')
    parser.add_argument('--maps', default=None, type=str, help='glob of map images (default: mowing evaluation maps)')
    parser.add_argument('--num_agents', default=None, type=int, help='default: as in training')
    parser.add_argument('--comm_range', default=None, type=float,
                        help='communication range [m], 0: unlimited (default: as in training)')
    parser.add_argument('--episodes', default=1, type=int, help='episodes per map')
    parser.add_argument('--max_steps', default=5000, type=int)
    parser.add_argument('--goal_coverage', default=0.99, type=float)
    parser.add_argument('--deterministic', default=True, action=BooleanOptionalAction)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out', default=None, type=str, help='output directory (default: <load>/eval)')
    args = parser.parse_args()

    params, env_cfg, train_cfg = load_checkpoint(args.load)
    files = sorted(glob.glob(args.maps)) if args.maps else eval_map_files()
    assert len(files) > 0, 'No maps found'
    maps = [load_map(f) for f in files]
    overrides = env_cfg.to_dict()
    overrides.update(
        canvas_size=canvas_size_for([m[0].shape[0] for m in maps], env_cfg),
        max_episode_steps=args.max_steps,
        max_non_new_steps=args.max_steps)
    if args.num_agents is not None:
        overrides['num_agents'] = args.num_agents
    if args.comm_range is not None:
        overrides['comm_range'] = args.comm_range
    env_cfg = EnvConfig.from_dict(overrides)
    out_dir = args.out or os.path.join(args.load, 'eval')
    os.makedirs(out_dir, exist_ok=True)

    env = MultiMowerEnv(env_cfg)
    actor, _ = make_networks(env_cfg, train_cfg)
    bank = build_bank(maps, env_cfg)
    env_params = EnvParams(bank=bank, map_logits=jnp.zeros(len(maps)), goal_coverage=jnp.float32(args.goal_coverage))
    map_ids = np.repeat(np.arange(len(maps)), args.episodes)
    print(f'{len(files)} maps x {args.episodes} episodes, {env_cfg.num_agents} robots, canvas {env_cfg.canvas_size}', flush=True)

    pos, coverage, done, info = run_episodes(
        env, actor, params['actor'], env_params, map_ids, jax.random.PRNGKey(args.seed),
        args.max_steps, args.deterministic)

    rows = []
    dt = env_cfg.step_size
    for e, m in enumerate(map_ids):
        ends = np.nonzero(done[:, e])[0]
        T = ends[0] + 1 if len(ends) > 0 else args.max_steps
        cov = coverage[:T, e]

        def time_to(c):
            idx = np.nonzero(cov >= c)[0]
            return (idx[0] + 1) * dt if len(idx) > 0 else float('nan')

        row = dict(map=os.path.basename(files[m]), episode=e % args.episodes, steps=int(T),
                   time=T * dt, coverage=float(cov[-1]), time_90=time_to(0.9),
                   time_95=time_to(0.95), time_99=time_to(0.99),
                   overlap=float(info['overlap'][T - 1, e]),
                   collisions=int(info['collisions'][T - 1, e]))
        rows.append(row)
        print('  '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in row.items()), flush=True)
        known, unknown = maps[m]
        paths = [pos[:T, e, n] * env_cfg.pixels_per_meter for n in range(env_cfg.num_agents)]
        name = os.path.splitext(row['map'])[0]
        draw_paths(os.path.join(out_dir, f'{name}_ep{row["episode"]}.png'), known, unknown, paths, env_cfg.radius_p)

    with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for k in ['coverage', 'time_90', 'time_99', 'overlap', 'collisions']:
        print(f'mean {k}: {np.nanmean([r[k] for r in rows]):.3f}')
    print('results in', out_dir)


if __name__ == '__main__':
    main()
