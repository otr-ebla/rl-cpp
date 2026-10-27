"""
Corridor test: one robot with a trained policy in a single long, narrow,
horizontal corridor, longer than the region covered by the largest scale of
the multi-scale maps (input_size * scale_factor^(num_maps-1) pixels, 76.8 m
with the default parameters, i.e. +-38.4 m around the robot along the
corridor when the robot is aligned with it).

The robot starts inside the corridor (the random spawn of the environment
almost never hits such a thin free region), covers it towards one end and
then has to come back over already covered ground: once the uncovered part
is farther away than the largest scale, nothing in the observation tells the
robot where to go. The test records, at every step, the position of the
robot, the coverage, the steps without new coverage and the distance along
the corridor to the nearest part that is still uncovered.

The policy is the shared actor of a MAPPO/IPPO checkpoint, run with
num_agents = 1 (no other robots in the observation), on the JAX device.

Writes to --out (default results/corridor_marl/<test name>):
  corridor.pdf    position along the corridor, covered extent and field of
                  view of the largest scale over time, coverage, distance to
                  the uncovered part, path
  steps.csv       the per step data shown in the PDF (dist_* and covered_*
                  in meters along the corridor, NaN: no uncovered part)
  summary.json    configuration and summary values
and the map to maps/corridor_<length>m_<width>m.png, so that the episode can
be watched with play_marl.py --start (see the end of the run).

With --precover the robot is dropped into a part of the corridor that is
already covered, e.g. covered from the left end to 100 m, start at 50 m
heading to the left (dead end), the uncovered part is 50 m away to the right.

Examples:
  python corridor_test_marl.py --load experiments/seeds/mappo_s0/checkpoint --length 120 --width 0.45
  python corridor_test_marl.py --width 0.45 --precover 0 100 --start 0.4167 --heading 180
"""
import argparse
import csv
import json
import math
import os
from argparse import BooleanOptionalAction

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from marl.config import EnvConfig
from marl.env import EnvParams, MultiMowerEnv
from marl.maps import build_bank, canvas_size_for, save_map
from marl.mappo import actor_apply, load_checkpoint, make_networks


def corridor_map(length_m, width_m, cfg: EnvConfig, wall_p=8):
    """
    Square map (side: corridor length + walls, rounded up to a multiple of 64
    pixels) that is obstacle everywhere except one horizontal corridor through
    the middle. Returns (known, unknown) indexed [x, y] and the corridor
    extent (x0, x1, y0, y1) in pixels.
    """
    ppm = cfg.pixels_per_meter
    length_p = int(round(length_m * ppm))
    width_p = int(round(width_m * ppm))
    side = int(math.ceil((length_p + 2 * wall_p) / 64) * 64)
    x0 = (side - length_p) // 2
    y0 = (side - width_p) // 2
    known = np.ones((side, side), dtype=bool)
    known[x0:x0 + length_p, y0:y0 + width_p] = False
    return known, np.zeros_like(known), (x0, x0 + length_p, y0, y0 + width_p)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--load', default='experiments/seeds/mappo_s0/checkpoint', type=str, help='checkpoint directory')
    parser.add_argument('--length', default=120, type=float, help='corridor length [m]')
    parser.add_argument('--width', default=0.6, type=float, help='corridor width [m]')
    parser.add_argument('--start', default=0.5, type=float, help='start position along the corridor (0 left end, 1 right end)')
    parser.add_argument('--heading', default=0, type=float, help='start heading [degrees], 0 = along the corridor to the right')
    parser.add_argument('--precover', nargs=2, default=None, type=float, metavar=('FROM', 'TO'),
                        help='part of the corridor that is already covered at the start [m from the left end]')
    parser.add_argument('--max_steps', default=10000, type=int)
    parser.add_argument('--goal_coverage', default=0.99, type=float)
    parser.add_argument('--uncovered_fraction', default=0.5, type=float,
                        help='a column of the corridor counts as uncovered when at least this fraction of its free pixels is not covered')
    parser.add_argument('--deterministic', default=True, action=BooleanOptionalAction)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out', default=None, type=str)
    args = parser.parse_args()

    params, env_cfg, train_cfg = load_checkpoint(args.load)
    known, unknown, (x0, x1, y0, y1) = corridor_map(args.length, args.width, env_cfg)
    name = f'corridor_{args.length:g}m_{args.width:g}m'
    map_file = os.path.join('maps', name + '.png')
    if args.precover:
        name += f'_precover{args.precover[0]:g}-{args.precover[1]:g}'
    name += f'_start{args.start:g}_heading{args.heading:g}'
    save_map(map_file, known, unknown)
    overrides = env_cfg.to_dict()
    overrides.update(
        num_agents=1,
        canvas_size=canvas_size_for([known.shape[0]], env_cfg),
        max_episode_steps=args.max_steps,
        max_non_new_steps=args.max_steps)
    env_cfg = EnvConfig.from_dict(overrides)
    ppm, mpp = env_cfg.pixels_per_meter, env_cfg.meters_per_pixel
    reach_m = env_cfg.input_size * env_cfg.scale_factor ** (env_cfg.num_maps - 1) * mpp
    out_dir = args.out or os.path.join('results', 'corridor_marl', name)
    os.makedirs(out_dir, exist_ok=True)

    start_pos = np.array([[(x0 + args.start * (x1 - x0)) * mpp, (y0 + y1) / 2 * mpp]], dtype=np.float32)
    start_pos[0, 0] = np.clip(start_pos[0, 0], x0 * mpp + 2 * env_cfg.mower_radius, x1 * mpp - 2 * env_cfg.mower_radius)
    start_heading = np.array([math.radians(args.heading)], dtype=np.float32)

    env = MultiMowerEnv(env_cfg)
    # fixed start pose instead of the random spawn
    env._spawn = lambda key, blocked, size: (jnp.asarray(start_pos), jnp.asarray(start_heading))
    actor, _ = make_networks(env_cfg, train_cfg)
    bank = build_bank([(known, unknown)], env_cfg)
    env_params = EnvParams(bank=bank, map_logits=jnp.zeros(1), goal_coverage=jnp.float32(args.goal_coverage))
    print(f'{map_file}: {known.shape[0]} px = {known.shape[0] * mpp:g} m, corridor {(x1 - x0) * mpp:g} x {(y1 - y0) * mpp:g} m, '
          f'largest scale {reach_m:g} m, canvas {env_cfg.canvas_size}, JAX device {jax.devices()[0]}', flush=True)

    S = env_cfg.canvas_size
    x_col = (jnp.arange(S) + 0.5) * mpp
    precover = np.zeros((S, S), dtype=bool)
    if args.precover:
        a, b = (x0 + np.clip(np.array(args.precover) * ppm, 0, x1 - x0)).astype(int)
        precover[a:b, :] = True
    free_col = jnp.sum(~bank.dilated[0], axis=1)  # free pixels per column x
    corridor_col = free_col > 0

    def column_stats(state):
        """Distance along the corridor to the nearest uncovered column, covered extent"""
        uncovered = jnp.sum(~state.dilated & (state.overlap == 0), axis=1)
        is_unc = corridor_col & (uncovered >= args.uncovered_fraction * free_col)
        is_cov = corridor_col & (uncovered < free_col)
        x = state.pos[0, 0]
        d = jnp.abs(x_col - x)
        dist = jnp.min(jnp.where(is_unc, d, jnp.inf))
        left = jnp.min(jnp.where(is_unc & (x_col < x), d, jnp.inf))
        right = jnp.min(jnp.where(is_unc & (x_col > x), d, jnp.inf))
        cov_lo = jnp.min(jnp.where(is_cov, x_col, jnp.inf))
        cov_hi = jnp.max(jnp.where(is_cov, x_col, -jnp.inf))
        return dist, left, right, cov_lo, cov_hi

    @jax.jit
    def rollout(key):
        k_reset, k_steps = jax.random.split(key)
        obs, state = env.reset(k_reset, env_params)
        if args.precover:
            overlap = jnp.maximum(state.overlap, jnp.asarray(precover, dtype=state.overlap.dtype))
            state = state.replace(overlap=overlap, coverage_px=jnp.sum((overlap > 0) & ~state.dilated).astype(jnp.float32))
            obs = env._observation(k_reset, state)

        def body(carry, k):
            obs, state, finished = carry
            k_act, k_step, k_obs = jax.random.split(k, 3)
            mean, log_std = actor_apply(actor, params['actor'], obs)
            action = mean if args.deterministic else mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
            # no automatic reset, the steps after the end of the episode are discarded
            state, _, done, info = env._step_state(k_step, state, action.reshape(1, -1))
            obs = env._observation(k_obs, state)
            out = dict(x=state.pos[0, 0], y=state.pos[0, 1], heading=state.heading[0],
                       coverage=info['coverage'], non_new_steps=state.non_new_steps,
                       collisions=state.collisions[0], done=done | finished)
            out.update(zip(['dist_uncovered', 'dist_left', 'dist_right', 'covered_lo', 'covered_hi'], column_stats(state)))
            return (obs, state, finished | done), out

        _, out = jax.lax.scan(body, (obs, state, jnp.bool_(False)), jax.random.split(k_steps, args.max_steps))
        return out

    out = {k: np.asarray(v) for k, v in jax.device_get(rollout(jax.random.PRNGKey(args.seed))).items()}
    ends = np.nonzero(out['done'])[0]
    T = ends[0] + 1 if len(ends) > 0 else args.max_steps
    out = {k: v[:T] for k, v in out.items() if k != 'done'}
    dt = env_cfg.step_size
    t = np.arange(1, T + 1) * dt
    xc = x0 * mpp  # positions relative to the left end of the corridor
    rows = dict(step=np.arange(1, T + 1), time=t, x=out['x'] - xc, y=out['y'] - y0 * mpp,
                heading_deg=np.degrees(out['heading']), coverage=out['coverage'],
                non_new_steps=out['non_new_steps'], collisions=out['collisions'],
                dist_uncovered=out['dist_uncovered'], dist_uncovered_left=out['dist_left'],
                dist_uncovered_right=out['dist_right'],
                covered_lo=out['covered_lo'] - xc, covered_hi=out['covered_hi'] - xc)
    rows = {k: np.where(np.isfinite(v), v, np.nan) if v.dtype.kind == 'f' else v for k, v in rows.items()}
    # steps in a row without new coverage of the free space (non_new_steps of the
    # environment is also reset by new pixels in the dilated band along the walls)
    stalled = np.zeros(T, dtype=np.int64)
    for i in range(1, T):
        stalled[i] = 0 if rows['coverage'][i] > rows['coverage'][i - 1] else stalled[i - 1] + 1
    rows['steps_without_new_coverage'] = stalled
    with open(os.path.join(out_dir, 'steps.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(rows.keys()))
        writer.writerows(zip(*[v.tolist() for v in rows.values()]))

    # summary
    def time_to(c):
        idx = np.nonzero(rows['coverage'] >= c)[0]
        return float(t[idx[0]]) if len(idx) > 0 else None

    beyond = rows['dist_uncovered'] > reach_m / 2
    summary = dict(
        checkpoint=args.load, map=map_file, corridor_length_m=(x1 - x0) * mpp, corridor_width_m=(y1 - y0) * mpp,
        largest_scale_m=reach_m, start_x_m=float(start_pos[0, 0] - xc), start_heading_deg=args.heading,
        precover_m=args.precover,
        deterministic=args.deterministic, seed=args.seed, steps=int(T), time_s=float(T * dt),
        success=bool(rows['coverage'][-1] >= args.goal_coverage), final_coverage=float(rows['coverage'][-1]),
        time_50=time_to(0.5), time_90=time_to(0.9), time_95=time_to(0.95), time_99=time_to(0.99),
        collisions=int(rows['collisions'][-1]),
        max_time_without_new_coverage_s=float(stalled.max() * dt),
        steps_uncovered_beyond_largest_scale=int(np.sum(beyond)),
        max_dist_uncovered_m=float(np.nanmax(rows['dist_uncovered'])) if np.any(np.isfinite(rows['dist_uncovered'])) else None,
        path_length_m=float(np.sum(np.hypot(np.diff(rows['x']), np.diff(rows['y'])))),
        reversals=int(np.sum(np.abs(np.diff(np.sign(np.diff(rows['x'])[np.abs(np.diff(rows['x'])) > 1e-3]))) > 0)))
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    for k, v in summary.items():
        print(f'  {k}: {v:.3f}' if isinstance(v, float) else f'  {k}: {v}')

    # figure
    L = (x1 - x0) * mpp
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=False,
                             gridspec_kw=dict(height_ratios=[3, 1.4, 1.6, 1.2]))
    ax = axes[0]
    ax.fill_between(t, rows['x'] - reach_m / 2, rows['x'] + reach_m / 2, color='tab:blue', alpha=0.10, lw=0,
                    label=f'largest scale along the corridor (±{reach_m / 2:g} m)')
    ax.fill_between(t, rows['covered_lo'], rows['covered_hi'], color='tab:green', alpha=0.35, lw=0,
                    label='covered extent')
    ax.plot(t, rows['x'], color='black', lw=0.9, label='robot')
    ax.axhline(0, color='0.3', lw=1.5)
    ax.axhline(L, color='0.3', lw=1.5)
    ax.set_ylim(-0.05 * L, 1.05 * L)
    ax.set_ylabel('position along the corridor [m]')
    title = f'{os.path.basename(os.path.dirname(os.path.abspath(args.load)))}, 1 robot, corridor {L:g} m x {(y1 - y0) * mpp:g} m'
    if args.precover:
        title += f', covered at the start: {args.precover[0]:g}-{args.precover[1]:g} m'
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)

    ax = axes[1]
    ax.plot(t, 100 * rows['coverage'], color='tab:green')
    ax.set_ylabel('coverage [%]')
    ax.set_ylim(0, 102)

    ax = axes[2]
    ax.plot(t, rows['dist_uncovered'], color='tab:red', label='distance to the nearest uncovered part')
    ax.axhline(reach_m / 2, color='tab:blue', ls='--', lw=1, label=f'half of the largest scale ({reach_m / 2:g} m)')
    ax.set_ylabel('distance [m]')
    ax2 = ax.twinx()
    ax2.plot(t, stalled * dt, color='0.5', lw=0.8, label='time without new coverage')
    ax2.set_ylabel('time without new coverage [s]', color='0.4')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)
    for a in axes[:3]:
        a.set_xlim(0, t[-1])
        a.set_xlabel('time [s]')

    ax = axes[3]
    W = (y1 - y0) * mpp
    ax.fill_between([0, L], 0, W, color='0.92', lw=0)
    sc = ax.scatter(rows['x'], rows['y'], c=t, s=1, cmap='viridis')
    ax.plot(rows['x'][0], rows['y'][0], 'o', color='tab:red', ms=5, label='start')
    ax.set_xlim(-0.01 * L, 1.01 * L)
    ax.set_ylim(0, W)
    ax.set_xlabel('position along the corridor [m]')
    ax.set_ylabel('across [m]')
    ax.set_title('path (vertical axis stretched)', fontsize=9)
    fig.colorbar(sc, ax=ax, label='time [s]', pad=0.01)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'corridor.pdf'))
    print('results in', out_dir)
    print(f'watch it: python play_marl.py --load {args.load} --maps {map_file} --num_agents 1 '
          f'--max_steps {args.max_steps} --start {start_pos[0, 0]:.3f} {start_pos[0, 1]:.3f} {args.heading:g}')


if __name__ == '__main__':
    main()
