"""
Corridor test of the original single-agent policy (rlm.mower_env.MowerEnv,
stable-baselines3, e.g. weights/mowing_tv1): the robot is dropped in the
middle of a single long, narrow, horizontal corridor, longer than twice the
region covered by the largest scale of the multi-scale maps
(input_size * scale_factor^(num_maps-1) pixels, 76.8 m with the default
parameters, i.e. +-38.4 m around the robot along the corridor when the robot
is aligned with it). Once one side is covered, the uncovered side is out of
view: nothing in the observation tells the robot where to go.

The start pose is fixed (the random spawn of the environment would put the
robot anywhere along the corridor). Evaluation settings as in eval.py
(deterministic actions, flip when stuck, max_non_new_steps).

Writes to --out (default results/corridor/<test name>):
  corridor.pdf    position along the corridor, covered extent and field of
                  view of the largest scale over time, coverage, distance to
                  the uncovered part, path
  steps.csv       the per step data shown in the PDF (dist_* and covered_*
                  in meters along the corridor, NaN: no uncovered part)
  summary.json    configuration and summary values
and the map to maps/corridor_<length>m_<width>m.png.

Runs in the environment of requirements.txt (conda env rlm), the policy on
the GPU if available. Example:
  python corridor_test.py --load weights/mowing_tv1 --length 160 --width 1.5
"""
import argparse
import csv
import importlib
import json
import math
import os
import time
from argparse import BooleanOptionalAction

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rlm.mower_env import MowerEnv


def corridor_map(length_m, width_m, meters_per_pixel, wall_p=8):
    """
    Square map (side: corridor length + walls, rounded up to a multiple of 64
    pixels) that is obstacle everywhere except one horizontal corridor through
    the middle. Returns the image as indexed by MowerEnv ([x, y], 255 free,
    0 obstacle) and the corridor extent (x0, x1, y0, y1) in pixels.
    """
    length_p = int(round(length_m / meters_per_pixel))
    width_p = int(round(width_m / meters_per_pixel))
    side = int(math.ceil((length_p + 2 * wall_p) / 64) * 64)
    x0 = (side - length_p) // 2
    y0 = (side - width_p) // 2
    img = np.zeros((side, side), dtype=np.uint8)
    img[x0:x0 + length_p, y0:y0 + width_p] = 255
    return img, (x0, x0 + length_p, y0, y0 + width_p)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--load', default='weights/mowing_tv1', type=str)
    parser.add_argument('--length', default=160, type=float, help='corridor length [m]')
    parser.add_argument('--width', default=1.5, type=float, help='corridor width [m]')
    parser.add_argument('--start', default=0.5, type=float, help='start position along the corridor (0 left end, 1 right end)')
    parser.add_argument('--heading', default=0, type=float, help='start heading [degrees], 0 = along the corridor to the right')
    parser.add_argument('--steps', default=20000, type=int, help='maximum episode length')
    parser.add_argument('--max_non_new_steps', default=5000, type=int)
    parser.add_argument('--goal_coverage', default=0.99, type=float)
    parser.add_argument('--flip_when_stuck', default=True, action=BooleanOptionalAction)
    parser.add_argument('--uncovered_fraction', default=0.5, type=float,
                        help='a column of the corridor counts as uncovered when at least this fraction of its free pixels is not covered')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--render', default=False, action=BooleanOptionalAction, help='show the pygame window')
    parser.add_argument('--render_mode', default='full', type=str)
    parser.add_argument('--out', default=None, type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)
    with open(os.path.join(args.load, 'agent_parameters.json')) as f:
        agent_args = json.load(f)
    with open(os.path.join(args.load, 'env_parameters.json')) as f:
        env_args = json.load(f)
    mpp = env_args['meters_per_pixel']
    img, (x0, x1, y0, y1) = corridor_map(args.length, args.width, mpp)
    name = f'corridor_{args.length:g}m_{args.width:g}m'
    map_file = os.path.join('maps', name + '.png')
    # undo the transform in MowerEnv._load_map, so that the map is loaded as generated
    cv2.imwrite(map_file, np.fliplr(img).T)
    name += f'_start{args.start:g}_heading{args.heading:g}'
    out_dir = args.out or os.path.join('results', 'corridor', os.path.basename(os.path.normpath(args.load)), name)
    os.makedirs(out_dir, exist_ok=True)

    env_args.update(max_episode_steps=args.steps, max_non_new_steps=args.max_non_new_steps, eval=True,
                    goal_coverage=args.goal_coverage, collision_ends_episode=False,
                    flip_when_stuck=args.flip_when_stuck, verbose=False, metrics_dir=None)
    env = MowerEnv(**env_args)
    env.eval_maps = [map_file]
    reach_m = env.input_size * env.scale_factor ** (env.num_maps - 1) * mpp

    # fixed start pose instead of the random spawn
    start_x = np.clip((x0 + args.start * (x1 - x0)) * mpp, x0 * mpp + 2 * env.mower_radius, x1 * mpp - 2 * env.mower_radius)
    start_m = np.array([start_x, (y0 + y1) / 2 * mpp])
    randomize_pose = env._randomize_pose

    def fixed_pose(mower_radius):
        randomize_pose(mower_radius)
        env.heading = math.radians(args.heading)
        env.noisy_heading = np.random.normal(loc=env.heading, scale=env.heading_noise) % (2 * math.pi)
        env.position_m = start_m.copy()
        env.position_p = env.position_m * env.pixels_per_meter
        env.noisy_position_m = np.random.normal(loc=env.position_m, scale=env.position_noise)
        env.noisy_position_p = env.noisy_position_m * env.pixels_per_meter
    env._randomize_pose = fixed_pose

    algo = getattr(importlib.import_module('stable_baselines3'), agent_args['algo'])
    model = algo.load(os.path.join(args.load, 'agent'), env=env, buffer_size=1000)
    print(f'{map_file}: {img.shape[0]} px = {img.shape[0] * mpp:g} m, corridor {(x1 - x0) * mpp:g} x {(y1 - y0) * mpp:g} m, '
          f'largest scale {reach_m:g} m, policy {args.load} ({agent_args["algo"]}) on {model.device}', flush=True)

    # free (not dilated) pixels of the corridor, as in MowerEnv._initialize_metrics
    obstacles = (img == 0).astype(float)
    obstacles[0, :] = 1; obstacles[-1, :] = 1; obstacles[:, 0] = 1; obstacles[:, -1] = 1
    if env.obstacle_dilation > 1:
        obstacles = cv2.dilate(obstacles, np.ones((env.obstacle_dilation,) * 2), iterations=1)
    free = obstacles[:, y0:y1] == 0  # corridor rows only
    free_col = free.sum(axis=1)
    corridor_col = free_col > 0
    x_col = (np.arange(img.shape[0]) + 0.5) * mpp

    obs = env.reset()
    keys = ['x', 'y', 'heading', 'coverage', 'non_new_steps', 'collisions',
            'dist_uncovered', 'dist_left', 'dist_right', 'covered_lo', 'covered_hi']
    out = {k: [] for k in keys}
    t0 = time.time()
    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if args.render:
            env.render(mode=args.render_mode)
        uncovered = np.sum(free & (env.coverage_map[:, y0:y1] == 0), axis=1)
        is_unc = corridor_col & (uncovered >= args.uncovered_fraction * free_col)
        is_cov = corridor_col & (uncovered < free_col)
        x = env.position_m[0]
        d = np.abs(x_col - x)
        nearest = lambda m: d[m].min() if m.any() else np.nan
        out['x'].append(x)
        out['y'].append(env.position_m[1])
        out['heading'].append(env.heading)
        out['coverage'].append(env.coverage_in_percent)
        out['non_new_steps'].append(env.non_new_steps)
        out['collisions'].append(env.num_collisions)
        out['dist_uncovered'].append(nearest(is_unc))
        out['dist_left'].append(nearest(is_unc & (x_col < x)))
        out['dist_right'].append(nearest(is_unc & (x_col > x)))
        out['covered_lo'].append(x_col[is_cov].min() if is_cov.any() else np.nan)
        out['covered_hi'].append(x_col[is_cov].max() if is_cov.any() else np.nan)
        if (step + 1) % 1000 == 0:
            print(f'  step {step + 1}: coverage {100 * env.coverage_in_percent:.1f}%, x {x - x0 * mpp:.1f} m, '
                  f'{(step + 1) / (time.time() - t0):.0f} steps/s', flush=True)
        if done:
            break
    out = {k: np.array(v, dtype=float) for k, v in out.items()}
    T = len(out['x'])

    dt = env.step_size
    t = np.arange(1, T + 1) * dt
    xc = x0 * mpp  # positions relative to the left end of the corridor
    rows = dict(step=np.arange(1, T + 1), time=t, x=out['x'] - xc, y=out['y'] - y0 * mpp,
                heading_deg=np.degrees(out['heading']), coverage=out['coverage'],
                non_new_steps=out['non_new_steps'].astype(int), collisions=out['collisions'].astype(int),
                dist_uncovered=out['dist_uncovered'], dist_uncovered_left=out['dist_left'],
                dist_uncovered_right=out['dist_right'],
                covered_lo=out['covered_lo'] - xc, covered_hi=out['covered_hi'] - xc)
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

    dx = np.diff(rows['x'])
    moving = np.sign(dx[np.abs(dx) > 1e-3])
    summary = dict(
        policy=args.load, algo=agent_args['algo'], map=map_file, corridor_length_m=(x1 - x0) * mpp,
        corridor_width_m=(y1 - y0) * mpp, largest_scale_m=reach_m, start_x_m=float(start_m[0] - xc),
        start_heading_deg=args.heading, flip_when_stuck=args.flip_when_stuck, seed=args.seed,
        steps=int(T), time_s=float(T * dt), success=bool(rows['coverage'][-1] >= args.goal_coverage),
        final_coverage=float(rows['coverage'][-1]),
        time_50=time_to(0.5), time_90=time_to(0.9), time_95=time_to(0.95), time_99=time_to(0.99),
        collisions=int(rows['collisions'][-1]),
        max_time_without_new_coverage_s=float(stalled.max() * dt),
        steps_uncovered_beyond_largest_scale=int(np.sum(rows['dist_uncovered'] > reach_m / 2)),
        max_dist_uncovered_m=float(np.nanmax(rows['dist_uncovered'])) if np.any(np.isfinite(rows['dist_uncovered'])) else None,
        path_length_m=float(np.sum(np.hypot(np.diff(rows['x']), np.diff(rows['y'])))),
        reversals=int(np.sum(moving[1:] != moving[:-1])))
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    for k, v in summary.items():
        print(f'  {k}: {v:.3f}' if isinstance(v, float) else f'  {k}: {v}')

    # figure
    L = (x1 - x0) * mpp
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), gridspec_kw=dict(height_ratios=[3, 1.4, 1.6, 1.2]))
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
    ax.set_title(f'{args.load} ({agent_args["algo"]}, single agent), corridor {L:g} m x {(y1 - y0) * mpp:g} m')
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
    ax.plot(start_m[0] - xc, start_m[1] - y0 * mpp, 'o', color='tab:red', ms=5, label='start')
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


if __name__ == '__main__':
    main()
