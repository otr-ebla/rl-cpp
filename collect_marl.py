"""
Collect evaluation data to compare trained multi-agent methods (MAPPO, IPPO,
MAPPO with limited communication, ...) for different numbers of robots, then
draw the comparison dashboard (see dashboard_marl.py).

A method is NAME=CHECKPOINTS, a glob of the checkpoints of the method trained
with different seeds (see train_seeds.py), optionally followed by environment
overrides, e.g. to evaluate a policy with another communication range than it
was trained with:

  python collect_marl.py --maps 'maps/test_large_*.png' --num_agents 2 4 8 \\
      --goal_coverage 0.95 --max_steps 30000 \\
      --run 'MAPPO=experiments/seeds/mappo_s*/checkpoint' \\
      --run 'IPPO=experiments/seeds/ippo_s*/checkpoint' \\
      --run 'MAPPO 5 m=experiments/seeds/mappo_comm5_s*/checkpoint' \\
      --run 'MAPPO tested at 5 m=experiments/seeds/mappo_s*/checkpoint,comm_range=5'

Every policy runs the same episodes: same maps, same start poses (same --seed)
and deterministic actions. The maps form a map set (--map_set, by default the
name of the --maps glob without the number), each map set is analysed on its
own in the dashboard.

Results are merged into <out>/episodes.csv (one row per episode) and
<out>/curves.csv (the time and the overlap when each coverage of 1, 2, ...,
99 % is first reached, one row per episode): rows of a (method, number of
robots, map set) that is collected again are replaced, so methods, seeds and
map sets can be added later. <out>/runs.json keeps the methods (their order
gives their colors in the dashboard) and the settings of each map set. Then
<out>/dashboard.pdf is drawn, with the values it shows in
<out>/dashboard_data/ (see dashboard_marl.py).

Per episode metrics:
  success            reached --goal_coverage within --max_steps
  coverage           final team coverage
  time_90/95/99      time to 90/95/99 % coverage [s] (nan if not reached)
  overlap            final overlap: covered again / covered
  overlap_90/95/99   overlap when 90/95/99 % coverage was first reached
  collisions         all collisions (walls, obstacles, robots) of all robots
  robot_collisions   collisions between robots (counted for each robot involved)
  path_length        mean distance driven per robot [m]
  imbalance          coefficient of variation of the area covered by each robot
                     (0 = every robot covered the same area)
  connectivity       mean fraction of the other robots each robot can
                     communicate with (directly or through other robots)
  belief_recall      mean fraction of the covered area each robot knows of
                     (1 with unlimited communication)
"""
import argparse
import csv
import glob
import json
import os
import re
import time
from argparse import BooleanOptionalAction

# Only allocate the GPU memory that is needed
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import jax.numpy as jnp
import numpy as np

from marl.config import EnvConfig
from marl.env import EnvParams, MultiMowerEnv
from marl.maps import build_bank, canvas_size_for, eval_map_files, load_map
from marl.mappo import actor_apply, load_checkpoint, make_networks

THRESHOLDS = (0.9, 0.95, 0.99)
LEVELS = np.arange(1, 100)  # coverage levels of the curves [%]


def parse_run(spec):
    """'NAME=GLOB[,key=value,...]' -> (name, sorted checkpoint paths, env overrides)"""
    name, rest = spec.split('=', 1)
    pattern, *overrides = rest.split(',')
    paths = sorted(glob.glob(pattern.strip()))
    assert paths, f'no checkpoint matches {pattern}'
    types = {f: type(v) for f, v in EnvConfig().to_dict().items()}
    out = {}
    for o in overrides:
        k, v = o.split('=', 1)
        assert k in types, f'unknown env option {k}'
        out[k] = v.lower() in ('1', 'true', 'yes') if types[k] is bool else types[k](v)
    return name.strip(), paths, out


def make_rollout(env, actor, max_steps):
    """
    Jitted function running E episodes in parallel with deterministic
    actions, without automatic reset: each environment is frozen once its
    episode is done. Returns the final statistics of each episode, and the
    step and overlap when each coverage level (LEVELS) was first reached.
    The actor parameters are an argument, so one compilation serves all seeds.
    """
    cfg = env.cfg
    N = cfg.num_agents
    levels = jnp.asarray(LEVELS / 100, dtype=jnp.float32)

    def step_env(actor_params, key, obs, state):
        """One step of one environment"""
        k_step, k_obs = jax.random.split(key)
        mean, _ = actor_apply(actor, actor_params, obs)
        new_state, _, done, _ = env._step_state(k_step, state, mean.reshape(N, -1))
        return env._observation(k_obs, new_state), new_state, done

    def comm_stats(state):
        """Per step communication statistics of one environment"""
        other = ~jnp.eye(N, dtype=bool)
        conn = jnp.sum(state.connected & other) / max(N * (N - 1), 1)
        if cfg.limited_comm:
            known = jnp.sum(state.belief_overlap > 0, axis=(1, 2))
            recall = jnp.mean(known / jnp.maximum(jnp.sum(state.overlap > 0), 1))
        else:
            recall = jnp.ones(())
        return conn, recall

    vstep = jax.vmap(step_env, in_axes=(None, 0, 0, 0))
    vstats = jax.vmap(comm_stats)

    @jax.jit
    def rollout(actor_params, keys, env_params, map_ids):
        """keys: (E, 2), one per episode, map_ids: (E,)"""
        E = map_ids.shape[0]
        K = env_params.bank.num_maps
        logits = jnp.where(jnp.arange(K)[None, :] == map_ids[:, None], 0.0, -jnp.inf)
        per_env = EnvParams(bank=None, map_logits=0, goal_coverage=None)
        params = env_params.replace(map_logits=logits)
        keys = jax.vmap(jax.random.split)(keys)                             # E x 2 x 2
        obs, state = jax.vmap(env.reset, in_axes=(0, per_env))(keys[:, 0], params)
        nan = jnp.full((E, len(LEVELS)), jnp.nan)
        carry = dict(
            obs=obs, state=state, key=keys[:, 1], t=jnp.zeros((), jnp.int32),
            active=jnp.ones(E, bool), steps=jnp.zeros(E, jnp.int32),
            path=jnp.zeros((E, N)), conn=jnp.zeros(E), recall=jnp.zeros(E),
            level_step=nan, level_overlap=nan)

        def cond(c):
            return (c['t'] < max_steps) & jnp.any(c['active'])

        def body(c):
            keys = jax.vmap(jax.random.split)(c['key'])                     # E x 2 x 2
            key, k = keys[:, 0], keys[:, 1]
            obs, state, done = vstep(actor_params, k, c['obs'], c['state'])
            act = c['active']
            keep = lambda new, old: jnp.where(act.reshape((E,) + (1,) * (new.ndim - 1)), new, old)
            state_k = jax.tree_util.tree_map(keep, state, c['state'])
            obs_k = jax.tree_util.tree_map(keep, obs, c['obs'])
            conn, recall = vstats(state_k)
            moved = jnp.linalg.norm(state_k.pos - c['state'].pos, axis=-1)
            coverage = state_k.coverage_px / state_k.free_space
            overlap = state_k.overlap_px / jnp.maximum(state_k.coverage_px, 1)
            first = jnp.isnan(c['level_step']) & (coverage[:, None] >= levels[None, :])
            return dict(
                obs=obs_k, state=state_k, key=key, t=c['t'] + 1,
                active=act & ~done, steps=c['steps'] + act,
                path=c['path'] + moved, conn=c['conn'] + jnp.where(act, conn, 0),
                recall=c['recall'] + jnp.where(act, recall, 0),
                level_step=jnp.where(first, (c['t'] + 1).astype(jnp.float32), c['level_step']),
                level_overlap=jnp.where(first, overlap[:, None], c['level_overlap']))

        c = jax.lax.while_loop(cond, body, carry)
        s = c['state']
        steps = jnp.maximum(c['steps'], 1)
        return dict(
            steps=c['steps'], coverage=s.coverage_px / s.free_space,
            overlap=s.overlap_px / jnp.maximum(s.coverage_px, 1),
            collisions=jnp.sum(s.collisions, axis=1), robot_collisions=jnp.sum(s.robot_collisions, axis=1),
            path=c['path'], covered_by=s.covered_by,
            connectivity=c['conn'] / steps, belief_recall=c['recall'] / steps,
            level_step=c['level_step'], level_overlap=c['level_overlap'])

    return rollout


class Evaluator:
    """Runs the episodes of one map set for the checkpoints of the methods"""

    def __init__(self, maps, map_names, map_set, args):
        self.maps, self.map_names, self.map_set, self.args = maps, map_names, map_set, args
        self.map_sizes = [m[0].shape[0] for m in maps]
        self.env_batch = args.env_batch
        self._rollouts = {}
        self._banks = {}

    def env_config(self, env_cfg, overrides, num_agents):
        cfg = env_cfg.to_dict()
        cfg.update(overrides)
        cfg.update(canvas_size=canvas_size_for(self.map_sizes, env_cfg), num_agents=num_agents,
                   max_episode_steps=self.args.max_steps, max_non_new_steps=self.args.max_steps)
        return EnvConfig.from_dict(cfg)

    def rollout(self, env_cfg, train_cfg):
        """Compiled once per environment and network configuration"""
        net = (train_cfg.hidden_dims, train_cfg.grouped_convs, train_cfg.squash_mean, train_cfg.min_log_std,
               train_cfg.init_log_std, train_cfg.neighbor_dims, train_cfg.neighbor_layers, train_cfg.neighbor_heads)
        k = (env_cfg, net)
        if k not in self._rollouts:
            actor, _ = make_networks(env_cfg, train_cfg)
            self._rollouts[k] = make_rollout(MultiMowerEnv(env_cfg), actor, self.args.max_steps)
        return self._rollouts[k]

    def env_params(self, env_cfg):
        if env_cfg not in self._banks:
            bank = build_bank(self.maps, env_cfg)
            self._banks[env_cfg] = EnvParams(bank=bank, map_logits=jnp.zeros(len(self.maps)),
                                             goal_coverage=jnp.float32(self.args.goal_coverage))
        return self._banks[env_cfg]

    def run(self, name, path, overrides, num_agents):
        """Rows of episodes.csv and curves.csv of one checkpoint"""
        args = self.args
        params, env_cfg, train_cfg = load_checkpoint(path)
        env_cfg = self.env_config(env_cfg, overrides, num_agents)
        rollout = self.rollout(env_cfg, train_cfg)
        env_params = self.env_params(env_cfg)

        map_ids = np.repeat(np.arange(len(self.maps)), args.episodes)
        episodes = np.tile(np.arange(args.episodes), len(self.maps))
        # one key per episode, the same for every method, seed and batch size: same start poses
        keys = jax.random.split(jax.random.PRNGKey(args.seed), len(map_ids))
        out = [None] * len(map_ids)
        start = 0
        while start < len(map_ids):
            ids = map_ids[start:start + self.env_batch]
            try:
                res = jax.device_get(rollout(params['actor'], keys[start:start + len(ids)], env_params,
                                             jnp.asarray(ids)))
            except jax.errors.JaxRuntimeError as e:
                if 'RESOURCE_EXHAUSTED' not in str(e) or self.env_batch == 1:
                    raise
                self.env_batch = max(1, self.env_batch // 2)
                print(f'out of GPU memory, {self.env_batch} episodes in parallel from now on', flush=True)
                continue
            for j in range(len(ids)):
                out[start + j] = {k: v[j] for k, v in res.items()}
            start += len(ids)
        res = {k: np.stack([o[k] for o in out]) for k in out[0]}

        dt = env_cfg.step_size
        common = dict(method=name, seed=train_cfg.seed, num_agents=num_agents, map_set=self.map_set)
        rows, curves = [], []
        for e, m in enumerate(map_ids):
            ep = dict(common, map=self.map_names[m], episode=int(episodes[e]))
            covered_by = res['covered_by'][e]
            row = dict(ep, checkpoint=path, comm_range=env_cfg.comm_range, map_size=self.map_sizes[m],
                       steps=int(res['steps'][e]), time=int(res['steps'][e]) * dt,
                       success=int(res['coverage'][e] >= args.goal_coverage), coverage=float(res['coverage'][e]))
            for th in THRESHOLDS:
                i = int(round(th * 100)) - 1
                row[f'time_{i + 1}'] = float(res['level_step'][e, i]) * dt
                row[f'overlap_{i + 1}'] = float(res['level_overlap'][e, i])
            row.update(overlap=float(res['overlap'][e]),
                       collisions=int(res['collisions'][e]), robot_collisions=int(res['robot_collisions'][e]),
                       path_length=float(res['path'][e].mean()),
                       imbalance=float(covered_by.std() / max(covered_by.mean(), 1e-6)),
                       connectivity=float(res['connectivity'][e]) if num_agents > 1 else float('nan'),
                       belief_recall=float(res['belief_recall'][e]))
            rows.append(row)
            curve = dict(ep)
            curve.update({f't{c}': float(res['level_step'][e, j]) * dt for j, c in enumerate(LEVELS)})
            curve.update({f'o{c}': float(res['level_overlap'][e, j]) for j, c in enumerate(LEVELS)})
            curves.append(curve)
        return rows, curves, env_cfg


def merge_csv(filename, rows, replaced):
    """
    Writes rows to filename, keeping the existing rows whose (method,
    num_agents, map_set) is not in replaced
    """
    key = lambda r: (r['method'], int(r['num_agents']), r['map_set'])
    old = []
    if os.path.exists(filename):
        with open(filename) as f:
            old = [r for r in csv.DictReader(f) if key(r) not in replaced]
    all_rows = old + rows
    if not all_rows:
        return
    fields = list(rows[0].keys()) if rows else list(old[0].keys())
    tmp = filename + '.tmp'
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(tmp, filename)


def default_map_set(patterns):
    """'maps/test_large_*.png' -> 'test_large'"""
    base = os.path.basename(patterns[0])
    return re.sub(r'[_*?\[\]0-9]*\.png$', '', base).rstrip('_') or 'maps'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', required=True, help='NAME=CHECKPOINT_GLOB[,env_option=value,...]')
    parser.add_argument('--num_agents', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--maps', nargs='+', default=None, help='globs of map images (default: mowing evaluation maps)')
    parser.add_argument('--map_set', default=None, type=str, help='name of the map set (default: from --maps)')
    parser.add_argument('--episodes', default=1, type=int, help='episodes per map and checkpoint')
    parser.add_argument('--max_steps', default=5000, type=int)
    parser.add_argument('--goal_coverage', default=0.99, type=float)
    parser.add_argument('--env_batch', default=50, type=int, help='episodes simulated in parallel')
    parser.add_argument('--seed', default=0, type=int, help='seed of the start poses')
    parser.add_argument('--out', default='results/eval', type=str)
    parser.add_argument('--dashboard', default=True, action=BooleanOptionalAction)
    parser.add_argument('--ref_agents', default=None, type=int, help='number of robots of the curves (dashboard)')
    args = parser.parse_args()

    files = sorted({f for p in args.maps for f in glob.glob(p)}) if args.maps else eval_map_files()
    assert len(files) > 0, 'No maps found'
    map_set = args.map_set or (default_map_set(args.maps) if args.maps else 'eval_mowing')
    maps = [load_map(f) for f in files]
    map_names = [os.path.basename(f) for f in files]
    os.makedirs(args.out, exist_ok=True)
    runs = [parse_run(spec) for spec in args.run]
    print(f'map set {map_set}: {len(files)} maps x {args.episodes} episodes, robots {args.num_agents}, '
          f'{sum(len(p) for _, p, _ in runs)} checkpoints, JAX device {jax.devices()[0]}', flush=True)

    runs_file = os.path.join(args.out, 'runs.json')
    meta = json.load(open(runs_file)) if os.path.exists(runs_file) else {}
    meta.setdefault('methods', {})
    meta.setdefault('map_sets', {})
    meta['map_sets'][map_set] = dict(maps=len(files), episodes=args.episodes, goal_coverage=args.goal_coverage,
                                     max_steps=args.max_steps, seed=args.seed, globs=args.maps)
    evaluator = Evaluator(maps, map_names, map_set, args)
    t_start = time.time()
    for name, paths, overrides in runs:
        for n in args.num_agents:
            rows, curves = [], []
            for path in paths:
                t0 = time.time()
                r, c, env_cfg = evaluator.run(name, path, overrides, n)
                rows += r
                curves += c
                mean = lambda k: np.nanmean([x[k] for x in r])
                p = round(100 * args.goal_coverage) if round(100 * args.goal_coverage) in (90, 95, 99) else 99
                print(f'{name:22s} seed {r[0]["seed"]:<2d} N={n:<2d} success {mean("success"):.2f}  '
                      f'coverage {mean("coverage"):.3f}  time_{p} {mean(f"time_{p}"):7.1f} s  '
                      f'overlap_{p} {100 * mean(f"overlap_{p}"):5.1f} %  robot coll. {mean("robot_collisions"):5.1f}'
                      f'  ({time.time() - t0:.0f} s, total {(time.time() - t_start) / 60:.0f} min)', flush=True)
            # written after all seeds, so that a method is never half collected
            merge_csv(os.path.join(args.out, 'episodes.csv'), rows, {(name, n, map_set)})
            merge_csv(os.path.join(args.out, 'curves.csv'), curves, {(name, n, map_set)})
        meta['methods'][name] = dict(checkpoints=paths, overrides=overrides, comm_range=env_cfg.comm_range)
        with open(runs_file, 'w') as f:
            json.dump(meta, f, indent=2)

    if args.dashboard:
        from dashboard_marl import draw_dashboard
        out = os.path.join(args.out, 'dashboard.pdf')
        data_out = draw_dashboard(args.out, out, ref_agents=args.ref_agents)
        print('dashboard:', out)
        print('data shown in the dashboard:', data_out)


if __name__ == '__main__':
    main()
