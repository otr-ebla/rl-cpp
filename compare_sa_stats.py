"""
Paired comparison of two policies on the same episodes (collect_marl.py data),
e.g. the single-agent policy replicated on N robots against MAPPO, see
compare_sa.sh. The dashboard (dashboard_marl.py) gives no p-values with a
single training seed per method; here the question is about these trained
policies, so the population is the maps: for every map set, team size and
metric, the per map difference B - A (mean over the episodes of a map, which
share the start poses between methods) gets a bootstrap 95 % confidence
interval over maps and a Wilcoxon signed-rank p-value, Holm corrected within
each (map set, comparison, metric) over the team sizes.

Metrics:
  success        reached the goal coverage of the map set
  time_goal      time to the goal coverage [s], episodes that did not reach
                 it count with the time limit (a lower bound of their time)
  overlap_goal   overlap when the goal coverage was reached, only maps where
                 both policies reached it in every episode
  collisions     all collisions of all robots per episode
  robot_coll     collisions between robots per episode
  path_length    distance driven per robot [m]
  imbalance      coefficient of variation of the area covered by each robot

  python compare_sa_stats.py --data results/compare_sa \\
      --pair 'Single-agent xN + flip:MAPPO' --pair 'Single-agent xN + flip:MAPPO + flip'

Writes <data>/stats.csv (every comparison) and prints a summary.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

# (key, lower is better)
METRICS = [('success', False), ('time_goal', True), ('overlap_goal', True), ('collisions', True),
           ('robot_coll', True), ('path_length', None), ('imbalance', True)]


def load(data):
    rows, info = [], {}
    for d in sorted(glob.glob(os.path.join(data, '*', 'episodes.csv'))):
        rows += list(csv.DictReader(open(d)))
        runs = json.load(open(os.path.join(os.path.dirname(d), 'runs.json')))
        info.update(runs['map_sets'])
    return rows, info


def episode_metrics(r, goal, limit):
    g = round(100 * goal)
    t = float(r[f'time_{g}'])
    return dict(success=float(r['success']), time_goal=t if np.isfinite(t) else limit,
                overlap_goal=float(r[f'overlap_{g}']), collisions=float(r['collisions']),
                robot_coll=float(r['robot_collisions']), path_length=float(r['path_length']),
                imbalance=float(r['imbalance']))


def per_map(rows, info):
    """{(map_set, method, N): {map: {metric: mean over episodes}}}, overlap_goal nan unless all episodes reached it"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s = info[r['map_set']]
        acc[(r['map_set'], r['method'], int(r['num_agents']))][r['map']].append(
            episode_metrics(r, s['goal_coverage'], s['max_steps'] * 0.5))
    out = {}
    for k, maps in acc.items():
        out[k] = {m: {key: float(np.mean([e[key] for e in eps])) for key, _ in METRICS} for m, eps in maps.items()}
    return out


def holm(p):
    p = np.asarray(p, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    order = np.argsort(p[ok])
    vals = p[ok][order] * (ok.sum() - np.arange(ok.sum()))
    vals = np.minimum(np.maximum.accumulate(vals), 1)
    tmp = np.empty_like(vals)
    tmp[order] = vals
    adj[ok] = tmp
    return adj


def compare(a, b, rng, n_boot=10000):
    """a, b: per map values (paired). Returns dict of estimates, CI of mean(b - a), p-value"""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return None
    d = b - a
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(maps=len(d), estimate_A=a.mean(), estimate_B=b.mean(), diff=d.mean(), ci_low=np.percentile(boot, 2.5),
                ci_high=np.percentile(boot, 97.5), rel=d.mean() / abs(a.mean()) if a.mean() != 0 else np.nan,
                p=p, b_better=np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='results/compare_sa')
    parser.add_argument('--pair', action='append', required=True, help="'METHOD A:METHOD B', difference B - A")
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    rows, info = load(args.data)
    pm = per_map(rows, info)
    rng = np.random.default_rng(args.seed)
    results = []
    for pair in args.pair:
        A, B = [x.strip() for x in pair.split(':')]
        for ms in info:
            for key, lower in METRICS:
                group = []
                for n in sorted({k[2] for k in pm if k[0] == ms}):
                    if (ms, A, n) not in pm or (ms, B, n) not in pm:
                        continue
                    maps = sorted(set(pm[(ms, A, n)]) & set(pm[(ms, B, n)]))
                    a = np.array([pm[(ms, A, n)][m][key] for m in maps])
                    b = np.array([pm[(ms, B, n)][m][key] for m in maps])
                    r = compare(a, b, rng)
                    if r is None:
                        continue
                    if lower is not None:
                        better = (b < a) if lower else (b > a)
                        r['b_better'] = float(np.mean(better[np.isfinite(a) & np.isfinite(b)]))
                    group.append(dict(map_set=ms, A=A, B=B, metric=key, lower_is_better=lower, num_agents=n, **r))
                for g, p in zip(group, holm([g['p'] for g in group])):
                    g['p_holm'] = p
                    g['significant'] = bool(p < 0.05)
                results += group

    out = os.path.join(args.data, 'stats.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    for pair in args.pair:
        A, B = [x.strip() for x in pair.split(':')]
        print(f'\n===== {B}  vs  {A}   (difference B - A, 95 % CI over maps, * = Holm p < 0.05)')
        for ms in info:
            print(f'--- {ms}  (goal {info[ms]["goal_coverage"]:.0%}, {info[ms]["maps"]} maps x {info[ms]["episodes"]} episodes)')
            print(f'{"metric":13s}' + ''.join(f'{"N=" + str(n):>34s}' for n in (1, 2, 4, 8)))
            for key, _ in METRICS:
                cells = {r['num_agents']: r for r in results if r['A'] == A and r['B'] == B and r['map_set'] == ms
                         and r['metric'] == key}
                line = f'{key:13s}'
                for n in (1, 2, 4, 8):
                    r = cells.get(n)
                    if r is None:
                        line += f'{"-":>34s}'
                        continue
                    fmt = (lambda v: f'{v:.2f}') if key in ('success', 'imbalance') else (lambda v: f'{v:.3g}')
                    cell = f'{fmt(r["estimate_A"])}->{fmt(r["estimate_B"])} ({100 * r["rel"]:+.0f}%){"*" if r["significant"] else " "}'
                    line += f'{cell:>34s}'
                print(line)
    print(f'\nall comparisons: {out}')


if __name__ == '__main__':
    main()
