"""
One page comparison (PDF) of the single-agent policy replicated on N robots,
MAPPO and IPPO, from the data of compare_sa.sh (collect_marl.py): every
policy runs the same episodes (same maps, start poses and seed, deterministic
actions).

  python dashboard_compare.py --data results/compare_sa
  python dashboard_compare.py --data results/compare_sa --methods 'MAPPO + flip' 'IPPO + flip' --ref_agents 8

One column per map set, one row per metric, against the number of robots:
  success rate        reached the goal coverage of the map set
  time to the goal    episodes that did not reach it count with the step
                      limit (a lower bound of their time)
  team efficiency     T_ref(1) / (N T(N)) on the same map, T_ref(1): time of
                      the reference single robot (--reference, by default the
                      single-agent policy replicated on N robots), 1 = ideal
                      linear speedup over it
  overlap at the goal only episodes that reached the goal coverage
  robot collisions    per episode (symlog scale)
  imbalance           coefficient of variation of the area covered per robot
then the coverage over time of one team size (--ref_agents, median map) and,
across the whole page, one summary table: one row per method, one column per
metric (the ones above and the path length), mean over the map sets and team
sizes of the mean over maps, the best of each column highlighted. The paired
tests are in compare_sa_stats.py (stats.txt of compare_sa.sh).

Lines: mean over maps of the mean over the episodes (and seeds) of a map, band:
95 % bootstrap CI over maps. A method keeps its color with and without
flip_when_stuck: solid with flip, dashed without (when both are shown),
dotted for the last checkpoint ('(last)'). By default: the single-agent policy
replicated on N robots, MAPPO and IPPO, without flip_when_stuck, with the
team efficiency relative to the single-agent policy with 1 robot. Everything shown is also written to
<pdf name>_data/ (points.csv, table.csv). Plotting runs on the CPU, the data
comes from the GPU rollouts of collect_marl.py.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from dashboard_marl import AXIS, FONT, GRID, INK, INK_2, MUTED, SURFACE, load

# Categorical slots in fixed order (dataviz reference palette), by base method
BASE_COLORS = {'MAPPO': '#2a78d6', 'IPPO': '#eb6834', 'Single-agent fine-tuned xN': '#1baf7a',
               'Single-agent xN': '#eda100'}
EXTRA_COLORS = ['#e87ba4', '#008300', '#6250d6', '#e34948']
DEFAULT_METHODS = ['Single-agent xN', 'MAPPO', 'IPPO']
DEFAULT_REFERENCE = 'Single-agent xN'
DEFAULT_TITLE = 'Single agent xN vs MAPPO vs IPPO: multi-robot coverage'
# (key, row title, axis label, lower is better, scale of the shown values)
ROWS = [
    ('success', 'Success rate', 'episodes [%]', False, 100),
    ('time', 'Time to the goal coverage', 'time [min], log', True, 1),
    ('efficiency', 'Team efficiency vs single-agent with 1 robot', 'T_ref(1) / (N T(N))', False, 1),
    ('overlap', 'Overlap at the goal coverage', 'covered again / covered [%]', True, 100),
    ('robot_coll', 'Collisions between robots', 'per episode, symlog', True, 1),
    ('imbalance', 'Workload imbalance', 'CV of area per robot', True, 1),
]
# (key, column title, unit, lower is better, scale, format) of the summary table
TABLE_METRICS = [('success', 'Success', '%', False, 100, '{:.1f}'), ('time', 'Time to goal', 'min', True, 1, '{:.1f}'),
                 ('efficiency', 'Team efficiency', 'T_ref(1) / (N T(N))', False, 1, '{:.3f}'),
                 ('overlap', 'Overlap', '%', True, 100, '{:.1f}'),
                 ('robot_coll', 'Robot collisions', 'per episode', True, 1, '{:.1f}'),
                 ('imbalance', 'Imbalance', 'CV of area', True, 1, '{:.3f}'),
                 ('path', 'Path length', 'm per robot', True, 1, '{:.0f}')]


def base_name(method):
    return method.replace(' + flip', '').replace(' (last)', '').strip()


def style(method, methods):
    """Dotted for the last checkpoint, dashed without flip_when_stuck when the same method is also shown with it"""
    if '(last)' in method:
        return ':'
    return '--' if 'flip' not in method and f'{base_name(method)} + flip' in methods else '-'


class MapSet:
    """Per map values (mean over episodes and seeds) of one map set"""

    def __init__(self, name, episodes, curves, info, reference):
        self.name = name
        self.info = info
        self.p = int(round(100 * info.get('goal_coverage', 0.99)))
        rows = [r for r in episodes if r['map_set'] == name]
        self.limit = max(r['time'] for r in rows)
        self.maps = sorted({r['map'] for r in rows})
        self.methods = list(dict.fromkeys(r['method'] for r in rows))
        self.agents = sorted({int(r['num_agents']) for r in rows})
        acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for r in rows:
            t = r[f'time_{self.p}']
            reached = np.isfinite(t)
            v = dict(success=float(reached), time=(t if reached else self.limit) / 60,
                     overlap=r[f'overlap_{self.p}'] if reached else np.nan,
                     robot_coll=r['robot_collisions'], imbalance=r['imbalance'], path=r['path_length'])
            for k, x in v.items():
                acc[(r['method'], int(r['num_agents']))][k][r['map']].append(x)
        self.values = {}
        for key, metrics in acc.items():
            self.values[key] = {k: np.array([np.nanmean(by_map[m]) if np.any(np.isfinite(by_map.get(m, [np.nan])))
                                             else np.nan for m in self.maps]) for k, by_map in metrics.items()}
        # team efficiency against the time of the reference method with 1 robot on the same map
        self.reference = reference if (reference, 1) in self.values else None
        for (m, n), v in self.values.items():
            ref = self.values.get((self.reference or m, 1))
            v['efficiency'] = ref['time'] / (n * v['time']) if ref is not None else np.full(len(self.maps), np.nan)
        self.curves = defaultdict(list)
        for r in curves:
            if r['map_set'] == name:
                self.curves[(r['method'], int(r['num_agents']))].append(r)

    def get(self, method, n, key):
        v = self.values.get((method, n))
        return None if v is None else v[key]

    def median_time_to_coverage(self, method, n):
        """Median map of the mean time over episodes to each coverage level, inf where not reached"""
        rows = self.curves.get((method, n), [])
        if not rows:
            return None
        levels = np.arange(1, 100)
        by_map = defaultdict(list)
        for r in rows:
            by_map[r['map']].append([r[f't{c}'] for c in levels])
        per_map = np.array([np.where(np.isfinite(np.array(v)), np.array(v), np.inf).mean(axis=0)
                            for v in by_map.values()])
        return levels, np.median(per_map, axis=0) / 60


def mean_ci(x, rng, n_boot=10000):
    """Mean over maps and 95 % bootstrap CI"""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, (np.nan, np.nan)
    if len(x) == 1:
        return x[0], (x[0], x[0])
    boot = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return x.mean(), (np.percentile(boot, 2.5), np.percentile(boot, 97.5))


def style_axes(ax, title=None, ylabel=None, xlabel=None):
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, loc='left', fontsize=11, color=INK, fontweight='bold', pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


def metric_panel(ax, ms, methods, colors, row, rng, points):
    key, title, ylabel, _, scale = row
    for m in methods:
        xs, ys, lo, hi = [], [], [], []
        for n in ms.agents:
            x = ms.get(m, n, key)
            if x is None:
                continue
            est, ci = mean_ci(x, rng)
            points.append(dict(map_set=ms.name, metric=key, method=m, num_agents=n, estimate=est, ci_low=ci[0],
                               ci_high=ci[1], maps=int(np.isfinite(x).sum())))
            if np.isfinite(est):
                xs.append(n)
                ys.append(scale * est)
                lo.append(scale * ci[0])
                hi.append(scale * ci[1])
        if not xs:
            continue
        ax.fill_between(xs, lo, hi, color=colors[m], alpha=0.10, linewidth=0)
        ax.plot(xs, ys, color=colors[m], linestyle=style(m, methods), linewidth=2, marker='o', markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3, label=m)
    if key == 'time':
        ax.set_yscale('log')
    elif key == 'robot_coll':
        ax.set_yscale('symlog', linthresh=1)
        ax.set_ylim(bottom=0)
    elif key == 'success':
        ax.set_ylim(top=102)
    elif key == 'efficiency':
        ax.axhline(1, color=MUTED, linewidth=1, zorder=2)
        ax.annotate('ideal', (ms.agents[-1], 1), textcoords='offset points', xytext=(-2, 3), ha='right',
                    fontsize=8, color=MUTED)
        ax.set_ylim(bottom=0)
    ax.set_xscale('log', base=2)
    ax.set_xticks(ms.agents)
    ax.set_xticklabels([str(n) for n in ms.agents])
    ax.minorticks_off()
    target = f' ({ms.p} %)' if key in ('success', 'time', 'overlap') else ''
    style_axes(ax, title + target, ylabel, 'robots')


def coverage_panel(ax, ms, methods, colors, n):
    for m in methods:
        res = ms.median_time_to_coverage(m, n)
        if res is None:
            continue
        levels, t = res
        ok = np.isfinite(t)
        ax.plot(t[ok], levels[ok], color=colors[m], linestyle=style(m, methods), linewidth=2, label=m)
    ax.set_ylim(0, 101)
    style_axes(ax, f'Coverage over time, {n} robots (median map)', 'coverage [%]', 'time [min]')


def summary_table(ax, data, methods, colors, table):
    """
    One row per method, one column per metric: mean over the map sets and team
    sizes (each (map set, N) weighs the same) of the mean over maps, best value
    of each column in bold
    """
    from matplotlib.patches import Rectangle
    values = {}
    for m in methods:
        for key, *_ in TABLE_METRICS:
            cells = []
            for ms in data:
                for n in ms.agents:
                    x = ms.get(m, n, key)
                    if x is not None and np.any(np.isfinite(x)):
                        cells.append(np.nanmean(x))
            values[(m, key)] = float(np.mean(cells)) if cells else np.nan
            table.append(dict(method=m, metric=key, value=values[(m, key)], cells=len(cells)))
    ncol = len(TABLE_METRICS)
    ax.set_xlim(-2.2, ncol)
    ax.set_ylim(len(methods) + 1, 0)
    for j, (key, label, unit, lower, scale, fmt) in enumerate(TABLE_METRICS):
        ax.text(j + 0.5, 0.45, label, ha='center', va='center', fontsize=10, color=INK, fontweight='bold')
        ax.text(j + 0.5, 0.82, unit, ha='center', va='center', fontsize=8.5, color=MUTED)
        col = [round(scale * values[(m, key)], 6) for m in methods]
        finite = [v for v in col if np.isfinite(v)]
        best = (min(finite) if lower else max(finite)) if finite else np.nan
        for i, (m, v) in enumerate(zip(methods, col)):
            is_best = np.isfinite(v) and fmt.format(v) == fmt.format(best)
            if is_best:
                ax.add_patch(Rectangle((j + 0.03, i + 1.06), 0.94, 0.88, facecolor='#1baf7a', alpha=0.18,
                                       linewidth=0))
            ax.text(j + 0.5, i + 1.5, fmt.format(v) if np.isfinite(v) else '-', ha='center', va='center',
                    fontsize=10, color=INK if is_best else INK_2, fontweight='bold' if is_best else 'normal')
    for i, m in enumerate(methods):
        ax.plot([-2.15, -1.95], [i + 1.5] * 2, color=colors[m], linestyle=style(m, methods), linewidth=2.5)
        ax.text(-1.88, i + 1.5, m, ha='left', va='center', fontsize=10, color=INK)
        ax.axhline(i + 1, color=GRID, linewidth=0.8)
    ax.axhline(1, color=AXIS, linewidth=1)
    ax.axis('off')
    ax.set_title('Mean over all map sets and team sizes (N = ' + ', '.join(str(n) for n in data[0].agents) +
                 '), each (map set, N) weighs the same; best of each column highlighted',
                 loc='left', fontsize=11, color=INK, fontweight='bold', pad=8)


def draw(data_dir, out, methods=None, reference=DEFAULT_REFERENCE, ref_agents=4, map_sets=None, seed=0,
         title=DEFAULT_TITLE):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = FONT

    episodes, curves, meta, _ = load(data_dir)
    assert episodes, f'no data in {data_dir}'
    present = list(dict.fromkeys(r['method'] for r in episodes))
    methods = [m for m in (methods or DEFAULT_METHODS) if m in present]
    assert methods, f'none of the methods is in the data: {present}'
    colors, extra = {}, iter(EXTRA_COLORS)
    bases = {}
    for m in methods:
        b = base_name(m)
        if b not in bases:
            bases[b] = BASE_COLORS.get(b) or next(extra, MUTED)
        colors[m] = bases[b]
    sets = [s for s in (map_sets or list(meta['map_sets'])) if any(r['map_set'] == s for r in episodes)]
    rng = np.random.default_rng(seed)
    data = [MapSet(s, episodes, curves, meta['map_sets'][s], reference) for s in sets]

    n_rows = len(ROWS) + 2
    # heights in inches: header, metric rows, summary table
    head, row_h, line_h = 2.1, 3.0, 0.42
    table_h = line_h * (len(methods) + 1)
    height = head + row_h * (n_rows - 1) + table_h + 0.9
    fig = plt.figure(figsize=(6 * len(sets) + 1.5, height), facecolor=SURFACE)
    gs = fig.add_gridspec(n_rows, len(sets), height_ratios=[row_h] * (n_rows - 1) + [table_h],
                          hspace=0.6, wspace=0.42, left=0.08, right=0.98, top=1 - (head + 0.6) / height,
                          bottom=0.15 / height)
    points, table = [], []
    for c, ms in enumerate(data):
        for r, row in enumerate(ROWS):
            if row[0] == 'efficiency' and ms.reference:
                row = (row[0], f'Team efficiency vs {ms.reference} with 1 robot', *row[2:])
            ax = fig.add_subplot(gs[r, c])
            metric_panel(ax, ms, methods, colors, row, rng, points)
        n_ref = ref_agents if ref_agents in ms.agents else ms.agents[-1]
        coverage_panel(fig.add_subplot(gs[len(ROWS), c]), ms, methods, colors, n_ref)
    summary_table(fig.add_subplot(gs[len(ROWS) + 1, :]), data, methods, colors, table)

    # column headers: map set, maps, episodes, goal
    for c, ms in enumerate(data):
        ax = fig.axes[c * (len(ROWS) + 1)]
        bbox = ax.get_position()
        fig.text(bbox.x0, bbox.y1 + 0.75 / height, ms.name, fontsize=15, color=INK, fontweight='bold')
        fig.text(bbox.x0, bbox.y1 + 0.5 / height, f'{len(ms.maps)} maps x {ms.info.get("episodes", 1)} episodes, '
                                           f'goal {ms.p} % coverage, limit {ms.limit / 60:.0f} min',
                 fontsize=9.5, color=INK_2)
    top = 1 - 0.3 / height
    fig.text(0.02, top, title, fontsize=20, color=INK,
             fontweight='bold', va='top')
    ref = data[0].reference if data else None
    if any('flip' in m for m in methods):
        flips = 'Solid: with flip_when_stuck, dashed: without, dotted: last checkpoint. '
    else:
        flips = 'Without flip_when_stuck. '
    fig.text(0.02, top - 0.45 / height,
             'Same episodes for every policy (maps, start poses, seed), deterministic actions. Lines: mean over '
             'maps, band: 95 % bootstrap CI over maps.\n' + flips +
             f'Team efficiency relative to {ref or "each method"} with 1 robot on the same map.',
             fontsize=10, color=INK_2, va='top')
    handles = [plt.Line2D([], [], color=colors[m], linestyle=style(m, methods), linewidth=2, marker='o', markersize=5,
                          markeredgecolor=SURFACE) for m in methods]
    fig.legend(handles, methods, loc='upper left', bbox_to_anchor=(0.015, top - 0.95 / height), ncol=min(len(methods), 6),
               frameon=False, fontsize=10.5, labelcolor=INK, handlelength=2.6, columnspacing=2.2)

    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    data_out = os.path.splitext(out)[0] + '_data'
    os.makedirs(data_out, exist_ok=True)
    for name, rows in (('points', points), ('table', table)):
        if rows:
            with open(os.path.join(data_out, f'{name}.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    return data_out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='results/compare_sa', type=str, help='directory written by compare_sa.sh')
    parser.add_argument('--out', default=None, type=str, help='PDF file (default: <data>/dashboard_compare.pdf)')
    parser.add_argument('--methods', nargs='+', default=None, help='methods to show, in this order')
    parser.add_argument('--reference', default=DEFAULT_REFERENCE, type=str,
                        help='method whose time with 1 robot is the reference of the team efficiency')
    parser.add_argument('--ref_agents', default=4, type=int, help='number of robots of the coverage curves')
    parser.add_argument('--map_sets', nargs='+', default=None, help='map sets (columns), default: all')
    parser.add_argument('--seed', default=0, type=int, help='seed of the bootstrap')
    parser.add_argument('--title', default=DEFAULT_TITLE, type=str)
    args = parser.parse_args()
    out = args.out or os.path.join(args.data, 'dashboard_compare.pdf')
    data_out = draw(args.data, out, args.methods, args.reference, args.ref_agents, args.map_sets,
                    args.seed, args.title)
    print('one page comparison:', out)
    print('data shown in the dashboard:', data_out)


if __name__ == '__main__':
    main()
