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
                      the best single robot (--reference, by default the
                      single-agent policy fine-tuned with N = 1), 1 = ideal
                      linear speedup over it
  overlap at the goal only episodes that reached the goal coverage
  robot collisions    per episode (symlog scale)
  imbalance           coefficient of variation of the area covered per robot
then the coverage over time of one team size (--ref_agents, median map) and
the paired tests of the main pairs (--pair, by default MAPPO and IPPO against
the fine-tuned single agent and IPPO against MAPPO, all with flip_when_stuck,
i.e. under the same rules): relative difference B - A of the mean over maps,
Wilcoxon signed-rank test over the maps, Holm corrected within each (map set,
pair, metric) over the team sizes, as in compare_sa_stats.py.

Lines: mean over maps of the mean over the episodes (and seeds) of a map, band:
95 % bootstrap CI over maps. A method keeps its color with and without
flip_when_stuck: solid with flip, dashed without, dotted for the last
checkpoint ('(last)'). Everything shown is also written to
<pdf name>_data/ (points.csv, tests.csv). Plotting runs on the CPU, the data
comes from the GPU rollouts of collect_marl.py.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from compare_sa_stats import compare, holm
from dashboard_marl import AXIS, FONT, GRID, INK, INK_2, MUTED, SURFACE, load

# Categorical slots in fixed order (dataviz reference palette), by base method
BASE_COLORS = {'MAPPO': '#2a78d6', 'IPPO': '#eb6834', 'Single-agent fine-tuned xN': '#1baf7a',
               'Single-agent xN': '#eda100'}
EXTRA_COLORS = ['#e87ba4', '#008300', '#6250d6', '#e34948']
DEFAULT_METHODS = ['Single-agent xN + flip', 'Single-agent fine-tuned xN + flip', 'MAPPO + flip', 'MAPPO',
                   'IPPO + flip', 'IPPO']
DEFAULT_PAIRS = ['Single-agent fine-tuned xN + flip:MAPPO + flip', 'Single-agent fine-tuned xN + flip:IPPO + flip',
                 'MAPPO + flip:IPPO + flip']
DEFAULT_REFERENCE = 'Single-agent fine-tuned xN + flip'
# (key, row title, axis label, lower is better, scale of the shown values)
ROWS = [
    ('success', 'Success rate', 'episodes [%]', False, 100),
    ('time', 'Time to the goal coverage', 'time [min], log', True, 1),
    ('efficiency', 'Team efficiency vs best single robot', 'T_ref(1) / (N T(N))', False, 1),
    ('overlap', 'Overlap at the goal coverage', 'covered again / covered [%]', True, 100),
    ('robot_coll', 'Collisions between robots', 'per episode, symlog', True, 1),
    ('imbalance', 'Workload imbalance', 'CV of area per robot', True, 1),
]
TEST_METRICS = [('success', 'success', False), ('time', 'time', True), ('efficiency', 'efficiency', False),
                ('overlap', 'overlap', True), ('robot_coll', 'robot coll.', True), ('imbalance', 'imbalance', True),
                ('path', 'path length', True)]


def base_name(method):
    return method.replace(' + flip', '').replace(' (last)', '').strip()


def style(method):
    if '(last)' in method:
        return ':'
    return '-' if 'flip' in method else '--'


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
        ax.plot(xs, ys, color=colors[m], linestyle=style(m), linewidth=2, marker='o', markersize=5,
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
        ax.plot(t[ok], levels[ok], color=colors[m], linestyle=style(m), linewidth=2, label=m)
    ax.set_ylim(0, 101)
    style_axes(ax, f'Coverage over time, {n} robots (median map)', 'coverage [%]', 'time [min]')


def tests_panel(ax, ms, pairs, rng, tests):
    """Heatmap: rows = (pair, metric), columns = team sizes, relative difference B - A, * Holm p < 0.05"""
    from matplotlib.patches import Rectangle
    rows, cells = [], []
    for A, B in pairs:
        for key, label, lower in TEST_METRICS:
            group = []
            for n in ms.agents:
                a, b = ms.get(A, n, key), ms.get(B, n, key)
                r = compare(a, b, rng) if a is not None and b is not None else None
                group.append(r)
            ps = holm([r['p'] if r is not None else np.nan for r in group])
            row = []
            for n, r, q in zip(ms.agents, group, ps):
                if r is None:
                    row.append(None)
                    continue
                better = (r['diff'] < 0) if lower else (r['diff'] > 0)
                sig = bool(np.isfinite(q) and q < 0.05)
                row.append(dict(r, p_holm=q, significant=sig, b_better=better and r['diff'] != 0))
                tests.append(dict(map_set=ms.name, A=A, B=B, metric=key, lower_is_better=lower, num_agents=n,
                                  maps=r['maps'], estimate_A=r['estimate_A'], estimate_B=r['estimate_B'],
                                  diff=r['diff'], ci_low=r['ci_low'], ci_high=r['ci_high'], rel=r['rel'],
                                  p=r['p'], p_holm=q, significant=sig))
            rows.append((A, B, label))
            cells.append(row)
    if not rows:
        ax.axis('off')
        return
    # one header line per pair, then its metrics
    lines = []
    for i, (A, B, label) in enumerate(rows):
        if i % len(TEST_METRICS) == 0:
            lines.append(('header', f'{B}  vs  {A}', None))
        lines.append(('cells', label, cells[i]))
    ax.set_xlim(0, len(ms.agents))
    ax.set_ylim(len(lines), 0)
    for i, (kind, label, row) in enumerate(lines):
        if kind == 'header':
            if i > 0:
                ax.axhline(i + 0.15, color=AXIS, linewidth=0.8)
            ax.text(-0.25, i + 0.62, label, ha='left', va='center', fontsize=8.5, color=INK, fontweight='bold',
                    transform=ax.transData, clip_on=False)
            continue
        ax.text(-0.04, i + 0.5, label, ha='right', va='center', fontsize=8, color=INK_2)
        for j, c in enumerate(row):
            if c is None:
                text, face, alpha = '-', SURFACE, 1
            else:
                rel = c['rel'] if np.isfinite(c['rel']) else 0
                face = '#1baf7a' if c['b_better'] else '#e34948'
                alpha = (0.18 + 0.55 * min(abs(rel), 1)) if c['significant'] else 0.07
                if c['diff'] == 0:
                    face, alpha = SURFACE, 1
                text = (f'{100 * rel:+.0f} %' if np.isfinite(c['rel']) else f'{c["diff"]:+.2g}') + \
                       (' *' if c['significant'] else '')
            ax.add_patch(Rectangle((j + 0.02, i + 0.08), 0.96, 0.84, facecolor=face, alpha=alpha, linewidth=0))
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', fontsize=8,
                    color=INK if c is not None and c['significant'] else INK_2)
    ax.set_xticks(np.arange(len(ms.agents)) + 0.5)
    ax.set_xticklabels([f'N = {n}' for n in ms.agents])
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks([])
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('Paired tests over maps: B - A relative to A, * Holm p < 0.05\n'
                 'green: B better, red: B worse, pale: not significant',
                 loc='left', fontsize=10.5, color=INK, fontweight='bold', pad=22)


def draw(data_dir, out, methods=None, pairs=None, reference=DEFAULT_REFERENCE, ref_agents=4, map_sets=None, seed=0):
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
    pairs = [tuple(x.strip() for x in p.split(':')) for p in (pairs or DEFAULT_PAIRS)]
    pairs = [(a, b) for a, b in pairs if a in present and b in present]
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
    test_lines = max(1, len(pairs)) * (len(TEST_METRICS) + 1)
    # heights in inches: header, metric rows, tests
    head, row_h, line_h = 2.1, 3.0, 0.26
    height = head + row_h * (n_rows - 1) + line_h * test_lines + 0.8
    fig = plt.figure(figsize=(6 * len(sets) + 1.5, height), facecolor=SURFACE)
    gs = fig.add_gridspec(n_rows, len(sets), height_ratios=[row_h] * (n_rows - 1) + [line_h * test_lines + 0.6],
                          hspace=0.6, wspace=0.42, left=0.08, right=0.98, top=1 - (head + 0.6) / height,
                          bottom=0.15 / height)
    points, tests = [], []
    for c, ms in enumerate(data):
        for r, row in enumerate(ROWS):
            ax = fig.add_subplot(gs[r, c])
            metric_panel(ax, ms, methods, colors, row, rng, points)
        n_ref = ref_agents if ref_agents in ms.agents else ms.agents[-1]
        coverage_panel(fig.add_subplot(gs[len(ROWS), c]), ms, methods, colors, n_ref)
        tests_panel(fig.add_subplot(gs[len(ROWS) + 1, c]), ms, pairs, rng, tests)

    # column headers: map set, maps, episodes, goal
    for c, ms in enumerate(data):
        ax = fig.axes[c * (len(ROWS) + 2)]
        bbox = ax.get_position()
        fig.text(bbox.x0, bbox.y1 + 0.75 / height, ms.name, fontsize=15, color=INK, fontweight='bold')
        fig.text(bbox.x0, bbox.y1 + 0.5 / height, f'{len(ms.maps)} maps x {ms.info.get("episodes", 1)} episodes, '
                                           f'goal {ms.p} % coverage, limit {ms.limit / 60:.0f} min',
                 fontsize=9.5, color=INK_2)
    top = 1 - 0.3 / height
    fig.text(0.02, top, 'Single agent xN vs MAPPO vs IPPO: multi-robot coverage', fontsize=20, color=INK,
             fontweight='bold', va='top')
    ref = data[0].reference if data else None
    fig.text(0.02, top - 0.45 / height,
             'Same episodes for every policy (maps, start poses, seed), deterministic actions. Lines: mean over '
             'maps, band: 95 % bootstrap CI over maps.\nSolid: with flip_when_stuck, dashed: without, dotted: '
             f'last checkpoint. Team efficiency relative to {ref or "each method"} with 1 robot on the same map.',
             fontsize=10, color=INK_2, va='top')
    handles = [plt.Line2D([], [], color=colors[m], linestyle=style(m), linewidth=2, marker='o', markersize=5,
                          markeredgecolor=SURFACE) for m in methods]
    fig.legend(handles, methods, loc='upper left', bbox_to_anchor=(0.015, top - 0.95 / height), ncol=min(len(methods), 6),
               frameon=False, fontsize=10.5, labelcolor=INK, handlelength=2.6, columnspacing=2.2)

    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    data_out = os.path.splitext(out)[0] + '_data'
    os.makedirs(data_out, exist_ok=True)
    for name, rows in (('points', points), ('tests', tests)):
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
    parser.add_argument('--pair', action='append', default=None, help="'METHOD A:METHOD B' of the tests, B - A")
    parser.add_argument('--reference', default=DEFAULT_REFERENCE, type=str,
                        help='method whose time with 1 robot is the reference of the team efficiency')
    parser.add_argument('--ref_agents', default=4, type=int, help='number of robots of the coverage curves')
    parser.add_argument('--map_sets', nargs='+', default=None, help='map sets (columns), default: all')
    parser.add_argument('--seed', default=0, type=int, help='seed of the bootstrap')
    args = parser.parse_args()
    out = args.out or os.path.join(args.data, 'dashboard_compare.pdf')
    data_out = draw(args.data, out, args.methods, args.pair, args.reference, args.ref_agents, args.map_sets,
                    args.seed)
    print('one page comparison:', out)
    print('data shown in the dashboard:', data_out)


if __name__ == '__main__':
    main()
