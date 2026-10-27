"""
Draw the comparison dashboard (PDF) from the data collected by collect_marl.py.

  python dashboard_marl.py --data results/eval     # all map sets in results/eval/*/
  python dashboard_marl.py --data results/eval --map_sets test_large --ref_agents 8

Two pages per map set:
  overview: as a function of the number of robots, the time to the target
    coverage (the goal coverage of the map set, or --threshold), the overlap
    when it is reached, the speedup over the smallest team, the success rate,
    the collisions between robots, the workload imbalance and the belief recall
    (limited communication); for one team size (--ref_agents) the overlap as a
    function of the coverage, the coverage over time and a summary table.
  statistical tests: every pair of methods, for every team size and metric:
    difference with 95 % confidence interval, p-value (Holm corrected within
    each metric) and probability of improvement.

Every estimate is the mean over the maps of the mean over the training seeds,
and the confidence intervals and tests account for both the variance between
training seeds and between maps, see marl/stats.py.

Everything the PDF shows is also written to <pdf name>_data/ next to it, in
natural units (fractions, not percent; times in minutes):
  points.csv   every point of the panels against the number of robots and of
               the summary tables: map_set, panel, method, num_agents,
               estimate, ci_low, ci_high, seeds, maps
  curves.csv   the curves of one team size: overlap at each coverage level
               (estimate and CI) and median time to each coverage level
  tests.csv    every statistical test: map_set, metric, A, B, num_agents,
               estimates, difference and relative difference with CI, p-value,
               Holm-corrected p-value, probability of improvement, significant
  info.json    settings of the dashboard (data directory, target coverage of
               each map set, ...)
The raw data is in the data directory (episodes.csv, curves.csv, runs.json).
"""
import argparse
import csv
import itertools
import json
import os
import time
from collections import defaultdict

import numpy as np

from marl.stats import compare, holm, mean_ci

# Reference categorical palette (dataviz skill), fixed order: a method keeps its
# color, given by its position in runs.json
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
FONT = ['system-ui', 'Segoe UI', 'Helvetica Neue', 'Arial', 'DejaVu Sans', 'sans-serif']
LEVELS = np.arange(1, 100)

# (key, title, lower is better, how differences are shown: 'rel' %, 'pp' percentage points, 'abs')
TESTS = [
    ('time', 'Time to the target coverage', True, 'rel'),
    ('overlap_at', 'Overlap when the target coverage is reached', True, 'pp'),
    ('success', 'Success rate', False, 'pp'),
    ('robot_collisions', 'Collisions between robots per episode', True, 'abs'),
    ('imbalance', 'Workload imbalance (CV)', True, 'abs'),
]


class Recorder:
    """Collects the values drawn in the PDF, see the module docstring"""

    def __init__(self):
        self.tables = defaultdict(list)

    def add(self, table, **row):
        self.tables[table].append({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                   for k, v in row.items()})

    def save(self, directory, info):
        os.makedirs(directory, exist_ok=True)
        for name, rows in self.tables.items():
            fields = list(dict.fromkeys(k for r in rows for k in r))
            with open(os.path.join(directory, f'{name}.csv'), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        with open(os.path.join(directory, 'info.json'), 'w') as f:
            json.dump(info, f, indent=2)


def load(data_dir):
    """
    episodes, curves and runs.json of data_dir, or merged over its
    subdirectories when it has no episodes.csv (map sets collected in parallel,
    see eval_seeds.sh)
    """
    if os.path.exists(os.path.join(data_dir, 'episodes.csv')):
        dirs = [data_dir]
    else:
        dirs = sorted(os.path.join(data_dir, d) for d in os.listdir(data_dir)
                      if os.path.exists(os.path.join(data_dir, d, 'episodes.csv')))
    episodes, curves, meta = [], [], {'methods': {}, 'map_sets': {}}
    for d in dirs:
        episodes += read_csv(os.path.join(d, 'episodes.csv'))
        curves += read_csv(os.path.join(d, 'curves.csv'))
        if os.path.exists(os.path.join(d, 'runs.json')):
            m = json.load(open(os.path.join(d, 'runs.json')))
            for k, v in m.get('methods', {}).items():
                meta['methods'].setdefault(k, v)
            meta['map_sets'].update(m.get('map_sets', {}))
    return episodes, curves, meta, dirs


def read_csv(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            out.append(row)
    return out


class Data:
    """The collected data of one map set, as (seeds x maps) matrices"""

    def __init__(self, episodes, curves, meta, map_set, threshold=None, recorder=None):
        self.map_set = map_set
        self.rec = recorder or Recorder()
        self.info = meta.get('map_sets', {}).get(map_set, {})
        goal = threshold or self.info.get('goal_coverage', 0.99)
        self.p = int(round(100 * goal))
        assert self.p in (90, 95, 99), 'the target coverage must be 0.9, 0.95 or 0.99'
        self.episodes = [r for r in episodes if r['map_set'] == map_set]
        self.curves = [r for r in curves if r['map_set'] == map_set]
        self.meta = meta.get('methods', {})
        present = {r['method'] for r in self.episodes}
        self.methods = [m for m in self.meta if m in present] + sorted(present - set(self.meta))
        self.agents = sorted({int(r['num_agents']) for r in self.episodes})
        self.maps = sorted({r['map'] for r in self.episodes})
        self.map_index = {m: i for i, m in enumerate(self.maps)}
        self.seeds = {m: sorted({int(r['seed']) for r in self.episodes if r['method'] == m}) for m in self.methods}
        # episodes that did not reach the target count as the step limit
        self.max_time = max(r['time'] for r in self.episodes)
        self._groups = defaultdict(list)
        for r in self.episodes:
            self._groups[(r['method'], int(r['num_agents']))].append(r)
        self._curve_groups = defaultdict(list)
        for r in self.curves:
            self._curve_groups[(r['method'], int(r['num_agents']))].append(r)

    def comm_range(self, method):
        rows = [r for r in self.episodes if r['method'] == method]
        return rows[0]['comm_range'] if rows else 0

    def paired(self, a, b):
        """Whether two methods use the same trained policies"""
        ca, cb = self.meta.get(a, {}).get('checkpoints'), self.meta.get(b, {}).get('checkpoints')
        return ca is not None and ca == cb

    def value(self, key, r):
        if key == 'time':
            t = r[f'time_{self.p}']
            return (t if np.isfinite(t) else self.max_time) / 60
        if key == 'overlap_at':
            return r[f'overlap_{self.p}']
        if key == 'success':
            return float(np.isfinite(r[f'time_{self.p}']))
        return r[key]

    def matrix(self, method, n, value, rows=None):
        """(seeds x maps) matrix of value(row), mean over the episodes of a map"""
        rows = self._groups.get((method, n), []) if rows is None else rows
        seeds = self.seeds[method]
        sums = np.zeros((len(seeds), len(self.maps)))
        counts = np.zeros_like(sums)
        for r in rows:
            v = value(r)
            if np.isfinite(v):
                i, j = seeds.index(int(r['seed'])), self.map_index[r['map']]
                sums[i, j] += v
                counts[i, j] += 1
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    def metric(self, method, n, key):
        return self.matrix(method, n, lambda r: self.value(key, r))

    def has(self, method, n):
        return bool(self._groups.get((method, n)))

    def curve_matrix(self, method, n, column):
        return self.matrix(method, n, lambda r: r[column], self._curve_groups.get((method, n), []))


def style_axes(ax, title, ylabel, xlabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=12, color=INK, fontweight='bold', pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def plot_line(ax, xs, ys, cis, color, label, scale=1, upper=None, markers=True):
    """Line with its confidence band, clipped to [0, upper]"""
    ys, cis = np.array(ys, dtype=float), np.array(cis, dtype=float).reshape(-1, 2)
    lo = np.maximum(cis[:, 0], 0)
    hi = cis[:, 1] if upper is None else np.minimum(cis[:, 1], upper)
    ok = np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(np.array(xs)[ok], scale * lo[ok], scale * hi[ok], color=color, alpha=0.12, linewidth=0)
    kw = dict(marker='o', markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5) if markers else {}
    ax.plot(xs, scale * ys, color=color, linewidth=2, label=label, zorder=3, **kw)


def plot_vs_agents(ax, data, methods, colors, matrix, name, title, ylabel, percent=False, upper=None, ideal=None):
    """
    One line per method: estimate and 95 % CI of matrix(method, n), for each
    number of robots. The points are recorded as panel name.
    """
    agents = data.agents
    for m in methods:
        xs, ys, cis = [], [], []
        for n in agents:
            X = matrix(m, n) if data.has(m, n) else None
            if X is None:
                continue
            est, ci = mean_ci(X)
            data.rec.add('points', map_set=data.map_set, panel=name, method=m, num_agents=n, estimate=est,
                         ci_low=ci[0], ci_high=ci[1], seeds=X.shape[0], maps=int(np.isfinite(X).any(axis=0).sum()))
            if np.isfinite(est):
                xs.append(n)
                ys.append(est)
                cis.append(ci)
        if xs:
            plot_line(ax, xs, ys, cis, colors[m], m, 100 if percent else 1, upper)
    if ideal is not None:
        ax.plot(agents, ideal(np.array(agents)), color=MUTED, linewidth=1, zorder=2)
        ax.annotate('ideal', (agents[-1], ideal(np.array(agents[-1:]))[0]), textcoords='offset points',
                    xytext=(-4, 4), ha='right', fontsize=9, color=MUTED)
    ax.set_xscale('log', base=2)
    ax.set_xticks(agents)
    ax.set_xticklabels([str(n) for n in agents])
    ax.minorticks_off()
    style_axes(ax, title, ylabel, 'robots')


def reached_by_most(X):
    """Whether most maps have a value (the curves stop where most maps did not get)"""
    return np.mean(np.isfinite(X).any(axis=0)) >= 0.5


def plot_overlap_vs_coverage(ax, data, methods, colors, n):
    for m in methods:
        if not data.has(m, n):
            continue
        xs, ys, cis = [], [], []
        for c in LEVELS[4::2]:
            X = data.curve_matrix(m, n, f'o{c}')
            if reached_by_most(X):
                est, ci = mean_ci(X)
                data.rec.add('curves', map_set=data.map_set, curve='overlap_vs_coverage', method=m, num_agents=n,
                             coverage=c / 100, value=est, ci_low=ci[0], ci_high=ci[1])
                xs.append(c)
                ys.append(est)
                cis.append(ci)
        if xs:
            plot_line(ax, xs, ys, cis, colors[m], m, 100, markers=False)
    style_axes(ax, f'Overlap while covering, {n} robots', 'covered again / covered [%]', 'coverage [%]')


def plot_coverage_vs_time(ax, data, methods, colors, n):
    for m in methods:
        if not data.has(m, n):
            continue
        xs, ys = [], []
        for c in LEVELS:
            # median over maps (the maps have very different sizes) of the
            # mean over seeds, a map that a seed did not cover to c counts as
            # never reaching it, so the curve is monotonic
            X = data.curve_matrix(m, n, f't{c}')
            t = np.median(np.where(np.isfinite(X), X, np.inf).mean(axis=0))
            if np.isfinite(t):
                data.rec.add('curves', map_set=data.map_set, curve='median_time_to_coverage_min', method=m,
                             num_agents=n, coverage=c / 100, value=t / 60)
                xs.append(t / 60)
                ys.append(c)
        if xs:
            ax.plot(xs, ys, color=colors[m], linewidth=2, label=m)
    ax.set_ylim(0, 101)
    style_axes(ax, f'Coverage over time, {n} robots (median map)', 'coverage [%]', 'time [min]')


def fmt_ci(est, ci, k=1, digits=1, unit='', upper=None):
    """estimate [CI], the values are >= 0 and <= upper (if given)"""
    if not np.isfinite(est):
        return '-'
    lo, hi = max(ci[0], 0), ci[1] if upper is None else min(ci[1], upper)
    return f'{k * est:.{digits}f} [{k * lo:.{digits}f}, {k * hi:.{digits}f}]{unit}'


def draw_table(ax, cells, columns, colors=None, widths=None, bold=None, fontsize=9.5, scale_y=1.6):
    """Plain table, a color swatch before the first cell of each row (if colors)"""
    ax.axis('off')
    table = ax.table(cellText=cells, colLabels=columns, colWidths=widths, loc='upper center', cellLoc='left',
                     colLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, scale_y)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        cell.set_facecolor(SURFACE)
        cell.get_text().set_color(INK if i > 0 else INK_2)
        if i == 0 or (bold is not None and bold[i - 1][j]):
            cell.get_text().set_fontweight('bold')
    for i, c in enumerate(colors or [], start=1):
        cell = table[i, 0]
        text = cell.get_text().get_text()
        cell.get_text().set_text('')
        ax.annotate('■', (0, 0.5), xycoords=cell, xytext=(6, 0), textcoords='offset points',
                    va='center', color=c, fontsize=12)
        ax.annotate(text, (0, 0.5), xycoords=cell, xytext=(22, 0), textcoords='offset points',
                    va='center', color=INK, fontsize=fontsize)


def summary_table(ax, data, methods, colors, n):
    p = data.p
    columns = ['Method', 'Comm.', 'Seeds', f'Success ({p} %)', f'Time to {p} %', f'Overlap at {p} %',
               'Robot coll.', 'Imbalance', 'Belief recall', 'Connectivity']
    cells, row_colors = [], []
    for m in methods:
        if not data.has(m, n):
            continue
        comm = data.comm_range(m)

        def metric(key):
            X = data.metric(m, n, key)
            est, ci = mean_ci(X)
            data.rec.add('points', map_set=data.map_set, panel=f'summary_{key}', method=m, num_agents=n,
                         estimate=est, ci_low=ci[0], ci_high=ci[1], seeds=X.shape[0],
                         maps=int(np.isfinite(X).any(axis=0).sum()))
            return est, ci
        cells.append([
            m, 'unlimited' if comm == 0 else f'{comm:g} m', str(len(data.seeds[m])),
            fmt_ci(*metric('success'), 100, 0, ' %', upper=1), fmt_ci(*metric('time'), 1, 1, ' min'),
            fmt_ci(*metric('overlap_at'), 100, 1, ' %'), fmt_ci(*metric('robot_collisions'), 1, 1),
            fmt_ci(*metric('imbalance'), 1, 2), fmt_ci(*metric('belief_recall'), 100, 0, ' %', upper=1),
            fmt_ci(*metric('connectivity'), 100, 0, ' %', upper=1)])
        row_colors.append(colors[m])
    if cells:
        draw_table(ax, cells, columns, row_colors, widths=[0.15, 0.06, 0.04] + [0.75 / 7] * 7)
    ax.set_title(f'Summary, {n} robots: mean [95 % CI]', loc='left', fontsize=12, color=INK, fontweight='bold')


def header(fig, data, what, colors, plt):
    counts = sorted({len(s) for s in data.seeds.values()})
    seeds = f'{counts[0]}' if len(counts) == 1 else f'{counts[0]}-{counts[-1]}'
    fig.text(0.05, 0.965, f'Multi-agent coverage, {data.map_set}: {what}', fontsize=20, color=INK,
             fontweight='bold')
    fig.text(0.05, 0.948, f'{len(data.maps)} maps x {data.info.get("episodes", 1)} episode(s) x {seeds} training '
                          f'seed(s) per method and team size, deterministic actions, same start poses for every '
                          f'policy. Mean over maps of the mean over seeds, 95 % CI over seeds and maps.',
             fontsize=11, color=INK_2)
    handles = [plt.Line2D([], [], color=colors[m], linewidth=2, marker='o', markersize=6,
                          markeredgecolor=SURFACE, markeredgewidth=1.5) for m in data.methods]
    labels = [m if data.comm_range(m) == 0 else f'{m}  (comm. {data.comm_range(m):g} m)' for m in data.methods]
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.05, 0.94), ncol=min(len(labels), 6),
               frameon=False, fontsize=11, labelcolor=INK, handlelength=2.2, columnspacing=2.5)


def overview_page(fig, data, colors, ref_agents, plt):
    methods = data.methods
    p = data.p
    base = data.agents[0]
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.5], hspace=0.42, wspace=0.22,
                          left=0.05, right=0.98, top=0.9, bottom=0.02)
    ax = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]
    metric = lambda key: (lambda m, n: data.metric(m, n, key))

    plot_vs_agents(ax[0][0], data, methods, colors, metric('time'), f'time_to_{p}_min',
                   f'Time to {p} % coverage', 'time [min]  (not reached: step limit)')
    plot_vs_agents(ax[0][1], data, methods, colors, metric('overlap_at'), f'overlap_at_{p}',
                   f'Overlap when {p} % coverage is reached', 'covered again / covered [%]', percent=True)

    def speedup(m, n):
        # same policy and map: ratio of the times of the smallest team and of n robots
        if not data.has(m, base):
            return None
        return data.metric(m, base, 'time') / data.metric(m, n, 'time')
    plot_vs_agents(ax[0][2], data, methods, colors, speedup, f'speedup_vs_{base}_robots',
                   f'Speedup over {base} robot{"s" if base > 1 else ""}', f'time ratio (to {p} %)',
                   ideal=lambda n: n / base)

    plot_overlap_vs_coverage(ax[1][0], data, methods, colors, ref_agents)
    plot_coverage_vs_time(ax[1][1], data, methods, colors, ref_agents)
    plot_vs_agents(ax[1][2], data, methods, colors, metric('success'), f'success_{p}',
                   f'Success rate ({p} % coverage)', 'episodes [%]', percent=True, upper=1)
    ax[1][2].set_ylim(-3, 103)

    plot_vs_agents(ax[2][0], data, methods, colors, metric('robot_collisions'), 'robot_collisions',
                   'Collisions between robots', 'per episode')
    plot_vs_agents(ax[2][1], data, methods, colors, metric('imbalance'), 'imbalance',
                   'Workload imbalance', 'CV of area covered per robot')
    limited = [m for m in methods if data.comm_range(m) > 0]
    plot_vs_agents(ax[2][2], data, limited, colors, metric('belief_recall'), 'belief_recall',
                   'Belief recall (limited communication)', 'covered area known to each robot [%]',
                   percent=True, upper=1)
    ax[2][2].set_ylim(top=102)

    summary_table(fig.add_subplot(gs[3, :]), data, methods, colors, ref_agents)
    header(fig, data, 'method comparison', colors, plt)


def tests_page(fig, data, colors, plt):
    pairs = list(itertools.combinations(data.methods, 2))
    agents = data.agents
    gs = fig.add_gridspec(len(TESTS) + 1, 1, height_ratios=[0.3] + [1] * len(TESTS), hspace=0.55,
                          left=0.05, right=0.98, top=0.9, bottom=0.02)
    note = fig.add_subplot(gs[0])
    note.axis('off')
    note.text(0, 1, 'Each cell: difference B - A [95 % CI], Holm-corrected p-value (within each metric), and '
                    'P(B better): probability that B beats A on a random map, for a random seed of each '
                    '(0.5 = no difference).\nBold: significant at the 5 % level after the correction. A single '
                    'seed gives no p-value. Policies evaluated twice (same checkpoints) are paired by seed.',
              va='top', fontsize=10.5, color=INK_2, transform=note.transAxes, linespacing=1.6)
    for t, (key, title, lower, mode) in enumerate(TESTS):
        results = {}
        for a, b in pairs:
            for n in agents:
                if data.has(a, n) and data.has(b, n):
                    results[(a, b, n)] = compare(data.metric(a, n, key), data.metric(b, n, key),
                                                 paired_seeds=data.paired(a, b), lower_is_better=lower)
        adjusted = dict(zip(results, holm([r['p'] for r in results.values()])))
        for (a, b, n), r in results.items():
            q = adjusted[(a, b, n)]
            data.rec.add('tests', map_set=data.map_set, metric=key, target_coverage=data.p / 100,
                         lower_is_better=lower, A=a, B=b, num_agents=n, paired_seeds=data.paired(a, b),
                         seeds_A=len(data.seeds[a]), seeds_B=len(data.seeds[b]), estimate_A=r['a'], estimate_B=r['b'],
                         diff=r['diff'], diff_ci_low=r['diff_ci'][0], diff_ci_high=r['diff_ci'][1], rel=r['rel'],
                         rel_ci_low=r['rel_ci'][0], rel_ci_high=r['rel_ci'][1], p=r['p'], p_holm=q,
                         p_better=r['p_better'], significant=bool(np.isfinite(q) and q < 0.05))
        cells, bold = [], []
        for a, b in pairs:
            row, brow = [f'{b}  vs  {a}'], [False]
            for n in agents:
                r = results.get((a, b, n))
                if r is None or not np.isfinite(r['diff']):
                    row.append('-')
                    brow.append(False)
                    continue
                if mode == 'rel':
                    d = f'{100 * r["rel"]:+.1f} % [{100 * r["rel_ci"][0]:+.1f}, {100 * r["rel_ci"][1]:+.1f}]'
                elif mode == 'pp':
                    d = f'{100 * r["diff"]:+.1f} pp [{100 * r["diff_ci"][0]:+.1f}, {100 * r["diff_ci"][1]:+.1f}]'
                else:
                    d = f'{r["diff"]:+.2f} [{r["diff_ci"][0]:+.2f}, {r["diff_ci"][1]:+.2f}]'
                q = adjusted[(a, b, n)]
                ps = 'p n/a' if not np.isfinite(q) else ('p<0.001' if q < 0.001 else f'p={q:.3f}')
                row.append(f'{d}   {ps}   P(B better) {r["p_better"]:.2f}')
                brow.append(bool(np.isfinite(q) and q < 0.05))
            cells.append(row)
            bold.append(brow)
        ax = fig.add_subplot(gs[t + 1])
        if cells:
            draw_table(ax, cells, ['B  vs  A'] + [f'{n} robots' for n in agents], bold=bold,
                       widths=[0.16] + [0.84 / len(agents)] * len(agents), fontsize=8.5, scale_y=1.45)
        else:
            ax.axis('off')
        unit = {'rel': 'relative difference', 'pp': 'percentage points', 'abs': 'difference'}[mode]
        target = f' ({data.p} %)' if key in ('time', 'overlap_at', 'success') else ''
        ax.set_title(f'{title}{target}: {unit}, {"lower" if lower else "higher"} is better', loc='left',
                     fontsize=12, color=INK, fontweight='bold')
    header(fig, data, 'statistical tests', colors, plt)


def draw_dashboard(data_dir, out, ref_agents=None, map_sets=None, threshold=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = FONT

    episodes, curves, meta, dirs = load(data_dir)
    assert episodes, f'no data in {data_dir}'
    all_methods = list(meta.get('methods', {}))
    all_methods += sorted({r['method'] for r in episodes} - set(all_methods))
    colors = {m: SERIES[i % len(SERIES)] for i, m in enumerate(all_methods)}
    sets = map_sets or list(dict.fromkeys(r['map_set'] for r in episodes))
    recorder = Recorder()
    info = dict(data_dirs=[os.path.abspath(d) for d in dirs], pdf=os.path.abspath(out), created=time.strftime('%Y-%m-%d %H:%M'),
                units='fractions (not percent), times in minutes', map_sets={})

    with PdfPages(out) as pdf:
        for s in sets:
            data = Data(episodes, curves, meta, s, threshold, recorder)
            if not data.episodes:
                print(f'no data for map set {s}')
                continue
            n_ref = ref_agents if ref_agents in data.agents else (4 if 4 in data.agents else data.agents[-1])
            info['map_sets'][s] = dict(target_coverage=data.p / 100, ref_agents=n_ref, maps=len(data.maps),
                                       num_agents=data.agents, methods=data.methods, seeds=data.seeds,
                                       step_limit_min=data.max_time / 60)
            for page, size in ((lambda f: overview_page(f, data, colors, n_ref, plt), (18, 19)),
                               (lambda f: tests_page(f, data, colors, plt), (18, 22))):
                fig = plt.figure(figsize=size, facecolor=SURFACE)
                page(fig)
                pdf.savefig(fig, facecolor=SURFACE)
                plt.close(fig)
    data_out = os.path.splitext(out)[0] + '_data'
    recorder.save(data_out, info)
    return data_out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='results/eval', type=str, help='directory written by collect_marl.py')
    parser.add_argument('--out', default=None, type=str, help='PDF file (default: <data>/dashboard.pdf)')
    parser.add_argument('--ref_agents', default=None, type=int, help='number of robots of the curves and table')
    parser.add_argument('--map_sets', nargs='+', default=None, help='map sets to show (default: all)')
    parser.add_argument('--threshold', default=None, type=float,
                        help='target coverage 0.9, 0.95 or 0.99 (default: goal coverage of each map set)')
    args = parser.parse_args()
    out = args.out or os.path.join(args.data, 'dashboard.pdf')
    data_out = draw_dashboard(args.data, out, args.ref_agents, args.map_sets, args.threshold)
    print('dashboard:', out)
    print('data shown in the dashboard:', data_out)


if __name__ == '__main__':
    main()
