"""
Statistics for comparing methods trained with several seeds and evaluated on
many maps.

The data of one method is a (seeds x maps) matrix X: X[s, m] is the metric of
the policy trained with seed s on map m (mean over the episodes on that map),
nan where it is not defined (e.g. overlap at 99 % coverage when 99 % was not
reached). The point estimate is the mean over maps of the mean over seeds.

The uncertainty has two sources, the training seeds and the maps (the
population of maps the test maps are drawn from), so the variance of an
estimate is estimated as
    var(seed scores) / seeds + var(map scores) / maps
where a seed score is the mean over the maps of one trained policy and a map
score the mean over the seeds on one map. Confidence intervals and p-values
use the t distribution with Welch-Satterthwaite degrees of freedom: with few
seeds, a bootstrap underestimates the variance between seeds (with 5 seeds, 14 %
false positives at the 5 % level in a simulation, see tests/test_marl.py),
this test keeps the false positive rate at or below the nominal level.

In a comparison the maps are paired (the variance of the per map
differences is used), and the seeds are paired only if both methods use the
same trained policies (e.g. one policy evaluated with two communication
ranges). With a single seed the variance between seeds is unknown: confidence
intervals then only contain the map variance and no p-value is given.

The probability of improvement (Agarwal et al., Deep Reinforcement Learning at
the Edge of the Statistical Precipice, NeurIPS 2021) is the probability that
method B is better than method A on a random map, for a random trained policy
of each.
"""
import warnings

import numpy as np
from scipy import stats


def _nanmean(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(x, axis=axis)


def _var_of_mean(x):
    """(variance of the mean of the finite values of x, degrees of freedom)"""
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, 0
    return np.var(x, ddof=1) / len(x), len(x) - 1


def _t_interval(est, components, alpha):
    """
    components: [(variance, dof)], nan variances (unknown, e.g. one seed) are
    left out. Returns (low, high), p-value of est = 0, standard error, dof.
    """
    known = [(v, d) for v, d in components if np.isfinite(v) and d > 0]
    se2 = sum(v for v, _ in known)
    if not known:
        return (est, est), np.nan, 0.0, 0
    if se2 <= 0:
        # no variance at all (e.g. identical policies)
        return (est, est), 1.0 if est == 0 else 0.0, 0.0, 0
    dof = se2 ** 2 / sum(v ** 2 / d for v, d in known if v > 0)
    se = np.sqrt(se2)
    t = stats.t.ppf(1 - alpha / 2, dof)
    p = 2 * stats.t.sf(abs(est) / se, dof)
    return (est - t * se, est + t * se), p, se, dof


def estimate(X):
    """Mean over maps of the mean over seeds, ignoring nan"""
    return float(_nanmean(_nanmean(np.asarray(X, dtype=float), axis=0)))


def mean_ci(X, alpha=0.05):
    """Estimate and (low, high) 1 - alpha confidence interval of one method, see the module docstring"""
    X = np.asarray(X, dtype=float)
    est = estimate(X)
    if not np.isfinite(est):
        return np.nan, (np.nan, np.nan)
    components = [_var_of_mean(_nanmean(X, axis=1)), _var_of_mean(_nanmean(X, axis=0))]
    ci, _, _, _ = _t_interval(est, components, alpha)
    return est, ci


def compare(XA, XB, paired_seeds=False, alpha=0.05, lower_is_better=True):
    """
    Compares method B with method A on the same maps (columns of XA, XB).

    Returns a dict:
      a, b:      estimates of A and B
      diff:      b - a, with the (low, high) confidence interval diff_ci
      rel:       (b - a) / |a|, with rel_ci (the uncertainty of a is ignored)
      p:         two-sided p-value of diff = 0 (nan with a single seed)
      p_better:  probability of improvement of B over A (0.5: no difference)
    """
    XA = np.asarray(XA, dtype=float)
    XB = np.asarray(XB, dtype=float)
    assert XA.shape[1] == XB.shape[1], 'methods must be evaluated on the same maps'
    a, b = estimate(XA), estimate(XB)
    nan = (np.nan, np.nan)
    if not (np.isfinite(a) and np.isfinite(b)):
        return dict(a=a, b=b, diff=np.nan, diff_ci=nan, rel=np.nan, rel_ci=nan, p=np.nan, p_better=np.nan)
    d = b - a
    map_diff = _nanmean(XB, axis=0) - _nanmean(XA, axis=0)
    if paired_seeds:
        assert XA.shape[0] == XB.shape[0]
        seed_parts = [_var_of_mean(_nanmean(XB - XA, axis=1))]
    else:
        seed_parts = [_var_of_mean(_nanmean(XA, axis=1)), _var_of_mean(_nanmean(XB, axis=1))]
    ci, p, _, _ = _t_interval(d, seed_parts + [_var_of_mean(map_diff)], alpha)
    if any(not np.isfinite(v) for v, _ in seed_parts):
        p = np.nan
    scale = 1 / abs(a) if a != 0 else np.nan
    return dict(a=a, b=b, diff=d, diff_ci=ci, rel=d * scale, rel_ci=(ci[0] * scale, ci[1] * scale), p=p,
                p_better=prob_better(XA, XB, lower_is_better, paired_seeds))


def prob_better(XA, XB, lower_is_better=True, paired_seeds=False):
    """
    Probability that B beats A on a random map, for a random seed of each (for
    the same seed if paired), ties count 1/2
    """
    if paired_seeds:
        da, db = XA, XB                                                     # S x M
    else:
        da, db = XA[:, None, :], XB[None, :, :]                             # Sa x Sb x M
    better = (db < da) if lower_is_better else (db > da)
    valid = np.isfinite(da) & np.isfinite(db)
    score = np.where(valid, better + 0.5 * (db == da), np.nan)
    per_map = _nanmean(score.reshape(-1, XA.shape[1]), axis=0)
    return float(_nanmean(per_map))


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order), nan stays nan"""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.nonzero(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out
