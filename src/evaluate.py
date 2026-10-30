"""MAP@K evaluator and bootstrap intervals.

The competition page only names the metric "MAP@5". We ASSUME the standard Kaggle definition
(as in ml_metrics.apk):
    AP@K = sum_{i=1..K} P(i) * rel(i) / min(K, |truth|)
    rel(i) = 1 if the i-th recommended id is in the truth set and did not appear earlier in the list
             (a repeated id counts once), P(i) = hits within the top i / i.
The order of the truth list is irrelevant. MAP@K = mean AP@K over users.
"""
import numpy as np
import pandas as pd


def ap_at_k(recommended, truth, k=5) -> float:
    truth = set(truth)
    if not truth:
        return 0.0
    seen, hits, s = set(), 0, 0.0
    for i, r in enumerate(list(recommended)[:k], start=1):
        if r in truth and r not in seen:
            hits += 1
            s += hits / i
        seen.add(r)
    return s / min(k, len(truth))


def map_at_k(recs: dict, truth: dict, k=5):
    """Mean AP over the users in `truth` (users missing from recs score 0). Returns (mean, per-user Series)."""
    ap = pd.Series({u: ap_at_k(recs.get(u, []), t, k) for u, t in truth.items()}, name="ap", dtype=float)
    return float(ap.mean()), ap


def _boot_idx(n_users, n, seed):
    return np.random.default_rng(seed).integers(0, n_users, size=(n, n_users))


def bootstrap_ci(ap: pd.Series, n=1000, seed=42):
    """95% percentile interval of the mean, resampling users with replacement."""
    v = ap.to_numpy()
    means = v[_boot_idx(len(v), n, seed)].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_diff(ap_a: pd.Series, ap_b: pd.Series, n=1000, seed=42):
    """Mean of (a - b) over common users and its 95% percentile interval (same user resample for both)."""
    a, b = ap_a.align(ap_b, join="inner")
    d = (a - b).to_numpy()
    means = d[_boot_idx(len(d), n, seed)].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
