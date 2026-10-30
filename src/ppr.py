"""Method A: Personalized PageRank on a show-show co-view graph.

Graph nodes = every show in the split's TRAINING interactions (pool or not). Users x shows weight
matrix B (softfrac | count08 | log_softfrac); edges from B01 = (B > 0):
  raw              C = B01^T B01, diagonal removed (users who watched both)
  cosine           C_ij / sqrt(n_i n_j), n_i = users of show i
  weighted_cosine  cosine of the weighted columns of B
T = row-normalised C. Isolated nodes (no edge) get a self-loop T_ii = 1 so every row sums to 1.
PPR per user u: R_{t+1} = (1 - d) p_u + d R_t T, R_0 = p_u (p_u = L1-normalised seed weights),
d = continue probability; up to 50 iterations, stop when max_u ||R_{t+1} - R_t||_1 < 1e-6.
Score s_ui = R_u[i] / max(deg_i, 1) ** beta; candidates = pool shows with s > 0, ranked by s desc,
tv_show_id asc; padded with the protocol's popularity list. The graph is dense: 2,973 (val) and
4,363 (test) nodes give 71 MB / 152 MB in float64, far below 1 GB.
"""
import time
from dataclasses import dataclass
from itertools import islice

import numpy as np
import pandas as pd
import scipy.sparse as sps

from src.protocol import K, THR, pad_with_popularity


def user_show_weights(train: pd.DataFrame, kind: str) -> pd.Series:
    """Series indexed (user_id, tv_show_id) of positive weights."""
    if kind in ("softfrac", "log_softfrac"):
        w = train.frac.astype(float)
    elif kind == "count08":
        w = (train.frac >= THR).astype(float)
    else:
        raise ValueError(kind)
    s = w.groupby([train.user_id, train.tv_show_id]).sum()
    if kind == "log_softfrac":
        s = np.log1p(s)
    return s[s > 0]


@dataclass
class Graph:
    shows: np.ndarray          # node ids, sorted int64
    T: np.ndarray              # dense row-stochastic transition matrix
    deg: np.ndarray            # number of neighbours (unweighted degree)
    n_users_per_show: np.ndarray
    n_isolated: int
    n_edges: int               # undirected, diagonal excluded
    build_seconds: float
    weights: pd.Series         # (user_id, tv_show_id) -> seed weight
    users: np.ndarray          # uint64 users with >= 1 positive weight (row order of B)
    B: sps.csr_matrix          # users x shows weights


def build_graph(train: pd.DataFrame, edge: str, weight: str, dtype=np.float64) -> Graph:
    t0 = time.perf_counter()
    shows = np.sort(train.tv_show_id.unique()).astype(np.int64)
    w = user_show_weights(train, weight)
    u = w.index.get_level_values(0).to_numpy(np.uint64)
    s = w.index.get_level_values(1).to_numpy(np.int64)
    users, ui = np.unique(u, return_inverse=True)
    si = np.searchsorted(shows, s)
    B = sps.csr_matrix((w.to_numpy(float), (ui, si)), shape=(len(users), len(shows)))
    B01 = (B > 0).astype(np.float64)
    n_i = np.asarray(B01.sum(axis=0)).ravel()
    if edge == "raw":
        C = (B01.T @ B01).toarray()
    elif edge == "cosine":
        C = (B01.T @ B01).toarray()
        nrm = np.sqrt(n_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            C = np.nan_to_num(C / np.outer(nrm, nrm))
    elif edge == "weighted_cosine":
        C = (B.T @ B).toarray()
        nrm = np.sqrt(np.diag(C).copy())
        with np.errstate(divide="ignore", invalid="ignore"):
            C = np.nan_to_num(C / np.outer(nrm, nrm))
    else:
        raise ValueError(edge)
    np.fill_diagonal(C, 0.0)
    deg = (C > 0).sum(axis=1)
    rs = C.sum(axis=1)
    iso = rs == 0
    C[iso, iso] = 1.0          # self-loop for isolated nodes
    rs[iso] = 1.0
    T = (C / rs[:, None]).astype(dtype)
    return Graph(shows, T, deg, n_i, int(iso.sum()), int((deg.sum()) // 2), time.perf_counter() - t0,
                 w, users, B)


def personalisation(g: Graph, users: np.ndarray, dtype=np.float64):
    """Dense P (len(users) x N) of L1-normalised seed weights; has_seed flags users with any seed."""
    row = {int(x): i for i, x in enumerate(g.users)}
    P = np.zeros((len(users), len(g.shows)), dtype=dtype)
    has = np.zeros(len(users), bool)
    for k, x in enumerate(users):
        i = row.get(int(x))
        if i is None:
            continue
        r = g.B.getrow(i)
        P[k, r.indices] = r.data / r.data.sum()
        has[k] = True
    return P, has


def ppr(P: np.ndarray, T: np.ndarray, d: float, max_iter=50, tol=1e-6, n_iter=None, batch=1000):
    """Batched power iteration. n_iter fixes the number of iterations (for linearity checks)."""
    R = np.empty_like(P)
    iters, deltas = [], []
    steps = n_iter if n_iter is not None else max_iter
    for s in range(0, len(P), batch):
        p = P[s:s + batch]
        r = p.copy()
        it, delta = 0, np.inf
        for it in range(1, steps + 1):
            r_new = (1 - d) * p + d * (r @ T)
            delta = float(np.abs(r_new - r).sum(axis=1).max()) if len(p) else 0.0
            r = r_new
            if n_iter is None and delta < tol:
                break
        R[s:s + batch] = r
        iters.append(it)
        deltas.append(delta)
    return R, {"iterations_per_batch": iters, "max_iterations": max(iters) if iters else 0,
               "final_max_l1_change": max(deltas) if deltas else 0.0, "converged": max(deltas, default=0) < tol}


def lazy_pop(pop, exclude, k=K):
    """First k ids of the popularity ranking not in `exclude` (for new_only padding)."""
    return list(islice((x for x in pop if int(x) not in exclude), k))


def recommend(R, has_seed, users, g: Graph, pool, beta, pop, mode="full", history=None, k=K, batch=1000):
    """Returns (recs: int(user) -> k ids, n_padded: int(user) -> count, raw_top: int(user) -> PPR ids)."""
    in_pool = np.isin(g.shows, pool)
    pen = np.maximum(g.deg, 1).astype(float) ** beta
    col = {int(x): i for i, x in enumerate(g.shows)}
    recs, padded, raw = {}, {}, {}
    for s in range(0, len(users), batch):
        S = R[s:s + batch] / pen
        S[:, ~in_pool] = 0
        ub = users[s:s + batch]
        if mode == "new_only":
            for j, x in enumerate(ub):
                h = [col[t] for t in history.get(int(x), ()) if t in col]
                S[j, h] = 0
        order = np.argsort(-S, axis=1, kind="stable")[:, :k]   # nodes sorted by id -> ties by id asc
        top_s = np.take_along_axis(S, order, axis=1)
        for j, x in enumerate(ub):
            x = int(x)
            lst = [int(g.shows[c]) for c, v in zip(order[j], top_s[j]) if v > 0] if has_seed[s + j] else []
            fill = pop if mode == "full" else lazy_pop(pop, history.get(x, set()) | set(lst), k)
            recs[x] = pad_with_popularity(lst, fill, k)
            padded[x] = k - len(lst)
            raw[x] = lst
    return recs, padded, raw
