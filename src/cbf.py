"""Method B: content-based filtering on schedule metadata.

Item vectors for every show in the split's metadata table, built from blocks that are each
L2-normalised, multiplied by a block weight, concatenated and L2-normalised again:
  category  one-hot
  genres    multi-hot over genre_1..3 with position weights 1.0 / 0.7 / 0.5 (genre_1 is primary)
  year      decade one-hot from the first 4 digits of year_of_production ('1996-', '2006 - ???')
  director  comma-separated tokens, lowercased, binary tf x idf, idf = ln((1 + N) / (1 + df)) + 1
  actors    same as director
  title     lowercase words of >= 3 letters (digits and 1-2 letter fragments such as 'с.', 'эп', 'т/с'
            dropped) minus episode noise words, tf-idf as above
Missing metadata -> zero block. A show whose vector is all zero keeps a zero vector (cosine 0).
User profile = sum_s w_us * x_s over the user's training shows, L2-normalised.
Score = cosine(profile, x_s) over pool shows (all features are >= 0, so cosine is in [0, 1]).
Ties: pool columns are ordered by the pop_recent ranking (which itself is score desc, tv_show_id asc),
and a stable sort on the score (rounded to 12 decimals) keeps that order, i.e. ties -> pop_recent rank asc
-> tv_show_id asc.
cbf_pop = (1 - lam) * cosine + lam * pop_norm is CBF WITH A POPULARITY PRIOR (a blend, not pure CBF).
"""
import math
import re

import numpy as np
import pandas as pd
import scipy.sparse as sps

from src.ppr import lazy_pop
from src.protocol import K, THR, pad_with_popularity

FEATURE_SETS = {"F1": ["category", "genres"], "F2": ["category", "genres", "year"],
                "F3": ["category", "genres", "year", "director", "actors"],
                "F4": ["category", "genres", "year", "director", "actors", "title"]}
GENRE_POS = (1.0, 0.7, 0.5)
ROUND = 12  # scores are rounded to this many decimals before ranking (ties -> pop rank)
TITLE_NOISE = {"сезон", "сезона", "сезоне", "серия", "серии", "выпуск", "часть", "эпизод", "эпизоды"}


def parse_decade(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    m = re.match(r"\s*(\d{4})", str(s))
    return f"{int(m.group(1)) // 10 * 10}s" if m else None


def list_tokens(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    return sorted({t.strip().lower() for t in str(s).split(",") if t.strip()})


def title_tokens(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    return sorted({w for w in re.findall(r"[^\W\d_]{3,}", str(s).lower()) if w not in TITLE_NOISE})


def raw_block(meta: pd.DataFrame, block: str):
    """List (one per show, meta order) of {token: raw value} before normalisation."""
    out = []
    for r in meta.itertuples():
        if block == "category":
            d = {r.category: 1.0} if isinstance(r.category, str) else {}
        elif block == "genres":
            d = {}
            for g, w in zip([r.genre_1, r.genre_2, r.genre_3], GENRE_POS):
                if isinstance(g, str) and g.strip():
                    d[g.strip().lower()] = d.get(g.strip().lower(), 0.0) + w
        elif block == "year":
            dec = parse_decade(r.year)
            d = {dec: 1.0} if dec else {}
        elif block in ("director", "actors"):
            d = {t: 1.0 for t in list_tokens(getattr(r, block))}
        elif block == "title":
            d = {t: 1.0 for t in title_tokens(r.title)}
        else:
            raise ValueError(block)
        out.append(d)
    if block in ("director", "actors", "title"):
        n = len(out)
        df = {}
        for d in out:
            for t in d:
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        out = [{t: v * idf[t] for t, v in d.items()} for d in out]
    return out


def to_csr(dicts, prefix):
    vocab = sorted({t for d in dicts for t in d})
    col = {t: i for i, t in enumerate(vocab)}
    rows, cols, vals = [], [], []
    for i, d in enumerate(dicts):
        for t, v in d.items():
            rows.append(i)
            cols.append(col[t])
            vals.append(v)
    M = sps.csr_matrix((vals, (rows, cols)), shape=(len(dicts), len(vocab)))
    return M, [f"{prefix}:{t}" for t in vocab]


def l2_rows(M):
    n = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    n[n == 0] = 1.0
    return sps.diags(1.0 / n) @ M


class Items:
    def __init__(self, meta: pd.DataFrame, feature_set, block_weights=None):
        blocks = FEATURE_SETS[feature_set] if isinstance(feature_set, str) else list(feature_set)
        bw = {b: 1.0 for b in blocks}
        bw.update(block_weights or {})
        self.ids = meta.index.to_numpy(np.int64)
        self.row = {int(x): i for i, x in enumerate(self.ids)}
        mats, names, self.blocks = [], [], {}
        start = 0
        for b in blocks:
            M, nm = to_csr(raw_block(meta, b), b)
            mats.append(l2_rows(M) * bw[b])
            names += nm
            self.blocks[b] = (start, start + len(nm))
            start += len(nm)
        self.X = l2_rows(sps.hstack(mats).tocsr()).tocsr()
        self.names = names
        self.zero = np.asarray(self.X.multiply(self.X).sum(axis=1)).ravel() == 0


def user_weights(train, kind, decay_hl, train_end):
    if kind == "softfrac":
        w = train.frac.astype(float)
    elif kind == "count08":
        w = (train.frac >= THR).astype(float)
    else:
        raise ValueError(kind)
    if decay_hl:
        age = (train_end - train.airing_start).dt.total_seconds() / (7 * 86400)
        w = w * 0.5 ** (age / decay_hl)
    s = w.groupby([train.user_id, train.tv_show_id]).sum()
    return s[s > 0]


def profiles(items: Items, train, users, kind, decay_hl, train_end):
    """Sparse L2-normalised profiles aligned with `users`; has flags users with a profile."""
    w = user_weights(train, kind, decay_hl, train_end)
    uidx = {int(u): i for i, u in enumerate(users)}
    u = [uidx.get(int(x)) for x in w.index.get_level_values(0)]
    s = [items.row.get(int(x)) for x in w.index.get_level_values(1)]
    keep = [a is not None and b is not None for a, b in zip(u, s)]
    W = sps.csr_matrix((w.to_numpy()[keep], (np.array(u, dtype=object)[keep].astype(int),
                                            np.array(s, dtype=object)[keep].astype(int))),
                       shape=(len(users), len(items.ids)))
    Pm = W @ items.X
    has = np.asarray(Pm.multiply(Pm).sum(axis=1)).ravel() > 0
    return l2_rows(Pm).tocsr(), has, W


def cosine_scores(P, items: Items, pool_by_pop):
    """Dense users x pool cosine matrix; columns in pop_recent rank order."""
    Xp = items.X[[items.row[int(x)] for x in pool_by_pop]]
    return (P @ Xp.T).toarray()


def top_k_stable(B, k):
    """Same result as np.argsort(-B, kind='stable')[:, :k]: candidates with score >= the k-th largest
    value, kept in column order, then stable-sorted by score (ties keep the column = pop-rank order)."""
    thr = -np.partition(-B, k - 1, axis=1)[:, k - 1]
    out = np.empty((len(B), k), dtype=np.int64)
    for i in range(len(B)):
        idx = np.flatnonzero(B[i] >= thr[i])
        out[i] = idx[np.argsort(-B[i, idx], kind="stable")[:k]]
    return out


def rank_lists(S, has, users, pool_by_pop, pop, mode="full", history=None, k=K, batch=1000):
    """Top-k per user from score matrix S (columns in pop rank order). Returns recs, padded, tie flags."""
    recs, padded, tie = {}, {}, {}
    col = {int(x): j for j, x in enumerate(pool_by_pop)}
    for s0 in range(0, len(users), batch):
        # round to 12 decimals: shows with identical vectors can differ by ~1e-16 in floating point,
        # which would otherwise override the intended pop-rank tie-break
        B = np.round(S[s0:s0 + batch], ROUND)
        ub = users[s0:s0 + batch]
        if mode == "new_only":
            for j, u in enumerate(ub):
                h = [col[t] for t in history.get(int(u), ()) if t in col]
                B[j, h] = -1.0
        order = top_k_stable(B, k + 1)
        top = np.take_along_axis(B, order, axis=1)
        for j, u in enumerate(ub):
            u = int(u)
            lst = [int(pool_by_pop[c]) for c, v in zip(order[j, :k], top[j, :k]) if v > 0] if has[s0 + j] else []
            fill = pop if mode == "full" else lazy_pop(pop, history.get(u, set()) | set(lst), k)
            recs[u] = pad_with_popularity(lst, fill, k)
            padded[u] = k - len(lst)
            tie[u] = bool(has[s0 + j] and top[j, k - 1] > 0 and top[j, k - 1] == top[j, k])
    return recs, padded, tie
