"""Non-learned baselines: global popularity and personal frequency (PF).

PF is a BASELINE, not one of the two required methods.
All rankings are restricted to the split's candidate pool.
"""
import numpy as np
import pandas as pd

from src.protocol import K, THR, pad_with_popularity

POP_VARIANTS = ["pop_users", "pop_pairs", "pop_hours", "pop_recent"]
PF_VARIANTS = ["PF_count_0.8", "PF_count_0.5", "PF_softfrac", "PF_seconds",
               "PF_decay_hl2", "PF_decay_hl4", "PF_decay_hl8"]


def popularity(train, pool, variant, train_end, thr=THR, recent_weeks=4) -> np.ndarray:
    """Pool ids by score desc, tv_show_id asc; pool shows without training signal last (id asc)."""
    t = train[train.tv_show_id.isin(pool)]
    q = t[t.frac >= thr]
    if variant == "pop_users":
        s = q.groupby("tv_show_id").user_id.nunique()
    elif variant == "pop_pairs":
        s = q.groupby("tv_show_id").size()
    elif variant == "pop_hours":
        s = t.groupby("tv_show_id").watched_seconds.sum()
    elif variant == "pop_recent":
        r = q[q.airing_start >= train_end - pd.Timedelta(weeks=recent_weeks)]
        s = r.groupby("tv_show_id").user_id.nunique()
    else:
        raise ValueError(variant)
    s = s.reindex(pool, fill_value=0).astype(float)
    df = pd.DataFrame({"id": pool, "s": s.values}).sort_values(["s", "id"], ascending=[False, True], kind="stable")
    return df.id.to_numpy(np.int64)


def pf_scores(train, pool, variant, train_end) -> pd.DataFrame:
    """Per (user, show in pool) score > 0, plus total watched seconds, ranked within user."""
    t = train[train.tv_show_id.isin(pool)]
    if variant == "PF_count_0.8":
        w = (t.frac >= 0.8).astype(float)
    elif variant == "PF_count_0.5":
        w = (t.frac >= 0.5).astype(float)
    elif variant == "PF_softfrac":
        w = t.frac.astype(float)
    elif variant == "PF_seconds":
        w = t.watched_seconds.astype(float)
    elif variant.startswith("PF_decay_hl"):
        hl = float(variant.removeprefix("PF_decay_hl"))
        age_weeks = (train_end - t.airing_start).dt.total_seconds() / (7 * 86400)
        w = (t.frac >= THR).astype(float) * 0.5 ** (age_weeks / hl)
    else:
        raise ValueError(variant)
    keys = [t.user_id, t.tv_show_id]
    df = pd.DataFrame({"score": w.groupby(keys).sum(), "secs": t.watched_seconds.groupby(keys).sum()}).reset_index()
    df = df[df.score > 0]
    return df.sort_values(["user_id", "score", "secs", "tv_show_id"], ascending=[True, False, False, True],
                          kind="stable", ignore_index=True)


def pf_recommend(train, pool, users, variant, pop_ranking, train_end, k=K):
    """Returns (recs: int(user) -> k ids, n_padded: int(user) -> number of popularity fill-ins)."""
    top = pf_scores(train, pool, variant, train_end).groupby("user_id").head(k)
    personal = {int(u): g.tv_show_id.tolist() for u, g in top.groupby("user_id", sort=False)}
    recs, padded = {}, {}
    for u in users:
        u = int(u)
        p = personal.get(u, [])
        recs[u] = pad_with_popularity(p, pop_ranking, k)
        padded[u] = k - len(p)
    return recs, padded


def pop_recommend(users, pop_ranking, k=K):
    top = [int(x) for x in pop_ranking[:k]]
    return {int(u): list(top) for u in users}, {int(u): 0 for u in users}
