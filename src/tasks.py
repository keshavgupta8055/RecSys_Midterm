"""Tasks for the hybrid: inner (weeks 11-15 -> 16-20), val (= get_split('val')), test (= get_split('test')).

make_task(name) returns a Task with the history interactions, windows, users, truth, pool, pop_recent
ranking, metadata and the labelled airings needed for exposure counts. The target-window schedule is used
ONLY to count which show airs on which channel when; it is never joined to viewing.
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from src.baselines import popularity
from src.labels import build_targets, iso_week_start, n_distinct_shows
from src.protocol import K, THR, build_meta, get_split, labelled_airings, load_interactions, pool_ids, submission_users

WK = {w: pd.Timestamp(iso_week_start(w)) for w in (11, 15, 16, 20, 21, 31, 43)}


@dataclass
class Task:
    name: str
    train: pd.DataFrame                  # history interactions only
    hist_start: pd.Timestamp
    hist_end: pd.Timestamp               # exclusive
    tgt_start: pd.Timestamp
    tgt_end: pd.Timestamp                # exclusive
    users: np.ndarray                    # uint64
    sub_mask: np.ndarray
    pool: np.ndarray                     # sorted int64
    pop: np.ndarray                      # pool ids in pop_recent order
    popnorm: np.ndarray                  # aligned with pop
    meta: pd.DataFrame
    lab: pd.DataFrame                    # labelled airings (duration > 0) of the schedules available to the task
    truth: dict = field(default_factory=dict)
    user_str: list = field(default_factory=list)

    @property
    def hist_weeks(self):
        return (self.hist_end - self.hist_start).days / 7

    @property
    def tgt_weeks(self):
        return (self.tgt_end - self.tgt_start).days / 7


def pop_norm(train, pop, hist_end):
    """pop_recent distinct qualifying users (last 4 history weeks) / max, aligned with the pop ranking."""
    q = train[(train.frac >= THR) & (train.airing_start >= hist_end - pd.Timedelta(weeks=4))]
    c = q.groupby("tv_show_id").user_id.nunique().reindex(pop, fill_value=0).to_numpy(float)
    return c / c.max()


@lru_cache(maxsize=4)
def make_task(name: str) -> Task:
    inter = load_interactions()
    _, sub_u = submission_users()
    if name == "inner":
        hs, he, ts, te = WK[11], WK[16], WK[16], WK[21]
        lab = labelled_airings(["11-20"])
        lab = lab[lab.start_time < te]                      # nothing after the inner target window
        train = inter[(inter.airing_start >= hs) & (inter.airing_start < he)].reset_index(drop=True)
        pool = pool_ids(lab, ts, te)
        tg = build_targets(inter, ts, te, THR)
        nds = n_distinct_shows(tg)
        users = np.sort(nds[nds >= K].index.to_numpy(np.uint64))
        top = tg[(tg["rank"] <= K) & tg.user_id.isin(users)]
        truth = {int(u): g.tv_show_id.tolist() for u, g in top.groupby("user_id", sort=True)}
        assert set(x for v in truth.values() for x in v) <= set(pool.tolist())
        meta = build_meta(lab)
        pop = popularity(train, pool, "pop_recent", he)
        t = Task("inner", train, hs, he, ts, te, users, np.isin(users, sub_u), pool, pop, pop_norm(train, pop, he),
                 meta, lab, truth=truth)
    elif name in ("val", "test"):
        sp = get_split(name)
        lab = labelled_airings(["11-20", "21-30"] + (["31-42"] if name == "test" else []))
        te = WK[31] if name == "val" else WK[43]
        lab = lab[lab.start_time < te]
        pop = popularity(sp.train, sp.pool, "pop_recent", sp.train_end)
        t = Task(name, sp.train, sp.train_start, sp.train_end, sp.train_end, te, sp.users, sp.sub_mask, sp.pool, pop,
                 pop_norm(sp.train, pop, sp.train_end), sp.meta, lab, truth=sp.truth, user_str=sp.user_str)
    else:
        raise ValueError(name)
    # no-leakage assertions
    assert t.train.airing_start.max() < t.tgt_start, name
    assert t.train.airing_start.min() >= t.hist_start, name
    assert t.lab.start_time.max() < t.tgt_end, name
    if name == "inner":
        assert t.tgt_end <= WK[21]   # inner labels never use val-target viewing
    return t


def task_summary(t: Task) -> dict:
    hist = t.train.groupby("user_id").tv_show_id.agg(set)
    pairs = [(u, s) for u, v in t.truth.items() for s in v]
    in_hist = np.mean([s in hist.get(np.uint64(u), set()) for u, s in pairs]) if pairs else np.nan
    return {"task": t.name, "users": len(t.users), "sub_users": int(t.sub_mask.sum()), "pool_size": len(t.pool),
            "history_rows": len(t.train), "history_start": t.hist_start.date(), "history_end_excl": t.hist_end.date(),
            "history_weeks": t.hist_weeks, "target_start": t.tgt_start.date(), "target_end_excl": t.tgt_end.date(),
            "target_weeks": t.tgt_weeks, "truth_pairs": len(pairs), "share_truth_pairs_in_history": in_hist}
