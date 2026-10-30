"""Shared evaluation protocol: every method imports get_split() and pad_with_popularity().

Split "val":  train = interactions with airing_start in ISO weeks 11-20 [2020-03-09, 2020-05-18);
              truth = rank <= 5 of build_targets(inter, W21, W31, thr=0.8);
              users = users with >= 5 distinct qualifying shows in the target window.
Split "test": train = all interactions (weeks 11-30); users = the 1,260 template users (template order).
Pool: labelled tv_show_ids (tv_show_id != 0, duration > 0) with >= 1 airing whose start lies inside the
      target window (val [W21, W31), test [W31, W43)), taken from all schedule files available to the
      split: the files hand over per channel mid-morning, so e.g. 11-20 holds 05-18 airings (week 21).
The 31-42 schedule is used only for the test pool and show metadata, never joined to viewing.
User ids are kept as uint64 (and the template's original strings for the test split).
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from src import io_utils as io
from src.join import INTERACTIONS_PATH
from src.labels import build_targets, iso_week_start, n_distinct_shows

THR = 0.8
K = 5
W = {w: pd.Timestamp(iso_week_start(w)) for w in (11, 17, 21, 26, 27, 31, 43)}
META_COLS = {"tv_show_title": "title", "tv_show_category": "category", "tv_show_genre_1": "genre_1",
             "tv_show_genre_2": "genre_2", "tv_show_genre_3": "genre_3", "year_of_production": "year",
             "director": "director", "actors": "actors"}


@dataclass
class Split:
    kind: str
    train: pd.DataFrame
    pool: np.ndarray                 # sorted int64 tv_show_ids
    users: np.ndarray                # uint64, evaluation / submission order
    sub_mask: np.ndarray             # bool, aligned with users
    meta: pd.DataFrame               # index tv_show_id
    train_start: pd.Timestamp
    train_end: pd.Timestamp          # exclusive
    truth: dict = field(default_factory=dict)       # int(user) -> list of 5 ids (val only)
    user_str: list = field(default_factory=list)    # original template strings (test only)
    info: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def load_interactions() -> pd.DataFrame:
    return pd.read_parquet(INTERACTIONS_PATH)


def labelled_airings(periods) -> pd.DataFrame:
    parts = []
    for p in periods:
        s = io.load_schedule(p)
        s = s[(s.tv_show_id != 0) & (s.duration > 0)]
        parts.append(s[["channel_id", "start_time", "duration", "tv_show_id", *META_COLS]])
    return pd.concat(parts, ignore_index=True)


def build_meta(lab: pd.DataFrame) -> pd.DataFrame:
    """Per tv_show_id: most frequent non-null value of each metadata column (ties: smallest value)."""
    meta = pd.DataFrame(index=pd.Index(np.sort(lab.tv_show_id.unique()), name="tv_show_id"))
    for c, name in META_COLS.items():
        m = (lab.groupby(["tv_show_id", c]).size().reset_index(name="n")
             .sort_values(["tv_show_id", "n", c], ascending=[True, False, True])
             .drop_duplicates("tv_show_id").set_index("tv_show_id")[c])
        meta[name] = m.reindex(meta.index)
    meta["n_airings_schedules"] = lab.groupby("tv_show_id").size().reindex(meta.index)
    return meta


def submission_users():
    """Template user ids as original strings (order kept) and as uint64 (round-trip asserted)."""
    import csv
    with open(io.DATA / io.SUBMISSION_FILE, newline="") as f:
        strs = [r["user_id"] for r in csv.DictReader(f)]
    u = np.array([int(s) for s in strs], dtype=np.uint64)  # explicit dtype: no float coercion
    assert [str(int(x)) for x in u] == strs
    return strs, u


def pad_with_popularity(lst, pop_ranking, k=K):
    """Deduplicate lst (keep order), truncate to k, then fill from pop_ranking skipping duplicates."""
    out, seen = [], set()
    for x in list(lst) + list(pop_ranking):
        x = int(x)
        if x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) == k:
            break
    return out


def pool_ids(lab: pd.DataFrame, start, end) -> np.ndarray:
    m = (lab.start_time >= start) & (lab.start_time < end)
    return np.sort(lab.loc[m, "tv_show_id"].unique()).astype(np.int64)


@lru_cache(maxsize=2)
def get_split(kind: str) -> Split:
    inter = load_interactions()
    sub_strs, sub_u = submission_users()
    if kind == "val":
        lab = labelled_airings(["11-20", "21-30"])
        train = inter[(inter.airing_start >= W[11]) & (inter.airing_start < W[21])].reset_index(drop=True)
        pool = pool_ids(lab, W[21], W[31])
        lit = labelled_airings(["21-30"])
        info = {"pool_size": len(pool),
                "pool_size_literal_schedule_21_30_only_from_W21": int(lit[lit.start_time >= W[21]].tv_show_id.nunique())}
        tg = build_targets(inter, W[21], W[31], THR)
        nds = n_distinct_shows(tg)
        users = np.sort(nds[nds >= K].index.values.astype(np.uint64))
        top = tg[(tg["rank"] <= K) & tg.user_id.isin(users)]
        truth = {int(u): g.tv_show_id.tolist() for u, g in top.groupby("user_id", sort=True)}
        assert all(len(v) == K for v in truth.values()) and len(truth) == len(users)
        missing = {t for v in truth.values() for t in v} - set(pool.tolist())
        assert not missing, f"truth ids outside pool: {missing}"
        sp = Split("val", train, pool, users, np.isin(users, sub_u), build_meta(lab), W[11], W[21],
                   truth=truth, info=info)
    elif kind == "test":
        lab = labelled_airings(["11-20", "21-30", "31-42"])
        train = inter[(inter.airing_start >= W[11]) & (inter.airing_start < W[31])].reset_index(drop=True)
        pool = pool_ids(lab, W[31], W[43])
        info = {"pool_size": len(pool),
                "pool_size_literal_schedule_31_42_only": int(labelled_airings(["31-42"]).tv_show_id.nunique())}
        sp = Split("test", train, pool, sub_u, np.ones(len(sub_u), bool), build_meta(lab), W[11], W[31],
                   user_str=sub_strs, info=info)
    else:
        raise ValueError(kind)
    sp.info.update({"train_rows": len(sp.train), "users": len(sp.users), "sub_users": int(sp.sub_mask.sum())})
    return sp


if __name__ == "__main__":
    for k in ["val", "test"]:
        s = get_split(k)
        print(k, s.info, "train", s.train_start.date(), "->", s.train_end.date(), "(exclusive)")
