"""Target construction mimicking the competition label recipe.

A (user, airing) pair qualifies when start <= airing_start < end and frac >= thr
(frac = watched seconds of merged views / airing duration, see src/join.py).
Per (user, tv_show_id): n_qualifying_airings = number of qualifying airings,
watched_seconds_total = sum of watched_seconds over those qualifying airings.

Ranking (full list, not truncated; callers take rank <= 5):
    n_qualifying_airings desc, then watched_seconds_total desc, then tv_show_id asc.
The competition only specifies "most frequent"; how its labels break ties is
unobservable, so the two secondary keys are our own explicit convention.
"""
from datetime import datetime

import pandas as pd

COLS = ["user_id", "rank", "tv_show_id", "n_qualifying_airings", "watched_seconds_total"]


def iso_week_start(week: int, year: int = 2020) -> datetime:
    """Monday 00:00 of an ISO week, e.g. 21 -> 2020-05-18, 31 -> 2020-07-27."""
    return datetime.fromisocalendar(year, week, 1)


def build_targets(interactions: pd.DataFrame, start, end, thr: float = 0.8) -> pd.DataFrame:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    m = (interactions.airing_start >= start) & (interactions.airing_start < end) & (interactions.frac >= thr)
    t = (interactions.loc[m]
         .groupby(["user_id", "tv_show_id"], sort=False)
         .agg(n_qualifying_airings=("frac", "size"), watched_seconds_total=("watched_seconds", "sum"))
         .reset_index()
         .sort_values(["user_id", "n_qualifying_airings", "watched_seconds_total", "tv_show_id"],
                      ascending=[True, False, False, True], kind="stable", ignore_index=True))
    t["rank"] = t.groupby("user_id").cumcount() + 1
    return t[COLS]


def n_distinct_shows(targets: pd.DataFrame) -> pd.Series:
    """Number of distinct qualifying tv_show_ids per user (users without any are absent)."""
    return targets.groupby("user_id").size().rename("n_distinct_shows")
