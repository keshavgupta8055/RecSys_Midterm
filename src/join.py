"""Step 2: join merged views to schedule airings (weeks 11-30 only).

Rules (see outputs/LOG.md, Step 2):
- exact duplicate view rows are dropped;
- view interval = [start_time, start_time + duration) in integer seconds of naive
  local wall-clock time (NOT end_time: 117 rows differ by +3600 s at the 2020-03-29 DST jump);
- overlapping or touching views of one (user_id, vsetv_id) are merged into blocks
  (a new block starts when v_start > running max of previous v_end);
- airings = schedules 11-20 + 21-30, duration <= 0 dropped, duplicated
  (channel_id, start_time) dropped keeping the first; airing interval = [start, start + duration);
- overlap = max(0, min(v_end, a_end) - max(v_start, a_start)), summed per (user, airing);
  frac = watched_seconds / airing duration.
The 31-42 schedule is never loaded here.
"""
import time

import numpy as np
import pandas as pd

from src import io_utils as io

TEST_START = pd.Timestamp("2020-07-27")  # ISO week 31 start; nothing from here on is kept
INTERACTIONS_PATH = io.CACHE / "interactions.parquet"


def to_sec(ts) -> np.ndarray:
    return np.asarray(ts, dtype="datetime64[s]").astype(np.int64)


def load_views_dedup():
    v = io.load_views()
    n0 = len(v)
    v = v.drop_duplicates(ignore_index=True)
    v_start = to_sec(v.start_time)
    out = pd.DataFrame({"user_id": v.user_id.values, "channel_id": v.vsetv_id.values,
                        "v_start": v_start, "v_end": v_start + v.duration.values.astype(np.int64)})
    return out, {"view_rows_raw": n0, "view_exact_duplicates_dropped": n0 - len(out), "view_rows_dedup": len(out)}


def merge_views(v: pd.DataFrame):
    """Merge overlapping/touching views within (user_id, channel_id). Returns blocks with n_raw."""
    v = v.sort_values(["user_id", "channel_id", "v_start"], kind="stable", ignore_index=True)
    u, ch, s, e = (v[c].values for c in ["user_id", "channel_id", "v_start", "v_end"])
    first = np.ones(len(v), bool)
    first[1:] = (u[1:] != u[:-1]) | (ch[1:] != ch[:-1])
    runmax = v.groupby(np.cumsum(first), sort=False).v_end.cummax().values
    prev = np.empty(len(v), np.int64)
    prev[0] = np.iinfo(np.int64).min
    prev[1:] = runmax[:-1]
    new = first | (s > prev)
    starts = np.flatnonzero(new)
    m = pd.DataFrame({"user_id": u[starts], "channel_id": ch[starts], "v_start": s[starts],
                      "v_end": np.maximum.reduceat(e, starts),
                      "n_raw": np.diff(np.append(starts, len(v)))})
    stats = {"merged_rows": len(m), "watched_seconds_before_merge": int((e - s).sum()),
             "watched_seconds_after_merge": int((m.v_end - m.v_start).sum()),
             "blocks_from_2plus_raw_views": int((m.n_raw > 1).sum())}
    return m, stats


def load_airings():
    a = pd.concat([io.load_schedule("11-20"), io.load_schedule("21-30")], ignore_index=True)
    n0 = len(a)
    a = a[a.duration > 0]
    n_nonpos = n0 - len(a)
    n1 = len(a)
    a = a.drop_duplicates(["channel_id", "start_time"], keep="first")
    st = to_sec(a.start_time)
    out = pd.DataFrame({"channel_id": a.channel_id.values, "a_start": st,
                        "a_end": st + a.duration.values.astype(np.int64),
                        "duration": a.duration.values, "tv_show_id": a.tv_show_id.values})
    out = out.sort_values(["channel_id", "a_start"], kind="stable", ignore_index=True)
    out["labelled"] = out.tv_show_id != 0
    return out, {"airings_raw_11_30": n0, "airings_dropped_nonpos_duration": n_nonpos,
                 "airings_dropped_dup_channel_start": n1 - len(out), "airings_kept": len(out),
                 "airings_labelled": int(out.labelled.sum())}


def channel_overlaps(vs, ve, st, en):
    """All (view, airing) pairs with positive overlap on one channel.

    Airings sorted by start; ends may be non-monotone (slightly overlapping airings), so the
    lower bound uses the running max of airing ends: every airing before the first index with
    cummax(end) > v_start ends at or before v_start. Candidates in [lo, hi) are then filtered
    on overlap > 0.
    """
    cm = np.maximum.accumulate(en)
    lo = np.searchsorted(cm, vs, side="right")
    hi = np.searchsorted(st, ve, side="left")
    cnt = np.maximum(hi - lo, 0)
    tot = int(cnt.sum())
    vi = np.repeat(np.arange(len(vs)), cnt)
    ai = np.repeat(lo, cnt) + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    ov = np.minimum(ve[vi], en[ai]) - np.maximum(vs[vi], st[ai])
    k = ov > 0
    return vi[k], ai[k], ov[k]


def union_intervals(st, en):
    cm = np.maximum.accumulate(en)
    new = np.ones(len(st), bool)
    new[1:] = st[1:] > cm[:-1]
    s = np.flatnonzero(new)
    return st[s], np.maximum.reduceat(en, s)


def build(merge=True, check=True):
    """Return (interactions for labelled airings incl. those >= TEST_START, per-user accounting, stats)."""
    t0 = time.perf_counter()
    v, stats = load_views_dedup()
    if merge:
        v, ms = merge_views(v)
        stats.update(ms)
    else:
        v = v.sort_values(["user_id", "channel_id", "v_start"], kind="stable", ignore_index=True)
    a, ast = load_airings()
    stats.update(ast)

    ucode, users = pd.factorize(v.user_id, sort=True)
    nu = len(users)
    vdur = (v.v_end - v.v_start).values
    acct = {k: np.zeros(nu) for k in ["total", "absent_channel", "labelled", "unlabelled", "covered_by_union",
                                      "on_airings_from_test_start"]}
    acct["total"] = np.bincount(ucode, vdur, minlength=nu)
    test_sec = to_sec(np.array([TEST_START.to_datetime64()]))[0]

    a_ch = a.channel_id.values
    ch_keys, ch_first = np.unique(a_ch, return_index=True)
    ch_last = np.append(ch_first[1:], len(a))
    a_bounds = {c: (f, l) for c, f, l in zip(ch_keys, ch_first, ch_last)}
    v_ch = v.channel_id.values
    order = np.argsort(v_ch, kind="stable")
    vch_sorted = v_ch[order]
    vk, vf = np.unique(vch_sorted, return_index=True)
    vl = np.append(vf[1:], len(order))

    parts = []
    for c, f, l in zip(vk, vf, vl):
        idx = order[f:l]
        if c not in a_bounds:
            acct["absent_channel"] += np.bincount(ucode[idx], vdur[idx], minlength=nu)
            continue
        s0, s1 = a_bounds[c]
        st, en = a.a_start.values[s0:s1], a.a_end.values[s0:s1]
        vs, ve = v.v_start.values[idx], v.v_end.values[idx]
        vi, ai, ov = channel_overlaps(vs, ve, st, en)
        uc = ucode[idx][vi]
        lab = a.labelled.values[s0:s1][ai]
        acct["labelled"] += np.bincount(uc[lab], ov[lab], minlength=nu)
        acct["unlabelled"] += np.bincount(uc[~lab], ov[~lab], minlength=nu)
        late = st[ai] >= test_sec
        acct["on_airings_from_test_start"] += np.bincount(uc[late], ov[late], minlength=nu)
        us, ue = union_intervals(st, en)
        uvi, _, uov = channel_overlaps(vs, ve, us, ue)
        acct["covered_by_union"] += np.bincount(ucode[idx][uvi], uov, minlength=nu)
        # aggregate labelled overlaps to (user, airing)
        key = uc[lab].astype(np.int64) * (s1 - s0) + ai[lab]
        uk, inv = np.unique(key, return_inverse=True)
        parts.append(pd.DataFrame({"ucode": (uk // (s1 - s0)).astype(np.int32),
                                   "aidx": (uk % (s1 - s0)) + s0,
                                   "watched_seconds": np.bincount(inv, ov[lab]).astype(np.int64)}))

    x = pd.concat(parts, ignore_index=True)
    ai = x.aidx.values
    inter = pd.DataFrame({
        "user_id": users[x.ucode.values].astype(np.uint64),
        "tv_show_id": a.tv_show_id.values[ai],
        "channel_id": a.channel_id.values[ai],
        "airing_start": a.a_start.values[ai].astype("datetime64[s]"),
        "airing_duration": a.duration.values[ai],
        "watched_seconds": x.watched_seconds.values,
    })
    inter["frac"] = inter.watched_seconds / inter.airing_duration
    inter["iso_week"] = inter.airing_start.dt.isocalendar().week.astype(np.int16).values
    inter = inter.sort_values(["user_id", "airing_start", "channel_id"], kind="stable", ignore_index=True)
    stats["frac_max"] = float(inter.frac.max())
    stats["frac_gt_1_pairs"] = int((inter.frac > 1 + 1e-9).sum())
    if check and merge:
        assert stats["frac_gt_1_pairs"] == 0, "frac > 1 after merging"
    acct = pd.DataFrame(acct)
    acct.insert(0, "user_id", users.astype(np.uint64))
    stats["build_seconds"] = round(time.perf_counter() - t0, 1)
    return inter, acct, stats


def main():
    import json

    import psutil
    inter, acct, stats = build()
    late = inter.airing_start >= TEST_START
    stats["labelled_pairs_all"] = len(inter)
    stats["labelled_pairs_dropped_airing_in_week31"] = int(late.sum())
    inter = inter[~late].reset_index(drop=True)
    stats["labelled_pairs_saved"] = len(inter)
    inter.to_parquet(INTERACTIONS_PATH, index=False)
    acct.to_parquet(io.CACHE / "watchtime_per_user.parquet", index=False)
    mi = psutil.Process().memory_info()
    stats["peak_memory_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 1024**2)
    (io.ROOT / "outputs" / "join_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(inter.dtypes.to_string())
    print(inter.head().to_string())


if __name__ == "__main__":
    main()
