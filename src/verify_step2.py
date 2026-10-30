"""Step 2 verification (C1-C4). Usage: python -m src.verify_step2 {brute,hand,repro,merge}"""
import calendar
import csv
import hashlib
import subprocess
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src import io_utils as io
from src.join import INTERACTIONS_PATH, build

OUT = io.ROOT / "outputs"
SEED = 42


def secs(dt: datetime) -> int:
    return calendar.timegm(dt.timetuple())


# ------------------------------------------------------------------ C1 brute force
def naive_pairs(view_rows, airings_by_ch):
    """Pure Python: dedup, merge per (user, channel), double loop over views x airings of the channel."""
    seen, views = set(), []
    for r in view_rows:
        if r not in seen:
            seen.add(r)
            views.append(r)
    by_key = {}
    for (u, ch, st, dur) in views:
        s = secs(st)
        by_key.setdefault((u, ch), []).append([s, s + dur])
    merged = []
    for (u, ch), ivs in by_key.items():
        ivs.sort()
        cur = list(ivs[0])
        for s, e in ivs[1:]:
            if s <= cur[1]:
                cur[1] = max(cur[1], e)
            else:
                merged.append((u, ch, cur[0], cur[1]))
                cur = [s, e]
        merged.append((u, ch, cur[0], cur[1]))
    acc = {}
    for (u, ch, s, e) in merged:
        for (a_s, a_e, show) in airings_by_ch.get(ch, []):
            ov = min(e, a_e) - max(s, a_s)
            if ov > 0:
                acc[(u, ch, a_s)] = acc.get((u, ch, a_s), 0) + ov
    return acc, merged


def naive_airings():
    a = pd.concat([io.load_schedule("11-20"), io.load_schedule("21-30")], ignore_index=True)
    out, seen = {}, set()
    for ch, st, dur, show in zip(a.channel_id.tolist(), a.start_time.tolist(), a.duration.tolist(),
                                 a.tv_show_id.tolist()):
        if dur <= 0 or (ch, st) in seen:
            continue
        seen.add((ch, st))
        s = secs(st.to_pydatetime())
        out.setdefault(ch, []).append((s, s + dur, show))
    return out


def brute():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    v = io.load_views()
    sub_users = np.sort(io.load_submission().user_id.values)  # keep uint64: np.array(list of big ints) -> float64
    sub = set(int(u) for u in sub_users)
    users = np.sort(v.user_id.unique())
    long_user = int(v.loc[v.duration.idxmax(), "user_id"])
    pick = [long_user] + list(rng.choice(sub_users, 5, replace=False))
    rest = [u for u in rng.choice(users, 60, replace=False) if u not in pick]
    pick = [int(u) for u in pick] + [int(u) for u in rest[:30 - len(pick)]]
    assert len(set(pick)) == 30 and set(pick) <= set(int(u) for u in users), "picked ids not in view log"
    print(f"picked {len(pick)} users; long-view user {long_user} (max duration {v.duration.max()} s); "
          f"submission users among picks: {sum(u in sub for u in pick)}")

    air = naive_airings()
    inter = pd.read_parquet(INTERACTIONS_PATH)
    inter = inter[inter.user_id.isin(pick)]
    pipe = {(int(u), int(c), secs(pd.Timestamp(t).to_pydatetime())): int(w) for u, c, t, w in
            zip(inter.user_id, inter.channel_id, inter.airing_start, inter.watched_seconds)}
    labelled_starts = {(ch, s) for ch, lst in air.items() for (s, e, show) in lst if show != 0}

    rows, n_pairs, n_mis = [], 0, 0
    vv = v[v.user_id.isin(pick)]
    for u in pick:
        uv = vv[vv.user_id == u]
        raw = list(zip([int(x) for x in uv.user_id], uv.vsetv_id.tolist(),
                       [t.to_pydatetime() for t in uv.start_time], uv.duration.tolist()))
        acc, merged = naive_pairs(raw, air)
        naive_lab = {k: w for k, w in acc.items() if (k[1], k[2]) in labelled_starts}
        pipe_u = {k: w for k, w in pipe.items() if k[0] == u}
        keys = set(naive_lab) | set(pipe_u)
        mis = sum(naive_lab.get(k) != pipe_u.get(k) for k in keys)
        max_span = max((sum(1 for (a_s, a_e, _) in air.get(ch, []) if min(e, a_e) - max(s, a_s) > 0)
                        for (_, ch, s, e) in merged), default=0)
        rows.append({"user_id": u, "submission_user": u in sub, "raw_views": len(raw), "merged_views": len(merged),
                     "max_airings_spanned_by_one_view": max_span, "pairs_naive": len(naive_lab),
                     "pairs_pipeline": len(pipe_u), "pairs_compared": len(keys), "mismatches": mis})
        n_pairs += len(keys)
        n_mis += mis
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "verify_bruteforce.csv", index=False)
    print(f"\nTOTAL labelled (user, airing) pairs compared: {n_pairs:,}; mismatches: {n_mis}; "
          f"runtime {time.perf_counter() - t0:.1f} s")
    assert n_mis == 0


# ------------------------------------------------------------------ C2 hand examples
VIEW_FMT, SCHED_FMT = "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S"


def hand():
    rng = np.random.default_rng(SEED)
    with open(io.DATA / "submission.csv", newline="") as f:
        sub = [r["user_id"] for r in csv.DictReader(f)]
    cand = set(rng.choice(sub, 20, replace=False).tolist())
    raw = {u: [] for u in cand}
    with open(io.DATA / "dataset11-30.csv", newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        for r in rd:
            if r[0] in raw:
                raw[r[0]].append(r)
    inter = pd.read_parquet(INTERACTIONS_PATH)

    def parse(r):
        s = datetime.strptime(r[2], VIEW_FMT)
        return s, s + timedelta(seconds=int(r[4]))

    # choose user: one with two raw views on one channel that overlap each other and both overlap a labelled airing
    chosen = None
    for u in sorted(cand):
        rows = sorted(raw[u], key=lambda r: (int(r[1]), r[2]))
        ui = inter[inter.user_id == np.uint64(int(u))]
        for r1, r2 in zip(rows, rows[1:]):
            if r1[1] != r2[1]:
                continue
            (s1, e1), (s2, e2) = parse(r1), parse(r2)
            if s2 >= e1:  # need a genuine overlap so merging changes the result
                continue
            ch = int(r1[1])
            hit = ui[(ui.channel_id == ch) & (ui.airing_start < e1) &
                     (ui.airing_start + pd.to_timedelta(ui.airing_duration, unit="s") > s2)]
            if len(hit):
                chosen = (u, ch, hit.iloc[0])
                break
        if chosen:
            break
    assert chosen, "no merged-overlap example found among 20 sampled submission users"
    u, _, merged_row = chosen
    ui = inter[inter.user_id == np.uint64(int(u))]
    key = pd.MultiIndex.from_arrays([ui.channel_id, ui.airing_start])
    not_c = ~key.isin([(merged_row.channel_id, merged_row.airing_start)])
    examples = [("A: frac >= 0.8", ui[(ui.frac >= 0.8) & not_c].iloc[0]),
                ("B: frac < 0.8", ui[(ui.frac < 0.8) & (ui.frac > 0.2) & not_c].iloc[0]),
                ("C: two overlapping raw views merged", merged_row)]

    lines = [f"Hand-computed examples, submission user {u} (raw lines read with the csv module only)",
             f"view-log header: {header}"]
    for title, ex in examples:
        ch, a_start = int(ex.channel_id), pd.Timestamp(ex.airing_start).to_pydatetime()
        a_line = None
        for fn in [io.SCHEDULE_FILES["11-20"], io.SCHEDULE_FILES["21-30"]]:
            with open(io.DATA / fn, newline="", encoding="utf-8") as f:
                rd = csv.reader(f)
                sh = next(rd)
                for r in rd:
                    if r[0] == str(ch) and r[2] == a_start.strftime(SCHED_FMT):
                        a_line = r
                        break
            if a_line:
                break
        a_s = datetime.strptime(a_line[2], SCHED_FMT)
        a_e = a_s + timedelta(seconds=int(a_line[3]))
        vlines = sorted([r for r in raw[u] if int(r[1]) == ch and parse(r)[0] < a_e and parse(r)[1] > a_s],
                        key=lambda r: r[2])
        lines += ["", "=" * 100, title, f"raw airing line: {dict(zip(sh, a_line[:7]))}",
                  f"  airing interval: [{a_s}, {a_s} + {a_line[3]} s = {a_e})"]
        lines.append(f"raw view lines on channel {ch} overlapping the airing ({len(vlines)}):")
        for r in vlines:
            s, e = parse(r)
            lines.append(f"  {r}  ->  view interval [{s}, {s} + {r[4]} s = {e})")
        # merge by hand
        blocks = []
        for r in vlines:
            s, e = parse(r)
            if blocks and s <= blocks[-1][1]:
                lines.append(f"  merge: start {s} <= current block end {blocks[-1][1]} -> block end = "
                             f"max({blocks[-1][1]}, {e}) = {max(blocks[-1][1], e)}")
                blocks[-1][1] = max(blocks[-1][1], e)
            else:
                blocks.append([s, e])
        # note: neighbours outside the airing could extend a block, but cannot change its overlap with the airing
        total = 0
        for s, e in blocks:
            lo, hi = max(s, a_s), min(e, a_e)
            ov = max(0, int((hi - lo).total_seconds()))
            lines.append(f"  block [{s}, {e}): overlap = min({e}, {a_e}) - max({s}, {a_s}) = {hi} - {lo} = {ov} s")
            total += ov
        frac = total / int(a_line[3])
        unmerged = sum(max(0, int((min(parse(r)[1], a_e) - max(parse(r)[0], a_s)).total_seconds())) for r in vlines)
        lines.append(f"  watched_seconds = {total} s; frac = {total} / {a_line[3]} = {frac:.6f}"
                     f"   (summing raw views without merging would give {unmerged} s)")
        lines.append(f"  pipeline: watched_seconds = {int(ex.watched_seconds)}, frac = {ex.frac:.6f} -> "
                     f"{'MATCH' if total == int(ex.watched_seconds) else 'MISMATCH'}")
    txt = "\n".join(lines)
    (OUT / "verify_hand_examples.txt").write_text(txt, encoding="utf-8")
    print(txt)


# ------------------------------------------------------------------ C3 reproducibility
def repro():
    (io.CACHE / "repro").mkdir(exist_ok=True)
    rows = []
    for run in [1, 2]:
        for p in io.CACHE.glob("*.parquet"):
            p.unlink()
        t0 = time.perf_counter()
        subprocess.run([sys.executable, "-m", "src.join"], check=True, cwd=io.ROOT, capture_output=True)
        dt = time.perf_counter() - t0
        df = pd.read_parquet(INTERACTIONS_PATH)
        h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
        df.to_parquet(io.CACHE / "repro" / f"interactions_run{run}.parquet", index=False)
        rows.append({"run": run, "wall_seconds_incl_csv_parse": round(dt, 1), "rows": len(df), "sha256": h})
        print(rows[-1])
    pd.testing.assert_frame_equal(pd.read_parquet(io.CACHE / "repro" / "interactions_run1.parquet"),
                                  pd.read_parquet(io.CACHE / "repro" / "interactions_run2.parquet"))
    for run in [1, 2]:
        (io.CACHE / "repro" / f"interactions_run{run}.parquet").unlink()
    print("assert_frame_equal: identical")
    pd.DataFrame(rows).assign(identical=rows[0]["sha256"] == rows[1]["sha256"]).to_csv(
        OUT / "verify_reproducibility.csv", index=False)


# ------------------------------------------------------------------ C4 merge sensitivity
def merge():
    res = []
    for m in [True, False]:
        inter, _, st = build(merge=m, check=m)
        inter = inter[inter.airing_start < pd.Timestamp("2020-07-27")]
        res.append({"merge_views": m, "labelled_pairs": len(inter), "pairs_frac_ge_0.8": int((inter.frac >= 0.8).sum()),
                    "pairs_frac_gt_1": int((inter.frac > 1 + 1e-9).sum()), "frac_max": float(inter.frac.max()),
                    "watched_seconds_on_labelled": int(inter.watched_seconds.sum()), "build_seconds": st["build_seconds"]})
        print(res[-1])
    df = pd.DataFrame(res)
    df.loc[2] = ["difference (no_merge - merge)"] + [df.iloc[1, i] - df.iloc[0, i] for i in range(1, df.shape[1])]
    df.to_csv(OUT / "merge_sensitivity.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    {"brute": brute, "hand": hand, "repro": repro, "merge": merge}[sys.argv[1]]()
