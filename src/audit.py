"""Step 1 data audit. Writes CSVs to outputs/, PNGs to outputs/figures/.

Audit only: no view-log/schedule join and no model.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import io_utils as io

np.random.seed(42)
OUT = io.ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DPI = 150
T0 = time.perf_counter()

EXPECTED_ROWS = {"view_log": 5_821_987, "schedule_11-20": 1_074_235,
                 "schedule_21-30": 1_074_394, "schedule_31-42": 1_295_378, "submission": 1_260}
facts = []  # (fact, value, expected)


def fact(name, value, expected=""):
    facts.append((name, value, expected))
    print(f"{name}: {value}" + (f"  (expected {expected})" if expected != "" else ""))


def save(df, name, index=False):
    df.to_csv(OUT / name, index=index)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)


def describe(s, pct=(0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)):
    d = s.describe(percentiles=list(pct))
    return d.rename_axis("stat").reset_index(name="value")


# ---------------------------------------------------------------- 1. raw schema
print("\n=== 1. Raw schema (pandas inference on raw CSV, first 5 rows) ===")
schema_rows = []
for f in [io.VIEW_FILE, *io.SCHEDULE_FILES.values(), io.SUBMISSION_FILE]:
    raw = pd.read_csv(io.DATA / f, nrows=5, dtype=str)
    inferred = pd.read_csv(io.DATA / f, nrows=1000)
    print(f"\n--- {f}\n{raw.to_string()}")
    for c in raw.columns:
        schema_rows.append({"file": f, "column": c, "inferred_dtype_first1000": str(inferred[c].dtype),
                            "example": raw[c].iloc[0]})
    raw.to_csv(OUT / f"head5_{f}", index=False)
save(pd.DataFrame(schema_rows), "raw_schema.csv")

# ---------------------------------------------------------------- 2. row counts
print("\n=== 2. Row counts ===")
v = io.load_views()
sched = {p: io.load_schedule(p) for p in io.SCHEDULE_FILES}
sub = io.load_submission()
counts = {"view_log": len(v), **{f"schedule_{p}": len(s) for p, s in sched.items()}, "submission": len(sub)}
rc = pd.DataFrame([{"table": k, "rows": n, "expected": EXPECTED_ROWS[k], "diff": n - EXPECTED_ROWS[k]}
                   for k, n in counts.items()])
print(rc.to_string(index=False))
save(rc, "row_counts.csv")
for _, r in rc.iterrows():
    fact(f"rows {r.table}", f"{r.rows:,}", f"{r.expected:,}")

# ---------------------------------------------------------------- 3. view log basics
print("\n=== 3. View log basics ===")
nulls = v.isna().sum()
fact("view log null cells (all cols)", int(nulls.sum()))
fact("view log distinct users", f"{v.user_id.nunique():,}")
fact("view log distinct vsetv_id", f"{v.vsetv_id.nunique():,}")
fact("view log min start_time", v.start_time.min())
fact("view log max start_time", v.start_time.max())
fact("view log max end_time", v.end_time.max())
fact("view log exact duplicate rows", int(v.duplicated().sum()))
iso = v.start_time.dt.isocalendar().week
fact("view log ISO week range (start_time)", f"{int(iso.min())}-{int(iso.max())}")
save(pd.DataFrame({"column": nulls.index, "nulls": nulls.values}), "viewlog_nulls.csv")

# ---------------------------------------------------------------- 4. duration
print("\n=== 4. View duration ===")
d = v.duration
dd = describe(d)
print(dd.to_string(index=False))
save(dd, "viewlog_duration_describe.csv")
fact("view duration median (s)", float(d.median()), "~396")
fact("view duration max (s)", int(d.max()), "~342,145")
fact("view duration min (s)", int(d.min()))
fact("view duration == 0 rows", int((d == 0).sum()))
diff = (v.end_time - v.start_time).dt.total_seconds().astype("int64") - d.astype("int64")
viol = diff != 0
fact("rows where duration != end_time - start_time", int(viol.sum()))
vd = diff[viol].value_counts().head(20).rename_axis("diff_seconds(end-start-duration)").reset_index(name="rows")
save(vd, "viewlog_duration_mismatch_top20.csv")
fact("rows with end_time < start_time", int((v.end_time < v.start_time).sum()))
fact("views longer than 24h (duration>86400)", int((d > 86400).sum()))

# ---------------------------------------------------------------- 5. time ranges
print("\n=== 5. Time ranges ===")
tr = [{"table": "view_log", "min_start": v.start_time.min(), "max_start": v.start_time.max(),
       "iso_week_min": int(iso.min()), "iso_week_max": int(iso.max()),
       "timestamp_raw_format": "YYYY-MM-DD HH:MM:SS (naive, no tz)"}]
for p, s in sched.items():
    w = s.start_time.dt.isocalendar().week
    end = s.start_time + pd.to_timedelta(s.duration, unit="s")
    tr.append({"table": f"schedule_{p}", "min_start": s.start_time.min(), "max_start": s.start_time.max(),
               "max_airing_end": end.max(), "iso_week_min": int(w.min()), "iso_week_max": int(w.max()),
               "timestamp_raw_format": "DD.MM.YYYY HH:MM:SS (naive, no tz)"})
tr = pd.DataFrame(tr)
print(tr.to_string(index=False))
save(tr, "time_ranges.csv")
# views falling inside each schedule's coverage
for p, s in sched.items():
    lo, hi = s.start_time.min(), (s.start_time + pd.to_timedelta(s.duration, unit="s")).max()
    inside = ((v.start_time >= lo) & (v.start_time < hi)).sum()
    fact(f"view rows with start_time inside schedule_{p} span", f"{inside:,}")
# start-time second / minute granularity (helps the later join)
fact("view start_time with seconds != 0 (share)", round(float((v.start_time.dt.second != 0).mean()), 4))
for p, s in sched.items():
    fact(f"schedule_{p} start_time with seconds != 0 (share)", round(float((s.start_time.dt.second != 0).mean()), 4))

# ---------------------------------------------------------------- 6. channel overlap
print("\n=== 6. vsetv_id vs channel_id overlap ===")
vch = set(v.vsetv_id.unique())
rows_per_ch = v.vsetv_id.value_counts()
ov = []
for p, s in sched.items():
    sch = set(s.channel_id.unique())
    inter = vch & sch
    ov.append({"schedule": p, "view_channels": len(vch), "schedule_channels": len(sch),
               "intersection": len(inter), "view_channels_not_in_schedule": len(vch - sch),
               "schedule_channels_not_in_views": len(sch - vch),
               "view_rows_on_matched_channels": int(rows_per_ch[list(inter)].sum()),
               "share_view_rows_matched": round(float(rows_per_ch[list(inter)].sum() / len(v)), 4)})
all_sch = set().union(*[set(s.channel_id.unique()) for s in sched.values()])
ov = pd.DataFrame(ov)
print(ov.to_string(index=False))
save(ov, "channel_overlap.csv")
missing = sorted(vch - all_sch)
save(rows_per_ch[missing].rename_axis("vsetv_id").reset_index(name="view_rows"),
     "viewlog_channels_missing_from_all_schedules.csv")
fact("view channels absent from every schedule", len(missing))

# ---------------------------------------------------------------- 7. schedule content
print("\n=== 7. Schedule content ===")
META = ["tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "year_of_production", "director", "actors"]
sc, miss, cats = [], [], []
for p, s in sched.items():
    lab = s[s.tv_show_id != 0]
    sc.append({"schedule": p, "rows": len(s), "labelled_rows": len(lab),
               "labelled_share": round(len(lab) / len(s), 4),
               "distinct_tv_show_id": lab.tv_show_id.nunique(),
               "distinct_channels": s.channel_id.nunique(),
               "distinct_channels_with_labelled": lab.channel_id.nunique(),
               "null_tv_show_id": int(s.tv_show_id.isna().sum()),
               "negative_tv_show_id": int((s.tv_show_id < 0).sum()),
               "nonpositive_duration": int((s.duration <= 0).sum()),
               "exact_duplicate_rows": int(s.duplicated().sum())})
    for c in META + ["tv_show_category", "tv_show_title"]:
        miss.append({"schedule": p, "column": c, "missing_share_labelled": round(float(lab[c].isna().mean()), 4),
                     "missing_share_unlabelled": round(float(s.loc[s.tv_show_id == 0, c].isna().mean()), 4)})
    cats.append(lab.tv_show_category.value_counts(dropna=False).rename(p))
sc = pd.DataFrame(sc)
miss = pd.DataFrame(miss)
cats = pd.concat(cats, axis=1).fillna(0).astype(int).rename_axis("tv_show_category")
print(sc.to_string(index=False)); print(miss.to_string(index=False)); print(cats.to_string())
save(sc, "schedule_summary.csv"); save(miss, "schedule_missingness_labelled.csv")
save(cats.reset_index(), "schedule_category_counts_labelled.csv")
for _, r in sc.iterrows():
    fact(f"schedule_{r.schedule} labelled share", r.labelled_share, "~0.108" if r.schedule == "11-20" else "")
# Are metadata constant within a tv_show_id?
lab_all = pd.concat([s[s.tv_show_id != 0].assign(schedule=p) for p, s in sched.items()], ignore_index=True)
cons = lab_all.groupby("tv_show_id")[["tv_show_category", "tv_show_genre_1", "tv_show_title"]].nunique()
fact("tv_show_ids with >1 category", int((cons.tv_show_category > 1).sum()))
fact("tv_show_ids with >1 genre_1", int((cons.tv_show_genre_1 > 1).sum()))
fact("tv_show_ids with >1 distinct title (share)", round(float((cons.tv_show_title > 1).mean()), 4))
ids = {p: set(s.loc[s.tv_show_id != 0, "tv_show_id"].unique()) for p, s in sched.items()}
train_ids = ids["11-20"] | ids["21-30"]
fact("labelled tv_show_ids in 31-42 schedule", len(ids["31-42"]))
fact("... of which also aired (labelled) in 11-30 schedules", len(ids["31-42"] & train_ids))
fact("tv_show_ids also in 11-20 & 21-30", len(ids["11-20"] & ids["21-30"]))

# ---------------------------------------------------------------- 8. series level
print("\n=== 8. Airings per tv_show_id ===")
apt = []
for p, s in sched.items():
    a = s.loc[s.tv_show_id != 0].groupby("tv_show_id").size()
    apt.append(describe(a).assign(schedule=p))
    print(p, "airings per tv_show_id:", a.describe().round(1).to_dict())
apt = pd.concat(apt)
save(apt.pivot(index="stat", columns="schedule", values="value").reset_index(), "airings_per_tv_show_id_describe.csv")
a_all = lab_all.groupby("tv_show_id").size()
dist_titles = lab_all.groupby("tv_show_id").tv_show_title.nunique()
save(pd.DataFrame({"airings": a_all, "distinct_titles": dist_titles}).sort_values("airings", ascending=False)
     .join(lab_all.drop_duplicates("tv_show_id").set_index("tv_show_id")[["tv_show_title", "tv_show_category"]])
     .head(30).reset_index(), "top30_tv_show_id_by_airings.csv")
fact("tv_show_ids (all 3 schedules) with exactly 1 airing (share)", round(float((a_all == 1).mean()), 4))
fact("median airings per tv_show_id (all 3 schedules)", float(a_all.median()))
fact("median distinct titles per tv_show_id", float(dist_titles.median()))

# ---------------------------------------------------------------- 9. airing duration
print("\n=== 9. Airing duration (labelled) ===")
ad = describe(lab_all.duration)
print(ad.to_string(index=False))
save(ad, "airing_duration_labelled_describe.csv")
fact("labelled airing duration median (s)", float(lab_all.duration.median()))

# schedule overlaps within channel (relevant for join later)
ovl = []
for p, s in sched.items():
    t = s[["channel_id", "start_time", "duration"]].sort_values(["channel_id", "start_time"])
    end = t.start_time + pd.to_timedelta(t.duration, unit="s")
    nxt = t.groupby("channel_id").start_time.shift(-1)
    gap = (nxt - end).dt.total_seconds()
    ovl.append({"schedule": p, "consecutive_pairs": int(gap.notna().sum()),
                "overlapping_pairs(gap<0)": int((gap < 0).sum()), "exact_contiguous(gap==0)": int((gap == 0).sum()),
                "gaps(gap>0)": int((gap > 0).sum()), "duplicate_channel_start": int(t.duplicated(["channel_id", "start_time"]).sum())})
ovl = pd.DataFrame(ovl)
print(ovl.to_string(index=False))
save(ovl, "schedule_contiguity.csv")

# ---------------------------------------------------------------- 10. submission
print("\n=== 10. submission.csv ===")
fact("submission columns", ",".join(sub.columns), "user_id,tv_show_id")
fact("submission duplicate user_ids", int(sub.user_id.duplicated().sum()))
fact("submission tv_show_id distinct values", ",".join(sub.tv_show_id.unique()[:5]))
upc = v.groupby("user_id").size()
in_log = sub.user_id.isin(upc.index)
fact("submission users present in view log (weeks 11-30)", int(in_log.sum()))
fact("submission users with 0 view rows in weeks 11-30", int((~in_log).sum()))
fact("view-log users not in submission", int((~upc.index.isin(sub.user_id)).sum()))
sub_rows = upc.reindex(sub.user_id).fillna(0)
save(describe(sub_rows).rename(columns={"value": "view_rows_per_submission_user"}), "submission_users_viewrows_describe.csv")

# ---------------------------------------------------------------- 11. per-user rows
print("\n=== 11. View rows per user ===")
pu = describe(upc)
print(pu.to_string(index=False))
save(pu, "viewlog_rows_per_user_describe.csv")
fact("view rows/user min", int(upc.min()))
fact("view rows/user median", float(upc.median()))
fact("view rows/user p90", float(upc.quantile(0.9)))
fact("view rows/user max", int(upc.max()))

# ---------------------------------------------------------------- plots
print("\n=== Plots ===")
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.logspace(np.log10(d[d > 0].min()), np.log10(d.max()), 80)
ax.hist(d[d > 0], bins=bins, color="#3b6ea5")
ax.set_xscale("log")
ax.axvline(d.median(), color="#c0392b", ls="--", label=f"median = {d.median():.0f} s")
ax.set(title="View duration distribution (weeks 11-30, duration > 0)", xlabel="View duration (seconds, log scale)",
       ylabel="Number of view-log rows")
ax.legend(); savefig(fig, "01_view_duration_hist_logx.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(upc, bins=np.logspace(0, np.log10(upc.max()), 50), color="#3b6ea5")
ax.set_xscale("log"); ax.set_yscale("log")
ax.axvline(upc.median(), color="#c0392b", ls="--", label=f"median = {upc.median():.0f} rows")
ax.set(title="View-log rows per user (weeks 11-30, log-binned)", xlabel="Number of view-log rows per user (log)",
       ylabel="Number of users (log)"); ax.legend()
savefig(fig, "02_views_per_user_loglog.png")

fig, ax = plt.subplots(figsize=(10, 5))
per_day = v.groupby(v.start_time.dt.floor("D")).size()
ax.plot(per_day.index, per_day.values, color="#3b6ea5")
ax.set(title="View-log rows per day (by start_time)", xlabel="Date", ylabel="Number of view-log rows")
fig.autofmt_xdate(); savefig(fig, "03_views_per_day.png")
save(per_day.rename_axis("date").reset_index(name="view_rows"), "viewlog_rows_per_day.csv")
low = per_day.nsmallest(5)
fact("5 lowest days (view rows)", "; ".join(f"{i.date()}={n}" for i, n in low.items()))
fact("days with view rows", len(per_day))
users_per_week = v.groupby(iso).user_id.nunique()
save(users_per_week.rename_axis("iso_week").reset_index(name="active_users"), "viewlog_active_users_per_week.csv")
fact("active users per ISO week (min..max)", f"{users_per_week.min()}..{users_per_week.max()}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(sc.schedule, 1 - sc.labelled_share, label="tv_show_id = 0 (unlabelled)", color="#bbbbbb")
ax.bar(sc.schedule, sc.labelled_share, bottom=1 - sc.labelled_share, label="tv_show_id != 0 (labelled)", color="#3b6ea5")
for i, r in sc.iterrows():
    ax.text(i, 1 - r.labelled_share / 2, f"{r.labelled_share:.1%}", ha="center", va="center", color="white", fontsize=10)
ax.set(title="Labelled vs unlabelled schedule rows per file", xlabel="Schedule file (weeks)", ylabel="Share of rows")
ax.legend(loc="lower right"); savefig(fig, "04_schedule_labelled_share.png")

fig, ax = plt.subplots(figsize=(9, 6))
top = lab_all.tv_show_category.value_counts().head(20)[::-1]
ax.barh(top.index.astype(str), top.values, color="#3b6ea5")
ax.set(title=f"Categories among labelled schedule rows, weeks 11-42 (top-20 requested; only {len(top)} exist)",
       xlabel="Number of labelled airings",
       ylabel="tv_show_category")
savefig(fig, "05_top20_categories_labelled.png")

fig, ax = plt.subplots(figsize=(8, 5))
ld = lab_all.duration[lab_all.duration > 0]
ax.hist(ld, bins=np.logspace(np.log10(ld.min()), np.log10(ld.max()), 80), color="#3b6ea5")
ax.set_xscale("log")
ax.axvline(ld.median(), color="#c0392b", ls="--", label=f"median = {ld.median():.0f} s")
ax.set(title="Airing duration of labelled schedule rows (weeks 11-42)", xlabel="Airing duration (seconds, log scale)",
       ylabel="Number of airings"); ax.legend()
savefig(fig, "06_airing_duration_labelled_hist.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(a_all, bins=np.logspace(0, np.log10(a_all.max()), 60), color="#3b6ea5")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set(title="Airings per tv_show_id (labelled rows, weeks 11-42)", xlabel="Number of airings per tv_show_id (log)",
       ylabel="Number of tv_show_ids (log)")
savefig(fig, "07_airings_per_tv_show_id.png")
save(a_all.value_counts().sort_index().rename_axis("airings").reset_index(name="tv_show_ids"),
     "airings_per_tv_show_id_distribution.csv")

# ---------------------------------------------------------------- key facts
kf = pd.DataFrame(facts, columns=["fact", "value", "expected"])
save(kf, "key_facts_step1.csv")
import psutil
mi = psutil.Process().memory_info()
peak = getattr(mi, "peak_wset", mi.rss)  # peak working set on Windows
print(f"\nTotal audit runtime: {time.perf_counter() - T0:.1f} s; peak process memory: {peak / 1024**2:.0f} MB")
