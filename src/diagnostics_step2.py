"""Step 2 diagnostics (A7 watch-time accounting, D1-D7). No model is built here."""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from src import io_utils as io
from src.join import INTERACTIONS_PATH, load_views_dedup
from src.labels import build_targets, iso_week_start, n_distinct_shows

np.random.seed(42)
T0 = time.perf_counter()
OUT = io.ROOT / "outputs"
FIG = OUT / "figures"
DPI = 150
BLUE, ORANGE, GREY, RED = "#3b6ea5", "#e08a2c", "#bbbbbb", "#c0392b"
CAT_EN = {"Детям": "Kids", "Сериалы": "Series", "Фильмы": "Films", "Развлечения": "Entertainment",
          "Инфо": "Info", "Познавательное": "Educational", "Другое": "Other", "Спорт": "Sport"}
summary = {}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)


inter = pd.read_parquet(INTERACTIONS_PATH)
sub_users = io.load_submission().user_id.values
wt = pd.read_parquet(io.CACHE / "watchtime_per_user.parquet")
views, _ = load_views_dedup()
all_users = np.sort(views.user_id.unique())
W11, W21, W25, W31 = (pd.Timestamp(iso_week_start(w)) for w in (11, 21, 25, 31))
GROUPS = {"all_users": all_users, "submission_users": sub_users}

# ------------------------------------------------------------------ A7 watch-time accounting
rows = []
for g, us in GROUPS.items():
    w = wt[wt.user_id.isin(us)].sum(numeric_only=True)
    double = w.labelled + w.unlabelled - w.covered_by_union
    gap = w.total - w.absent_channel - w.covered_by_union
    parts = [("(i) labelled airings (tv_show_id != 0)", w.labelled),
             ("(ii) unlabelled airings (tv_show_id = 0)", w.unlabelled),
             ("(iii) channels absent from schedules", w.absent_channel),
             ("(iv) gaps, not covered by any airing", gap),
             ("correction: seconds counted twice in (i)+(ii) where airings overlap", -double)]
    for name, s in parts:
        rows.append({"group": g, "bucket": name, "seconds": int(s), "share": s / w.total})
    rows.append({"group": g, "bucket": "TOTAL merged watched seconds", "seconds": int(w.total), "share": 1.0})
acct = pd.DataFrame(rows)
assert np.allclose(acct[acct.bucket != "TOTAL merged watched seconds"].groupby("group").seconds.sum(),
                   acct[acct.bucket == "TOTAL merged watched seconds"].set_index("group").seconds)
acct.to_csv(OUT / "watchtime_accounting.csv", index=False)
print(acct.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 3.8))
names = {"(i) labelled airings (tv_show_id != 0)": ("labelled airings", BLUE),
         "(ii) unlabelled airings (tv_show_id = 0)": ("unlabelled airings (tv_show_id = 0)", GREY),
         "(iii) channels absent from schedules": ("channels absent from schedules", ORANGE),
         "(iv) gaps, not covered by any airing": ("schedule gaps", RED)}
for i, g in enumerate(GROUPS):
    left = 0
    for b, (lab, col) in names.items():
        s = acct[(acct.group == g) & (acct.bucket == b)].share.iloc[0]
        ax.barh(i, s, left=left, color=col, label=lab if i == 0 else None)
        if s > 0.03:
            ax.text(left + s / 2, i, f"{s:.1%}", ha="center", va="center", color="white", fontsize=10)
        left += s
ax.set_yticks([0, 1], ["all users (4,914)", "submission users (1,260)"])
ax.set_xlim(0, 1)
ax.set(title="Where merged watch time falls (weeks 11-30)", xlabel="Share of merged watched seconds")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=9, frameon=False)
savefig(fig, "13_watchtime_accounting.png")

# ------------------------------------------------------------------ D1 threshold sensitivity
train = inter[(inter.airing_start >= W11) & (inter.airing_start < W31)]
summary["labelled_pairs_weeks_11_30"] = len(train)
summary["labelled_pairs_week10"] = int((inter.airing_start < W11).sum())
rows = []
for g, us in GROUPS.items():
    d = train[train.user_id.isin(us)]
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        q = d[d.frac >= thr]
        nds = q.groupby("user_id").tv_show_id.nunique()
        rows.append({"group": g, "thr": thr, "qualifying_pairs": len(q), "distinct_users": q.user_id.nunique(),
                     "distinct_tv_show_ids": q.tv_show_id.nunique(), "users_with_ge5_shows": int((nds >= 5).sum())})
ts = pd.DataFrame(rows)
ts.to_csv(OUT / "threshold_sensitivity.csv", index=False)
print(ts.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(train.frac, bins=100, range=(0, 1), color=BLUE)
ax.axvline(0.8, color=RED, ls="--", label="threshold 0.8")
ax.set_yscale("log")
ax.set(title="Screen-time fraction of labelled (user, airing) pairs, weeks 11-30",
       xlabel="frac = watched seconds / airing duration", ylabel="Number of pairs (log)")
ax.legend(); savefig(fig, "08_frac_hist.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for g, col in zip(GROUPS, [BLUE, ORANGE]):
    t = ts[ts.group == g]
    axes[0].plot(t.thr, t.qualifying_pairs, "o-", color=col, label=g.replace("_", " "))
    axes[1].plot(t.thr, t.users_with_ge5_shows, "o-", color=col, label=g.replace("_", " "))
axes[0].set(title="Qualifying (user, airing) pairs vs threshold", xlabel="Screen-time threshold",
            ylabel="Qualifying pairs (weeks 11-30)")
axes[1].set(title="Users with >= 5 distinct qualifying shows vs threshold", xlabel="Screen-time threshold",
            ylabel="Users")
for ax in axes:
    ax.axvline(0.8, color=GREY, ls=":"); ax.legend()
savefig(fig, "09_threshold_sensitivity.png")

# ------------------------------------------------------------------ D2 per ISO week
q8 = inter[inter.frac >= 0.8]
vw = pd.Series(views.v_start.values.astype("datetime64[s]")).dt.isocalendar().week.values
active = pd.Series(views.user_id.values).groupby(vw).nunique()
wk = pd.DataFrame({"iso_week": range(10, 31)})
wk["qualifying_pairs"] = wk.iso_week.map(q8.groupby("iso_week").size()).fillna(0).astype(int)
wk["active_users_viewlog"] = wk.iso_week.map(active).fillna(0).astype(int)
wk["qualifying_users"] = wk.iso_week.map(q8.groupby("iso_week").user_id.nunique()).fillna(0).astype(int)
wk.to_csv(OUT / "weekly_qualifying.csv", index=False)
print(wk.to_string(index=False))
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(wk.iso_week, wk.active_users_viewlog, "o-", color=GREY, label="active users (any view)")
ax.plot(wk.iso_week, wk.qualifying_users, "o-", color=BLUE, label="users with >= 1 qualifying airing")
ax.set(title="Weekly activity (ISO weeks 10-30, threshold 0.8)", xlabel="ISO week of airing / view start",
       ylabel="Users")
ax2 = ax.twinx()
ax2.bar(wk.iso_week, wk.qualifying_pairs, color=ORANGE, alpha=0.3, label="qualifying pairs (right axis)")
ax2.set_ylabel("Qualifying (user, airing) pairs")
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)  # lines above bars
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
ax.set_xticks(wk.iso_week)
savefig(fig, "10_weekly_qualifying.png")

# ------------------------------------------------------------------ D3 distinct qualifying shows per user
t_all = build_targets(inter, W11, W31, 0.8)
nds = n_distinct_shows(t_all)
rows, hist = [], {}
for g, us in GROUPS.items():
    n = nds.reindex(us).fillna(0).astype(int)
    hist[g] = n
    rows.append({"group": g, "users": len(n), "users_0_shows": int((n == 0).sum()), "users_ge1": int((n >= 1).sum()),
                 "users_ge5": int((n >= 5).sum()), "median_shows": float(n.median()), "p90_shows": float(n.quantile(.9)),
                 "max_shows": int(n.max())})
nd = pd.DataFrame(rows)
nd.to_csv(OUT / "n_distinct_shows_summary.csv", index=False)
pd.DataFrame({g: h.value_counts().sort_index() for g, h in hist.items()}).fillna(0).astype(int) \
    .rename_axis("n_distinct_shows").reset_index().to_csv(OUT / "n_distinct_shows_distribution.csv", index=False)
print(nd.to_string(index=False))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mx = int(hist["all_users"].max())
for ax, bins, ttl in [(axes[0], np.arange(0, 32) - 0.5, "zoom 0-30, one bar per value"),
                      (axes[1], np.arange(0, mx + 10, 10), "full range, bins of 10")]:
    for g, col in zip(GROUPS, [BLUE, ORANGE]):
        ax.hist(hist[g], bins=bins, histtype="step", lw=2, color=col,
                label=f"{g.replace('_', ' ')} (n={len(hist[g]):,})")
    ax.axvline(4.5 if ax is axes[0] else 5, color=RED, ls="--", label="5 shows (full top-5 possible)")
    ax.set(title=f"Distinct qualifying shows per user ({ttl})",
           xlabel="Distinct tv_show_ids with >= 1 qualifying airing (weeks 11-30, thr 0.8)", ylabel="Users")
    ax.legend()
savefig(fig, "11_n_distinct_shows.png")

# ------------------------------------------------------------------ D4 candidate windows
WINDOWS = {"W1 weeks 21-30": (W21, W31), "W2 weeks 25-30": (W25, W31), "W3 weeks 11-20": (W11, W21)}
targets = {k: build_targets(inter, s, e, 0.8) for k, (s, e) in WINDOWS.items()}
rows = []
for k, (s, e) in WINDOWS.items():
    n = n_distinct_shows(targets[k])
    for g, us in GROUPS.items():
        m = n.reindex(us).fillna(0)
        rows.append({"window": k, "start": s.date(), "end_exclusive": e.date(), "group": g, "users": len(us),
                     "ge1_shows": int((m >= 1).sum()), "ge3_shows": int((m >= 3).sum()), "ge5_shows": int((m >= 5).sum())})
cw = pd.DataFrame(rows)
cw.to_csv(OUT / "candidate_window_target_counts.csv", index=False)
print(cw.to_string(index=False))

# ------------------------------------------------------------------ D5 repeat consumption W1 vs W3
sched = pd.concat([io.load_schedule("11-20"), io.load_schedule("21-30")], ignore_index=True)
lab_sched = sched[(sched.tv_show_id != 0) & (sched.duration > 0)]
ids_1120 = set(io.load_schedule("11-20").query("tv_show_id != 0").tv_show_id.unique())
w1 = targets["W1 weeks 21-30"]
n1 = n_distinct_shows(w1)
w3 = inter[(inter.airing_start >= W11) & (inter.airing_start < W21)]
w3_q = set(zip(w3.user_id[w3.frac >= 0.8], w3.tv_show_id[w3.frac >= 0.8]))
w3_any = set(zip(w3.user_id, w3.tv_show_id))
rows = []
for g, us in GROUPS.items():
    elig = n1[(n1 >= 5) & n1.index.isin(us)].index
    top5 = w1[(w1["rank"] <= 5) & w1.user_id.isin(elig)]
    pairs = list(zip(top5.user_id, top5.tv_show_id))
    q = np.array([p in w3_q for p in pairs]); a = np.array([p in w3_any for p in pairs])
    in1120 = top5.tv_show_id.isin(ids_1120).values
    rows.append({"group": g, "users_ge5_in_W1": len(elig), "top5_pairs": len(pairs),
                 "share_pairs_qualifying_in_W3": q.mean(), "share_pairs_any_watch_in_W3": a.mean(),
                 "distinct_target_shows": top5.tv_show_id.nunique(),
                 "distinct_target_shows_with_0_airings_in_11-20_schedule": top5.tv_show_id[~in1120].nunique(),
                 "share_pairs_show_absent_from_11-20_schedule": 1 - in1120.mean()})
rc = pd.DataFrame(rows)
rc.to_csv(OUT / "repeat_consumption_W1_vs_W3.csv", index=False)
print(rc.T.to_string())

# ------------------------------------------------------------------ D6 popularity and categories
titles = (lab_sched.groupby(["tv_show_id", "tv_show_title"]).size().reset_index(name="n")
          .sort_values(["tv_show_id", "n"], ascending=[True, False]).drop_duplicates("tv_show_id")
          .set_index("tv_show_id").tv_show_title)
cat = lab_sched.drop_duplicates("tv_show_id").set_index("tv_show_id").tv_show_category
q11 = train[train.frac >= 0.8]
pop = q11.groupby("tv_show_id").agg(distinct_users=("user_id", "nunique"), qualifying_pairs=("user_id", "size"))
pop = pop.sort_values(["distinct_users", "qualifying_pairs"], ascending=False).head(20)
pop["title_most_frequent"] = titles.reindex(pop.index).values
pop["category"] = cat.reindex(pop.index).values
pop["category_en"] = pop.category.map(CAT_EN)
pop.reset_index().to_csv(OUT / "top20_shows_by_distinct_users.csv", index=False, encoding="utf-8-sig")
print(pop.to_string())
cs = pd.DataFrame({"share_of_qualifying_pairs": q11.tv_show_id.map(cat).value_counts(normalize=True),
                   "share_of_labelled_airings_11_30": lab_sched.tv_show_category.value_counts(normalize=True)}).fillna(0)
cs = cs.sort_values("share_of_labelled_airings_11_30")
cs.index = [f"{CAT_EN.get(c, c)} ({c})" for c in cs.index]
cs.rename_axis("category").reset_index().to_csv(OUT / "category_share_qualifying_vs_schedule.csv", index=False,
                                                 encoding="utf-8-sig")
print(cs.to_string())
fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(cs))
ax.barh(y - 0.2, cs.share_of_labelled_airings_11_30, height=0.4, color=GREY, label="labelled schedule airings")
ax.barh(y + 0.2, cs.share_of_qualifying_pairs, height=0.4, color=BLUE, label="qualifying (user, airing) pairs")
ax.set_yticks(y, cs.index)
ax.set(title="Category share: what airs vs what is watched (frac >= 0.8, weeks 11-30)", xlabel="Share")
ax.legend(loc="lower right"); savefig(fig, "12_category_share.png")

mi = psutil.Process().memory_info()
summary["runtime_seconds"] = round(time.perf_counter() - T0, 1)
summary["peak_memory_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 1024**2)
(OUT / "diagnostics_step2_stats.json").write_text(json.dumps(summary, indent=2))
print(summary)
