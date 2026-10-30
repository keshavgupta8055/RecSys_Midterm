"""Step 3: validation of baselines, ceiling/robustness/breakdown, example users, test submissions.

Run: python -m src.run_step3     (no learned model is built here)
"""
import hashlib
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from src import io_utils as io
from src.baselines import PF_VARIANTS, POP_VARIANTS, pf_recommend, pf_scores, pop_recommend, popularity
from src.evaluate import ap_at_k, bootstrap_ci, map_at_k, paired_bootstrap_diff
from src.labels import build_targets, n_distinct_shows
from src.make_submission import write_submission
from src.protocol import K, THR, W, get_split, labelled_airings, load_interactions, submission_users

np.random.seed(42)
T0 = time.perf_counter()
OUT, FIG = io.ROOT / "outputs", io.ROOT / "outputs" / "figures"
DPI = 150
BLUE, ORANGE, GREY, RED, GREEN = "#3b6ea5", "#e08a2c", "#9a9a9a", "#c0392b", "#4a9a5b"
summary = {}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)


def mean_ci(ap):
    lo, hi = bootstrap_ci(ap)
    return float(ap.mean()), lo, hi


# ====================================================================== validation split
sp = get_split("val")
print("VAL split:", sp.info)
summary["val_split"] = sp.info
SUB = set(int(u) for u in sp.users[sp.sub_mask])
truth = sp.truth


def split_pop(ap):
    return {"ALL": ap, "SUB": ap[ap.index.isin(SUB)]}


def evaluate(name, family, recs, padded, t0):
    _, ap = map_at_k(recs, truth)
    row = {"method": name, "family": family}
    for pop, a in split_pop(ap).items():
        m, lo, hi = mean_ci(a)
        row.update({f"map5_{pop}": m, f"ci_lo_{pop}": lo, f"ci_hi_{pop}": hi, f"n_{pop}": len(a),
                    f"mean_padded_{pop}": float(np.mean([padded[u] for u in a.index]))})
    RUNTIMES[name] = round(time.perf_counter() - t0, 2)  # kept out of the hashed results CSV
    return row, ap


RUNTIMES = {}


rows, APS, RECS, POPRANK = [], {}, {}, {}
for v in POP_VARIANTS:
    t0 = time.perf_counter()
    POPRANK[v] = popularity(sp.train, sp.pool, v, sp.train_end)
    RECS[v], padded = pop_recommend(sp.users, POPRANK[v])
    r, APS[v] = evaluate(v, "popularity baseline", RECS[v], padded, t0)
    rows.append(r)
    print(r)
res = pd.DataFrame(rows)
res["avg_all_sub"] = (res.map5_ALL + res.map5_SUB) / 2
POP = res.sort_values(["avg_all_sub", "method"], ascending=[False, True]).method.iloc[0]
print("chosen popularity baseline (best mean of ALL and SUB MAP@5):", POP)

for v in PF_VARIANTS:
    t0 = time.perf_counter()
    RECS[v], padded = pf_recommend(sp.train, sp.pool, sp.users, v, POPRANK[POP], sp.train_end)
    r, APS[v] = evaluate(v, f"personal-frequency baseline (padded with {POP})", RECS[v], padded, t0)
    rows.append(r)
    print(r)
res = pd.DataFrame(rows)
res["avg_all_sub"] = (res.map5_ALL + res.map5_SUB) / 2
PF = res[res.method.isin(PF_VARIANTS)].sort_values(["avg_all_sub", "method"], ascending=[False, True]).method.iloc[0]
res["chosen"] = res.method.isin([POP, PF])
res.to_csv(OUT / "baseline_results_val.csv", index=False)
pd.Series(RUNTIMES, name="runtime_s").rename_axis("method").reset_index().to_csv(
    OUT / "baseline_runtimes_val.csv", index=False)
print(res[["method", "map5_ALL", "ci_lo_ALL", "ci_hi_ALL", "map5_SUB", "ci_lo_SUB", "ci_hi_SUB",
           "mean_padded_ALL", "mean_padded_SUB"]].round(4).to_string(index=False))
print("runtimes (s):", RUNTIMES)
print("chosen personal baseline:", PF)
summary.update({"chosen_pop": POP, "chosen_pf": PF})

pd_rows = []
for pop in ["ALL", "SUB"]:
    d, lo, hi = paired_bootstrap_diff(split_pop(APS[PF])[pop], split_pop(APS[POP])[pop])
    pd_rows.append({"population": pop, "comparison": f"{PF} - {POP}", "mean_diff": d, "ci_lo": lo, "ci_hi": hi})
    hl = [v for v in PF_VARIANTS if v != PF]
    for v in hl:
        d, lo, hi = paired_bootstrap_diff(split_pop(APS[PF])[pop], split_pop(APS[v])[pop])
        pd_rows.append({"population": pop, "comparison": f"{PF} - {v}", "mean_diff": d, "ci_lo": lo, "ci_hi": hi})
pdiff = pd.DataFrame(pd_rows)
pdiff.to_csv(OUT / "baseline_paired_diffs_val.csv", index=False)
print(pdiff.round(5).to_string(index=False))

# ====================================================================== C4 ceiling
tr = sp.train
hist_any = tr.groupby("user_id").tv_show_id.agg(set)
hist_q = tr[tr.frac >= THR].groupby("user_id").tv_show_id.agg(set)
ids_1120 = set(labelled_airings(["11-20"]).tv_show_id.unique().tolist())
crow = []
for pop, us in [("ALL", list(truth)), ("SUB", [u for u in truth if u in SUB])]:
    in_any = in_q = absent = 0
    ap_any, ap_q = [], []
    for u in us:
        t = truth[u]
        ha = hist_any.get(np.uint64(u), set())
        hq = hist_q.get(np.uint64(u), set())
        in_any += sum(x in ha for x in t)
        in_q += sum(x in hq for x in t)
        absent += sum(x not in ids_1120 for x in t)
        dummies = [-1, -2, -3, -4, -5]
        ap_any.append(ap_at_k(([x for x in t if x in ha] + dummies)[:K], t))
        ap_q.append(ap_at_k(([x for x in t if x in hq] + dummies)[:K], t))
    n = len(us) * K
    crow.append({"population": pop, "users": len(us), "truth_pairs": n,
                 "share_truth_in_history_any_watch": in_any / n, "share_truth_in_history_qualifying": in_q / n,
                 "ceiling_map5_history_any_watch": float(np.mean(ap_any)),
                 "ceiling_map5_history_qualifying": float(np.mean(ap_q)),
                 "share_truth_shows_absent_from_11_20_schedule": absent / n,
                 f"map5_{POP}": float(split_pop(APS[POP])[pop].mean()), f"map5_{PF}": float(split_pop(APS[PF])[pop].mean())})
ceil = pd.DataFrame(crow)
ceil.to_csv(OUT / "ceiling_analysis_val.csv", index=False)
print(ceil.T.to_string())

# ====================================================================== C5 robustness
inter = load_interactions()
clean = inter[inter.iso_week.isin([21, 22, 23, 24, 28, 29, 30])]
TRUTHS = {"main: weeks 21-30": build_targets(inter, W[21], W[31], THR),
          "clean: 21-24 + 28-30 (no 25-27)": build_targets(clean, W[21], W[31], THR),
          "weeks 21-25": build_targets(inter, W[21], W[26], THR),
          "weeks 26-30": build_targets(inter, W[26], W[31], THR)}
sub_all = set(int(u) for u in submission_users()[1])
rrows = []
for name, tg in TRUTHS.items():
    nds = n_distinct_shows(tg)
    us = nds[nds >= K].index.values.astype(np.uint64)
    top = tg[(tg["rank"] <= K) & tg.user_id.isin(us)]
    tdict = {int(u): g.tv_show_id.tolist() for u, g in top.groupby("user_id")}
    assert set(x for v in tdict.values() for x in v) <= set(sp.pool.tolist())
    pf_r, _ = pf_recommend(sp.train, sp.pool, us, PF, POPRANK[POP], sp.train_end)
    pop_r, _ = pop_recommend(us, POPRANK[POP])
    for meth, recs in [(POP, pop_r), (PF, pf_r)]:
        _, ap = map_at_k(recs, tdict)
        for pop, a in [("ALL", ap), ("SUB", ap[ap.index.isin(sub_all)])]:
            m, lo, hi = mean_ci(a)
            rrows.append({"truth_window": name, "method": meth, "population": pop, "users": len(a),
                          "map5": m, "ci_lo": lo, "ci_hi": hi})
rob = pd.DataFrame(rrows)
rob.to_csv(OUT / "robustness_target_windows_val.csv", index=False)
print(rob.round(4).to_string(index=False))

# ====================================================================== C6 breakdown
nds_train = tr[tr.frac >= THR].groupby("user_id").tv_show_id.nunique()
act = pd.Series({u: int(nds_train.get(np.uint64(u), 0)) for u in truth})
bins, labels = [-1, 4, 9, 19, 49, np.inf], ["0-4", "5-9", "10-19", "20-49", "50+"]
bucket = pd.cut(act, bins=bins, labels=labels)
brows = []
for pop in ["ALL", "SUB"]:
    for b in labels:
        for meth in [POP, PF]:
            a = split_pop(APS[meth])[pop]
            a = a[bucket.reindex(a.index) == b]
            m, lo, hi = mean_ci(a) if len(a) else (np.nan, np.nan, np.nan)
            brows.append({"population": pop, "activity_bucket": b, "method": meth, "users": len(a),
                          "map5": m, "ci_lo": lo, "ci_hi": hi})
bk = pd.DataFrame(brows)
bk.to_csv(OUT / "breakdown_activity_bucket_val.csv", index=False)
print(bk.round(4).to_string(index=False))

hrows = []
for meth in [POP, PF]:
    for pop in ["ALL", "SUB"]:
        us = split_pop(APS[meth])[pop].index
        hits = np.array([[RECS[meth][u][i] in set(truth[u]) for i in range(K)] for u in us])
        for i in range(K):
            hrows.append({"method": meth, "population": pop, "rank": i + 1,
                          "hit_rate_at_rank": hits[:, i].mean(), "share_of_all_hits": hits[:, i].sum() / hits.sum(),
                          "mean_hits_per_user": hits.sum(1).mean()})
hr = pd.DataFrame(hrows)
hr.to_csv(OUT / "hit_rank_distribution_val.csv", index=False)
print(hr.round(4).to_string(index=False))

drows = []
for meth in [POP, PF]:
    for pop in ["ALL", "SUB"]:
        a = split_pop(APS[meth])[pop]
        d = a.describe(percentiles=[.25, .5, .75, .9]).to_dict()
        d.update({"method": meth, "population": pop, "share_ap_0": float((a == 0).mean()),
                  "share_ap_1": float((a == 1).mean())})
        drows.append(d)
pd.DataFrame(drows).to_csv(OUT / "per_user_ap_distribution_val.csv", index=False)

# ====================================================================== B4 hand-computed MAP check
meta = sp.meta
lines = [f"Hand-computed AP@5 check, personal baseline {PF} (validation split)"]
cand = None
for u in sorted(SUB):  # first SUB user with 2-4 hits and at least one miss before the last hit
    hit = [r in set(truth[u]) for r in RECS[PF][u]]
    if 2 <= sum(hit) <= 4 and not all(hit[:max(i for i, h in enumerate(hit) if h) + 1]):
        cand = u
        break
u = cand
rec, t = RECS[PF][u], truth[u]
lines += [f"user_id = {u} (SUB user)",
          "recommended (rank: id title [category] -> hit?):"]
for i, r in enumerate(rec, 1):
    lines.append(f"  {i}: {r} {meta.title.get(r)} [{meta.category.get(r)}] -> {'HIT' if r in t else 'miss'}")
lines.append("truth (5 ids, order irrelevant): " + ", ".join(f"{x} {meta.title.get(x)}" for x in t))
hits, terms = 0, []
for i, r in enumerate(rec, 1):
    if r in t:
        hits += 1
        lines.append(f"  rank {i}: hit #{hits} -> P@{i} = {hits}/{i} = {hits / i:.6f}")
        terms.append((hits, i))
    else:
        lines.append(f"  rank {i}: miss -> contributes 0")
num = sum(h / i for h, i in terms)
lines.append(f"  AP@5 = ({' + '.join(f'{h}/{i}' for h, i in terms)}) / min(5, |truth|=5) = {num:.6f} / 5 = {num / 5:.6f}")
lines.append(f"  ap_at_k(...) = {ap_at_k(rec, t):.6f} -> {'MATCH' if abs(ap_at_k(rec, t) - num / 5) < 1e-12 else 'MISMATCH'}")
(OUT / "verify_map_hand.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

# ====================================================================== D example users
nsub = act[act.index.isin(SUB)]
elig = nsub[nsub >= 5]
p25, p90 = float(np.percentile(elig, 25)), float(np.percentile(elig, 90))


def closest(target):
    d = pd.DataFrame({"u": elig.index, "d": (elig.values - target).__abs__()})
    return int(d.sort_values(["d", "u"]).u.iloc[0])


ex = {"light": closest(p25), "heavy": closest(p90)}
(OUT / "example_users.json").write_text(json.dumps(
    {k: {"user_id": str(v), "distinct_qualifying_training_shows": int(elig[v])} for k, v in ex.items()} |
    {"rule": f"eligible = SUB val users with >= 5 truth shows and >= 5 distinct qualifying training shows "
             f"(n={len(elig)}); light = closest to p25 = {p25}; heavy = closest to p90 = {p90}; ties smallest id"},
    indent=2))
prow = []
q = tr[tr.frac >= THR]
for role, u in ex.items():
    uq = q[q.user_id == np.uint64(u)].groupby("tv_show_id").size()
    uh = tr[tr.user_id == np.uint64(u)].groupby("tv_show_id").watched_seconds.sum() / 3600
    top10 = pd.DataFrame({"q": uq, "h": uh.reindex(uq.index)}).reset_index().sort_values(
        ["q", "h", "tv_show_id"], ascending=[False, False, True]).head(10)
    for i, r in enumerate(top10.itertuples(), 1):
        prow.append({"role": role, "user_id": str(u), "section": "train_top10", "rank": i, "tv_show_id": r.tv_show_id,
                     "title": meta.title.get(r.tv_show_id), "category": meta.category.get(r.tv_show_id),
                     "qualifying_count": int(r.q), "watched_hours": round(float(r.h), 3)})
    tg = TRUTHS["main: weeks 21-30"]
    tu = tg[(tg.user_id == np.uint64(u)) & (tg["rank"] <= K)]
    for r in tu.itertuples():
        prow.append({"role": role, "user_id": str(u), "section": "truth_val", "rank": r.rank, "tv_show_id": r.tv_show_id,
                     "title": meta.title.get(r.tv_show_id), "category": meta.category.get(r.tv_show_id),
                     "qualifying_count": int(r.n_qualifying_airings),
                     "watched_hours": round(r.watched_seconds_total / 3600, 3)})
    for meth in POP_VARIANTS + PF_VARIANTS:
        for i, x in enumerate(RECS[meth][u], 1):
            prow.append({"role": role, "user_id": str(u), "section": f"val_top5_{meth}", "rank": i, "tv_show_id": x,
                         "title": meta.title.get(x), "category": meta.category.get(x),
                         "hit": x in set(truth[u])})
    ha, hq = hist_any.get(np.uint64(u), set()), hist_q.get(np.uint64(u), set())
    print(f"example {role}: user {u}, {int(elig[u])} distinct qualifying training shows; truth in history "
          f"(any watch) {sum(x in ha for x in truth[u])}/5, (qualifying) {sum(x in hq for x in truth[u])}/5; "
          f"AP {POP}={APS[POP][u]:.4f}, {PF}={APS[PF][u]:.4f}")
    summary[f"example_{role}"] = {"user_id": str(u), "train_shows": int(elig[u]),
                                  "truth_in_hist_any": sum(x in ha for x in truth[u]),
                                  "truth_in_hist_q": sum(x in hq for x in truth[u])}
pd.DataFrame(prow).to_csv(OUT / "example_users_profiles.csv", index=False, encoding="utf-8-sig")

# ====================================================================== E test submissions
ts = get_split("test")
print("TEST split:", ts.info)
summary["test_split"] = ts.info
pop_test = popularity(ts.train, ts.pool, POP, ts.train_end)
pop_recs, _ = pop_recommend(ts.users, pop_test)
pf_recs, pf_pad = pf_recommend(ts.train, ts.pool, ts.users, PF, pop_test, ts.train_end)
to_str = dict(zip((int(u) for u in ts.users), ts.user_str))
p1 = write_submission("sub01_popularity", {to_str[u]: v for u, v in pop_recs.items()}, ts)
p2 = write_submission("sub02_personal_baseline", {to_str[u]: v for u, v in pf_recs.items()}, ts)
tq = ts.train[ts.train.frac >= THR]
n_q_any = tq.groupby("user_id").tv_show_id.nunique().reindex(ts.users, fill_value=0)
padn = pd.Series(pf_pad)
summary["test_pf"] = {"users_with_any_padding": int((padn > 0).sum()), "users_fully_padded": int((padn == K).sum()),
                      "mean_padded": float(padn.mean()),
                      "users_without_qualifying_training_show": int((n_q_any.values == 0).sum())}
print("test PF padding:", summary["test_pf"])

# ====================================================================== F plots
res_p = res.set_index("method")
fig, ax = plt.subplots(figsize=(10, 6.5))
order = POP_VARIANTS + PF_VARIANTS
y = np.arange(len(order))
for off, pop, col in [(-0.2, "ALL", BLUE), (0.2, "SUB", ORANGE)]:
    m = res_p.loc[order, f"map5_{pop}"]
    err = [m - res_p.loc[order, f"ci_lo_{pop}"], res_p.loc[order, f"ci_hi_{pop}"] - m]
    ax.barh(y + off, m, height=0.4, xerr=err, color=col, capsize=2,
            label=f"{pop} (n={int(res_p[f'n_{pop}'].iloc[0]):,})")
ax.set_yticks(y, [f"{o}{'  (chosen)' if o in (POP, PF) else ''}" for o in order])
ax.invert_yaxis()
ax.set(title="Validation MAP@5 of baselines (train weeks 11-20, truth weeks 21-30), 95% bootstrap CI",
       xlabel="MAP@5", xlim=(0, res_p.ci_hi_SUB.max() * 1.3))
ax.legend(loc="lower right")
savefig(fig, "14_baselines_map5.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
edges = np.linspace(0, 1, 26)
for ax, pop in zip(axes, ["ALL", "SUB"]):
    for meth, col in [(POP, GREY), (PF, BLUE)]:
        a = split_pop(APS[meth])[pop]
        ax.hist(a, bins=edges, histtype="step", lw=2, color=col, label=f"{meth} (mean {a.mean():.3f})")
    ax.set(title=f"Per-user AP@5, {pop} users (n={len(split_pop(APS[PF])[pop]):,})", xlabel="AP@5", ylabel="Users")
    ax.legend()
savefig(fig, "15_per_user_ap_hist.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, pop in zip(axes, ["ALL", "SUB"]):
    x = np.arange(len(labels))
    for off, meth, col in [(-0.2, POP, GREY), (0.2, PF, BLUE)]:
        d = bk[(bk.population == pop) & (bk.method == meth)].set_index("activity_bucket").loc[labels]
        ax.bar(x + off, d.map5, width=0.4, color=col, yerr=[d.map5 - d.ci_lo, d.ci_hi - d.map5], capsize=3, label=meth)
    n = bk[(bk.population == pop) & (bk.method == PF)].set_index("activity_bucket").loc[labels].users
    ax.set_xticks(x, [f"{b}\n(n={k})" for b, k in zip(labels, n)])
    ax.set(title=f"MAP@5 by training activity, {pop} users",
           xlabel="Distinct qualifying shows in training weeks 11-20", ylabel="MAP@5")
    ax.legend()
savefig(fig, "16_map5_by_activity.png")

fig, ax = plt.subplots(figsize=(8, 5))
for off, meth, col in [(-0.2, POP, GREY), (0.2, PF, BLUE)]:
    d = hr[(hr.method == meth) & (hr.population == "ALL")]
    ax.bar(d["rank"] + off, d.hit_rate_at_rank, width=0.4, color=col, label=meth)
ax.set(title="Hit rate by recommendation rank (ALL users, validation)", xlabel="Rank in recommended top-5",
       ylabel="Share of users whose item at this rank is a truth show")
ax.legend(); savefig(fig, "17_hit_rank.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
c = ceil.set_index("population")
x = np.arange(2)
for off, col_, lab, cl in [(-0.3, f"map5_{POP}", POP, GREY), (-0.1, f"map5_{PF}", PF, BLUE),
                           (0.1, "ceiling_map5_history_qualifying", "ceiling: perfect ranker, qualifying history", GREEN),
                           (0.3, "ceiling_map5_history_any_watch", "ceiling: perfect ranker, any-watch history", ORANGE)]:
    axes[0].bar(x + off, c.loc[["ALL", "SUB"], col_], width=0.2, color=cl, label=lab)
axes[0].set_xticks(x, ["ALL", "SUB"])
axes[0].set(title="MAP@5: baselines vs history-only ceiling", ylabel="MAP@5", ylim=(0, 1))
axes[1].set_ylim(0, 1)
axes[0].legend(fontsize=8)
for off, col_, lab, cl in [(-0.25, "share_truth_in_history_any_watch", "in history (any watch)", ORANGE),
                           (0, "share_truth_in_history_qualifying", "in history (qualifying)", GREEN),
                           (0.25, "share_truth_shows_absent_from_11_20_schedule", "show absent from 11-20 schedule", RED)]:
    axes[1].bar(x + off, c.loc[["ALL", "SUB"], col_], width=0.25, color=cl, label=lab)
axes[1].set_xticks(x, ["ALL", "SUB"])
axes[1].set(title="Truth (user, show) pairs: share in the user's training history", ylabel="Share of truth pairs")
axes[1].legend(fontsize=8)
savefig(fig, "18_ceiling.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
wins = list(TRUTHS)
for ax, pop in zip(axes, ["ALL", "SUB"]):
    x = np.arange(len(wins))
    for off, meth, col in [(-0.2, POP, GREY), (0.2, PF, BLUE)]:
        d = rob[(rob.population == pop) & (rob.method == meth)].set_index("truth_window").loc[wins]
        ax.bar(x + off, d.map5, width=0.4, color=col, yerr=[d.map5 - d.ci_lo, d.ci_hi - d.map5], capsize=3, label=meth)
    n = rob[(rob.population == pop) & (rob.method == PF)].set_index("truth_window").loc[wins].users
    short = ["weeks 21-30\n(main)", "21-24 + 28-30\n(excl. 25-27)", "weeks 21-25", "weeks 26-30"]
    ax.set_xticks(x, [f"{w}\n(n={k:,})" for w, k in zip(short, n)], fontsize=9)
    ax.set_xlabel("Target window used to build the truth")
    ax.set(title=f"MAP@5 across target windows, {pop} users (train weeks 11-20)", ylabel="MAP@5")
    ax.legend()
savefig(fig, "19_robustness_windows.png")

# ====================================================================== manifest
mi = psutil.Process().memory_info()
summary["runtime_seconds"] = round(time.perf_counter() - T0, 1)
summary["peak_memory_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 1024**2)
files = ["baseline_results_val.csv", "baseline_paired_diffs_val.csv", "ceiling_analysis_val.csv",
         "robustness_target_windows_val.csv", "breakdown_activity_bucket_val.csv", "hit_rank_distribution_val.csv",
         "per_user_ap_distribution_val.csv", "verify_map_hand.txt", "example_users.json",
         "example_users_profiles.csv", "submissions/sub01_popularity.csv", "submissions/sub02_personal_baseline.csv"]
summary["sha256"] = {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in files}
(OUT / "step3_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "sha256"}, indent=2))
