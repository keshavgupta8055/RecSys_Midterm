"""Step 4: Method A = Personalized PageRank. Run: python -m src.run_step4"""
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
from src.baselines import pf_recommend, pop_recommend, popularity
from src.evaluate import ap_at_k, bootstrap_ci, map_at_k, paired_bootstrap_diff
from src.labels import build_targets, n_distinct_shows
from src.make_submission import write_submission
from src.ppr import build_graph, lazy_pop, personalisation, ppr, recommend
from src.protocol import K, THR, W, get_split, load_interactions, pad_with_popularity

np.random.seed(42)
T0 = time.perf_counter()
OUT, FIG = io.ROOT / "outputs", io.ROOT / "outputs" / "figures"
DPI = 150
BLUE, ORANGE, GREY, RED, GREEN = "#3b6ea5", "#e08a2c", "#9a9a9a", "#c0392b", "#4a9a5b"
EDGES, DS, WEIGHTS, BETAS = ["raw", "cosine", "weighted_cosine"], [0.3, 0.5, 0.7, 0.85], ["softfrac", "count08"], [0, 0.5]
summary = {}
proc = psutil.Process()


def peak_mb():
    mi = proc.memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 1024**2)


def log(*a):
    print(*a, flush=True)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)


# ====================================================================== setup
sp = get_split("val")
SUB = set(int(u) for u in sp.users[sp.sub_mask])
truth = sp.truth
tr = sp.train
log(f"[B4] val graph training airing_start min {tr.airing_start.min()} max {tr.airing_start.max()}")
assert tr.airing_start.max() < W[21]
summary["leakage_val"] = {"min": str(tr.airing_start.min()), "max": str(tr.airing_start.max())}

POPV = popularity(tr, sp.pool, "pop_recent", sp.train_end)
pop_recs, _ = pop_recommend(sp.users, POPV)
pf_recs, pf_pad = pf_recommend(tr, sp.pool, sp.users, "PF_softfrac", POPV, sp.train_end)
step3 = pd.read_csv(OUT / "baseline_results_val.csv").set_index("method")
hist = {int(u): set(s) for u, s in tr.groupby("user_id").tv_show_id.agg(set).items()}
nq = tr[tr.frac >= THR].groupby("user_id").tv_show_id.nunique()


def evaluate(recs, tdict=None, sub=SUB):
    tdict = truth if tdict is None else tdict
    _, ap = map_at_k(recs, tdict)
    res = {}
    for pop, a in (("ALL", ap), ("SUB", ap[ap.index.isin(sub)])):
        lo, hi = bootstrap_ci(a)
        res[pop] = {"map5": float(a.mean()), "ci_lo": lo, "ci_hi": hi, "n": len(a), "ap": a}
    return res


E_POP, E_PF = evaluate(pop_recs), evaluate(pf_recs)
for m, e in [("pop_recent", E_POP), ("PF_softfrac", E_PF)]:
    for pop in ["ALL", "SUB"]:
        assert abs(e[pop]["map5"] - step3.loc[m, f"map5_{pop}"]) < 1e-12, (m, pop)
log("baselines reproduce Step 3 exactly:", {m: (round(e['ALL']['map5'], 4), round(e['SUB']['map5'], 4))
                                             for m, e in [("pop_recent", E_POP), ("PF_softfrac", E_PF)]})

# ====================================================================== A6 grid (float32)
log("[A6] grid: 3 edges x 4 d x 2 seed weights x 2 beta (PPR in float32; beta reuses R)")
rows, RT = [], []
for edge in EDGES:
    for weight in WEIGHTS:
        g = build_graph(tr, edge, weight, dtype=np.float32)
        P, has = personalisation(g, sp.users, dtype=np.float32)
        for d in DS:
            t0 = time.perf_counter()
            R, info = ppr(P, g.T, d)
            t_ppr = time.perf_counter() - t0
            for beta in BETAS:
                t1 = time.perf_counter()
                recs, padded, _ = recommend(R, has, sp.users, g, sp.pool, beta, POPV)
                e = evaluate(recs)
                row = {"edge": edge, "seed_weight": weight, "d": d, "beta": beta}
                for pop in ["ALL", "SUB"]:
                    row.update({f"map5_{pop}": e[pop]["map5"], f"ci_lo_{pop}": e[pop]["ci_lo"],
                                f"ci_hi_{pop}": e[pop]["ci_hi"]})
                row.update({"avg_all_sub": (e["ALL"]["map5"] + e["SUB"]["map5"]) / 2,
                            "mean_padded_ALL": float(np.mean([padded[u] for u in truth])),
                            "users_without_seed": int((~has).sum()), "ppr_iterations_max": info["max_iterations"],
                            "converged_1e-6": info["converged"], "final_max_l1_change": info["final_max_l1_change"]})
                rows.append(row)
                RT.append({**{k: row[k] for k in ["edge", "seed_weight", "d", "beta"]},
                           "graph_build_seconds": round(g.build_seconds, 2), "ppr_seconds": round(t_ppr, 2),
                           "recommend_eval_seconds": round(time.perf_counter() - t1, 2),
                           "runtime_seconds": round(g.build_seconds + t_ppr + time.perf_counter() - t1, 2)})
                log(f"  {edge:16s} {weight:8s} d={d:<5} beta={beta:<4} ALL {row['map5_ALL']:.4f} SUB {row['map5_SUB']:.4f} "
                    f"iters {info['max_iterations']} {RT[-1]['runtime_seconds']}s")
        del g, P
grid = pd.DataFrame(rows).merge(pd.DataFrame(RT), on=["edge", "seed_weight", "d", "beta"])
grid.to_csv(OUT / "ppr_grid_val.csv", index=False)
rank = grid.assign(er=grid.edge.map({e: i for i, e in enumerate(EDGES)}),
                   wr=grid.seed_weight.map({"count08": 0, "softfrac": 1}))
best = rank.sort_values(["avg_all_sub", "runtime_seconds", "er", "beta", "wr"],
                        ascending=[False, True, True, True, True]).iloc[0]
CFG = {"edge": best.edge, "seed_weight": best.seed_weight, "d": float(best.d), "beta": float(best.beta)}
summary["selected"] = CFG
summary["selected_grid_float32"] = {"map5_ALL": float(best.map5_ALL), "map5_SUB": float(best.map5_SUB)}
log("[A6] selected (best mean of ALL and SUB MAP@5; tie -> faster -> simpler):", CFG, summary["selected_grid_float32"])
summary["grid_seconds"] = round(time.perf_counter() - T0, 1)

# ====================================================================== selected config, float64
ROB = {"main: weeks 21-30": build_targets(load_interactions(), W[21], W[31], THR)}
inter = load_interactions()
ROB["clean: 21-24 + 28-30 (no 25-27)"] = build_targets(inter[inter.iso_week.isin([21, 22, 23, 24, 28, 29, 30])],
                                                       W[21], W[31], THR)
ROB["weeks 21-25"] = build_targets(inter, W[21], W[26], THR)
ROB["weeks 26-30"] = build_targets(inter, W[26], W[31], THR)
ROB_T = {}
for k, tg in ROB.items():
    nds = n_distinct_shows(tg)
    us = nds[nds >= K].index
    top = tg[(tg["rank"] <= K) & tg.user_id.isin(us)]
    ROB_T[k] = {int(u): g_.tv_show_id.tolist() for u, g_ in top.groupby("user_id")}
assert ROB_T["main: weeks 21-30"] == truth
U = np.array(sorted(set().union(*[set(t) for t in ROB_T.values()])), dtype=np.uint64)


def run_selected(train, users, pool, pop, cfg, history_=None):
    t0 = time.perf_counter()
    g = build_graph(train, cfg["edge"], cfg["seed_weight"])
    P, has = personalisation(g, users)
    t1 = time.perf_counter()
    R, info = ppr(P, g.T, cfg["d"])
    t_ppr = time.perf_counter() - t1
    full = recommend(R, has, users, g, pool, cfg["beta"], pop)
    new = recommend(R, has, users, g, pool, cfg["beta"], pop, mode="new_only", history=history_) if history_ else None
    return {"g": g, "P": P, "has": has, "R": R, "info": info, "full": full, "new": new,
            "graph_seconds": g.build_seconds, "ppr_seconds": t_ppr, "total_seconds": time.perf_counter() - t0}


S = run_selected(tr, U, sp.pool, POPV, CFG, hist)
G = S["g"]
uidx = {int(u): i for i, u in enumerate(U)}
ppr_recs = {u: S["full"][0][u] for u in truth}
ppr_pad = {u: S["full"][1][u] for u in truth}
E_PPR = evaluate(ppr_recs)
log("[C1] selected PPR float64:", {p: round(E_PPR[p]["map5"], 4) for p in ["ALL", "SUB"]},
    "iterations", S["info"]["iterations_per_batch"], "converged", S["info"]["converged"])
summary["ppr_val"] = {p: {k: v for k, v in E_PPR[p].items() if k != "ap"} for p in ["ALL", "SUB"]}
summary["ppr_val_iterations"] = S["info"]["iterations_per_batch"]
summary["ppr_val_mean_padded_ALL"] = float(np.mean(list(ppr_pad.values())))
summary["graph_val"] = {"nodes": len(G.shows), "undirected_edges": G.n_edges,
                        "density": G.n_edges / (len(G.shows) * (len(G.shows) - 1) / 2), "isolated_nodes": G.n_isolated,
                        "seed_users": len(G.users)}

# ---------------------------------------------------------------- B3 sanity
rs = G.T.sum(axis=1)
Rs = S["R"][S["has"]].sum(axis=1)
summary["sanity"] = {"T_row_sum_min": float(rs.min()), "T_row_sum_max": float(rs.max()),
                     "isolated_nodes_selfloop": G.n_isolated,
                     "R_row_sum_min": float(Rs.min()), "R_row_sum_max": float(Rs.max())}
gs = build_graph(tr, CFG["edge"], "softfrac")
Ps, hs = personalisation(gs, sp.users)
R01, _ = ppr(Ps, gs.T, 0.01)
r01, _, raw01 = recommend(R01, hs, sp.users, gs, sp.pool, 0, POPV)
same = np.array([r01[u] == pf_recs[u] for u in truth])
n_pool_seed = pd.Series({u: len(hist.get(u, set()) & set(sp.pool.tolist())) for u in truth})
summary["sanity"]["d0.01_top5_equals_PF_softfrac_share"] = float(same.mean())
summary["sanity"]["d0.01_share_among_users_with_ge5_pool_seeds"] = float(same[(n_pool_seed >= 5).values].mean())
log("[B3]", summary["sanity"])
del gs, Ps, R01

# ---------------------------------------------------------------- B2 naive cross-check
rng = np.random.default_rng(42)
pick = rng.choice(sp.users, 10, replace=False)  # uint64 array -> uint64 picks
assert pick.dtype == np.uint64 and all(int(u) in truth for u in pick)
log("[B2] naive pure-Python PPR for 10 users ...")
T = G.T
adj = []
for i in range(len(G.shows)):
    nz = np.flatnonzero(T[i])
    adj.append((nz.tolist(), T[i, nz].tolist()))
col = {int(x): i for i, x in enumerate(G.shows)}
pool_set = set(sp.pool.tolist())
deg = G.deg.tolist()
nrows = []
for u in pick:
    u = int(u)
    seed = {}
    for sh, fr in zip(tr.tv_show_id[tr.user_id == np.uint64(u)].tolist(), tr.frac[tr.user_id == np.uint64(u)].tolist()):
        w = fr if CFG["seed_weight"] == "softfrac" else (1.0 if fr >= THR else 0.0)
        seed[col[sh]] = seed.get(col[sh], 0.0) + w
    seed = {i: w for i, w in seed.items() if w > 0}
    tot = sum(seed.values())
    p = [0.0] * len(adj)
    for i, w in seed.items():
        p[i] = w / tot
    r = list(p)
    d = CFG["d"]
    t0 = time.perf_counter()
    for it in range(1, 51):
        new = [(1 - d) * x for x in p]
        for i in range(len(adj)):
            ri = r[i]
            if ri:
                idx, val = adj[i]
                for j, t in zip(idx, val):
                    new[j] += d * ri * t
        delta = sum(abs(a - b) for a, b in zip(new, r))
        r = new
        if delta < 1e-6:
            break
    Rv, _ = ppr(S["P"][[uidx[u]]], T, d, n_iter=it)
    diff = float(np.max(np.abs(np.array(r) - Rv[0])))
    diff_main = float(np.max(np.abs(np.array(r) - S["R"][uidx[u]])))
    sc = [(r[i] / max(deg[i], 1) ** CFG["beta"], int(G.shows[i])) for i in range(len(adj)) if int(G.shows[i]) in pool_set]
    sc = [x for x in sc if x[0] > 0]
    sc.sort(key=lambda x: (-x[0], x[1]))
    naive_top = pad_with_popularity([x[1] for x in sc[:K]], POPV)
    nrows.append({"user_id": str(u), "naive_iterations": it, "max_abs_diff_same_iterations": diff,
                  "max_abs_diff_vs_main_run": diff_main, "top5_naive": " ".join(map(str, naive_top)),
                  "top5_vectorised": " ".join(map(str, ppr_recs[u])), "top5_identical": naive_top == ppr_recs[u],
                  "seconds": round(time.perf_counter() - t0, 1)})
    log("   ", nrows[-1])
nv = pd.DataFrame(nrows)
nv.to_csv(OUT / "verify_ppr_naive.csv", index=False)
assert (nv.max_abs_diff_same_iterations < 1e-8).all() and nv.top5_identical.all()
summary["naive_check"] = {"users": len(nv), "max_abs_diff": float(nv.max_abs_diff_same_iterations.max()),
                          "all_top5_identical": bool(nv.top5_identical.all())}
del adj

# ---------------------------------------------------------------- B5 determinism


def recs_hash(recs):
    return hashlib.sha256(json.dumps({str(k): v for k, v in sorted(recs.items())}).encode()).hexdigest()


h1 = recs_hash(ppr_recs)
S2 = run_selected(tr, sp.users, sp.pool, POPV, CFG)
h2 = recs_hash({u: S2["full"][0][u] for u in truth})
summary["determinism"] = {"hash_run1": h1, "hash_run2": h2, "identical": h1 == h2}
log("[B5]", summary["determinism"])
assert h1 == h2
del S2

# ====================================================================== C1 comparison
METHODS = {"pop_recent": (pop_recs, E_POP), "PF_softfrac": (pf_recs, E_PF), "PPR": (ppr_recs, E_PPR)}
crow = []
for pop in ["ALL", "SUB"]:
    for m, (_, e) in METHODS.items():
        crow.append({"population": pop, "method": m, "map5": e[pop]["map5"], "ci_lo": e[pop]["ci_lo"],
                     "ci_hi": e[pop]["ci_hi"], "n": e[pop]["n"], "comparison": "", "mean_diff": np.nan,
                     "diff_ci_lo": np.nan, "diff_ci_hi": np.nan})
    for base in ["PF_softfrac", "pop_recent"]:
        dlt, lo, hi = paired_bootstrap_diff(E_PPR[pop]["ap"], METHODS[base][1][pop]["ap"])
        crow.append({"population": pop, "method": "", "comparison": f"PPR - {base}", "mean_diff": dlt,
                     "diff_ci_lo": lo, "diff_ci_hi": hi})
cmp_ = pd.DataFrame(crow)
cmp_.to_csv(OUT / "ppr_vs_baselines_val.csv", index=False)
log(cmp_.round(4).to_string(index=False))

# ====================================================================== C2 familiar vs new
frows = []
for m, (recs, _) in METHODS.items():
    for pop, us in [("ALL", list(truth)), ("SUB", [u for u in truth if u in SUB])]:
        fam = np.array([[x in hist.get(u, set()) for x in recs[u]] for u in us])
        hit = np.array([[x in set(truth[u]) for x in recs[u]] for u in us])
        for slot in list(range(K)) + ["all"]:
            f = fam[:, slot] if slot != "all" else fam.ravel()
            h = hit[:, slot] if slot != "all" else hit.ravel()
            frows.append({"method": m, "population": pop, "slot": slot + 1 if slot != "all" else "all",
                          "share_familiar": f.mean(), "precision_familiar": h[f].mean() if f.any() else np.nan,
                          "precision_new": h[~f].mean() if (~f).any() else np.nan,
                          "share_of_hits_new": (h & ~f).sum() / h.sum() if h.sum() else np.nan,
                          "mean_new_items_per_list": (~fam).sum(1).mean() if slot == "all" else np.nan})
fam_df = pd.DataFrame(frows)
fam_df.to_csv(OUT / "ppr_familiar_vs_new_val.csv", index=False)
log(fam_df[fam_df.slot == "all"].round(4).to_string(index=False))

# new_only mode
truth_new = {u: [x for x in t if x not in hist.get(u, set())] for u, t in truth.items()}
truth_new = {u: t for u, t in truth_new.items() if t}
ppr_new = {u: S["new"][0][u] for u in truth_new}
pop_new = {u: pad_with_popularity([], lazy_pop(POPV, hist.get(u, set()))) for u in truth_new}
nrow, NEW_AP = [], {}
for m, recs in [("PPR_new_only", ppr_new), ("pop_recent_new_only", pop_new)]:
    ap = pd.Series({u: ap_at_k(recs[u], truth_new[u]) for u in truth_new})
    rec = pd.Series({u: len(set(recs[u]) & set(truth_new[u])) / len(truth_new[u]) for u in truth_new})
    NEW_AP[m] = (ap, rec)
    for pop, mask in [("ALL", np.ones(len(ap), bool)), ("SUB", ap.index.isin(SUB))]:
        lo, hi = bootstrap_ci(ap[mask])
        rlo, rhi = bootstrap_ci(rec[mask])
        nrow.append({"method": m, "population": pop, "users_with_truth_new": int(mask.sum()),
                     "map5_vs_truth_new": ap[mask].mean(), "ci_lo": lo, "ci_hi": hi,
                     "recall5_vs_truth_new": rec[mask].mean(), "recall_ci_lo": rlo, "recall_ci_hi": rhi})
for pop, mask_fn in [("ALL", lambda a: a), ("SUB", lambda a: a[a.index.isin(SUB)])]:
    dlt, lo, hi = paired_bootstrap_diff(mask_fn(NEW_AP["PPR_new_only"][0]), mask_fn(NEW_AP["pop_recent_new_only"][0]))
    nrow.append({"method": "PPR_new_only - pop_recent_new_only (MAP, paired)", "population": pop,
                 "map5_vs_truth_new": dlt, "ci_lo": lo, "ci_hi": hi})
newdf = pd.DataFrame(nrow)
newdf.to_csv(OUT / "ppr_new_only_val.csv", index=False)
log(newdf.round(4).to_string(index=False))

# ====================================================================== C3 breakdowns
act = pd.Series({u: int(nq.get(np.uint64(u), 0)) for u in truth})
labels = ["0-4", "5-9", "10-19", "20-49", "50+"]
bucket = pd.cut(act, [-1, 4, 9, 19, 49, np.inf], labels=labels)
brow = []
for pop in ["ALL", "SUB"]:
    for b in labels:
        for m, (_, e) in METHODS.items():
            a = e[pop]["ap"]
            a = a[bucket.reindex(a.index) == b]
            lo, hi = bootstrap_ci(a)
            brow.append({"population": pop, "activity_bucket": b, "method": m, "users": len(a), "map5": a.mean(),
                         "ci_lo": lo, "ci_hi": hi})
bk = pd.DataFrame(brow)
bk.to_csv(OUT / "ppr_breakdown_activity_val.csv", index=False)

sub_all = set(int(x) for x in get_split("test").users)
rrow = []
for k, td in ROB_T.items():
    us = np.array(sorted(td), dtype=np.uint64)
    pf_r, _ = pf_recommend(tr, sp.pool, us, "PF_softfrac", POPV, sp.train_end)
    pop_r, _ = pop_recommend(us, POPV)
    ppr_r = {int(u): S["full"][0][int(u)] for u in us}
    for m, recs in [("pop_recent", pop_r), ("PF_softfrac", pf_r), ("PPR", ppr_r)]:
        e = evaluate(recs, td, sub_all)
        for pop in ["ALL", "SUB"]:
            rrow.append({"truth_window": k, "method": m, "population": pop, "users": e[pop]["n"],
                         "map5": e[pop]["map5"], "ci_lo": e[pop]["ci_lo"], "ci_hi": e[pop]["ci_hi"]})
rob = pd.DataFrame(rrow)
rob.to_csv(OUT / "ppr_robustness_windows_val.csv", index=False)
log(rob.round(4).to_string(index=False))

hrow, drow = [], []
for m, (recs, e) in METHODS.items():
    for pop in ["ALL", "SUB"]:
        us = e[pop]["ap"].index
        hits = np.array([[recs[u][i] in set(truth[u]) for i in range(K)] for u in us])
        for i in range(K):
            hrow.append({"method": m, "population": pop, "rank": i + 1, "hit_rate_at_rank": hits[:, i].mean(),
                         "share_of_all_hits": hits[:, i].sum() / hits.sum()})
        a = e[pop]["ap"]
        dd = a.describe(percentiles=[.25, .5, .75, .9]).to_dict()
        dd.update({"method": m, "population": pop, "share_ap_0": float((a == 0).mean()), "share_ap_1": float((a == 1).mean())})
        drow.append(dd)
pd.DataFrame(hrow).to_csv(OUT / "ppr_hit_rank_val.csv", index=False)
pd.DataFrame(drow).to_csv(OUT / "ppr_per_user_ap_val.csv", index=False)

# ====================================================================== C4 coverage / diversity
meta = sp.meta
cat = meta.category
vrow, mix = [], {}
for m, (recs, _) in METHODS.items():
    lists = [recs[u] for u in truth]
    cats = [[cat.get(x) for x in l] for l in lists]
    pairdiv = [np.mean([a != b for i, a in enumerate(c) for b in c[i + 1:]]) for c in cats]
    vrow.append({"method": m, "distinct_shows_recommended": len({x for l in lists for x in l}), "pool_size": len(sp.pool),
                 "catalogue_coverage": len({x for l in lists for x in l}) / len(sp.pool),
                 "share_lists_ge3_categories": float(np.mean([len(set(c)) >= 3 for c in cats])),
                 "mean_pairwise_category_diversity": float(np.mean(pairdiv))})
    mix[m] = pd.Series([c for cs in cats for c in cs]).value_counts(normalize=True)
mix["truth"] = pd.Series([cat.get(x) for t in truth.values() for x in t]).value_counts(normalize=True)
cov = pd.DataFrame(vrow)
cov.to_csv(OUT / "ppr_coverage_diversity_val.csv", index=False)
pd.DataFrame(mix).fillna(0).rename_axis("category").reset_index().to_csv(
    OUT / "ppr_category_mix_val.csv", index=False, encoding="utf-8-sig")
log(cov.round(4).to_string(index=False))

# ====================================================================== E test file
ts = get_split("test")
log(f"[B4] test graph training airing_start min {ts.train.airing_start.min()} max {ts.train.airing_start.max()}")
assert ts.train.airing_start.max() < W[31]
summary["leakage_test"] = {"min": str(ts.train.airing_start.min()), "max": str(ts.train.airing_start.max())}
pop_test = popularity(ts.train, ts.pool, "pop_recent", ts.train_end)
ST = run_selected(ts.train, ts.users, ts.pool, pop_test, CFG)
to_str = dict(zip((int(u) for u in ts.users), ts.user_str))
write_submission("sub03_ppr", {to_str[u]: v for u, v in ST["full"][0].items()}, ts)
tpad = pd.Series(ST["full"][1])
summary["test"] = {"nodes": len(ST["g"].shows), "undirected_edges": ST["g"].n_edges,
                   "isolated_nodes": ST["g"].n_isolated, "iterations": ST["info"]["iterations_per_batch"],
                   "users_padded": int((tpad > 0).sum()), "users_no_seed": int((~ST["has"]).sum()),
                   "mean_padded": float(tpad.mean())}
log("[E]", summary["test"])

cost = pd.DataFrame([
    {"split": "val", "nodes": len(G.shows), "undirected_edges": G.n_edges, "users_scored": len(U),
     "graph_build_seconds": round(S["graph_seconds"], 2), "ppr_seconds_all_users": round(S["ppr_seconds"], 2),
     "ppr_iterations": max(S["info"]["iterations_per_batch"])},
    {"split": "test", "nodes": len(ST["g"].shows), "undirected_edges": ST["g"].n_edges, "users_scored": len(ts.users),
     "graph_build_seconds": round(ST["graph_seconds"], 2), "ppr_seconds_all_users": round(ST["ppr_seconds"], 2),
     "ppr_iterations": max(ST["info"]["iterations_per_batch"])}])

# ====================================================================== D example users
ex = json.loads((OUT / "example_users.json").read_text())
erow = []
for role in ["light", "heavy"]:
    u = int(ex[role]["user_id"])
    i = uidx[u]
    p = S["P"][i]
    seeds = np.flatnonzero(p)
    top_seeds = seeds[np.argsort(-p[seeds], kind="stable")][:10]
    parts = []
    for j in top_seeds:
        q = np.zeros_like(p)
        q[j] = p[j]
        parts.append(q)
    rest = p.copy()
    rest[top_seeds] = 0
    parts.append(rest)
    parts.append(p)
    Rd, _ = ppr(np.vstack(parts), G.T, CFG["d"], n_iter=50)
    comp, total = Rd[:-1], Rd[-1]
    sum_err = float(np.abs(comp.sum(0) - total).max())
    assert sum_err < 1e-8
    pen = np.maximum(G.deg, 1) ** CFG["beta"]
    uq = tr[(tr.user_id == np.uint64(u)) & (tr.frac >= THR)].groupby("tv_show_id").size()
    seed_names = [f"{int(G.shows[j])} {meta.title.get(int(G.shows[j]))}" for j in top_seeds] + ["(all other seeds)"]
    log(f"\n[D] {role} user {u}: decomposition check max|sum(parts) - total| = {sum_err:.2e}")
    for section, recs in [("PPR_full", S["full"][0][u]), ("PPR_new_only", S["new"][0][u]),
                          ("PF_softfrac", pf_recs[u]), ("pop_recent", pop_recs[u])]:
        for rnk, x in enumerate(recs, 1):
            c = col.get(x)
            row = {"role": role, "user_id": str(u), "list": section, "rank": rnk, "tv_show_id": x,
                   "title": meta.title.get(x), "category": meta.category.get(x), "hit": x in set(truth[u]),
                   "in_training_history": x in hist.get(u, set()), "training_qualifying_count": int(uq.get(x, 0))}
            if section.startswith("PPR") and c is not None:
                row["ppr_score"] = float(S["R"][i, c] / pen[c])
                if section == "PPR_full":
                    sh = comp[:, c] / total[c]
                    o = np.argsort(-sh, kind="stable")[:3]
                    for n_, k_ in enumerate(o, 1):
                        row[f"seed{n_}"] = seed_names[k_]
                        row[f"seed{n_}_share"] = round(float(sh[k_]), 4)
            erow.append(row)
    ov = len(set(S["full"][0][u]) & set(pf_recs[u]))
    log(f"  overlap PPR top-5 vs PF_softfrac top-5: {ov}")
    summary[f"example_{role}_overlap_ppr_pf"] = ov
exdf = pd.DataFrame(erow)
exdf.to_csv(OUT / "example_users_ppr.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 30):
    log(exdf.drop(columns=["user_id"]).to_string(index=False))

# ====================================================================== F plots
gb = grid[(grid.seed_weight == CFG["seed_weight"]) & (grid.beta == CFG["beta"])]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, pop in zip(axes, ["ALL", "SUB"]):
    M = gb.pivot(index="d", columns="edge", values=f"map5_{pop}")[EDGES]
    im = ax.imshow(M.values, cmap="Blues", aspect="auto")
    for (a, b), v in np.ndenumerate(M.values):
        ax.text(b, a, f"{v:.4f}", ha="center", va="center",
                color="white" if v > (M.values.min() + M.values.max()) / 2 else "black", fontsize=10)
    ax.set_xticks(range(len(EDGES)), EDGES)
    ax.set_yticks(range(len(M.index)), [str(x) for x in M.index])
    ax.set(title=f"PPR MAP@5, {pop} users (seed {CFG['seed_weight']}, beta {CFG['beta']})",
           xlabel="Edge weight", ylabel="d (continue probability)")
    fig.colorbar(im, ax=ax, shrink=0.8)
savefig(fig, "20_ppr_grid_heatmap.png")

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(2)
for off, (m, (_, e)), c in zip([-0.27, 0, 0.27], METHODS.items(), [GREY, BLUE, GREEN]):
    v = [e[p]["map5"] for p in ["ALL", "SUB"]]
    err = [[e[p]["map5"] - e[p]["ci_lo"] for p in ["ALL", "SUB"]], [e[p]["ci_hi"] - e[p]["map5"] for p in ["ALL", "SUB"]]]
    ax.bar(x + off, v, width=0.27, yerr=err, capsize=4, color=c, label=m if m != "PPR" else "PPR (Method A)")
    for xx, vv in zip(x + off, v):
        ax.text(xx, vv / 2, f"{vv:.3f}", ha="center", color="white", fontsize=9)
ax.set_xticks(x, ["ALL (n=4,030)", "SUB (n=1,150)"])
ax.set(title="Validation MAP@5 with 95% bootstrap CI", ylabel="MAP@5")
ax.legend(); savefig(fig, "21_map5_three_methods.png")

fa = fam_df[(fam_df.slot == "all") & (fam_df.population == "ALL")].set_index("method").loc[list(METHODS)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
xm = np.arange(len(METHODS))
axes[0].bar(xm, fa.share_familiar, color=BLUE, label="familiar (in training history)")
axes[0].bar(xm, 1 - fa.share_familiar, bottom=fa.share_familiar, color=ORANGE, label="new")
axes[0].set_xticks(xm, list(METHODS))
axes[0].set(title="Composition of recommended items (ALL users)", ylabel="Share of recommended items", ylim=(0, 1.25))
axes[0].legend(loc="upper center", ncol=2)
axes[1].bar(xm - 0.2, fa.precision_familiar, width=0.4, color=BLUE, label="precision of familiar items")
axes[1].bar(xm + 0.2, fa.precision_new.fillna(0), width=0.4, color=ORANGE, label="precision of new items")
axes[1].set_xticks(xm, list(METHODS))
axes[1].set(title="Share of recommended items that are in the truth (ALL users)", ylabel="Precision",
            ylim=(0, fa.precision_familiar.max() * 1.35))
axes[1].legend(loc="upper center", ncol=2)
savefig(fig, "22_familiar_vs_new.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, pop in zip(axes, ["ALL", "SUB"]):
    xb = np.arange(len(labels))
    for off, m, c in zip([-0.27, 0, 0.27], METHODS, [GREY, BLUE, GREEN]):
        dd = bk[(bk.population == pop) & (bk.method == m)].set_index("activity_bucket").loc[labels]
        ax.bar(xb + off, dd.map5, width=0.27, color=c, yerr=[dd.map5 - dd.ci_lo, dd.ci_hi - dd.map5], capsize=2, label=m)
    nn = bk[(bk.population == pop) & (bk.method == "PPR")].set_index("activity_bucket").loc[labels].users
    ax.set_xticks(xb, [f"{b}\n(n={k})" for b, k in zip(labels, nn)])
    ax.set(title=f"MAP@5 by training activity, {pop} users", xlabel="Distinct qualifying training shows", ylabel="MAP@5")
    ax.legend()
savefig(fig, "23_map5_by_activity_three_methods.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
na = newdf[newdf.method.isin(["PPR_new_only", "pop_recent_new_only"])]
for ax, (metric, lo, hi, ttl) in zip(axes, [("map5_vs_truth_new", "ci_lo", "ci_hi", "MAP@5 vs truth_new"),
                                            ("recall5_vs_truth_new", "recall_ci_lo", "recall_ci_hi", "Recall@5 vs truth_new")]):
    for off, m, c in [(-0.2, "pop_recent_new_only", GREY), (0.2, "PPR_new_only", GREEN)]:
        dd = na[na.method == m].set_index("population").loc[["ALL", "SUB"]]
        ax.bar(np.arange(2) + off, dd[metric], width=0.4, color=c, yerr=[dd[metric] - dd[lo], dd[hi] - dd[metric]],
               capsize=4, label=m)
    nn = na[na.method == "PPR_new_only"].set_index("population").loc[["ALL", "SUB"]].users_with_truth_new
    ax.set_xticks(np.arange(2), [f"ALL (n={int(nn['ALL']):,})", f"SUB (n={int(nn['SUB']):,})"])
    ax.set(title=f"Discovery of unseen shows: {ttl}", ylabel=ttl.split(" vs")[0])
    ax.legend()
savefig(fig, "24_new_only_discovery.png")

fig, ax = plt.subplots(figsize=(8, 5))
for gg, lab, c in [(G, f"val graph (weeks 11-20, {len(G.shows):,} nodes)", BLUE),
                   (ST["g"], f"test graph (weeks 11-30, {len(ST['g'].shows):,} nodes)", ORANGE)]:
    dg = gg.deg[gg.deg > 0]
    ax.hist(dg, bins=np.logspace(0, np.log10(dg.max()), 40), histtype="step", lw=2, color=c, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set(title=f"Degree distribution of the co-view graph ({CFG['edge']}, seed {CFG['seed_weight']})",
       xlabel="Node degree (number of neighbour shows, log)", ylabel="Number of shows (log)")
ax.legend(); savefig(fig, "25_ppr_degree_distribution.png")
dd_ = pd.Series(G.deg).describe(percentiles=[.1, .5, .9]).round(1).to_dict()
summary["graph_val"]["degree_summary"] = dd_

# ====================================================================== cost + manifest
summary["runtime_seconds"] = round(time.perf_counter() - T0, 1)
summary["peak_memory_mb"] = peak_mb()
cost["peak_memory_mb_whole_step"] = summary["peak_memory_mb"]
cost.to_csv(OUT / "ppr_cost.csv", index=False)
log(cost.to_string(index=False))
HASHED = ["ppr_grid_val.csv", "ppr_vs_baselines_val.csv", "ppr_familiar_vs_new_val.csv", "ppr_new_only_val.csv",
          "ppr_breakdown_activity_val.csv", "ppr_robustness_windows_val.csv", "ppr_hit_rank_val.csv",
          "ppr_per_user_ap_val.csv", "ppr_coverage_diversity_val.csv", "ppr_category_mix_val.csv",
          "verify_ppr_naive.csv", "example_users_ppr.csv", "submissions/sub03_ppr.csv"]


def content_hash(f):
    if f.endswith(".csv") and not f.startswith("submissions"):
        df = pd.read_csv(OUT / f)
        df = df.drop(columns=[c for c in df.columns if "seconds" in c or c == "runtime"], errors="ignore")
        return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    return hashlib.sha256((OUT / f).read_bytes()).hexdigest()


summary["sha256_excluding_timing_columns"] = {f: content_hash(f) for f in HASHED}
(OUT / "step4_summary.json").write_text(json.dumps(summary, indent=2, default=str))
log(json.dumps({k: v for k, v in summary.items() if not k.startswith("sha256")}, indent=2, default=str))
