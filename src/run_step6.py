"""Step 6: hybrid = candidate generation + learned re-ranker, rank fusion, temporal backtest.
Run: python -m src.run_step6
"""
import dataclasses
import hashlib
import itertools
import json
import time
import warnings

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import io_utils as io
from src.evaluate import ap_at_k, bootstrap_ci, map_at_k, paired_bootstrap_diff
from src.hybrid_features import (CAT_EN, FEATURES, GROUPS, SRC, SRC_K, build, history_stats, in_candidates)
from src.labels import build_targets, n_distinct_shows
from src.make_submission import write_submission
from src.protocol import K, THR, W, labelled_airings, load_interactions, pad_with_popularity
from src.tasks import WK, make_task, task_summary

np.random.seed(42)
warnings.filterwarnings("ignore", message=".*eval_at.*")              # eval_at given via params on purpose
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")  # LightGBM 4.7 API notice
T0 = time.perf_counter()
OUT, FIG = io.ROOT / "outputs", io.ROOT / "outputs" / "figures"
DPI = 150
NTHREADS = 6
summary, COST = {}, []
proc = psutil.Process()
COL = {"pop_recent": "#9a9a9a", "PF_softfrac": "#3b6ea5", "PPR": "#4a9a5b", "cbf_pure": "#c0392b", "cbf_pop": "#e08a2c",
       "M0_PF_exposure": "#7d4fa1", "M1_fusion": "#17becf", "M2_logreg": "#8c564b", "M3_lambdarank": "#d62728",
       "M4_lgb_binary": "#bcbd22"}


def log(*a):
    print(*a, flush=True)


def peak_mb():
    mi = proc.memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 1024**2)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)


# ====================================================================== A tasks + B/C features
TASKS = {n: make_task(n) for n in ["inner", "val", "test"]}
pd.DataFrame([task_summary(t) for t in TASKS.values()]).to_csv(OUT / "tasks_summary.csv", index=False)
log(pd.read_csv(OUT / "tasks_summary.csv").to_string(index=False))
FULL, SRCS = {}, {}
for n, t in TASKS.items():
    log(f"[C] building candidates + features for {n} ...")
    F, src, tm = build(t)
    FULL[n], SRCS[n] = F, src
    COST.append({"task": n, "stage": "feature build (all sources + candidates + features)", "seconds": round(tm["total_seconds"], 1),
                 **{k: round(v, 1) for k, v in tm.items() if k != "total_seconds"}, "peak_memory_mb_so_far": peak_mb()})
    log("   ", COST[-1])
assert not [c for c in FEATURES if "truth" in c or "target_watch" in c]
FM = {n: F[in_candidates(F, SRC_K)].reset_index(drop=True) for n, F in FULL.items()}
summary["feature_hash_superset"] = {n: hashlib.sha256(pd.util.hash_pandas_object(F, index=False).values.tobytes()).hexdigest()
                                    for n, F in FULL.items()}


# ---------------------------------------------------------------------- helpers
def top5(Fm, score, users, pop, k=K):
    order = np.lexsort((Fm.rank_pop.to_numpy(), -np.asarray(score, float), Fm.user_id.to_numpy()))
    u = Fm.user_id.to_numpy()[order]
    s = Fm.tv_show_id.to_numpy()[order]
    first = np.r_[True, u[1:] != u[:-1]]
    pos = np.arange(len(u)) - np.maximum.accumulate(np.where(first, np.arange(len(u)), 0))
    lists = {}
    for uu, ss in zip(u[pos < k], s[pos < k]):
        lists.setdefault(int(uu), []).append(int(ss))
    return {int(x): pad_with_popularity(lists.get(int(x), []), pop, k) for x in users}


_PAD = {}


def fast_map(Fm, score):
    """Mean AP@5 for eval users (|truth| = 5, >= 5 candidates each); equals map_at_k of top5() (asserted).
    Rows are laid out in a users x max-candidates matrix and sorted per row (score desc, pop rank asc)."""
    # the cache holds a reference to Fm, so its id cannot be reused by another (temporary) frame
    if id(Fm) not in _PAD or _PAD[id(Fm)][0] is not Fm:
        uc = pd.factorize(Fm.user_id.to_numpy())[0]
        cnt = np.bincount(uc)
        pos = np.arange(len(uc)) - np.r_[0, np.cumsum(cnt)[:-1]][uc]
        R2 = np.full((len(cnt), cnt.max()), 1e9)
        L2 = np.zeros((len(cnt), cnt.max()))
        R2[uc, pos] = Fm.rank_pop.to_numpy()
        L2[uc, pos] = Fm.label.to_numpy()
        _PAD[id(Fm)] = (Fm, uc, pos, R2, L2)
    _, uc, pos, R2, L2 = _PAD[id(Fm)]
    S2 = np.full(R2.shape, -np.inf)
    S2[uc, pos] = np.asarray(score, float)
    o = np.lexsort((R2, -S2), axis=-1)[:, :K]
    h = np.take_along_axis(L2, o, axis=1)
    return float(((h * h.cumsum(1)) / np.arange(1, K + 1)).sum(1).mean() / K)


def evaluate(recs, t, truth=None):
    truth = t.truth if truth is None else truth
    _, ap = map_at_k(recs, truth)
    sub = set(int(u) for u in t.users[t.sub_mask])
    res = {}
    for p, a in (("ALL", ap), ("SUB", ap[ap.index.isin(sub)])):
        lo, hi = bootstrap_ci(a)
        res[p] = {"map5": float(a.mean()), "ci_lo": lo, "ci_hi": hi, "n": len(a), "ap": a}
    return res


def src_top5(src, t, kind):
    if kind in ("pf", "pfx"):
        lst = src[f"{kind}_list"].query("rank <= 5")
        lists = {int(u): g.tv_show_id.tolist() for u, g in lst.groupby("user_id")}
    else:
        Rm = src[f"{kind}_R"]
        cols = t.pool if kind == "ppr" else t.pop
        lists = {}
        for i, u in enumerate(t.users):
            r = Rm[i]
            idx = np.flatnonzero((r > 0) & (r <= 5))
            lists[int(u)] = [int(cols[j]) for j in idx[np.argsort(r[idx])]]
    return {int(u): pad_with_popularity(lists.get(int(u), []), t.pop) for u in t.users}


# ---------------------------------------------------------------------- reproduce Steps 3-5 (val)
tv, ti, tt = TASKS["val"], TASKS["inner"], TASKS["test"]
BASE = {"pop_recent": {int(u): [int(x) for x in tv.pop[:5]] for u in tv.users},
        "PF_softfrac": src_top5(SRCS["val"], tv, "pf"), "M0_PF_exposure": src_top5(SRCS["val"], tv, "pfx"),
        "PPR": src_top5(SRCS["val"], tv, "ppr"), "cbf_pop": src_top5(SRCS["val"], tv, "cbfpop"),
        "cbf_pure": src_top5(SRCS["val"], tv, "cbf")}
REF = {"pop_recent": (0.23980727874276261, 0.2655594202898551), "PF_softfrac": (0.32366335814722915, 0.3599913043478261),
       "M0_PF_exposure": (0.34063192721257235, 0.3802608695652174), "PPR": (0.325645988420182, 0.36395652173913046),
       "cbf_pop": (0.2783217535153019, 0.31113043478260877), "cbf_pure": (0.18152522746071134, 0.19921739130434785)}
E = {}
for m, r in BASE.items():
    E[m] = evaluate(r, tv)
    assert abs(E[m]["ALL"]["map5"] - REF[m][0]) < 1e-12 and abs(E[m]["SUB"]["map5"] - REF[m][1]) < 1e-12, m
log("[A] Steps 3-5 reproduced exactly:", {m: round(E[m]["ALL"]["map5"], 4) for m in E})
b3 = pd.read_csv(OUT / "baseline_results_val.csv")
summary["baseline_results_val_sha256"] = hashlib.sha256((OUT / "baseline_results_val.csv").read_bytes()).hexdigest()

# ====================================================================== B candidate statistics
crow, only_rows = [], []
for n in ["inner", "val", "test"]:
    t, Fm = TASKS[n], FM[n]
    cnt = Fm.groupby("user_id").size()
    r = {"task": n, "users": len(t.users), "rows": len(Fm), "mean_candidates": cnt.mean(), "median_candidates": cnt.median(),
         "max_candidates": cnt.max(), "min_candidates": cnt.min()}
    if t.truth:
        r["candidate_recall_truth_pairs"] = Fm.label.sum() / (5 * len(t.truth))
        r["history_ceiling_any_watch"] = task_summary(t)["share_truth_pairs_in_history"]
        tp = Fm[Fm.label == 1]
        member = np.column_stack([(tp[f"rank_{s}"] <= SRC_K[s]).to_numpy() for s in SRC])
        for j, s in enumerate(SRC):
            only = member[:, j] & (member.sum(1) == 1)
            only_rows.append({"task": n, "source": s, "k": SRC_K[s],
                              "truth_pairs_in_source": int(member[:, j].sum()),
                              "truth_pairs_found_only_by_source": int(only.sum()),
                              "share_of_truth_pairs_only_by_source": only.sum() / (5 * len(t.truth))})
    crow.append(r)
cand = pd.DataFrame(crow)
cand.to_csv(OUT / "hybrid_candidates_summary.csv", index=False)
pd.DataFrame(only_rows).to_csv(OUT / "hybrid_candidates_by_source.csv", index=False)
log(cand.round(4).to_string(index=False))
log(pd.DataFrame(only_rows).round(4).to_string(index=False))

# ====================================================================== D verification
# D1 naive features
rng = np.random.default_rng(42)
pick = rng.choice(tv.users, 6, replace=False)
assert pick.dtype == np.uint64 and all(int(u) in tv.truth for u in pick)
inter_all = load_interactions()
lab = tv.lab
NAIVE_COLS = ["raw_softfrac", "raw_q", "raw_c05", "raw_secs", "raw_weeks_q", "raw_weeks_any", "raw_a_train", "raw_a_target",
              "hs_softfrac_pw", "hs_q_pw", "hs_c05_pw", "hs_hours_pw", "hs_share_softfrac", "hs_share_q",
              "hs_rank_inv_softfrac", "hs_mean_frac", "hs_n_channels", "rh_weeks_q_share", "rh_weeks_any_share",
              "rh_rec_q", "rh_rec_any", "rh_first_seen", "ex_a_target_us_pw", "ex_a_train_us_pw", "ex_a_target_glob_pw",
              "ex_a_train_glob_pw", "raw_ex_ratio_us", "ex_rate_ratio_us", "ex_zero_us", "ex_other_channels_only",
              "ex_glob_rate_ratio"]
hw, tw = tv.hist_weeks, tv.tgt_weeks
he, hs0, ts0, te0 = tv.hist_end.to_pydatetime(), tv.hist_start.to_pydatetime(), tv.tgt_start.to_pydatetime(), tv.tgt_end.to_pydatetime()
nrows, hand_case = [], None
Fv = FM["val"]
for u in pick:
    u = int(u)
    fu = Fv[Fv.user_id == np.uint64(u)]
    zero_hist = fu[(fu.ex_zero_us == 1) & (fu.new_show == 0)]
    s_zero = (zero_hist if len(zero_hist) else fu[fu.ex_zero_us == 1]).iloc[0]
    s_new = fu[(fu.new_show == 1) & (fu.tv_show_id != s_zero.tv_show_id)].iloc[0]
    sel = [s_zero, s_new]
    rest = fu[~fu.tv_show_id.isin([s.tv_show_id for s in sel])]
    sel += [rest.iloc[i] for i in sorted(np.random.default_rng(42).choice(len(rest), 3, replace=False))]
    rows = [r for r in inter_all[inter_all.user_id == np.uint64(u)].itertuples() if hs0 <= r.airing_start.to_pydatetime() < he]
    per = {}
    for r in rows:                         # naive per-show accumulation over raw interaction rows
        d = per.setdefault(r.tv_show_id, {"soft": 0.0, "q": 0, "c05": 0, "secs": 0, "n": 0, "wq": set(), "wa": set(),
                                          "last_q": None, "last": None, "first": None, "ch": set()})
        d["soft"] += r.frac
        d["q"] += r.frac >= THR
        d["c05"] += r.frac >= 0.5
        d["secs"] += r.watched_seconds
        d["n"] += 1
        d["wa"].add(r.iso_week)
        d["ch"].add(r.channel_id)
        a = r.airing_start.to_pydatetime()
        d["last"] = a if d["last"] is None or a > d["last"] else d["last"]
        d["first"] = a if d["first"] is None or a < d["first"] else d["first"]
        if r.frac >= THR:
            d["wq"].add(r.iso_week)
            d["last_q"] = a if d["last_q"] is None or a > d["last_q"] else d["last_q"]
    tot_soft = sum(d["soft"] for d in per.values())
    tot_q = sum(d["q"] for d in per.values())
    ranked = sorted(per.items(), key=lambda kv: -kv[1]["soft"])
    for s_row in sel:
        s = int(s_row.tv_show_id)
        d = per.get(s)
        chans = d["ch"] if d else set()
        a_tr = a_tg = g_tr = g_tg = 0
        for r in lab[lab.tv_show_id == s].itertuples():          # naive loop over schedule rows of the show
            st = r.start_time.to_pydatetime()
            if hs0 <= st < he:
                g_tr += 1
                a_tr += r.channel_id in chans
            if ts0 <= st < te0:
                g_tg += 1
                a_tg += r.channel_id in chans
        soft = d["soft"] if d else 0.0
        rank_s = 1 + sum(1 for _, v in ranked if v["soft"] > soft) if d else None
        nv = {"raw_softfrac": soft, "raw_q": d["q"] if d else 0, "raw_c05": d["c05"] if d else 0,
              "raw_secs": d["secs"] if d else 0, "raw_weeks_q": len(d["wq"]) if d else 0,
              "raw_weeks_any": len(d["wa"]) if d else 0, "raw_a_train": a_tr, "raw_a_target": a_tg,
              "hs_softfrac_pw": soft / hw, "hs_q_pw": (d["q"] if d else 0) / hw, "hs_c05_pw": (d["c05"] if d else 0) / hw,
              "hs_hours_pw": (d["secs"] if d else 0) / 3600 / hw, "hs_share_softfrac": soft / tot_soft,
              "hs_share_q": ((d["q"] if d else 0) / tot_q) if tot_q else 0.0,
              "hs_rank_inv_softfrac": 1 / rank_s if d else 0.0, "hs_mean_frac": soft / d["n"] if d else 0.0,
              "hs_n_channels": len(chans), "rh_weeks_q_share": (len(d["wq"]) if d else 0) / hw,
              "rh_weeks_any_share": (len(d["wa"]) if d else 0) / hw,
              "rh_rec_q": ((he - d["last_q"]).total_seconds() / 604800 / hw) if d and d["last_q"] else 1.0,
              "rh_rec_any": ((he - d["last"]).total_seconds() / 604800 / hw) if d else 1.0,
              "rh_first_seen": ((d["first"] - hs0).total_seconds() / 604800 / hw) if d else 1.0,
              "ex_a_target_us_pw": a_tg / tw, "ex_a_train_us_pw": a_tr / hw, "ex_a_target_glob_pw": g_tg / tw,
              "ex_a_train_glob_pw": g_tr / hw, "raw_ex_ratio_us": a_tg / max(a_tr, 1),
              "ex_rate_ratio_us": (a_tg / tw) / max(a_tr / hw, 1 / hw), "ex_zero_us": float(a_tg == 0),
              "ex_other_channels_only": float(a_tg == 0 and g_tg > 0),
              "ex_glob_rate_ratio": (g_tg / tw) / max(g_tr / hw, 1 / hw)}
        diff = max(abs(float(nv[c]) - float(s_row[c])) for c in NAIVE_COLS)
        nrows.append({"user_id": str(u), "tv_show_id": s, "new_show": bool(s_row.new_show), "ex_zero_us": bool(s_row.ex_zero_us),
                      "features_compared": len(NAIVE_COLS), "max_abs_diff": diff})
        if (hand_case is None or hand_case[3]["raw_q"] < 2) and d and d["q"] >= 1 and a_tr > 0 and a_tg > 0:
            hand_case = (u, s, [r for r in rows if r.tv_show_id == s], nv, chans, a_tr, a_tg, g_tr, g_tg)
nvdf = pd.DataFrame(nrows)
nvdf.to_csv(OUT / "verify_features_naive.csv", index=False)
log(nvdf.to_string(index=False))
assert (nvdf.max_abs_diff < 1e-9).all()
summary["naive_features"] = {"pairs": len(nvdf), "features_each": len(NAIVE_COLS), "max_abs_diff": float(nvdf.max_abs_diff.max())}

u, s, rws, nv, chans, a_tr, a_tg, g_tr, g_tg = hand_case
L = [f"Hand-worked features: val task, user {u}, show {s} ({tv.meta.title.get(s)}); history {tv.hist_start.date()} .. "
     f"{tv.hist_end.date()} (exclusive, {hw:g} weeks), target {tv.tgt_start.date()} .. {tv.tgt_end.date()} ({tw:g} weeks)",
     "raw interaction rows (airing_start, channel, airing_duration, watched_seconds, frac, iso_week):"]
for r in rws:
    L.append(f"  {r.airing_start}  ch {r.channel_id}  dur {r.airing_duration}  watched {r.watched_seconds}  frac {r.frac:.6f}"
             f"  week {r.iso_week}{'  (qualifying)' if r.frac >= THR else ''}")
L.append(f"softfrac = " + " + ".join(f"{r.frac:.6f}" for r in rws) + f" = {nv['raw_softfrac']:.6f}")
L.append(f"qualifying count = #rows with frac >= 0.8 = {nv['raw_q']}")
L.append(f"weeks_active (distinct ISO weeks with a qualifying watch) = {sorted({r.iso_week for r in rws if r.frac >= THR})}"
         f" -> {nv['raw_weeks_q']}; share = {nv['raw_weeks_q']} / {hw:g} = {nv['rh_weeks_q_share']:.6f}")
L.append(f"user's channels for this show C_us = {sorted(chans)}")
L.append(f"A_train_us = labelled airings of the show starting in the history window on C_us = {a_tr} (all channels: {g_tr})")
L.append(f"A_target_us = labelled airings in the target window on C_us = {a_tg} (all channels: {g_tg}); schedule only")
L.append(f"exposure ratio = A_target_us / max(A_train_us, 1) = {a_tg} / {max(a_tr, 1)} = {nv['raw_ex_ratio_us']:.6f}; "
         f"rate ratio = ({a_tg}/{tw:g}) / max({a_tr}/{hw:g}, 1/{hw:g}) = {nv['ex_rate_ratio_us']:.6f}")
vrow = Fv[(Fv.user_id == np.uint64(u)) & (Fv.tv_show_id == s)].iloc[0]
L.append("vectorised: " + ", ".join(f"{c}={vrow[c]:.6f}" for c in ["raw_softfrac", "raw_q", "raw_weeks_q", "raw_a_train",
                                                                     "raw_a_target", "raw_ex_ratio_us"]))
(OUT / "verify_features_hand.txt").write_text("\n".join(L), encoding="utf-8")
log("\n".join(L))

# D2c history shift sanity
tshift = dataclasses.replace(tv, train=tv.train[tv.train.airing_start < WK[20]].reset_index(drop=True), hist_end=WK[20])
h_a, h_b = history_stats(tv), history_stats(tshift)
mm = h_a.merge(h_b, on=["user_id", "tv_show_id"], how="left", suffixes=("", "_s"))
changed = float((mm.softfrac != mm.softfrac_s.fillna(0)).mean())
summary["history_shift_test"] = {"pairs": len(mm), "share_pairs_softfrac_changed": changed,
                                 "pairs_disappearing": int(mm.softfrac_s.isna().sum())}
log("[D2c]", summary["history_shift_test"])
assert changed > 0

# D3 determinism (inner rebuilt)
F2, _, _ = build(ti)
summary["feature_determinism_inner"] = bool(pd.util.hash_pandas_object(F2, index=False).equals(
    pd.util.hash_pandas_object(FULL["inner"], index=False)))
log("[D3] inner features rebuilt identical:", summary["feature_determinism_inner"])
assert summary["feature_determinism_inner"]
del F2

# ====================================================================== E models
Fi, Fv, Ft = FM["inner"], FM["val"], FM["test"]
assert Fi.groupby("user_id").size().min() >= K and Fv.groupby("user_id").size().min() >= K
for s_chk in [Fv.hs_softfrac_pw.to_numpy(), np.random.default_rng(1).random(len(Fv)), np.zeros(len(Fv))]:
    assert abs(fast_map(Fv, s_chk) - map_at_k(top5(Fv, s_chk, tv.users, tv.pop), tv.truth)[0]) < 1e-12


def comp_scores(F):
    expo = np.where(F.raw_a_target > 0, F.raw_a_target, 0.1 * F.raw_a_target_glob)
    return {"pf": F.raw_softfrac.to_numpy(), "pfx": (F.raw_softfrac * (1 + expo) ** 0.25).to_numpy(),
            "ppr": F.ppr_score_rel.to_numpy(), "cbfpop": F.cbfpop_score.to_numpy(), "pop": F.pop_norm.to_numpy()}


M1C = ["pf", "pfx", "ppr", "cbfpop", "pop"]


def zfuse_parts(F):
    sc = comp_scores(F)
    out = {}
    for c in M1C:
        x = pd.Series(sc[c])
        g = x.groupby(F.user_id.to_numpy())
        sd = g.transform("std").fillna(0).to_numpy()
        out[c] = np.where(sd > 0, (x.to_numpy() - g.transform("mean").to_numpy()) / np.where(sd > 0, sd, 1), 0.0)
    return out


def rrf(F, w, k0):
    s = np.zeros(len(F))
    for c, wc in zip(M1C, w):
        if wc:
            r = F[f"rank_{c}"].to_numpy()
            s += np.where(np.isnan(r), 0.0, wc / (k0 + np.nan_to_num(r, nan=1e9)))
    return s


log("[E/M1] fusion grids on inner (w in {0, 0.5, 1, 2}^5, k0 in {10, 30, 60})")
t0 = time.perf_counter()
grid_rows = []
WS = [w for w in itertools.product([0, 0.5, 1, 2], repeat=5) if any(w)]
Zi = zfuse_parts(Fi)
for w in WS:
    for k0 in [10, 30, 60]:
        grid_rows.append({"fusion": "rrf", "k0": k0, **{f"w_{c}": x for c, x in zip(M1C, w)}, "inner_map5": fast_map(Fi, rrf(Fi, w, k0))})
    zs = sum(x * Zi[c] for c, x in zip(M1C, w))
    grid_rows.append({"fusion": "zscore", "k0": np.nan, **{f"w_{c}": x for c, x in zip(M1C, w)}, "inner_map5": fast_map(Fi, zs)})
m1g = pd.DataFrame(grid_rows)
m1g.to_csv(OUT / "hybrid_m1_fusion_grid_inner.csv", index=False)
best_rrf = m1g[m1g.fusion == "rrf"].sort_values(["inner_map5"], ascending=False, kind="stable").iloc[0]
best_z = m1g[m1g.fusion == "zscore"].sort_values(["inner_map5"], ascending=False, kind="stable").iloc[0]
COST.append({"task": "inner", "stage": f"M1 fusion grid ({len(m1g)} configs)", "seconds": round(time.perf_counter() - t0, 1)})
log("  best rrf:", best_rrf.to_dict())
log("  best zscore:", best_z.to_dict())


def m1_score(F, row):
    w = [row[f"w_{c}"] for c in M1C]
    if row.fusion == "rrf":
        return rrf(F, w, row.k0)
    z = zfuse_parts(F)
    return sum(x * z[c] for c, x in zip(M1C, w))


M1_ROW = best_rrf if best_rrf.inner_map5 >= best_z.inner_map5 else best_z
summary["M1"] = {"rrf": best_rrf.to_dict(), "zscore": best_z.to_dict(), "chosen_on_inner": M1_ROW.fusion}


# ---------------------------------------------------------------------- learned models
def user_folds(F, k, seed=42):
    us = np.sort(F.user_id.unique())
    perm = np.random.default_rng(seed).permutation(len(us))
    fold = pd.Series(perm % k, index=us)
    return fold.reindex(F.user_id.to_numpy()).to_numpy()


def lgb_params(kind, p):
    base = {"learning_rate": p["lr"], "num_leaves": p["leaves"], "min_child_samples": p["mcs"], "random_state": 42,
            "deterministic": True, "force_row_wise": True, "n_jobs": NTHREADS, "verbose": -1}
    if kind == "rank":
        base.update({"objective": "lambdarank", "metric": "ndcg", "eval_at": [5]})
    else:
        base.update({"objective": "binary", "metric": "binary_logloss"})
    return base


def fit_lgb(kind, p, F, feats, n_estimators=None):
    """Early stopping on a user-grouped 20% hold-out of F, then refit on all of F with the best iteration."""
    y = F.graded.to_numpy() if kind == "rank" else F.label.to_numpy()
    X = F[feats].to_numpy(np.float32)
    if n_estimators is None:
        hold = user_folds(F, 5, seed=7) == 0
        m = (lgb.LGBMRanker if kind == "rank" else lgb.LGBMClassifier)(n_estimators=3000, **lgb_params(kind, p))
        fit_kw = {"eval_set": [(X[hold], y[hold])], "callbacks": [lgb.early_stopping(50, verbose=False)]}
        if kind == "rank":
            fit_kw.update(group=F[~hold].groupby("user_id", sort=False).size().to_numpy(),
                          eval_group=[F[hold].groupby("user_id", sort=False).size().to_numpy()])
        m.fit(X[~hold], y[~hold], **fit_kw)
        n_estimators = max(int(m.best_iteration_ or 1), 1)
    m = (lgb.LGBMRanker if kind == "rank" else lgb.LGBMClassifier)(n_estimators=n_estimators, **lgb_params(kind, p))
    kw = {"group": F.groupby("user_id", sort=False).size().to_numpy()} if kind == "rank" else {}
    m.fit(X, y, **kw)
    return m, n_estimators


def lgb_predict(m, F, feats, kind):
    X = F[feats].to_numpy(np.float32)
    return m.predict(X) if kind == "rank" else m.predict_proba(X)[:, 1]


def fit_lr(F, feats, C):
    sc = StandardScaler().fit(F[feats].to_numpy())
    m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(F[feats].to_numpy()), F.label.to_numpy())
    return (sc, m)


def lr_predict(mm_, F, feats):
    sc, m = mm_
    return m.decision_function(sc.transform(F[feats].to_numpy()))


LGB_GRID = [{"leaves": a, "lr": b, "mcs": c} for a in [7, 15, 31] for b in [0.03, 0.1] for c in [20, 100]]
fold_i = user_folds(Fi, 3)
FOLDS = [(Fi[fold_i != f].reset_index(drop=True), Fi[fold_i == f].reset_index(drop=True)) for f in range(3)]
cv_rows = []
log("[E] model selection by 3-fold user-grouped CV on the inner task (never on val)")
t0 = time.perf_counter()
for C in [0.1, 1.0]:
    sc_ = []
    for f_tr, f_te in FOLDS:
        mdl = fit_lr(f_tr, FEATURES, C)
        sc_.append(fast_map(f_te, lr_predict(mdl, f_te, FEATURES)))
    cv_rows.append({"model": "M2_logreg", "C": C, "cv_map5_inner": float(np.mean(sc_))})
    log("  ", cv_rows[-1])
COST.append({"task": "inner", "stage": "M2 CV (2 x 3 fits)", "seconds": round(time.perf_counter() - t0, 1)})
for kind, name in [("rank", "M3_lambdarank"), ("binary", "M4_lgb_binary")]:
    t0 = time.perf_counter()
    for p in LGB_GRID:
        sc_, its = [], []
        for f_tr, f_te in FOLDS:
            mdl, n_it = fit_lgb(kind, p, f_tr, FEATURES)
            sc_.append(fast_map(f_te, lgb_predict(mdl, f_te, FEATURES, kind)))
            its.append(n_it)
        cv_rows.append({"model": name, **p, "cv_map5_inner": float(np.mean(sc_)), "mean_best_iteration": float(np.mean(its))})
        log("  ", cv_rows[-1])
    COST.append({"task": "inner", "stage": f"{name} CV ({len(LGB_GRID)} configs x 3 folds)", "seconds": round(time.perf_counter() - t0, 1)})
cvdf = pd.DataFrame(cv_rows)
cvdf.to_csv(OUT / "hybrid_model_selection_inner_cv.csv", index=False)
BEST = {}
BEST["M2_logreg"] = cvdf[cvdf.model == "M2_logreg"].sort_values("cv_map5_inner", ascending=False, kind="stable").iloc[0]
for name in ["M3_lambdarank", "M4_lgb_binary"]:
    b = cvdf[cvdf.model == name].sort_values("cv_map5_inner", ascending=False, kind="stable").iloc[0]
    BEST[name] = {"leaves": int(b.leaves), "lr": float(b.lr), "mcs": int(b.mcs)}
summary["selected_hyperparameters"] = {"M2_logreg": {"C": float(BEST["M2_logreg"].C)}, "M3_lambdarank": BEST["M3_lambdarank"],
                                       "M4_lgb_binary": BEST["M4_lgb_binary"]}
log("[E] selected hyper-parameters:", summary["selected_hyperparameters"])

# ---------------------------------------------------------------------- P1: train inner -> val
MODELS, NEST, SCORES_V = {}, {}, {}
t0 = time.perf_counter()
MODELS["M2_logreg"] = fit_lr(Fi, FEATURES, float(BEST["M2_logreg"].C))
COST.append({"task": "inner", "stage": "M2 final fit", "seconds": round(time.perf_counter() - t0, 1)})
t0 = time.perf_counter()
SCORES_V["M2_logreg"] = lr_predict(MODELS["M2_logreg"], Fv, FEATURES)
COST.append({"task": "val", "stage": "M2 inference", "seconds": round(time.perf_counter() - t0, 2)})
for kind, name in [("rank", "M3_lambdarank"), ("binary", "M4_lgb_binary")]:
    t0 = time.perf_counter()
    MODELS[name], NEST[name] = fit_lgb(kind, BEST[name], Fi, FEATURES)
    COST.append({"task": "inner", "stage": f"{name} final fit (early stopping + refit)", "seconds": round(time.perf_counter() - t0, 1)})
    t0 = time.perf_counter()
    SCORES_V[name] = lgb_predict(MODELS[name], Fv, FEATURES, kind)
    COST.append({"task": "val", "stage": f"{name} inference", "seconds": round(time.perf_counter() - t0, 2)})
SCORES_V["M1_fusion"] = m1_score(Fv, M1_ROW)
SCORES_V["M1_rrf"] = m1_score(Fv, best_rrf)
SCORES_V["M1_zscore"] = m1_score(Fv, best_z)
summary["n_estimators"] = NEST
RECS = dict(BASE)
for m, sc in SCORES_V.items():
    RECS[m] = top5(Fv, sc, tv.users, tv.pop)
    E[m] = evaluate(RECS[m], tv)
HYBRIDS = ["M1_fusion", "M2_logreg", "M3_lambdarank", "M4_lgb_binary"]
p1 = []
for m in ["pop_recent", "PF_softfrac", "PPR", "cbf_pure", "cbf_pop", "M0_PF_exposure", "M1_rrf", "M1_zscore"] + HYBRIDS:
    r = {"method": m}
    for p in ["ALL", "SUB"]:
        r.update({f"map5_{p}": E[m][p]["map5"], f"ci_lo_{p}": E[m][p]["ci_lo"], f"ci_hi_{p}": E[m][p]["ci_hi"]})
    r["avg_all_sub"] = (r["map5_ALL"] + r["map5_SUB"]) / 2
    p1.append(r)
p1 = pd.DataFrame(p1)
HYB = p1[p1.method.isin(HYBRIDS)].sort_values("avg_all_sub", ascending=False, kind="stable").method.iloc[0]
dl = []
for m in HYBRIDS:
    for base in ["M0_PF_exposure", "PF_softfrac", "PPR", "cbf_pop", "cbf_pure", "pop_recent"]:
        for p in ["ALL", "SUB"]:
            d, lo, hi = paired_bootstrap_diff(E[m][p]["ap"], E[base][p]["ap"])
            dl.append({"model": m, "vs": base, "population": p, "mean_diff": d, "ci_lo": lo, "ci_hi": hi})
dldf = pd.DataFrame(dl)
p1.to_csv(OUT / "hybrid_p1_results_val.csv", index=False)
dldf.to_csv(OUT / "hybrid_p1_paired_diffs_val.csv", index=False)
hd = dldf[(dldf.model == HYB) & (dldf.vs == "M0_PF_exposure") & (dldf.population == "ALL")].iloc[0]
FINAL = HYB if hd.ci_lo > 0 else "M0_PF_exposure"
summary["P1"] = {"best_hybrid": HYB, "hybrid_vs_M0_ALL": [hd.mean_diff, hd.ci_lo, hd.ci_hi], "final_strategy": FINAL}
log(p1.round(4).to_string(index=False))
log(dldf[dldf.vs.isin(["M0_PF_exposure", "PF_softfrac"])].round(4).to_string(index=False))
log("[E] best hybrid under P1:", HYB, "| final strategy by the rule:", FINAL)

# ---------------------------------------------------------------------- P2: 5-fold user-grouped CV on val
fold_v = user_folds(Fv, 5)
p2 = []
t0 = time.perf_counter()
for name in ["M2_logreg", "M3_lambdarank", "M4_lgb_binary"]:
    oof = np.zeros(len(Fv))
    for f in range(5):
        tr_, te_ = Fv[fold_v != f].reset_index(drop=True), fold_v == f
        if name == "M2_logreg":
            mdl = fit_lr(tr_, FEATURES, float(BEST[name].C))
            oof[te_] = lr_predict(mdl, Fv[te_], FEATURES)
        else:
            kind = "rank" if name == "M3_lambdarank" else "binary"
            mdl, _ = fit_lgb(kind, BEST[name], tr_, FEATURES, n_estimators=NEST[name])
            oof[te_] = lgb_predict(mdl, Fv[te_], FEATURES, kind)
    e = evaluate(top5(Fv, oof, tv.users, tv.pop), tv)
    p2.append({"model": name, **{f"{k}_{p}": e[p][k] for p in ["ALL", "SUB"] for k in ["map5", "ci_lo", "ci_hi"]}})
    log("  [P2]", p2[-1])
COST.append({"task": "val", "stage": "P2 5-fold CV (3 models)", "seconds": round(time.perf_counter() - t0, 1)})
pd.DataFrame(p2).to_csv(OUT / "hybrid_p2_cv_val.csv", index=False)

# ---------------------------------------------------------------------- D2a permutation / random tests
rngp = np.random.default_rng(42)
lgb_kind = {"M3_lambdarank": "rank", "M4_lgb_binary": "binary"}
EXPL = HYB if HYB in lgb_kind else "M3_lambdarank"           # LightGBM model used for SHAP explanations
Fi_perm = Fi.copy()
perm = rngp.permutation(len(Fi_perm))
Fi_perm["graded"] = Fi_perm.graded.to_numpy()[perm]
Fi_perm["label"] = Fi_perm.label.to_numpy()[perm]
mp, _ = fit_lgb("rank", BEST["M3_lambdarank"], Fi_perm, FEATURES, n_estimators=NEST["M3_lambdarank"])
perm_map = map_at_k(top5(Fv, lgb_predict(mp, Fv, FEATURES, "rank"), tv.users, tv.pop), tv.truth)[0]
rand_map = map_at_k(top5(Fv, rngp.random(len(Fv)), tv.users, tv.pop), tv.truth)[0]
uu = list(tv.truth)
pidx = rngp.permutation(len(uu))          # permute positions: permuting the big-int ids would turn them into float64
shuf = {uu[i]: tv.truth[uu[j]] for i, j in enumerate(pidx)}
shuf_map = map_at_k(RECS["M3_lambdarank"], shuf)[0]
summary["leakage_tests"] = {"M3_trained_on_permuted_inner_labels_val_map": perm_map,
                            "random_reranking_inside_candidates_val_map": rand_map,
                            "real_M3_vs_truth_shuffled_across_val_users_map": shuf_map,
                            "real_M3_val_map": E["M3_lambdarank"]["ALL"]["map5"]}
log("[D2a]", summary["leakage_tests"])

# ====================================================================== F1 ablation + importance
t0 = time.perf_counter()
ab = []
p3 = BEST["M3_lambdarank"]
for g, cols in GROUPS.items():
    feats = [f for f in FEATURES if f not in cols]
    mdl, n_ = fit_lgb("rank", p3, Fi, feats)
    e = evaluate(top5(Fv, lgb_predict(mdl, Fv, feats, "rank"), tv.users, tv.pop), tv)
    ab.append({"experiment": f"drop {g}", "group": g, "n_features": len(feats), "map5_ALL": e["ALL"]["map5"],
               "map5_SUB": e["SUB"]["map5"], "delta_ALL": e["ALL"]["map5"] - E["M3_lambdarank"]["ALL"]["map5"],
               "delta_SUB": e["SUB"]["map5"] - E["M3_lambdarank"]["SUB"]["map5"]})
    log("  [F1]", ab[-1])
base_feats = GROUPS["history_strength"]
mdl, _ = fit_lgb("rank", p3, Fi, base_feats)
e0 = evaluate(top5(Fv, lgb_predict(mdl, Fv, base_feats, "rank"), tv.users, tv.pop), tv)
ab.append({"experiment": "history_strength only", "group": "history_strength", "n_features": len(base_feats),
           "map5_ALL": e0["ALL"]["map5"], "map5_SUB": e0["SUB"]["map5"], "delta_ALL": 0.0, "delta_SUB": 0.0})
for g, cols in GROUPS.items():
    if g == "history_strength":
        continue
    feats = base_feats + cols
    mdl, _ = fit_lgb("rank", p3, Fi, feats)
    e = evaluate(top5(Fv, lgb_predict(mdl, Fv, feats, "rank"), tv.users, tv.pop), tv)
    ab.append({"experiment": f"history_strength + {g}", "group": g, "n_features": len(feats), "map5_ALL": e["ALL"]["map5"],
               "map5_SUB": e["SUB"]["map5"], "delta_ALL": e["ALL"]["map5"] - e0["ALL"]["map5"],
               "delta_SUB": e["SUB"]["map5"] - e0["SUB"]["map5"]})
    log("  [F1]", ab[-1])
abdf = pd.DataFrame(ab)
abdf.to_csv(OUT / "hybrid_ablation_val.csv", index=False)
COST.append({"task": "inner->val", "stage": "ablation (16 M3 fits)", "seconds": round(time.perf_counter() - t0, 1)})
m3 = MODELS["M3_lambdarank"]
contrib = m3.predict(Fv[FEATURES].to_numpy(np.float32), pred_contrib=True)
imp = pd.DataFrame({"feature": FEATURES, "gain": m3.booster_.feature_importance("gain"),
                    "mean_abs_shap_val": np.abs(contrib[:, :-1]).mean(0)})
imp["group"] = imp.feature.map({f: g for g, fs in GROUPS.items() for f in fs})
imp = imp.sort_values("mean_abs_shap_val", ascending=False, ignore_index=True)
imp.to_csv(OUT / "hybrid_feature_importance_M3.csv", index=False)
log(imp.head(15).round(4).to_string(index=False))

# ====================================================================== F2 candidate-size sensitivity
sens = []
t0 = time.perf_counter()
for kk in [10, 20, 30, 50]:
    ks = {s: kk for s in SRC}
    Fi_k = FULL["inner"][in_candidates(FULL["inner"], ks)].reset_index(drop=True)
    Fv_k = FULL["val"][in_candidates(FULL["val"], ks)].reset_index(drop=True)
    mdl, _ = fit_lgb("rank", p3, Fi_k, FEATURES)
    e = evaluate(top5(Fv_k, lgb_predict(mdl, Fv_k, FEATURES, "rank"), tv.users, tv.pop), tv)
    sens.append({"k_per_source": kk, "mean_candidates_val": len(Fv_k) / len(tv.users),
                 "candidate_recall_val": Fv_k.label.sum() / (5 * len(tv.truth)), "map5_ALL": e["ALL"]["map5"],
                 "map5_SUB": e["SUB"]["map5"]})
    log("  [F2]", sens[-1])
sens.append({"k_per_source": "main (30/30/20/30/30/20)", "mean_candidates_val": len(Fv) / len(tv.users),
             "candidate_recall_val": Fv.label.sum() / (5 * len(tv.truth)), "map5_ALL": E["M3_lambdarank"]["ALL"]["map5"],
             "map5_SUB": E["M3_lambdarank"]["SUB"]["map5"]})
sensdf = pd.DataFrame(sens)
sensdf.to_csv(OUT / "hybrid_candidate_size_sensitivity.csv", index=False)
COST.append({"task": "inner->val", "stage": "candidate-size sensitivity (4 M3 fits)", "seconds": round(time.perf_counter() - t0, 1)})

# ====================================================================== F3 breakdowns
SHOW = ["M0_PF_exposure", "PF_softfrac", "PPR", "cbf_pop", "pop_recent", HYB]
SHOW = list(dict.fromkeys(SHOW))
hist = {int(u): set(s) for u, s in tv.train.groupby("user_id").tv_show_id.agg(set).items()}
nq = tv.train[tv.train.frac >= THR].groupby("user_id").tv_show_id.nunique()
act = pd.Series({u: int(nq.get(np.uint64(u), 0)) for u in tv.truth})
labels = ["0-4", "5-9", "10-19", "20-49", "50+"]
bucket = pd.cut(act, [-1, 4, 9, 19, 49, np.inf], labels=labels)
cat = tv.meta.category
topcat = pd.Series({u: cat.get(t[0]) for u, t in tv.truth.items()})
rows = []
for m in SHOW:
    for p in ["ALL", "SUB"]:
        a = E[m][p]["ap"]
        for dim, key in [("activity_bucket", bucket), ("category_of_top_truth_show", topcat)]:
            for val_, grp in a.groupby(key.reindex(a.index).to_numpy()):
                lo, hi = bootstrap_ci(grp)
                rows.append({"breakdown": dim, "value": CAT_EN.get(val_, val_), "method": m, "population": p,
                             "users": len(grp), "map5": grp.mean(), "ci_lo": lo, "ci_hi": hi})
bkd = pd.DataFrame(rows)
bkd.to_csv(OUT / "hybrid_breakdowns_val.csv", index=False)

inter = load_interactions()
ROB = {"main: weeks 21-30": build_targets(inter, W[21], W[31], THR),
       "clean: 21-24 + 28-30 (no 25-27)": build_targets(inter[inter.iso_week.isin([21, 22, 23, 24, 28, 29, 30])], W[21], W[31], THR),
       "weeks 21-25": build_targets(inter, W[21], W[26], THR), "weeks 26-30": build_targets(inter, W[26], W[31], THR)}
rrow = []
for k, tg in ROB.items():
    nds = n_distinct_shows(tg)
    us = nds[nds >= K].index
    td = {int(u): g.tv_show_id.tolist() for u, g in tg[(tg["rank"] <= K) & tg.user_id.isin(us)].groupby("user_id")}
    for m in SHOW:
        e = evaluate({u: RECS[m][u] for u in td}, tv, td)
        for p in ["ALL", "SUB"]:
            rrow.append({"truth_window": k, "method": m, "population": p, "users": e[p]["n"], "map5": e[p]["map5"],
                         "ci_lo": e[p]["ci_lo"], "ci_hi": e[p]["ci_hi"]})
pd.DataFrame(rrow).to_csv(OUT / "hybrid_robustness_windows_val.csv", index=False)

truth_new = {u: [x for x in t if x not in hist.get(u, set())] for u, t in tv.truth.items()}
truth_new = {u: t for u, t in truth_new.items() if t}
watched_any = set(tv.train.tv_show_id.unique().tolist())
in_1120 = set(labelled_airings(["11-20"]).tv_show_id.unique().tolist())   # same definition as Step 5
cold = [(u, x) for u, t in tv.truth.items() for x in t if x not in watched_any or x not in in_1120]
vr = []
for m in SHOW:
    R_ = RECS[m]
    us = list(tv.truth)
    fam = np.array([[x in hist.get(u, set()) for x in R_[u]] for u in us])
    hit = np.array([[x in set(tv.truth[u]) for x in R_[u]] for u in us])
    f, h = fam.ravel(), hit.ravel()
    lists = [R_[u] for u in us]
    cats = [[cat.get(x) for x in l] for l in lists]
    a = E[m]["ALL"]["ap"]
    vr.append({"method": m, **{f"hit_rate_rank{i + 1}": hit[:, i].mean() for i in range(K)},
               "share_ap_0": float((a == 0).mean()), "share_ap_1": float((a == 1).mean()), "median_ap": float(a.median()),
               "share_familiar": f.mean(), "precision_familiar": h[f].mean(),
               "precision_new": h[~f].mean() if (~f).any() else np.nan,
               "share_of_hits_new": (h & ~f).sum() / h.sum(), "mean_new_items_per_list": (~fam).sum(1).mean(),
               "recall5_truth_new": float(np.mean([len(set(R_[u]) & set(t)) / len(t) for u, t in truth_new.items()])),
               "cold_truth_pairs": len(cold), "recall_cold_pairs": float(np.mean([x in R_[u] for u, x in cold])),
               "distinct_shows_recommended": len({x for l in lists for x in l}),
               "catalogue_coverage": len({x for l in lists for x in l}) / len(tv.pool),
               "share_lists_ge3_categories": float(np.mean([len(set(c)) >= 3 for c in cats])),
               **{f"catmix_{CAT_EN.get(k_, k_)}": v for k_, v in pd.Series([c for cs in cats for c in cs]).value_counts(normalize=True).items()}})
tmix = pd.Series([cat.get(x) for t in tv.truth.values() for x in t]).value_counts(normalize=True)
vr.append({"method": "truth", **{f"catmix_{CAT_EN.get(k_, k_)}": v for k_, v in tmix.items()}})
vrdf = pd.DataFrame(vr)
vrdf.to_csv(OUT / "hybrid_lists_analysis_val.csv", index=False)
log(vrdf.round(4).T.to_string())

# ====================================================================== F4 failure analysis
d_ap = (E[HYB]["ALL"]["ap"] - E["M0_PF_exposure"]["ALL"]["ap"]).rename("d")
order_ = pd.DataFrame({"d": d_ap, "u": d_ap.index}).sort_values(["d", "u"])
loss, gain = order_.u.iloc[:200].tolist(), order_.u.iloc[::-1].iloc[:200].tolist()
hs_v = SRCS["val"]["hs"].set_index(["user_id", "tv_show_id"])


def grp_stats(us, name):
    tn = [len(truth_new.get(u, [])) / 5 for u in us]
    news_t = [np.mean([cat.get(x) == "Инфо" for x in tv.truth[u]]) for u in us]
    news_r = [np.mean([cat.get(x) == "Инфо" for x in RECS[HYB][u]]) for u in us]
    ratios = []
    for u in us:
        for x in tv.truth[u]:
            k_ = (np.uint64(u), x)
            if k_ in hs_v.index:
                r_ = hs_v.loc[k_]
                ratios.append(r_.a_target / max(r_.a_train, 1))
    return {"group": name, "users": len(us), "mean_ap_change": float(d_ap.reindex(us).mean()),
            "mean_q_shows_history": float(act.reindex(us).mean()), "share_truth_new_to_user": float(np.mean(tn)),
            "share_news_in_truth": float(np.mean(news_t)), f"share_news_in_{HYB}_top5": float(np.mean(news_r)),
            "median_exposure_ratio_truth_in_history": float(np.median(ratios)) if ratios else np.nan}


fa = pd.DataFrame([grp_stats(loss, "200 largest losses vs M0"), grp_stats(gain, "200 largest gains vs M0"),
                   grp_stats(list(tv.truth), "all val users")])
fa["share_users_with_zero_change"] = [np.nan, np.nan, float((d_ap == 0).mean())]
fa.to_csv(OUT / "hybrid_failure_groups_val.csv", index=False)
log(fa.round(4).to_string(index=False))
expl_m = MODELS[EXPL]
feat_names = FEATURES
cases = []
for u in loss:
    m0l, hl = RECS["M0_PF_exposure"][u], RECS[HYB][u]
    lost = [x for x in m0l if x in set(tv.truth[u]) and x not in hl]
    wrong = [x for x in hl if x not in set(tv.truth[u]) and x not in m0l]
    if lost and wrong:
        rows_ = Fv[(Fv.user_id == np.uint64(u)) & Fv.tv_show_id.isin([lost[0], wrong[0]])]
        if len(rows_) < 2:
            continue
        ctb = expl_m.predict(rows_[FEATURES].to_numpy(np.float32), pred_contrib=True)
        for (_, rr), cc in zip(rows_.iterrows(), ctb):
            top = np.argsort(-np.abs(cc[:-1]))[:4]
            cases.append({"user_id": str(u), "ap_change": float(d_ap[u]), "tv_show_id": int(rr.tv_show_id),
                          "title": tv.meta.title.get(int(rr.tv_show_id)), "category": cat.get(int(rr.tv_show_id)),
                          "role": "truth show dropped by hybrid" if rr.tv_show_id == lost[0] else "non-truth show promoted by hybrid",
                          "rank_in_M0": (m0l.index(int(rr.tv_show_id)) + 1) if int(rr.tv_show_id) in m0l else None,
                          "rank_in_hybrid": (hl.index(int(rr.tv_show_id)) + 1) if int(rr.tv_show_id) in hl else None,
                          "explained_by": EXPL, "shap_bias": float(cc[-1]), "shap_total": float(cc.sum()),
                          **{f"shap{i + 1}": f"{feat_names[j]}={cc[j]:+.4f} (value {rr[feat_names[j]]:.4g})" for i, j in enumerate(top)},
                          "hs_softfrac_pw": rr.hs_softfrac_pw, "ex_a_target_us_pw": rr.ex_a_target_us_pw,
                          "ex_rate_ratio_us": rr.ex_rate_ratio_us, "rh_rec_q": rr.rh_rec_q})
    if len(cases) >= 6:
        break
pd.DataFrame(cases).to_csv(OUT / "hybrid_failure_cases_val.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60):
    log(pd.DataFrame(cases).drop(columns=["explained_by"]).to_string(index=False))

# ====================================================================== H test (P3) + example users
t0 = time.perf_counter()
Fiv = pd.concat([Fi.assign(_g=0), Fv.assign(_g=1)], ignore_index=True)
Fiv["user_id_task"] = Fiv.user_id.astype(str) + "_" + Fiv._g.astype(str)
Fiv = Fiv.sort_values(["_g", "user_id"], kind="stable", ignore_index=True)
P3M = {}
for name in ["M3_lambdarank", "M4_lgb_binary"]:
    kind = lgb_kind[name]
    y = Fiv.graded.to_numpy() if kind == "rank" else Fiv.label.to_numpy()
    mdl = (lgb.LGBMRanker if kind == "rank" else lgb.LGBMClassifier)(n_estimators=NEST[name], **lgb_params(kind, BEST[name]))
    kw = {"group": Fiv.groupby(["_g", "user_id"], sort=False).size().to_numpy()} if kind == "rank" else {}
    mdl.fit(Fiv[FEATURES].to_numpy(np.float32), y, **kw)
    P3M[name] = mdl
P3M["M2_logreg"] = fit_lr(Fiv, FEATURES, float(BEST["M2_logreg"].C))
COST.append({"task": "inner+val", "stage": "P3 final fits (M2, M3, M4)", "seconds": round(time.perf_counter() - t0, 1)})
t0 = time.perf_counter()
TS = {"M1_fusion": m1_score(Ft, M1_ROW)}
for name in ["M3_lambdarank", "M4_lgb_binary"]:
    TS[name] = lgb_predict(P3M[name], Ft, FEATURES, lgb_kind[name])
TS["M2_logreg"] = lr_predict(P3M["M2_logreg"], Ft, FEATURES)
COST.append({"task": "test", "stage": "P3 inference (all models)", "seconds": round(time.perf_counter() - t0, 2)})
best_learned = p1[p1.method.isin(["M2_logreg", "M3_lambdarank", "M4_lgb_binary"])].sort_values(
    "avg_all_sub", ascending=False, kind="stable").method.iloc[0]
SUB06 = HYB if FINAL == HYB else best_learned
to_str = dict(zip((int(u) for u in tt.users), tt.user_str))
rec06 = top5(Ft, TS[SUB06], tt.users, tt.pop)
rec07 = top5(Ft, TS["M1_fusion"], tt.users, tt.pop)
write_submission("sub06_hybrid", {to_str[u]: v for u, v in rec06.items()}, tt)
write_submission("sub07_fusion", {to_str[u]: v for u, v in rec07.items()}, tt)
s05 = pd.read_csv(OUT / "submissions" / "sub05_pf_exposure.csv", dtype=str)
s05 = {r.user_id: [int(x) for x in r.tv_show_id.split()] for r in s05.itertuples()}
hist_t = {int(u): s for u, s in tt.train.groupby("user_id").tv_show_id.agg(set).items()}
cnt_t = Ft.groupby("user_id").size()
only_pop = Ft.groupby("user_id").apply(lambda g: bool(g[[f"rank_{s}" for s in SRC if s != "pop"]].isna().all().all()),
                                       include_groups=False)
summary["test"] = {"sub06_model": SUB06, "sub06_selected_by_rule": FINAL == SUB06,
                   "users_without_candidates": int(len(tt.users) - len(cnt_t)),
                   "users_with_only_popularity_candidates": int(only_pop.sum()),
                   "users_padded_sub06": int(sum(1 for u in tt.users if int(u) not in cnt_t.index or cnt_t[np.uint64(u)] < 5)),
                   "mean_overlap_sub06_vs_sub05": float(np.mean([len(set(rec06[int(u)]) & set(s05[to_str[int(u)]])) for u in tt.users])),
                   "mean_overlap_sub07_vs_sub05": float(np.mean([len(set(rec07[int(u)]) & set(s05[to_str[int(u)]])) for u in tt.users])),
                   "share_identical_lists_sub06_vs_sub05": float(np.mean([rec06[int(u)] == s05[to_str[int(u)]] for u in tt.users]))}
log("[H]", summary["test"])

exu = json.loads((OUT / "example_users.json").read_text())
erow = []
for role in ["light", "heavy"]:
    u = int(exu[role]["user_id"])
    lists = {f"hybrid ({HYB})": RECS[HYB][u], "pop_recent": RECS["pop_recent"][u], "PF_softfrac": RECS["PF_softfrac"][u],
             "PPR": RECS["PPR"][u], "cbf_pure": RECS["cbf_pure"][u], "cbf_pop": RECS["cbf_pop"][u],
             "PF_soft_exposure_g0.25 (M0)": RECS["M0_PF_exposure"][u], "truth": tv.truth[u]}
    fu = Fv[Fv.user_id == np.uint64(u)].set_index("tv_show_id")
    for sec, lst in lists.items():
        for r_, x in enumerate(lst, 1):
            row = {"role": role, "user_id": str(u), "list": sec, "rank": r_, "tv_show_id": x, "title": tv.meta.title.get(x),
                   "category": cat.get(x), "hit": x in set(tv.truth[u]), "in_history": x in hist.get(u, set())}
            if sec.startswith("hybrid") and x in fu.index:
                fr = fu.loc[[x]]
                row["model_score"] = float(SCORES_V[HYB][(Fv.user_id == np.uint64(u)).to_numpy() & (Fv.tv_show_id == x).to_numpy()][0])
                cc = expl_m.predict(fr[FEATURES].to_numpy(np.float32), pred_contrib=True)[0]
                for i, j in enumerate(np.argsort(-np.abs(cc[:-1]))[:3]):
                    row[f"shap{i + 1}"] = f"{FEATURES[j]}={cc[j]:+.4f}"
                row["shap_model"] = EXPL
            erow.append(row)
    names = [k for k in lists if k != "truth"]
    for a_, b_ in itertools.combinations(names, 2):
        erow.append({"role": role, "user_id": str(u), "list": "overlap", "title": f"{a_} vs {b_}",
                     "rank": len(set(lists[a_]) & set(lists[b_]))})
    log(f"[G] {role} {u}: hybrid hits {sum(x in set(tv.truth[u]) for x in lists[names[0]])}, AP hybrid "
        f"{E[HYB]['ALL']['ap'][u]:.4f} vs M0 {E['M0_PF_exposure']['ALL']['ap'][u]:.4f}")
exdf = pd.DataFrame(erow)
exdf.to_csv(OUT / "example_users_hybrid.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.width", 250, "display.max_columns", 14, "display.max_colwidth", 34):
    log(exdf[(exdf.list != "overlap") & exdf.list.str.startswith("hybrid")].drop(columns=["user_id"]).to_string(index=False))

# ====================================================================== F5/F6 cost + gap
ceil = pd.read_csv(OUT / "ceiling_analysis_val.csv").set_index("population")
summary["gap_closed"] = {p: (E[HYB][p]["map5"] - E["PF_softfrac"][p]["map5"]) /
                            (ceil.loc[p, "ceiling_map5_history_any_watch"] - E["PF_softfrac"][p]["map5"]) for p in ["ALL", "SUB"]}
summary["gap_closed_M0"] = {p: (E["M0_PF_exposure"][p]["map5"] - E["PF_softfrac"][p]["map5"]) /
                               (ceil.loc[p, "ceiling_map5_history_any_watch"] - E["PF_softfrac"][p]["map5"]) for p in ["ALL", "SUB"]}
log("[F6] gap closed:", summary["gap_closed"], "M0:", summary["gap_closed_M0"])

# ====================================================================== I plots
order_m = ["pop_recent", "cbf_pure", "cbf_pop", "PF_softfrac", "PPR", "M0_PF_exposure", "M1_fusion", "M2_logreg",
           "M3_lambdarank", "M4_lgb_binary"]
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(2)
wd = 0.085
for k, m in enumerate(order_m):
    v = [E[m][p]["map5"] for p in ["ALL", "SUB"]]
    err = [[E[m][p]["map5"] - E[m][p]["ci_lo"] for p in ["ALL", "SUB"]], [E[m][p]["ci_hi"] - E[m][p]["map5"] for p in ["ALL", "SUB"]]]
    ax.bar(x + (k - 4.5) * wd, v, width=wd, yerr=err, capsize=2, color=COL[m], label=m + (" (final)" if m == FINAL else ""))
for p_, xx in zip(["ALL", "SUB"], x):
    ax.hlines(E["PF_softfrac"][p_]["map5"], xx - 0.45, xx + 0.45, colors="black", linestyles="--", lw=1)
    ax.hlines(ceil.loc[p_, "ceiling_map5_history_any_watch"], xx - 0.45, xx + 0.45, colors="red", linestyles=":", lw=1.2)
ax.plot([], [], "k--", lw=1, label="PF_softfrac level")
ax.plot([], [], "r:", lw=1.2, label="history ceiling")
ax.set_xticks(x, ["ALL (n=4,030)", "SUB (n=1,150)"])
ax.set(title="Validation MAP@5 (P1: hybrids trained on the inner task, weeks 11-15 -> 16-20), 95% CI", ylabel="MAP@5",
       ylim=(0, 0.8))
ax.legend(fontsize=7.5, ncol=4, loc="upper left")
savefig(fig, "32_hybrid_map5_all_methods.png")

ad = abdf[abdf.experiment.str.startswith("drop")].sort_values("delta_ALL")
aa = abdf[abdf.experiment.str.startswith("history_strength +")].sort_values("delta_ALL")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].barh(ad.group, ad.delta_ALL, color=["#c0392b" if v < 0 else "#4a9a5b" for v in ad.delta_ALL])
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set(title=f"Drop one feature group (M3, P1): change in MAP@5 ALL\n(full model {E['M3_lambdarank']['ALL']['map5']:.4f})",
            xlabel="Delta MAP@5 ALL")
axes[1].barh(aa.group, aa.delta_ALL, color=["#c0392b" if v < 0 else "#4a9a5b" for v in aa.delta_ALL])
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set(title=f"Add one group to history-strength only\n(history only {e0['ALL']['map5']:.4f})", xlabel="Delta MAP@5 ALL")
savefig(fig, "33_hybrid_ablation.png")

top15 = imp.head(15).iloc[::-1]
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
axes[0].barh(top15.feature, top15.mean_abs_shap_val, color="#d62728")
axes[0].set(title="M3 mean |SHAP| on val candidates (top-15)", xlabel="Mean |contribution|")
tg15 = imp.sort_values("gain", ascending=False).head(15).iloc[::-1]
axes[1].barh(tg15.feature, tg15.gain, color="#3b6ea5")
axes[1].set(title="M3 total gain importance (top-15)", xlabel="Gain")
savefig(fig, "34_hybrid_importance.png")

bs = pd.DataFrame(only_rows)
bs = bs[bs.task == "val"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
xx = np.arange(len(bs))
axes[0].bar(xx - 0.2, bs.truth_pairs_in_source / (5 * len(tv.truth)), width=0.4, color="#3b6ea5", label="truth pairs in source top-k")
axes[0].bar(xx + 0.2, bs.share_of_truth_pairs_only_by_source, width=0.4, color="#e08a2c", label="found ONLY by this source")
axes[0].set_xticks(xx, [f"{s}\n(top-{k})" for s, k in zip(bs.source, bs.k)])
axes[0].axhline(task_summary(tv)["share_truth_pairs_in_history"], color="red", ls=":", label="history ceiling (any watch)")
axes[0].set(title="Candidate recall by source (val)", ylabel="Share of truth (user, show) pairs")
axes[0].legend(fontsize=8)
sn = sensdf[sensdf.k_per_source.apply(lambda v: isinstance(v, (int, np.integer)))]
ax2 = axes[1]
ax2.plot(sn.k_per_source, sn.map5_ALL, "o-", color="#d62728", label="M3 MAP@5 ALL")
ax2.set(title="Candidate-size sensitivity (same k for every source)", xlabel="k per source", ylabel="MAP@5 ALL")
ax3 = ax2.twinx()
ax3.plot(sn.k_per_source, sn.candidate_recall_val, "s--", color="#3b6ea5", label="candidate recall")
ax3.set_ylabel("Candidate recall (val)")
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="lower right")
savefig(fig, "35_hybrid_candidates.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, p in zip(axes, ["ALL", "SUB"]):
    d = bkd[(bkd.breakdown == "activity_bucket") & (bkd.population == p)]
    xb = np.arange(len(labels))
    for k, m in enumerate(SHOW):
        dd = d[d.method == m].set_index("value").reindex(labels)
        ax.bar(xb + (k - (len(SHOW) - 1) / 2) * 0.13, dd.map5, width=0.13, color=COL[m], label=m)
    nn = d[d.method == SHOW[0]].set_index("value").reindex(labels).users
    ax.set_xticks(xb, [f"{b}\n(n={int(v)})" for b, v in zip(labels, nn)])
    ax.set(title=f"MAP@5 by training activity, {p} users", xlabel="Distinct qualifying history shows", ylabel="MAP@5")
    ax.legend(fontsize=7)
savefig(fig, "36_hybrid_activity_buckets.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(d_ap, bins=np.linspace(-1, 1, 41), color="#d62728")
axes[0].set_yscale("log")
axes[0].set(title=f"Per-user AP change: {HYB} minus M0 (val ALL)\nmean {d_ap.mean():+.4f}, zero change {float((d_ap == 0).mean()):.1%}",
            xlabel="AP@5 change", ylabel="Users (log)")
mets = ["share_truth_new_to_user", "share_news_in_truth", f"share_news_in_{HYB}_top5"]
xg = np.arange(len(mets))
for k, (gname, c) in enumerate(zip(fa.group, ["#c0392b", "#4a9a5b", "#9a9a9a"])):
    axes[1].bar(xg + (k - 1) * 0.27, fa.loc[k, mets], width=0.27, color=c, label=gname)
axes[1].set_xticks(xg, ["truth shows new\nto the user", "news share\nin truth", "news share in\nhybrid top-5"])
axes[1].set(title="Loss vs gain users (vs M0)", ylabel="Share")
axes[1].legend(fontsize=8)
savefig(fig, "37_hybrid_failure_analysis.png")

# ====================================================================== cost + manifest
COST.append({"task": "all", "stage": "whole step", "seconds": round(time.perf_counter() - T0, 1), "peak_memory_mb_so_far": peak_mb()})
pd.DataFrame(COST).to_csv(OUT / "hybrid_cost.csv", index=False)
log(pd.DataFrame(COST).to_string(index=False))
summary["runtime_seconds"] = round(time.perf_counter() - T0, 1)
summary["peak_memory_mb"] = peak_mb()
HASHED = ["tasks_summary.csv", "hybrid_candidates_summary.csv", "hybrid_candidates_by_source.csv", "verify_features_naive.csv",
          "verify_features_hand.txt", "hybrid_m1_fusion_grid_inner.csv", "hybrid_model_selection_inner_cv.csv",
          "hybrid_p1_results_val.csv", "hybrid_p1_paired_diffs_val.csv", "hybrid_p2_cv_val.csv", "hybrid_ablation_val.csv",
          "hybrid_feature_importance_M3.csv", "hybrid_candidate_size_sensitivity.csv", "hybrid_breakdowns_val.csv",
          "hybrid_robustness_windows_val.csv", "hybrid_lists_analysis_val.csv", "hybrid_failure_groups_val.csv",
          "hybrid_failure_cases_val.csv", "example_users_hybrid.csv", "submissions/sub06_hybrid.csv",
          "submissions/sub07_fusion.csv"]


def content_hash(f):
    if f.endswith(".csv") and not f.startswith("submissions"):
        df = pd.read_csv(OUT / f)
        df = df.drop(columns=[c for c in df.columns if "seconds" in c], errors="ignore")
        return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    return hashlib.sha256((OUT / f).read_bytes()).hexdigest()


summary["sha256_excluding_timing_columns"] = {f: content_hash(f) for f in HASHED}
(OUT / "step6_summary.json").write_text(json.dumps(summary, indent=2, default=str))
log(json.dumps({k: v for k, v in summary.items() if not k.startswith("sha256")}, indent=2, default=str))
