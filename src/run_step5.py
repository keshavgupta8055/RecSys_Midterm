"""Step 5: Method B = content-based filtering (CBF) + schedule-exposure diagnostic. Run: python -m src.run_step5"""
import calendar
import hashlib
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from src import io_utils as io
from src.baselines import pf_recommend, pop_recommend, popularity
from src.cbf import (FEATURE_SETS, GENRE_POS, ROUND, Items, cosine_scores, list_tokens, parse_decade,
                     profiles, rank_lists, title_tokens)
from src.evaluate import ap_at_k, bootstrap_ci, map_at_k, paired_bootstrap_diff
from src.labels import build_targets, n_distinct_shows
from src.make_submission import write_submission
from src.ppr import build_graph, lazy_pop, personalisation, ppr, recommend
from src.protocol import K, THR, W, get_split, labelled_airings, load_interactions, pad_with_popularity

np.random.seed(42)
T0 = time.perf_counter()
OUT, FIG = io.ROOT / "outputs", io.ROOT / "outputs" / "figures"
DPI = 150
COL = {"pop_recent": "#9a9a9a", "PF_softfrac": "#3b6ea5", "PPR": "#4a9a5b", "cbf_pure": "#c0392b",
       "cbf_pop": "#e08a2c", "PF_exposure": "#7d4fa1"}
LAMS = [0.05, 0.1, 0.2, 0.35, 0.5]
summary = {}
proc = psutil.Process()


def log(*a):
    print(*a, flush=True)


def peak_mb():
    mi = proc.memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 1024**2)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)


# ====================================================================== setup
sp = get_split("val")
tr, truth, meta = sp.train, sp.truth, sp.meta
SUB = set(int(u) for u in sp.users[sp.sub_mask])
log(f"[B4] val training airing_start {tr.airing_start.min()} .. {tr.airing_start.max()}; metadata from schedules only")
assert tr.airing_start.max() < W[21]
POP = popularity(tr, sp.pool, "pop_recent", sp.train_end)          # pool ids in pop rank order


def pop_counts(train, pool_by_pop, train_end):
    q = train[(train.frac >= THR) & (train.airing_start >= train_end - pd.Timedelta(weeks=4))]
    c = q.groupby("tv_show_id").user_id.nunique().reindex(pool_by_pop, fill_value=0).to_numpy(float)
    return c / c.max()


POPNORM = pop_counts(tr, POP, sp.train_end)
hist = {int(u): set(s) for u, s in tr.groupby("user_id").tv_show_id.agg(set).items()}
nq = tr[tr.frac >= THR].groupby("user_id").tv_show_id.nunique()
step3 = pd.read_csv(OUT / "baseline_results_val.csv").set_index("method")


def evaluate(recs, tdict=None, sub=SUB):
    tdict = truth if tdict is None else tdict
    _, ap = map_at_k(recs, tdict)
    res = {}
    for pop, a in (("ALL", ap), ("SUB", ap[ap.index.isin(sub)])):
        lo, hi = bootstrap_ci(a)
        res[pop] = {"map5": float(a.mean()), "ci_lo": lo, "ci_hi": hi, "n": len(a), "ap": a}
    return res


pop_recs, _ = pop_recommend(sp.users, POP)
pf_recs, pf_pad = pf_recommend(tr, sp.pool, sp.users, "PF_softfrac", POP, sp.train_end)
E = {"pop_recent": evaluate(pop_recs), "PF_softfrac": evaluate(pf_recs)}
for m in E:
    for p in ["ALL", "SUB"]:
        assert abs(E[m][p]["map5"] - step3.loc[m, f"map5_{p}"]) < 1e-12
gP = build_graph(tr, "raw", "softfrac")
Pp, hasp = personalisation(gP, sp.users)
Rp, _ = ppr(Pp, gP.T, 0.85)
ppr_recs = recommend(Rp, hasp, sp.users, gP, sp.pool, 0.5, POP)[0]
ppr_new_all = recommend(Rp, hasp, sp.users, gP, sp.pool, 0.5, POP, mode="new_only", history=hist)[0]
E["PPR"] = evaluate(ppr_recs)
assert abs(E["PPR"]["ALL"]["map5"] - 0.325645988420182) < 1e-12
del gP, Pp, Rp
RECS = {"pop_recent": pop_recs, "PF_softfrac": pf_recs, "PPR": ppr_recs}
log("baselines + PPR reproduce Steps 3-4:", {m: round(E[m]["ALL"]["map5"], 4) for m in E})

# ====================================================================== A5 grid
log("[A5] CBF grid: F1-F4 x {softfrac, count08} x decay {none, 4wk} x {pure, pop lam 0.05..0.5}")
rows = []
SCACHE = {}


def run_combo(fs, weight, decay, bw=None):
    t0 = time.perf_counter()
    it = Items(meta, fs, bw)
    t_items = time.perf_counter() - t0
    P, has, Wm = profiles(it, tr, sp.users, weight, decay, sp.train_end)
    S = cosine_scores(P, it, POP)
    return it, P, has, Wm, S, t_items, time.perf_counter() - t0 - t_items


def eval_variant(S, has, lam):
    t0 = time.perf_counter()
    Sv = S if lam is None else (1 - lam) * S + lam * POPNORM[None, :]
    recs, padded, tie = rank_lists(Sv, has, sp.users, POP, POP)
    e = evaluate(recs)
    return recs, padded, tie, e, time.perf_counter() - t0


def grid_row(fs, weight, decay, lam, bw, e, padded, tie, has, t_items, t_score, t_rank):
    r = {"feature_set": fs, "seed_weight": weight, "decay_hl_weeks": decay or "none",
         "variant": "cbf_pure" if lam is None else "cbf_pop", "lam": 0.0 if lam is None else lam,
         "block_weights": json.dumps(bw or {}, sort_keys=True)}
    for p in ["ALL", "SUB"]:
        r.update({f"map5_{p}": e[p]["map5"], f"ci_lo_{p}": e[p]["ci_lo"], f"ci_hi_{p}": e[p]["ci_hi"]})
    r.update({"avg_all_sub": (e["ALL"]["map5"] + e["SUB"]["map5"]) / 2,
              "mean_padded_ALL": float(np.mean([padded[u] for u in truth])),
              "share_top5_tie_at_5th_6th": float(np.mean([tie[u] for u in truth])),
              "users_without_profile": int((~has).sum()),
              "item_build_seconds": round(t_items, 2), "profile_scoring_seconds": round(t_score, 2),
              "ranking_eval_seconds": round(t_rank, 2), "runtime_seconds": round(t_items + t_score + t_rank, 2)})
    return r


for fs in FEATURE_SETS:
    for weight in ["softfrac", "count08"]:
        for decay in [None, 4]:
            it, P, has, Wm, S, t_items, t_score = run_combo(fs, weight, decay)
            for lam in [None] + LAMS:
                _, padded, tie, e, t_rank = eval_variant(S, has, lam)
                rows.append(grid_row(fs, weight, decay, lam, None, e, padded, tie, has, t_items, t_score, t_rank))
                log(f"  {fs} {weight:8s} decay={str(decay):4s} {rows[-1]['variant']:8s} lam={rows[-1]['lam']:<4} "
                    f"ALL {e['ALL']['map5']:.4f} SUB {e['SUB']['map5']:.4f}")
grid = pd.DataFrame(rows)
SIMPLE = {"F1": 0, "F2": 1, "F3": 2, "F4": 3}


def select(g):
    g = g.assign(fr=g.feature_set.map(SIMPLE), dr=(g.decay_hl_weeks != "none").astype(int),
                 bwr=(g.block_weights != "{}").astype(int))
    return g.sort_values(["avg_all_sub", "runtime_seconds", "fr", "dr", "bwr", "lam"],
                         ascending=[False, True, True, True, True, True]).iloc[0]


log("[A1] block-weight grid on the best configuration of each family")
for fam in ["cbf_pure", "cbf_pop"]:
    b = select(grid[grid.variant == fam])
    decay = None if b.decay_hl_weeks == "none" else int(b.decay_hl_weeks)
    lam = None if fam == "cbf_pure" else float(b.lam)
    for gw, cw in [(1.0, 0.5), (2.0, 1.0), (2.0, 0.5)]:
        bw = {"genres": gw, "category": cw}
        it, P, has, Wm, S, t_items, t_score = run_combo(b.feature_set, b.seed_weight, decay, bw)
        _, padded, tie, e, t_rank = eval_variant(S, has, lam)
        rows.append(grid_row(b.feature_set, b.seed_weight, decay, lam, bw, e, padded, tie, has, t_items, t_score, t_rank))
        log(f"  {fam} {b.feature_set} {bw}: ALL {e['ALL']['map5']:.4f} SUB {e['SUB']['map5']:.4f}")
grid = pd.DataFrame(rows)
grid.to_csv(OUT / "cbf_grid_val.csv", index=False)
SEL = {}
for fam in ["cbf_pure", "cbf_pop"]:
    b = select(grid[grid.variant == fam])
    SEL[fam] = {"feature_set": b.feature_set, "seed_weight": b.seed_weight,
                "decay": None if b.decay_hl_weeks == "none" else int(b.decay_hl_weeks),
                "lam": None if fam == "cbf_pure" else float(b.lam), "block_weights": json.loads(b.block_weights)}
    log(f"[A5] selected {fam}:", SEL[fam], f"ALL {b.map5_ALL:.4f} SUB {b.map5_SUB:.4f}")
summary["selected"] = SEL
summary["grid_seconds"] = round(time.perf_counter() - T0, 1)


def run_selected(fam, split, train, users, pop, popnorm, history_=None):
    c = SEL[fam]
    t0 = time.perf_counter()
    it = Items(split.meta, c["feature_set"], c["block_weights"])
    t_items = time.perf_counter() - t0
    P, has, Wm = profiles(it, train, users, c["seed_weight"], c["decay"], split.train_end)
    S = cosine_scores(P, it, pop)
    Sv = S if c["lam"] is None else (1 - c["lam"]) * S + c["lam"] * popnorm[None, :]
    full = rank_lists(Sv, has, users, pop, pop)
    t_all = time.perf_counter() - t0
    new = rank_lists(Sv, has, users, pop, pop, mode="new_only", history=history_) if history_ else None
    return {"it": it, "P": P, "has": has, "W": Wm, "S": S, "Sv": Sv, "full": full, "new": new,
            "item_seconds": t_items, "profile_scoring_seconds": t_all - t_items}


SR = {f: run_selected(f, sp, tr, sp.users, POP, POPNORM, hist) for f in ["cbf_pure", "cbf_pop"]}
for f in SR:
    RECS[f] = SR[f]["full"][0]
    E[f] = evaluate(RECS[f])
    summary[f"{f}_val"] = {p: {k: v for k, v in E[f][p].items() if k != "ap"} for p in ["ALL", "SUB"]}
    summary[f"{f}_val"]["tie_share"] = float(np.mean([SR[f]["full"][2][u] for u in truth]))
    summary[f"{f}_val"]["mean_padded"] = float(np.mean([SR[f]["full"][1][u] for u in truth]))
log({f: summary[f"{f}_val"] for f in SR})

# ====================================================================== B3 sanity
it = SR["cbf_pure"]["it"]
nrm = np.sqrt(np.asarray(it.X.multiply(it.X).sum(axis=1)).ravel())
S0 = SR["cbf_pure"]["S"]
single = [(i, u) for i, u in enumerate(sp.users) if SR["cbf_pure"]["W"][i].nnz == 1]
ok = appl = 0
pos = {int(x): j for j, x in enumerate(POP)}
for i, u in single:
    top = RECS["cbf_pure"][int(u)]
    in_pool_identical = np.abs(S0[i] - 1.0) < 1e-12
    if in_pool_identical.any():
        appl += 1
        ok += any(abs(S0[i, pos[x]] - 1.0) < 1e-12 for x in top if x in pos)
summary["sanity"] = {"max_abs_norm_deviation": float(np.abs(nrm[~it.zero] - 1).max()), "zero_vector_shows": int(it.zero.sum()),
                     "items": int(it.X.shape[0]), "cosine_min": float(S0.min()), "cosine_max": float(S0.max()),
                     "single_show_users": len(single), "single_show_users_with_identical_vector_in_pool": appl,
                     "of_which_identical_vector_show_in_top5": ok}
log("[B3]", summary["sanity"])
assert ok == appl and S0.min() >= -1e-12 and S0.max() <= 1 + 1e-12

# ====================================================================== B2 naive cross-check


def naive_items(meta_df, blocks, bw):
    """Pure-Python item vectors {feature: value} (same tokenisers, own idf/normalisation)."""
    rows_ = list(meta_df.itertuples())
    raw = {b: [] for b in blocks}
    for r in rows_:
        for b in blocks:
            d = {}
            if b == "category" and isinstance(r.category, str):
                d[r.category] = 1.0
            elif b == "genres":
                for gname, w in zip([r.genre_1, r.genre_2, r.genre_3], GENRE_POS):
                    if isinstance(gname, str) and gname.strip():
                        d[gname.strip().lower()] = d.get(gname.strip().lower(), 0.0) + w
            elif b == "year" and parse_decade(r.year):
                d[parse_decade(r.year)] = 1.0
            elif b in ("director", "actors"):
                for t in list_tokens(getattr(r, b)):
                    d[t] = 1.0
            elif b == "title":
                for t in title_tokens(r.title):
                    d[t] = 1.0
            raw[b].append(d)
    for b in ("director", "actors", "title"):
        if b in raw:
            dfc = {}
            for d in raw[b]:
                for t in d:
                    dfc[t] = dfc.get(t, 0) + 1
            n = len(raw[b])
            raw[b] = [{t: v * (math.log((1 + n) / (1 + dfc[t])) + 1) for t, v in d.items()} for d in raw[b]]
    vecs = {}
    for k, r in enumerate(rows_):
        v = {}
        for b in blocks:
            d = raw[b][k]
            nn = math.sqrt(sum(x * x for x in d.values()))
            for t, x in d.items():
                v[(b, t)] = x / nn * bw.get(b, 1.0) if nn else 0.0
        nn = math.sqrt(sum(x * x for x in v.values()))
        vecs[int(r.Index)] = {f: x / nn for f, x in v.items()} if nn else {}
    return vecs


rng = np.random.default_rng(42)
pick = rng.choice(sp.users, 10, replace=False)
assert pick.dtype == np.uint64 and all(int(u) in truth for u in pick)
uidx = {int(u): i for i, u in enumerate(sp.users)}
pop_rank = {int(x): j for j, x in enumerate(POP)}
nrows = []
end_ts = calendar.timegm(sp.train_end.timetuple())
for fam in ["cbf_pure", "cbf_pop"]:
    c = SEL[fam]
    vecs = naive_items(meta, FEATURE_SETS[c["feature_set"]], c["block_weights"])
    for u in pick:
        u = int(u)
        rws = tr[tr.user_id == np.uint64(u)]
        wsum = {}
        for sh, fr, ts in zip(rws.tv_show_id.tolist(), rws.frac.tolist(), rws.airing_start.tolist()):
            w = fr if c["seed_weight"] == "softfrac" else (1.0 if fr >= THR else 0.0)
            if c["decay"]:
                age = (end_ts - calendar.timegm(ts.timetuple())) / (7 * 86400)
                w *= 0.5 ** (age / c["decay"])
            wsum[sh] = wsum.get(sh, 0.0) + w
        prof = {}
        for sh, w in wsum.items():
            if w > 0:
                for f, x in vecs.get(sh, {}).items():
                    prof[f] = prof.get(f, 0.0) + w * x
        nn = math.sqrt(sum(x * x for x in prof.values()))
        prof = {f: x / nn for f, x in prof.items()}
        sc = {}
        for j, sh in enumerate(POP.tolist()):
            cos = sum(prof.get(f, 0.0) * x for f, x in vecs[sh].items())
            sc[sh] = cos if c["lam"] is None else (1 - c["lam"]) * cos + c["lam"] * POPNORM[j]
        vec_row = SR[fam]["Sv"][uidx[u]]
        diff = max(abs(sc[sh] - vec_row[j]) for j, sh in enumerate(POP.tolist()))
        rs = {sh: float(np.round(v, ROUND)) for sh, v in sc.items()}
        order = sorted([sh for sh in rs if rs[sh] > 0], key=lambda sh: (-rs[sh], pop_rank[sh]))
        naive_top = pad_with_popularity(order[:K], POP)
        nrows.append({"variant": fam, "user_id": str(u), "pool_scores_compared": len(sc), "max_abs_diff": diff,
                      "top5_naive": " ".join(map(str, naive_top)), "top5_vectorised": " ".join(map(str, RECS[fam][u])),
                      "top5_identical": naive_top == RECS[fam][u]})
        log("   [B2]", nrows[-1])
nv = pd.DataFrame(nrows)
nv.to_csv(OUT / "verify_cbf_naive.csv", index=False)
assert (nv.max_abs_diff < 1e-10).all() and nv.top5_identical.all()
summary["naive_check"] = {"pairs": len(nv), "max_abs_diff": float(nv.max_abs_diff.max()),
                          "all_top5_identical": bool(nv.top5_identical.all())}


# ====================================================================== B5 determinism
def rhash(r):
    return hashlib.sha256(json.dumps({str(k): v for k, v in sorted(r.items())}).encode()).hexdigest()


det = {}
for fam in SR:
    again = run_selected(fam, sp, tr, sp.users, POP, POPNORM)["full"][0]
    det[fam] = {"hash1": rhash(RECS[fam]), "hash2": rhash(again), "identical": rhash(RECS[fam]) == rhash(again)}
    assert det[fam]["identical"]
summary["determinism"] = det
log("[B5]", det)

# ====================================================================== D exposure


def exposure_tables(split, periods, tgt_lo, tgt_hi):
    lab = labelled_airings(periods)
    tr_ = split.train
    a_tr = lab[(lab.start_time >= split.train_start) & (lab.start_time < split.train_end)]
    a_tg = lab[(lab.start_time >= tgt_lo) & (lab.start_time < tgt_hi)]
    c_tr = a_tr.groupby(["tv_show_id", "channel_id"]).size().rename("a_train")
    c_tg = a_tg.groupby(["tv_show_id", "channel_id"]).size().rename("a_target")
    glob_tg = a_tg.groupby("tv_show_id").size()
    usc = tr_[["user_id", "tv_show_id", "channel_id"]].drop_duplicates()
    usc = usc.join(c_tr, on=["tv_show_id", "channel_id"]).join(c_tg, on=["tv_show_id", "channel_id"]).fillna(0)
    us = usc.groupby(["user_id", "tv_show_id"])[["a_train", "a_target"]].sum()
    keys = [tr_.user_id, tr_.tv_show_id]
    us["softfrac"] = tr_.frac.groupby(keys).sum()
    us["q"] = (tr_.frac >= THR).astype(float).groupby(keys).sum()
    us["secs"] = tr_.watched_seconds.groupby(keys).sum()
    us = us.reset_index()
    us["a_target_global"] = us.tv_show_id.map(glob_tg).fillna(0)
    us = us[us.tv_show_id.isin(split.pool)].copy()
    us["exposure"] = np.where(us.a_target > 0, us.a_target, 0.1 * us.a_target_global)
    return us


EXPO = exposure_tables(sp, ["11-20", "21-30"], W[21], W[31])
log(f"[D] val exposure table: {len(EXPO):,} (user, history show in pool) pairs; target-window schedule used only for "
    f"show-channel-time counts, never joined to viewing")
EXPO["truth_rel"] = [t in set(truth.get(int(u), ())) for u, t in zip(EXPO.user_id, EXPO.tv_show_id)]
ev = EXPO[EXPO.user_id.isin(sp.users)].copy()
ev["ratio"] = ev.a_target / np.maximum(ev.a_train, 1)
drow = []
for rel, g in ev.groupby("truth_rel"):
    for col in ["a_target", "ratio", "a_train", "exposure"]:
        q = g[col].quantile([.1, .25, .5, .75, .9]).to_dict()
        drow.append({"pairs": "truth show" if rel else "non-truth show", "variable": col, "n": len(g),
                     "mean": g[col].mean(), **{f"q{int(k * 100)}": v for k, v in q.items()}})
diag = pd.DataFrame(drow)
bins = [-0.5, 0.5, 1.5, 3.5, 7.5, 15.5, 31.5, 63.5, np.inf]
blab = ["0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64+"]
ev["bin"] = pd.cut(ev.a_target, bins, labels=blab)
hr = ev.groupby("bin", observed=False).agg(pairs=("truth_rel", "size"), hit_rate=("truth_rel", "mean")).reset_index()
diag.to_csv(OUT / "exposure_diagnostics_val.csv", index=False)
hr.to_csv(OUT / "exposure_hitrate_by_atarget_val.csv", index=False)
log(diag.round(3).to_string(index=False))
log(hr.round(4).to_string(index=False))


def expo_variants(us):
    v = {}
    for gm in [0, 0.25, 0.5, 1.0]:
        v[f"PF_soft_exposure_g{gm}"] = us.softfrac * (1 + us.exposure) ** gm
    for a, b in [(0.5, 1), (1, 2), (0, 0.001)]:
        v[f"PF_rate_exposure_q_a{a}_b{b}"] = (us.q + a) / (us.a_train + b) * us.exposure
        v[f"PF_rate_exposure_soft_a{a}_b{b}"] = (us.softfrac + a) / (us.a_train + b) * us.exposure
    return v


def expo_recs(us, score, users, pop):
    c = pd.DataFrame({"user_id": us.user_id.values, "tv_show_id": us.tv_show_id.values, "score": score.values,
                      "secs": us.secs.values})
    c = c[c.score > 0].sort_values(["user_id", "score", "secs", "tv_show_id"], ascending=[True, False, False, True],
                                   kind="stable").groupby("user_id").head(K)
    personal = {int(u): g.tv_show_id.tolist() for u, g in c.groupby("user_id", sort=False)}
    recs, pad = {}, {}
    for u in users:
        p = personal.get(int(u), [])
        recs[int(u)] = pad_with_popularity(p, pop)
        pad[int(u)] = K - len(p)
    return recs, pad


erows, EXR = [], {}
for name, sc in expo_variants(EXPO).items():
    recs, pad = expo_recs(EXPO, sc, sp.users, POP)
    e = evaluate(recs)
    EXR[name] = (recs, e)
    r = {"method": name, "family": "PF-exposure baseline (not Method B)"}
    for p in ["ALL", "SUB"]:
        dlt, lo, hi = paired_bootstrap_diff(e[p]["ap"], E["PF_softfrac"][p]["ap"])
        r.update({f"map5_{p}": e[p]["map5"], f"ci_lo_{p}": e[p]["ci_lo"], f"ci_hi_{p}": e[p]["ci_hi"],
                  f"diff_vs_PF_{p}": dlt, f"diff_ci_lo_{p}": lo, f"diff_ci_hi_{p}": hi})
    r["avg_all_sub"] = (r["map5_ALL"] + r["map5_SUB"]) / 2
    r["mean_padded_ALL"] = float(np.mean([pad[u] for u in truth]))
    erows.append(r)
    log(f"  {name:34s} ALL {r['map5_ALL']:.4f} SUB {r['map5_SUB']:.4f} dPF {r['diff_vs_PF_ALL']:+.4f}")
assert EXR["PF_soft_exposure_g0"][0] == pf_recs, "gamma 0 must reproduce PF_softfrac"
ex = pd.DataFrame(erows)
ex.to_csv(OUT / "exposure_baselines_val.csv", index=False)
BEST_EX = ex.sort_values(["avg_all_sub", "method"], ascending=[False, True]).method.iloc[0]
RECS["PF_exposure"] = EXR[BEST_EX][0]
E["PF_exposure"] = EXR[BEST_EX][1]
ceil = pd.read_csv(OUT / "ceiling_analysis_val.csv").set_index("population")
gap = {p: (E["PF_exposure"][p]["map5"] - E["PF_softfrac"][p]["map5"]) /
          (ceil.loc[p, "ceiling_map5_history_any_watch"] - E["PF_softfrac"][p]["map5"]) for p in ["ALL", "SUB"]}
summary["best_exposure"] = {"method": BEST_EX, "map5_ALL": E["PF_exposure"]["ALL"]["map5"],
                            "map5_SUB": E["PF_exposure"]["SUB"]["map5"], "share_of_gap_PF_to_ceiling_closed": gap}
log("[D4] best exposure baseline", summary["best_exposure"])

# ====================================================================== C1 comparison
METHODS = ["pop_recent", "PF_softfrac", "PPR", "cbf_pure", "cbf_pop", "PF_exposure"]
crow = []
for p in ["ALL", "SUB"]:
    for m in METHODS:
        crow.append({"population": p, "method": m if m != "PF_exposure" else f"PF_exposure ({BEST_EX})",
                     "map5": E[m][p]["map5"], "ci_lo": E[m][p]["ci_lo"], "ci_hi": E[m][p]["ci_hi"], "n": E[m][p]["n"]})
    for a, b in [("cbf_pure", "pop_recent"), ("cbf_pure", "PF_softfrac"), ("cbf_pure", "PPR"), ("cbf_pop", "pop_recent"),
                 ("cbf_pop", "PF_softfrac"), ("cbf_pop", "PPR"), ("cbf_pop", "cbf_pure"), ("PF_exposure", "PF_softfrac")]:
        dlt, lo, hi = paired_bootstrap_diff(E[a][p]["ap"], E[b][p]["ap"])
        crow.append({"population": p, "comparison": f"{a} - {b}", "mean_diff": dlt, "diff_ci_lo": lo, "diff_ci_hi": hi})
cmp_ = pd.DataFrame(crow)
cmp_.to_csv(OUT / "cbf_vs_baselines_val.csv", index=False)
log(cmp_.round(4).to_string(index=False))

# ====================================================================== C2 familiar vs new + new_only
frows = []
for m in METHODS:
    for p, us in [("ALL", list(truth)), ("SUB", [u for u in truth if u in SUB])]:
        fam = np.array([[x in hist.get(u, set()) for x in RECS[m][u]] for u in us])
        hit = np.array([[x in set(truth[u]) for x in RECS[m][u]] for u in us])
        f, h = fam.ravel(), hit.ravel()
        frows.append({"method": m, "population": p, "share_familiar": f.mean(),
                      "precision_familiar": h[f].mean() if f.any() else np.nan,
                      "precision_new": h[~f].mean() if (~f).any() else np.nan,
                      "share_of_hits_new": (h & ~f).sum() / h.sum() if h.sum() else np.nan,
                      "mean_new_items_per_list": (~fam).sum(1).mean()})
fam_df = pd.DataFrame(frows)
fam_df.to_csv(OUT / "cbf_familiar_vs_new_val.csv", index=False)
log(fam_df.round(4).to_string(index=False))

truth_new = {u: [x for x in t if x not in hist.get(u, set())] for u, t in truth.items()}
truth_new = {u: t for u, t in truth_new.items() if t}
NEWR = {"cbf_pure_new_only": SR["cbf_pure"]["new"][0], "cbf_pop_new_only": SR["cbf_pop"]["new"][0],
        "PPR_new_only": ppr_new_all,
        "pop_recent_new_only": {u: pad_with_popularity([], lazy_pop(POP, hist.get(u, set()))) for u in truth_new}}
nrow = []
NAP = {}
for m, recs in NEWR.items():
    ap = pd.Series({u: ap_at_k(recs[u], truth_new[u]) for u in truth_new})
    rc = pd.Series({u: len(set(recs[u]) & set(truth_new[u])) / len(truth_new[u]) for u in truth_new})
    NAP[m] = ap
    for p, mask in [("ALL", np.ones(len(ap), bool)), ("SUB", ap.index.isin(SUB))]:
        lo, hi = bootstrap_ci(ap[mask])
        rlo, rhi = bootstrap_ci(rc[mask])
        nrow.append({"method": m, "population": p, "users_with_truth_new": int(mask.sum()),
                     "map5_vs_truth_new": ap[mask].mean(), "ci_lo": lo, "ci_hi": hi,
                     "recall5_vs_truth_new": rc[mask].mean(), "recall_ci_lo": rlo, "recall_ci_hi": rhi})
s4 = pd.read_csv(OUT / "ppr_new_only_val.csv")
for m in ["PPR_new_only", "pop_recent_new_only"]:
    ref = s4[(s4.method == m) & (s4.population == "ALL")].map5_vs_truth_new.iloc[0]
    assert abs(NAP[m].mean() - ref) < 1e-12, (m, NAP[m].mean(), ref)
for m in ["cbf_pure_new_only", "cbf_pop_new_only"]:
    for p in ["ALL", "SUB"]:
        a = NAP[m] if p == "ALL" else NAP[m][NAP[m].index.isin(SUB)]
        b = NAP["pop_recent_new_only"] if p == "ALL" else NAP["pop_recent_new_only"][NAP["pop_recent_new_only"].index.isin(SUB)]
        dlt, lo, hi = paired_bootstrap_diff(a, b)
        nrow.append({"method": f"{m} - pop_recent_new_only (paired MAP)", "population": p, "map5_vs_truth_new": dlt,
                     "ci_lo": lo, "ci_hi": hi})
newdf = pd.DataFrame(nrow)
newdf.to_csv(OUT / "cbf_new_only_val.csv", index=False)
log(newdf.round(4).to_string(index=False))

# ====================================================================== C3 cold items
watched_any = set(tr.tv_show_id.unique().tolist())
in_1120 = set(labelled_airings(["11-20"]).tv_show_id.unique().tolist())
cold = [(u, x) for u, t in truth.items() for x in t if x not in watched_any or x not in in_1120]
crow2 = []
for m in METHODS:
    for p, cp in [("ALL", cold), ("SUB", [(u, x) for u, x in cold if u in SUB])]:
        crow2.append({"method": m, "population": p, "cold_truth_pairs": len(cp),
                      "cold_pairs_never_watched_in_training": sum(x not in watched_any for _, x in cp),
                      "cold_pairs_absent_from_11_20_schedule": sum(x not in in_1120 for _, x in cp),
                      "recall_of_cold_pairs": float(np.mean([x in RECS[m][u] for u, x in cp])) if cp else np.nan})
for m in ["cbf_pure_new_only", "cbf_pop_new_only", "PPR_new_only"]:
    crow2.append({"method": m, "population": "ALL", "cold_truth_pairs": len(cold),
                  "recall_of_cold_pairs": float(np.mean([x in NEWR[m][u] for u, x in cold]))})
cold_df = pd.DataFrame(crow2)
cold_df.to_csv(OUT / "cbf_cold_items_val.csv", index=False)
log(cold_df.round(4).to_string(index=False))

# ====================================================================== C4 breakdowns
act = pd.Series({u: int(nq.get(np.uint64(u), 0)) for u in truth})
labels = ["0-4", "5-9", "10-19", "20-49", "50+"]
bucket = pd.cut(act, [-1, 4, 9, 19, 49, np.inf], labels=labels)
brow = []
for p in ["ALL", "SUB"]:
    for bl in labels:
        for m in METHODS:
            a = E[m][p]["ap"]
            a = a[bucket.reindex(a.index) == bl]
            lo, hi = bootstrap_ci(a)
            brow.append({"population": p, "activity_bucket": bl, "method": m, "users": len(a), "map5": a.mean(),
                         "ci_lo": lo, "ci_hi": hi})
bk = pd.DataFrame(brow)
bk.to_csv(OUT / "cbf_breakdown_activity_val.csv", index=False)

inter = load_interactions()
ROB = {"main: weeks 21-30": build_targets(inter, W[21], W[31], THR),
       "clean: 21-24 + 28-30 (no 25-27)": build_targets(inter[inter.iso_week.isin([21, 22, 23, 24, 28, 29, 30])],
                                                        W[21], W[31], THR),
       "weeks 21-25": build_targets(inter, W[21], W[26], THR), "weeks 26-30": build_targets(inter, W[26], W[31], THR)}
sub_all = set(int(x) for x in get_split("test").users)
rrow = []
for k, tg in ROB.items():
    nds = n_distinct_shows(tg)
    us = nds[nds >= K].index
    top = tg[(tg["rank"] <= K) & tg.user_id.isin(us)]
    td = {int(u): g.tv_show_id.tolist() for u, g in top.groupby("user_id")}
    assert set(td) <= set(truth)
    for m in METHODS:
        e = evaluate({u: RECS[m][u] for u in td}, td, sub_all)
        for p in ["ALL", "SUB"]:
            rrow.append({"truth_window": k, "method": m, "population": p, "users": e[p]["n"], "map5": e[p]["map5"],
                         "ci_lo": e[p]["ci_lo"], "ci_hi": e[p]["ci_hi"]})
rob = pd.DataFrame(rrow)
rob.to_csv(OUT / "cbf_robustness_windows_val.csv", index=False)

hrow, arow = [], []
for m in METHODS:
    for p in ["ALL", "SUB"]:
        us = E[m][p]["ap"].index
        hits = np.array([[RECS[m][u][i] in set(truth[u]) for i in range(K)] for u in us])
        for i in range(K):
            hrow.append({"method": m, "population": p, "rank": i + 1, "hit_rate_at_rank": hits[:, i].mean()})
        a = E[m][p]["ap"]
        d = a.describe(percentiles=[.25, .5, .75, .9]).to_dict()
        d.update({"method": m, "population": p, "share_ap_0": float((a == 0).mean()), "share_ap_1": float((a == 1).mean())})
        arow.append(d)
pd.DataFrame(hrow).to_csv(OUT / "cbf_hit_rank_val.csv", index=False)
pd.DataFrame(arow).to_csv(OUT / "cbf_per_user_ap_val.csv", index=False)

# ====================================================================== C5 coverage / diversity
cat = meta.category
vrow, mix = [], {}
for m in METHODS:
    lists = [RECS[m][u] for u in truth]
    cats = [[cat.get(x) for x in l] for l in lists]
    dist = len({x for l in lists for x in l})
    vrow.append({"method": m, "distinct_shows_recommended": dist, "pool_size": len(sp.pool),
                 "catalogue_coverage": dist / len(sp.pool),
                 "share_lists_ge3_categories": float(np.mean([len(set(c)) >= 3 for c in cats])),
                 "mean_pairwise_category_diversity": float(np.mean([np.mean([a != b for i, a in enumerate(c)
                                                                             for b in c[i + 1:]]) for c in cats]))})
    mix[m] = pd.Series([c for cs in cats for c in cs]).value_counts(normalize=True)
mix["truth"] = pd.Series([cat.get(x) for t in truth.values() for x in t]).value_counts(normalize=True)
cov = pd.DataFrame(vrow)
cov.to_csv(OUT / "cbf_coverage_diversity_val.csv", index=False)
mixdf = pd.DataFrame(mix).fillna(0)
mixdf.rename_axis("category").reset_index().to_csv(OUT / "cbf_category_mix_val.csv", index=False, encoding="utf-8-sig")
log(cov.round(4).to_string(index=False))

# ====================================================================== F test files
ts = get_split("test")
log(f"[B4] test training airing_start {ts.train.airing_start.min()} .. {ts.train.airing_start.max()}")
assert ts.train.airing_start.max() < W[31]
POP_T = popularity(ts.train, ts.pool, "pop_recent", ts.train_end)
POPNORM_T = pop_counts(ts.train, POP_T, ts.train_end)
to_str = dict(zip((int(u) for u in ts.users), ts.user_str))
TS = {f: run_selected(f, ts, ts.train, ts.users, POP_T, POPNORM_T) for f in ["cbf_pure", "cbf_pop"]}
CBF_SUB = "cbf_pop" if E["cbf_pop"]["ALL"]["map5"] + E["cbf_pop"]["SUB"]["map5"] > \
    E["cbf_pure"]["ALL"]["map5"] + E["cbf_pure"]["SUB"]["map5"] else "cbf_pure"
write_submission("sub04_cbf", {to_str[u]: v for u, v in TS[CBF_SUB]["full"][0].items()}, ts)
EXPO_T = exposure_tables(ts, ["11-20", "21-30", "31-42"], W[31], W[43])
ex_recs_t, ex_pad_t = expo_recs(EXPO_T, expo_variants(EXPO_T)[BEST_EX], ts.users, POP_T)
write_submission("sub05_pf_exposure", {to_str[u]: v for u, v in ex_recs_t.items()}, ts)
summary["test"] = {"sub04_variant": CBF_SUB,
                   "sub04_users_padded": int(sum(v > 0 for v in TS[CBF_SUB]["full"][1].values())),
                   "sub04_users_without_profile": int((~TS[CBF_SUB]["has"]).sum()),
                   "sub05_method": BEST_EX, "sub05_users_padded": int(sum(v > 0 for v in ex_pad_t.values())),
                   "sub05_users_without_history_in_pool": int(sum(v == K for v in ex_pad_t.values())),
                   "test_items_shape": list(TS[CBF_SUB]["it"].X.shape)}
log("[F]", summary["test"])

# ====================================================================== C6 cost
cost = []
for name, R in [("val", SR), ("test", TS)]:
    for f in ["cbf_pure", "cbf_pop"]:
        X = R[f]["it"].X
        cost.append({"split": name, "variant": f, "items": X.shape[0], "features": X.shape[1], "nnz": X.nnz,
                     "item_build_seconds": round(R[f]["item_seconds"], 2),
                     "profile_scoring_ranking_seconds": round(R[f]["profile_scoring_seconds"], 2),
                     "users": len(R[f]["has"])})

# ====================================================================== E example users
exu = json.loads((OUT / "example_users.json").read_text())
ppr_ex = RECS["PPR"]
erow = []
itp = SR["cbf_pure"]["it"]
for role in ["light", "heavy"]:
    u = int(exu[role]["user_id"])
    i = uidx[u]
    pvec = SR["cbf_pure"]["P"][i].toarray().ravel()
    for blk in ["category", "genres"]:
        a, b = itp.blocks[blk]
        o = np.argsort(-pvec[a:b], kind="stable")[:5]
        for r_, j in enumerate(o, 1):
            if pvec[a + j] > 0:
                erow.append({"role": role, "user_id": str(u), "section": f"profile_top_{blk}", "rank": r_,
                             "title": itp.names[a + j].split(":", 1)[1], "score": round(float(pvec[a + j]), 4)})
    wrow = SR["cbf_pure"]["W"][i]
    contrib = {int(itp.ids[j]): float(w * (itp.X[j] @ pvec)[0]) for j, w in zip(wrow.indices, wrow.data)}
    tot = sum(contrib.values())
    for r_, (sh, cv) in enumerate(sorted(contrib.items(), key=lambda kv: -kv[1])[:5], 1):
        erow.append({"role": role, "user_id": str(u), "section": "profile_top_contributing_training_shows", "rank": r_,
                     "tv_show_id": sh, "title": meta.title.get(sh), "category": meta.category.get(sh),
                     "score": round(cv / tot, 4)})
    hist_rows = [itp.row[s] for s in hist.get(u, set()) if s in itp.row]
    lists = {"cbf_pure": RECS["cbf_pure"][u], "cbf_pop": RECS["cbf_pop"][u],
             "cbf_pure_new_only": SR["cbf_pure"]["new"][0][u], "PF_softfrac": pf_recs[u], "PPR": ppr_ex[u],
             "pop_recent": pop_recs[u], f"PF_exposure ({BEST_EX})": RECS["PF_exposure"][u]}
    for sec, lst in lists.items():
        for r_, x in enumerate(lst, 1):
            row = {"role": role, "user_id": str(u), "section": sec, "rank": r_, "tv_show_id": x,
                   "title": meta.title.get(x), "category": meta.category.get(x),
                   "genres": ", ".join(str(g) for g in meta.loc[x, ["genre_1", "genre_2", "genre_3"]] if isinstance(g, str)),
                   "hit": x in set(truth[u]), "in_training_history": x in hist.get(u, set())}
            if sec.startswith("cbf"):
                row["score"] = round(float(SR["cbf_pure"]["S"][i, pop_rank[x]] if sec != "cbf_pop"
                                           else SR["cbf_pop"]["Sv"][i, pop_rank[x]]), 6)
                sims = (itp.X[hist_rows] @ itp.X[itp.row[x]].T).toarray().ravel()
                o = [k for k in np.argsort(-sims, kind="stable") if int(itp.ids[hist_rows[k]]) != x][:2]
                for n_, k in enumerate(o, 1):
                    sid = int(itp.ids[hist_rows[k]])
                    row[f"similar_training_show_{n_}"] = f"{sid} {meta.title.get(sid)}"
                    row[f"similar_training_show_{n_}_cos"] = round(float(sims[k]), 4)
            erow.append(row)
    names = list(lists)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ov = len(set(lists[names[a]]) & set(lists[names[b]]))
            erow.append({"role": role, "user_id": str(u), "section": "overlap", "title": f"{names[a]} vs {names[b]}",
                         "score": ov})
    log(f"[E] {role} {u}: overlaps " + "; ".join(f"{names[a]}~{names[b]}={len(set(lists[names[a]]) & set(lists[names[b]]))}"
                                                  for a in range(len(names)) for b in range(a + 1, len(names))))
exdf = pd.DataFrame(erow)
exdf.to_csv(OUT / "example_users_cbf.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.width", 260, "display.max_columns", 16, "display.max_colwidth", 28):
    log(exdf[exdf.section != "overlap"].drop(columns=["user_id"]).to_string(index=False))

# ====================================================================== G plots
ref = E["PF_softfrac"]["ALL"]["map5"]
best_lam = SEL["cbf_pop"]["lam"]
g0 = grid[grid.block_weights == "{}"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
for ax, (fam, lam) in zip(axes, [("cbf_pure", 0.0), ("cbf_pop", best_lam)]):
    d = g0[(g0.variant == fam) & (np.isclose(g0.lam, lam))]
    x = np.arange(4)
    for off, (wt, dc), c in zip([-0.3, -0.1, 0.1, 0.3], [("softfrac", "none"), ("softfrac", "4"), ("count08", "none"),
                                                          ("count08", "4")], ["#3b6ea5", "#9cc3e6", "#e08a2c", "#f5c58f"]):
        dd = d[(d.seed_weight == wt) & (d.decay_hl_weeks.astype(str) == dc)].set_index("feature_set").loc[list(FEATURE_SETS)]
        ax.bar(x + off, dd.map5_ALL, width=0.2, color=c, label=f"{wt}, decay {dc}")
    ax.axhline(ref, color="black", ls="--", lw=1, label=f"PF_softfrac {ref:.4f}")
    ax.axhline(E["pop_recent"]["ALL"]["map5"], color="grey", ls=":", lw=1, label="pop_recent")
    ax.set_xticks(x, ["F1 cat+genre", "F2 +year", "F3 +dir+actors", "F4 +title"])
    ax.set(title=f"{fam}{'' if fam == 'cbf_pure' else f' (lam = {lam})'}: validation MAP@5, ALL users",
           ylabel="MAP@5", xlabel="Feature set")
    ax.legend(fontsize=8, loc="upper left")
savefig(fig, "26_cbf_grid.png")

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(2)
wd = 0.13
for k, m in enumerate(METHODS):
    v = [E[m][p]["map5"] for p in ["ALL", "SUB"]]
    err = [[E[m][p]["map5"] - E[m][p]["ci_lo"] for p in ["ALL", "SUB"]], [E[m][p]["ci_hi"] - E[m][p]["map5"] for p in ["ALL", "SUB"]]]
    lab = m if m != "PF_exposure" else f"PF_exposure ({BEST_EX})"
    ax.bar(x + (k - 2.5) * wd, v, width=wd, color=COL[m], yerr=err, capsize=3, label=lab)
    for xx, p in zip(x + (k - 2.5) * wd, ["ALL", "SUB"]):
        ax.text(xx, E[m][p]["ci_hi"] + 0.006, f"{E[m][p]['map5']:.3f}", ha="center", va="bottom", fontsize=7,
                rotation=90)
ax.set_xticks(x, ["ALL (n=4,030)", "SUB (n=1,150)"])
ax.set(title="Validation MAP@5 with 95% bootstrap CI", ylabel="MAP@5", ylim=(0, 0.5))
ax.legend(fontsize=8, loc="upper left", ncol=2)
savefig(fig, "27_map5_all_methods.png")

fa = fam_df[fam_df.population == "ALL"].set_index("method").loc[METHODS]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
xm = np.arange(len(METHODS))
axes[0].bar(xm, fa.share_familiar, color="#3b6ea5", label="familiar (in training history)")
axes[0].bar(xm, 1 - fa.share_familiar, bottom=fa.share_familiar, color="#e08a2c", label="new")
axes[0].set_xticks(xm, METHODS, rotation=20)
axes[0].set(title="Composition of recommended items (ALL users)", ylabel="Share of recommended items", ylim=(0, 1.25))
axes[0].legend(loc="upper center", ncol=2)
axes[1].bar(xm - 0.2, fa.precision_familiar, width=0.4, color="#3b6ea5", label="precision of familiar items")
axes[1].bar(xm + 0.2, fa.precision_new.fillna(0), width=0.4, color="#e08a2c", label="precision of new items")
axes[1].set_xticks(xm, METHODS, rotation=20)
axes[1].set(title="Share of recommended items that are in the truth (ALL)", ylabel="Precision",
            ylim=(0, fa.precision_familiar.max() * 1.35))
axes[1].legend(loc="upper center", ncol=2)
savefig(fig, "28_familiar_vs_new_all_methods.png")

na = newdf[newdf.method.isin(list(NEWR))]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
nm_col = {"pop_recent_new_only": "#9a9a9a", "PPR_new_only": "#4a9a5b", "cbf_pure_new_only": "#c0392b",
          "cbf_pop_new_only": "#e08a2c"}
for ax, (metric, lo, hi, ttl) in zip(axes, [("map5_vs_truth_new", "ci_lo", "ci_hi", "MAP@5"),
                                            ("recall5_vs_truth_new", "recall_ci_lo", "recall_ci_hi", "Recall@5")]):
    for k, (m, c) in enumerate(nm_col.items()):
        dd = na[na.method == m].set_index("population").loc[["ALL", "SUB"]]
        ax.bar(np.arange(2) + (k - 1.5) * 0.2, dd[metric], width=0.2, color=c,
               yerr=[dd[metric] - dd[lo], dd[hi] - dd[metric]], capsize=3, label=m)
    nn = na[na.method == "PPR_new_only"].set_index("population").users_with_truth_new
    ax.set_xticks(np.arange(2), [f"ALL (n={int(nn['ALL']):,})", f"SUB (n={int(nn['SUB']):,})"])
    ax.set(title=f"Discovery of unseen shows: {ttl} vs truth_new", ylabel=ttl)
    ax.legend(fontsize=8)
savefig(fig, "29_new_only_discovery_all.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(hr.bin.astype(str), hr.hit_rate, color="#7d4fa1")
for xx, (h_, n_) in enumerate(zip(hr.hit_rate, hr.pairs)):
    axes[0].text(xx, h_ + 0.005, f"n={n_:,}", ha="center", fontsize=7)
axes[0].set(title="Hit rate of (user, history show) pairs vs target-window airings\non the user's channels (A_target_us)",
            xlabel="A_target_us (airings of the show in weeks 21-30 on channels the user watched it on)",
            ylabel="Share of pairs whose show is in the user's truth")
for rel, c, lab in [(True, "#c0392b", "truth shows"), (False, "#9a9a9a", "non-truth history shows")]:
    r = ev[ev.truth_rel == rel].ratio
    axes[1].hist(np.clip(r, 0, 5), bins=np.linspace(0, 5, 51), histtype="step", lw=2, color=c, density=True,
                 label=f"{lab} (median {r.median():.2f}, n={len(r):,})")
axes[1].set(title="Ratio A_target_us / max(A_train_us, 1) (clipped at 5)", xlabel="Ratio", ylabel="Density")
axes[1].legend()
savefig(fig, "30_exposure_diagnostics.png")

CAT_EN = {"Детям": "Kids", "Сериалы": "Series", "Фильмы": "Films", "Развлечения": "Entertainment", "Инфо": "Info",
          "Познавательное": "Educational", "Другое": "Other", "Спорт": "Sport"}
mx = mixdf[METHODS + ["truth"]]
mx.index = [f"{CAT_EN.get(c, c)}" for c in mx.index]
mx = mx.loc[mx["truth"].sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(12, 5))
xc = np.arange(len(mx))
cols = METHODS + ["truth"]
for k, m in enumerate(cols):
    ax.bar(xc + (k - 3) * 0.12, mx[m], width=0.12, color=COL.get(m, "black"), label=m)
ax.set_xticks(xc, mx.index)
ax.set(title="Category mix of recommended items vs truth (validation, ALL users)", ylabel="Share", xlabel="Category")
ax.legend(fontsize=8, ncol=2)
savefig(fig, "31_category_mix_all_methods.png")

# ====================================================================== manifest
summary["runtime_seconds"] = round(time.perf_counter() - T0, 1)
summary["peak_memory_mb"] = peak_mb()
pd.DataFrame(cost).assign(peak_memory_mb_whole_step=summary["peak_memory_mb"]).to_csv(OUT / "cbf_cost.csv", index=False)
log(pd.DataFrame(cost).to_string(index=False))
HASHED = ["cbf_grid_val.csv", "cbf_vs_baselines_val.csv", "cbf_familiar_vs_new_val.csv", "cbf_new_only_val.csv",
          "cbf_cold_items_val.csv", "cbf_breakdown_activity_val.csv", "cbf_robustness_windows_val.csv",
          "cbf_hit_rank_val.csv", "cbf_per_user_ap_val.csv", "cbf_coverage_diversity_val.csv", "cbf_category_mix_val.csv",
          "exposure_diagnostics_val.csv", "exposure_hitrate_by_atarget_val.csv", "exposure_baselines_val.csv",
          "verify_cbf_naive.csv", "example_users_cbf.csv", "submissions/sub04_cbf.csv",
          "submissions/sub05_pf_exposure.csv"]


def content_hash(f):
    if f.endswith(".csv") and not f.startswith("submissions"):
        df = pd.read_csv(OUT / f)
        df = df.drop(columns=[c for c in df.columns if "seconds" in c], errors="ignore")
        return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    return hashlib.sha256((OUT / f).read_bytes()).hexdigest()


summary["sha256_excluding_timing_columns"] = {f: content_hash(f) for f in HASHED}
(OUT / "step5_summary.json").write_text(json.dumps(summary, indent=2, default=str))
log(json.dumps({k: v for k, v in summary.items() if not k.startswith("sha256")}, indent=2, default=str))
