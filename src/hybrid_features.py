"""Candidate generation and features for the hybrid re-ranker (vectorised; no per-user loops for features
except the max-cosine-to-top-5 term, which is a small sparse product per user).

Sources (selected configs from Steps 3-5): PF_softfrac, PF_soft_exposure_g0.25, PPR (raw, softfrac, d 0.85,
beta 0.5), cbf_pop (F4, count08, 4-week decay, lam 0.35), cbf_pure (F4, count08, category 0.5), pop_recent.
A superset with the top-KMAX of every source is built once; the per-source ranks are kept so that any
per-source cut-off (main: SRC_K) is a row filter.
Features only use history interactions and schedule counts; the target-window schedule is used only to count
which show airs on which channel when (never joined to viewing). Counts are divided by window weeks so the
5-, 10- and 20-week history windows are comparable.
"""
import time

import numpy as np
import pandas as pd

from src.cbf import ROUND, Items, cosine_scores, profiles
from src.ppr import build_graph, personalisation, ppr
from src.protocol import K, THR

SRC = ["pf", "pfx", "ppr", "cbfpop", "cbf", "pop"]
SRC_K = {"pf": 30, "pfx": 30, "ppr": 20, "cbfpop": 30, "cbf": 30, "pop": 20}
KMAX = 50
CATS = ["Детям", "Сериалы", "Фильмы", "Развлечения", "Инфо", "Познавательное", "Другое", "Спорт"]
CAT_EN = dict(zip(CATS, ["kids", "series", "films", "entertainment", "info", "educational", "other", "sport"]))

GROUPS = {
    "history_strength": ["hs_softfrac_pw", "hs_q_pw", "hs_c05_pw", "hs_hours_pw", "hs_share_softfrac", "hs_share_q",
                         "hs_rank_inv_softfrac", "hs_rank_inv_q", "hs_mean_frac", "hs_n_channels"],
    "recency_habit": ["rh_weeks_q_share", "rh_weeks_any_share", "rh_rec_q", "rh_rec_any", "rh_first_seen"],
    "exposure": ["ex_a_target_us_pw", "ex_a_train_us_pw", "ex_a_target_glob_pw", "ex_a_train_glob_pw",
                 "ex_log_a_target_us_pw", "ex_log_a_train_us_pw", "ex_log_a_target_glob_pw", "ex_log_a_train_glob_pw",
                 "ex_rate_ratio_us", "ex_zero_us", "ex_other_channels_only", "ex_glob_rate_ratio"],
    "ppr": ["ppr_score_rel", "ppr_rank_inv"],
    "cbf": ["cbf_cos", "cbfpop_score", "cbf_rank_inv", "cbfpop_rank_inv", "cbf_maxcos_top5"],
    "popularity": ["pop_norm", "pop_rank_inv", "pop_hist", "new_show", "never_watched"],
    "user_context": ["u_q_shows_pw", "u_hours_pw", "u_activity_bucket", "u_mean_softfrac", "u_news_share"],
    "show_attributes": [f"cat_{CAT_EN[c]}" for c in CATS] + ["sh_dur_median_min", "sh_n_channels_target"],
}
FEATURES = [f for g in GROUPS.values() for f in g]


# ---------------------------------------------------------------------------- history / exposure tables
def history_stats(t):
    """Per (user, show) statistics over the task's history interactions."""
    H = t.train
    q = H.frac >= THR
    keys = [H.user_id, H.tv_show_id]
    g = pd.DataFrame({
        "softfrac": H.frac.groupby(keys).sum(),
        "q": q.astype(float).groupby(keys).sum(),
        "c05": (H.frac >= 0.5).astype(float).groupby(keys).sum(),
        "secs": H.watched_seconds.astype(float).groupby(keys).sum(),
        "n_air": H.frac.groupby(keys).size().astype(float),
        "weeks_any": H.iso_week.groupby(keys).nunique().astype(float),
        "last_any": H.airing_start.groupby(keys).max(),
        "first_any": H.airing_start.groupby(keys).min(),
        "n_channels": H.channel_id.groupby(keys).nunique().astype(float),
    })
    Hq = H[q]
    kq = [Hq.user_id, Hq.tv_show_id]
    g["weeks_q"] = Hq.iso_week.groupby(kq).nunique().astype(float)
    g["last_q"] = Hq.airing_start.groupby(kq).max()
    g["weeks_q"] = g.weeks_q.fillna(0.0)
    return g.reset_index()


def exposure_stats(t):
    """Per (user, show) A_train_us / A_target_us on the user's channels for that show; per show global counts."""
    lab = t.lab
    a_tr = lab[(lab.start_time >= t.hist_start) & (lab.start_time < t.hist_end)]
    a_tg = lab[(lab.start_time >= t.tgt_start) & (lab.start_time < t.tgt_end)]
    c_tr = a_tr.groupby(["tv_show_id", "channel_id"]).size().rename("a_train")
    c_tg = a_tg.groupby(["tv_show_id", "channel_id"]).size().rename("a_target")
    usc = t.train[["user_id", "tv_show_id", "channel_id"]].drop_duplicates()
    usc = usc.join(c_tr, on=["tv_show_id", "channel_id"]).join(c_tg, on=["tv_show_id", "channel_id"]).fillna(0)
    us = usc.groupby(["user_id", "tv_show_id"])[["a_train", "a_target"]].sum().reset_index()
    show = pd.DataFrame({"a_target_glob": a_tg.groupby("tv_show_id").size(), "a_train_glob": a_tr.groupby("tv_show_id").size(),
                         "n_channels_target": a_tg.groupby("tv_show_id").channel_id.nunique(),
                         "dur_median_target": a_tg.groupby("tv_show_id").duration.median()})
    return us, show


# ---------------------------------------------------------------------------- source rankings
def list_ranks(df, score, users):
    """Ranks within user for a long table (user_id, tv_show_id, score, secs): score desc, secs desc, id asc."""
    d = df[df[score] > 0].sort_values(["user_id", score, "secs", "tv_show_id"], ascending=[True, False, False, True],
                                       kind="stable")
    d = d[d.user_id.isin(users)]
    return pd.DataFrame({"user_id": d.user_id.values, "tv_show_id": d.tv_show_id.values,
                         "rank": d.groupby("user_id").cumcount().values + 1})


def dense_ranks(S, batch=500):
    """Rank matrix (int16, 0 = not ranked because score <= 0); stable: ties keep the column order."""
    R = np.zeros(S.shape, np.int16)
    ar = np.arange(1, S.shape[1] + 1, dtype=np.int16)
    for s in range(0, len(S), batch):
        B = S[s:s + batch]
        order = np.argsort(-B, axis=1, kind="stable")
        rk = np.empty(order.shape, np.int16)
        np.put_along_axis(rk, order, np.broadcast_to(ar, order.shape), axis=1)
        rk[B <= 0] = 0
        R[s:s + batch] = rk
    return R


def sources(t):
    """All source scores/ranks for the task's users. Returns dict with timing."""
    out, tm = {}, {}
    users = t.users
    t0 = time.perf_counter()
    hs = history_stats(t)
    ex_us, ex_show = exposure_stats(t)
    hs = hs.merge(ex_us, on=["user_id", "tv_show_id"], how="left").fillna({"a_train": 0, "a_target": 0})
    hs["a_target_glob"] = hs.tv_show_id.map(ex_show.a_target_glob).fillna(0)
    inpool = hs[hs.tv_show_id.isin(t.pool)].copy()
    inpool["expo"] = np.where(inpool.a_target > 0, inpool.a_target, 0.1 * inpool.a_target_glob)
    inpool["pfx"] = inpool.softfrac * (1 + inpool.expo) ** 0.25
    out["pf_list"] = list_ranks(inpool, "softfrac", users)
    out["pfx_list"] = list_ranks(inpool, "pfx", users)
    out["hs"], out["ex_show"] = hs, ex_show
    tm["history_exposure_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    g = build_graph(t.train, "raw", "softfrac")
    P, has = personalisation(g, users)
    R, info = ppr(P, g.T, 0.85)
    node = {int(x): i for i, x in enumerate(g.shows)}
    cols = np.array([node.get(int(x), -1) for x in t.pool])
    pen = np.maximum(g.deg, 1).astype(float) ** 0.5
    S = np.zeros((len(users), len(t.pool)))
    ok = cols >= 0
    S[:, ok] = R[:, cols[ok]] / pen[cols[ok]]
    S[~has] = 0
    out["ppr_S"], out["ppr_R"] = S, dense_ranks(S)
    out["ppr_info"] = {"nodes": len(g.shows), "edges": g.n_edges, "iterations": info["iterations_per_batch"]}
    del R, P, g
    tm["ppr_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    it = Items(t.meta, "F4", {"category": 0.5, "genres": 1.0})
    Pc, hc, _ = profiles(it, t.train, users, "count08", None, t.hist_end)
    Sc = np.round(cosine_scores(Pc, it, t.pop), ROUND)
    Sc[~hc] = 0
    it2 = Items(t.meta, "F4")
    Pp, hp, _ = profiles(it2, t.train, users, "count08", 4, t.hist_end)
    Sp = np.round(0.65 * cosine_scores(Pp, it2, t.pop) + 0.35 * t.popnorm[None, :], ROUND)
    Sp[~hp] = 0
    out.update({"cbf_S": Sc, "cbf_R": dense_ranks(Sc), "cbfpop_S": Sp, "cbfpop_R": dense_ranks(Sp), "cbf_items": it})
    tm["cbf_seconds"] = time.perf_counter() - t0
    out["timing"] = tm
    return out


# ---------------------------------------------------------------------------- candidates + features
def build(t, kmax=KMAX):
    """Candidate superset (top-kmax of every source) with features and labels. Returns (df, sources, timing)."""
    t_all = time.perf_counter()
    src = sources(t)
    users = t.users
    uidx = pd.Series(np.arange(len(users)), index=users.astype(np.uint64))
    pool_id_col = {int(x): j for j, x in enumerate(t.pool)}      # PPR matrix columns: pool sorted by id
    pop_col = {int(x): j for j, x in enumerate(t.pop)}           # CBF matrices + pop rank: pop order

    t0 = time.perf_counter()
    parts = [src["pf_list"].query("rank <= @kmax")[["user_id", "tv_show_id"]],
             src["pfx_list"].query("rank <= @kmax")[["user_id", "tv_show_id"]]]
    for key, colmap in [("ppr_R", t.pool), ("cbf_R", t.pop), ("cbfpop_R", t.pop)]:
        Rm = src[key]
        r, c = np.nonzero((Rm > 0) & (Rm <= kmax))
        parts.append(pd.DataFrame({"user_id": users[r], "tv_show_id": np.asarray(colmap)[c]}))
    parts.append(pd.DataFrame({"user_id": np.repeat(users, kmax), "tv_show_id": np.tile(t.pop[:kmax], len(users))}))
    C = pd.concat(parts, ignore_index=True).drop_duplicates().sort_values(["user_id", "tv_show_id"], ignore_index=True)
    C["user_id"] = C.user_id.astype(np.uint64)
    ui = uidx.reindex(C.user_id.values).to_numpy()
    ci = C.tv_show_id.map(pool_id_col).to_numpy()
    cp = C.tv_show_id.map(pop_col).to_numpy()
    for name, lst in [("pf", src["pf_list"]), ("pfx", src["pfx_list"])]:
        C = C.merge(lst.rename(columns={"rank": f"rank_{name}"}), on=["user_id", "tv_show_id"], how="left")
    C["rank_ppr"] = src["ppr_R"][ui, ci].astype(float)
    C["rank_cbf"] = src["cbf_R"][ui, cp].astype(float)
    C["rank_cbfpop"] = src["cbfpop_R"][ui, cp].astype(float)
    C["rank_pop"] = cp + 1.0
    for s in SRC:
        C[f"rank_{s}"] = C[f"rank_{s}"].replace(0, np.nan)
    tm_cand = time.perf_counter() - t0

    t0 = time.perf_counter()
    hw, tw = t.hist_weeks, t.tgt_weeks
    hs = src["hs"]
    C = C.merge(hs.drop(columns=["a_target_glob"]), on=["user_id", "tv_show_id"], how="left")
    newm = C.softfrac.isna()
    for c in ["softfrac", "q", "c05", "secs", "n_air", "weeks_any", "weeks_q", "n_channels", "a_train", "a_target"]:
        C[c] = C[c].fillna(0.0)
    ut = hs.groupby("user_id").agg(u_softfrac=("softfrac", "sum"), u_q=("q", "sum"), u_secs=("secs", "sum"),
                                   u_shows=("softfrac", "size"), u_q_shows=("q", lambda x: float((x > 0).sum())))
    cat = t.meta.category
    news = hs.assign(n=hs.tv_show_id.map(cat).eq("Инфо")).query("n").groupby("user_id").secs.sum()
    ut["u_news"] = news.reindex(ut.index).fillna(0.0)
    C = C.merge(ut, left_on="user_id", right_index=True, how="left")
    hs_r = hs[["user_id", "tv_show_id", "softfrac", "q"]].copy()
    hs_r["rk_s"] = hs_r.groupby("user_id").softfrac.rank(ascending=False, method="min")
    hs_r["rk_q"] = hs_r.q.where(hs_r.q > 0).groupby(hs_r.user_id).rank(ascending=False, method="min")
    C = C.merge(hs_r[["user_id", "tv_show_id", "rk_s", "rk_q"]], on=["user_id", "tv_show_id"], how="left")

    F = pd.DataFrame({"user_id": C.user_id.values, "tv_show_id": C.tv_show_id.values})
    for s in SRC:
        F[f"rank_{s}"] = C[f"rank_{s}"].values
    # raw values kept for verification (not model features)
    for c in ["softfrac", "q", "c05", "secs", "weeks_q", "weeks_any", "a_train", "a_target"]:
        F[f"raw_{c}"] = C[c].values
    F["hs_softfrac_pw"] = C.softfrac / hw
    F["hs_q_pw"] = C.q / hw
    F["hs_c05_pw"] = C.c05 / hw
    F["hs_hours_pw"] = C.secs / 3600 / hw
    F["hs_share_softfrac"] = (C.softfrac / C.u_softfrac).fillna(0.0)
    F["hs_share_q"] = (C.q / C.u_q.where(C.u_q > 0)).fillna(0.0)
    F["hs_rank_inv_softfrac"] = (1 / C.rk_s).fillna(0.0)
    F["hs_rank_inv_q"] = (1 / C.rk_q).fillna(0.0)
    F["hs_mean_frac"] = (C.softfrac / C.n_air.where(C.n_air > 0)).fillna(0.0)
    F["hs_n_channels"] = C.n_channels
    wk = pd.Timedelta(weeks=1)
    F["rh_weeks_q_share"] = C.weeks_q / hw
    F["rh_weeks_any_share"] = C.weeks_any / hw
    F["rh_rec_q"] = ((t.hist_end - C.last_q) / wk / hw).fillna(1.0).astype(float)
    F["rh_rec_any"] = ((t.hist_end - C.last_any) / wk / hw).fillna(1.0).astype(float)
    F["rh_first_seen"] = ((C.first_any - t.hist_start) / wk / hw).fillna(1.0).astype(float)
    ex_show = src["ex_show"]
    a_tg_g = C.tv_show_id.map(ex_show.a_target_glob).fillna(0.0)
    a_tr_g = C.tv_show_id.map(ex_show.a_train_glob).fillna(0.0)
    F["raw_a_target_glob"] = a_tg_g.values
    F["ex_a_target_us_pw"] = C.a_target / tw
    F["ex_a_train_us_pw"] = C.a_train / hw
    F["ex_a_target_glob_pw"] = a_tg_g / tw
    F["ex_a_train_glob_pw"] = a_tr_g / hw
    for c in ["a_target_us", "a_train_us", "a_target_glob", "a_train_glob"]:
        F[f"ex_log_{c}_pw"] = np.log1p(F[f"ex_{c}_pw"])
    F["raw_ex_ratio_us"] = C.a_target / np.maximum(C.a_train, 1)
    F["ex_rate_ratio_us"] = F.ex_a_target_us_pw / np.maximum(F.ex_a_train_us_pw, 1 / hw)
    F["ex_zero_us"] = (C.a_target == 0).astype(float)
    F["ex_other_channels_only"] = ((C.a_target == 0) & (a_tg_g > 0)).astype(float)
    F["ex_glob_rate_ratio"] = F.ex_a_target_glob_pw / np.maximum(F.ex_a_train_glob_pw, 1 / hw)
    sp_ = src["ppr_S"][ui, ci]
    umax = pd.Series(sp_).groupby(ui).transform("max").to_numpy()
    F["ppr_score_rel"] = np.where(umax > 0, sp_ / np.where(umax > 0, umax, 1), 0.0)
    F["ppr_rank_inv"] = (1 / C.rank_ppr).fillna(0.0).values
    F["cbf_cos"] = src["cbf_S"][ui, cp]
    F["cbfpop_score"] = src["cbfpop_S"][ui, cp]
    F["cbf_rank_inv"] = (1 / C.rank_cbf).fillna(0.0).values
    F["cbfpop_rank_inv"] = (1 / C.rank_cbfpop).fillna(0.0).values
    F["cbf_maxcos_top5"] = maxcos_top5(F, hs, src["cbf_items"])
    F["pop_norm"] = t.popnorm[cp]
    F["pop_rank_inv"] = 1 / (cp + 1.0)
    q_users = t.train[t.train.frac >= THR].groupby("tv_show_id").user_id.nunique()
    F["pop_hist"] = C.tv_show_id.map(q_users).fillna(0).values / t.train.user_id.nunique()
    F["new_show"] = newm.astype(float).values
    F["never_watched"] = (~C.tv_show_id.isin(set(t.train.tv_show_id.unique().tolist()))).astype(float).values
    F["u_q_shows_pw"] = C.u_q_shows.fillna(0.0).values / hw
    F["u_hours_pw"] = C.u_secs.fillna(0.0).values / 3600 / hw
    F["u_activity_bucket"] = pd.cut(C.u_q_shows.fillna(0.0), [-1, 4, 9, 19, 49, np.inf], labels=False).astype(float).values
    F["u_mean_softfrac"] = (C.u_softfrac / C.u_shows).fillna(0.0).values
    F["u_news_share"] = (C.u_news / C.u_secs.where(C.u_secs > 0)).fillna(0.0).values
    sc = C.tv_show_id.map(cat)
    for c in CATS:
        F[f"cat_{CAT_EN[c]}"] = (sc == c).astype(float).values
    F["sh_dur_median_min"] = C.tv_show_id.map(ex_show.dur_median_target).fillna(0.0).values / 60
    F["sh_n_channels_target"] = C.tv_show_id.map(ex_show.n_channels_target).fillna(0.0).values
    tr_rank = {(u, s): r for u, v in t.truth.items() for r, s in enumerate(v, 1)}
    rk = np.array([tr_rank.get((int(u), int(s)), 0) for u, s in zip(F.user_id, F.tv_show_id)])
    F["graded"] = np.where(rk > 0, 6 - rk, 0)
    F["label"] = (rk > 0).astype(int)
    F[FEATURES] = F[FEATURES].astype(np.float64)
    tm_feat = time.perf_counter() - t0
    timing = {**src["timing"], "candidates_seconds": tm_cand, "features_seconds": tm_feat,
              "total_seconds": time.perf_counter() - t_all}
    return F, src, timing


def maxcos_top5(F, hs, items):
    """Max cosine (cbf_pure item vectors) between a candidate and the user's top-5 history shows by softfrac."""
    top = hs.sort_values(["user_id", "softfrac", "tv_show_id"], ascending=[True, False, True]).groupby("user_id").head(5)
    top = top[top.tv_show_id.isin(items.row)]
    hist_rows = {int(u): [items.row[int(s)] for s in g.tv_show_id] for u, g in top.groupby("user_id")}
    out = np.zeros(len(F))
    X = items.X
    for u, idx in F.groupby("user_id", sort=False).indices.items():
        h = hist_rows.get(int(u))
        if not h:
            continue
        rows = [items.row.get(int(s), -1) for s in F.tv_show_id.values[idx]]
        ok = np.array([r >= 0 for r in rows])
        if ok.any():
            sims = (X[[r for r in rows if r >= 0]] @ X[h].T).toarray()
            out[idx[ok]] = sims.max(axis=1)
    return out


def in_candidates(F, ks):
    """Boolean mask: row is in the top-ks[source] of at least one source."""
    m = np.zeros(len(F), bool)
    for s, k in ks.items():
        m |= (F[f"rank_{s}"] <= k).to_numpy()
    return m
