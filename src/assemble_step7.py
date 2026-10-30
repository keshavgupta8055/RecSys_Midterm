"""Step 7: assemble the final comparison, report-ready example tables, limitations evidence and the figure index
from files that already exist (no model is trained or re-run). Run: python -m src.assemble_step7
"""
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import io_utils as io
from src.evaluate import ap_at_k
from src.hybrid_features import FEATURES

OUT, FIG = io.ROOT / "outputs", io.ROOT / "outputs" / "figures"
DPI = 150
CAT_EN = {"Детям": "Kids", "Сериалы": "Series", "Фильмы": "Films", "Развлечения": "Entertainment", "Инфо": "Info",
          "Познавательное": "Educational", "Другое": "Other", "Спорт": "Sport"}
S = {k: json.loads((OUT / f"step{k}_summary.json").read_text()) for k in (3, 4, 5, 6)}
r = lambda f: pd.read_csv(OUT / f)
issues = []


def log(*a):
    print(*a, flush=True)


# ====================================================================== consistency of accuracy numbers
p1 = r("hybrid_p1_results_val.csv").set_index("method")
b3 = r("baseline_results_val.csv").set_index("method")
pv = r("ppr_vs_baselines_val.csv")
cv = r("cbf_vs_baselines_val.csv")
ex = r("exposure_baselines_val.csv").set_index("method")
checks = []
for pop in ["ALL", "SUB"]:
    checks += [("pop_recent", pop, p1.loc["pop_recent", f"map5_{pop}"], b3.loc["pop_recent", f"map5_{pop}"], "baseline_results_val.csv"),
               ("PF_softfrac", pop, p1.loc["PF_softfrac", f"map5_{pop}"], b3.loc["PF_softfrac", f"map5_{pop}"], "baseline_results_val.csv"),
               ("PPR", pop, p1.loc["PPR", f"map5_{pop}"], pv[(pv.population == pop) & (pv.method == "PPR")].map5.iloc[0], "ppr_vs_baselines_val.csv"),
               ("PPR", pop, p1.loc["PPR", f"map5_{pop}"], S[4]["ppr_val"][pop]["map5"], "step4_summary.json"),
               ("cbf_pure", pop, p1.loc["cbf_pure", f"map5_{pop}"], S[5]["cbf_pure_val"][pop]["map5"], "step5_summary.json"),
               ("cbf_pop", pop, p1.loc["cbf_pop", f"map5_{pop}"], S[5]["cbf_pop_val"][pop]["map5"], "step5_summary.json"),
               ("cbf_pop", pop, p1.loc["cbf_pop", f"map5_{pop}"], cv[(cv.population == pop) & (cv.method == "cbf_pop")].map5.iloc[0], "cbf_vs_baselines_val.csv"),
               ("M0", pop, p1.loc["M0_PF_exposure", f"map5_{pop}"], ex.loc["PF_soft_exposure_g0.25", f"map5_{pop}"], "exposure_baselines_val.csv"),
               ("M0", pop, p1.loc["M0_PF_exposure", f"map5_{pop}"], S[5]["best_exposure"][f"map5_{pop}"], "step5_summary.json")]
for m, pop, a, b, src in checks:
    if abs(a - b) > 1e-12:
        issues.append(f"MAP@5 {m} {pop}: hybrid_p1_results_val.csv {a} vs {src} {b}")
if issues:
    raise SystemExit("INCONSISTENT NUMBERS - stopping:\n" + "\n".join(issues))
log(f"[check] {len(checks)} cross-file MAP@5 comparisons agree to 1e-12")

# ====================================================================== A final comparison


def pdiff(df, comp, pop, neg=False, cols=("mean_diff", "diff_ci_lo", "diff_ci_hi")):
    row = df[(df.population == pop) & (df.comparison == comp)].iloc[0]
    m, lo, hi = (row[c] for c in cols)
    return (-m, -hi, -lo) if neg else (m, lo, hi)


bpd = r("baseline_paired_diffs_val.csv")
hpd = r("hybrid_p1_paired_diffs_val.csv")
DIFF = {}  # (method, vs, pop) -> (mean, lo, hi, source)
for pop in ["ALL", "SUB"]:
    DIFF[("pop_recent", "PF", pop)] = (*pdiff(bpd, "PF_softfrac - pop_recent", pop, True, ("mean_diff", "ci_lo", "ci_hi")),
                                       "baseline_paired_diffs_val.csv (sign flipped)")
    DIFF[("PPR", "PF", pop)] = (*pdiff(pv, "PPR - PF_softfrac", pop), "ppr_vs_baselines_val.csv")
    DIFF[("cbf_pure", "PF", pop)] = (*pdiff(cv, "cbf_pure - PF_softfrac", pop), "cbf_vs_baselines_val.csv")
    DIFF[("cbf_pop", "PF", pop)] = (*pdiff(cv, "cbf_pop - PF_softfrac", pop), "cbf_vs_baselines_val.csv")
    DIFF[("M0_PF_exposure", "PF", pop)] = (*pdiff(cv, "PF_exposure - PF_softfrac", pop), "cbf_vs_baselines_val.csv")
    DIFF[("PF_softfrac", "M0", pop)] = (*pdiff(cv, "PF_exposure - PF_softfrac", pop, True), "cbf_vs_baselines_val.csv (sign flipped)")
    for m in ["M1_fusion", "M2_logreg", "M3_lambdarank", "M4_lgb_binary"]:
        for vs, key in [("PF_softfrac", "PF"), ("M0_PF_exposure", "M0")]:
            row = hpd[(hpd.model == m) & (hpd.vs == vs) & (hpd.population == pop)].iloc[0]
            DIFF[(m, key, pop)] = (row.mean_diff, row.ci_lo, row.ci_hi, "hybrid_p1_paired_diffs_val.csv")
hc = r("hybrid_cost.csv").set_index("stage")
pc = r("ppr_cost.csv").set_index("split")
cc = r("cbf_cost.csv").set_index(["split", "variant"])
brt = r("baseline_runtimes_val.csv").set_index("method").runtime_s
fb = "feature build (all sources + candidates + features)"
hcv = lambda task, col="seconds": float(hc[hc.task == task].loc[fb, col])
feat_iv = hcv("inner") + hcv("val")
NMS = "not measured separately"
META = {
    "pop_recent": ("baseline", "Step 3", "train weeks 11-20 -> val weeks 21-30", "none (distinct qualifying users, last 4 training weeks)",
                   f"{brt['pop_recent']} s score+evaluate (baseline_runtimes_val.csv)", NMS, brt["pop_recent"], S[3]["peak_memory_mb"]),
    "PF_softfrac": ("baseline", "Step 3", "train weeks 11-20 -> val weeks 21-30", "none (sum of frac per history show)",
                    f"{brt['PF_softfrac']} s score+evaluate (baseline_runtimes_val.csv)", NMS, brt["PF_softfrac"], S[3]["peak_memory_mb"]),
    "PPR": ("Method A", "Step 4", "train weeks 11-20 -> val weeks 21-30",
            f"no learned weights; d 0.85, beta 0.5, raw edges, softfrac seeds chosen from 48 configs; graph {S[4]['graph_val']['nodes']} nodes",
            f"graph {pc.loc['val', 'graph_build_seconds']} s + PPR {pc.loc['val', 'ppr_seconds_all_users']} s (ppr_cost.csv)",
            f"graph {pc.loc['test', 'graph_build_seconds']} s + PPR {pc.loc['test', 'ppr_seconds_all_users']} s",
            pc.loc["val", "graph_build_seconds"] + pc.loc["val", "ppr_seconds_all_users"], S[4]["peak_memory_mb"]),
    "cbf_pure": ("Method B", "Step 5", "train weeks 11-20 -> val weeks 21-30",
                 "no learned weights; F4 blocks, count08 profile, category weight 0.5 (102-config grid); items x features 8,234 x 23,841",
                 f"items {cc.loc[('val', 'cbf_pure'), 'item_build_seconds']} s + profile/score/rank {cc.loc[('val', 'cbf_pure'), 'profile_scoring_ranking_seconds']} s (cbf_cost.csv)",
                 f"items {cc.loc[('test', 'cbf_pure'), 'item_build_seconds']} s + {cc.loc[('test', 'cbf_pure'), 'profile_scoring_ranking_seconds']} s",
                 cc.loc[("val", "cbf_pure"), "item_build_seconds"] + cc.loc[("val", "cbf_pure"), "profile_scoring_ranking_seconds"], S[5]["peak_memory_mb"]),
    "cbf_pop": ("Method B (+popularity prior)", "Step 5", "train weeks 11-20 -> val weeks 21-30",
                "lam 0.35, 4-week decay, count08 (102-config grid)",
                f"items {cc.loc[('val', 'cbf_pop'), 'item_build_seconds']} s + profile/score/rank {cc.loc[('val', 'cbf_pop'), 'profile_scoring_ranking_seconds']} s (cbf_cost.csv)",
                f"items {cc.loc[('test', 'cbf_pop'), 'item_build_seconds']} s + {cc.loc[('test', 'cbf_pop'), 'profile_scoring_ranking_seconds']} s",
                cc.loc[("val", "cbf_pop"), "item_build_seconds"] + cc.loc[("val", "cbf_pop"), "profile_scoring_ranking_seconds"], S[5]["peak_memory_mb"]),
    "M0_PF_exposure": ("baseline (PF-exposure)", "Step 5 (re-derived in Step 6)", "train weeks 11-20 -> val weeks 21-30",
                       "gamma 0.25 (chosen from 10 exposure variants)",
                       f"history + exposure tables {hcv('val', 'history_exposure_seconds')} s (hybrid_cost.csv, inside the Step 6 val build)",
                       f"history + exposure tables {hcv('test', 'history_exposure_seconds')} s",
                       hcv("val", "history_exposure_seconds"), S[6]["peak_memory_mb"]),
    "M1_fusion": ("non-learned hybrid", "Step 6", "P1: weights tuned on inner (weeks 11-15 -> 16-20), evaluated on val",
                  f"5 fusion weights (z-score fusion {S[6]['M1']['zscore']['w_pfx']} x PF_exposure, {S[6]['M1']['zscore']['w_cbfpop']} x cbf_pop) from 4,092 configs",
                  f"feature builds inner {hcv('inner')} s + val {hcv('val')} s; tuning grid {float(hc.loc['M1 fusion grid (4092 configs)', 'seconds'])} s; scoring {NMS}",
                  f"test feature build {hcv('test')} s; scoring {NMS}", feat_iv, S[6]["peak_memory_mb"]),
    "M2_logreg": ("hybrid (learned)", "Step 6", "P1: trained on inner, evaluated on val",
                  f"{len(FEATURES)} coefficients + intercept, C {S[6]['selected_hyperparameters']['M2_logreg']['C']}",
                  f"features {feat_iv:.1f} s + fit {float(hc.loc['M2 final fit', 'seconds'])} s + inference {float(hc.loc['M2 inference', 'seconds'])} s; CV {float(hc.loc['M2 CV (2 x 3 fits)', 'seconds'])} s",
                  f"test features {hcv('test')} s; P3 fit and inference {NMS} (all models together: {float(hc.loc['P3 final fits (M2, M3, M4)', 'seconds'])} s / {float(hc.loc['P3 inference (all models)', 'seconds'])} s)",
                  feat_iv + float(hc.loc["M2 final fit", "seconds"]) + float(hc.loc["M2 inference", "seconds"]), S[6]["peak_memory_mb"]),
    "M3_lambdarank": ("hybrid (learned) - FINAL", "Step 6", "P1: trained on inner, evaluated on val",
                      f"{S[6]['n_estimators']['M3_lambdarank']} trees, 31 leaves, lr 0.1, {len(FEATURES)} features",
                      f"features {feat_iv:.1f} s + fit {float(hc.loc['M3_lambdarank final fit (early stopping + refit)', 'seconds'])} s + inference {float(hc.loc['M3_lambdarank inference', 'seconds'])} s; CV {float(hc.loc['M3_lambdarank CV (12 configs x 3 folds)', 'seconds'])} s",
                      f"test features {hcv('test')} s; P3 fit and inference {NMS} (all models together: {float(hc.loc['P3 final fits (M2, M3, M4)', 'seconds'])} s / {float(hc.loc['P3 inference (all models)', 'seconds'])} s)",
                      feat_iv + float(hc.loc["M3_lambdarank final fit (early stopping + refit)", "seconds"]) + float(hc.loc["M3_lambdarank inference", "seconds"]),
                      S[6]["peak_memory_mb"]),
    "M4_lgb_binary": ("hybrid (learned)", "Step 6", "P1: trained on inner, evaluated on val",
                      f"{S[6]['n_estimators']['M4_lgb_binary']} trees, 31 leaves, lr 0.03, {len(FEATURES)} features",
                      f"features {feat_iv:.1f} s + fit {float(hc.loc['M4_lgb_binary final fit (early stopping + refit)', 'seconds'])} s + inference {float(hc.loc['M4_lgb_binary inference', 'seconds'])} s; CV {float(hc.loc['M4_lgb_binary CV (12 configs x 3 folds)', 'seconds'])} s",
                      f"test features {hcv('test')} s; P3 fit and inference {NMS}",
                      feat_iv + float(hc.loc["M4_lgb_binary final fit (early stopping + refit)", "seconds"]) + float(hc.loc["M4_lgb_binary inference", "seconds"]),
                      S[6]["peak_memory_mb"]),
}
rows = []
for m, (role, step, proto, params, cval, ctest, tot, peak) in META.items():
    row = {"method": m, "role": role, "evaluated_in": step, "protocol": proto}
    for pop in ["ALL", "SUB"]:
        row.update({f"map5_{pop}": p1.loc[m, f"map5_{pop}"], f"ci_lo_{pop}": p1.loc[m, f"ci_lo_{pop}"], f"ci_hi_{pop}": p1.loc[m, f"ci_hi_{pop}"]})
    for vs in ["PF", "M0"]:
        for pop in ["ALL", "SUB"]:
            d = DIFF.get((m, vs, pop))
            if (m == "PF_softfrac" and vs == "PF") or (m == "M0_PF_exposure" and vs == "M0"):
                row[f"diff_vs_{vs}_{pop}"] = "reference"
            elif d is None:
                row[f"diff_vs_{vs}_{pop}"] = "not computed in Steps 3-6 (per-user APs not stored)"
            else:
                row[f"diff_vs_{vs}_{pop}"] = f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"
    row.update({"learned_parameters_or_config": params, "cost_val": cval, "cost_test": ctest,
                "total_seconds_val_for_fig38": round(tot, 2), "peak_memory_mb_whole_step": peak,
                "cost_source_note": "timings from the cost CSVs = LAST of the two reproducibility runs of that step"})
    rows.append(row)
final = pd.DataFrame(rows)
final.to_csv(OUT / "final_method_comparison.csv", index=False)
with pd.option_context("display.width", 220, "display.max_colwidth", 40):
    log(final[["method", "role", "map5_ALL", "ci_lo_ALL", "ci_hi_ALL", "map5_SUB", "diff_vs_PF_ALL", "diff_vs_M0_ALL",
               "total_seconds_val_for_fig38"]].round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
colors = {"baseline": "#9a9a9a", "Method A": "#4a9a5b", "Method B": "#c0392b", "Method B (+popularity prior)": "#e08a2c",
          "baseline (PF-exposure)": "#7d4fa1", "non-learned hybrid": "#17becf", "hybrid (learned)": "#8c564b",
          "hybrid (learned) - FINAL": "#d62728"}
for _, rr in final.iterrows():
    ax.errorbar(rr.total_seconds_val_for_fig38, rr.map5_ALL, yerr=[[rr.map5_ALL - rr.ci_lo_ALL], [rr.ci_hi_ALL - rr.map5_ALL]],
                fmt="o", color=colors[rr.role], capsize=3, ms=7)
    off = {"M1_fusion": (-62, -14), "M2_logreg": (8, -16), "M3_lambdarank": (-80, 10), "M4_lgb_binary": (8, 6)}.get(rr.method, (6, 4))
    ax.annotate(rr.method, (rr.total_seconds_val_for_fig38, rr.map5_ALL), textcoords="offset points", xytext=off, fontsize=8,
                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "grey"} if rr.method in off else None)
ax.set_xscale("log")
ax.set(title="Accuracy vs measured cost on validation (MAP@5 ALL, 95% CI)",
       xlabel="Measured seconds to produce val recommendations (log; excludes hyper-parameter search; see CSV)",
       ylabel="Validation MAP@5 (ALL, n = 4,030)")
for role, c in colors.items():
    ax.plot([], [], "o", color=c, label=role)
ax.legend(fontsize=7, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "38_accuracy_vs_cost.png", dpi=DPI)
plt.close(fig)

# ====================================================================== B example tables + facts
eh = r("example_users_hybrid.csv")
ep = r("example_users_ppr.csv")
ec = r("example_users_cbf.csv")
pr = r("example_users_profiles.csv")
exu = json.loads((OUT / "example_users.json").read_text())
LISTS = {"pop_recent": "pop_recent", "PF_softfrac": "PF_softfrac", "PPR": "PPR", "cbf_pure": "cbf_pure", "cbf_pop": "cbf_pop",
         "M0 exposure": "PF_soft_exposure_g0.25 (M0)", "final hybrid M3": "hybrid (M3_lambdarank)"}


def lst(df, col, role, name):
    d = df[(df.role == role) & (df[col] == name)].sort_values("rank")
    return [int(x) for x in d.tv_show_id]


tab_rows, facts, md = [], [], []
for role in ["light", "heavy"]:
    uid = exu[role]["user_id"]
    L = {k: lst(eh, "list", role, v) for k, v in LISTS.items()}
    truth = lst(eh, "list", role, "truth")
    # cross-file assertions (lists already printed in earlier steps)
    assert L["PPR"] == lst(ep, "list", role, "PPR_full") == lst(ec, "section", role, "PPR")
    assert L["PF_softfrac"] == lst(ep, "list", role, "PF_softfrac") == lst(ec, "section", role, "PF_softfrac") == lst(pr, "section", role, "val_top5_PF_softfrac")
    assert L["pop_recent"] == lst(ep, "list", role, "pop_recent") == lst(ec, "section", role, "pop_recent") == lst(pr, "section", role, "val_top5_pop_recent")
    assert L["cbf_pure"] == lst(ec, "section", role, "cbf_pure") and L["cbf_pop"] == lst(ec, "section", role, "cbf_pop")
    assert L["M0 exposure"] == lst(ec, "section", role, "PF_exposure (PF_soft_exposure_g0.25)")
    assert truth == lst(pr, "section", role, "truth_val")
    info = pd.concat([eh[["tv_show_id", "title", "category"]], ec[["tv_show_id", "title", "category"]].dropna(),
                      pr[["tv_show_id", "title", "category"]]]).dropna().drop_duplicates("tv_show_id")
    info = info.assign(tv_show_id=info.tv_show_id.astype(int)).set_index("tv_show_id")
    eh_r = eh[(eh.role == role) & (eh.list != "overlap")]
    hist_flag = {int(x): bool(f) for x, f in zip(eh_r.tv_show_id, eh_r.in_history)}
    tr10 = pr[(pr.role == role) & (pr.section == "train_top10")].sort_values("rank")
    for k, l in L.items():
        for i, x in enumerate(l, 1):
            tab_rows.append({"role": role, "user_id": uid, "method": k, "rank": i, "tv_show_id": x, "title": info.title.get(x),
                             "category_en": CAT_EN.get(info.category.get(x), info.category.get(x)), "hit": x in truth,
                             "in_training_history": hist_flag.get(x)})
    for i, x in enumerate(truth, 1):
        tab_rows.append({"role": role, "user_id": uid, "method": "validation truth", "rank": i, "tv_show_id": x,
                         "title": info.title.get(x), "category_en": CAT_EN.get(info.category.get(x))})
    for t_ in tr10.itertuples():
        tab_rows.append({"role": role, "user_id": uid, "method": "training top-10", "rank": int(t_.rank), "tv_show_id": int(t_.tv_show_id),
                         "title": t_.title, "category_en": CAT_EN.get(t_.category, t_.category),
                         "qualifying_count": int(t_.qualifying_count), "watched_hours": t_.watched_hours})
    # facts
    aps = {k: ap_at_k(l, set(truth)) for k, l in L.items()}
    for k, l in L.items():
        facts.append({"role": role, "user_id": uid, "method": k, "hits": sum(x in truth for x in l), "ap5": aps[k],
                      "recommended_in_training_history": sum(bool(hist_flag.get(x)) for x in l)})
    ov = eh[(eh.role == role) & (eh.list == "overlap")][["title", "rank"]].rename(columns={"title": "pair", "rank": "common_ids"})
    pf_only = [x for x in L["PF_softfrac"] if x not in L["final hybrid M3"]]
    hy_only = [x for x in L["final hybrid M3"] if x not in L["PF_softfrac"]]
    hy = eh[(eh.role == role) & (eh.list == "hybrid (M3_lambdarank)")].set_index("tv_show_id")
    md.append(f"## {role} user {uid}\n")
    md.append(f"Selected by the Step 3 rule: {exu['rule']}. This user has {exu[role]['distinct_qualifying_training_shows']} "
              f"distinct qualifying training shows.\n")
    md.append("| method | hits | AP@5 (val) | recommended shows already in training history |\n|---|---|---|---|")
    for f_ in [f for f in facts if f["role"] == role]:
        md.append(f"| {f_['method']} | {f_['hits']} | {f_['ap5']:.4f} | {f_['recommended_in_training_history']} |")
    md.append(f"\nShows that differ between PF_softfrac and the final hybrid: {len(hy_only)} "
              f"(in PF only: {pf_only}; in hybrid only: {hy_only}).")
    for x in hy_only:
        h_ = hy.loc[x]
        md.append(f"- hybrid-only show {x} ({info.title.get(x)}, {CAT_EN.get(info.category.get(x))}), hit = {h_.hit}: "
                  f"top-3 SHAP (M3) {h_.shap1}; {h_.shap2}; {h_.shap3}")
    for x in pf_only:
        md.append(f"- PF-only show {x} ({info.title.get(x)}), hit = {x in truth}: no SHAP stored (the show is not in the hybrid top-5, "
                  f"and Step 6 stored contributions only for hybrid top-5 items)")
    md.append("\nOverlaps (number of common ids) between method pairs (from example_users_hybrid.csv):\n")
    md.append("| pair | common ids |\n|---|---|")
    for o in ov.itertuples():
        md.append(f"| {o.pair} | {int(o.common_ids)} |")
    md.append("")
pd.DataFrame(tab_rows).to_csv(OUT / "report_example_tables.csv", index=False, encoding="utf-8-sig")
fdf = pd.DataFrame(facts)
log(fdf.round(4).to_string(index=False))
hyb_ap_logged = {"light": (0.55, 0.52), "heavy": (0.60, 0.60)}  # Step 6 log: AP hybrid vs M0
for role, (a, b) in hyb_ap_logged.items():
    f_ = fdf[fdf.role == role].set_index("method").ap5
    assert abs(f_["final hybrid M3"] - a) < 5e-5 and abs(f_["M0 exposure"] - b) < 5e-5, role

head = ["# Example-user facts (validation, from existing files)\n",
        "All numbers below come from outputs/example_users_*.csv (Steps 3-6) and are recomputed only with ap_at_k on "
        "those stored lists. **Two users are an illustration, not evidence: none of the per-user differences below is "
        "statistically significant; the significance statements in the report rest on the 4,030 / 1,150 validation users "
        "(final_method_comparison.csv).** No statement is made about user enjoyment or discovery value.\n"]
(OUT / "report_example_facts.md").write_text("\n".join(head + md), encoding="utf-8")

# markdown tables
mdt = ["# Report example tables (validation, weeks 21-30)\n",
       "Cells: tv_show_id, short title, English category. **Bold** = hit (show in the user's validation truth top-5). "
       "Lists are copied from outputs/example_users_hybrid.csv and asserted equal to the Step 3-5 example files.\n"]
tr_ = pd.DataFrame(tab_rows)


def short(t, n=26):
    t = str(t)
    return t if len(t) <= n else t[: n - 1] + "…"


for role in ["light", "heavy"]:
    d = tr_[tr_.role == role]
    mdt.append(f"## {role} user {exu[role]['user_id']}\n")
    mdt.append("| rank | " + " | ".join(LISTS) + " |")
    mdt.append("|---" * (len(LISTS) + 1) + "|")
    for k in range(1, 6):
        cells = []
        for m in LISTS:
            x = d[(d.method == m) & (d["rank"] == k)].iloc[0]
            c = f"{x.tv_show_id} {short(x.title)} ({x.category_en})"
            cells.append(f"**{c}**" if x.hit else c)
        mdt.append(f"| {k} | " + " | ".join(cells) + " |")
    mdt.append("\nValidation truth top-5: " + "; ".join(f"{x.tv_show_id} {short(x.title, 40)} ({x.category_en})"
                                                      for x in d[d.method == "validation truth"].itertuples()))
    mdt.append("\nTraining top-10 (weeks 11-20):\n\n| rank | tv_show_id | title | category | qualifying airings | watched hours |\n|---|---|---|---|---|---|")
    for x in d[d.method == "training top-10"].itertuples():
        mdt.append(f"| {x.rank} | {x.tv_show_id} | {short(x.title, 45)} | {x.category_en} | {int(x.qualifying_count)} | {x.watched_hours} |")
    mdt.append("")
(OUT / "report_example_tables.md").write_text("\n".join(mdt), encoding="utf-8")

for role, num in [("light", 39), ("heavy", 40)]:
    d = tr_[tr_.role == role]
    fig = plt.figure(figsize=(8.3, 7.4))
    ax = fig.add_axes([0.01, 0.42, 0.98, 0.5])
    ax.axis("off")
    cell, colr = [], []
    for k in range(1, 6):
        row_, crow_ = [], []
        for m in LISTS:
            x = d[(d.method == m) & (d["rank"] == k)].iloc[0]
            row_.append(f"{x.tv_show_id}\n{short(x.title, 17)}\n{x.category_en}")
            crow_.append("#b9e3c0" if x.hit else "white")
        cell.append(row_)
        colr.append(crow_)
    t = ax.table(cellText=cell, cellColours=colr, colLabels=[textwrap.fill(m, 11) for m in LISTS],
                 rowLabels=[str(k) for k in range(1, 6)], loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(6.6)
    t.scale(1, 3.1)
    fig.suptitle(f"Example user ({role}) {exu[role]['user_id']}: validation top-5 per method (green = hit)", fontsize=9.5, y=0.98)
    ax2 = fig.add_axes([0.01, 0.02, 0.98, 0.36])
    ax2.axis("off")
    tt = d[d.method == "validation truth"]
    t10 = d[d.method == "training top-10"]
    lines = ["Validation truth top-5:"] + [f"  {x.rank}. {x.tv_show_id}  {short(x.title, 46)}  ({x.category_en})" for x in tt.itertuples()]
    lines += ["", "Training top-10 (qualifying airings, weeks 11-20):"]
    lines += [f"  {x.rank}. {x.tv_show_id}  {short(x.title, 46)}  ({x.category_en}, {int(x.qualifying_count)})" for x in t10.itertuples()]
    ax2.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=7, family="DejaVu Sans")
    fig.savefig(FIG / f"{num}_example_user_{role}.png", dpi=DPI)
    plt.close(fig)

# ====================================================================== C limitations evidence
sched = r("schedule_summary.csv")
wta = r("watchtime_accounting.csv")
wq = r("weekly_qualifying.csv").set_index("iso_week")
days = r("viewlog_rows_per_day.csv")
cold = r("cbf_cold_items_val.csv")
ceil = r("ceiling_analysis_val.csv").set_index("population")
newo = r("cbf_new_only_val.csv")
la = r("hybrid_lists_analysis_val.csv").set_index("method")
fc = r("hybrid_failure_cases_val.csv")
unl = wta[(wta.group == "all_users") & wta.bucket.str.startswith("(ii)")].share.iloc[0]
full_days = pd.date_range("2020-03-09", "2020-07-26").strftime("%Y-%m-%d")
missing = sorted(set(full_days) - set(days.date.astype(str)))
cold_all = cold[(cold.method == "M3_lambdarank") | (cold.method == "PF_softfrac")]
n_cold = int(cold[(cold.method == "PF_softfrac") & (cold.population == "ALL")].cold_truth_pairs.iloc[0])
nn = lambda m, p: newo[(newo.method == m) & (newo.population == p)].map5_vs_truth_new.iloc[0]
grid_sizes = {"PPR": len(r("ppr_grid_val.csv")), "CBF": len(r("cbf_grid_val.csv")), "M1": len(r("hybrid_m1_fusion_grid_inner.csv")),
              "LightGBM": len(r("hybrid_model_selection_inner_cv.csv").query("model != 'M2_logreg'"))}
LIM = [
    ("Only a small share of schedule rows carries a tv_show_id",
     f"labelled share per schedule file: {', '.join(f'{x:.2%}' for x in sched.labelled_share)}", "schedule_summary.csv"),
    ("A large part of watch time falls on unlabelled airings (tv_show_id = 0) and is invisible to every method",
     f"{unl:.2%} of merged watched seconds (all users)", "watchtime_accounting.csv"),
    ("Label recipe: the competition's tie handling is unobservable; our tie-break is count, watched seconds, tv_show_id",
     "assumption (no number)", "outputs/PROTOCOL.md, src/labels.py"),
    ("Metric: the competition page only names MAP@5; the standard Kaggle AP@K definition is assumed",
     "assumption; unit-tested on 9 cases", "src/evaluate.py, outputs/test_evaluate_output.txt"),
    ("Validation target window contains weeks with about half the usual viewing",
     f"qualifying pairs weeks 25/26/27: {wq.loc[25, 'qualifying_pairs']:,} / {wq.loc[26, 'qualifying_pairs']:,} / {wq.loc[27, 'qualifying_pairs']:,} "
     f"vs week 24: {wq.loc[24, 'qualifying_pairs']:,} and week 28: {wq.loc[28, 'qualifying_pairs']:,}", "weekly_qualifying.csv"),
    ("A day is missing from the view log", f"days without any view row: {', '.join(missing)}", "viewlog_rows_per_day.csv"),
    ("Truth pairs on shows nobody watched in training cannot be recovered by history methods",
     f"{n_cold:,} of {5 * 4030:,} truth pairs ({n_cold / (5 * 4030):.2%}); recall of these by the final hybrid "
     f"{la.loc['M3_lambdarank', 'recall_cold_pairs']:.4f}", "cbf_cold_items_val.csv, hybrid_lists_analysis_val.csv"),
    ("Some truth shows did not even air in the training schedule",
     f"{ceil.loc['ALL', 'share_truth_shows_absent_from_11_20_schedule']:.2%} (ALL) / {ceil.loc['SUB', 'share_truth_shows_absent_from_11_20_schedule']:.2%} (SUB) of truth pairs",
     "ceiling_analysis_val.csv"),
    ("Ceiling of any history-only ranker", f"MAP@5 {ceil.loc['ALL', 'ceiling_map5_history_any_watch']:.4f} (ALL) / "
     f"{ceil.loc['SUB', 'ceiling_map5_history_any_watch']:.4f} (SUB)", "ceiling_analysis_val.csv"),
    ("The co-view graph is nearly complete, so PPR adds little beyond the user's own history",
     f"density {S[4]['graph_val']['density']:.4f}; PPR - PF_softfrac {DIFF[('PPR', 'PF', 'ALL')][0]:+.4f} ALL",
     "step4_summary.json, ppr_vs_baselines_val.csv"),
    ("Content-based filtering is weaker than popularity on unseen shows",
     f"MAP@5 vs truth_new: cbf_pure {nn('cbf_pure_new_only', 'ALL'):.4f}, cbf_pop {nn('cbf_pop_new_only', 'ALL'):.4f}, "
     f"pop_recent {nn('pop_recent_new_only', 'ALL'):.4f}", "cbf_new_only_val.csv"),
    ("The final re-ranker concentrates on few shows",
     f"{int(la.loc['M3_lambdarank', 'distinct_shows_recommended'])} distinct shows recommended on val "
     f"(M0: {int(la.loc['M0_PF_exposure', 'distinct_shows_recommended'])})", "hybrid_lists_analysis_val.csv"),
    ("The final re-ranker over-recommends Info and almost never recommends Films",
     f"Info {la.loc['M3_lambdarank', 'catmix_info']:.2%} vs truth {la.loc['truth', 'catmix_info']:.2%}; "
     f"Films {la.loc['M3_lambdarank', 'catmix_films']:.2%} vs truth {la.loc['truth', 'catmix_films']:.2%}", "hybrid_lists_analysis_val.csv"),
    ("Selection on validation is optimistic (best of many configurations)",
     f"grid sizes: PPR {grid_sizes['PPR']} configs, CBF {grid_sizes['CBF']} configs, M1 {grid_sizes['M1']:,} configs (tuned on inner), "
     f"LightGBM {grid_sizes['LightGBM']} configs x 3 folds (on inner); P1 never fits on val labels",
     "ppr_grid_val.csv, cbf_grid_val.csv, hybrid_m1_fusion_grid_inner.csv, hybrid_model_selection_inner_cv.csv"),
    ("Exposure features use the target-period schedule (public 31-42 schedule on Kaggle), counting airings only",
     "design statement; dropping the exposure group costs the re-ranker "
     f"{r('hybrid_ablation_val.csv').set_index('experiment').loc['drop exposure', 'delta_ALL']:+.4f} MAP@5 ALL",
     "hybrid_ablation_val.csv, outputs/LOG.md Step 5/6"),
    ("Kaggle public/private halves of about 630 users limit what a leaderboard difference can show",
     "NOT quantified: per-user AP vectors were not stored in Steps 3-6, so the requested bootstrap at n = 630 is skipped "
     "(it would need re-running a model, which Step 7 forbids)", "-"),
]
lim = pd.DataFrame(LIM, columns=["limitation", "supporting_number", "source_file"])
lim.to_csv(OUT / "limitations_evidence.csv", index=False, encoding="utf-8-sig")
lmd = ["# Limitations: evidence\n", "| # | limitation | supporting number | source |", "|---|---|---|---|"]
lmd += [f"| {i} | {a} | {b} | {c} |" for i, (a, b, c) in enumerate(LIM, 1)]
lmd += ["\n## Failure cases supported by results (hybrid_failure_cases_val.csv; SHAP of M3)\n"]
for u, g in fc.groupby("user_id", sort=False):
    lmd.append(f"- user {u} (AP change vs M0 {g.ap_change.iloc[0]:+.4f}):")
    for x in g.itertuples():
        lmd.append(f"  - {x.role}: {x.tv_show_id} {x.title} ({CAT_EN.get(x.category, x.category)}); M0 rank {x.rank_in_M0}, "
                   f"hybrid rank {x.rank_in_hybrid}; top SHAP: {x.shap1}; {x.shap2}")
lmd.append("- The first case (user 18133386382515269463) has no training history at all: all candidates come from popularity, "
           "and the re-ranker promotes a frequently aired cartoon (ex_a_target_glob_pw) over a truth news show.")
(OUT / "limitations_evidence.md").write_text("\n".join(lmd), encoding="utf-8")

# ====================================================================== D figure index + contact sheet
DESC = {
    "01": ("View duration histogram (log x)", "validation"), "02": ("View-log rows per user (log-log)", "validation"),
    "03": ("View-log rows per day; dip around 2020-07-03", "limitations"), "04": ("Labelled vs unlabelled schedule rows per file", "limitations"),
    "05": ("Categories among labelled schedule rows", "business-algorithmic analysis"), "06": ("Airing duration of labelled rows", "validation"),
    "07": ("Airings per tv_show_id (series level)", "validation"), "08": ("Screen-time fraction of labelled pairs, threshold 0.8", "validation"),
    "09": ("Threshold sensitivity: pairs and users with >= 5 shows", "validation"), "10": ("Weekly activity; weeks 25-27 dip", "limitations"),
    "11": ("Distinct qualifying shows per user", "validation"), "12": ("Category share: what airs vs what is watched", "business-algorithmic analysis"),
    "13": ("Where merged watch time falls (labelled / unlabelled / gaps)", "limitations"),
    "14": ("Baseline variants MAP@5 with CI", "method comparison"), "15": ("Per-user AP histogram, PF vs popularity", "validation"),
    "16": ("Baseline MAP@5 by training activity", "business-algorithmic analysis"), "17": ("Hit rate by rank, baselines", "validation"),
    "18": ("History-only ceiling analysis", "limitations"), "19": ("Baseline robustness across target windows", "validation"),
    "20": ("PPR grid heatmap (d x edge weight)", "method comparison"), "21": ("pop / PF / PPR MAP@5 with CI", "method comparison"),
    "22": ("Familiar vs new composition and precision (PPR)", "business-algorithmic analysis"),
    "23": ("MAP@5 by activity: pop / PF / PPR", "business-algorithmic analysis"), "24": ("Discovery of unseen shows: PPR vs popularity", "business-algorithmic analysis"),
    "25": ("PPR graph degree distribution (val, test)", "limitations"), "26": ("CBF grid by feature set", "method comparison"),
    "27": ("MAP@5 of all Step 5 methods with CI", "method comparison"), "28": ("Familiar vs new for all Step 5 methods", "business-algorithmic analysis"),
    "29": ("New-only discovery: CBF, PPR, popularity", "business-algorithmic analysis"), "30": ("Exposure diagnostics: hit rate vs A_target_us", "business-algorithmic analysis"),
    "31": ("Category mix of recommendations vs truth", "business-algorithmic analysis"),
    "32": ("MAP@5 of every method incl. hybrids (P1) with ceiling", "method comparison"), "33": ("Hybrid ablation by feature group", "method comparison"),
    "34": ("Re-ranker SHAP and gain importance (top-15)", "business-algorithmic analysis"), "35": ("Candidate recall by source; candidate-size sensitivity", "validation"),
    "36": ("MAP@5 by activity: final hybrid vs others", "business-algorithmic analysis"), "37": ("Failure analysis: AP change vs M0", "limitations"),
    "38": ("Accuracy vs measured cost, all methods", "method comparison"), "39": ("Example user (light): top-5 per method", "business-algorithmic analysis"),
    "40": ("Example user (heavy): top-5 per method", "business-algorithmic analysis"),
}
pngs = sorted(p.name for p in FIG.glob("*.png"))
idx = [{"number": int(p[:2]), "file": f"outputs/figures/{p}", "shows": DESC[p[:2]][0], "serves": DESC[p[:2]][1]} for p in pngs]
pd.DataFrame(idx).to_csv(OUT / "figure_index.csv", index=False)
n = len(pngs)
cols = 6
rows_ = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows_, cols, figsize=(cols * 3.2, rows_ * 2.3))
for ax, p in zip(axes.ravel(), pngs):
    ax.imshow(mpimg.imread(FIG / p))
    ax.set_title(f"{p[:2]}: {DESC[p[:2]][0][:38]}", fontsize=7)
    ax.axis("off")
for ax in axes.ravel()[n:]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "figures_contact_sheet.png", dpi=110)
plt.close(fig)
log(f"[D] {n} figures indexed; contact sheet written")
log("[done] files written; issues:", issues or "none")
