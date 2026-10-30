"""Offline smoke check (no data, seconds): headline numbers agree between result CSVs and step summaries, and every
file recorded in the reproducibility CSVs still has the recorded hash. Run: python -m src.smoke_check
"""
import hashlib
import json

import pandas as pd

from src import io_utils as io

OUT = io.ROOT / "outputs"
S = {k: json.loads((OUT / f"step{k}_summary.json").read_text()) for k in (3, 4, 5, 6)}
p1 = pd.read_csv(OUT / "hybrid_p1_results_val.csv").set_index("method")
b3 = pd.read_csv(OUT / "baseline_results_val.csv").set_index("method")
fm = pd.read_csv(OUT / "final_method_comparison.csv").set_index("method")
ok, fail = 0, []


def check(name, a, b, tol=1e-12):
    global ok
    if abs(float(a) - float(b)) <= tol:
        ok += 1
    else:
        fail.append(f"{name}: {a} != {b}")


for p in ["ALL", "SUB"]:
    for m in ["pop_recent", "PF_softfrac"]:
        check(f"{m} {p} p1 vs baseline_results", p1.loc[m, f"map5_{p}"], b3.loc[m, f"map5_{p}"])
    check(f"PPR {p} p1 vs step4_summary", p1.loc["PPR", f"map5_{p}"], S[4]["ppr_val"][p]["map5"])
    check(f"cbf_pop {p} p1 vs step5_summary", p1.loc["cbf_pop", f"map5_{p}"], S[5]["cbf_pop_val"][p]["map5"])
    check(f"M0 {p} p1 vs step5_summary", p1.loc["M0_PF_exposure", f"map5_{p}"], S[5]["best_exposure"][f"map5_{p}"])
    for m in ["pop_recent", "PF_softfrac", "PPR", "cbf_pop", "M0_PF_exposure", "M3_lambdarank"]:
        check(f"{m} {p} p1 vs final_method_comparison", p1.loc[m, f"map5_{p}"], fm.loc[m, f"map5_{p}"])
check("M3 - M0 ALL diff vs step6_summary", pd.read_csv(OUT / "hybrid_p1_paired_diffs_val.csv").query(
    "model == 'M3_lambdarank' and vs == 'M0_PF_exposure' and population == 'ALL'").mean_diff.iloc[0], S[6]["P1"]["hybrid_vs_M0_ALL"][0])
assert S[3]["chosen_pop"] == "pop_recent" and S[3]["chosen_pf"] == "PF_softfrac" and S[6]["P1"]["final_strategy"] == "M3_lambdarank"


def content_hash(f, drop):
    if f.endswith(".csv") and not f.startswith("submissions"):
        df = pd.read_csv(OUT / f)
        df = df.drop(columns=[c for c in df.columns if drop(c)], errors="ignore")
        return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    return hashlib.sha256((OUT / f).read_bytes()).hexdigest()


RULE = {3: None, 4: lambda c: "seconds" in c or c == "runtime", 5: lambda c: "seconds" in c, 6: lambda c: "seconds" in c}
nh = 0
for k, drop in RULE.items():
    rep = pd.read_csv(OUT / f"step{k}_reproducibility.csv")
    for f, h in zip(rep.file, rep.sha256_run2):
        got = hashlib.sha256((OUT / f).read_bytes()).hexdigest() if drop is None else content_hash(f, drop)
        nh += 1
        if got != h:
            fail.append(f"step {k} hash changed: {f}")
lines = [f"headline-number checks passed: {ok}", f"files re-hashed against step3-6 reproducibility records: {nh}",
         "failures: " + ("none" if not fail else "; ".join(fail))]
print("\n".join(lines))
(OUT / "smoke_check_output.txt").write_text("\n".join(lines) + "\n")
assert not fail
