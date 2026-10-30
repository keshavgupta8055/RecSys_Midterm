"""Designate one submission file as final.

Usage: python -m src.designate_final <name> [--out-dir DIR]
Validates outputs/submissions/<name>.csv with the shared validator, copies it to <out-dir>/final_submission.csv
(default: submission/) and writes <out-dir>/FINAL.txt with name, sha256, rows, validation result and the val MAP@5
of the method that produced it.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from src import io_utils as io
from src.make_submission import validate

OUT = io.ROOT / "outputs"
METHOD = {"sub01_popularity": "pop_recent", "sub02_personal_baseline": "PF_softfrac", "sub03_ppr": "PPR",
          "sub04_cbf": "cbf_pop", "sub05_pf_exposure": "M0_PF_exposure", "sub06_hybrid": "M3_lambdarank",
          "sub07_fusion": "M1_fusion"}


def test_pool():
    """Test pool = labelled shows airing in [2020-07-27, 2020-10-19), as in src.protocol (needs the cached schedules)."""
    from src.protocol import get_split
    return get_split("test").pool


def val_map(method):
    import pandas as pd
    p1 = pd.read_csv(OUT / "hybrid_p1_results_val.csv").set_index("method")
    return {p: float(p1.loc[method, f"map5_{p}"]) for p in ["ALL", "SUB"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--out-dir", default=str(io.ROOT / "submission"))
    a = ap.parse_args()
    name = a.name.removesuffix(".csv")
    src = OUT / "submissions" / f"{name}.csv"
    with open(src, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = [(r[0], [int(x) for x in r[1].split()]) for r in rd]
    assert header == ["user_id", "tv_show_id"], header
    print(f"validating {src.name}:")
    validate(rows, test_pool())
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "final_submission.csv"
    shutil.copyfile(src, dst)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    m = METHOD.get(name)
    vm = val_map(m) if m else None
    info = {"name": name, "source": src.relative_to(io.ROOT).as_posix(), "sha256": sha, "rows": len(rows), "header": header,
            "validation": "passed (1,260 rows, template order, 5 distinct ints, no zeros, all ids in the test pool, exact header)",
            "method": m, "val_map5": vm,
            "val_map5_protocol": "P1 (trained on inner, evaluated on val)" if m and m.startswith("M") and m != "M0_PF_exposure"
            else "trained on weeks 11-20, evaluated on weeks 21-30"}
    (out / "FINAL.txt").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
