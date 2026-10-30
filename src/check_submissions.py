"""Re-validate every file in outputs/submissions/ and compare them. Run: python -m src.check_submissions"""
import csv
import hashlib

import numpy as np
import pandas as pd

from src import io_utils as io
from src.designate_final import test_pool
from src.make_submission import validate

SUB = io.ROOT / "outputs" / "submissions"


def read(p):
    with open(p, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = [(r[0], [int(x) for x in r[1].split()]) for r in rd]
    assert header == ["user_id", "tv_show_id"], (p.name, header)
    return rows


pool = test_pool()
files = sorted(SUB.glob("*.csv"))
data = {}
for p in files:
    print(f"{p.name}:")
    rows = read(p)
    validate(rows, pool)
    data[p.stem] = dict(rows)
names = list(data)
users = list(data[names[0]])
summary = []
for n in names:
    summary.append({"file": f"{n}.csv", "rows": len(data[n]), "sha256": hashlib.sha256((SUB / f"{n}.csv").read_bytes()).hexdigest(),
                    "users_identical_to_sub05": sum(data[n][u] == data["sub05_pf_exposure"][u] for u in users),
                    "users_identical_to_sub06": sum(data[n][u] == data["sub06_hybrid"][u] for u in users)})
S = pd.DataFrame(summary)
M = pd.DataFrame([[np.mean([len(set(data[a][u]) & set(data[b][u])) for u in users]) for b in names] for a in names],
                 index=names, columns=names)
S = S.join(M.add_prefix("mean_overlap_with_").reset_index(drop=True))
S.to_csv(io.ROOT / "outputs" / "submissions_check.csv", index=False)
print(S[["file", "rows", "sha256", "users_identical_to_sub05", "users_identical_to_sub06"]].to_string(index=False))
print("\nmean number of common ids per user (out of 5):")
print(M.round(2).to_string())
