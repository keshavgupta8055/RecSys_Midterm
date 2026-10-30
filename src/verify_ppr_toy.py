"""Toy PPR example computed with exact fractions, compared to src.ppr. Run: python -m src.verify_ppr_toy"""
from fractions import Fraction as F

import numpy as np
import pandas as pd

from src import io_utils as io
from src.ppr import build_graph, ppr

SHOWS = ["A", "B", "C", "D"]
IDS = {"A": 1, "B": 2, "C": 3, "D": 4}
WATCH = {"u1": ["A", "B"], "u2": ["A", "B", "C"], "u3": ["C", "D"]}
D = F(1, 2)
lines = []


def out(s=""):
    lines.append(s)
    print(s)


out("Toy Personalized PageRank (raw co-count graph, d = 1/2, seed = user u1's single show A)")
for u, w in WATCH.items():
    out(f"  {u} watched {w}")
n = len(SHOWS)
C = [[F(0)] * n for _ in range(n)]
for i, a in enumerate(SHOWS):
    for j, b in enumerate(SHOWS):
        if i != j:
            C[i][j] = F(sum(a in w and b in w for w in WATCH.values()))
out("\nco-count matrix C (users who watched both, diagonal removed):")
out("      " + "  ".join(f"{s:>5}" for s in SHOWS))
for i, a in enumerate(SHOWS):
    out(f"  {a}:  " + "  ".join(f"{str(x):>5}" for x in C[i]))
T = [[C[i][j] / sum(C[i]) for j in range(n)] for i in range(n)]
out("\nT = row-normalised C:")
for i, a in enumerate(SHOWS):
    out(f"  {a}: " + "  ".join(f"{str(x):>5}" for x in T[i]) + f"   (row sum {sum(T[i])})")

P = [F(1), F(0), F(0), F(0)]
R = list(P)
out(f"\nR_0 = P = {[str(x) for x in R]}")
for t in (1, 2):
    out(f"\nIteration {t}: R_{t} = (1 - d) * P + d * (R_{t - 1} T)")
    new = []
    for j, b in enumerate(SHOWS):
        terms = [(R[i], T[i][j]) for i in range(n) if R[i] != 0 and T[i][j] != 0]
        walk = sum(r * tt for r, tt in terms)
        expr = " + ".join(f"{r}*{tt}" for r, tt in terms) or "0"
        val = (1 - D) * P[j] + D * walk
        out(f"  {b}: (1/2)*{P[j]} + (1/2)*({expr}) = {(1 - D) * P[j]} + (1/2)*{walk} = {val}")
        new.append(val)
    R = new
    out(f"  R_{t} = {[str(x) for x in R]}  (sum = {sum(R)})")
hand = np.array([float(x) for x in R])

# vectorised implementation on the same toy data
rows = [{"user_id": np.uint64(k + 1), "tv_show_id": IDS[s], "frac": 1.0, "watched_seconds": 1}
        for k, (u, w) in enumerate(WATCH.items()) for s in w]
toy = pd.DataFrame(rows)
g = build_graph(toy, "raw", "softfrac")
p = np.zeros((1, 4))
p[0, 0] = 1.0
R2, _ = ppr(p, g.T, 0.5, n_iter=2)
Rc, info = ppr(p, g.T, 0.5, max_iter=1000, tol=1e-15)
closed = 0.5 * p @ np.linalg.inv(np.eye(4) - 0.5 * g.T)
out("\nsrc.ppr after 2 iterations:  " + np.array2string(R2[0], precision=15))
out(f"hand (fractions) after 2:    {np.array2string(hand, precision=15)}")
diff = float(np.abs(R2[0] - hand).max())
out(f"max |vectorised - hand| = {diff:.3e} -> {'OK (< 1e-12)' if diff < 1e-12 else 'FAIL'}")
assert diff < 1e-12
out(f"\nconverged power iteration ({info['max_iterations']} iterations): {np.array2string(Rc[0], precision=12)}")
out(f"closed form (1-d) p (I - dT)^-1:            {np.array2string(closed[0], precision=12)}")
cd = float(np.abs(Rc[0] - closed[0]).max())
out(f"max |power iteration - closed form| = {cd:.3e}; after 2 iterations the gap to the fixed point is "
    f"{float(np.abs(R2[0] - closed[0]).max()):.4f}")
assert cd < 1e-12
(io.ROOT / "outputs" / "verify_ppr_toy.txt").write_text("\n".join(lines), encoding="utf-8")
