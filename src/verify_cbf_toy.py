"""Toy CBF example with exact (sympy) arithmetic, compared to src.cbf. Run: python -m src.verify_cbf_toy"""
import numpy as np
import pandas as pd
import sympy as sy

from src import io_utils as io
from src.cbf import GENRE_POS, Items, cosine_scores, profiles

SHOWS = {  # id: (name, category, [genres], director)
    1: ("A", "Kids", ["cartoon", "comedy"], "X"),
    2: ("B", "Kids", ["cartoon"], "Y"),
    3: ("C", "Films", ["comedy", "drama"], "X"),
    4: ("D", "Films", ["drama"], None),
}
USER = {1: 2, 2: 1}  # show id -> softfrac weight (A watched fully twice, B once)
lines = []


def out(s=""):
    lines.append(s)
    print(s)


out("Toy content-based filtering: blocks category + genres + director, block weights 1")
for i, (n, c, g, d) in SHOWS.items():
    out(f"  show {n} (id {i}): category={c}, genres={g}, director={d}")
out(f"  user history (softfrac weights): " + ", ".join(f"{SHOWS[i][0]}={w}" for i, w in USER.items()))

pos = [sy.Rational(str(x)) for x in GENRE_POS]
cats = sorted({v[1] for v in SHOWS.values()})
gens = sorted({g for v in SHOWS.values() for g in v[2]})
dirs = sorted({v[3] for v in SHOWS.values() if v[3]})
N = len(SHOWS)
df = {d: sum(v[3] == d for v in SHOWS.values()) for d in dirs}
idf = {d: sy.log(sy.Rational(1 + N, 1 + df[d])) + 1 for d in dirs}
out(f"\nfeature order: category {cats}; genres {gens}; director {dirs}")
out("director idf = ln((1+N)/(1+df)) + 1 with N = 4: " + ", ".join(f"{d}: ln(5/{1 + df[d]}) + 1" for d in dirs))


def unit(v):
    n = sy.sqrt(sum(x**2 for x in v))
    return [sy.nsimplify(0)] * len(v) if n == 0 else [sy.simplify(x / n) for x in v]


vec = {}
for i, (n, c, g, d) in SHOWS.items():
    vc = unit([sy.Integer(int(c == x)) for x in cats])
    vg_raw = [sum((pos[k] for k, gg in enumerate(g) if gg == x), sy.Integer(0)) for x in gens]
    vg = unit(vg_raw)
    vd = unit([idf[x] if d == x else sy.Integer(0) for x in dirs])
    full = unit(vc + vg + vd)
    vec[i] = full
    out(f"\nshow {n}: category block {vc}")
    out(f"         genres raw {vg_raw} -> / sqrt({sum(x**2 for x in vg_raw)}) = {vg}")
    out(f"         director block {vd}   " + ("(one token -> unit after L2, idf cancels)" if d else "(missing -> zero block)"))
    out(f"         concatenated, divided by sqrt(number of non-empty blocks) -> x_{n} = {full}")

prof_raw = [sum(USER[i] * vec[i][j] for i in USER) for j in range(len(vec[1]))]
prof = unit(prof_raw)
out(f"\nprofile raw = 2*x_A + 1*x_B = {[sy.simplify(x) for x in prof_raw]}")
out(f"profile (L2-normalised) = {prof}")
hand = {}
out("\ncosine(profile, x_s) = profile . x_s (both unit length):")
for i, (n, *_ ) in SHOWS.items():
    terms = [f"({sy.simplify(a)})*({sy.simplify(b)})" for a, b in zip(prof, vec[i]) if a != 0 and b != 0]
    val = sy.simplify(sum(a * b for a, b in zip(prof, vec[i])))
    hand[i] = float(sy.N(val, 30))
    out(f"  {n}: {' + '.join(terms) or '0'} = {val} = {hand[i]:.15f}")

meta = pd.DataFrame({"title": [v[0] for v in SHOWS.values()], "category": [v[1] for v in SHOWS.values()],
                     "genre_1": [v[2][0] for v in SHOWS.values()],
                     "genre_2": [v[2][1] if len(v[2]) > 1 else None for v in SHOWS.values()],
                     "genre_3": [None] * 4, "year": [None] * 4, "director": [v[3] for v in SHOWS.values()],
                     "actors": [None] * 4}, index=pd.Index(list(SHOWS), name="tv_show_id"))
items = Items(meta, ["category", "genres", "director"])
train = pd.DataFrame({"user_id": np.array([7, 7, 7], dtype=np.uint64), "tv_show_id": [1, 1, 2],
                      "frac": [1.0, 1.0, 1.0], "airing_start": pd.to_datetime(["2020-03-10"] * 3)})
P, has, _ = profiles(items, train, np.array([7], dtype=np.uint64), "softfrac", None, pd.Timestamp("2020-05-18"))
S = cosine_scores(P, items, np.array(list(SHOWS)))[0]
out("\nsrc.cbf cosines: " + ", ".join(f"{SHOWS[i][0]}={s:.15f}" for i, s in zip(SHOWS, S)))
diff = max(abs(S[k] - hand[i]) for k, i in enumerate(SHOWS))
out(f"max |src.cbf - exact| = {diff:.3e} -> {'OK (< 1e-12)' if diff < 1e-12 else 'FAIL'}")
assert diff < 1e-12
(io.ROOT / "outputs" / "verify_cbf_toy.txt").write_text("\n".join(lines), encoding="utf-8")
