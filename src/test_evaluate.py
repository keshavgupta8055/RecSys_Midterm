"""Unit tests for ap_at_k / map_at_k (plain asserts). Run: python -m src.test_evaluate"""
from src.evaluate import ap_at_k, map_at_k

T = ["a", "b", "c", "d", "e"]
CASES = [
    ("perfect list", ["a", "b", "c", "d", "e"], 1.0),
    ("same ids, other order", ["e", "c", "a", "d", "b"], 1.0),
    ("one hit at rank 1 only", ["a", "x", "y", "z", "w"], 1 / 5),
    ("one hit at rank 5 only", ["x", "y", "z", "w", "a"], (1 / 5) / 5),
    ("hits at ranks 1 and 3", ["a", "x", "b", "y", "z"], (1 + 2 / 3) / 5),
    ("no hit", ["v", "w", "x", "y", "z"], 0.0),
    ("duplicate ids count once", ["a", "a", "b", "c", "d"], (1 + 2 / 3 + 3 / 4 + 4 / 5) / 5),
    ("duplicate of a hit only (a,a,a,a,a)", ["a", "a", "a", "a", "a"], 1 / 5),
    ("list longer than k is truncated", ["x", "y", "z", "w", "v", "a"], 0.0),
]

lines = []
for name, rec, exp in CASES:
    got = ap_at_k(rec, set(T), 5)
    ok = abs(got - exp) < 1e-12
    lines.append(f"{'PASS' if ok else 'FAIL'}  {name:40s} rec={rec}  expected={exp:.6f}  computed={got:.6f}")
    print(lines[-1])
    assert ok, name

m, ap = map_at_k({1: ["a", "b", "c", "d", "e"], 2: ["x", "y", "z", "w", "a"]}, {1: T, 2: T, 3: T})
exp = (1.0 + 0.04 + 0.0) / 3
lines.append(f"{'PASS' if abs(m - exp) < 1e-12 else 'FAIL'}  map_at_k with a missing user scores it 0: "
             f"expected={exp:.6f} computed={m:.6f}")
print(lines[-1])
assert abs(m - exp) < 1e-12
print("all evaluator tests passed")

if __name__ == "__main__":
    from src import io_utils as io
    (io.ROOT / "outputs" / "test_evaluate_output.txt").write_text("\n".join(lines) + "\nall evaluator tests passed\n")
