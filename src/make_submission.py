"""Write and validate a Kaggle submission file in the template's row order."""
import csv

from src import io_utils as io

SUB_DIR = io.ROOT / "outputs" / "submissions"
HEADER = ["user_id", "tv_show_id"]


def template_users():
    with open(io.DATA / io.SUBMISSION_FILE, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        users = [r[0] for r in rd]
    assert header == HEADER, header
    return users


def validate(rows, pool):
    """rows: list of (user_str, [5 ints]). Asserts every rule and prints a checklist."""
    tmpl = template_users()
    pool = set(int(x) for x in pool)
    checks = {
        "1,260 rows": len(rows) == 1260,
        "same users and order as template": [u for u, _ in rows] == tmpl,
        "each row exactly 5 distinct integers": all(len(r) == 5 and len(set(r)) == 5 and all(isinstance(x, int) for x in r)
                                                    for _, r in rows),
        "no zeros": all(0 not in r for _, r in rows),
        "all ids in test pool": all(set(r) <= pool for _, r in rows),
    }
    for k, ok in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {k}")
    assert all(checks.values()), checks


def write_submission(name, recs, split):
    """recs: original user string -> list of 5 ids. Writes outputs/submissions/<name>.csv."""
    rows = [(u, [int(x) for x in recs[u]]) for u in split.user_str]
    print(f"validating {name}:")
    validate(rows, split.pool)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    path = SUB_DIR / f"{name}.csv"
    with open(path, "w", newline="") as f:
        f.write(",".join(HEADER) + "\n")
        for u, r in rows:
            f.write(f"{u},{' '.join(map(str, r))}\n")
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    assert lines[0] == "user_id,tv_show_id" and len(lines) == 1261
    print(f"  [OK] exact header; wrote {path.name}; first 3 rows:")
    for line in lines[1:4]:
        print("   ", line)
    return path
