"""T2 acceptance checks on the built MATRIX store. Exits non-zero if any check fails."""

import sys

import pandas as pd

from stage1.matrix_store import MatrixStore
from stage1.paths import DB, ROOT

EXPECTED_ROWS = 39_508_102


def main() -> int:
    store = MatrixStore(DB / "matrix.duckdb")
    test_diseases = pd.read_csv(ROOT / "config" / "test_diseases.csv")
    failures: list[str] = []

    total = store.row_count() + store.label_count()
    print(f"rows: {store.row_count():,} scored + {store.label_count():,} labels = {total:,}")
    if total != EXPECTED_ROWS:
        failures.append(f"expected {EXPECTED_ROWS:,} rows, found {total:,}")

    overlap = store.con.execute(
        "SELECT count(*) FROM labels l JOIN scores s USING (source, target)"
    ).fetchone()[0]
    print(f"labelled pairs that are also scored: {overlap}")
    if overlap:
        failures.append("some labelled pairs are also scored")

    for mondo_id, name in zip(test_diseases["mondo_id"], test_diseases["name"], strict=True):
        agreement = store.rank_agreement(mondo_id, "rank_drug", "untransformed_treat_score")
        if agreement["n"] == 0:
            print(f"  {name}: not scored by MATRIX")
        elif agreement["n"] != agreement["exact_matches"]:
            failures.append(f"{name}: rank_drug differs from untransformed order {agreement}")

    print("FAILED: " + "; ".join(failures) if failures else "all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
