"""DuckDB store for the MATRIX score file (about 39.5M drug-disease pairs)."""

from pathlib import Path

import duckdb
import pandas as pd

# A pair counts as already known if any of these flags is set.
KNOWN_FLAGS = (
    "is_known_positive",
    "ec_indications_list",
    "ec_indications_list_on_label",
    "ec_indications_list_off_label",
    "off_label",
)
ORDER_COLUMNS = ("transformed_treat_score", "untransformed_treat_score")


def normalise_column(name: str) -> str:
    """The README writes some columns with spaces ('not treat score'); use underscores."""
    return name.strip().replace(" ", "_")


class MatrixStore:
    def __init__(self, db_path: Path | str, read_only: bool = True):
        self.con = duckdb.connect(str(db_path), read_only=read_only)

    @classmethod
    def build(
        cls, db_path: Path | str, shard_glob: str, memory_limit: str = "20GB"
    ) -> "MatrixStore":
        con = duckdb.connect(str(db_path))
        con.execute(f"SET memory_limit = '{memory_limit}'")
        con.execute(f"SET temp_directory = '{Path(db_path).parent}/duckdb_tmp'")
        described = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [shard_glob]).fetchall()
        select = ", ".join(
            f'"{name}" AS {normalise_column(name)}'
            if normalise_column(name) != name
            else f'"{name}"'
            for name, *_ in described
        )
        # The file mixes two kinds of row. Scored pairs have scores and ranks. Label rows are the
        # known positive and negative training pairs: they carry no scores, and a pair that is
        # labelled is not scored.
        con.execute(
            f"""
            CREATE OR REPLACE TABLE scores AS
            SELECT {select} FROM read_parquet(?)
            WHERE untransformed_treat_score IS NOT NULL
            ORDER BY target, transformed_treat_score DESC
            """,
            [shard_glob],
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE labels AS
            SELECT {select} FROM read_parquet(?)
            WHERE untransformed_treat_score IS NULL
            ORDER BY target
            """,
            [shard_glob],
        )
        con.close()
        return cls(db_path)

    def close(self) -> None:
        self.con.close()

    def row_count(self) -> int:
        return self.con.execute("SELECT count(*) FROM scores").fetchone()[0]

    def label_count(self) -> int:
        return self.con.execute("SELECT count(*) FROM labels").fetchone()[0]

    def labels_for(self, mondo_id: str) -> pd.DataFrame:
        """Known positive and negative training pairs for a disease (these have no scores)."""
        return self.con.execute(
            "SELECT source, target, y, is_known_positive, is_known_negative FROM labels "
            "WHERE target = ? ORDER BY y DESC, source",
            [mondo_id],
        ).df()

    def columns(self) -> list[str]:
        return [row[0] for row in self.con.execute("DESCRIBE scores").fetchall()]

    def top_drugs(
        self,
        mondo_id: str,
        n: int = 100,
        include_known: bool = True,
        by: str = "transformed_treat_score",
    ) -> pd.DataFrame:
        """Drugs for one disease, best first. `by` picks the score that orders them."""
        if by not in ORDER_COLUMNS:
            raise ValueError(f"order by one of {ORDER_COLUMNS}, not {by!r}")
        known = " OR ".join(f"coalesce({flag}, false)" for flag in KNOWN_FLAGS)
        query = f"""
            WITH ranked AS (
                SELECT *,
                    ({known}) AS already_known,
                    row_number() OVER (
                        ORDER BY {by} DESC, source_ec_id
                    ) AS rank_in_disease
                FROM scores WHERE target = ?
            )
            SELECT *,
                CASE WHEN already_known THEN NULL
                     ELSE row_number() OVER (
                         PARTITION BY already_known ORDER BY rank_in_disease
                     ) END AS rank_among_new
            FROM ranked
        """
        outer = "SELECT * FROM ({q}) WHERE {cond} ORDER BY rank_in_disease LIMIT ?"
        cond = "true" if include_known else "NOT already_known"
        return self.con.execute(outer.format(q=query, cond=cond), [mondo_id, n]).df()

    def coverage(self, mondo_ids: list[str]) -> pd.DataFrame:
        return self.con.execute(
            "SELECT target, count(*) AS n_drugs FROM scores WHERE target IN (SELECT unnest(?)) "
            "GROUP BY target ORDER BY target",
            [mondo_ids],
        ).df()

    def rank_agreement(self, mondo_id: str, stored_column: str, by: str) -> dict[str, int]:
        """Compare a stored rank column with the rank recomputed from `by` for one disease."""
        allowed = set(self.columns())
        if stored_column not in allowed or by not in allowed:
            raise ValueError(f"unknown column: {stored_column!r} or {by!r}")
        row = self.con.execute(
            f"""
            SELECT count(*), count(*) FILTER (WHERE stored = recomputed) FROM (
                SELECT {stored_column} AS stored,
                       rank() OVER (ORDER BY {by} DESC) AS recomputed
                FROM scores WHERE target = ?
            )
            """,
            [mondo_id],
        ).fetchone()
        return {"n": row[0], "exact_matches": row[1]}
