import pandas as pd
import pytest

from stage1.matrix_store import MatrixStore, normalise_column

BOOL_COLS = [
    "is_known_positive",
    "is_known_negative",
    "trial_sig_better",
    "trial_non_sig_better",
    "trial_sig_worse",
    "trial_non_sig_worse",
    "off_label",
    "ec_indications_list_off_label",
    "ec_indications_list_on_label",
    "ec_indications_list",
]


def _shard(rows):
    """Build a frame with MATRIX's columns, using the spaced names found in the README."""
    frame = pd.DataFrame(rows)
    for col in BOOL_COLS:
        if col not in frame:
            frame[col] = False
    frame["not treat score"] = 0.1
    frame["unknown score"] = 1.0 - frame["untransformed_treat_score"] - 0.1
    if "source" not in frame:
        frame["source"] = "CHEBI:" + frame["source_ec_id"]
    # Real scored rows have y = NULL; only the label rows carry 1 or 0.
    frame["y"] = (frame["y"] if "y" in frame else pd.Series([pd.NA] * len(frame))).astype("Int32")
    for col in ["untransformed_rank", "rank", "rank_drug", "rank_disease"]:
        frame[col] = 0
    for col in ["quantile_rank", "quantile_drug", "quantile_disease"]:
        frame[col] = 0.0
    return frame


@pytest.fixture
def store(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    rows_a = [
        {
            "target": "MONDO:1",
            "source_ec_id": "EC:1",
            "untransformed_treat_score": 0.2,
            "transformed_treat_score": 0.30,
        },
        {
            "target": "MONDO:1",
            "source_ec_id": "EC:2",
            "untransformed_treat_score": 0.6,
            "transformed_treat_score": 0.90,
            "is_known_positive": True,
            "y": 1,
        },
        {
            "target": "MONDO:2",
            "source_ec_id": "EC:1",
            "untransformed_treat_score": 0.5,
            "transformed_treat_score": 0.70,
        },
    ]
    rows_b = [
        {
            "target": "MONDO:1",
            "source_ec_id": "EC:3",
            "untransformed_treat_score": 0.4,
            "transformed_treat_score": 0.50,
        },
        {
            "target": "MONDO:2",
            "source_ec_id": "EC:2",
            "untransformed_treat_score": 0.1,
            "transformed_treat_score": 0.20,
        },
    ]
    _shard(rows_a).to_parquet(shards / "a.parquet")
    _shard(rows_b).to_parquet(shards / "b.parquet")
    labels = _shard(
        [
            {
                "target": "MONDO:1",
                "source": "CHEBI:99",
                "source_ec_id": None,
                "untransformed_treat_score": float("nan"),
                "transformed_treat_score": float("nan"),
                "is_known_positive": True,
                "y": 1,
            },
            {
                "target": "MONDO:2",
                "source": "CHEBI:98",
                "source_ec_id": None,
                "untransformed_treat_score": float("nan"),
                "transformed_treat_score": float("nan"),
                "is_known_negative": True,
                "y": 0,
            },
        ]
    )
    labels.to_parquet(shards / "c.parquet")
    return MatrixStore.build(tmp_path / "m.duckdb", str(shards / "*.parquet"))


def test_normalise_column_replaces_spaces():
    assert normalise_column("not treat score") == "not_treat_score"
    assert normalise_column("rank_drug") == "rank_drug"


def test_build_loads_scored_rows_from_all_shards(store):
    assert store.row_count() == 5


def test_unscored_label_rows_go_to_a_separate_table(store):
    assert store.label_count() == 2
    labels = store.labels_for("MONDO:1")
    assert labels["source"].tolist() == ["CHEBI:99"]
    assert labels["y"].tolist() == [1]
    assert store.labels_for("MONDO:404").empty


def test_top_drugs_orders_by_transformed_score(store):
    top = store.top_drugs("MONDO:1", n=10)
    assert top["source_ec_id"].tolist() == ["EC:2", "EC:3", "EC:1"]
    assert top["rank_in_disease"].tolist() == [1, 2, 3]


def test_top_drugs_can_order_by_untransformed_score(tmp_path):
    shards = tmp_path / "s"
    shards.mkdir()
    rows = [
        {
            "target": "MONDO:1",
            "source_ec_id": "EC:A",
            "untransformed_treat_score": 0.9,
            "transformed_treat_score": 0.5,
        },
        {
            "target": "MONDO:1",
            "source_ec_id": "EC:B",
            "untransformed_treat_score": 0.1,
            "transformed_treat_score": 0.8,
        },
    ]
    _shard(rows).to_parquet(shards / "a.parquet")
    built = MatrixStore.build(tmp_path / "m.duckdb", str(shards / "*.parquet"))
    by_transformed = built.top_drugs("MONDO:1")
    by_raw = built.top_drugs("MONDO:1", by="untransformed_treat_score")
    assert by_transformed["source_ec_id"].tolist() == ["EC:B", "EC:A"]
    assert by_raw["source_ec_id"].tolist() == ["EC:A", "EC:B"]


def test_top_drugs_rejects_unknown_score_column(store):
    with pytest.raises(ValueError):
        store.top_drugs("MONDO:1", by="not_a_column; DROP TABLE scores")


def test_top_drugs_respects_n(store):
    assert len(store.top_drugs("MONDO:1", n=2)) == 2


def test_known_pairs_are_flagged_and_rank_among_new_skips_them(store):
    top = store.top_drugs("MONDO:1", n=10)
    flagged = top.set_index("source_ec_id")
    assert flagged["already_known"].to_dict() == {"EC:2": True, "EC:3": False, "EC:1": False}
    assert flagged["rank_among_new"].isna().to_dict()["EC:2"]
    assert flagged["rank_among_new"].dropna().astype(int).to_dict() == {"EC:3": 1, "EC:1": 2}


def test_top_drugs_can_hide_known_pairs_without_renumbering_rank_in_disease(store):
    top = store.top_drugs("MONDO:1", n=10, include_known=False)
    assert top["source_ec_id"].tolist() == ["EC:3", "EC:1"]
    assert top["rank_in_disease"].tolist() == [2, 3]
    assert top["rank_among_new"].tolist() == [1, 2]


def test_unknown_disease_returns_empty_frame(store):
    top = store.top_drugs("MONDO:404", n=10)
    assert top.empty
    assert "source_ec_id" in top.columns


def test_disease_coverage_counts_drugs_per_disease(store):
    coverage = store.coverage(["MONDO:1", "MONDO:2", "MONDO:404"])
    assert coverage.set_index("target")["n_drugs"].to_dict() == {"MONDO:1": 3, "MONDO:2": 2}


def test_recomputed_rank_matches_stored_rank_when_consistent(tmp_path):
    shards = tmp_path / "s"
    shards.mkdir()
    frame = _shard(
        [
            {
                "target": "MONDO:1",
                "source_ec_id": "EC:1",
                "untransformed_treat_score": 0.2,
                "transformed_treat_score": 0.3,
            },
            {
                "target": "MONDO:1",
                "source_ec_id": "EC:2",
                "untransformed_treat_score": 0.6,
                "transformed_treat_score": 0.9,
            },
        ]
    )
    frame["rank_drug"] = [2, 1]
    frame.to_parquet(shards / "a.parquet")
    built = MatrixStore.build(tmp_path / "m.duckdb", str(shards / "*.parquet"))
    report = built.rank_agreement(
        "MONDO:1", stored_column="rank_drug", by="transformed_treat_score"
    )
    assert report == {"n": 2, "exact_matches": 2}
