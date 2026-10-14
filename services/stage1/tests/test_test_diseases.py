import pytest

from stage1.crosswalk import DEFAULT_NODES, PrimeKGCrosswalk
from stage1.disease_lookup import DEFAULT_PARQUET, DiseaseLookup
from stage1.test_diseases import TEST_DISEASES, build_table

needs_data = pytest.mark.skipif(
    not (DEFAULT_PARQUET.exists() and DEFAULT_NODES.exists()),
    reason="reference data not downloaded",
)


def test_set_has_twenty_unique_diseases_across_three_groups():
    assert len(TEST_DISEASES) == 20
    assert len({mondo for _, mondo, _, _ in TEST_DISEASES}) == 20
    assert {group for _, _, group, _ in TEST_DISEASES} == {"common", "hard", "rare"}


@needs_data
def test_every_test_disease_resolves_to_its_pinned_mondo_id():
    table = build_table(DiseaseLookup.from_parquet(), PrimeKGCrosswalk.from_csv())
    assert len(table) == 20


@needs_data
def test_only_idiopathic_pulmonary_fibrosis_lacks_a_txgnn_node():
    table = build_table(DiseaseLookup.from_parquet(), PrimeKGCrosswalk.from_csv())
    assert table[table["txgnn_node_index"].isna()]["name"].tolist() == [
        "Idiopathic pulmonary fibrosis"
    ]
