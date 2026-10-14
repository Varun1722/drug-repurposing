import pandas as pd
import pytest

from stage1.crosswalk import PrimeKGCrosswalk, mondo_ids_from_node_id, txgnn_key


def test_single_node_id_is_padded_to_seven_digits():
    assert mondo_ids_from_node_id("5180") == ["MONDO:0005180"]


def test_grouped_node_id_splits_on_underscore():
    assert mondo_ids_from_node_id("10820_13060") == ["MONDO:0010820", "MONDO:0013060"]


def test_long_ids_are_not_truncated():
    assert mondo_ids_from_node_id("100065") == ["MONDO:0100065"]


def test_txgnn_key_turns_plain_numbers_into_float_strings():
    # TxGNN's own convert2str writes "7739" as "7739.0" and leaves grouped ids and DrugBank ids.
    assert txgnn_key("7739") == "7739.0"
    assert txgnn_key("10820_13060") == "10820_13060"
    assert txgnn_key("DB00316") == "DB00316"


def _nodes():
    return pd.DataFrame(
        [
            {
                "node_index": 1,
                "node_id": "5180",
                "node_type": "disease",
                "node_name": "Parkinson disease",
                "node_source": "MONDO",
            },
            {
                "node_index": 2,
                "node_id": "10820_13060",
                "node_type": "disease",
                "node_name": "autosomal recessive Parkinson disease",
                "node_source": "MONDO_grouped",
            },
            {
                "node_index": 3,
                "node_id": "DB00316",
                "node_type": "drug",
                "node_name": "Acetaminophen",
                "node_source": "DrugBank",
            },
            {
                "node_index": 4,
                "node_id": "9796",
                "node_type": "gene/protein",
                "node_name": "PHYHIP",
                "node_source": "NCBI",
            },
        ]
    )


@pytest.fixture
def crosswalk():
    return PrimeKGCrosswalk(_nodes())


def test_disease_maps_to_its_node(crosswalk):
    hit = crosswalk.disease("MONDO:0005180")
    assert hit.node_index == 1
    assert hit.node_name == "Parkinson disease"
    assert hit.grouped is False
    assert hit.member_mondo_ids == ("MONDO:0005180",)


def test_grouped_disease_reports_all_members(crosswalk):
    hit = crosswalk.disease("MONDO:0013060")
    assert hit.node_index == 2
    assert hit.grouped is True
    assert hit.member_mondo_ids == ("MONDO:0010820", "MONDO:0013060")


def test_unmapped_disease_is_none(crosswalk):
    assert crosswalk.disease("MONDO:0000001") is None


def test_drug_maps_by_drugbank_id(crosswalk):
    hit = crosswalk.drug("DB00316")
    assert hit.node_index == 3
    assert hit.node_name == "Acetaminophen"
    assert crosswalk.drug("DB99999") is None
    assert crosswalk.drug(None) is None


def test_non_disease_nodes_are_ignored(crosswalk):
    assert crosswalk.disease("MONDO:0009796") is None


def test_disease_node_carries_the_key_txgnn_uses_internally(crosswalk):
    assert crosswalk.disease("MONDO:0005180").txgnn_key == "5180.0"
    assert crosswalk.disease("MONDO:0013060").txgnn_key == "10820_13060"
