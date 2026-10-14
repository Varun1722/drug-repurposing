import numpy as np
import pandas as pd
import pytest

from stage1.disease_lookup import DiseaseLookup


def _row(mondo_id, name, synonyms=(), matrix_filter=True, prevalence=None, deleted=False):
    return {
        "id": mondo_id,
        "name": name,
        "synonyms": np.array(list(synonyms), dtype=object),
        "official_matrix_filter": matrix_filter,
        "prevalence_world": prevalence,
        "deleted": deleted,
    }


@pytest.fixture
def lookup():
    df = pd.DataFrame(
        [
            _row(
                "MONDO:0005180",
                "Parkinson disease",
                ["Parkinson's disease", "PD"],
                prevalence="10-99 in 100,000",
            ),
            _row("MONDO:0004975", "Alzheimer disease", ["Alzheimer's disease", "AD"]),
            _row("MONDO:0005147", "type 2 diabetes mellitus", ["T2DM", "adult-onset diabetes"]),
            _row("MONDO:0005148", "type 1 diabetes mellitus", ["T1DM", "AD"]),
            _row(
                "MONDO:0009061",
                "cystic fibrosis",
                [],
                matrix_filter=False,
                prevalence="1-9 in 100,000",
            ),
            _row("MONDO:0000001", "retired disease", [], deleted=True),
        ]
    )
    return DiseaseLookup(df)


def test_exact_name_ignores_case(lookup):
    result = lookup.resolve("PARKINSON DISEASE")
    assert result.status == "exact"
    assert result.best.mondo_id == "MONDO:0005180"
    assert result.best.match_type == "name"


def test_synonym_with_apostrophe_matches(lookup):
    result = lookup.resolve("Alzheimers disease")
    assert result.status == "exact"
    assert result.best.mondo_id == "MONDO:0004975"
    assert result.best.match_type == "synonym"


def test_mondo_id_passthrough(lookup):
    result = lookup.resolve("MONDO:0009061")
    assert result.status == "exact"
    assert result.best.name == "cystic fibrosis"
    assert result.best.match_type == "id"


def test_unknown_mondo_id_is_none_not_error(lookup):
    result = lookup.resolve("MONDO:9999999")
    assert result.status == "none"
    assert result.matches == []


def test_shared_synonym_is_ambiguous(lookup):
    # "AD" is a synonym of two diseases and is not any disease's name.
    result = lookup.resolve("AD")
    assert result.status == "ambiguous"
    assert {m.mondo_id for m in result.matches} == {"MONDO:0004975", "MONDO:0005148"}


def test_name_match_beats_synonym_match():
    df = pd.DataFrame(
        [
            _row("MONDO:1", "flu", []),
            _row("MONDO:2", "influenza", ["flu"]),
        ]
    )
    result = DiseaseLookup(df).resolve("flu")
    assert result.status == "exact"
    assert result.best.mondo_id == "MONDO:1"


def test_deleted_diseases_are_not_returned(lookup):
    assert lookup.resolve("retired disease").status == "none"


def test_misspelling_returns_close_matches(lookup):
    result = lookup.resolve("parkinsen disease")
    assert result.status == "close"
    assert result.matches[0].mondo_id == "MONDO:0005180"
    assert result.matches[0].match_type == "close"


def test_partial_text_returns_contains_matches(lookup):
    result = lookup.resolve("diabetes")
    assert result.status == "close"
    assert {m.mondo_id for m in result.matches} >= {"MONDO:0005147", "MONDO:0005148"}


def test_empty_query_is_none(lookup):
    assert lookup.resolve("   ").status == "none"


def test_match_carries_matrix_filter_and_prevalence(lookup):
    cf = lookup.resolve("cystic fibrosis").best
    assert cf.in_matrix_filter is False
    assert cf.prevalence_world == "1-9 in 100,000"
    pd_match = lookup.resolve("Parkinson disease").best
    assert pd_match.in_matrix_filter is True
    assert lookup.resolve("type 2 diabetes mellitus").best.prevalence_world is None
