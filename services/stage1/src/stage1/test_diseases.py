"""T1: the 20 diseases used for first manual tests, resolved to MONDO and TxGNN ids.

Expected MONDO ids are pinned so that a change in the Every Cure list shows up as an error
instead of silently swapping the disease under test.
"""

import pandas as pd

from stage1.crosswalk import PrimeKGCrosswalk
from stage1.disease_lookup import DiseaseLookup
from stage1.paths import ROOT

OUTPUT = ROOT / "config" / "test_diseases.csv"

# (query, expected MONDO id, group, why it is in the set)
TEST_DISEASES = [
    (
        "type 2 diabetes mellitus",
        "MONDO:0005148",
        "common",
        "Very common; many approved drugs, so it exposes popularity bias",
    ),
    (
        "Alzheimer disease",
        "MONDO:0004975",
        "common",
        "Large unmet need, many failed trials; merged PrimeKG node of 21 diseases",
    ),
    (
        "rheumatoid arthritis",
        "MONDO:0008383",
        "common",
        "Autoimmune; many approved drugs; JAK-inhibitor-style repurposing precedent",
    ),
    (
        "asthma",
        "MONDO:0004979",
        "common",
        "Common respiratory disease; outside the MATRIX official filter",
    ),
    (
        "major depressive disorder",
        "MONDO:0002009",
        "common",
        "Psychiatric; different drug classes from the other common diseases",
    ),
    (
        "essential hypertension",
        "MONDO:0001134",
        "common",
        "Common cardiovascular disease; merged PrimeKG node; outside the MATRIX filter",
    ),
    (
        "Parkinson disease",
        "MONDO:0005180",
        "common",
        "Neurodegenerative; merged PrimeKG node of 18 diseases; active repurposing trials",
    ),
    (
        "ulcerative colitis",
        "MONDO:0005101",
        "common",
        "Inflammatory bowel disease; known repurposing history",
    ),
    (
        "psoriasis",
        "MONDO:0005083",
        "common",
        "Dermatology; many approved drugs; outside the MATRIX filter",
    ),
    (
        "tuberculosis",
        "MONDO:0018076",
        "common",
        "Infection; antimicrobial rather than chronic-disease drugs",
    ),
    (
        "pancreatic ductal adenocarcinoma",
        "MONDO:0005184",
        "hard",
        "Cancer with few effective drugs; tests cancer coverage",
    ),
    (
        "idiopathic pulmonary fibrosis",
        "MONDO:0800504",
        "hard",
        "Not a PrimeKG disease, so it exercises the MATRIX-only path",
    ),
    (
        "nonalcoholic steatohepatitis",
        "MONDO:0007027",
        "hard",
        "Now called MASH; many repurposing trials; tests synonym lookup",
    ),
    (
        "heart failure",
        "MONDO:0005252",
        "hard",
        "SGLT2 inhibitors moved from diabetes to heart failure, a real repurposing success",
    ),
    (
        "amyotrophic lateral sclerosis",
        "MONDO:0004976",
        "rare",
        "Few drugs; merged PrimeKG node of 21 diseases",
    ),
    (
        "cystic fibrosis",
        "MONDO:0009061",
        "rare",
        "Rare; mostly targeted therapies, so few repurposing candidates",
    ),
    (
        "Duchenne muscular dystrophy",
        "MONDO:0010679",
        "rare",
        "Rare; steroid standard of care; active trial pipeline",
    ),
    (
        "Huntington disease",
        "MONDO:0007739",
        "rare",
        "Rare (under 1 in 100,000); no disease-modifying drug, so a cold-start case",
    ),
    ("Friedreich ataxia", "MONDO:0100339", "rare", "Rare; very few drugs; merged PrimeKG node"),
    ("Rett syndrome", "MONDO:0010726", "rare", "Rare; very few drugs; a cold-start case"),
]


def build_table(lookup: DiseaseLookup, crosswalk: PrimeKGCrosswalk) -> pd.DataFrame:
    rows = []
    for query, expected, group, why in TEST_DISEASES:
        result = lookup.resolve(query)
        match = result.best
        if result.status != "exact" or match.mondo_id != expected:
            found = [(m.mondo_id, m.name) for m in result.matches]
            raise ValueError(f"{query!r}: expected {expected}, got {result.status} {found}")
        node = crosswalk.disease(match.mondo_id)
        rows.append(
            {
                "query": query,
                "mondo_id": match.mondo_id,
                "name": match.name,
                "group": group,
                "why": why,
                "in_matrix_filter": match.in_matrix_filter,
                "prevalence_world": match.prevalence_world,
                "txgnn_node_id": None if node is None else node.node_id,
                "txgnn_node_index": None if node is None else node.node_index,
                "txgnn_node_name": None if node is None else node.node_name,
                "txgnn_grouped_members": 0 if node is None else len(node.member_mondo_ids),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    table = build_table(DiseaseLookup.from_parquet(), PrimeKGCrosswalk.from_csv())
    OUTPUT.parent.mkdir(exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(table[["mondo_id", "name", "group", "in_matrix_filter", "txgnn_node_index"]].to_string())
    unmapped = table[table["txgnn_node_index"].isna()]["name"].tolist()
    print(f"\nwrote {OUTPUT}\nno TxGNN node: {unmapped}")


if __name__ == "__main__":
    main()
