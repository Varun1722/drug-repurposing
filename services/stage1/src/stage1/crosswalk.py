"""Map Every Cure ids (MONDO, DrugBank) to TxGNN's PrimeKG node indices.

PrimeKG disease nodes carry either one MONDO number ("5180") or several joined by "_" when
PrimeKG merged near-duplicate diseases ("10820_13060"). Drug nodes carry DrugBank ids.
The `txgnn` and `mondo_txgnn` columns of the Every Cure disease list are disease-area
labels, not ids, so they are not used here.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stage1.paths import RAW

DEFAULT_NODES = RAW / "txgnn_primekg" / "node.csv"


def mondo_ids_from_node_id(node_id: str) -> list[str]:
    return [f"MONDO:{int(part):07d}" for part in node_id.split("_")]


def txgnn_key(node_id: str) -> str:
    """The id string TxGNN keeps internally (its `convert2str`): plain numbers become floats."""
    if "_" in node_id:
        return node_id
    try:
        return str(float(node_id))
    except ValueError:
        return node_id


@dataclass(frozen=True)
class DiseaseNode:
    node_index: int
    node_id: str
    node_name: str
    grouped: bool
    member_mondo_ids: tuple[str, ...]

    @property
    def txgnn_key(self) -> str:
        return txgnn_key(self.node_id)


@dataclass(frozen=True)
class DrugNode:
    node_index: int
    node_id: str
    node_name: str


class PrimeKGCrosswalk:
    def __init__(self, nodes: pd.DataFrame):
        self._diseases: dict[str, DiseaseNode] = {}
        for row in nodes[nodes["node_type"] == "disease"].itertuples(index=False):
            members = tuple(mondo_ids_from_node_id(row.node_id))
            node = DiseaseNode(
                node_index=int(row.node_index),
                node_id=row.node_id,
                node_name=row.node_name,
                grouped=len(members) > 1,
                member_mondo_ids=members,
            )
            for mondo_id in members:
                self._diseases[mondo_id] = node
        self._drugs = {
            row.node_id: DrugNode(int(row.node_index), row.node_id, row.node_name)
            for row in nodes[nodes["node_type"] == "drug"].itertuples(index=False)
        }

    @classmethod
    def from_csv(cls, path: Path = DEFAULT_NODES) -> "PrimeKGCrosswalk":
        return cls(pd.read_csv(path, sep="\t", dtype=str).astype({"node_index": int}))

    def disease(self, mondo_id: str) -> DiseaseNode | None:
        return self._diseases.get(mondo_id)

    def drug(self, drugbank_id: str | None) -> DrugNode | None:
        return None if drugbank_id is None else self._drugs.get(drugbank_id)

    @property
    def n_diseases(self) -> int:
        return len({node.node_index for node in self._diseases.values()})

    @property
    def n_drugs(self) -> int:
        return len(self._drugs)
