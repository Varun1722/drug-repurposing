"""Resolve free-text disease queries to MONDO ids using the Every Cure disease list.

Resolution order: MONDO id, exact name, exact synonym, then close matches (substring, then
fuzzy). A query that does not resolve returns close matches rather than raising.
"""

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stage1.paths import RAW

DEFAULT_PARQUET = RAW / "disease_list" / "data" / "train-00000-of-00001.parquet"
_MONDO_ID = re.compile(r"^MONDO:\d+$", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise(text: str) -> str:
    """Lowercase, drop apostrophes, and collapse every other symbol run to one space."""
    lowered = text.lower().replace("'", "").replace("’", "")
    return _NON_ALNUM.sub(" ", lowered).strip()


@dataclass(frozen=True)
class DiseaseMatch:
    mondo_id: str
    name: str
    match_type: str  # id | name | synonym | contains | close
    matched_text: str
    in_matrix_filter: bool
    prevalence_world: str | None


@dataclass(frozen=True)
class LookupResult:
    query: str
    status: str  # exact | ambiguous | close | none
    matches: list[DiseaseMatch]

    @property
    def best(self) -> DiseaseMatch | None:
        return self.matches[0] if self.matches else None


class DiseaseLookup:
    def __init__(self, diseases: pd.DataFrame):
        live = diseases[~diseases["deleted"].astype(bool)]
        self._records = {row.id: row for row in live.itertuples(index=False)}
        self._by_name: dict[str, list[str]] = {}
        self._by_synonym: dict[str, list[str]] = {}
        for row in self._records.values():
            self._by_name.setdefault(normalise(row.name), []).append(row.id)
            for synonym in row.synonyms if row.synonyms is not None else []:
                key = normalise(synonym)
                ids = self._by_synonym.setdefault(key, [])
                if row.id not in ids:
                    ids.append(row.id)

    @classmethod
    def from_parquet(cls, path: Path = DEFAULT_PARQUET) -> "DiseaseLookup":
        return cls(pd.read_parquet(path))

    def _match(self, mondo_id: str, match_type: str, matched_text: str) -> DiseaseMatch:
        row = self._records[mondo_id]
        prevalence = row.prevalence_world
        return DiseaseMatch(
            mondo_id=mondo_id,
            name=row.name,
            match_type=match_type,
            matched_text=matched_text,
            in_matrix_filter=bool(row.official_matrix_filter),
            prevalence_world=None if pd.isna(prevalence) else str(prevalence),
        )

    def resolve(self, query: str, limit: int = 5) -> LookupResult:
        text = query.strip()
        key = normalise(text)
        if not key:
            return LookupResult(query, "none", [])

        if _MONDO_ID.match(text):
            mondo_id = "MONDO:" + text.split(":")[1]
            if mondo_id in self._records:
                return LookupResult(query, "exact", [self._match(mondo_id, "id", mondo_id)])
            return LookupResult(query, "none", [])

        for index, match_type in ((self._by_name, "name"), (self._by_synonym, "synonym")):
            ids = index.get(key)
            if ids:
                matches = [self._match(i, match_type, text) for i in ids]
                return LookupResult(query, "exact" if len(ids) == 1 else "ambiguous", matches)

        close = self._close_matches(key, limit)
        return LookupResult(query, "close" if close else "none", close)

    def _close_matches(self, key: str, limit: int) -> list[DiseaseMatch]:
        seen: set[str] = set()
        out: list[DiseaseMatch] = []

        def add(mondo_id: str, match_type: str, matched: str) -> None:
            if mondo_id not in seen and len(out) < limit:
                seen.add(mondo_id)
                out.append(self._match(mondo_id, match_type, matched))

        contains = sorted(
            (name for name in self._by_name if key in name), key=lambda n: (len(n), n)
        )
        for name in contains:
            for mondo_id in self._by_name[name]:
                add(mondo_id, "contains", name)

        fuzzy = difflib.get_close_matches(key, list(self._by_name), n=limit, cutoff=0.8)
        for name in fuzzy:
            for mondo_id in self._by_name[name]:
                add(mondo_id, "close", name)
        return out
