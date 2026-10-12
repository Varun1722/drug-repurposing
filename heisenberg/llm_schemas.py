"""Pydantic schemas for LLM calls that must return structured JSON.

Each model is sent to OpenAI as a strict ``json_schema`` response format
(see ``review.query_openai_structured``) and then re-validated on the way
back.  Mirrors lib/llm-schemas.ts -- keep the two in sync.

Rules for OpenAI strict mode (checked by ``to_strict_json_schema``):
  - the root is an object, so lists are wrapped (``{"proteins": [...]}``)
  - every field is required: no defaults.  A value that may be absent is
    ``Optional[X]`` *without* a default, i.e. required-but-nullable
  - keys are generated in declaration order, so reasoning fields
    (rationale, hypothesis) come before the decision they justify

Written for Python 3.9: use ``typing.Optional``/``List``, not ``X | None``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    # extra="forbid" emits ``additionalProperties: false``, which strict
    # mode requires on every object.
    model_config = ConfigDict(extra="forbid")


class ProteinList(_StrictModel):
    proteins: List[str] = Field(
        description="Short protein names or gene symbols, e.g. EGFR, BRAF, KRAS G12D"
    )


class DrugNameList(_StrictModel):
    drug_names: List[str] = Field(
        description="Generic (non-brand) drug names, e.g. atorvastatin"
    )


class DrugEntry(_StrictModel):
    drug: str
    proteins: List[str] = Field(
        description="Target protein symbols this drug may bind to"
    )
    mechanism: str
    fda_status: str
    category: Literal["mainstream", "repurposing_candidate"]
    trial_status_for_targets: str = Field(
        description='Clinical-trial status for the listed targets, '
        'or "No known trials for these targets"'
    )


class DrugProteinMap(_StrictModel):
    drugs: List[DrugEntry]


class ReasoningDecision(_StrictModel):
    rationale: str = Field(description="Analysis of the docking patterns, 2-3 sentences")
    hypothesis: str = Field(description="A specific scientific hypothesis, 1 sentence")
    action: Literal["expand_3d_similar", "expand_class", "proceed"]
    seed_cids: Optional[List[int]] = Field(
        description="PubChem CIDs to seed 3D similarity search; "
        "null unless action is expand_3d_similar"
    )
    drug_class: Optional[str] = Field(
        description="Drug class to expand; null unless action is expand_class"
    )
    drug_names: Optional[List[str]] = Field(
        description="Drug names in that class; null unless action is expand_class"
    )


# Every schema a prompt may name in its ``output.schema`` front matter.
SCHEMAS: Dict[str, Type[BaseModel]] = {
    model.__name__: model
    for model in (ProteinList, DrugNameList, DrugProteinMap, ReasoningDecision)
}


# ---------------------------------------------------------------------------
# Strict-mode schema generation
# ---------------------------------------------------------------------------

_FORBIDDEN_KEYWORDS = frozenset({
    "oneOf", "allOf", "not", "if", "then", "else", "default",
    "patternProperties", "propertyNames", "minProperties", "maxProperties",
})
_MAX_OBJECT_DEPTH = 10  # OpenAI's limit on object nesting in a strict schema


def _check(node: Any, path: str, object_depth: int) -> None:
    if not isinstance(node, dict):
        return

    for keyword in _FORBIDDEN_KEYWORDS & node.keys():
        raise ValueError(
            f'Schema at {path} uses "{keyword}", which OpenAI strict mode '
            f"does not support"
        )

    properties = node.get("properties")
    depth = object_depth
    if node.get("type") == "object" or isinstance(properties, dict):
        depth += 1
        if depth > _MAX_OBJECT_DEPTH:
            raise ValueError(
                f"Schema at {path} nests objects more than "
                f"{_MAX_OBJECT_DEPTH} levels deep"
            )
        if node.get("additionalProperties") is not False:
            raise ValueError(f"Object at {path} must set additionalProperties: false")
        missing = set(properties or {}) - set(node.get("required", []))
        if missing:
            raise ValueError(
                f"Object at {path} has non-required properties "
                f"({', '.join(sorted(missing))}). Strict mode needs every "
                f"field required; use Optional[X] with no default for "
                f"nullable values"
            )

    # `properties` / `$defs` map names to schemas: walk the values, so a field
    # called "default" is not mistaken for the keyword.
    for map_key in ("properties", "$defs", "definitions"):
        children = node.get(map_key)
        if isinstance(children, dict):
            for name, child in children.items():
                _check(child, f"{path}/{map_key}/{name}", depth)
    _check(node.get("items"), f"{path}/items", depth)
    for i, variant in enumerate(node.get("anyOf") or []):
        _check(variant, f"{path}/anyOf/{i}", depth)


@lru_cache(maxsize=None)
def to_strict_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """JSON Schema for ``model``, verified to be OpenAI strict-mode safe.

    Raises ``ValueError`` naming the offending path if the model would be
    rejected (a default value, a missing ``required`` key, a root union, ...),
    instead of surfacing as an opaque HTTP 400 at request time.  The returned
    dict is shared via the cache; treat it as read-only.
    """
    schema = model.model_json_schema()
    if schema.get("type") != "object" or "anyOf" in schema:
        raise ValueError(
            "Root schema must be a plain object model; root-level unions "
            "are not supported"
        )
    _check(schema, "#", 0)
    return schema
