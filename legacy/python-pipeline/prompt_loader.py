"""Load and render prompt files from ``prompts/``.

A prompt is one ``prompts/<group>/<name>.prompt`` file (format documented in
prompts/README.md): YAML front matter, then a ``{{role "system"}}`` section
and a ``{{role "user"}}`` section containing ``{{variable}}`` placeholders.

Rendering is deliberately strict -- a missing, extra, mistyped or unknown
variable raises ``PromptError`` instead of silently producing a half-filled
prompt.  Values are substituted in a single pass and never re-scanned, so text
that happens to contain ``{{...}}`` (e.g. a paper abstract) is inserted
verbatim.

The TypeScript twin is lib/prompts.ts; both implement the same spec and are
checked against prompts/_parity/ so they render byte-identical text.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_ID_RE = re.compile(r"^[a-z0-9_]+/[a-z0-9_]+$")
_ROLE_RE = re.compile(r'^\{\{role "(system|user)"\}\}[ \t]*$', re.MULTILINE)
_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_TRAILING_WHITESPACE = " \t\n\r\f\v"  # ASCII only, so Python and TypeScript agree
_VARIABLE_TYPES = ("string", "integer", "boolean")

# id -> content hash of every prompt rendered in this process, so a run can
# record exactly which prompt versions produced its output.
_USED: Dict[str, str] = {}


class PromptError(Exception):
    """A prompt file is malformed, or it was rendered with the wrong variables."""


class RenderedPrompt(NamedTuple):
    id: str
    hash: str
    system: str
    user: str
    output_schema: Optional[str]


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    hash: str
    description: str
    variables: Dict[str, str]  # name -> "string" | "integer" | "boolean"
    output_schema: Optional[str]
    system: str
    user: str


def parse_prompt(prompt_id: str, source: str) -> PromptTemplate:
    """Parse the text of a ``.prompt`` file.  Raises ``PromptError`` if malformed."""
    text = source.replace("\r\n", "\n")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    if not text.startswith("---\n"):
        raise PromptError(f"{prompt_id}: file must start with '---' front matter")
    end = text.find("\n---\n", 3)
    if end == -1:
        raise PromptError(f"{prompt_id}: front matter is not closed with '---'")
    try:
        meta = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError as e:
        raise PromptError(f"{prompt_id}: invalid front matter YAML: {e}") from e
    if not isinstance(meta, dict):
        raise PromptError(f"{prompt_id}: front matter must be a mapping")
    body = text[end + len("\n---\n"):]

    variables = _read_variables(prompt_id, meta)
    output_schema = _read_output_schema(prompt_id, meta)
    system, user = _split_roles(prompt_id, body)

    used = set(_PLACEHOLDER_RE.findall(system)) | set(_PLACEHOLDER_RE.findall(user))
    for section_name, section in (("system", system), ("user", user)):
        leftover = _PLACEHOLDER_RE.sub("", section)
        if "{{" in leftover or "}}" in leftover:
            raise PromptError(
                f"{prompt_id}: malformed placeholder in the {section_name} section "
                f"(placeholders are exactly {{{{name}}}}, no spaces)"
            )
    undeclared = sorted(used - set(variables))
    unused = sorted(set(variables) - used)
    if undeclared:
        raise PromptError(f"{prompt_id}: placeholders not declared in input.schema: {undeclared}")
    if unused:
        raise PromptError(f"{prompt_id}: input.schema variables never used in the body: {unused}")

    return PromptTemplate(
        id=prompt_id,
        hash=digest,
        description=str(meta.get("description", "")),
        variables=variables,
        output_schema=output_schema,
        system=system,
        user=user,
    )


def _read_variables(prompt_id: str, meta: dict) -> Dict[str, str]:
    schema = (meta.get("input") or {}).get("schema") or {}
    if not isinstance(schema, dict):
        raise PromptError(f"{prompt_id}: input.schema must be a mapping of name -> type")
    variables: Dict[str, str] = {}
    for name, var_type in schema.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise PromptError(f"{prompt_id}: invalid variable name {name!r}")
        if var_type not in _VARIABLE_TYPES:
            raise PromptError(
                f"{prompt_id}: variable {name!r} has type {var_type!r}; "
                f"allowed: {', '.join(_VARIABLE_TYPES)}"
            )
        variables[name] = var_type
    return variables


def _read_output_schema(prompt_id: str, meta: dict) -> Optional[str]:
    schema = (meta.get("output") or {}).get("schema")
    if schema is not None and not isinstance(schema, str):
        raise PromptError(f"{prompt_id}: output.schema must be the name of a code-defined schema")
    return schema


def _split_roles(prompt_id: str, body: str) -> tuple:
    markers = list(_ROLE_RE.finditer(body))
    if [m.group(1) for m in markers] != ["system", "user"]:
        found = [m.group(1) for m in markers]
        raise PromptError(
            f'{prompt_id}: body needs exactly one {{{{role "system"}}}} followed by one '
            f'{{{{role "user"}}}} marker (found {found})'
        )
    if body[: markers[0].start()].strip():
        raise PromptError(f"{prompt_id}: text before the first role marker")

    def section(start: int, stop: int) -> str:
        # Drop the newline that ends the marker line and trailing whitespace;
        # everything else is the prompt text, byte for byte.
        text = body[start:stop]
        return (text[1:] if text.startswith("\n") else text).rstrip(_TRAILING_WHITESPACE)

    system = section(markers[0].end(), markers[1].start())
    user = section(markers[1].end(), len(body))
    if not system or not user:
        raise PromptError(f"{prompt_id}: system and user sections must not be empty")
    return system, user


@lru_cache(maxsize=None)
def load_prompt(prompt_id: str) -> PromptTemplate:
    if not _ID_RE.match(prompt_id):
        raise PromptError(f"invalid prompt id {prompt_id!r} (expected '<group>/<name>')")
    path = PROMPTS_DIR / f"{prompt_id}.prompt"
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise PromptError(f"unknown prompt {prompt_id!r} (no file {path})") from None
    return parse_prompt(prompt_id, source)


def render_prompt(prompt_id: str, **variables) -> RenderedPrompt:
    """Render ``prompt_id`` with ``variables``.  Every declared variable must be
    supplied with the declared type, and nothing else may be passed."""
    template = load_prompt(prompt_id)

    missing = sorted(set(template.variables) - set(variables))
    extra = sorted(set(variables) - set(template.variables))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing variables {missing}")
        if extra:
            parts.append(f"unexpected variables {extra}")
        raise PromptError(f"{prompt_id}: " + "; ".join(parts))

    values = {name: _to_text(prompt_id, name, template.variables[name], value)
              for name, value in variables.items()}

    def substitute(section: str) -> str:
        return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], section)

    _USED[prompt_id] = template.hash
    return RenderedPrompt(
        id=prompt_id,
        hash=template.hash,
        system=substitute(template.system),
        user=substitute(template.user),
        output_schema=template.output_schema,
    )


def _to_text(prompt_id: str, name: str, var_type: str, value: object) -> str:
    if var_type == "string" and isinstance(value, str):
        return value
    if var_type == "integer" and isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if var_type == "boolean" and isinstance(value, bool):
        return "true" if value else "false"
    raise PromptError(
        f"{prompt_id}: variable {name!r} must be a {var_type}, got {type(value).__name__}"
    )


def list_prompt_ids() -> List[str]:
    """Every prompt id on disk (directories starting with '_' are not prompts)."""
    ids = []
    for path in sorted(PROMPTS_DIR.glob("*/*.prompt")):
        if path.parent.name.startswith("_"):
            continue
        ids.append(f"{path.parent.name}/{path.stem}")
    return ids


def used_prompts() -> Dict[str, str]:
    """``{prompt id: content hash}`` for every prompt rendered so far in this process."""
    return dict(_USED)


def reset_used_prompts() -> None:
    _USED.clear()
