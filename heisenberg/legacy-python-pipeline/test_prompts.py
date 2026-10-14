"""Tests for the prompt loader (prompt_loader.py) and the prompt files in prompts/.

No API key and no network: these only read files and render text.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import llm
import prompt_loader as pl
from llm_schemas import SCHEMAS, DrugNameList, ProteinList

GOOD = '''---
name: demo
description: "A demo prompt: with a colon."
input:
  schema:
    disease: string
    round: integer
---
{{role "system"}}
You study {{disease}}.
{{role "user"}}
Round {{round}} of {{disease}}.
'''


def parse(source=GOOD):
    return pl.parse_prompt("demo/demo", source)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_reads_metadata_and_sections():
    t = parse()
    assert t.variables == {"disease": "string", "round": "integer"}
    assert t.description == "A demo prompt: with a colon."
    assert t.system == "You study {{disease}}."
    assert t.user == "Round {{round}} of {{disease}}."
    assert t.output_schema is None


def test_hash_is_stable_and_changes_with_content():
    assert parse().hash == parse().hash
    assert parse(GOOD.replace("You study", "You research")).hash != parse().hash
    assert parse(GOOD.replace("\n", "\r\n")).hash == parse().hash  # CRLF-normalised


@pytest.mark.parametrize("source, message", [
    ("no front matter", "must start with"),
    ("---\nname: x\n", "not closed"),
    (GOOD.replace('{{role "system"}}', '{{role "user"}}'), "exactly one"),
    (GOOD.replace('{{role "user"}}', '{{role "system"}}'), "exactly one"),
    (GOOD.replace('{{role "system"}}\n', "stray text\n" + '{{role "system"}}\n'), "before the first role"),
    (GOOD.replace("{{round}}", "{{rnd}}"), "not declared"),
    (GOOD.replace("Round {{round}} of ", "Round of "), "never used"),
    (GOOD.replace("{{round}}", "{{ round }}"), "malformed placeholder"),
    (GOOD.replace("integer", "float"), "allowed:"),
    (GOOD.replace('You study {{disease}}.', ""), "must not be empty"),
    ("---\n- a list\n---\n" + GOOD.split("---\n", 2)[2], "mapping"),
])
def test_parse_rejects_malformed_prompts(source, message):
    with pytest.raises(pl.PromptError, match=message):
        parse(source)


# ---------------------------------------------------------------------------
# Rendering (against a temporary prompts directory)
# ---------------------------------------------------------------------------

@pytest.fixture
def demo(tmp_path, monkeypatch):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "demo.prompt").write_text(GOOD, encoding="utf-8")
    structured = GOOD.replace("---\n{{role", "output:\n  schema: ProteinList\n---\n{{role", 1)
    (tmp_path / "demo" / "structured.prompt").write_text(structured, encoding="utf-8")
    monkeypatch.setattr(pl, "PROMPTS_DIR", tmp_path)
    pl.load_prompt.cache_clear()
    pl.reset_used_prompts()
    yield tmp_path
    pl.load_prompt.cache_clear()


def test_render_substitutes_variables(demo):
    r = pl.render_prompt("demo/demo", disease="asthma", round=2)
    assert r.system == "You study asthma."
    assert r.user == "Round 2 of asthma."


def test_render_is_single_pass_and_inserts_values_verbatim(demo):
    tricky = "{{round}} $& $1 \\n é {{ x }}"
    r = pl.render_prompt("demo/demo", disease=tricky, round=2)
    assert r.system == f"You study {tricky}."  # {{round}} inside a value is NOT expanded
    assert r.user == f"Round 2 of {tricky}."


def test_render_rejects_wrong_variables(demo):
    with pytest.raises(pl.PromptError, match=r"missing variables \['round'\]"):
        pl.render_prompt("demo/demo", disease="x")
    with pytest.raises(pl.PromptError, match=r"unexpected variables \['extra'\]"):
        pl.render_prompt("demo/demo", disease="x", round=1, extra="y")


@pytest.mark.parametrize("kwargs", [
    {"disease": 5, "round": 1},        # string expected
    {"disease": "x", "round": "1"},    # integer expected
    {"disease": "x", "round": True},   # bool is not an integer
    {"disease": None, "round": 1},
])
def test_render_rejects_wrong_types(demo, kwargs):
    with pytest.raises(pl.PromptError, match="must be a"):
        pl.render_prompt("demo/demo", **kwargs)


@pytest.mark.parametrize("bad_id", ["../etc/passwd", "demo", "demo/", "Demo/demo", "a/b/c", "demo/de mo"])
def test_invalid_ids_are_rejected(demo, bad_id):
    with pytest.raises(pl.PromptError, match="invalid prompt id"):
        pl.load_prompt(bad_id)


def test_unknown_prompt(demo):
    with pytest.raises(pl.PromptError, match="unknown prompt"):
        pl.load_prompt("demo/missing")


def test_used_prompts_records_hashes(demo):
    assert pl.used_prompts() == {}
    r = pl.render_prompt("demo/demo", disease="x", round=1)
    assert pl.used_prompts() == {"demo/demo": r.hash}


def test_query_prompt_refuses_structured_prompts_and_vice_versa(demo, monkeypatch):
    monkeypatch.setattr(llm, "query_openai", lambda s, u: "reply")
    monkeypatch.setattr(llm, "query_openai_structured", lambda s, u, m: m(proteins=[]))
    assert llm.query_prompt("demo/demo", disease="x", round=1) == "reply"
    with pytest.raises(pl.PromptError, match="query_prompt_structured"):
        llm.query_prompt("demo/structured", disease="x", round=1)
    assert llm.query_prompt_structured("demo/structured", ProteinList, disease="x", round=1).proteins == []
    with pytest.raises(pl.PromptError, match="was called with DrugNameList"):
        llm.query_prompt_structured("demo/structured", DrugNameList, disease="x", round=1)
    with pytest.raises(pl.PromptError, match="declares output.schema None"):
        llm.query_prompt_structured("demo/demo", ProteinList, disease="x", round=1)


# ---------------------------------------------------------------------------
# The real prompt files
# ---------------------------------------------------------------------------

REAL_IDS = pl.list_prompt_ids()


def test_there_are_prompts():
    assert len(REAL_IDS) >= 18


@pytest.mark.parametrize("prompt_id", REAL_IDS)
def test_every_shipped_prompt_parses(prompt_id):
    template = pl.load_prompt(prompt_id)
    assert template.id == prompt_id
    assert template.description, "every prompt needs a description"
    assert template.system and template.user
    # front-matter `name` matches the file name
    meta_name = (pl.PROMPTS_DIR / f"{prompt_id}.prompt").read_text("utf-8").split("\nname: ")[1].split("\n")[0]
    assert meta_name == prompt_id.split("/")[1]


@pytest.mark.parametrize("prompt_id", REAL_IDS)
def test_declared_output_schemas_exist_in_code(prompt_id):
    schema = pl.load_prompt(prompt_id).output_schema
    assert schema is None or schema in SCHEMAS, f"{prompt_id}: unknown output.schema {schema!r}"


def test_no_python_source_still_holds_an_inline_prompt():
    """Prompts belong in prompts/, not in code: no module may call the raw
    client with an inline system prompt (only llm.py itself defines it)."""
    here = Path(__file__).parent
    offenders = []
    for path in sorted(here.glob("*.py")):
        if path.name in ("llm.py",) or path.name.startswith("test_"):
            continue
        text = path.read_text("utf-8")
        if "query_openai(" in text or "query_openai_structured(" in text:
            offenders.append(path.name)
    assert not offenders, f"raw LLM calls with inline prompts in: {offenders}"


# ---------------------------------------------------------------------------
# Cross-language parity (the TypeScript twin runs the same fixtures)
# ---------------------------------------------------------------------------

PARITY = pl.PROMPTS_DIR / "_parity"


def test_parity_fixtures_cover_every_prompt():
    vars_ = json.loads((PARITY / "vars.json").read_text("utf-8"))
    rendered = json.loads((PARITY / "rendered.json").read_text("utf-8"))
    assert sorted(vars_) == sorted(rendered) == REAL_IDS


@pytest.mark.parametrize("prompt_id", REAL_IDS)
def test_render_matches_parity_fixture(prompt_id):
    vars_ = json.loads((PARITY / "vars.json").read_text("utf-8"))[prompt_id]
    expected = json.loads((PARITY / "rendered.json").read_text("utf-8"))[prompt_id]
    r = pl.render_prompt(prompt_id, **vars_)
    assert {"hash": r.hash, "system": r.system, "user": r.user} == expected


def test_parity_fixtures_are_up_to_date():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "scripts" / "update_prompt_parity.py"), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
