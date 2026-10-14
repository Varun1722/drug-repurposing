"""Pipeline-level failure handling: what gets recorded in ``state["errors"]``,
when a stage degrades, and when the run stops.

Every external call is replaced by a fake; files are written under tmp_path.
"""

import json

import pytest

pytest.importorskip("rdkit")  # pipeline imports agent2, which needs RDKit

import pipeline  # noqa: E402
from llm import LLMOutputError  # noqa: E402
from resilience import EscalationRequired, ExternalCallError  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Run in a scratch dir (pipeline_state.json etc.) without sleeping."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("time.sleep", lambda _: None)


def _err(source="pubchem", kind="server"):
    return ExternalCallError(source, kind, f"{source}: {kind} (HTTP 503) after 4 attempt(s)",
                             status=503, attempts=4, retryable=True)


def _state(**overrides):
    state = pipeline.new_state("glioblastoma")
    state.update(overrides)
    return state


def _drugs(*names, protein="MGMT"):
    return [{"drug": n, "proteins": [protein], "mechanism": "m", "fda_status": "Approved"}
            for n in names]


@pytest.fixture
def structure_fakes(monkeypatch):
    """Healthy RCSB / PubChem fakes for stage_structure; tests override pieces."""
    monkeypatch.setattr(pipeline, "search_pdb", lambda p: [{"pdb_id": "1ABC"}])
    monkeypatch.setattr(pipeline, "pick_best_structure", lambda c: {"pdb_id": "1ABC"})
    monkeypatch.setattr(pipeline, "download_pdb", lambda i, d: f"{d}/{i}.pdb")
    monkeypatch.setattr(pipeline, "lookup_smiles",
                        lambda n: {"cid": 1, "smiles": "C" * len(n)})
    monkeypatch.setattr(pipeline, "search_compounds_for_target", lambda p, m: [])


# ---------------------------------------------------------------------------
# State basics
# ---------------------------------------------------------------------------

def test_new_state_starts_with_no_errors():
    assert _state()["errors"] == []


def test_state_files_from_before_error_recording_load_with_an_errors_list(tmp_path):
    (tmp_path / pipeline.STATE_FILE).write_text(json.dumps({"disease": "d", "status": "x"}))
    assert pipeline.load_state()["errors"] == []


def test_run_stage_records_a_fatal_error_saves_state_and_reraises():
    state = _state(status="structures_complete")

    def boom(s):
        raise _err("gnina", "no_results")

    with pytest.raises(ExternalCallError):
        pipeline._run_stage(state, "docking", boom)

    saved = json.load(open(pipeline.STATE_FILE))
    assert saved["status"] == "structures_complete"        # --resume re-runs the failed stage
    assert [(e["stage"], e["source"], e["kind"], e["fatal"]) for e in saved["errors"]] == [
        ("docking", "gnina", "no_results", True)]


def test_run_stage_passes_the_result_through():
    state = _state()
    assert pipeline._run_stage(state, "x", lambda s, a, b=0: (s is state, a, b), 1, b=2) == (True, 1, 2)
    assert state["errors"] == []


# ---------------------------------------------------------------------------
# stage_structure
# ---------------------------------------------------------------------------

def test_a_failed_lookup_skips_that_drug_and_is_recorded(structure_fakes, monkeypatch):
    def lookup(name):
        if name == "flaky":
            raise _err()
        return {"cid": 1, "smiles": "CCO"}

    monkeypatch.setattr(pipeline, "lookup_smiles", lookup)
    state = _state(protein_targets=["MGMT"], drugs=_drugs("flaky", "fine"))
    pipeline.stage_structure(state)

    assert [l["name"] for l in state["targets"][0]["ligands"]] == ["fine"]
    assert [(e["stage"], e["source"], e["target"], e["fatal"]) for e in state["errors"]] == [
        ("structure", "pubchem", "MGMT/flaky", False)]
    assert state["status"] == "structures_complete"


def test_not_found_is_not_an_error(structure_fakes, monkeypatch):
    monkeypatch.setattr(pipeline, "lookup_smiles", lambda n: None)
    state = _state(protein_targets=["MGMT"], drugs=_drugs("ghost"))
    pipeline.stage_structure(state)

    assert state["targets"][0]["ligands"] == [] and state["errors"] == []


def test_three_consecutive_pubchem_failures_escalate_with_the_errors_saved(structure_fakes, monkeypatch):
    def always_down(name):
        raise _err()

    monkeypatch.setattr(pipeline, "lookup_smiles", always_down)
    state = _state(protein_targets=["MGMT"], drugs=_drugs(*"abcde"))

    with pytest.raises(EscalationRequired) as exc:
        pipeline.stage_structure(state)

    assert exc.value.source == "pubchem" and "--resume" in str(exc.value)
    assert len(state["errors"]) == 3          # recorded before the guard tripped; d, e never tried


def test_a_success_between_failures_prevents_escalation(structure_fakes, monkeypatch):
    results = iter([_err(), _err(), {"cid": 1, "smiles": "CC"}, _err(), _err(), {"cid": 2, "smiles": "CCC"}])

    def lookup(name):
        r = next(results)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(pipeline, "lookup_smiles", lookup)
    state = _state(protein_targets=["MGMT"], drugs=_drugs(*"abcdef"))
    pipeline.stage_structure(state)         # no EscalationRequired

    assert len(state["errors"]) == 4 and len(state["targets"][0]["ligands"]) == 2


def test_failed_pdb_download_keeps_the_target_without_a_file(structure_fakes, monkeypatch):
    def download(pdb_id, directory):
        raise _err("rcsb", "not_found")

    monkeypatch.setattr(pipeline, "download_pdb", download)
    state = _state(protein_targets=["MGMT"], drugs=_drugs("x"))
    pipeline.stage_structure(state)

    target = state["targets"][0]
    assert target["pdb_id"] == "1ABC" and target["pdb_file"] is None
    assert [(e["source"], e["kind"], e["target"]) for e in state["errors"]] == [
        ("rcsb", "not_found", "MGMT")]


def test_failed_pdb_search_still_tries_the_bare_gene_name(structure_fakes, monkeypatch):
    queries = []

    def search(name):
        queries.append(name)
        if name == "KRAS G12D":
            raise _err("rcsb", "server")
        return [{"pdb_id": "4OBE"}]

    monkeypatch.setattr(pipeline, "search_pdb", search)
    state = _state(protein_targets=["KRAS G12D"], drugs=[])
    pipeline.stage_structure(state)

    assert queries == ["KRAS G12D", "KRAS"]
    assert state["targets"][0]["pdb_id"] == "1ABC"  # fake picker
    assert len(state["errors"]) == 1


def test_failed_compound_search_is_recorded_and_the_known_drugs_are_kept(structure_fakes, monkeypatch):
    def search(protein, n):
        raise _err()

    monkeypatch.setattr(pipeline, "search_compounds_for_target", search)
    state = _state(protein_targets=["MGMT"], drugs=_drugs("known"))
    pipeline.stage_structure(state)

    assert [l["name"] for l in state["targets"][0]["ligands"]] == ["known"]
    assert state["errors"][0]["target"] == "MGMT"


# ---------------------------------------------------------------------------
# stage_docking
# ---------------------------------------------------------------------------

def _target(protein, tmp_path, ligands=("a", "b"), with_file=True):
    pdb = tmp_path / f"{protein}.pdb"
    if with_file:
        pdb.write_text("HEADER\nEND\n")
    return {"protein": protein, "pdb_id": "1ABC", "pdb_file": str(pdb),
            "ligands": [{"name": n, "smiles": "C", "mechanism": "m",
                         "fda_status": "A", "source": "s"} for n in ligands]}


def _fake_docking(monkeypatch, per_protein):
    """run_docking fake: ``per_protein[pdb stem]`` is a result list, an
    exception to raise, or a ``(results, [error dicts])`` pair."""
    def run_docking(protein_pdb_path, ligands, errors=None, **kwargs):
        outcome = per_protein[protein_pdb_path.rsplit("/", 1)[1][:-4]]
        if isinstance(outcome, Exception):
            raise outcome
        results, chunk_errors = outcome if isinstance(outcome, tuple) else (outcome, [])
        errors.extend(chunk_errors)
        return results, 1.0

    monkeypatch.setattr(pipeline, "run_docking", run_docking)


def _res(name, score=0.5):
    return {"name": name, "confidence_score": score, "confidence_raw": score}


def test_partial_docking_failure_is_recorded_per_target(tmp_path, monkeypatch):
    _fake_docking(monkeypatch, {
        "KRAS": ([_res("a")], [{"kind": "server", "message": "chunk 1 failed",
                                "chunk": 1, "count": 1}]),
        "EGFR": _err("gnina", "server"),
    })
    state = _state(round=1, targets=[_target("KRAS", tmp_path), _target("EGFR", tmp_path)])
    pipeline.stage_docking(state)

    assert [r["name"] for r in state["all_docking_results"]] == ["a"]
    assert state["all_docking_results"][0]["protein_target"] == "KRAS"
    summary = [(e["target"], e["source"], e["kind"]) for e in state["errors"]]
    assert ("KRAS", "gnina", "server") in summary and ("EGFR", "gnina", "server") in summary
    chunk = next(e for e in state["errors"] if e["target"] == "KRAS")
    assert chunk["chunk"] == 1 and chunk["count"] == 1 and chunk["stage"] == "docking"
    assert state["status"] == "docking_round_1_complete"


def test_targets_without_a_structure_or_ligands_are_recorded_as_skipped(tmp_path, monkeypatch):
    _fake_docking(monkeypatch, {"KRAS": [_res("a")]})
    state = _state(round=1, targets=[
        _target("KRAS", tmp_path),
        _target("NOPDB", tmp_path, with_file=False),
        _target("NOLIG", tmp_path, ligands=()),
    ])
    (tmp_path / "NOLIG.pdb").write_text("x")
    pipeline.stage_docking(state)

    assert {(e["target"], e["kind"]) for e in state["errors"]} == {
        ("NOPDB", "skipped_no_structure"), ("NOLIG", "skipped_no_ligands")}


def test_round_one_with_nothing_docked_stops_the_run(tmp_path, monkeypatch):
    _fake_docking(monkeypatch, {"KRAS": _err("gnina", "server")})
    state = _state(round=1, targets=[_target("KRAS", tmp_path)])

    with pytest.raises(ExternalCallError) as exc:
        pipeline.stage_docking(state)

    assert (exc.value.source, exc.value.kind) == ("gnina", "no_results")
    assert state["all_docking_results"] == []
    assert state["status"] != "docking_round_1_complete"   # round 1 did not complete


def test_an_empty_expansion_round_is_recorded_not_fatal(tmp_path, monkeypatch):
    _fake_docking(monkeypatch, {"KRAS": _err("gnina", "server")})
    state = _state(round=2, targets=[_target("KRAS", tmp_path)],
                   all_docking_results=[_res("earlier")])
    pipeline.stage_docking(state)           # no raise

    assert [r["name"] for r in state["all_docking_results"]] == ["earlier"]
    assert ("docking", "gnina", "no_results", False) in [
        (e["stage"], e["source"], e["kind"], e["fatal"]) for e in state["errors"]]


# ---------------------------------------------------------------------------
# Reasoning + expansion
# ---------------------------------------------------------------------------

def test_an_openai_outage_in_reasoning_is_not_turned_into_proceed(monkeypatch):
    def down(*a, **k):
        raise _err("openai", "server")

    monkeypatch.setattr(pipeline, "query_prompt_structured", down)
    state = _state(round=1, all_docking_results=[_res("a")])
    with pytest.raises(ExternalCallError):
        pipeline.analyze_and_decide(state)
    assert state["expansion_history"] == []     # nothing was decided


@pytest.fixture
def expansion_state():
    return _state(
        round=1, protein_targets=["KRAS"],
        targets=[{"protein": "KRAS", "pdb_id": "1", "pdb_file": "k.pdb", "ligands": []}],
        all_docking_results=[{**_res("seed", 0.9), "source": "pubchem_cid_100"},
                             {**_res("seed2", 0.8), "source": "pubchem_cid_200"}],
        expansion_history=[{"round": 1, "action": "expand_3d_similar", "rationale": "r"}],
    )


def test_expansion_survives_a_failed_cid_and_tells_the_next_round(expansion_state, monkeypatch):
    def similar(cid, max_results):
        if cid == 100:
            raise _err()
        return [{"cid": 5, "smiles": "CCO", "iupac_name": "ethanol"}]

    docked = []
    monkeypatch.setattr(pipeline, "search_3d_similar", similar)
    monkeypatch.setattr(pipeline, "stage_docking",
                        lambda s, targets_to_dock=None: docked.append(targets_to_dock) or s)
    decision = {"action": "expand_3d_similar", "seed_cids": [100, 200]}
    pipeline.execute_expansion(expansion_state, decision)

    assert [t["ligands"][0]["name"] for t in docked[0]] == ["ethanol"]     # CID 200 still expanded
    outcome = expansion_state["expansion_history"][-1]["outcome"]
    assert "1 new ligand entries found" in outcome and "1 PubChem call(s) failed (server)" in outcome
    assert [(e["stage"], e["target"]) for e in expansion_state["errors"]] == [("expansion", "CID 100")]
    # ... and it is saved, so the next reasoning prompt (which gets expansion_history) sees it
    assert "outcome" in json.load(open(pipeline.STATE_FILE))["expansion_history"][-1]


def test_expansion_that_finds_nothing_because_pubchem_is_down_says_so(expansion_state, monkeypatch):
    def down(cid, max_results):
        raise _err()

    monkeypatch.setattr(pipeline, "search_3d_similar", down)
    monkeypatch.setattr(pipeline, "stage_docking", lambda *a, **k: pytest.fail("nothing to dock"))
    pipeline.execute_expansion(expansion_state, {"action": "expand_3d_similar", "seed_cids": [100, 200]})

    outcome = expansion_state["expansion_history"][-1]["outcome"]
    assert outcome == "0 new ligand entries found; 2 PubChem call(s) failed (server)"
    assert len(expansion_state["errors"]) == 2      # no escalation during expansion


def test_unusable_drug_name_reply_is_recorded(expansion_state, monkeypatch):
    def refused(*a, **k):
        raise LLMOutputError("refusal", "Model refused: no")

    monkeypatch.setattr(pipeline, "query_prompt_structured", refused)
    pipeline.execute_expansion(
        expansion_state, {"action": "expand_class", "drug_class": "statins", "drug_names": []})

    assert [(e["stage"], e["source"], e["kind"]) for e in expansion_state["errors"]] == [
        ("expansion", "openai", "refusal")]
    assert expansion_state["expansion_history"][-1]["outcome"].startswith("0 new ligand")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_report_discloses_non_fatal_errors_and_leaves_fatal_ones_out(monkeypatch):
    import results

    monkeypatch.setattr(results, "load_docking_data", lambda path: {"targets": []})
    monkeypatch.setattr(pipeline, "summarise_target", lambda t: t)
    monkeypatch.setattr(pipeline, "build_overview_block", lambda d, s: "overview")
    monkeypatch.setattr(pipeline, "build_classified_overview", lambda s, d: "classified")
    for name in ("methodology", "results", "conclusion"):
        monkeypatch.setattr(pipeline, f"generate_{name}", lambda *a, _n=name, **k: f"{_n} text")

    state = _state(errors=[
        {"stage": "structure", "source": "pubchem", "kind": "server", "message": "lookup failed",
         "target": "MGMT/flaky", "fatal": False},
        {"stage": "report", "source": "openai", "kind": "server", "message": "dead", "fatal": True},
    ])
    pipeline.stage_report(state)

    md = state["results_md"]
    assert "### Data Quality Notes" in md
    assert "structure / pubchem / server (MGMT/flaky): lookup failed" in md
    assert "dead" not in md
    assert md.index("Data Quality Notes") < md.index("## Conclusion")


def test_report_has_no_data_quality_section_when_nothing_failed(monkeypatch):
    import results

    monkeypatch.setattr(results, "load_docking_data", lambda path: {"targets": []})
    monkeypatch.setattr(pipeline, "build_overview_block", lambda d, s: "o")
    monkeypatch.setattr(pipeline, "build_classified_overview", lambda s, d: "c")
    for name in ("methodology", "results", "conclusion"):
        monkeypatch.setattr(pipeline, f"generate_{name}", lambda *a, _n=name, **k: f"{_n} text")
    state = _state()
    pipeline.stage_report(state)
    assert "Data Quality Notes" not in state["results_md"]


# ---------------------------------------------------------------------------
# run / resume / CLI
# ---------------------------------------------------------------------------

@pytest.fixture
def stages(monkeypatch):
    """Replace every stage with a recorder; returns the call list."""
    calls = []

    def stage(name, status):
        def fn(state, *args, **kwargs):
            calls.append(name)
            state["status"] = status
            return state
        return fn

    monkeypatch.setattr(pipeline, "stage_literature", stage("literature", "literature_complete"))
    monkeypatch.setattr(pipeline, "stage_structure", stage("structure", "structures_complete"))
    monkeypatch.setattr(pipeline, "stage_docking", stage("docking", "docking_round_1_complete"))
    monkeypatch.setattr(pipeline, "stage_report", stage("report", "report_complete"))
    monkeypatch.setattr(pipeline, "stage_paper", stage("paper", "complete"))
    monkeypatch.setattr(pipeline, "analyze_and_decide",
                        lambda state: calls.append("decide") or {"action": "proceed"})
    return calls


def _save(state):
    pipeline.save_state(state)


def test_resume_from_initialized_reruns_literature_first(stages):
    _save(_state(status="initialized"))
    pipeline.resume_pipeline()
    assert stages == ["literature", "structure", "docking", "decide", "report", "paper"]


def test_resume_from_structures_complete_skips_structure_retrieval(stages):
    _save(_state(status="structures_complete"))
    pipeline.resume_pipeline()
    assert stages == ["docking", "decide", "report", "paper"]


def test_resume_from_literature_complete_is_unchanged(stages):
    _save(_state(status="literature_complete"))
    pipeline.resume_pipeline()
    assert stages == ["structure", "docking", "decide", "report", "paper"]


def test_resume_after_docking_goes_straight_to_the_report(stages, monkeypatch):
    monkeypatch.setattr(pipeline, "_write_docking_output", lambda s: None)
    _save(_state(status="docking_round_1_complete"))
    pipeline.resume_pipeline()
    assert stages == ["report", "paper"]


def test_a_failure_during_resume_is_recorded_and_the_status_stays(stages, monkeypatch):
    def boom(state, *a, **k):
        raise _err("gnina", "no_results")

    monkeypatch.setattr(pipeline, "stage_docking", boom)
    _save(_state(status="structures_complete"))

    with pytest.raises(ExternalCallError):
        pipeline.resume_pipeline()

    saved = json.load(open(pipeline.STATE_FILE))
    assert saved["status"] == "structures_complete"
    assert saved["errors"][-1]["fatal"] is True and saved["errors"][-1]["stage"] == "docking"


def test_cli_stops_with_a_resume_hint_when_a_stage_fails(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["pipeline.py", "glioblastoma"])

    def boom(disease, max_rounds):
        raise EscalationRequired("pubchem", 3, _err())

    monkeypatch.setattr(pipeline, "run_pipeline", boom)
    with pytest.raises(SystemExit) as exc:
        pipeline.main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "PIPELINE STOPPED" in err and "python pipeline.py --resume" in err


def test_a_programming_error_is_not_dressed_up_as_a_pipeline_stop(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pipeline.py", "glioblastoma"])

    def bug(disease, max_rounds):
        raise KeyError("bug")

    monkeypatch.setattr(pipeline, "run_pipeline", bug)
    with pytest.raises(KeyError):
        pipeline.main()


# ---------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------

def test_expansion_outcome_reports_what_was_actually_docked(expansion_state, monkeypatch):
    """Docking can fail after the search succeeded; the next reasoning step must be told."""
    monkeypatch.setattr(pipeline, "search_3d_similar",
                        lambda cid, n: [{"cid": 5, "smiles": "CCO", "iupac_name": "ethanol"}])
    monkeypatch.setattr(pipeline, "stage_docking",
                        lambda s, targets_to_dock=None: s)       # round 2 docks nothing
    pipeline.execute_expansion(expansion_state, {"action": "expand_3d_similar", "seed_cids": [100]})
    assert expansion_state["expansion_history"][-1]["outcome"] == "1 new ligand entries found; 0 docked"


def test_expansion_outcome_counts_docked_results(expansion_state, monkeypatch):
    monkeypatch.setattr(pipeline, "search_3d_similar",
                        lambda cid, n: [{"cid": 5, "smiles": "CCO", "iupac_name": "ethanol"}])

    def dock(state, targets_to_dock=None):
        state["all_docking_results"].append(_res("ethanol"))
        return state

    monkeypatch.setattr(pipeline, "stage_docking", dock)
    pipeline.execute_expansion(expansion_state, {"action": "expand_3d_similar", "seed_cids": [100]})
    assert expansion_state["expansion_history"][-1]["outcome"] == "1 new ligand entries found; 1 docked"


def test_resume_after_a_reasoning_failure_runs_reasoning_again(stages, monkeypatch):
    monkeypatch.setattr(pipeline, "_write_docking_output", lambda s: None)
    _save(_state(status="docking_round_1_complete", round=1, expansion_history=[]))
    pipeline.resume_pipeline()
    assert stages == ["decide", "report", "paper"]


def test_resume_does_not_redo_reasoning_that_already_happened(stages, monkeypatch):
    monkeypatch.setattr(pipeline, "_write_docking_output", lambda s: None)
    _save(_state(status="docking_round_1_complete", round=1,
                 expansion_history=[{"round": 1, "action": "proceed", "rationale": "r"}]))
    pipeline.resume_pipeline()
    assert stages == ["report", "paper"]


def test_resume_runs_the_expansion_a_resumed_reasoning_chooses(stages, monkeypatch):
    monkeypatch.setattr(pipeline, "_write_docking_output", lambda s: None)
    monkeypatch.setattr(pipeline, "analyze_and_decide",
                        lambda state: stages.append("decide") or {"action": "expand_class"})
    monkeypatch.setattr(pipeline, "execute_expansion",
                        lambda state, decision: stages.append("expand") or state)
    _save(_state(status="docking_round_1_complete", round=1, expansion_history=[]))
    pipeline.resume_pipeline()
    assert stages == ["decide", "expand", "report", "paper"]


def _err_entry(stage, fatal=False):
    return {"stage": stage, "source": "pubchem", "kind": "server", "message": "m",
            "fatal": fatal, "target": "t"}


def test_resume_forgets_non_fatal_errors_of_stages_it_reruns_but_keeps_the_rest(stages):
    errors = [_err_entry("literature"), _err_entry("structure"), _err_entry("structure", fatal=True)]
    _save(_state(status="literature_complete", errors=errors))
    state = pipeline.resume_pipeline()
    saved = [(e["stage"], e["fatal"]) for e in state["errors"]]
    assert ("literature", False) in saved           # literature did not re-run
    assert ("structure", False) not in saved        # structure re-ran: its old gaps are stale
    assert ("structure", True) in saved             # fatal entries stay as history


def test_resume_from_structures_complete_keeps_structure_errors(stages):
    _save(_state(status="structures_complete", errors=[_err_entry("structure")]))
    state = pipeline.resume_pipeline()
    assert [e["stage"] for e in state["errors"]] == ["structure"]


def test_a_failed_optional_pubmed_search_degrades_instead_of_aborting(monkeypatch):
    searches = []

    def search(query, *a, **k):
        searches.append(query)
        if len(searches) == 1:
            return [{"pmid": "1", "title": "t", "summary": "s", "authors": [], "published": "2020", "link": "l"}]
        raise _err("pubmed", "rate_limited")      # the drug + repurposing searches

    monkeypatch.setattr(pipeline, "search_pubmed", search)
    monkeypatch.setattr(pipeline, "query_prompt", lambda pid, **v: f"text:{pid}")
    monkeypatch.setattr(pipeline, "extract_proteins_from_text", lambda t: ["KRAS"])
    monkeypatch.setattr(pipeline, "_discover_repurposing_candidates", lambda d, p: "repurposing")
    monkeypatch.setattr(pipeline, "_extract_drug_protein_map", lambda *a: {"drugs": []})
    state = _state()
    pipeline.stage_literature(state)

    assert state["status"] == "literature_complete"
    assert [(e["stage"], e["source"], e["target"], e["fatal"]) for e in state["errors"]] == [
        ("literature", "pubmed", "drug papers", False),
        ("literature", "pubmed", "repurposing papers", False)]


def test_the_disease_pubmed_search_is_still_required(monkeypatch):
    def down(*a, **k):
        raise _err("pubmed", "server")

    monkeypatch.setattr(pipeline, "search_pubmed", down)
    with pytest.raises(ExternalCallError):
        pipeline.stage_literature(_state())
