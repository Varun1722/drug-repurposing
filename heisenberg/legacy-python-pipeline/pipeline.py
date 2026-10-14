#!/usr/bin/env python3
"""
Autonomous Drug Discovery Pipeline

Single orchestrator that runs the full pipeline: literature review →
structure retrieval → docking → **reasoning loop** → report → paper.

The reasoning loop is the agentic part: after each docking round,
OpenAI analyzes results, looks for patterns (e.g. drug-class
clustering), and decides whether to expand the search (3D similarity,
more drugs of a promising class) or proceed to synthesis.

Usage:
    python pipeline.py "pancreatic ductal adenocarcinoma"
    python pipeline.py "pleural mesothelioma" --max-rounds 3
    python pipeline.py --resume   # resume from last saved state
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Import existing pipeline functions (NO rewrites)
# ---------------------------------------------------------------------------

from llm import LLMOutputError, query_prompt, query_prompt_structured
from prompt_loader import used_prompts
from review import (
    search_pubmed,
    extract_proteins_from_text,
    extract_proteins_fallback,
    format_papers_for_prompt,
    build_disease_query,
    build_drug_query,
    build_repurposing_query,
    _discover_repurposing_candidates,
    _extract_drug_protein_map,
    _build_references,
)
from llm_schemas import DrugNameList, ReasoningDecision
from resilience import (
    EscalationRequired,
    ExternalCallError,
    FailureGuard,
    compact_error,
    record_error,
    record_exception,
    summarize_errors,
)
from agent2 import (
    search_pdb,
    pick_best_structure,
    download_pdb,
    lookup_smiles,
    search_compounds_for_target,
    search_3d_similar,
    canonicalize_smiles,
)
from agent3 import run_docking
from results import (
    summarise_target,
    classify_compound,
    build_overview_block,
    build_classified_overview,
    format_target_summary_for_prompt,
    generate_methodology,
    generate_results,
    generate_conclusion,
)
from final import (
    generate_abstract,
    merge_introduction_and_background,
    merge_drug_landscape,
    merge_results as merge_results_section,
    merge_conclusions,
    _get_review_intro_sections,
    _get_review_drug_sections,
    _get_review_conclusion,
    _get_results_methodology,
    _get_results_body,
    _get_results_conclusion,
    _extract_references,
    _deduplicate_references,
    _strip_leading_header,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATE_FILE = "pipeline_state.json"
STRUCTURES_DIR = "structures"
# Default for --max-rounds: how many times the reasoning loop may expand the
# search after the first docking round.  The run's actual limit lives in
# state["max_rounds"] (set from --max-rounds, saved so --resume keeps it);
# lib/rounds.ts holds the same default for the web app.
MAX_EXPANSION_ROUNDS = 2
PUBCHEM_DELAY = 0.25


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

def new_state(disease: str, max_rounds: int = MAX_EXPANSION_ROUNDS) -> dict:
    """Create a fresh pipeline state."""
    return {
        "disease": disease,
        "status": "initialized",
        "round": 0,
        "max_rounds": max_rounds,   # the one limit on expansion rounds for this run
        "protein_targets": [],
        "drugs": [],
        "targets": [],          # agent2-style target entries (protein, pdb_id, pdb_file, ligands)
        "all_docking_results": [],  # accumulated across rounds
        "hypotheses": [],
        "expansion_history": [],
        "review_md": "",
        "results_md": "",
        "final_paper_md": "",
        # Failed external calls and other degradations (see resilience.record_error).
        # Non-fatal ones are listed in the report; a fatal one stopped the run.
        "errors": [],
        "started_at": datetime.now().isoformat(),
    }


def save_state(state: dict):
    """Persist pipeline state to disk."""
    # Record which prompt versions (content hashes) produced this run's output.
    state.setdefault("prompt_versions", {}).update(used_prompts())
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, STATE_FILE)


def load_state() -> dict:
    """Load pipeline state from disk."""
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
    state.setdefault("errors", [])   # state files from before error recording
    state.setdefault("max_rounds", MAX_EXPANSION_ROUNDS)   # ... and before max_rounds
    return state


def _forget_stale_errors(state: dict, stages: set) -> None:
    """On resume, drop the non-fatal errors of stages that are about to run
    again: they described a failed attempt and would otherwise still be listed
    as missing data after a successful re-run.  Fatal entries are kept as history."""
    state["errors"] = [e for e in state.get("errors", [])
                       if e.get("fatal") or e.get("stage") not in stages]


def _run_stage(state: dict, name: str, fn, *args, **kwargs):
    """Run one pipeline stage.  If it fails, record the failure in the state
    and save it (status stays at the last completed stage, so ``--resume``
    re-runs this one) before re-raising."""
    try:
        return fn(state, *args, **kwargs)
    except Exception as exc:
        record_exception(state, stage=name, exc=exc, fatal=True)
        save_state(state)
        raise


# ---------------------------------------------------------------------------
# Stage 1: Literature review + target identification
# ---------------------------------------------------------------------------

def stage_literature(state: dict) -> dict:
    """Run literature search, identify protein targets and drug candidates."""
    disease = state["disease"]

    print(f"\n{'='*60}")
    print(f"  Stage 1: Literature Review — {disease}")
    print(f"{'='*60}\n")

    # Search PubMed
    print(">> Searching PubMed for disease papers …")
    disease_papers = search_pubmed(build_disease_query(disease))
    disease_papers_text = format_papers_for_prompt(disease_papers)

    # Generate literature review via OpenAI
    print(">> Generating literature review …")
    first_review = query_prompt(
        "literature/review_first_pass",
        disease=disease, papers=disease_papers_text,
    )

    # Extract protein targets
    print(">> Extracting protein targets …")
    proteins = extract_proteins_from_text(first_review)
    if not proteins:
        print("    Extraction from review text yielded nothing, retrying "
              "with a direct query …")
        proteins = extract_proteins_fallback(disease)
    if not proteins:
        raise RuntimeError(
            f"Could not identify any protein targets for {disease!r}."
        )
    print(f"    Targets: {proteins}")

    # Search for drug papers
    time.sleep(0.5)
    print(">> Searching PubMed for drug papers …")

    def optional_search(query: str, what: str) -> list:
        # These two searches only enrich the drug section ("(No papers found.)" is
        # handled below), so a PubMed failure after retries degrades them to empty,
        # recorded in state, instead of discarding the review work done so far.
        try:
            return search_pubmed(query)
        except ExternalCallError as e:
            print(f"  WARNING: PubMed search for {what} failed: {compact_error(e)}",
                  file=sys.stderr)
            record_exception(state, stage="literature", exc=e, target=what)
            return []

    drug_papers = optional_search(build_drug_query(disease, proteins), "drug papers")
    time.sleep(0.5)
    repurpose_papers = optional_search(build_repurposing_query(proteins),
                                       "repurposing papers")

    # Dedup
    seen_ids = {p["pmid"] for p in drug_papers}
    for rp in repurpose_papers:
        if rp["pmid"] not in seen_ids:
            drug_papers.append(rp)
            seen_ids.add(rp["pmid"])

    drug_papers_text = format_papers_for_prompt(drug_papers) if drug_papers else "(No papers found.)"

    # Generate drug section
    print(">> Generating drug analysis …")
    drug_review = query_prompt(
        "literature/drug_analysis",
        disease=disease,
        proteins=", ".join(proteins),
        papers=drug_papers_text,
    )

    # Repurposing candidates
    print(">> Searching for repurposing candidates …")
    repurposing_review = _discover_repurposing_candidates(disease, proteins)

    # Assemble review markdown
    all_papers = disease_papers + drug_papers
    references_block = _build_references(all_papers)
    today = datetime.now().strftime("%B %d, %Y")
    review_md = (
        f"# Literature Review: {disease.title()}\n\n"
        f"*Auto-generated on {today}*\n\n---\n\n"
        f"{first_review}\n\n---\n\n"
        f"{drug_review}\n\n---\n\n"
        f"{repurposing_review}\n\n---\n\n"
        f"## Consolidated References\n\n{references_block}\n"
    )

    with open("review.md", "w", encoding="utf-8") as f:
        f.write(review_md)

    # Extract drug-protein map
    print(">> Extracting drug-protein map …")
    combined = drug_review + "\n\n" + repurposing_review
    drug_map = _extract_drug_protein_map(combined, disease, proteins)

    with open("review.json", "w", encoding="utf-8") as f:
        json.dump(drug_map, f, indent=2, ensure_ascii=False)
    if "llm_error" in drug_map:
        record_error(state, stage="literature", source="openai",
                     kind=drug_map["llm_error"]["kind"],
                     message=f"Drug-protein map unusable: {drug_map['llm_error']['message']}")

    # Update state
    state["protein_targets"] = proteins
    state["drugs"] = drug_map.get("drugs", [])
    state["review_md"] = review_md
    state["status"] = "literature_complete"
    save_state(state)

    print(f"\n  >> Literature complete: {len(proteins)} targets, {len(state['drugs'])} drugs")
    return state


# ---------------------------------------------------------------------------
# Stage 2: Structure retrieval
# ---------------------------------------------------------------------------

def _attempt(state: dict, guard: FailureGuard, stage: str, what: str,
             target: str, fn, *args, default=None):
    """Run one external step.  Returns ``fn(*args)``.

    "Nothing found" is a normal return value and resets ``guard``.  On
    ``ExternalCallError`` the failure is recorded in ``state["errors"]`` and
    counted against ``guard``, which raises ``EscalationRequired`` after
    repeated failures to the same source; otherwise ``default`` is returned
    and the stage carries on without that one item.
    """
    try:
        value = fn(*args)
    except ExternalCallError as e:
        print(f"  WARNING: {what} failed for {target}: {compact_error(e)}",
              file=sys.stderr)
        record_exception(state, stage=stage, exc=e, target=target)
        guard.failed(e)
        return default
    guard.ok()
    return value


def stage_structure(state: dict, extra_ligands: dict | None = None) -> dict:
    """Retrieve PDB structures and drug SMILES.

    extra_ligands: optional dict mapping protein name → list of
      {"name": str, "smiles": str, "mechanism": str, "source": str}
      from expansion rounds.
    """
    disease = state["disease"]
    proteins = state["protein_targets"]
    drugs = state["drugs"]

    print(f"\n{'='*60}")
    print(f"  Stage 2: Structure Retrieval")
    print(f"{'='*60}\n")

    targets = []
    rcsb_guard = FailureGuard("rcsb")
    pubchem_guard = FailureGuard("pubchem")

    for protein in proteins:
        print(f"\n{'─'*40}")
        print(f"  Processing: {protein}")
        print(f"{'─'*40}\n")

        # Search PDB
        candidates = _attempt(state, rcsb_guard, "structure", "PDB search",
                              protein, search_pdb, protein, default=[])
        if not candidates:
            gene = protein.split()[0]
            if gene != protein:
                candidates = _attempt(state, rcsb_guard, "structure", "PDB search",
                                      protein, search_pdb, gene, default=[])

        pdb_id = None
        pdb_file = None
        if candidates:
            best = _attempt(state, rcsb_guard, "structure", "PDB metadata",
                            protein, pick_best_structure, candidates)
            if best:
                pdb_id = best["pdb_id"]
                pdb_file = _attempt(state, rcsb_guard, "structure", "PDB download",
                                    protein, download_pdb, pdb_id, STRUCTURES_DIR)

        # Look up SMILES for known drugs
        ligands = []
        for drug_info in drugs:
            if protein not in drug_info.get("proteins", []):
                continue
            drug_name = drug_info["drug"]
            time.sleep(PUBCHEM_DELAY)
            result = _attempt(state, pubchem_guard, "structure", "SMILES lookup",
                              f"{protein}/{drug_name}", lookup_smiles, drug_name)
            ligands.append({
                "name": drug_name,
                "smiles": result["smiles"] if result else "",
                "mechanism": drug_info.get("mechanism", ""),
                "fda_status": drug_info.get("fda_status", "Unknown"),
                "source": f"pubchem_cid_{result['cid']}" if result else "no_smiles",
            })

        # Discover bioactive compounds
        extra = _attempt(state, pubchem_guard, "structure", "compound search",
                         protein, search_compounds_for_target, protein, 50,
                         default=[])
        existing_smiles = {l["smiles"] for l in ligands if l["smiles"]}
        for c in extra:
            if c["smiles"] and c["smiles"] not in existing_smiles:
                ligands.append({
                    "name": c.get("iupac_name", f"CID_{c['cid']}"),
                    "smiles": c["smiles"],
                    "mechanism": "Bioactive — PubChem target search",
                    "fda_status": "Unknown — requires verification",
                    "source": f"pubchem_cid_{c['cid']}",
                })
                existing_smiles.add(c["smiles"])

        # Add expansion ligands if provided
        if extra_ligands and protein in extra_ligands:
            for lig in extra_ligands[protein]:
                if lig["smiles"] and lig["smiles"] not in existing_smiles:
                    ligands.append(lig)
                    existing_smiles.add(lig["smiles"])

        # Drop empty SMILES
        ligands = [l for l in ligands if l["smiles"]]

        targets.append({
            "protein": protein,
            "pdb_id": pdb_id,
            "pdb_file": pdb_file,
            "ligands": ligands,
        })

        print(f"  >> {protein}: PDB={pdb_id}, {len(ligands)} ligands")

    state["targets"] = targets
    state["status"] = "structures_complete"
    save_state(state)

    # Also write agent2_output.json for compatibility
    with open("agent2_output.json", "w", encoding="utf-8") as f:
        json.dump({"disease": disease, "targets": targets}, f, indent=2)

    return state


# ---------------------------------------------------------------------------
# Stage 3: Docking
# ---------------------------------------------------------------------------

def stage_docking(state: dict, targets_to_dock: list[dict] | None = None) -> dict:
    """Run GNINA docking on all targets (or a subset for expansion rounds)."""
    targets = targets_to_dock or state["targets"]
    round_num = state["round"]

    print(f"\n{'='*60}")
    print(f"  Stage 3: GNINA Docking — Round {round_num}")
    print(f"{'='*60}\n")

    round_results = []

    for target in targets:
        protein = target["protein"]
        pdb_file = target.get("pdb_file")
        ligands = target.get("ligands", [])

        if not pdb_file or not os.path.exists(pdb_file):
            print(f"  WARNING: No PDB file for {protein}, skipping")
            record_error(state, stage="docking", source="pipeline",
                         kind="skipped_no_structure", target=protein,
                         message=f"{protein} was not docked: no structure file")
            continue

        dock_ligands = [
            {"name": l["name"], "smiles": l["smiles"]}
            for l in ligands if l.get("smiles")
        ]

        if not dock_ligands:
            print(f"  WARNING: No ligands for {protein}, skipping")
            record_error(state, stage="docking", source="pipeline",
                         kind="skipped_no_ligands", target=protein,
                         message=f"{protein} was not docked: no ligands with a SMILES")
            continue

        print(f"\n  Docking {len(dock_ligands)} ligands against {protein} …")
        dock_errors: list = []
        try:
            results, elapsed = run_docking(
                protein_pdb_path=pdb_file,
                ligands=dock_ligands,
                errors=dock_errors,
            )
        except ExternalCallError as e:
            # Every chunk failed: keep going with the other targets.
            print(f"  WARNING: docking {protein} failed: {compact_error(e)}",
                  file=sys.stderr)
            record_exception(state, stage="docking", exc=e, target=protein)
            continue
        for d in dock_errors:
            record_error(state, stage="docking", source="gnina", target=protein, **d)

        # Merge metadata back
        ligand_meta = {l["name"]: l for l in ligands}
        for r in results:
            meta = ligand_meta.get(r["name"], {})
            r["mechanism"] = meta.get("mechanism", "")
            r["fda_status"] = meta.get("fda_status", "")
            r["source"] = meta.get("source", "")
            r["protein_target"] = protein
            r["pdb_id"] = target.get("pdb_id", "")
            r["round"] = round_num

        round_results.extend(results)

        print(f"  Top 5 for {protein}:")
        for i, r in enumerate(results[:5]):
            print(f"    {i+1}. {r['name'][:40]:40s} score={r['confidence_score']:.4f}")

    if not round_results:
        # Round 1 with nothing docked means no data to reason about or write up:
        # stop (state is saved by _run_stage) instead of producing an empty paper.
        # A later expansion round that adds nothing is merely recorded.
        message = (f"Round {round_num} docked no ligands; see the docking errors "
                   f"recorded in {STATE_FILE}.")
        if round_num <= 1:
            raise ExternalCallError("gnina", "no_results", message)
        record_error(state, stage="docking", source="gnina", kind="no_results",
                     message=message)

    state["all_docking_results"].extend(round_results)
    state["status"] = f"docking_round_{round_num}_complete"
    save_state(state)

    # Write agent3_output.json for compatibility
    _write_docking_output(state)

    return state


def _write_docking_output(state: dict):
    """Write agent3_output.json from accumulated results."""
    results_by_protein = {}
    for r in state["all_docking_results"]:
        prot = r.get("protein_target", "unknown")
        results_by_protein.setdefault(prot, []).append(r)

    target_entries = []
    for target in state["targets"]:
        protein = target["protein"]
        results = results_by_protein.get(protein, [])
        results.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)
        target_entries.append({
            "protein": protein,
            "pdb_id": target.get("pdb_id"),
            "pdb_file": target.get("pdb_file"),
            "status": "completed",
            "num_ligands_total": len(target.get("ligands", [])),
            "num_ligands_docked": len(results),
            "docking_time_seconds": 0,
            "results": results,
        })

    output = {
        "disease": state["disease"],
        "status": "completed",
        "completed_targets": len(target_entries),
        "total_targets": len(target_entries),
        "total_docking_time_seconds": 0,
        "targets": target_entries,
    }
    with open("agent3_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Stage 4: Reasoning loop (THE AGENTIC PART)
# ---------------------------------------------------------------------------

def analyze_and_decide(state: dict) -> dict:
    """Ask OpenAI to analyze docking results and decide next action.

    This is the core agentic reasoning: the LLM examines the results,
    looks for patterns (drug class clustering, structural similarity),
    forms a hypothesis, and decides whether to expand the search or
    proceed to paper generation.

    Returns: {"action": "expand_3d_similar"|"expand_class"|"proceed",
              "rationale": str, "hypothesis": str, ...}
    """
    disease = state["disease"]
    round_num = state["round"]
    results = state["all_docking_results"]
    hypotheses = state["hypotheses"]

    # Build a summary of top results for the prompt
    top_results = sorted(results, key=lambda x: x.get("confidence_score", 0), reverse=True)[:30]
    results_summary = []
    for r in top_results:
        results_summary.append({
            "name": r["name"],
            "score": round(r.get("confidence_score", 0), 4),
            "mechanism": r.get("mechanism", "")[:100],
            "fda_status": r.get("fda_status", "")[:60],
            "protein": r.get("protein_target", ""),
            "source": r.get("source", ""),
            "round": r.get("round", 1),
        })

    print(f"\n{'='*60}")
    print(f"  Stage 4: Reasoning — Round {round_num}")
    print(f"{'='*60}\n")

    # Docking round 1 is the initial one, so ``max_rounds`` expansions end at
    # docking round max_rounds + 1.  The loop in run_pipeline never reaches
    # this; it guards direct callers and old state files.  There is nothing to
    # ask the model at the limit, so do not.
    max_round = state.get("max_rounds", MAX_EXPANSION_ROUNDS) + 1
    if round_num >= max_round:
        print("  Max rounds reached, proceeding to synthesis.")
        return {
            "action": "proceed",
            "rationale": "Max rounds reached, proceeding to synthesis.",
            "hypothesis": "",
            "seed_cids": None,
            "drug_class": None,
            "drug_names": None,
        }

    print(">> Asking OpenAI to analyze results …")

    try:
        decision = query_prompt_structured(
            "analysis/reasoning_decision", ReasoningDecision,
            disease=disease,
            round=round_num,
            max_round=max_round,
            hypotheses=json.dumps(hypotheses),
            expansion_history=json.dumps(state["expansion_history"]),
            top_results=json.dumps(results_summary, indent=2),
        ).model_dump()
    except LLMOutputError as e:
        # Don't expand on a guess; say why in the rationale so it lands in
        # expansion_history and the report instead of vanishing.
        print(f"  WARNING: reasoning reply unusable ({e.kind}): {e}", file=sys.stderr)
        record_error(state, stage="reasoning", source="openai", kind=e.kind,
                     message=f"Reasoning reply unusable: {e}")
        decision = {
            "action": "proceed",
            "rationale": f"(Reasoning step produced no usable result: {e.kind}. "
                         f"Proceeding to synthesis.)",
            "hypothesis": "",
            "seed_cids": None,
            "drug_class": None,
            "drug_names": None,
        }

    print(f"\n  Decision: {decision['action']}")
    print(f"  Rationale: {decision['rationale']}")
    print(f"  Hypothesis: {decision['hypothesis']}")

    # Record
    state["hypotheses"].append(decision.get("hypothesis", ""))
    state["expansion_history"].append({
        "round": round_num,
        "action": decision["action"],
        "rationale": decision["rationale"],
    })
    save_state(state)

    return decision


def execute_expansion(state: dict, decision: dict) -> dict:
    """Execute an expansion decision: fetch new compounds and dock them."""
    action = decision["action"]
    disease = state["disease"]

    print(f"\n{'='*60}")
    print(f"  Expansion: {action}")
    print(f"{'='*60}\n")

    new_ligands_by_protein = {}  # protein → [ligand dicts]
    failures: list[ExternalCallError] = []   # PubChem lookups that failed (not "no hit")

    def lookup(what: str, target: str, fn, *args):
        """Best-effort PubChem call for an expansion: a failure is recorded
        and the item skipped.  No escalation here: round-1 results are already
        saved, so a flaky PubChem should shrink the expansion, not end the run."""
        try:
            return fn(*args)
        except ExternalCallError as e:
            print(f"  WARNING: {what} failed for {target}: {compact_error(e)}",
                  file=sys.stderr)
            record_exception(state, stage="expansion", exc=e, target=target)
            failures.append(e)
            return None

    if action == "expand_3d_similar":
        seed_cids = decision.get("seed_cids") or []
        if not seed_cids:
            # Fall back to top-scoring compounds that have CIDs
            for r in sorted(
                state["all_docking_results"],
                key=lambda x: x.get("confidence_score", 0),
                reverse=True,
            ):
                src = r.get("source", "")
                if src.startswith("pubchem_cid_"):
                    try:
                        cid = int(src.split("_")[-1])
                        seed_cids.append(cid)
                    except ValueError:
                        pass
                if len(seed_cids) >= 5:
                    break

        print(f"  Expanding 3D similarity for CIDs: {seed_cids}")
        for cid in seed_cids:
            time.sleep(PUBCHEM_DELAY)
            similar = lookup("3D similarity search", f"CID {cid}",
                             search_3d_similar, cid, 10) or []
            for c in similar:
                if c["smiles"]:
                    lig = {
                        "name": c.get("iupac_name", f"CID_{c['cid']}"),
                        "smiles": c["smiles"],
                        "mechanism": f"3D-similar to CID {cid} (ST≥0.80, CT≥0.50)",
                        "fda_status": "Unknown — requires verification",
                        "source": f"pubchem_3dsim_cid_{c['cid']}_from_{cid}",
                    }
                    # Add to all protein targets
                    for protein in state["protein_targets"]:
                        new_ligands_by_protein.setdefault(protein, []).append(lig)

    elif action == "expand_class":
        drug_names = decision.get("drug_names") or []
        drug_class = decision.get("drug_class") or ""

        if not drug_names and drug_class:
            # Ask OpenAI for specific drug names in this class
            print(f"  Asking OpenAI for {drug_class} drugs …")
            try:
                drug_names = query_prompt_structured(
                    "analysis/class_expansion", DrugNameList,
                    drug_class=drug_class,
                    proteins=", ".join(state["protein_targets"]),
                    exclude=", ".join(r["name"] for r in state["all_docking_results"][:20]) or "(none)",
                ).drug_names
            except LLMOutputError as e:
                # No names -> no new ligands -> "No new compounds found" below.
                print(f"  WARNING: drug-name reply unusable ({e.kind}): {e}",
                      file=sys.stderr)
                record_error(state, stage="expansion", source="openai",
                             kind=e.kind, message=f"Drug-name reply unusable: {e}")

        print(f"  Looking up SMILES for {len(drug_names)} drugs: {drug_names}")
        for name in drug_names[:10]:
            time.sleep(PUBCHEM_DELAY)
            result = lookup("SMILES lookup", str(name), lookup_smiles, str(name))
            if result and result["smiles"]:
                lig = {
                    "name": str(name),
                    "smiles": result["smiles"],
                    "mechanism": f"Class expansion: {drug_class}",
                    "fda_status": "FDA-approved (class expansion)",
                    "source": f"pubchem_cid_{result['cid']}",
                }
                for protein in state["protein_targets"]:
                    new_ligands_by_protein.setdefault(protein, []).append(lig)

    # Count new compounds
    total_new = sum(len(v) for v in new_ligands_by_protein.values())
    print(f"\n  >> Added {total_new} new ligands across {len(new_ligands_by_protein)} targets")

    # Tell the next reasoning step what actually happened, so it does not
    # re-propose an expansion that failed for lack of data, not lack of merit.
    outcome = f"{total_new} new ligand entries found"
    if failures:
        kinds = ", ".join(sorted({f.kind for f in failures}))
        outcome += f"; {len(failures)} PubChem call(s) failed ({kinds})"

    def set_outcome(text: str) -> None:
        if state["expansion_history"]:
            state["expansion_history"][-1]["outcome"] = text
            save_state(state)

    set_outcome(outcome)

    if not new_ligands_by_protein:
        print("  >> No new compounds found, proceeding to synthesis")
        return state

    # Build docking targets for just the new compounds
    expansion_targets = []
    for target in state["targets"]:
        protein = target["protein"]
        new_ligs = new_ligands_by_protein.get(protein, [])
        if new_ligs and target.get("pdb_file"):
            expansion_targets.append({
                "protein": protein,
                "pdb_id": target.get("pdb_id"),
                "pdb_file": target["pdb_file"],
                "ligands": new_ligs,
            })

    # Dock the new compounds
    state["round"] += 1
    docked_before = len(state["all_docking_results"])
    state = stage_docking(state, targets_to_dock=expansion_targets)
    # Docking can fail after the search succeeded: report what was actually docked.
    set_outcome(f"{outcome}; {len(state['all_docking_results']) - docked_before} docked")
    return state


# ---------------------------------------------------------------------------
# Stage 5 & 6: Report generation + paper assembly
# ---------------------------------------------------------------------------

def stage_report(state: dict) -> dict:
    """Generate results.md from docking data."""
    disease = state["disease"]

    print(f"\n{'='*60}")
    print(f"  Stage 5: Report Generation")
    print(f"{'='*60}\n")

    # Load docking data from the file we wrote
    from results import load_docking_data
    data = load_docking_data("agent3_output.json")

    target_summaries = [summarise_target(t) for t in data.get("targets", [])]
    overview = build_overview_block(data, target_summaries)
    classified = build_classified_overview(target_summaries, disease)

    # Include reasoning history in the report
    reasoning_note = ""
    if state["hypotheses"]:
        reasoning_note = (
            "\n\n### Autonomous Reasoning History\n\n"
            "The following hypotheses were generated and tested autonomously "
            "by the pipeline's reasoning loop:\n\n"
        )
        for i, (hyp, exp) in enumerate(
            zip(state["hypotheses"], state["expansion_history"]), 1
        ):
            reasoning_note += (
                f"**Round {i}:** {hyp}\n"
                f"- Action: {exp['action']}\n"
                f"- Rationale: {exp['rationale']}\n\n"
            )

    # Disclose inputs that are missing because an external call failed, so a
    # reader does not mistake a gap for a negative result.
    error_lines = summarize_errors(state.get("errors", []))
    errors_note = ""
    if error_lines:
        errors_note = (
            "\n\n### Data Quality Notes\n\n"
            "Some external calls failed even after retries, so the affected "
            "items are missing from this analysis:\n\n"
            + "\n".join(f"- {line}" for line in error_lines)
            + "\n"
        )

    print(">> Generating Methodology …")
    methodology = generate_methodology(disease, overview)

    print(">> Generating Results …")
    results_text = generate_results(disease, overview, classified)

    print(">> Generating Conclusion …")
    conclusion = generate_conclusion(disease, overview, classified)

    # Strip duplicate headers
    methodology = _strip_leading_header(methodology)
    results_text = _strip_leading_header(results_text)
    conclusion = _strip_leading_header(conclusion)

    today = datetime.now().strftime("%B %d, %Y")
    results_md = (
        f"# Molecular Docking Analysis: {disease.title()}\n\n"
        f"*Auto-generated on {today}*\n\n---\n\n"
        f"## Methodology\n\n{methodology}\n\n---\n\n"
        f"## Results\n\n{results_text}\n\n"
        f"{reasoning_note}{errors_note}\n\n---\n\n"
        f"## Conclusion\n\n{conclusion}\n"
    )

    with open("results.md", "w", encoding="utf-8") as f:
        f.write(results_md)

    state["results_md"] = results_md
    state["status"] = "report_complete"
    save_state(state)

    return state


def stage_paper(state: dict) -> dict:
    """Assemble the final research paper from review.md + results.md."""
    disease = state["disease"]

    print(f"\n{'='*60}")
    print(f"  Stage 6: Final Paper Assembly")
    print(f"{'='*60}\n")

    review_md = state.get("review_md", "")
    results_md = state.get("results_md", "")

    if not review_md:
        with open("review.md", "r") as f:
            review_md = f.read()
    if not results_md:
        with open("results.md", "r") as f:
            results_md = f.read()

    # Extract sections
    review_intro = _get_review_intro_sections(review_md)
    review_drugs = _get_review_drug_sections(review_md)
    review_conclusion = _get_review_conclusion(review_md)
    results_methodology = _get_results_methodology(results_md)
    results_body = _get_results_body(results_md)
    results_conclusion = _get_results_conclusion(results_md)
    review_refs = _extract_references(review_md)

    print(">> Generating abstract …")
    abstract = _strip_leading_header(generate_abstract(review_md, results_md))

    print(">> Merging Introduction …")
    intro = _strip_leading_header(
        merge_introduction_and_background(review_intro, results_md)
    )

    print(">> Merging Therapeutic Landscape …")
    landscape = _strip_leading_header(merge_drug_landscape(review_drugs))

    methodology = _strip_leading_header(results_methodology)

    print(">> Merging Results …")
    docking_results = _strip_leading_header(merge_results_section(results_body))

    print(">> Merging Conclusions …")
    conclusion = _strip_leading_header(
        merge_conclusions(review_conclusion, results_conclusion)
    )

    unified_refs = _deduplicate_references(review_refs, "")

    today = datetime.now().strftime("%B %d, %Y")
    final_md = (
        f"# Drug Repurposing for {disease.title()}: A Computational Docking "
        f"and Literature Review Study\n\n"
        f"*{today}*\n\n---\n\n"
        f"## Abstract\n\n{abstract}\n\n---\n\n"
        f"## 1. Introduction and Background\n\n{intro}\n\n---\n\n"
        f"## 2. Therapeutic Landscape\n\n{landscape}\n\n---\n\n"
        f"## 3. Methodology\n\n{methodology}\n\n---\n\n"
        f"## 4. Computational Docking Results\n\n{docking_results}\n\n---\n\n"
        f"## 5. Conclusion\n\n{conclusion}\n\n---\n\n"
        f"## References\n\n{unified_refs}\n"
    )

    with open("final_paper.md", "w", encoding="utf-8") as f:
        f.write(final_md)

    state["final_paper_md"] = final_md
    state["status"] = "complete"
    save_state(state)

    print(f"\n  >> Final paper: final_paper.md ({len(final_md):,} chars)")
    return state


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(disease: str, max_rounds: int = MAX_EXPANSION_ROUNDS):
    """Run the full autonomous pipeline.

    ``max_rounds`` is the most times the reasoning loop may expand the search
    after the first docking round.  Each pass asks the model once; it ends
    early when the model says "proceed".  An expansion that finds nothing
    still uses up a pass.
    """
    if max_rounds < 0:
        raise ValueError(f"max_rounds must be >= 0, got {max_rounds}")
    state = new_state(disease, max_rounds)

    print(f"\n{'#'*60}")
    print(f"  Autonomous Drug Discovery Pipeline")
    print(f"  Target: {disease}")
    print(f"  Max expansion rounds: {max_rounds}")
    print(f"{'#'*60}\n")

    # Stage 1: Literature
    state = _run_stage(state, "literature", stage_literature)

    # Stage 2: Structure retrieval
    state = _run_stage(state, "structure", stage_structure)

    # Stage 3: First docking round
    state["round"] = 1
    state = _run_stage(state, "docking", stage_docking)

    # Stage 4: Reasoning loop
    for _ in range(max_rounds):
        decision = _run_stage(state, "reasoning", analyze_and_decide)

        if decision["action"] == "proceed":
            print("\n  >> Reasoning decided: proceed to synthesis")
            break

        # Execute expansion and re-dock
        state = _run_stage(state, "expansion", execute_expansion, decision)

        # Analyze again after expansion
        # (the loop continues and will call analyze_and_decide again)

    # Stage 5: Report
    state = _run_stage(state, "report", stage_report)

    # Stage 6: Paper
    state = _run_stage(state, "paper", stage_paper)

    # Final summary
    total_compounds = len(state["all_docking_results"])
    total_rounds = state["round"]
    print(f"\n{'#'*60}")
    print(f"  Pipeline Complete")
    print(f"  Disease:         {disease}")
    print(f"  Docking rounds:  {total_rounds}")
    print(f"  Total compounds: {total_compounds}")
    print(f"  Hypotheses:      {len(state['hypotheses'])}")
    print(f"  Output:          final_paper.md")
    if state["errors"]:
        print(f"  Recorded errors: {len(state['errors'])} (see \"errors\" in {STATE_FILE})")
    print(f"{'#'*60}\n")

    return state


def resume_pipeline():
    """Resume from the last saved state."""
    if not os.path.exists(STATE_FILE):
        print(f"ERROR: No state file found ({STATE_FILE})", file=sys.stderr)
        sys.exit(1)

    state = load_state()
    status = state["status"]
    disease = state["disease"]

    print(f"\n  Resuming pipeline for '{disease}' (status: {status})")

    def structure_to_expansion(state: dict) -> dict:
        """Stages 2-4 (one reasoning round), as the original resume did."""
        if status != "structures_complete":
            state = _run_stage(state, "structure", stage_structure)
        state["round"] = 1
        state = _run_stage(state, "docking", stage_docking)
        decision = _run_stage(state, "reasoning", analyze_and_decide)
        if decision["action"] != "proceed":
            state = _run_stage(state, "expansion", execute_expansion, decision)
        return state

    if status == "complete":
        print("  Pipeline already complete.")

    elif status == "report_complete":
        _forget_stale_errors(state, {"paper"})
        state = _run_stage(state, "paper", stage_paper)

    elif "docking" in status:
        # Reasoning appends one expansion_history entry per docking round.  If
        # the last round has none, the run stopped before (or during) reasoning,
        # and resuming straight to the report would silently skip it.
        owed = min(state["round"], state["max_rounds"])   # no reasoning past the limit
        if len(state["expansion_history"]) < owed:
            print("  Reasoning for the last docking round had not completed; running it.")
            _forget_stale_errors(state, {"reasoning", "expansion", "report", "paper"})
            decision = _run_stage(state, "reasoning", analyze_and_decide)
            if decision["action"] != "proceed":
                state = _run_stage(state, "expansion", execute_expansion, decision)
        else:
            _forget_stale_errors(state, {"report", "paper"})
        # Rewrite agent3_output.json with correct schema before report
        _write_docking_output(state)
        state = _run_stage(state, "report", stage_report)
        state = _run_stage(state, "paper", stage_paper)

    else:
        rerun = {"literature", "structure", "docking", "reasoning", "expansion",
                 "report", "paper"}
        if status != "initialized":
            rerun.discard("literature")
        if status == "structures_complete":
            rerun.discard("structure")
        _forget_stale_errors(state, rerun)
        if status == "initialized":
            # A failure in the very first stage leaves a state with no literature yet.
            state = _run_stage(state, "literature", stage_literature)
        elif status not in ("literature_complete", "structures_complete"):
            print(f"  Unknown status: {status}. Starting from structures.")
        state = structure_to_expansion(state)
        state = _run_stage(state, "report", stage_report)
        state = _run_stage(state, "paper", stage_paper)

    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Drug Discovery Pipeline"
    )
    parser.add_argument(
        "disease",
        nargs="?",
        type=str,
        help="Disease to analyze (e.g. 'pancreatic ductal adenocarcinoma')",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last saved pipeline state.",
    )
    parser.add_argument(
        "--max-rounds",
        type=_non_negative_int,
        default=MAX_EXPANSION_ROUNDS,
        help=f"Max expansion rounds after the first docking round, 0 to skip "
             f"expansion (default: {MAX_EXPANSION_ROUNDS}).",
    )
    args = parser.parse_args()

    if not args.resume and not args.disease:
        parser.print_help()
        sys.exit(1)

    try:
        if args.resume:
            resume_pipeline()
        else:
            run_pipeline(args.disease, max_rounds=args.max_rounds)
    except (ExternalCallError, EscalationRequired) as e:
        # _run_stage has already saved the state with this error recorded.
        print(f"\nPIPELINE STOPPED: {e}", file=sys.stderr)
        print(f"Progress is saved in {STATE_FILE}. Fix the cause, then run: "
              f"python pipeline.py --resume", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
