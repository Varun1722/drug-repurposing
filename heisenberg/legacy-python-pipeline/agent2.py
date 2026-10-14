#!/usr/bin/env python3
"""
Agent 2 — Structure Retrieval Agent

Reads Agent 1's output (review.json) and for each protein target:
  1. Searches the RCSB Protein Data Bank for the best 3D structure
  2. Downloads the .pdb file
  3. Queries PubChem for drug SMILES strings
  4. Discovers additional bioactive compounds via PubChem assay data

Also exposes search_3d_similar() for post-docking 3D similarity expansion
(shape-Tanimoto >= 0.80, color-Tanimoto >= 0.50) — call after Agent 3
identifies top hits to find structurally analogous candidates.

Outputs a structured JSON (agent2_output.json) for Agent 3 (DiffDock).

No AI / LLM calls — just two REST APIs (RCSB PDB + PubChem).

Failure handling: "no results" (a 404, or RCSB's 204) returns ``[]`` / ``None``.
A call that still fails after retries raises ``resilience.ExternalCallError``
instead of being mistaken for "no results"; the caller decides what to skip and
records it (see pipeline.stage_structure and main() below).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests
from rdkit import Chem

from resilience import (
    EscalationRequired,
    ExternalCallError,
    FailureGuard,
    compact_error,
    request_with_retry,
    response_json,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

STRUCTURES_DIR = "structures"
OUTPUT_FILE = "agent2_output.json"

PUBCHEM_DELAY = 0.25  # PubChem allows 5 req/s — stay conservative
PDB_DELAY = 0.5

# PubChem answers 404 for "no such record" and 400 (PUGREST.BadRequest) for an
# identifier it cannot parse.  For identifiers that come from an LLM both mean
# "no match", not "PubChem is down".
_PUBCHEM_NO_MATCH = (400, 404)

# PubChem property names we request vs. the keys it actually returns
# (the API sometimes renames them, e.g. CanonicalSMILES -> ConnectivitySMILES).
_SMILES_KEYS = ("CanonicalSMILES", "ConnectivitySMILES", "SMILES")


def _extract_smiles(props: dict) -> str:
    """Pull the SMILES string from a PubChem property dict, trying known keys."""
    for key in _SMILES_KEYS:
        val = props.get(key, "")
        if val:
            return val
    return ""


def canonicalize_smiles(smiles: str) -> str:
    """Convert a SMILES string to canonical isomeric SMILES via RDKit.

    GNINA requires canonical isomeric SMILES so that stereochemistry
    and aromatic perception are unambiguous.  Returns the original string
    unchanged if RDKit cannot parse it (better to attempt docking with a
    raw SMILES than to silently drop the compound).
    """
    if not smiles:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[RDKit] WARNING: could not parse SMILES, keeping raw: {smiles[:60]}")
        return smiles
    return Chem.MolToSmiles(mol, isomericSmiles=True)


# ---------------------------------------------------------------------------
# RCSB PDB helpers
# ---------------------------------------------------------------------------

def search_pdb(protein_name: str, max_results: int = 10) -> list[dict]:
    """Search RCSB PDB for structures matching a protein name.

    Filters: Human, X-ray diffraction, resolution < 3 Å.
    Returns list of {pdb_id, score} sorted by resolution (best first).
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": protein_name},
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entity_source_organism.ncbi_scientific_name",
                        "operator": "exact_match",
                        "value": "Homo sapiens",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": 3.0,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc",
                }
            ],
            "results_content_type": ["experimental"],
            "paginate": {"start": 0, "rows": max_results},
        },
    }

    print(f"[PDB] Searching for: {protein_name}")
    # RCSB answers 204 No Content when nothing matches: that is a result, not an error.
    resp = request_with_retry("post", RCSB_SEARCH_URL, source="rcsb",
                              json=query, timeout=30, allow_status=(204,))
    if resp.status_code == 204:
        print(f"[PDB] Found 0 structures for '{protein_name}'")
        return []
    data = response_json(resp, "rcsb")

    results = []
    for hit in data.get("result_set", []):
        pdb_id = hit.get("identifier", "")
        score = hit.get("score", 0)
        results.append({"pdb_id": pdb_id, "score": score})

    print(f"[PDB] Found {len(results)} structures for '{protein_name}'")
    return results


def get_pdb_metadata(pdb_id: str) -> dict:
    """Get metadata for a PDB entry (resolution, title, ligand count).

    An entry RCSB does not know (404) gets empty metadata; a failed request
    raises ``ExternalCallError`` rather than pretending the entry has no ligand.
    """
    url = f"{RCSB_DATA_URL}/{pdb_id}"
    resp = request_with_retry("get", url, source="rcsb", timeout=15,
                              allow_status=(404,))
    if resp.status_code == 404:
        print(f"[PDB] No metadata for {pdb_id} (404)")
        return {
            "pdb_id": pdb_id,
            "title": "",
            "resolution": None,
            "has_ligand": False,
            "nonpolymer_count": 0,
        }
    data = response_json(resp, "rcsb")

    entry_info = data.get("rcsb_entry_info") or {}
    resolution = None
    res_val = entry_info.get("resolution_combined", [None])
    if isinstance(res_val, list) and res_val:
        resolution = res_val[0]
    elif isinstance(res_val, (int, float)):
        resolution = res_val

    title = (data.get("struct") or {}).get("title", "")
    nonpolymer_count = entry_info.get("nonpolymer_entity_count") or 0

    return {
        "pdb_id": pdb_id,
        "title": title,
        "resolution": resolution,
        "has_ligand": nonpolymer_count > 0,
        "nonpolymer_count": nonpolymer_count,
    }


def pick_best_structure(candidates: list[dict]) -> dict | None:
    """From a list of PDB search results, pick the best one.

    Prefers structures that have a bound ligand, then highest resolution.
    A candidate whose metadata cannot be fetched is skipped (it cannot be
    judged); if none can be fetched the last ``ExternalCallError`` is raised.
    """
    if not candidates:
        return None

    enriched = []
    last_error: ExternalCallError | None = None
    for c in candidates[:10]:
        try:
            enriched.append(get_pdb_metadata(c["pdb_id"]))
        except ExternalCallError as e:
            print(f"[PDB] Skipping {c['pdb_id']}: {compact_error(e)}")
            last_error = e
        time.sleep(PDB_DELAY)
    if not enriched:
        raise last_error

    # Prefer structures with ligands, then sort by resolution
    with_ligand = [e for e in enriched if e["has_ligand"]]
    pool = with_ligand if with_ligand else enriched
    pool.sort(key=lambda x: x["resolution"] or 999)

    best = pool[0]
    tag = "with ligand" if best["has_ligand"] else "no ligand"
    print(
        f"[PDB] Best structure ({tag}): {best['pdb_id']} "
        f"(resolution: {best['resolution']}Å, "
        f"non-polymer entities: {best['nonpolymer_count']})"
    )
    return best


def download_pdb(pdb_id: str, output_dir: str) -> str:
    """Download a structure file from RCSB. Returns the local .pdb file path.

    Tries .pdb first; falls back to .cif (mmCIF) for newer entries where
    RCSB no longer provides the legacy PDB format.  When a .cif is
    downloaded it is converted to .pdb via ``gemmi`` so that downstream
    tools (GNINA / ProDy) always receive PDB format.
    """
    os.makedirs(output_dir, exist_ok=True)

    pdb_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    if os.path.exists(pdb_path):
        print(f"[PDB] {pdb_id}.pdb already exists, skipping download")
        return pdb_path

    # Try .pdb first, fall back to .cif
    for ext in ("pdb", "cif"):
        url = f"{RCSB_DOWNLOAD_URL}/{pdb_id}.{ext}"
        print(f"[PDB] Downloading {url} …")

        resp = request_with_retry("get", url, source="rcsb", timeout=60,
                                  allow_status=(404,))
        if resp.status_code == 404:
            if ext == "pdb":
                print(f"[PDB] .pdb not available for {pdb_id}, trying .cif …")
            continue

        if ext == "pdb":
            with open(pdb_path, "wb") as f:
                f.write(resp.content)
            size_kb = len(resp.content) / 1024
            print(f"[PDB] Saved {pdb_id}.pdb ({size_kb:.1f} KB)")
            return pdb_path

        # .cif downloaded — convert to .pdb with gemmi
        cif_path = os.path.join(output_dir, f"{pdb_id}.cif")
        with open(cif_path, "wb") as f:
            f.write(resp.content)

        import gemmi
        st = gemmi.read_structure(cif_path)
        st.write_pdb(pdb_path)
        os.remove(cif_path)
        size_kb = os.path.getsize(pdb_path) / 1024
        print(f"[PDB] Converted {pdb_id}.cif → .pdb ({size_kb:.1f} KB)")
        return pdb_path

    raise ExternalCallError(
        "rcsb", "not_found",
        f"rcsb: no .pdb or .cif file available for {pdb_id}",
    )


# ---------------------------------------------------------------------------
# PubChem helpers
# ---------------------------------------------------------------------------

def lookup_smiles(drug_name: str) -> dict | None:
    """Look up a drug's SMILES and properties from PubChem by name.

    Tries the raw name first; if PubChem does not know it and the name contains
    parenthetical aliases like ``daraxonrasib (RMC-6236)``, retries
    with each component individually.  Returns None only for "not found";
    a PubChem failure raises ``ExternalCallError`` (trying aliases against a
    service that is down would not help).
    """
    result = _lookup_smiles_single(drug_name)
    if result:
        return result

    # Try sub-parts: "daraxonrasib (RMC-6236)" -> ["daraxonrasib", "RMC-6236"]
    import re
    parts = re.split(r"[()]", drug_name)
    parts = [p.strip() for p in parts if p.strip()]
    for part in parts:
        if part == drug_name:
            continue
        time.sleep(PUBCHEM_DELAY)
        result = _lookup_smiles_single(part)
        if result:
            return result

    return None


def _lookup_smiles_single(drug_name: str) -> dict | None:
    """Query PubChem for a single compound name."""
    encoded = requests.utils.quote(drug_name)
    url = (
        f"{PUBCHEM_BASE}/compound/name/{encoded}"
        f"/property/CanonicalSMILES,IUPACName,MolecularFormula/JSON"
    )

    # 404 = unknown name; 400 = a name PubChem cannot parse (LLM-written names
    # can be odd).  Neither is an outage, so neither may count as a failure.
    resp = request_with_retry("get", url, source="pubchem", timeout=15,
                              allow_status=_PUBCHEM_NO_MATCH)
    if resp.status_code in _PUBCHEM_NO_MATCH:
        print(f"[PubChem] '{drug_name}' not found")
        return None
    data = response_json(resp, "pubchem")

    props = (data.get("PropertyTable", {}).get("Properties") or [{}])[0]
    result = {
        "cid": props.get("CID"),
        "smiles": canonicalize_smiles(_extract_smiles(props)),
        "iupac_name": props.get("IUPACName", ""),
        "molecular_formula": props.get("MolecularFormula", ""),
    }
    smiles_preview = result["smiles"][:60]
    if len(result["smiles"]) > 60:
        smiles_preview += "…"
    print(
        f"[PubChem] Found '{drug_name}': "
        f"CID={result['cid']}, SMILES={smiles_preview}"
    )
    return result


def search_compounds_for_target(
    protein_name: str, max_compounds: int = 50
) -> list[dict]:
    """Find bioactive compounds for a protein target via PubChem.

    Uses PubChem's assay-target and gene-symbol endpoints to discover
    compounds that are pharmacologically active against the given target.

    Returns ``[]`` when PubChem has no active compounds for the target.  If a
    lookup *failed* and nothing was found, or the property fetch failed,
    ``ExternalCallError`` is raised so an outage is not read as "no compounds".
    """
    gene_symbol = protein_name.split()[0]  # "KRAS G12D" -> "KRAS"
    cids: list[int] = []
    lookup_errors: list[ExternalCallError] = []

    def collect_cids(url: str) -> None:
        resp = request_with_retry("get", url, source="pubchem", timeout=30,
                                  allow_status=_PUBCHEM_NO_MATCH)
        if resp.status_code == 200:
            data = response_json(resp, "pubchem")
            for entry in data.get("InformationList", {}).get("Information", []):
                cids.extend(entry.get("CID", []))

    # Strategy 1: bioassay target search by gene symbol
    url = (
        f"{PUBCHEM_BASE}/assay/target/genesymbol/"
        f"{requests.utils.quote(gene_symbol)}/cids/JSON"
        f"?cids_type=active"
    )
    print(f"[PubChem] Searching bioassays for gene target: {gene_symbol}")
    try:
        collect_cids(url)
    except ExternalCallError as e:
        print(f"[PubChem] Assay target search failed: {compact_error(e)}")
        lookup_errors.append(e)

    # Strategy 2 (fallback): gene/symbol pharmacologically-active compounds
    if not cids:
        url = (
            f"{PUBCHEM_BASE}/gene/symbol/"
            f"{requests.utils.quote(gene_symbol)}/cids/JSON"
            f"?cids_type=pharmacologically_active"
        )
        try:
            time.sleep(PUBCHEM_DELAY)
            collect_cids(url)
        except ExternalCallError as e:
            print(f"[PubChem] Gene symbol search failed: {compact_error(e)}")
            lookup_errors.append(e)

    # Deduplicate and limit
    cids = list(set(cids))[:max_compounds]

    if not cids:
        if lookup_errors:
            raise lookup_errors[-1]
        print(f"[PubChem] No bioactive compounds found for target '{protein_name}'")
        return []

    print(
        f"[PubChem] Found {len(cids)} active CIDs for '{gene_symbol}', "
        f"fetching properties …"
    )

    # Batch-fetch properties
    compounds = []
    batch_size = 100  # PubChem allows up to ~100 CIDs per request

    for i in range(0, len(cids), batch_size):
        batch = cids[i : i + batch_size]
        cid_str = ",".join(str(c) for c in batch)
        url = (
            f"{PUBCHEM_BASE}/compound/cid/{cid_str}"
            f"/property/CanonicalSMILES,IUPACName,MolecularFormula/JSON"
        )

        time.sleep(PUBCHEM_DELAY)
        resp = request_with_retry("get", url, source="pubchem", timeout=30,
                                  allow_status=(404,))
        if resp.status_code == 200:
            data = response_json(resp, "pubchem")
            for props in data.get("PropertyTable", {}).get("Properties", []):
                compounds.append(
                    {
                        "cid": props.get("CID"),
                        "smiles": canonicalize_smiles(_extract_smiles(props)),
                        "iupac_name": props.get("IUPACName", ""),
                        "molecular_formula": props.get("MolecularFormula", ""),
                    }
                )

    print(f"[PubChem] Retrieved properties for {len(compounds)} compounds")
    return compounds


def search_3d_similar(cid: int, max_results: int = 20) -> list[dict]:
    """Find compounds with similar 3D conformations to a given CID.

    Uses PubChem's fastsimilarity_3d endpoint which applies fixed thresholds
    of shape-Tanimoto >= 0.80 and color-Tanimoto >= 0.50.  This finds
    compounds whose 3D shape and functional-group orientation resemble the
    query — useful for discovering structurally analogous drug candidates.

    Returns a list of dicts with cid, smiles, iupac_name, molecular_formula;
    ``[]`` if the CID has no 3D conformer or no neighbours.  A failed request
    raises ``ExternalCallError``.
    """
    # Step 1: get similar CIDs
    url = (
        f"{PUBCHEM_BASE}/compound/fastsimilarity_3d/cid/{cid}/cids/JSON"
    )
    print(f"[PubChem 3D] Searching 3D-similar compounds for CID {cid} …")

    resp = request_with_retry("get", url, source="pubchem", timeout=60,
                              allow_status=(404,))
    if resp.status_code == 404:
        print(f"[PubChem 3D] No 3D conformer for CID {cid}")
        return []
    data = response_json(resp, "pubchem")

    similar_cids = data.get("IdentifierList", {}).get("CID", [])
    # Remove self-hit and limit
    similar_cids = [c for c in similar_cids if c != cid][:max_results]

    if not similar_cids:
        print(f"[PubChem 3D] No similar compounds found for CID {cid}")
        return []

    print(f"[PubChem 3D] Found {len(similar_cids)} similar CIDs, fetching properties …")

    # Step 2: batch-fetch properties
    cid_str = ",".join(str(c) for c in similar_cids)
    prop_url = (
        f"{PUBCHEM_BASE}/compound/cid/{cid_str}"
        f"/property/CanonicalSMILES,IUPACName,MolecularFormula/JSON"
    )

    compounds = []
    time.sleep(PUBCHEM_DELAY)
    resp = request_with_retry("get", prop_url, source="pubchem", timeout=30,
                              allow_status=(404,))
    if resp.status_code == 200:
        data = response_json(resp, "pubchem")
        for props in data.get("PropertyTable", {}).get("Properties", []):
            compounds.append(
                {
                    "cid": props.get("CID"),
                    "smiles": canonicalize_smiles(_extract_smiles(props)),
                    "iupac_name": props.get("IUPACName", ""),
                    "molecular_formula": props.get("MolecularFormula", ""),
                }
            )

    print(f"[PubChem 3D] Retrieved {len(compounds)} similar compounds for CID {cid}")
    return compounds


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _step(errors: list, guard: FailureGuard, what: str, target: str, fn, *args,
          default=None):
    """Run one external step.  Returns ``fn(*args)``; on ``ExternalCallError``
    records it in ``errors``, counts it against ``guard`` (which raises
    ``EscalationRequired`` after repeated failures) and returns ``default``."""
    try:
        value = fn(*args)
    except ExternalCallError as e:
        print(f"[Agent2] {what} failed for {target}: {compact_error(e)}",
              file=sys.stderr)
        errors.append({
            "stage": "structure",
            "source": e.source,
            "kind": e.kind,
            "message": compact_error(e),
            "target": target,
        })
        guard.failed(e)
        return default
    guard.ok()
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Agent 2: Retrieve protein structures and drug SMILES for docking."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="review.json",
        help="Input JSON from Agent 1 (default: review.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=OUTPUT_FILE,
        help=f"Output JSON for Agent 3 (default: {OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--structures-dir",
        type=str,
        default=STRUCTURES_DIR,
        help=f"Directory to save .pdb files (default: {STRUCTURES_DIR}).",
    )
    parser.add_argument(
        "--max-extra-compounds",
        type=int,
        default=50,
        help="Max extra compounds to discover per protein target (default: 50).",
    )
    args = parser.parse_args()

    # ----- Load Agent 1 output -----
    print(f"\n{'=' * 60}")
    print(f"  Agent 2 — Structure Retrieval")
    print(f"{'=' * 60}\n")

    if not os.path.exists(args.input):
        print(f"ERROR: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        agent1_data = json.load(f)

    disease = agent1_data.get("disease", "unknown")
    protein_targets = agent1_data.get("protein_targets", [])
    drugs = agent1_data.get("drugs", [])

    print(f"  Disease:      {disease}")
    print(f"  Targets:      {protein_targets}")
    print(f"  Known drugs:  {[d['drug'] for d in drugs]}")
    print()

    # ----- Process each protein target -----
    output_targets = []
    errors: list[dict] = []   # failed external calls (not the same as "no results")
    rcsb_guard = FailureGuard("rcsb")
    pubchem_guard = FailureGuard("pubchem")

    try:
        _process_targets(protein_targets, drugs, args, output_targets, errors,
                         rcsb_guard, pubchem_guard)
    except EscalationRequired as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ----- Write output -----
    output = {
        "disease": disease,
        "targets": output_targets,
        "errors": errors,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ----- Summary -----
    total_ligands = sum(len(t["ligands"]) for t in output_targets)
    print(f"\n{'=' * 60}")
    print(f"  Agent 2 complete")
    print(f"  Output:       {args.output}")
    print(f"  Targets:      {len(output_targets)}")
    print(f"  Total ligands: {total_ligands}")
    print(f"  PDB files:    {args.structures_dir}/")
    if errors:
        print(f"  Failed calls: {len(errors)} (see \"errors\" in the output)")
    print(f"{'=' * 60}\n")


def _process_targets(protein_targets, drugs, args, output_targets, errors,
                     rcsb_guard, pubchem_guard):
    """Steps 1-4 for every protein target; appends to ``output_targets``."""
    for protein in protein_targets:
        print(f"\n{'─' * 40}")
        print(f"  Processing target: {protein}")
        print(f"{'─' * 40}\n")

        # --- Step 1: Search PDB for best structure ---
        print(">> Step 1: Searching RCSB PDB …")
        candidates = _step(errors, rcsb_guard, "PDB search", protein,
                           search_pdb, protein, default=[])

        # Fallback: strip mutation notation and retry with bare gene name
        if not candidates:
            gene_name = protein.split()[0]
            if gene_name != protein:
                print(f"[PDB] Retrying with gene name only: {gene_name}")
                candidates = _step(errors, rcsb_guard, "PDB search", protein,
                                   search_pdb, gene_name, default=[])

        pdb_id = None
        pdb_file = None

        if candidates:
            best = _step(errors, rcsb_guard, "PDB metadata", protein,
                         pick_best_structure, candidates)
            if best:
                pdb_id = best["pdb_id"]

                # --- Step 2: Download PDB file ---
                print(f"\n>> Step 2: Downloading PDB structure {pdb_id} …")
                pdb_file = _step(errors, rcsb_guard, "PDB download", protein,
                                 download_pdb, pdb_id, args.structures_dir)
        else:
            print(f"[PDB] WARNING: No structures found for '{protein}'")

        # --- Step 3: Look up SMILES for drugs from Agent 1 ---
        print(f"\n>> Step 3: Looking up SMILES for known drugs …")
        ligands = []

        for drug_info in drugs:
            if protein not in drug_info.get("proteins", []):
                continue

            drug_name = drug_info["drug"]
            time.sleep(PUBCHEM_DELAY)
            pubchem_result = _step(errors, pubchem_guard, "SMILES lookup",
                                   f"{protein}/{drug_name}", lookup_smiles,
                                   drug_name)

            ligand = {
                "name": drug_name,
                "smiles": pubchem_result["smiles"] if pubchem_result else "",
                "mechanism": drug_info.get("mechanism", ""),
                "fda_status": drug_info.get("fda_status", "Unknown"),
                "source": (
                    f"pubchem_cid_{pubchem_result['cid']}"
                    if pubchem_result
                    else "agent1_no_smiles"
                ),
            }
            ligands.append(ligand)

        # --- Step 4: Discover additional compounds via PubChem ---
        print(
            f"\n>> Step 4: Searching PubChem for additional "
            f"compounds targeting {protein} …"
        )
        extra_compounds = _step(errors, pubchem_guard, "compound search",
                                protein, search_compounds_for_target, protein,
                                args.max_extra_compounds, default=[])

        # Merge extras, dedup by SMILES
        existing_smiles = {lig["smiles"] for lig in ligands if lig["smiles"]}
        for compound in extra_compounds:
            if compound["smiles"] and compound["smiles"] not in existing_smiles:
                ligands.append(
                    {
                        "name": compound.get("iupac_name", f"CID_{compound['cid']}"),
                        "smiles": compound["smiles"],
                        "mechanism": "Bioactive compound — discovered via PubChem target search",
                        "fda_status": "Unknown — requires verification",
                        "source": f"pubchem_cid_{compound['cid']}",
                    }
                )
                existing_smiles.add(compound["smiles"])

        # Drop any ligands that have no SMILES (can't dock without structure)
        valid_ligands = [lig for lig in ligands if lig["smiles"]]
        skipped = len(ligands) - len(valid_ligands)
        if skipped:
            print(f"[Agent2] Skipped {skipped} ligand(s) with no SMILES string")

        target_entry = {
            "protein": protein,
            "pdb_id": pdb_id,
            "pdb_file": pdb_file,
            "ligands": valid_ligands,
        }
        output_targets.append(target_entry)

        print(
            f"\n  >> {protein}: PDB={pdb_id}, "
            f"{len(valid_ligands)} ligands ready for docking"
        )


if __name__ == "__main__":
    main()
