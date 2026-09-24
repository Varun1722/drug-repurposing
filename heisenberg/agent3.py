#!/usr/bin/env python3
"""
Agent 3 — GNINA Docking Simulation Agent

Reads Agent 2's output (agent2_output.json) containing protein structures and
drug SMILES strings, sends them to GNINA running on a Modal serverless CPU
worker (modal_app/gnina_worker.py), and returns ranked results with
confidence scores normalized to 0-1.

Pipeline mode (reads Agent 2 output):
    python agent3.py

Manual mode:
    python agent3.py --protein structures/6GJ8.pdb --ligands ligands.json

Test mode:
    python agent3.py --test

Environment variables:
    GNINA_ENDPOINT_URL  - URL of the deployed Modal worker's web endpoint
                          (printed by `modal deploy modal_app/gnina_worker.py`)
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE = "agent2_output.json"
OUTPUT_FILE = "agent3_output.json"

EXAMPLE_LIGANDS = [
    {"name": "sotorasib", "smiles": "C=CC(=O)N1CCC(CC1)n2c(=O)c3cc(F)c(cc3n2c4ccc(cc4)c5nc(cnc5OC)N)OC"},
    {"name": "adagrasib", "smiles": "Cc1c(F)c(C)c(Cl)c(Nc2nc3c(c(n2)C(=O)N4CCC(CC4)N5CC(C)C(F)(F)C5)ccn3C(C)C)c1F"},
    {"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    {"name": "ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"},
    {"name": "caffeine", "smiles": "Cn1c(=O)c2c(ncn2C)n(c1=O)C"},
]


# ---------------------------------------------------------------------------
# GNINA worker helpers
# ---------------------------------------------------------------------------

def encode_pdb(pdb_path: str) -> str:
    """Base64-encode a PDB file."""
    with open(pdb_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of at most chunk_size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def submit_chunk(endpoint_url: str, protein_pdb_b64: str, ligand_chunk: list[dict],
                 samples_per_complex: int, chunk_idx: int):
    """Submit a single chunk to the GNINA Modal worker and return its output.

    This is a plain synchronous HTTP call — the worker docks the whole chunk
    and replies once it's done, no submit/poll needed.
    """
    payload = {
        "protein_pdb_b64": protein_pdb_b64,
        "ligands": ligand_chunk,
        "samples_per_complex": samples_per_complex,
    }

    print(f"    [chunk {chunk_idx}] Docking {len(ligand_chunk)} ligands …")
    resp = requests.post(endpoint_url, json=payload, timeout=600)

    if not resp.ok:
        print(f"    [chunk {chunk_idx}] GNINA worker error {resp.status_code}: {resp.text[:500]}", flush=True)
        return {"error": f"Chunk {chunk_idx} HTTP {resp.status_code}", "results": []}

    output = resp.json()
    for err in output.get("errors", []):
        print(f"    [chunk {chunk_idx}] {err['name']}: {err['error']}", flush=True)
    print(f"    [chunk {chunk_idx}] Done ({output.get('processing_time_seconds', '?')}s)", flush=True)
    return output


def run_docking(
    protein_pdb_path: str,
    ligands: list[dict],
    endpoint_url: str = None,
    chunk_size: int = 10,
    samples_per_complex: int = 10,
) -> list[dict]:
    """
    Run GNINA molecular docking via a Modal serverless CPU worker.

    Args:
        protein_pdb_path: Path to the protein .pdb file
        ligands: List of dicts with "name" and "smiles" keys
        endpoint_url: GNINA Modal worker URL
        chunk_size: Number of ligands per worker request
        samples_per_complex: Max poses to generate per drug-protein pair

    Returns:
        List of result dicts sorted by confidence_score (descending).
    """
    endpoint_url = endpoint_url or os.environ.get("GNINA_ENDPOINT_URL")

    if not endpoint_url:
        raise ValueError(
            "GNINA_ENDPOINT_URL not set. Deploy modal_app/gnina_worker.py "
            "with `modal deploy` and set the printed URL."
        )

    protein_pdb_b64 = encode_pdb(protein_pdb_path)

    chunks = chunk_list(ligands, chunk_size)
    n_chunks = len(chunks)
    print(f"  Docking {len(ligands)} ligands in {n_chunks} chunk(s) of ≤{chunk_size}")

    all_results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=min(n_chunks, 8)) as executor:
        futures = {
            executor.submit(
                submit_chunk, endpoint_url, protein_pdb_b64, chunk,
                samples_per_complex, i
            ): i
            for i, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                output = future.result()
                if output is None:
                    print(f"    [chunk {chunk_idx}] Warning: empty output")
                    continue
                chunk_results = output.get("results", [])
                all_results.extend(chunk_results)
            except Exception as e:
                print(f"    [chunk {chunk_idx}] Error: {e}")

    elapsed = time.time() - start_time

    all_results.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)

    print(f"  Completed {len(all_results)}/{len(ligands)} ligands in {elapsed:.1f}s")
    return all_results, elapsed


# ---------------------------------------------------------------------------
# Pipeline mode: read Agent 2 output, dock all targets
# ---------------------------------------------------------------------------

def _flush_output(output_file: str, data: dict):
    """Atomically write the current pipeline state to disk."""
    tmp = output_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, output_file)


def run_pipeline(
    input_file: str,
    output_file: str,
    samples_per_complex: int = 10,
    chunk_size: int = 10,
    endpoint_url: str = None,
) -> dict:
    """
    Read agent2_output.json, run docking for every target, write agent3_output.json.
    The output file is updated live after each target so progress can be monitored.
    """
    print(f"\n{'=' * 60}")
    print(f"  Agent 3 — GNINA Docking Simulation")
    print(f"{'=' * 60}\n")

    if not os.path.exists(input_file):
        print(f"ERROR: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(input_file) as f:
        agent2_data = json.load(f)

    disease = agent2_data.get("disease", "unknown")
    targets = agent2_data.get("targets", [])

    print(f"  Disease:      {disease}")
    print(f"  Targets:      {len(targets)}")
    for t in targets:
        print(f"    • {t['protein']} — PDB: {t.get('pdb_id', 'N/A')}, {len(t.get('ligands', []))} ligands")
    print()

    # Pre-build all target entries with "queued" status so the JSON shows
    # the full queue from the start.
    output_targets = []
    for target in targets:
        output_targets.append({
            "protein": target["protein"],
            "pdb_id": target.get("pdb_id"),
            "pdb_file": target.get("pdb_file"),
            "status": "queued",
            "num_ligands_total": len(target.get("ligands", [])),
            "num_ligands_docked": 0,
            "docking_time_seconds": 0,
            "results": [],
        })

    output = {
        "disease": disease,
        "status": "running",
        "completed_targets": 0,
        "total_targets": len(targets),
        "total_docking_time_seconds": 0,
        "targets": output_targets,
    }
    _flush_output(output_file, output)

    total_time = 0

    for idx, target in enumerate(targets):
        protein = target["protein"]
        pdb_file = target.get("pdb_file")
        ligands = target.get("ligands", [])

        print(f"\n{'─' * 40}")
        print(f"  Docking: {protein} ({len(ligands)} ligands)")
        print(f"{'─' * 40}\n")

        # Mark this target as "docking" and flush
        output_targets[idx]["status"] = "docking"
        _flush_output(output_file, output)

        if not pdb_file or not os.path.exists(pdb_file):
            print(f"  WARNING: PDB file not found: {pdb_file}")
            print(f"  Skipping target {protein}")
            output_targets[idx]["status"] = "error"
            output_targets[idx]["error"] = f"PDB file not found: {pdb_file}"
            output["completed_targets"] += 1
            _flush_output(output_file, output)
            continue

        if not ligands:
            print(f"  WARNING: No ligands for target {protein}")
            output_targets[idx]["status"] = "error"
            output_targets[idx]["error"] = "No ligands provided"
            output["completed_targets"] += 1
            _flush_output(output_file, output)
            continue

        # Format ligands for the worker (needs "name" and "smiles")
        dock_ligands = [
            {"name": lig["name"], "smiles": lig["smiles"]}
            for lig in ligands
            if lig.get("smiles")
        ]

        results, elapsed = run_docking(
            protein_pdb_path=pdb_file,
            ligands=dock_ligands,
            endpoint_url=endpoint_url,
            chunk_size=chunk_size,
            samples_per_complex=samples_per_complex,
        )
        total_time += elapsed

        # Merge Agent 2 metadata back into results
        ligand_meta = {lig["name"]: lig for lig in ligands}
        for r in results:
            meta = ligand_meta.get(r["name"], {})
            r["mechanism"] = meta.get("mechanism", "")
            r["fda_status"] = meta.get("fda_status", "")
            r["source"] = meta.get("source", "")

        output_targets[idx].update({
            "status": "completed",
            "num_ligands_docked": len(results),
            "docking_time_seconds": round(elapsed, 2),
            "results": results,
        })
        output["completed_targets"] += 1
        output["total_docking_time_seconds"] = round(total_time, 2)
        _flush_output(output_file, output)

        # Print top 5 for this target
        print(f"\n  Top hits for {protein}:")
        print(f"  {'Rank':<6} {'Drug':<30} {'Score':<10} {'Raw':<10}")
        print(f"  {'-' * 56}")
        for i, r in enumerate(results[:5]):
            name = r["name"][:28]
            print(f"  {i+1:<6} {name:<30} {r['confidence_score']:<10.4f} {r['confidence_raw']:<10.4f}")

    # ----- Final write -----
    output["status"] = "completed"
    output["total_docking_time_seconds"] = round(total_time, 2)
    _flush_output(output_file, output)

    # ----- Summary -----
    total_docked = sum(t.get("num_ligands_docked", 0) for t in output_targets)
    print(f"\n{'=' * 60}")
    print(f"  Agent 3 complete")
    print(f"  Output:         {output_file}")
    print(f"  Targets docked: {len([t for t in output_targets if t['status'] == 'completed'])}")
    print(f"  Total ligands:  {total_docked}")
    print(f"  Total time:     {total_time:.1f}s")
    print(f"{'=' * 60}\n")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_results_table(results: list[dict]):
    """Print a formatted results table."""
    print(f"\nTop results:")
    print(f"{'Rank':<6} {'Drug':<30} {'Score':<10} {'Raw':<10}")
    print("-" * 56)
    for i, r in enumerate(results[:10]):
        name = r["name"][:28]
        print(f"{i+1:<6} {name:<30} {r['confidence_score']:<10.4f} {r['confidence_raw']:<10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Agent 3: GNINA Docking Simulation Agent"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=INPUT_FILE,
        help=f"Input JSON from Agent 2 (default: {INPUT_FILE}). Used in pipeline mode.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=OUTPUT_FILE,
        help=f"Output JSON for Agent 4 (default: {OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--protein",
        type=str,
        default=None,
        help="Path to protein .pdb file (manual mode — bypasses Agent 2 input).",
    )
    parser.add_argument(
        "--ligands",
        type=str,
        default=None,
        help="Path to JSON file with ligands [{name, smiles}, ...] (manual mode).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use built-in example ligands (requires --protein).",
    )
    parser.add_argument("--endpoint-url", type=str, default=None,
                        help="GNINA Modal worker URL (defaults to GNINA_ENDPOINT_URL env var).")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10,
                        help="Max poses per drug-protein pair (default: 10)")

    args = parser.parse_args()

    # ----- Manual mode -----
    if args.protein:
        if args.test:
            ligands = EXAMPLE_LIGANDS
        elif args.ligands:
            with open(args.ligands) as f:
                ligands = json.load(f)
        else:
            print("Error: --protein requires --ligands <file.json> or --test")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print(f"  Agent 3 — GNINA Docking Simulation (manual mode)")
        print(f"{'=' * 60}\n")

        results, elapsed = run_docking(
            protein_pdb_path=args.protein,
            ligands=ligands,
            endpoint_url=args.endpoint_url,
            chunk_size=args.chunk_size,
            samples_per_complex=args.samples,
        )

        _print_results_table(results)

        output = {
            "results": results,
            "processing_time_seconds": round(elapsed, 2),
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
        return

    # ----- Pipeline mode -----
    run_pipeline(
        input_file=args.input,
        output_file=args.output,
        samples_per_complex=args.samples,
        chunk_size=args.chunk_size,
        endpoint_url=args.endpoint_url,
    )


if __name__ == "__main__":
    main()
