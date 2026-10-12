#!/usr/bin/env python3
"""
Docking Results Report Generator

Reads GNINA simulation output (agent3_output.json) and uses the
OpenAI API to generate a research-paper–style Methodology, Results,
and Conclusion section, saved as results.md.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

from llm import query_prompt

def _strip_leading_header(text: str) -> str:
    """Remove a leading Markdown header (e.g. '## Methodology\\n') that
    the LLM often echoes back, which would duplicate the header we
    already add during document assembly."""
    stripped = text.lstrip()
    # Match one or more '#' followed by any text, then a newline
    cleaned = re.sub(r"^#{1,4}\s+[^\n]*\n+", "", stripped, count=1)
    return cleaned


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def load_docking_data(path: str) -> dict:
    """Load and return the agent3_output.json data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarise_target(target: dict) -> dict:
    """Extract a compact summary of a single protein target's docking
    results (dropping bulky SDF payloads)."""
    results_compact = []
    for r in target.get("results", []):
        results_compact.append({
            "rank": len(results_compact) + 1,
            "name": r["name"],
            "confidence_score": round(r["confidence_score"], 4),
            "confidence_raw": round(r["confidence_raw"], 4),
            "mechanism": r.get("mechanism", ""),
            "fda_status": r.get("fda_status", ""),
            "source": r.get("source", ""),
            "num_poses": len(r.get("all_poses", [])),
        })
    # Sort by confidence_score descending (should already be, but ensure)
    results_compact.sort(key=lambda x: x["confidence_score"], reverse=True)
    # Assign ranks after sorting
    for i, r in enumerate(results_compact, 1):
        r["rank"] = i

    return {
        "protein": target["protein"],
        "pdb_id": target["pdb_id"],
        "num_ligands_total": target.get("num_ligands_total", 0),
        "num_ligands_docked": target.get("num_ligands_docked", 0),
        "docking_time_seconds": target.get("docking_time_seconds", 0),
        "results": results_compact,
    }


def format_target_summary_for_prompt(target_summary: dict) -> str:
    """Format a target summary as human-readable text for LLM prompts."""
    lines = [
        f"### Protein: {target_summary['protein']}  (PDB: {target_summary['pdb_id']})",
        f"- Ligands submitted: {target_summary['num_ligands_total']}",
        f"- Ligands successfully docked: {target_summary['num_ligands_docked']}",
        f"- Docking wall-clock time: {target_summary['docking_time_seconds']:.1f} s",
        "",
        "| Rank | Compound | Confidence | Mechanism | FDA Status | Source |",
        "|------|----------|------------|-----------|------------|--------|",
    ]
    for r in target_summary["results"]:
        lines.append(
            f"| {r['rank']} | {r['name'][:80]} | {r['confidence_score']:.4f} "
            f"| {r['mechanism'][:60]} | {r['fda_status'][:40]} | {r['source']} |"
        )
    return "\n".join(lines)


def classify_compound(compound: dict, disease: str) -> str:
    """Classify a compound as 'disease_purposed', 'repurposing', or 'novel'.

    - disease_purposed : FDA status explicitly mentions this disease
                         (e.g. 'FDA-approved for pancreatic cancer').
    - repurposing      : FDA-approved for a *different* indication.
    - novel            : Not FDA-approved / unknown / research compound.
    """
    fda = (compound.get("fda_status") or "").lower()
    disease_lower = disease.lower()

    # Extract the core disease word (e.g. 'pancreatic') for flexible matching
    disease_words = [w for w in disease_lower.split() if len(w) > 3]

    if "fda" in fda and "approved" in fda:
        # Check whether the approval is for THIS disease
        if any(w in fda for w in disease_words) or disease_lower in fda:
            return "disease_purposed"
        return "repurposing"

    return "novel"


def build_overview_block(data: dict, target_summaries: list[dict]) -> str:
    """Build a textual overview of the entire docking run for the LLM."""
    lines = [
        f"Disease: {data['disease']}",
        f"Overall status: {data['status']}",
        f"Total protein targets: {data['total_targets']}",
        f"Total docking wall-clock time: {data['total_docking_time_seconds']:.1f} seconds",
        "",
    ]
    for ts in target_summaries:
        lines.append(format_target_summary_for_prompt(ts))
        lines.append("")
    return "\n".join(lines)


def build_classified_overview(
    target_summaries: list[dict], disease: str
) -> dict:
    """Split every target's results into three buckets and return a dict
    with formatted text blocks for:
      - 'disease_purposed' (drugs approved for this disease)
      - 'repurposing'      (drugs approved for other indications)
      - 'novel'            (research / unknown compounds)
    Each value is a formatted text string ready for an LLM prompt.
    """
    buckets: dict[str, list[str]] = {
        "disease_purposed": [],
        "repurposing": [],
        "novel": [],
    }

    for ts in target_summaries:
        # Partition results
        categorised: dict[str, list[dict]] = {
            "disease_purposed": [],
            "repurposing": [],
            "novel": [],
        }
        for r in ts["results"]:
            cat = classify_compound(r, disease)
            categorised[cat].append(r)

        for cat, compounds in categorised.items():
            if not compounds:
                continue
            header = (
                f"### Protein: {ts['protein']}  (PDB: {ts['pdb_id']})\n"
                f"- Compounds in this category: {len(compounds)}\n"
            )
            table = (
                "| Rank | Compound | Confidence | Mechanism "
                "| FDA Status | Source |\n"
                "|------|----------|------------|-----------|"
                "------------|--------|\n"
            )
            for i, c in enumerate(compounds, 1):
                table += (
                    f"| {i} | {c['name'][:80]} | {c['confidence_score']:.4f} "
                    f"| {c['mechanism'][:60]} | {c['fda_status'][:40]} "
                    f"| {c['source']} |\n"
                )
            buckets[cat].append(header + table)

    return {k: "\n".join(v) if v else "(none)" for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_methodology(disease: str, overview: str) -> str:
    """Ask OpenAI to write the Methodology section."""
    return query_prompt("report/methodology", disease=disease, overview=overview)


def generate_results(
    disease: str,
    overview: str,
    classified: dict,
) -> str:
    """Ask OpenAI to write the Results section, split into
    disease-purposed drugs, a repurposing leaderboard, and novel compounds."""
    return query_prompt(
        "report/results",
        disease=disease,
        disease_title=disease.title(),
        overview=overview,
        disease_purposed=classified["disease_purposed"],
        repurposing=classified["repurposing"],
        novel=classified["novel"],
    )


def generate_conclusion(
    disease: str,
    overview: str,
    classified: dict,
) -> str:
    """Ask OpenAI to write the Conclusion section, emphasising the
    drug-repurposing findings."""
    return query_prompt(
        "report/conclusion",
        disease=disease,
        overview=overview,
        repurposing=classified["repurposing"],
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown results report from GNINA "
                    "docking output."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="agent3_output.json",
        help="Path to the GNINA results JSON (default: agent3_output.json).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="results.md",
        help="Output Markdown file (default: results.md).",
    )
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    # ----- Step 1: Load and parse docking data -----
    print("\n>> Step 1: Loading docking data …")
    if not os.path.isfile(input_file):
        print(f"ERROR: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    data = load_docking_data(input_file)
    disease = data.get("disease", "unknown disease")

    print(f"    Disease     : {disease}")
    print(f"    Targets     : {data.get('total_targets', '?')}")
    print(f"    Status      : {data.get('status', '?')}")

    target_summaries = [summarise_target(t) for t in data.get("targets", [])]
    overview = build_overview_block(data, target_summaries)

    total_ligands = sum(ts["num_ligands_docked"] for ts in target_summaries)
    print(f"    Ligands     : {total_ligands} (docked)")
    print(f"    Total time  : {data.get('total_docking_time_seconds', 0):.1f} s")

    print(f"\n{'='*60}")
    print(f"  Docking Results Report  —  {disease}")
    print(f"{'='*60}\n")

    # ----- Step 2: Classify compounds -----
    print("\n>> Step 2: Classifying compounds …")
    classified = build_classified_overview(target_summaries, disease)
    for cat, label in [
        ("disease_purposed", "Approved for this disease"),
        ("repurposing", "Repurposing candidates"),
        ("novel", "Novel / research"),
    ]:
        count = classified[cat].count("|") // 7  # rough row count
        print(f"    {label:30s}: ~{count} compounds")

    # ----- Step 3: Generate Methodology -----
    print("\n>> Step 3: Generating Methodology section …")
    methodology = generate_methodology(disease, overview)

    # ----- Step 4: Generate Results -----
    print("\n>> Step 4: Generating Results section …")
    results = generate_results(disease, overview, classified)

    # ----- Step 5: Generate Conclusion -----
    print("\n>> Step 5: Generating Conclusion section …")
    conclusion = generate_conclusion(disease, overview, classified)

    # ----- Step 6: Assemble final document -----
    print("\n>> Step 6: Assembling final Markdown document …")

    # Strip leading duplicate headers that the LLM may echo back
    methodology = _strip_leading_header(methodology)
    results = _strip_leading_header(results)
    conclusion = _strip_leading_header(conclusion)

    today = datetime.now().strftime("%B %d, %Y")
    final_md = (
        f"# Molecular Docking Analysis: {disease.title()}\n\n"
        f"*Auto-generated on {today} using GNINA, Modal, and "
        f"OpenAI.*\n\n"
        f"---\n\n"
        f"## Methodology\n\n"
        f"{methodology}\n\n"
        f"---\n\n"
        f"## Results\n\n"
        f"{results}\n\n"
        f"---\n\n"
        f"## Conclusion\n\n"
        f"{conclusion}\n"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_md)

    print(f"\n{'='*60}")
    print(f"  Report saved to: {output_file}")
    print(f"  Targets analysed: {len(target_summaries)}")
    print(f"  Total ligands: {total_ligands}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
