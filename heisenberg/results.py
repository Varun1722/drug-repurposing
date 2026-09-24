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

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

def query_openai(system_prompt: str, user_prompt: str) -> str:
    """Send a chat-completion request to the OpenAI API and return the
    assistant's reply text."""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set.",
              file=sys.stderr)
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # No custom temperature — some models (e.g. gpt-5.6-terra) only
        # support the default (1) and reject any other value.
        "max_completion_tokens": 8000,
    }

    print("[OpenAI] Sending request …")
    resp = requests.post(OPENAI_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    reply = data["choices"][0]["message"]["content"]
    print(f"[OpenAI] Received {len(reply)} chars.")
    return reply


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
    system = (
        "You are an expert computational biologist writing the methodology "
        "section of a peer-reviewed research paper. Write in formal academic "
        "prose using Markdown formatting. Include specific technical details. "
        "Use inline citations like [1], [2] when citing GNINA or Modal. "
        "Do NOT invent results — describe only the experimental setup."
    )
    user = (
        f"Write a **Methodology** section for a research paper studying "
        f"potential drug candidates for **{disease}** via molecular docking.\n\n"
        f"Use the following factual details:\n\n"
        f"1. **Molecular docking engine**: GNINA — a CNN-scored molecular "
        f"docking tool built on AutoDock Vina's search algorithm, which "
        f"rescores generated poses with an ensemble of trained convolutional "
        f"neural networks. GNINA generates multiple binding poses per "
        f"ligand–protein pair and ranks them by a learned CNN pose score "
        f"(0 – 1, higher = better predicted binding).\n\n"
        f"2. **Compute infrastructure**: All docking simulations were run on "
        f"Modal serverless CPU workers. Each job was submitted to a dedicated "
        f"serverless function, enabling parallel CPU inference without "
        f"managing local hardware.\n\n"
        f"3. **Docking protocol**: For every ligand–protein pair, 10 binding "
        f"poses were generated. The pose with the highest confidence score "
        f"was retained as the representative result for ranking.\n\n"
        f"4. **Protein structures**: Target proteins were obtained from the "
        f"RCSB Protein Data Bank (PDB) in standard .pdb format.\n\n"
        f"5. **Ligand sources**: Candidate compounds were retrieved from "
        f"PubChem and curated drug databases, including both FDA-approved "
        f"drugs and bioactive research compounds identified through target-"
        f"based searches.\n\n"
        f"6. **Run summary**:\n{overview}\n\n"
        f"Structure the section with clear sub-headings (e.g., "
        f"'Molecular Docking Engine', 'Compute Infrastructure', "
        f"'Docking Protocol', 'Protein Target Selection', "
        f"'Ligand Library Construction')."
    )
    return query_openai(system, user)


def generate_results(
    disease: str,
    overview: str,
    classified: dict,
) -> str:
    """Ask OpenAI to write the Results section, split into
    disease-purposed drugs, a repurposing leaderboard, and novel compounds."""
    system = (
        "You are an expert computational biologist writing the results "
        "section of a peer-reviewed drug-repurposing research paper. "
        "Write in formal academic prose using Markdown formatting. "
        "Present the data objectively — interpret findings only where "
        "directly supported by the numbers. Include Markdown tables."
    )
    user = (
        f"Write a **Results** section for a research paper whose goal is "
        f"to identify **drug repurposing candidates** for "
        f"**{disease}** via molecular docking.\n\n"
        f"## Full docking data\n\n{overview}\n\n"
        f"The compounds have been classified into three categories:\n\n"
        f"### A. Drugs Currently Approved for {disease.title()}\n\n"
        f"{classified['disease_purposed']}\n\n"
        f"### B. Repurposing Candidates — FDA-Approved for Other Indications\n\n"
        f"{classified['repurposing']}\n\n"
        f"### C. Novel / Research Compounds (Not Yet FDA-Approved)\n\n"
        f"{classified['novel']}\n\n"
        f"The Results section MUST include:\n\n"
        f"1. **Overview of Docking Runs** — number of targets, ligands, "
        f"and total compute time.\n"
        f"2. **Drugs Currently Purposed for {disease.title()}** — "
        f"present a ranked table of the disease-purposed drugs and their "
        f"confidence scores per target. Comment on their performance as a "
        f"baseline.  If none exist, state that explicitly.\n"
        f"3. **Drug Repurposing Leaderboard** — THIS IS THE MOST IMPORTANT "
        f"SECTION.  Present a single **combined leaderboard table** of ALL "
        f"FDA-approved drugs that are approved for OTHER indications "
        f"(not {disease}), ranked by their best GNINA confidence "
        f"score across all targets.  Columns: Rank, Drug Name, Best "
        f"Confidence Score, Target Protein, Original Approved Indication, "
        f"Proposed Mechanism.  Below the table, discuss the top 5 "
        f"candidates in detail — why their scores are noteworthy, what "
        f"their original indication is, and what evidence supports "
        f"repurposing potential.\n"
        f"4. **Novel / Research Compounds** — briefly present top-scoring "
        f"non-FDA compounds for completeness.\n"
        f"5. **Cross-Target Comparison** — highlight any compound that "
        f"scores well against multiple targets.\n"
        f"6. **Score Distribution Analysis** — compare score ranges "
        f"between disease-purposed drugs, repurposing candidates, and "
        f"novel compounds.\n\n"
        f"Use specific numbers from the data. Do not fabricate results."
    )
    return query_openai(system, user)


def generate_conclusion(
    disease: str,
    overview: str,
    classified: dict,
) -> str:
    """Ask OpenAI to write the Conclusion section, emphasising the
    drug-repurposing findings."""
    system = (
        "You are an expert computational biologist writing the conclusion "
        "of a peer-reviewed drug-repurposing research paper. Write in "
        "formal academic prose using Markdown formatting. Ground all "
        "statements in the data provided; clearly distinguish "
        "computational predictions from experimental evidence."
    )
    user = (
        f"Write a **Conclusion** section for a research paper whose "
        f"primary goal was to identify **drug repurposing candidates** "
        f"for **{disease}** via molecular docking.\n\n"
        f"Data summary:\n\n{overview}\n\n"
        f"Repurposing leaderboard data:\n\n{classified['repurposing']}\n\n"
        f"The Conclusion MUST cover:\n\n"
        f"1. **Summary of Key Findings** — which repurposed drugs showed "
        f"the strongest predicted binding and for which target(s). "
        f"Highlight how they compare to drugs already approved for "
        f"{disease} (if any were tested).\n"
        f"2. **Top Repurposing Candidates** — name the most promising "
        f"FDA-approved drugs from other therapeutic areas, their original "
        f"indications, and why their docking scores suggest they merit "
        f"further investigation for {disease}.\n"
        f"3. **Clinical Implications** — what these docking results "
        f"suggest for accelerated clinical trials, given that repurposed "
        f"drugs already have established safety profiles. Note these are "
        f"computational predictions requiring experimental validation.\n"
        f"4. **Limitations** — limitations of GNINA confidence scores "
        f"as proxies for actual binding affinity, absence of molecular "
        f"dynamics / free-energy perturbation follow-up, limited target "
        f"conformational sampling, etc.\n"
        f"5. **Future Directions** — recommended next steps: MD "
        f"simulations, in vitro binding assays, ADMET profiling, and "
        f"preclinical testing for the top repurposing candidates."
    )
    return query_openai(system, user)


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
