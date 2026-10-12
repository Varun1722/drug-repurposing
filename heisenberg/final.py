#!/usr/bin/env python3
"""
Final Paper Generator

Reads review.md (literature review) and results.md (docking results),
merges them into a single polished research paper with an abstract,
de-duplicated content, and a unified reference list, then outputs
final_paper.md.

Uses the OpenAI API for generating the abstract and for
intelligently merging overlapping content.
"""

import argparse
import os
import re
import sys
from datetime import datetime

from llm import query_prompt

# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _strip_leading_header(text: str) -> str:
    """Remove a leading Markdown header that would duplicate our own."""
    stripped = text.lstrip()
    return re.sub(r"^#{1,4}\s+[^\n]*\n+", "", stripped, count=1)


def _read_md(path: str) -> str:
    """Read a Markdown file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_references(md_text: str) -> str:
    """Pull out the consolidated references section from a Markdown file."""
    # Look for "## Consolidated References" or "## References" near the end
    patterns = [
        r"(## Consolidated References.*)",
        r"(## References\s*\n(?:\s*\[?\d+\]?.*\n?)+)",
    ]
    for pat in patterns:
        m = re.search(pat, md_text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


def _extract_sections(md_text: str) -> list[tuple[str, str]]:
    """Split a Markdown document into (heading, body) tuples at ## level."""
    parts = re.split(r"^(## .+)$", md_text, flags=re.MULTILINE)
    sections = []
    # parts[0] is content before the first ## heading
    preamble = parts[0].strip()
    if preamble:
        sections.append(("_preamble", preamble))
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((heading, body))
    return sections


def _deduplicate_references(review_refs: str, results_refs: str) -> str:
    """Merge reference blocks, keeping unique entries only.

    Returns a re-numbered reference block.
    """
    ref_pattern = re.compile(
        r"\[(\d+)\]\s*(.*?)(?=\n\[|\n*$)", re.DOTALL
    )

    seen_titles: dict[str, str] = {}  # normalised title → full entry
    ordered: list[str] = []

    for block in [review_refs, results_refs]:
        for m in ref_pattern.finditer(block):
            entry = m.group(2).strip()
            # Normalise: lowercase, strip punctuation for dedup key
            key = re.sub(r"[^a-z0-9]", "", entry.lower())[:120]
            if key and key not in seen_titles:
                seen_titles[key] = entry
                ordered.append(entry)

    lines = []
    for i, entry in enumerate(ordered, 1):
        lines.append(f"[{i}] {entry}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def generate_abstract(review_text: str, results_text: str) -> str:
    """Generate a concise, publication-quality abstract."""
    return query_prompt(
        "paper/abstract",
        review_text=review_text[:6000],
        results_text=results_text[:6000],
    )


def merge_introduction_and_background(
    review_text: str, results_text: str
) -> str:
    """Merge the introduction, molecular landscape, and protein-target
    sections from the review into a single polished Introduction & Background
    section. Preserve inline citation numbers."""
    return query_prompt("paper/introduction", sections=review_text)


def merge_drug_landscape(review_drug_text: str) -> str:
    """Polish the FDA-approved drugs and repurposing-candidates sections from
    the review into a single Therapeutic Landscape section."""
    return query_prompt("paper/therapeutic_landscape", drug_text=review_drug_text)


def merge_results(results_text: str) -> str:
    """Polish the docking results section into final-paper quality."""
    return query_prompt("paper/results", results_text=results_text)


def merge_conclusions(
    review_conclusion: str, results_conclusion: str
) -> str:
    """Merge the two conclusion sections into one unified conclusion."""
    return query_prompt(
        "paper/conclusion",
        review_conclusion=review_conclusion,
        results_conclusion=results_conclusion,
    )


# ---------------------------------------------------------------------------
# Content extraction helpers
# ---------------------------------------------------------------------------

def _get_review_intro_sections(review_md: str) -> str:
    """Extract Introduction, Molecular Landscape, Key Protein Targets,
    and Current Therapeutic Strategies from review.md."""
    sections = _extract_sections(review_md)
    target_headings = {
        "introduction", "molecular", "genetic", "landscape",
        "key protein", "protein targets", "therapeutic strategies",
        "clinical relevance",
    }
    collected = []
    for heading, body in sections:
        heading_lower = heading.lower()
        if any(kw in heading_lower for kw in target_headings):
            collected.append(f"{heading}\n\n{body}")
    return "\n\n---\n\n".join(collected) if collected else ""


def _get_review_drug_sections(review_md: str) -> str:
    """Extract FDA drug and repurposing sections from review.md."""
    sections = _extract_sections(review_md)
    target_headings = {
        "fda", "drug", "repurpos", "candidate", "interaction",
    }
    collected = []
    for heading, body in sections:
        heading_lower = heading.lower()
        if any(kw in heading_lower for kw in target_headings):
            collected.append(f"{heading}\n\n{body}")
    # Also grab anything between the first --- after protein targets
    # and the consolidated references
    repurpose_match = re.search(
        r"(### FDA-Approved Drugs.*?)(?=## Consolidated References|$)",
        review_md, re.DOTALL
    )
    if repurpose_match:
        collected.append(repurpose_match.group(1).strip())
    return "\n\n---\n\n".join(collected) if collected else ""


def _get_review_conclusion(review_md: str) -> str:
    """Extract the conclusion from review.md."""
    sections = _extract_sections(review_md)
    for heading, body in sections:
        if "conclusion" in heading.lower() or "future" in heading.lower():
            return body
    # Fallback: find ### Conclusion
    m = re.search(
        r"### Conclusion.*?\n(.*?)(?=###|\n## |$)", review_md, re.DOTALL
    )
    return m.group(1).strip() if m else ""


def _get_results_methodology(results_md: str) -> str:
    """Extract the full Methodology block from results.md.

    The LLM often outputs methodology sub-sections (Molecular Docking
    Engine, Compute Infrastructure, etc.) at the ## level rather than ###,
    so we collect everything from '## Methodology' up to '## Results' and
    demote the sub-headings to ### for clean nesting in the final paper.
    """
    sections = _extract_sections(results_md)
    collecting = False
    collected: list[str] = []
    for heading, body in sections:
        h = heading.lower()
        if "methodol" in h:
            collecting = True
            if body.strip():
                collected.append(body)
            continue
        if collecting:
            if "result" in h or "conclusion" in h:
                break
            # Demote ## to ### so it nests under the paper's ## Methodology
            sub_heading = heading.replace("## ", "### ", 1)
            collected.append(f"{sub_heading}\n\n{body}")
    return "\n\n".join(collected)


def _get_results_body(results_md: str) -> str:
    """Extract the Results section from results.md."""
    sections = _extract_sections(results_md)
    for heading, body in sections:
        if "result" in heading.lower() and "methodol" not in heading.lower():
            return body
    return ""


def _get_results_conclusion(results_md: str) -> str:
    """Extract the Conclusion from results.md."""
    sections = _extract_sections(results_md)
    for heading, body in sections:
        if "conclusion" in heading.lower():
            return body
    return ""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a final research paper from review.md and "
                    "results.md."
    )
    parser.add_argument(
        "--review",
        type=str,
        default="review.md",
        help="Path to the literature review Markdown (default: review.md).",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results.md",
        help="Path to the docking results Markdown (default: results.md).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="final_paper.md",
        help="Output Markdown file (default: final_paper.md).",
    )
    args = parser.parse_args()

    # Validate inputs
    for path, label in [(args.review, "Review"), (args.results, "Results")]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Final Paper Generator")
    print(f"{'='*60}\n")

    # ----- Step 1: Load source documents -----
    print(">> Step 1: Loading source documents …")
    review_md = _read_md(args.review)
    results_md = _read_md(args.results)
    print(f"    review.md  : {len(review_md):,} chars")
    print(f"    results.md : {len(results_md):,} chars")

    # Detect the disease from the review title
    title_match = re.search(r"#.*?:\s*(.+)", review_md)
    disease = title_match.group(1).strip() if title_match else "Disease"

    # ----- Step 2: Extract sections -----
    print("\n>> Step 2: Extracting sections from source documents …")
    review_intro = _get_review_intro_sections(review_md)
    review_drugs = _get_review_drug_sections(review_md)
    review_conclusion = _get_review_conclusion(review_md)
    results_methodology = _get_results_methodology(results_md)
    results_body = _get_results_body(results_md)
    results_conclusion = _get_results_conclusion(results_md)
    review_refs = _extract_references(review_md)

    sections_found = sum(1 for s in [
        review_intro, review_drugs, review_conclusion,
        results_methodology, results_body, results_conclusion,
    ] if s)
    print(f"    Sections extracted: {sections_found}/6")

    # ----- Step 3: Generate abstract -----
    print("\n>> Step 3: Generating abstract …")
    abstract = generate_abstract(review_md, results_md)
    abstract = _strip_leading_header(abstract)

    # ----- Step 4: Polish Introduction & Background -----
    print("\n>> Step 4: Merging Introduction & Background …")
    intro_body = merge_introduction_and_background(review_intro, results_md)
    intro_body = _strip_leading_header(intro_body)

    # ----- Step 5: Polish Methodology -----
    print("\n>> Step 5: Polishing Methodology …")
    # Methodology comes solely from results.md — just clean it up
    methodology = _strip_leading_header(results_methodology)

    # ----- Step 6: Polish Therapeutic Landscape -----
    print("\n>> Step 6: Merging Therapeutic Landscape …")
    drug_landscape = merge_drug_landscape(review_drugs)
    drug_landscape = _strip_leading_header(drug_landscape)

    # ----- Step 7: Polish Docking Results -----
    print("\n>> Step 7: Polishing Docking Results …")
    docking_results = merge_results(results_body)
    docking_results = _strip_leading_header(docking_results)

    # ----- Step 8: Merge Conclusions -----
    print("\n>> Step 8: Merging Conclusions …")
    conclusion = merge_conclusions(review_conclusion, results_conclusion)
    conclusion = _strip_leading_header(conclusion)

    # ----- Step 9: Deduplicate references -----
    print("\n>> Step 9: Building unified reference list …")
    unified_refs = _deduplicate_references(review_refs, "")

    # ----- Step 10: Assemble final paper -----
    print("\n>> Step 10: Assembling final paper …")

    today = datetime.now().strftime("%B %d, %Y")

    final_md = (
        f"# Drug Repurposing for {disease}: A Computational Docking "
        f"and Literature Review Study\n\n"
        f"*{today}*\n\n"
        f"---\n\n"
        f"## Abstract\n\n"
        f"{abstract}\n\n"
        f"---\n\n"
        f"## 1. Introduction and Background\n\n"
        f"{intro_body}\n\n"
        f"---\n\n"
        f"## 2. Therapeutic Landscape\n\n"
        f"{drug_landscape}\n\n"
        f"---\n\n"
        f"## 3. Methodology\n\n"
        f"{methodology}\n\n"
        f"---\n\n"
        f"## 4. Computational Docking Results\n\n"
        f"{docking_results}\n\n"
        f"---\n\n"
        f"## 5. Conclusion\n\n"
        f"{conclusion}\n\n"
        f"---\n\n"
        f"## References\n\n"
        f"{unified_refs}\n"
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(final_md)

    print(f"\n{'='*60}")
    print(f"  Final paper saved to: {args.output}")
    print(f"  Total length: {len(final_md):,} chars")
    print(f"  References: {unified_refs.count('[')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
