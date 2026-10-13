#!/usr/bin/env python3
"""
Disease Literature Review Generator

Uses the PubMed E-utilities API to find papers, the OpenAI API to
synthesize a full literature review (in Markdown) about a specific disease,
targetable proteins, and FDA-approved drugs that may interact with them,
and the Exa API for live web search where the write-up needs fresh
grounding.
"""

import argparse
import json
import os
from dotenv import load_dotenv
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

import defusedxml.ElementTree as ET

from llm import LLMOutputError, query_prompt, query_prompt_structured
from llm_schemas import DrugProteinMap, ProteinList
from resilience import (
    DEFAULT_POLICY,
    call_with_retry,
    request_with_retry,
    response_json,
    run_cli,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
EXA_SEARCH_URL = "https://api.exa.ai/search"

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_MAX_RESULTS = 15  # fetch up to 15 papers per query
PUBMED_TOOL_NAME = "drug-repurposing-pipeline"
# Optional — raises NCBI's rate limit from 3 req/sec to 10 req/sec.
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
# Optional — NCBI asks integrators to self-identify via a contact email as a
# courtesy so they can reach out before blocking an IP, rather than requiring it.
NCBI_CONTACT_EMAIL = os.environ.get("NCBI_CONTACT_EMAIL", "")


# ---------------------------------------------------------------------------
# PubMed helpers
# ---------------------------------------------------------------------------

def _ncbi_params(extra: dict) -> dict:
    """Add the shared tool/email/api_key identification params NCBI asks
    E-utilities integrators to include."""
    params = {**extra, "tool": PUBMED_TOOL_NAME}
    if NCBI_CONTACT_EMAIL:
        params["email"] = NCBI_CONTACT_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def _ncbi_get(url: str, what: str) -> bytes:
    """GET an E-utilities URL, retrying transient failures (NCBI answers 429
    when the 3 req/s courtesy limit is exceeded).  Raises ``ExternalCallError``
    (source ``pubmed``) when it still fails."""
    def attempt() -> bytes:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read()

    # The label omits the URL: it carries the api_key.
    return call_with_retry(attempt, source="pubmed", policy=DEFAULT_POLICY,
                           label=what)


def search_pubmed(query: str, max_results: int = PUBMED_MAX_RESULTS) -> list[dict]:
    """Search PubMed and return a list of paper metadata dicts.

    Raises ``ExternalCallError`` if PubMed cannot be reached, rather than
    returning an empty list that reads as "no papers"."""
    esearch_params = urllib.parse.urlencode(_ncbi_params({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json",
    }))
    esearch_url = f"{PUBMED_ESEARCH_URL}?{esearch_params}"
    print(f"[PubMed] Searching: {query!r}  (max {max_results} results)")

    esearch_data = json.loads(_ncbi_get(esearch_url, "esearch"))

    pmids = esearch_data.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        print("[PubMed] No results.")
        return []

    efetch_params = urllib.parse.urlencode(_ncbi_params({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }))
    efetch_url = f"{PUBMED_EFETCH_URL}?{efetch_params}"

    xml_data = _ncbi_get(efetch_url, "efetch")

    papers = _parse_pubmed_xml(xml_data)
    print(f"[PubMed] Retrieved {len(papers)} papers with usable abstracts "
          f"(of {len(pmids)} matched).")
    return papers


def _parse_pubmed_xml(xml_data: bytes) -> list[dict]:
    """Parse a PubMed EFetch XML response into paper metadata dicts,
    skipping citation-only records that have no abstract."""
    root = ET.fromstring(xml_data)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue
        art = medline.find("Article")
        if art is None:
            continue

        pmid_el = medline.find("PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""

        title_el = art.find("ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Structured abstracts (Background/Methods/Results/...) come as
        # multiple <AbstractText> elements — concatenate all of them, not
        # just the first, or most of the abstract silently disappears.
        abstract_parts = []
        for ab_text in art.findall("./Abstract/AbstractText"):
            text = "".join(ab_text.itertext()).strip()
            if not text:
                continue
            label = ab_text.get("Label")
            abstract_parts.append(f"{label}: {text}" if label else text)
        summary = " ".join(abstract_parts)

        if not title or not summary:
            continue  # citation-only record with no usable abstract

        authors = []
        for author in art.findall("./AuthorList/Author"):
            last = author.find("LastName")
            if last is None or not last.text:
                continue
            fore = author.find("ForeName")
            name = f"{fore.text} {last.text}" if fore is not None and fore.text else last.text
            authors.append(name)

        published = ""
        pub_date = art.find("./Journal/JournalIssue/PubDate")
        if pub_date is not None:
            year_el = pub_date.find("Year")
            if year_el is not None and year_el.text:
                published = year_el.text
            else:
                # Some records only have a free-text MedlineDate (e.g.
                # "2023 Jan-Feb") instead of a clean Year element.
                medline_date_el = pub_date.find("MedlineDate")
                if medline_date_el is not None and medline_date_el.text:
                    published = medline_date_el.text[:4]

        papers.append({
            "title": title,
            "summary": summary,
            "authors": authors,
            "published": published,
            "pmid": pmid,
            "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    return papers


def format_papers_for_prompt(papers: list[dict]) -> str:
    """Format paper metadata into a readable block for the LLM prompt."""
    lines = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors += " et al."
        lines.append(
            f"[{i}] {p['title']}\n"
            f"    Authors: {authors}\n"
            f"    Published: {p['published']}\n"
            f"    PMID: {p['pmid']}\n"
            f"    Abstract: {p['summary'][:500]}...\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exa helpers
# ---------------------------------------------------------------------------

def web_search_exa(query: str, num_results: int = 12) -> list[dict]:
    """Search the live web via Exa and return a list of
    {title, url, text} result dicts."""
    if not EXA_API_KEY:
        print("ERROR: EXA_API_KEY environment variable is not set.",
              file=sys.stderr)
        sys.exit(1)

    headers = {
        "x-api-key": EXA_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "type": "auto",
        "numResults": num_results,
        "contents": {"text": {"maxCharacters": 1500}},
    }

    print(f"[Exa] Searching: {query!r}")
    resp = request_with_retry("post", EXA_SEARCH_URL, source="exa",
                              json=payload, headers=headers, timeout=30)
    data = response_json(resp, "exa")
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "text": r.get("text", ""),
        }
        for r in data.get("results", [])
    ]
    print(f"[Exa] Retrieved {len(results)} results.")
    return results


def format_search_results_for_prompt(results: list[dict]) -> str:
    """Format Exa search results into a numbered, citable block for an LLM
    prompt."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] {r['title']}\n"
            f"    URL: {r['url']}\n"
            f"    {r['text'][:1000]}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_disease_query(disease: str) -> str:
    """Build a PubMed search query for disease biology / treatment papers.

    No field tags needed — PubMed's automatic term mapping already expands
    quoted phrases across title/abstract/MeSH terms by default."""
    return (
        f'("{disease}") AND (treatment OR therapy OR target OR protein '
        f"OR pathway OR molecular OR biomarker)"
    )


def build_drug_query(disease: str, proteins: list[str]) -> str:
    """Build a PubMed search query for FDA-approved drugs binding target
    proteins."""
    protein_clause = " OR ".join(f'"{p}"' for p in proteins[:6])
    return (
        f'("{disease}") AND (FDA OR "approved drug" OR inhibitor '
        f"OR therapeutic) AND ({protein_clause})"
    )


def build_repurposing_query(proteins: list[str]) -> str:
    """Build a broader PubMed query for drug repurposing, molecular docking,
    and off-target interaction studies for the given protein targets —
    intentionally NOT restricted to the disease."""
    protein_clause = " OR ".join(f'"{p}"' for p in proteins[:6])
    return (
        f'({protein_clause}) AND '
        f'("drug repurposing" OR "drug repositioning" OR "molecular docking" '
        f'OR "virtual screening" OR "off-target" OR "binding affinity" '
        f'OR "structure-activity" OR "polypharmacology")'
    )


def extract_proteins_from_text(text: str) -> list[str]:
    """Ask OpenAI to extract a concise list of protein targets from the
    first-pass review."""
    try:
        result = query_prompt_structured(
            "extraction/protein_list", ProteinList, text=text
        )
    except LLMOutputError as e:
        # Empty list -> the caller retries via extract_proteins_fallback.
        print(f"[OpenAI] Protein extraction unusable ({e.kind}): {e}",
              file=sys.stderr)
        return []
    return _clean_names(result.proteins)


def _clean_names(names: list[str]) -> list[str]:
    """Trim, drop blanks, and de-duplicate while keeping order."""
    return list(dict.fromkeys(n.strip() for n in names if n.strip()))


def extract_proteins_fallback(disease: str) -> list[str]:
    """Last-resort protein extraction: if pulling targets out of the review
    text failed entirely, search the web for known protein targets of this
    specific disease and extract from those results — grounded in real
    sources rather than asking the LLM to recall from parametric memory
    alone, and not a silently substituted, unrelated hardcoded list."""
    search_results = web_search_exa(
        f"protein targets implicated in {disease} pathophysiology and treatment",
        num_results=8,
    )
    if not search_results:
        return []
    search_results_text = format_search_results_for_prompt(search_results)

    try:
        result = query_prompt_structured(
            "extraction/protein_list_fallback", ProteinList,
            disease=disease, search_results=search_results_text,
        )
    except LLMOutputError as e:
        print(f"[OpenAI] Protein fallback unusable ({e.kind}): {e}",
              file=sys.stderr)
        return []
    return _clean_names(result.proteins)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown literature review about a disease."
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="The specific disease to review (e.g., 'pancreatic cancer', 'Alzheimer's disease').",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="review.md",
        help="Output Markdown file (default: review.md).",
    )
    args = parser.parse_args()

    disease = args.prompt.strip()
    output_file = args.output

    print(f"\n{'='*60}")
    print(f"  Literature Review Generator  —  {disease}")
    print(f"{'='*60}\n")

    # ----- Step 1: Search PubMed for papers on this disease -----
    print(">> Step 1: Searching PubMed for papers on", disease)
    disease_query = build_disease_query(disease)
    disease_papers = search_pubmed(disease_query, max_results=PUBMED_MAX_RESULTS)

    if not disease_papers:
        print("No papers found on PubMed. Try a different disease.",
              file=sys.stderr)
        sys.exit(1)

    disease_papers_text = format_papers_for_prompt(disease_papers)

    # ----- Step 2: Generate first-pass review (disease + proteins) -----
    print("\n>> Step 2: Generating literature review via OpenAI …")
    first_review = query_prompt(
        "literature/review_first_pass_cli",
        disease=disease, papers=disease_papers_text,
    )

    # ----- Step 3: Extract protein targets -----
    print("\n>> Step 3: Extracting protein targets …")
    proteins = extract_proteins_from_text(first_review)

    if not proteins:
        print("    Extraction from review text yielded nothing, retrying "
              "with a direct query …")
        proteins = extract_proteins_fallback(disease)

    if not proteins:
        print(f"ERROR: Could not identify any protein targets for "
              f"{disease!r}.", file=sys.stderr)
        sys.exit(1)

    print(f"    Identified proteins: {proteins}")

    # ----- Step 4a: Search PubMed for mainstream drugs -----
    # Brief pause to respect NCBI's courtesy rate limit (≤3 req/sec without
    # an API key; each search_pubmed() call itself makes 2 requests).
    time.sleep(0.5)
    print("\n>> Step 4a: Searching PubMed for mainstream drugs targeting these proteins …")
    drug_query = build_drug_query(disease, proteins)
    drug_papers = search_pubmed(drug_query, max_results=PUBMED_MAX_RESULTS)

    # ----- Step 4b: Broader repurposing / docking / off-target search -----
    time.sleep(0.5)
    print("\n>> Step 4b: Searching PubMed for drug-repurposing & off-target interaction studies …")
    repurpose_query = build_repurposing_query(proteins)
    repurpose_papers = search_pubmed(repurpose_query, max_results=PUBMED_MAX_RESULTS)

    # Deduplicate by PMID, keeping order
    seen_ids = {p["pmid"] for p in drug_papers}
    for rp in repurpose_papers:
        if rp["pmid"] not in seen_ids:
            drug_papers.append(rp)
            seen_ids.add(rp["pmid"])

    drug_papers_text = format_papers_for_prompt(drug_papers) if drug_papers else "(No papers found.)"

    # ----- Step 5a: Generate drug section -----
    print("\n>> Step 5a: Generating mainstream drug analysis via OpenAI …")

    # Reference numbering continues from the disease papers
    offset = len(disease_papers)
    re_numbered_drug_text = drug_papers_text
    if drug_papers:
        for i, _ in enumerate(drug_papers, 1):
            re_numbered_drug_text = re_numbered_drug_text.replace(
                f"[{i}]", f"[{i + offset}]", 1
            )

    drug_review = query_prompt(
        "literature/drug_analysis_cli",
        disease=disease,
        proteins=", ".join(proteins),
        first_ref_number=offset + 1,
        papers=re_numbered_drug_text,
    )

    # ----- Step 5b: Speculative / repurposing drug discovery via Exa + OpenAI -----
    print("\n>> Step 5b: Searching for non-obvious repurposable FDA drugs via Exa + OpenAI …")
    repurposing_review = _discover_repurposing_candidates(disease, proteins)

    # ----- Step 6: Assemble final document -----
    print("\n>> Step 6: Assembling final Markdown document …")

    # Build combined references
    all_papers = disease_papers + drug_papers
    references_block = _build_references(all_papers)

    today = datetime.now().strftime("%B %d, %Y")
    final_md = (
        f"# Literature Review: {disease.title()}\n\n"
        f"*Auto-generated on {today} using PubMed, OpenAI, and Exa.*\n\n"
        f"---\n\n"
        f"{first_review}\n\n"
        f"---\n\n"
        f"{drug_review}\n\n"
        f"---\n\n"
        f"{repurposing_review}\n\n"
        f"---\n\n"
        f"## Consolidated References\n\n{references_block}\n"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_md)

    # ----- Step 7: Generate review.json (drug → protein mapping) -----
    print("\n>> Step 7: Generating review.json (drug–protein mapping) …")
    combined_drug_text = drug_review + "\n\n" + repurposing_review
    drug_protein_json = _extract_drug_protein_map(
        combined_drug_text, disease, proteins
    )
    with open("review.json", "w", encoding="utf-8") as f:
        json.dump(drug_protein_json, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Review saved to: {output_file}")
    print(f"  Drug map saved to: review.json")
    print(f"  Total papers cited: {len(all_papers)}")
    print(f"{'='*60}\n")


def _discover_repurposing_candidates(
    disease: str, proteins: list[str]
) -> str:
    """Use Exa web search to find FDA-approved drugs from ANY therapeutic
    area with structural, mechanistic, or computational evidence of
    interacting with the target proteins — even if they are not currently
    studied for this disease — then have OpenAI write up the findings.

    Perplexity used to do its own web search here. OpenAI's chat completions
    have no live browsing, so we search Exa first and ground the write-up
    in those results instead."""
    search_query = (
        f"FDA-approved drugs from cardiology, psychiatry, infectious disease, "
        f"metabolic, or autoimmune therapeutic areas with known or "
        f"computationally predicted binding, off-target activity, or drug "
        f"repurposing potential against {', '.join(proteins)}"
    )
    search_results = web_search_exa(search_query, num_results=12)
    search_results_text = (
        format_search_results_for_prompt(search_results)
        if search_results else "(No web search results found.)"
    )

    return query_prompt(
        "literature/repurposing_candidates",
        disease=disease,
        proteins=", ".join(proteins),
        search_results=search_results_text,
    )


def _extract_drug_protein_map(drug_review_text: str, disease: str, proteins: list[str]) -> dict:
    """Ask OpenAI for the drugs mentioned in the review and the proteins
    each may target, as a schema-constrained ``DrugProteinMap``.

    If the reply is unusable (refusal, truncation, ...) the map comes back
    with no drugs and an ``llm_error`` field recording why, so the gap is
    visible in review.json rather than silent.
    """
    data: dict = {"drugs": []}
    try:
        drug_map = query_prompt_structured(
            "extraction/drug_protein_map", DrugProteinMap,
            disease=disease,
            proteins=", ".join(proteins),
            review_text=drug_review_text,
        )
        data = drug_map.model_dump()
    except LLMOutputError as e:
        print(f"[OpenAI] Drug-protein map unusable ({e.kind}): {e}",
              file=sys.stderr)
        data["llm_error"] = {"kind": e.kind, "message": str(e)}

    data["disease"] = disease
    data["protein_targets"] = proteins
    return data


def _build_references(papers: list[dict]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors += " et al."
        lines.append(
            f"[{i}] {authors}. \"{p['title']}.\" "
            f"PMID:{p['pmid']}, {p['published']}. {p['link']}"
        )
    return "\n\n".join(lines)


if __name__ == "__main__":
    run_cli(main)
