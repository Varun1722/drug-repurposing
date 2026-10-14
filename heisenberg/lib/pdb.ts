// RCSB PDB API client. Behavioural reference: legacy/python-pipeline/agent2.py
//
// "Nothing found" (RCSB's 204, or a 404) is a normal result: [] / empty
// metadata / null. A request that fails even after retries throws
// ExternalCallError, so an outage is not mistaken for "no structure exists".

import { fetchWithRetry, responseJson } from "./retry"

const PDB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
const PDB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"
const PDB_DOWNLOAD_URL = "https://files.rcsb.org/download"

export interface PdbCandidate {
  pdbId: string
  score: number
  title?: string
  resolution?: number
  hasLigands?: boolean
}

/**
 * Search RCSB PDB for protein structures.
 * Reference: legacy/python-pipeline/agent2.py search_pdb()
 */
export async function searchPdb(
  proteinName: string,
  maxResults: number = 10
): Promise<PdbCandidate[]> {
  const query = {
    query: {
      type: "group",
      logical_operator: "and",
      nodes: [
        {
          type: "terminal",
          service: "full_text",
          parameters: { value: proteinName },
        },
        {
          type: "terminal",
          service: "text",
          parameters: {
            attribute:
              "rcsb_entity_source_organism.ncbi_scientific_name",
            operator: "exact_match",
            value: "Homo sapiens",
          },
        },
        {
          type: "terminal",
          service: "text",
          parameters: {
            attribute: "exptl.method",
            operator: "exact_match",
            value: "X-RAY DIFFRACTION",
          },
        },
        {
          type: "terminal",
          service: "text",
          parameters: {
            attribute: "rcsb_entry_info.resolution_combined",
            operator: "less",
            value: 3.0,
          },
        },
      ],
    },
    return_type: "entry",
    request_options: {
      sort: [
        {
          sort_by: "rcsb_entry_info.resolution_combined",
          direction: "asc",
        },
      ],
      results_content_type: ["experimental"],
      paginate: { start: 0, rows: maxResults },
    },
  }

  const res = await fetchWithRetry(
    PDB_SEARCH_URL,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(query),
    },
    // 204 No Content is how RCSB says "no matches".
    { source: "rcsb", allowStatus: [204], timeoutMs: 30_000 }
  )
  if (res.status === 204) return []

  const data = await responseJson<{ result_set?: { identifier: string; score: number }[] }>(res, "rcsb")
  return (data.result_set || []).map((r) => ({
    pdbId: r.identifier,
    score: r.score,
  }))
}

/**
 * Get PDB entry metadata (resolution, ligand info).
 */
export async function getPdbMetadata(
  pdbId: string
): Promise<{ resolution?: number; hasLigands: boolean; title: string }> {
  const res = await fetchWithRetry(
    `${PDB_DATA_URL}/${pdbId}`,
    {},
    { source: "rcsb", allowStatus: [404], timeoutMs: 15_000 }
  )
  // An entry RCSB does not know: empty metadata. A failed request threw above.
  if (res.status === 404) return { hasLigands: false, title: "" }

  const data = await responseJson<Record<string, any>>(res, "rcsb")
  const resolution = data?.rcsb_entry_info?.resolution_combined?.[0] ?? undefined
  const title = data?.struct?.title ?? ""
  // Check for non-polymer entities (ligands)
  const hasLigands = (data?.rcsb_entry_info?.nonpolymer_entity_count ?? 0) > 0

  return { resolution, hasLigands, title }
}

/**
 * Pick the best PDB structure: prefer those with ligands, then best resolution.
 * Reference: legacy/python-pipeline/agent2.py pick_best_structure(). A candidate whose metadata cannot be
 * fetched is skipped (it cannot be judged); if none can be, the first failure
 * is thrown.
 */
export async function pickBestStructure(
  candidates: PdbCandidate[]
): Promise<PdbCandidate | null> {
  if (candidates.length === 0) return null

  // Enrich with metadata in parallel (check first 5 candidates)
  const settled = await Promise.allSettled(
    candidates.slice(0, 5).map(async (c) => {
      const meta = await getPdbMetadata(c.pdbId)
      return {
        ...c,
        resolution: meta.resolution,
        hasLigands: meta.hasLigands,
        title: meta.title,
      }
    })
  )
  const enriched = settled.flatMap((r) => (r.status === "fulfilled" ? [r.value] : []))
  if (enriched.length === 0) {
    const first = settled.find((r): r is PromiseRejectedResult => r.status === "rejected")
    throw first?.reason
  }

  // Sort: ligands first, then by resolution (lower is better)
  enriched.sort((a, b) => {
    if (a.hasLigands && !b.hasLigands) return -1
    if (!a.hasLigands && b.hasLigands) return 1
    return (a.resolution ?? 999) - (b.resolution ?? 999)
  })

  return enriched[0]
}

/**
 * Download PDB file and return as base64 string (no disk write).
 * Reference: legacy/python-pipeline/agent2.py download_pdb(), adapted for serverless.
 * Unlike the reference it does not convert .cif to .pdb (needs gemmi; a Python-service job).
 * Returns null when RCSB has neither a .pdb nor a .cif file for the id; a
 * failed download throws ExternalCallError.
 */
export async function downloadPdb(pdbId: string): Promise<string | null> {
  const upperPdbId = pdbId.toUpperCase()

  // Try .pdb format first, then .cif
  for (const ext of ["pdb", "cif"]) {
    const url = `${PDB_DOWNLOAD_URL}/${upperPdbId}.${ext}`
    const res = await fetchWithRetry(url, {}, { source: "rcsb", allowStatus: [404], timeoutMs: 60_000 })
    if (res.status === 404) continue

    const buffer = await res.arrayBuffer()
    // Convert to base64
    const bytes = new Uint8Array(buffer)
    let binary = ""
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i])
    }
    return btoa(binary)
  }

  return null
}
