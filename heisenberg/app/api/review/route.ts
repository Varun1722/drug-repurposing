import { queryOpenAI, parseJsonResponse } from "@/lib/openai"
import { webSearch, formatSearchResultsForPrompt } from "@/lib/exa"
import { searchCompoundsForTarget } from "@/lib/pubchem"
import type { ReviewDrug } from "@/lib/types"
import { shortenDrugName } from "@/lib/types"

export const maxDuration = 300 // 5 min — lit review makes several OpenAI + Exa calls

// ---------- PubMed helpers ----------

interface PubmedPaper {
  title: string
  summary: string
  authors: string[]
  published: string
  pmid: string
  link: string
}

const PUBMED_TOOL_NAME = "drug-repurposing-pipeline"

function ncbiParams(extra: Record<string, string>): URLSearchParams {
  const params = new URLSearchParams({ ...extra, tool: PUBMED_TOOL_NAME })
  // Optional courtesy/rate-limit params — NCBI asks integrators to
  // self-identify via email, and api_key raises the limit 3/sec -> 10/sec.
  if (process.env.NCBI_CONTACT_EMAIL) {
    params.set("email", process.env.NCBI_CONTACT_EMAIL)
  }
  if (process.env.NCBI_API_KEY) {
    params.set("api_key", process.env.NCBI_API_KEY)
  }
  return params
}

async function searchPubmed(
  query: string,
  maxResults: number = 15
): Promise<PubmedPaper[]> {
  const esearchParams = ncbiParams({
    db: "pubmed",
    term: query,
    retmax: String(maxResults),
    sort: "relevance",
    retmode: "json",
  })
  const esearchRes = await fetch(
    `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?${esearchParams.toString()}`,
    { signal: AbortSignal.timeout(30000) }
  )
  if (!esearchRes.ok) return []
  const esearchData = await esearchRes.json()
  const pmids: string[] = esearchData?.esearchresult?.idlist || []
  if (pmids.length === 0) return []

  const efetchParams = ncbiParams({
    db: "pubmed",
    id: pmids.join(","),
    retmode: "xml",
  })
  const efetchRes = await fetch(
    `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?${efetchParams.toString()}`,
    { signal: AbortSignal.timeout(30000) }
  )
  if (!efetchRes.ok) return []

  const xml = await efetchRes.text()
  return parsePubmedXml(xml)
}

function stripTags(text: string): string {
  const withoutTags = text.replace(/<[^>]+>/g, "")
  // A real XML parser decodes entities automatically; this regex-based
  // parser doesn't, so numeric/named entities (PubMed abstracts are full
  // of them, e.g. "&#xa0;") need decoding by hand.
  return withoutTags
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => String.fromCodePoint(parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec) => String.fromCodePoint(parseInt(dec, 10)))
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
}

function parsePubmedXml(xml: string): PubmedPaper[] {
  const papers: PubmedPaper[] = []
  const articleRegex = /<PubmedArticle>([\s\S]*?)<\/PubmedArticle>/g
  let match
  while ((match = articleRegex.exec(xml)) !== null) {
    const article = match[1]

    const pmidMatch = article.match(/<PMID[^>]*>(.*?)<\/PMID>/)
    const pmid = pmidMatch ? pmidMatch[1].trim() : ""

    const titleMatch = article.match(/<ArticleTitle[^>]*>([\s\S]*?)<\/ArticleTitle>/)
    const title = titleMatch ? stripTags(titleMatch[1]).trim() : ""

    // Structured abstracts (Background/Methods/Results/...) come as
    // multiple <AbstractText> elements — concatenate all of them, not
    // just the first, or most of the abstract silently disappears.
    const abstractParts: string[] = []
    const abstractTextRegex = /<AbstractText([^>]*)>([\s\S]*?)<\/AbstractText>/g
    let abMatch
    while ((abMatch = abstractTextRegex.exec(article)) !== null) {
      const text = stripTags(abMatch[2]).trim()
      if (!text) continue
      const labelMatch = abMatch[1].match(/Label="([^"]*)"/)
      abstractParts.push(labelMatch ? `${labelMatch[1]}: ${text}` : text)
    }
    const summary = abstractParts.join(" ")

    if (!title || !summary) continue // citation-only record with no usable abstract

    const authors: string[] = []
    const authorRegex = /<Author[^>]*>([\s\S]*?)<\/Author>/g
    let authorMatch
    while ((authorMatch = authorRegex.exec(article)) !== null) {
      const authorBlock = authorMatch[1]
      const lastMatch = authorBlock.match(/<LastName>(.*?)<\/LastName>/)
      if (!lastMatch) continue
      const foreMatch = authorBlock.match(/<ForeName>(.*?)<\/ForeName>/)
      authors.push(foreMatch ? `${foreMatch[1]} ${lastMatch[1]}` : lastMatch[1])
    }

    let published = ""
    const pubDateMatch = article.match(/<PubDate>([\s\S]*?)<\/PubDate>/)
    if (pubDateMatch) {
      const yearMatch = pubDateMatch[1].match(/<Year>(.*?)<\/Year>/)
      if (yearMatch) {
        published = yearMatch[1]
      } else {
        // Some records only have a free-text MedlineDate (e.g.
        // "2023 Jan-Feb") instead of a clean Year element.
        const medlineDateMatch = pubDateMatch[1].match(/<MedlineDate>(.*?)<\/MedlineDate>/)
        if (medlineDateMatch) published = medlineDateMatch[1].slice(0, 4)
      }
    }

    papers.push({
      title,
      summary,
      authors,
      published,
      pmid,
      link: `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`,
    })
  }
  return papers
}

function formatPapersForPrompt(papers: PubmedPaper[]): string {
  return papers
    .map((p, i) => {
      const authors =
        p.authors.slice(0, 3).join(", ") +
        (p.authors.length > 3 ? " et al." : "")
      return (
        `[${i + 1}] ${p.title}\n` +
        `    Authors: ${authors}\n` +
        `    Published: ${p.published}\n` +
        `    PMID: ${p.pmid}\n` +
        `    Abstract: ${p.summary.slice(0, 500)}...\n`
      )
    })
    .join("\n")
}

// ---------- Query builders ----------

function buildDiseaseQuery(disease: string): string {
  return `("${disease}") AND (treatment OR therapy OR target OR protein OR pathway OR molecular OR biomarker)`
}

function buildDrugQuery(disease: string, proteins: string[]): string {
  const proteinClause = proteins
    .slice(0, 6)
    .map((p) => `"${p}"`)
    .join(" OR ")
  return `("${disease}") AND (FDA OR "approved drug" OR inhibitor OR therapeutic) AND (${proteinClause})`
}

function buildRepurposingQuery(proteins: string[]): string {
  const proteinClause = proteins
    .slice(0, 6)
    .map((p) => `"${p}"`)
    .join(" OR ")
  return `(${proteinClause}) AND ("drug repurposing" OR "drug repositioning" OR "molecular docking" OR "virtual screening" OR "off-target" OR "binding affinity" OR "structure-activity" OR "polypharmacology")`
}

// ---------- Protein extraction ----------

async function extractProteinsFromText(text: string): Promise<string[]> {
  const system =
    "You are a biomedical research assistant. " +
    "Extract a concise list of protein targets from the following text. " +
    "Return ONLY a JSON array of short protein names/symbols, " +
    'e.g. ["EGFR", "HER2", "BRAF"]. No explanation.'
  const raw = await queryOpenAI(system, text)

  const match = raw.match(/\[[\s\S]*?\]/)
  if (match) {
    try {
      const parsed = JSON.parse(match[0])
      if (Array.isArray(parsed)) return parsed.map(String)
    } catch {
      // fallback
    }
  }
  return raw
    .split(",")
    .map((t) => t.trim().replace(/['"[\]]/g, ""))
    .filter(Boolean)
}

/**
 * Last-resort protein extraction: if pulling targets out of the review text
 * failed entirely, search the web for known protein targets of this
 * specific disease and extract from those results — grounded in real
 * sources rather than asking the LLM to recall from parametric memory
 * alone, and not a silently substituted, unrelated hardcoded list.
 */
async function extractProteinsFallback(disease: string): Promise<string[]> {
  const searchResults = await webSearch(
    `protein targets implicated in ${disease} pathophysiology and treatment`,
    8
  )
  if (searchResults.length === 0) return []
  const searchResultsText = formatSearchResultsForPrompt(searchResults)

  const system =
    "You are a biomedical research assistant. Based ONLY on the search " +
    "results provided, return ONLY a JSON array of 3-5 well-known protein " +
    "target names/symbols implicated in the disease given. No explanation."
  const user =
    `Disease: ${disease}\n\nSearch results:\n${searchResultsText}`
  const raw = await queryOpenAI(system, user)

  const match = raw.match(/\[[\s\S]*?\]/)
  if (match) {
    try {
      const parsed = JSON.parse(match[0])
      if (Array.isArray(parsed) && parsed.length > 0) return parsed.map(String)
    } catch {
      // fall through
    }
  }
  return []
}

// ---------- Repurposing candidates ----------

async function discoverRepurposingCandidates(
  disease: string,
  proteins: string[]
): Promise<string> {
  // Perplexity used to do its own web search here. OpenAI's chat completions
  // have no live browsing, so search Exa first and ground the write-up in
  // those results instead.
  const searchQuery =
    `FDA-approved drugs from cardiology, psychiatry, infectious disease, ` +
    `metabolic, or autoimmune therapeutic areas with known or computationally ` +
    `predicted binding, off-target activity, or drug repurposing potential ` +
    `against ${proteins.join(", ")}`
  const searchResults = await webSearch(searchQuery, 12)
  const searchResultsText = searchResults.length
    ? formatSearchResultsForPrompt(searchResults)
    : "(No web search results found.)"

  const system =
    "You are a computational pharmacology expert specialising in drug " +
    "repurposing and polypharmacology. Write a detailed Markdown section " +
    "for a literature review, grounded ONLY in the web search results " +
    "provided below. Be exhaustive: include drugs from cardiology, " +
    "psychiatry, infectious disease, metabolic disorders, autoimmune " +
    "conditions, and any other field. Cite sources using their [N] index " +
    "from the search results below."
  const user =
    `The following proteins have been identified as therapeutic targets ` +
    `in **${disease}**: ${proteins.join(", ")}.\n\n` +
    `Web search results on repurposing candidates:\n${searchResultsText}\n\n` +
    `Using ONLY the evidence above, identify FDA-approved drugs from ANY ` +
    `therapeutic area — not just oncology — that have known or ` +
    `computationally predicted interactions with these proteins. Consider:\n` +
    `- Molecular docking studies showing binding affinity\n` +
    `- Shared binding-site homology with known inhibitors\n` +
    `- Off-target activity reported in pharmacovigilance data\n` +
    `- Structural similarity (Tanimoto ≥ 0.5) to known ligands\n` +
    `- Drug-gene interaction databases (DGIdb, DrugBank, STITCH)\n` +
    `- Repurposing screens or virtual screening hits\n\n` +
    `Write sections in Markdown with at least 10 drugs, citing sources by [N].`
  return queryOpenAI(system, user)
}

// ---------- Drug-protein map extraction ----------

async function extractDrugProteinMap(
  drugReviewText: string,
  disease: string,
  proteins: string[]
): Promise<{ drugs: ReviewDrug[]; disease: string; protein_targets: string[] }> {
  const system =
    "You are a biomedical data-extraction assistant. " +
    'Return ONLY valid JSON with no extra text, no markdown fences. ' +
    'The JSON must be an object with a top-level key "drugs" whose value ' +
    "is an array of objects. Each object has: " +
    '"drug" (string), "proteins" (list of target protein symbols), ' +
    '"mechanism" (string), "fda_status" (string), ' +
    '"category" ("mainstream" or "repurposing_candidate"), ' +
    '"trial_status_for_targets" (string: describe if this drug has been tested ' +
    'in clinical trials specifically for the listed protein targets, e.g. ' +
    '"Phase 2 trial for KRAS G12C inhibition in NSCLC" or "No known trials for these targets").'
  const user =
    `From the following literature review sections about ${disease}, ` +
    `extract EVERY drug mentioned — both mainstream drugs for this ` +
    `disease AND repurposing candidates from other therapeutic areas — ` +
    `and which of these proteins each may ` +
    `bind to or react with: ${proteins.join(", ")}.\n\n` +
    `Mark drugs that are commonly used for ${disease} as ` +
    `"mainstream". Mark drugs from other therapeutic areas or ` +
    `speculative candidates as "repurposing_candidate".\n\n` +
    `Text:\n${drugReviewText}`
  const raw = await queryOpenAI(system, user)
  let data = parseJsonResponse(raw) as Record<string, unknown>

  if ("_parse_error" in data) {
    const fixSystem =
      "You are a JSON repair assistant. The user will give you malformed " +
      "JSON. Return ONLY the corrected, valid JSON. No explanation."
    const fixedRaw = await queryOpenAI(fixSystem, raw)
    data = parseJsonResponse(fixedRaw) as Record<string, unknown>
  }

  if (!("drugs" in data)) {
    data = { drugs: Array.isArray(data) ? data : [] }
  }

  const rawDrugs = (data.drugs as Record<string, unknown>[]) || []
  const drugs: ReviewDrug[] = rawDrugs.map((d) => ({
    drug: shortenDrugName((d.drug as string) || "", null, 40) || "Unknown",
    proteins: (d.proteins as string[]) || [],
    mechanism: (d.mechanism as string) || "",
    fdaStatus: (d.fdaStatus as string) || (d.fda_status as string) || "",
    category: (d.category as string) || "",
    trialStatusForTargets: (d.trialStatusForTargets as string) || (d.trial_status_for_targets as string) || "",
  }))

  return {
    drugs,
    disease,
    protein_targets: proteins,
  }
}

// ---------- Route handler (SSE streaming) ----------

export async function POST(req: Request) {
  try {
    const { disease } = await req.json()
    if (!disease) {
      return new Response(
        JSON.stringify({ error: "disease is required" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      )
    }

    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      async start(controller) {
        function send(data: object) {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify(data)}\n\n`)
          )
        }

        try {
          // Step 1: Search PubMed
          send({ type: "progress", message: "Searching PubMed for relevant papers..." })
          const diseasePapers = await searchPubmed(buildDiseaseQuery(disease))
          const diseasePapersText = formatPapersForPrompt(diseasePapers)
          send({ type: "progress", message: `Found ${diseasePapers.length} papers on PubMed` })

          // Step 2: Generate literature review via OpenAI
          send({ type: "progress", message: "Generating literature review with OpenAI..." })
          const systemReview =
            "You are an expert biomedical researcher. Write a detailed, scholarly " +
            "literature review in Markdown format. Use inline citations like " +
            "[1], [2], etc., referencing the papers provided."
          const userReview =
            `Using the following PubMed papers, write a comprehensive literature ` +
            `review about **${disease}**.\n\n` +
            `Include:\n` +
            `1. Introduction (epidemiology, significance)\n` +
            `2. Molecular and genetic landscape\n` +
            `3. **Key protein targets** for treatment\n` +
            `4. Current therapeutic strategies\n` +
            `5. References\n\n` +
            `Papers:\n${diseasePapersText}`
          const firstReview = await queryOpenAI(systemReview, userReview)

          // Step 3: Extract protein targets
          send({ type: "progress", message: "Extracting protein targets..." })
          let proteins = await extractProteinsFromText(firstReview)
          if (proteins.length === 0) {
            send({
              type: "progress",
              message: "Extraction from review text yielded nothing, retrying with a direct query...",
            })
            proteins = await extractProteinsFallback(disease)
          }
          if (proteins.length === 0) {
            throw new Error(
              `Could not identify any protein targets for "${disease}".`
            )
          }

          // Emit proteins immediately so frontend can populate
          send({ type: "proteins", proteins })

          // Step 4: Search PubMed for drug papers
          send({ type: "progress", message: "Searching for drug interaction papers..." })
          const drugPapers = await searchPubmed(
            buildDrugQuery(disease, proteins)
          )
          const repurposePapers = await searchPubmed(
            buildRepurposingQuery(proteins)
          )

          // Dedup
          const seenIds = new Set(drugPapers.map((p) => p.pmid))
          for (const rp of repurposePapers) {
            if (!seenIds.has(rp.pmid)) {
              drugPapers.push(rp)
              seenIds.add(rp.pmid)
            }
          }

          send({ type: "progress", message: `Found ${drugPapers.length} drug-related papers` })

          const drugPapersText = drugPapers.length
            ? formatPapersForPrompt(drugPapers)
            : "(No papers found.)"

          // Step 5: Generate drug analysis
          send({ type: "progress", message: "Analyzing FDA-approved drugs and candidates..." })
          const systemDrugs =
            "You are an expert pharmacology researcher. Write a detailed Markdown " +
            "section about FDA-approved drugs and repurposing candidates. " +
            "Be precise about drug names, mechanisms, and protein interactions."
          const userDrugs =
            `Continue the review on **${disease}**.\n\n` +
            `Protein targets: ${proteins.join(", ")}.\n\n` +
            `Write sections on:\n` +
            `1. FDA-Approved Drugs and Candidate Compounds per target\n` +
            `2. Drug-Protein Interaction Summary Table\n` +
            `3. Conclusion and Future Directions\n\n` +
            `Papers:\n${drugPapersText}`
          const drugReview = await queryOpenAI(systemDrugs, userDrugs)

          // Step 6: Repurposing candidates
          send({ type: "progress", message: "Discovering repurposing candidates across therapeutic areas..." })
          const repurposingReview = await discoverRepurposingCandidates(
            disease,
            proteins
          )

          // Step 7: Assemble review markdown
          const reviewMd =
            `# Literature Review: ${disease}\n\n` +
            `---\n\n${firstReview}\n\n---\n\n${drugReview}\n\n---\n\n${repurposingReview}\n\n`

          // Emit review text
          send({
            type: "review",
            reviewMd,
            papersAnalyzed: diseasePapers.length + drugPapers.length,
          })

          // Step 8: Extract drug-protein map
          send({ type: "progress", message: "Extracting drug-protein interaction map..." })
          const combined = drugReview + "\n\n" + repurposingReview
          const drugMap = await extractDrugProteinMap(combined, disease, proteins)

          // Step 9: Search PubChem for bioactive compounds targeting each protein
          send({ type: "progress", message: "Searching PubChem for bioactive compounds..." })
          const existingDrugNames = new Set(
            drugMap.drugs.map((d) => d.drug.toLowerCase())
          )
          const bioactiveDrugs: ReviewDrug[] = []

          const bioactiveResults = await Promise.allSettled(
            proteins.map((protein) => searchCompoundsForTarget(protein, 15))
          )

          for (let i = 0; i < proteins.length; i++) {
            const r = bioactiveResults[i]
            if (r.status !== "fulfilled") continue
            const protein = proteins[i]
            for (const c of r.value) {
              const name = shortenDrugName(c.iupacName, c.cid)
              if (existingDrugNames.has(name.toLowerCase())) continue
              existingDrugNames.add(name.toLowerCase())
              bioactiveDrugs.push({
                drug: name,
                proteins: [protein],
                mechanism: "Bioactive compound — PubChem target search",
                fdaStatus: "Unknown — requires verification",
                category: "repurposing_candidate",
              })
            }
          }

          if (bioactiveDrugs.length > 0) {
            send({
              type: "progress",
              message: `Found ${bioactiveDrugs.length} additional bioactive compounds from PubChem`,
            })
          }

          const allDrugs = [...drugMap.drugs, ...bioactiveDrugs]

          // Emit drugs
          send({ type: "drugs", drugs: allDrugs })

          // Final complete event
          send({
            type: "complete",
            proteins,
            drugs: allDrugs,
            reviewMd,
            papersAnalyzed: diseasePapers.length + drugPapers.length,
          })
        } catch (err) {
          send({
            type: "error",
            message: err instanceof Error ? err.message : "Unknown error",
          })
        }

        controller.close()
      },
    })

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    })
  } catch (error) {
    return new Response(
      JSON.stringify({
        error: error instanceof Error ? error.message : "Unknown error",
      }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    )
  }
}
