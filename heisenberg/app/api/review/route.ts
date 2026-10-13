import { queryPrompt, queryPromptStructured, LLMOutputError } from "@/lib/openai"
import {
  ProteinListSchema,
  DrugProteinMapSchema,
  type DrugProteinMap,
} from "@/lib/llm-schemas"
import { webSearch, formatSearchResultsForPrompt } from "@/lib/exa"
import { searchCompoundsForTarget } from "@/lib/pubchem"
import {
  errorPayload,
  fetchWithRetry,
  routeBudget,
  toWarning,
  type FailureWarning,
} from "@/lib/retry"
import type { OpenAICallOptions } from "@/lib/openai"
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
  // NCBI answers 429 above its courtesy rate limit: retry. A failure after the
  // retries throws ExternalCallError rather than returning [] (which would read
  // as "no papers" and send the model an empty literature section).
  const esearchRes = await fetchWithRetry(
    `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?${esearchParams.toString()}`,
    {},
    { source: "pubmed", timeoutMs: 30_000 }
  )
  const esearchData = await esearchRes.json()
  const pmids: string[] = esearchData?.esearchresult?.idlist || []
  if (pmids.length === 0) return []

  const efetchParams = ncbiParams({
    db: "pubmed",
    id: pmids.join(","),
    retmode: "xml",
  })
  const efetchRes = await fetchWithRetry(
    `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?${efetchParams.toString()}`,
    {},
    { source: "pubmed", timeoutMs: 30_000 }
  )

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

async function extractProteinsFromText(
  text: string,
  options: OpenAICallOptions
): Promise<string[]> {
  try {
    const result = await queryPromptStructured(
      "extraction/protein_list",
      { text },
      ProteinListSchema,
      options
    )
    return cleanNames(result.proteins)
  } catch (error) {
    if (!(error instanceof LLMOutputError)) throw error
    // Empty list -> the caller retries via extractProteinsFallback.
    console.error("Protein extraction: unusable LLM output:", error.kind, error.message)
    return []
  }
}

/** Trim, drop blanks, and de-duplicate while keeping order. */
function cleanNames(names: string[]): string[] {
  return [...new Set(names.map((n) => n.trim()).filter(Boolean))]
}

/**
 * Last-resort protein extraction: if pulling targets out of the review text
 * failed entirely, search the web for known protein targets of this
 * specific disease and extract from those results — grounded in real
 * sources rather than asking the LLM to recall from parametric memory
 * alone, and not a silently substituted, unrelated hardcoded list.
 */
async function extractProteinsFallback(
  disease: string,
  options: OpenAICallOptions
): Promise<string[]> {
  const searchResults = await webSearch(
    `protein targets implicated in ${disease} pathophysiology and treatment`,
    8
  )
  if (searchResults.length === 0) return []
  const searchResultsText = formatSearchResultsForPrompt(searchResults)

  try {
    const result = await queryPromptStructured(
      "extraction/protein_list_fallback",
      { disease, search_results: searchResultsText },
      ProteinListSchema,
      options
    )
    return cleanNames(result.proteins)
  } catch (error) {
    if (!(error instanceof LLMOutputError)) throw error
    console.error("Protein fallback: unusable LLM output:", error.kind, error.message)
    return []
  }
}

// ---------- Repurposing candidates ----------

async function discoverRepurposingCandidates(
  disease: string,
  proteins: string[],
  options: OpenAICallOptions
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

  return queryPrompt(
    "literature/repurposing_candidates",
    {
      disease,
      proteins: proteins.join(", "),
      search_results: searchResultsText,
    },
    options
  )
}

// ---------- Drug-protein map extraction ----------

async function extractDrugProteinMap(
  drugReviewText: string,
  disease: string,
  proteins: string[],
  options: OpenAICallOptions
): Promise<{
  drugs: ReviewDrug[]
  disease: string
  protein_targets: string[]
  warning?: FailureWarning
}> {
  let rawDrugs: DrugProteinMap["drugs"] = []
  let warning: FailureWarning | undefined
  try {
    const data = await queryPromptStructured(
      "extraction/drug_protein_map",
      { disease, proteins: proteins.join(", "), review_text: drugReviewText },
      DrugProteinMapSchema,
      options
    )
    rawDrugs = data.drugs
  } catch (error) {
    if (!(error instanceof LLMOutputError)) throw error
    // Keep the pipeline going with no literature-derived drugs (bioactive
    // compounds are still discovered per target), but make the gap visible.
    console.error("Drug map: unusable LLM output:", error.kind, error.message)
    warning = {
      source: "openai",
      kind: error.kind,
      message: `Drug-protein map unusable (${error.kind}); no literature-derived drugs`,
    }
  }

  const drugs: ReviewDrug[] = rawDrugs.map((d) => ({
    drug: shortenDrugName(d.drug, null, 40) || "Unknown",
    proteins: d.proteins,
    mechanism: d.mechanism,
    fdaStatus: d.fda_status,
    category: d.category,
    trialStatusForTargets: d.trial_status_for_targets,
  }))

  return {
    drugs,
    disease,
    protein_targets: proteins,
    warning,
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

        // Non-fatal failures (a step degraded, the review continued). Sent as
        // "warning" events as they happen and repeated in "complete".
        const warnings: FailureWarning[] = []
        function warn(warning: FailureWarning) {
          warnings.push(warning)
          send({ type: "warning", warning })
        }

        // Every OpenAI call below shares one deadline, so retries across the
        // whole review stay inside maxDuration.
        const budget = routeBudget(285_000)
        const callOptions = (): OpenAICallOptions => ({ maxTotalMs: budget() })

        try {
          // Step 1: Search PubMed
          send({ type: "progress", message: "Searching PubMed for relevant papers..." })
          const diseasePapers = await searchPubmed(buildDiseaseQuery(disease))
          const diseasePapersText = formatPapersForPrompt(diseasePapers)
          send({ type: "progress", message: `Found ${diseasePapers.length} papers on PubMed` })

          // Step 2: Generate literature review via OpenAI
          send({ type: "progress", message: "Generating literature review with OpenAI..." })
          const firstReview = await queryPrompt("literature/review_first_pass", {
            disease,
            papers: diseasePapersText,
          }, callOptions())

          // Step 3: Extract protein targets
          send({ type: "progress", message: "Extracting protein targets..." })
          let proteins = await extractProteinsFromText(firstReview, callOptions())
          if (proteins.length === 0) {
            send({
              type: "progress",
              message: "Extraction from review text yielded nothing, retrying with a direct query...",
            })
            proteins = await extractProteinsFallback(disease, callOptions())
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
          // These two searches only enrich the drug section ("(No papers found.)" is
          // handled below), so a PubMed failure after retries degrades them to empty,
          // with a warning, instead of discarding the review work done so far.
          const optionalSearch = async (query: string, what: string): Promise<PubmedPaper[]> => {
            try {
              return await searchPubmed(query)
            } catch (error) {
              warn(toWarning(error, what))
              return []
            }
          }
          const drugPapers = await optionalSearch(buildDrugQuery(disease, proteins), "drug papers")
          const repurposePapers = await optionalSearch(buildRepurposingQuery(proteins), "repurposing papers")

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
          const drugReview = await queryPrompt("literature/drug_analysis", {
            disease,
            proteins: proteins.join(", "),
            papers: drugPapersText,
          }, callOptions())

          // Step 6: Repurposing candidates
          send({ type: "progress", message: "Discovering repurposing candidates across therapeutic areas..." })
          const repurposingReview = await discoverRepurposingCandidates(
            disease,
            proteins,
            callOptions()
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
          const drugMap = await extractDrugProteinMap(combined, disease, proteins, callOptions())
          if (drugMap.warning) warn(drugMap.warning)

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
            if (r.status !== "fulfilled") {
              warn(toWarning(r.reason, proteins[i]))
              continue
            }
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
            warnings,
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
    const { status, body } = errorPayload(error)
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    })
  }
}
