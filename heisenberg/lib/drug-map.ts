// Turns the review text into a drug -> protein list.
//
// One extraction call over the whole review could come back cut off ("truncated") and
// the route then went on with no review drugs at all (NSCLC and COVID-19 in real runs;
// in the one run measured the model had spent the whole 8,000-token cap on hidden
// reasoning and written nothing, which is why lib/openai.ts no longer sets a cap). So:
// each review section is extracted on its own and at the same time (shorter input, a
// faster step); a cut-off reply is retried on the two halves of that text; what cannot
// be recovered costs a warning and only that piece, never the other pieces. An API
// failure still throws, so an outage is not read as "no drugs".

import { DrugProteinMapSchema, type DrugProteinMap } from "./llm-schemas"
import { LLMOutputError, queryPromptStructured, type OpenAICallOptions } from "./openai"
import type { FailureWarning } from "./retry"
import { shortenDrugName, type ReviewDrug } from "./types"

export interface DrugMapSection {
  /** Names the section in a warning, e.g. "drug analysis". */
  label: string
  text: string
}

/** A section is split at most this many times in a row: it becomes at most 4 pieces. */
const MAX_SPLITS = 2
/** A reply is not cut off by a section shorter than this, so splitting it would not help. */
const MIN_SPLIT_CHARS = 1000

type Piece = { drugs: ReviewDrug[]; warnings: FailureWarning[] }

/**
 * Cuts `text` in two at a blank line (else a line break) near the middle; no text
 * is lost. Null when it is too short to need it or has no line break to cut at.
 * shortcut: a table cut in two leaves its header row in the first half only; the
 * model still reads the second half's rows, and the first half is the one with the
 * header. Upgrade to repeating the header if drugs go missing from split tables.
 */
export function splitInHalf(text: string): [string, string] | null {
  if (text.length < MIN_SPLIT_CHARS) return null
  const middle = text.length / 2
  const candidates: { at: number; length: number }[] = []
  for (const separator of ["\n\n", "\n"]) {
    let nearest = -1
    for (let i = text.indexOf(separator); i !== -1; i = text.indexOf(separator, i + 1)) {
      if (nearest === -1 || Math.abs(i - middle) < Math.abs(nearest - middle)) nearest = i
    }
    if (nearest > 0 && nearest + separator.length < text.length) candidates.push({ at: nearest, length: separator.length })
  }
  // A blank line only if it leaves halves of similar size; otherwise whichever is closest to the middle.
  const balanced = candidates.find((c) => c.at >= text.length * 0.25 && c.at <= text.length * 0.75)
  const best = balanced ?? candidates.sort((a, b) => Math.abs(a.at - middle) - Math.abs(b.at - middle))[0]
  return best ? [text.slice(0, best.at), text.slice(best.at + best.length)] : null
}

const nameKey = (name: string) => name.trim().toLowerCase().replace(/\s+/g, " ")

/** One entry per drug name (ignoring case and spacing); the first entry's wording is kept. */
export function mergeDrugs(lists: ReviewDrug[][]): ReviewDrug[] {
  const byName = new Map<string, ReviewDrug>()
  for (const drug of lists.flat()) {
    const key = nameKey(drug.drug)
    const seen = byName.get(key)
    byName.set(
      key,
      seen
        ? {
            ...seen,
            proteins: [...new Set([...seen.proteins, ...drug.proteins])],
            mechanism: seen.mechanism || drug.mechanism,
            fdaStatus: seen.fdaStatus || drug.fdaStatus,
            category: seen.category === "mainstream" || drug.category === "mainstream" ? "mainstream" : seen.category,
            trialStatusForTargets: seen.trialStatusForTargets || drug.trialStatusForTargets,
          }
        : { ...drug, proteins: [...new Set(drug.proteins)] }
    )
  }
  return [...byName.values()]
}

const toReviewDrug = (d: DrugProteinMap["drugs"][number]): ReviewDrug => ({
  drug: shortenDrugName(d.drug, null, 40) || "Unknown",
  proteins: d.proteins,
  mechanism: d.mechanism,
  fdaStatus: d.fda_status,
  category: d.category,
  trialStatusForTargets: d.trial_status_for_targets,
})

async function extractPiece(
  label: string,
  text: string,
  disease: string,
  proteins: string[],
  callOptions: () => OpenAICallOptions,
  depth: number
): Promise<Piece> {
  try {
    const data = await queryPromptStructured(
      "extraction/drug_protein_map",
      { disease, proteins: proteins.join(", "), review_text: text },
      DrugProteinMapSchema,
      callOptions()
    )
    return { drugs: data.drugs.map(toReviewDrug), warnings: [] }
  } catch (error) {
    if (!(error instanceof LLMOutputError)) throw error
    const halves = error.kind === "truncated" && depth < MAX_SPLITS ? splitInHalf(text) : null
    if (halves) {
      const [first, second] = await Promise.all(
        halves.map((half) => extractPiece(label, half, disease, proteins, callOptions, depth + 1))
      )
      return { drugs: [...first.drugs, ...second.drugs], warnings: [...first.warnings, ...second.warnings] }
    }
    // Keep the pipeline going without this piece's drugs (bioactive compounds are
    // still found per target), but make the gap visible.
    console.error(`Drug map: unusable LLM output for the ${label} section:`, error.kind, error.message)
    return {
      drugs: [],
      warnings: [
        {
          source: "openai",
          kind: error.kind,
          message: `Drug-protein map unusable for the ${label} section (${error.kind}); its drugs are missing`,
        },
      ],
    }
  }
}

/**
 * The drugs named in each section and the proteins each may bind. `callOptions` is
 * called before every request so each one gets what is left of the route's time budget.
 */
export async function extractDrugMap(
  sections: DrugMapSection[],
  disease: string,
  proteins: string[],
  callOptions: () => OpenAICallOptions
): Promise<{ drugs: ReviewDrug[]; warnings: FailureWarning[] }> {
  const pieces = await Promise.all(
    sections
      .filter((section) => section.text.trim() !== "")
      .map((section) => extractPiece(section.label, section.text, disease, proteins, callOptions, 0))
  )
  return { drugs: mergeDrugs(pieces.map((p) => p.drugs)), warnings: pieces.flatMap((p) => p.warnings) }
}
