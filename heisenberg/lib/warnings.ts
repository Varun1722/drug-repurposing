// Wording for failures shown to the user. Routes report failed external calls
// as { source, kind, message, target? } warnings (see lib/retry.ts); this turns
// a list of them into one sentence for the chat.

import type { FailureWarning } from "./retry"

const SOURCE_LABELS: Record<string, string> = {
  pubchem: "PubChem",
  chembl: "ChEMBL",
  rcsb: "RCSB PDB",
  openai: "OpenAI",
  exa: "Exa web search",
  pubmed: "PubMed",
  gnina: "GNINA docking",
}

const MAX_TARGETS_SHOWN = 3

/**
 * "2 PubChem calls failed (MGMT/flaky, KRAS); 1 OpenAI call failed." Calls are
 * grouped by source, with a few of the affected items named. Returns "" for an
 * empty list.
 */
export function describeWarnings(warnings: readonly FailureWarning[]): string {
  const bySource = new Map<string, FailureWarning[]>()
  for (const w of warnings) bySource.set(w.source, [...(bySource.get(w.source) ?? []), w])

  return [...bySource.entries()]
    .map(([source, items]) => {
      const label = SOURCE_LABELS[source] ?? source
      const targets = [...new Set(items.map((i) => i.target).filter((t): t is string => !!t))]
      const shown = targets.slice(0, MAX_TARGETS_SHOWN).join(", ")
      const more = targets.length > MAX_TARGETS_SHOWN ? ", …" : ""
      const named = targets.length > 0 ? ` (${shown}${more})` : ""
      return `${items.length} ${label} call${items.length === 1 ? "" : "s"} failed${named}`
    })
    .join("; ")
}

/** The `error` string of a failed API response, or `fallback` if there is none. */
export async function errorMessageFrom(res: Response, fallback: string): Promise<string> {
  try {
    const body = (await res.json()) as { error?: unknown }
    return typeof body.error === "string" && body.error ? body.error : fallback
  } catch {
    return fallback
  }
}
