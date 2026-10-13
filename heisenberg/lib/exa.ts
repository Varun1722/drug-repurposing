// Exa web search client — replaces Perplexity's built-in web search for
// steps that need live grounding (e.g. broad repurposing-candidate discovery).

import { fetchWithRetry, responseJson } from "./retry"

const EXA_SEARCH_URL = "https://api.exa.ai/search"

export interface ExaSearchResult {
  title: string
  url: string
  text: string
}

export async function webSearch(
  query: string,
  numResults: number = 10
): Promise<ExaSearchResult[]> {
  const apiKey = process.env.EXA_API_KEY
  if (!apiKey) throw new Error("EXA_API_KEY not set")

  // A search has no side effects, so repeating it is safe. Throws
  // ExternalCallError if Exa still fails after the retries.
  const res = await fetchWithRetry(
    EXA_SEARCH_URL,
    {
      method: "POST",
      headers: {
        "x-api-key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        type: "auto",
        numResults,
        contents: { text: { maxCharacters: 1500 } },
      }),
    },
    { source: "exa", timeoutMs: 30_000 }
  )

  const data = await responseJson<{ results?: Record<string, unknown>[] }>(res, "exa")
  return (data.results || []).map((r) => ({
    title: (r.title as string) || "",
    url: (r.url as string) || "",
    text: (r.text as string) || "",
  }))
}

/**
 * Format Exa search results into a numbered, citable block for an LLM prompt.
 */
export function formatSearchResultsForPrompt(results: ExaSearchResult[]): string {
  return results
    .map(
      (r, i) =>
        `[${i + 1}] ${r.title}\n    URL: ${r.url}\n    ${r.text.slice(0, 1000)}\n`
    )
    .join("\n")
}
