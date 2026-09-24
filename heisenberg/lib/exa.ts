// Exa web search client — replaces Perplexity's built-in web search for
// steps that need live grounding (e.g. broad repurposing-candidate discovery).

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

  const res = await fetch(EXA_SEARCH_URL, {
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
    signal: AbortSignal.timeout(30000),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Exa API error ${res.status}: ${text}`)
  }

  const data = await res.json()
  return ((data.results as Record<string, unknown>[]) || []).map((r) => ({
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
