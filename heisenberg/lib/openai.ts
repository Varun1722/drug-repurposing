// OpenAI API client — replaces lib/perplexity.ts for LLM/intelligence calls.
// Live web search is handled separately via lib/exa.ts.

const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
const DEFAULT_MODEL = "gpt-5.6-terra"

export async function queryOpenAI(
  systemPrompt: string,
  userPrompt: string
): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) throw new Error("OPENAI_API_KEY not set")

  const res = await fetch(OPENAI_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: process.env.OPENAI_MODEL || DEFAULT_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      // No custom temperature — some models (e.g. gpt-5.6-terra) only
      // support the default (1) and reject any other value.
      max_completion_tokens: 8000,
    }),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`OpenAI API error ${res.status}: ${text}`)
  }

  const data = await res.json()
  return data.choices?.[0]?.message?.content ?? ""
}

/**
 * Robustly parse JSON from an LLM response.
 * Handles markdown fences, trailing commas, control chars.
 */
export function parseJsonResponse(raw: string): Record<string, unknown> {
  // Strip markdown code fences
  let cleaned = raw.replace(/```(?:json)?\s*/gi, "").replace(/```\s*/g, "")

  // Try to extract JSON object or array
  const jsonMatch = cleaned.match(/(\{[\s\S]*\}|\[[\s\S]*\])/)
  if (jsonMatch) {
    cleaned = jsonMatch[1]
  }

  // Remove trailing commas before } or ]
  cleaned = cleaned.replace(/,\s*([}\]])/g, "$1")

  // Remove control characters (except newline, tab)
  cleaned = cleaned.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, "")

  try {
    return JSON.parse(cleaned) as Record<string, unknown>
  } catch {
    // Last resort: try to find any JSON-like structure
    const fallback = cleaned.match(/\{[^{}]*\}/)
    if (fallback) {
      try {
        return JSON.parse(fallback[0]) as Record<string, unknown>
      } catch {
        // Give up
      }
    }
    return {}
  }
}
