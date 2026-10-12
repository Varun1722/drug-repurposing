// OpenAI API client — replaces lib/perplexity.ts for LLM/intelligence calls.
// Live web search is handled separately via lib/exa.ts.
//
// Two entry points:
//   queryOpenAI            free-text completions (prose, markdown sections)
//   queryOpenAIStructured  schema-constrained completions (Structured Outputs)
// and the same two driven by a prompts/*.prompt file:
//   queryPrompt            render the prompt file, then queryOpenAI
//   queryPromptStructured  render the prompt file, then queryOpenAIStructured
// Prefer the *Prompt helpers: prompt text lives in prompts/, not in routes.

import { z } from "zod/v4"
import { SCHEMAS } from "./llm-schemas"
import { PromptError, renderPrompt, type PromptValue } from "./prompts"
import { toStrictJsonSchema } from "./strict-schema"

const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
const DEFAULT_MODEL = "gpt-5.6-terra"

interface ChatChoice {
  finish_reason?: string | null
  message?: { content?: string | null; refusal?: string | null }
}

async function postChatCompletion(
  extra: Record<string, unknown>,
  systemPrompt: string,
  userPrompt: string
): Promise<ChatChoice | undefined> {
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
      ...extra,
    }),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`OpenAI API error ${res.status}: ${text}`)
  }

  const data = await res.json()
  return data.choices?.[0] as ChatChoice | undefined
}

export async function queryOpenAI(
  systemPrompt: string,
  userPrompt: string
): Promise<string> {
  const choice = await postChatCompletion({}, systemPrompt, userPrompt)
  return choice?.message?.content ?? ""
}

export type LLMOutputErrorKind =
  | "refusal"
  | "truncated"
  | "content_filter"
  | "empty"
  | "invalid_json"
  | "schema_mismatch"

/**
 * The API call succeeded but the reply is not a usable structured result.
 * Strict mode guarantees schema-valid JSON for a normal completion, so these
 * only occur when the model refused, ran out of tokens, was content-filtered,
 * or returned values that fail our own (stricter) Zod validation. Callers
 * decide the fallback; they should record `kind` instead of failing silently.
 */
export class LLMOutputError extends Error {
  constructor(
    readonly kind: LLMOutputErrorKind,
    message: string
  ) {
    super(message)
    this.name = "LLMOutputError"
  }
}

const jsonSchemaCache = new WeakMap<z.ZodType, Record<string, unknown>>()

function jsonSchemaFor(schema: z.ZodType): Record<string, unknown> {
  let jsonSchema = jsonSchemaCache.get(schema)
  if (!jsonSchema) {
    jsonSchema = toStrictJsonSchema(schema)
    jsonSchemaCache.set(schema, jsonSchema)
  }
  return jsonSchema
}

/**
 * Chat completion constrained to `schema` via `response_format: json_schema`
 * (strict). Returns the parsed, Zod-validated object. `name` labels the schema
 * in the request ([a-zA-Z0-9_-], max 64 chars).
 *
 * @throws LLMOutputError when the reply is a refusal, truncated, filtered,
 *         empty, or fails validation. Network/HTTP failures throw a plain Error.
 */
export async function queryOpenAIStructured<T extends z.ZodType>(
  systemPrompt: string,
  userPrompt: string,
  schema: T,
  name: string
): Promise<z.infer<T>> {
  const choice = await postChatCompletion(
    {
      response_format: {
        type: "json_schema",
        json_schema: { name, strict: true, schema: jsonSchemaFor(schema) },
      },
    },
    systemPrompt,
    userPrompt
  )

  const message = choice?.message
  if (message?.refusal) {
    throw new LLMOutputError("refusal", `Model refused: ${message.refusal}`)
  }
  if (choice?.finish_reason === "length") {
    throw new LLMOutputError(
      "truncated",
      "Reply hit max_completion_tokens before the JSON was complete"
    )
  }
  if (choice?.finish_reason === "content_filter") {
    throw new LLMOutputError("content_filter", "Reply stopped by the content filter")
  }
  if (!message?.content) {
    throw new LLMOutputError("empty", "Model returned no content")
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(message.content)
  } catch {
    throw new LLMOutputError("invalid_json", "Reply was not valid JSON")
  }

  const result = schema.safeParse(parsed)
  if (!result.success) {
    const issues = result.error.issues
      .map((issue) => `${issue.path.join(".") || "(root)"}: ${issue.message}`)
      .join("; ")
    throw new LLMOutputError("schema_mismatch", `Reply failed validation: ${issues}`)
  }
  return result.data
}

/** Render `prompts/<id>.prompt` and return the model's free-text reply. */
export async function queryPrompt(
  id: string,
  variables: Record<string, PromptValue>
): Promise<string> {
  const prompt = renderPrompt(id, variables)
  if (prompt.outputSchema !== null) {
    throw new PromptError(
      `${id} declares output.schema '${prompt.outputSchema}'; call queryPromptStructured instead`
    )
  }
  return queryOpenAI(prompt.system, prompt.user)
}

/**
 * Render `prompts/<id>.prompt` and return the reply parsed as `schema`.
 * The prompt file must declare `output.schema: <name>` for this exact schema
 * (see SCHEMAS in lib/llm-schemas.ts), which keeps each prompt tied to the
 * schema it was written for.
 */
export async function queryPromptStructured<T extends z.ZodType>(
  id: string,
  variables: Record<string, PromptValue>,
  schema: T
): Promise<z.infer<T>> {
  const prompt = renderPrompt(id, variables)
  const registry: Record<string, z.ZodType> = SCHEMAS
  const expected = prompt.outputSchema === null ? undefined : registry[prompt.outputSchema]
  if (expected === undefined || expected !== schema) {
    throw new PromptError(
      `${id} declares output.schema ${prompt.outputSchema === null ? "none" : `'${prompt.outputSchema}'`} ` +
        `but was called with a different schema`
    )
  }
  return queryOpenAIStructured(prompt.system, prompt.user, schema, prompt.outputSchema!)
}
