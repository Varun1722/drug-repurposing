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
//
// Failure handling (see lib/retry.ts): transient API failures (429/5xx,
// timeouts, dropped connections) are retried with backoff and, if they persist,
// throw ExternalCallError. An unusable *reply* throws LLMOutputError; an empty
// or malformed one is asked for once more first.

import { z } from "zod/v4"
import { SCHEMAS } from "./llm-schemas"
import { PromptError, renderPrompt, type PromptValue } from "./prompts"
import { compactError, ExternalCallError, fetchWithRetry, type RetryPolicy } from "./retry"
import { toStrictJsonSchema } from "./strict-schema"

const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
const DEFAULT_MODEL = "gpt-5.6-terra"

interface ChatChoice {
  finish_reason?: string | null
  message?: { content?: string | null; refusal?: string | null }
}

/**
 * Time budget for one call, covering every retry. A route's `maxDuration`
 * bounds what it can spend: pass a smaller `maxTotalMs` from a 60s route so a
 * stuck call fails with a clean error instead of the platform killing the
 * request. The default fits the longer (180-300s) routes.
 */
export interface OpenAICallOptions {
  maxTotalMs?: number
}

const OPENAI_POLICY: RetryPolicy = {
  maxAttempts: 4,
  baseDelayMs: 1_000,
  maxDelayMs: 15_000,
  maxTotalMs: 150_000,
  maxRetryAfterMs: 20_000,
}

async function postChatCompletion(
  extra: Record<string, unknown>,
  systemPrompt: string,
  userPrompt: string,
  options: OpenAICallOptions = {}
): Promise<ChatChoice | undefined> {
  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) throw new Error("OPENAI_API_KEY not set")

  // A chat completion has no side effects, so repeating it is safe.
  const res = await fetchWithRetry(
    OPENAI_API_URL,
    {
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
        // No max_completion_tokens: hidden reasoning counts toward it, and once the model spent all
        // 8,000 tokens on reasoning and wrote no reply (finish_reason "length"). The route's time
        // budget bounds a call instead. A model without reasoning rejects reasoning_effort (HTTP 400).
        reasoning_effort: "medium",
        ...extra,
      }),
    },
    {
      source: "openai",
      policy: { ...OPENAI_POLICY, ...(options.maxTotalMs ? { maxTotalMs: options.maxTotalMs } : {}) },
    }
  )

  let data: { choices?: ChatChoice[] }
  try {
    data = await res.json()
  } catch {
    throw new ExternalCallError("openai", "bad_response", "openai: response was not valid JSON")
  }
  const choice = data.choices?.[0]
  if (!choice) {
    throw new ExternalCallError("openai", "bad_response", "openai: response had no choices[0]")
  }
  return choice
}

export async function queryOpenAI(
  systemPrompt: string,
  userPrompt: string,
  options: OpenAICallOptions = {}
): Promise<string> {
  const choice = await postChatCompletion({}, systemPrompt, userPrompt, options)
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
  /** How many replies were tried (an empty/malformed one is asked for again once). */
  attempts: number

  constructor(
    readonly kind: LLMOutputErrorKind,
    message: string,
    attempts = 1
  ) {
    super(message)
    this.name = "LLMOutputError"
    this.attempts = attempts
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

// Kinds where asking again can help: the reply was empty or malformed, which a
// fresh sample usually fixes. A refusal, a content-filter stop or a truncated
// reply would just come back the same way.
const REPAIRABLE: ReadonlySet<LLMOutputErrorKind> = new Set(["empty", "invalid_json", "schema_mismatch"])
const MAX_OUTPUT_ATTEMPTS = 2

/**
 * Chat completion constrained to `schema` via `response_format: json_schema`
 * (strict). Returns the parsed, Zod-validated object. `name` labels the schema
 * in the request ([a-zA-Z0-9_-], max 64 chars).
 *
 * An empty or malformed reply is retried once, with a one-line description of
 * what was wrong appended to the prompt so the model can correct it.
 *
 * @throws LLMOutputError when the reply is a refusal, truncated, filtered, or
 *         still unusable on the second try. ExternalCallError when the API
 *         could not be reached (after retries).
 */
export async function queryOpenAIStructured<T extends z.ZodType>(
  systemPrompt: string,
  userPrompt: string,
  schema: T,
  name: string,
  options: OpenAICallOptions = {}
): Promise<z.infer<T>> {
  let prompt = userPrompt
  // One deadline across both tries: the re-ask gets what is left, not a fresh budget.
  const deadline = Date.now() + (options.maxTotalMs ?? OPENAI_POLICY.maxTotalMs)
  for (let attempt = 1; ; attempt++) {
    try {
      const remaining = Math.max(1, deadline - Date.now())
      return await structuredOnce(systemPrompt, prompt, schema, name, { maxTotalMs: remaining })
    } catch (error) {
      if (!(error instanceof LLMOutputError)) throw error
      error.attempts = attempt
      if (!REPAIRABLE.has(error.kind) || attempt >= MAX_OUTPUT_ATTEMPTS) throw error
      console.warn(`[openai] Reply unusable (${error.kind}); asking again.`)
      prompt =
        `${userPrompt}\n\nYour previous reply was rejected: ${compactError(error, 300)}\n` +
        `Reply again with JSON that satisfies the schema.`
    }
  }
}

async function structuredOnce<T extends z.ZodType>(
  systemPrompt: string,
  userPrompt: string,
  schema: T,
  name: string,
  options: OpenAICallOptions
): Promise<z.infer<T>> {
  const choice = await postChatCompletion(
    {
      response_format: {
        type: "json_schema",
        json_schema: { name, strict: true, schema: jsonSchemaFor(schema) },
      },
    },
    systemPrompt,
    userPrompt,
    options
  )

  const message = choice?.message
  if (message?.refusal) {
    throw new LLMOutputError("refusal", `Model refused: ${message.refusal}`)
  }
  if (choice?.finish_reason === "length") {
    throw new LLMOutputError(
      "truncated",
      "Reply was cut off at the model's output limit before the JSON was complete"
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
  variables: Record<string, PromptValue>,
  options: OpenAICallOptions = {}
): Promise<string> {
  const prompt = renderPrompt(id, variables)
  if (prompt.outputSchema !== null) {
    throw new PromptError(
      `${id} declares output.schema '${prompt.outputSchema}'; call queryPromptStructured instead`
    )
  }
  return queryOpenAI(prompt.system, prompt.user, options)
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
  schema: T,
  options: OpenAICallOptions = {}
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
  return queryOpenAIStructured(prompt.system, prompt.user, schema, prompt.outputSchema!, options)
}
