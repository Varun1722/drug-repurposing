// Tests for schema-constrained OpenAI calls (lib/openai.ts, lib/strict-schema.ts,
// lib/llm-schemas.ts). fetch is mocked: no API key, no network.

import assert from "node:assert/strict"
import { afterEach, beforeEach, describe, it } from "node:test"
import { z } from "zod/v4"

import {
  DrugNameListSchema,
  DrugProteinMapSchema,
  ProteinListSchema,
  ReasoningDecisionSchema,
  SCHEMAS,
} from "../lib/llm-schemas"
import {
  LLMOutputError,
  queryOpenAIStructured,
  queryPrompt,
  queryPromptStructured,
} from "../lib/openai"
import { PromptError } from "../lib/prompts"
import { ExternalCallError } from "../lib/retry"
import { toStrictJsonSchema } from "../lib/strict-schema"

describe("strict schema conversion", () => {
  for (const [name, schema] of Object.entries(SCHEMAS)) {
    it(`${name} is strict-mode safe`, () => {
      const json = toStrictJsonSchema(schema)
      assert.equal(json.additionalProperties, false)
      assert.ok(!JSON.stringify(json).includes("minimum"), "integer bounds should be stripped")
    })
  }

  it("decision schema puts reasoning before the action and requires every key", () => {
    const json = toStrictJsonSchema(ReasoningDecisionSchema) as { properties: object; required: string[] }
    const keys = Object.keys(json.properties)
    assert.deepEqual(keys.slice(0, 3), ["rationale", "hypothesis", "action"])
    assert.deepEqual(json.required, keys)
  })

  it("rejects optional, default, root unions and nested optionals", () => {
    assert.throws(() => toStrictJsonSchema(z.object({ a: z.string().optional() })), /non-required properties \(a\)[\s\S]*nullable/)
    assert.throws(() => toStrictJsonSchema(z.object({ a: z.string().default("x") })), /non-required|"default"/)
    assert.throws(
      () => toStrictJsonSchema(z.union([z.object({ a: z.string() }), z.object({ b: z.string() })])),
      /Root schema must be a plain object/
    )
    assert.throws(
      () => toStrictJsonSchema(z.object({ x: z.array(z.object({ y: z.number().optional() })) })),
      /#\/properties\/x\/items[\s\S]*non-required/
    )
  })

  it("allows field names that look like JSON Schema keywords", () => {
    toStrictJsonSchema(z.object({ default: z.string(), not: z.string() }))
  })
})

type Choice = { finish_reason?: string; message?: { content?: string | null; refusal?: string | null } }

describe("queryOpenAIStructured / queryPrompt*", () => {
  const realFetch = globalThis.fetch
  let lastBody: any

  let bodies: any[] = []
  let queue: Array<[Choice, number]> = []
  const realSetTimeout = globalThis.setTimeout
  const waits: number[] = []

  // `reply` sets the response for every request; `queue` is consumed first, one per request.
  const reply = (choice: Choice, status = 200, extra: { text?: string; headers?: Record<string, string> } = {}) => {
    globalThis.fetch = (async (_url: unknown, init: { body: string }) => {
      lastBody = JSON.parse(init.body)
      bodies.push(lastBody)
      const [c, st] = queue.shift() ?? [choice, status]
      return new Response(st >= 400 ? (extra.text ?? "boom") : JSON.stringify({ choices: [c] }), {
        status: st,
        headers: extra.headers,
      })
    }) as unknown as typeof fetch
  }
  const ok = (content: unknown, finish_reason = "stop"): Choice => ({ finish_reason, message: { content: JSON.stringify(content) } })

  beforeEach(() => {
    process.env.OPENAI_API_KEY = "test-key"
    bodies = []
    queue = []
    waits.length = 0
    // Backoff waits are recorded and skipped, so retry tests run instantly.
    globalThis.setTimeout = ((fn: () => void, ms?: number) => {
      waits.push(ms ?? 0)
      fn()
      return 0
    }) as unknown as typeof setTimeout
  })
  afterEach(() => {
    globalThis.fetch = realFetch
    globalThis.setTimeout = realSetTimeout
  })

  const call = () => queryOpenAIStructured("sys", "usr", ProteinListSchema, "ProteinList")
  const kindOf = async (choice: Choice) => {
    reply(choice)
    try {
      await call()
    } catch (e) {
      assert.ok(e instanceof LLMOutputError, String(e))
      return e.kind
    }
    return "no-error"
  }

  it("returns the parsed result and sends a strict json_schema request", async () => {
    reply(ok({ proteins: ["EGFR", "BRAF"] }))
    assert.deepEqual(await call(), { proteins: ["EGFR", "BRAF"] })
    assert.equal(lastBody.response_format.type, "json_schema")
    assert.equal(lastBody.response_format.json_schema.strict, true)
    assert.equal(lastBody.response_format.json_schema.name, "ProteinList")
    assert.equal(lastBody.response_format.json_schema.schema.additionalProperties, false)
    // No output cap: hidden reasoning counts toward it, and in a real run the model spent all
    // 8,000 tokens on reasoning and wrote no reply. Reasoning effort is set, not left to the model's default.
    assert.equal("max_completion_tokens" in lastBody, false)
    assert.equal(lastBody.reasoning_effort, "medium")
    assert.equal("temperature" in lastBody, false)
  })

  it("classifies unusable replies", async () => {
    assert.equal(await kindOf({ finish_reason: "stop", message: { content: null, refusal: "no" } }), "refusal")
    assert.equal(await kindOf({ finish_reason: "length", message: { content: '{"proteins":["EG' } }), "truncated")
    assert.equal(await kindOf({ finish_reason: "content_filter", message: { content: "{" } }), "content_filter")
    assert.equal(await kindOf({ finish_reason: "stop", message: { content: "" } }), "empty")
    assert.equal(await kindOf({ finish_reason: "stop", message: { content: "not json" } }), "invalid_json")
    assert.equal(await kindOf(ok({ proteins: "EGFR" })), "schema_mismatch")
  })

  it("retries transient API errors, then succeeds", async () => {
    reply(ok({ proteins: ["EGFR"] }))
    queue = [[{}, 503], [{}, 429]]
    assert.deepEqual(await call(), { proteins: ["EGFR"] })
    assert.equal(bodies.length, 3)
    assert.equal(waits.length, 2)
  })

  it("throws ExternalCallError (not LLMOutputError) when the API keeps failing", async () => {
    reply({}, 503)
    await assert.rejects(call(), (e: unknown) => {
      assert.ok(e instanceof ExternalCallError && !(e instanceof LLMOutputError))
      assert.equal(e.source, "openai")
      assert.equal(e.kind, "server")
      assert.equal(e.attempts, 4)
      return true
    })
    assert.equal(bodies.length, 4)
  })

  it("does not retry an exhausted credit balance", async () => {
    reply({}, 429, { text: JSON.stringify({ error: { code: "credit_balance_exhausted", type: "insufficient_quota" } }) })
    await assert.rejects(call(), (e: unknown) => e instanceof ExternalCallError && e.kind === "quota")
    assert.equal(bodies.length, 1)
    assert.equal(waits.length, 0)
  })

  it("does not retry a bad request", async () => {
    reply({}, 400)
    await assert.rejects(call(), (e: unknown) => e instanceof ExternalCallError && e.kind === "client")
    assert.equal(bodies.length, 1)
  })

  it("honours Retry-After on a rate limit", async () => {
    reply({}, 429, { headers: { "retry-after": "7" } })
    await assert.rejects(call(), (e: unknown) => e instanceof ExternalCallError && e.kind === "rate_limited")
    assert.ok(waits.length > 0 && waits.every((w) => w >= 7000))
  })

  it("surfaces a Retry-After longer than it will wait", async () => {
    reply({}, 429, { headers: { "retry-after": "600" } })
    await assert.rejects(call(), (e: unknown) => e instanceof ExternalCallError && e.attempts === 1)
    assert.equal(waits.length, 0)
  })

  it("a response with no choices is a bad_response", async () => {
    globalThis.fetch = (async () => new Response(JSON.stringify({ error: "?" }), { status: 200 })) as unknown as typeof fetch
    await assert.rejects(call(), (e: unknown) => e instanceof ExternalCallError && e.kind === "bad_response")
  })

  it("asks again once for a malformed reply, telling the model what was wrong", async () => {
    reply(ok({ proteins: ["EGFR"] }))
    queue = [[ok({ proteins: "EGFR" }), 200]]
    assert.deepEqual(await queryOpenAIStructured("sys", "original prompt", ProteinListSchema, "ProteinList"), {
      proteins: ["EGFR"],
    })
    assert.equal(bodies.length, 2)
    assert.equal(bodies[0].messages[1].content, "original prompt")
    assert.match(bodies[1].messages[1].content, /^original prompt\n\nYour previous reply was rejected: Reply failed validation/)
  })

  it("gives up after the second unusable reply and reports the attempt count", async () => {
    reply({ finish_reason: "stop", message: { content: "not json" } })
    await assert.rejects(call(), (e: unknown) => e instanceof LLMOutputError && e.kind === "invalid_json" && e.attempts === 2)
    assert.equal(bodies.length, 2)
  })

  it("does not ask again after a refusal, truncation or content filter", async () => {
    for (const choice of [
      { finish_reason: "stop", message: { content: null, refusal: "no" } },
      { finish_reason: "length", message: { content: '{"proteins":["EG' } },
      { finish_reason: "content_filter", message: { content: "{" } },
    ] as Choice[]) {
      bodies = []
      reply(choice)
      await assert.rejects(call(), (e: unknown) => e instanceof LLMOutputError && e.attempts === 1)
      assert.equal(bodies.length, 1)
    }
  })

  it("the re-ask shares the call's time budget instead of getting a fresh one", async () => {
    const real = AbortSignal.timeout
    const asked: number[] = []
    AbortSignal.timeout = ((ms: number) => {
      asked.push(ms)
      return real.call(AbortSignal, ms)
    }) as typeof AbortSignal.timeout
    try {
      reply(ok({ proteins: ["EGFR"] }))
      queue = [[ok({ proteins: "EGFR" }), 200]]
      // first reply is malformed; the second attempt must get what is left of 5s
      const slow = globalThis.fetch
      globalThis.fetch = (async (...args: Parameters<typeof fetch>) => {
        await new Promise((r) => realSetTimeout(r, 80))
        return slow(...args)
      }) as typeof fetch
      await queryOpenAIStructured("s", "u", ProteinListSchema, "ProteinList", { maxTotalMs: 5_000 })
    } finally {
      AbortSignal.timeout = real
    }
    assert.equal(asked.length, 2)
    assert.ok(asked[1] < asked[0], `second attempt should get less time: ${asked}`)
  })

  it("lets a route cap the time budget of a call", async () => {
    // maxTotalMs 1ms leaves no room for a second attempt, so the first failure is final.
    reply({}, 503)
    await assert.rejects(
      queryOpenAIStructured("s", "u", ProteinListSchema, "ProteinList", { maxTotalMs: 1 }),
      (e: unknown) => e instanceof ExternalCallError && e.attempts === 1
    )
  })

  it("validates enums with zod on the way back", async () => {
    reply(ok({ rationale: "r", hypothesis: "h", action: "delete_everything", seed_cids: null, drug_class: null, drug_names: null }))
    await assert.rejects(
      queryOpenAIStructured("s", "u", ReasoningDecisionSchema, "ReasoningDecision"),
      (e: unknown) => e instanceof LLMOutputError && e.kind === "schema_mismatch"
    )
  })

  it("queryPromptStructured renders the prompt file and names the schema from its front matter", async () => {
    reply(ok({ drug_names: ["atorvastatin"] }))
    const result = await queryPromptStructured(
      "analysis/class_expansion",
      { drug_class: "statins", proteins: "KRAS", exclude: "(none)" },
      DrugNameListSchema
    )
    assert.deepEqual(result, { drug_names: ["atorvastatin"] })
    assert.equal(lastBody.response_format.json_schema.name, "DrugNameList")
    assert.match(lastBody.messages[1].content, /^List 10 FDA-approved drugs in the class: statins/)
  })

  it("keeps prompts tied to their schema and to the right helper", async () => {
    reply(ok({ drug_names: [] }))
    const vars = { drug_class: "statins", proteins: "KRAS", exclude: "(none)" }
    await assert.rejects(queryPromptStructured("analysis/class_expansion", vars, ProteinListSchema), PromptError)
    await assert.rejects(queryPromptStructured("analysis/class_expansion", vars, DrugProteinMapSchema), PromptError)
    await assert.rejects(queryPrompt("analysis/class_expansion", vars), /call queryPromptStructured instead/)
    await assert.rejects(
      queryPromptStructured("chat/answer", { context_block: "", question: "q" }, ProteinListSchema),
      /declares output.schema none/
    )
  })

  it("queryPrompt returns free text from a plain prompt", async () => {
    reply({ finish_reason: "stop", message: { content: "an answer" } })
    assert.equal(await queryPrompt("chat/answer", { context_block: "Context:\nx\n\n", question: "Which?" }), "an answer")
    assert.equal(lastBody.response_format, undefined)
    assert.match(lastBody.messages[1].content, /^Context:\nx\n\nQuestion: Which\?$/)
    assert.equal("max_completion_tokens" in lastBody, false)
    assert.equal(lastBody.reasoning_effort, "medium")
  })
})
