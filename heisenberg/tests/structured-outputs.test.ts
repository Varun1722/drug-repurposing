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

  const reply = (choice: Choice, status = 200) => {
    globalThis.fetch = (async (_url: unknown, init: { body: string }) => {
      lastBody = JSON.parse(init.body)
      return { ok: status === 200, status, text: async () => "boom", json: async () => ({ choices: [choice] }) }
    }) as unknown as typeof fetch
  }
  const ok = (content: unknown, finish_reason = "stop"): Choice => ({ finish_reason, message: { content: JSON.stringify(content) } })

  beforeEach(() => {
    process.env.OPENAI_API_KEY = "test-key"
  })
  afterEach(() => {
    globalThis.fetch = realFetch
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
    assert.equal(lastBody.max_completion_tokens, 8000)
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

  it("keeps HTTP errors as plain Errors", async () => {
    reply({}, 429)
    await assert.rejects(call(), (e: Error) => !(e instanceof LLMOutputError) && /OpenAI API error 429/.test(e.message))
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
  })
})
