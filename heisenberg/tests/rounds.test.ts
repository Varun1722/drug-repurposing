// The web app's single expansion-round limit (lib/rounds.ts) and the
// /api/analyze route that enforces it.

import assert from "node:assert/strict"
import { afterEach, beforeEach, describe, it } from "node:test"

import { POST } from "../app/api/analyze/route"
import { MAX_DOCKING_ROUND, MAX_EXPANSION_ROUNDS, mayExpandAfter } from "../lib/rounds"
import { mockFetch, res } from "./_fetch"

describe("lib/rounds", () => {
  it("round 1 is the initial docking, so N expansions end at docking round N + 1", () => {
    assert.equal(MAX_DOCKING_ROUND, MAX_EXPANSION_ROUNDS + 1)
    assert.equal(MAX_EXPANSION_ROUNDS, 2) // same default as pipeline.py
  })

  it("may expand until the limit, never at or past it", () => {
    const allowed = [1, 2, 3, 4, 5].filter(mayExpandAfter)
    assert.deepEqual(allowed, Array.from({ length: MAX_EXPANSION_ROUNDS }, (_, i) => i + 1))
  })
})

describe("POST /api/analyze", () => {
  let http: ReturnType<typeof mockFetch>
  beforeEach(() => {
    process.env.OPENAI_API_KEY = "test-key"
    http = mockFetch({})
  })
  afterEach(() => http.restore())

  const call = (round: number) =>
    POST(
      new Request("http://x/api/analyze", {
        method: "POST",
        body: JSON.stringify({
          disease: "pancreatic cancer",
          round,
          dockingResults: [{ name: "a", confidenceScore: 0.9, mechanism: "", fdaStatus: "", proteinTarget: "KRAS", source: "", round: 1 }],
        }),
      })
    )

  const modelSays = (decision: object) => {
    http.restore()
    http = mockFetch({
      "api.openai.com": res(200, { choices: [{ finish_reason: "stop", message: { content: JSON.stringify(decision) } }] }),
    })
  }
  const expand = { rationale: "r", hypothesis: "h", action: "expand_class", seed_cids: null, drug_class: "statins", drug_names: ["a"] }

  it("at the limit it proceeds without asking the model", async () => {
    modelSays(expand)
    const body = await (await call(MAX_DOCKING_ROUND)).json()
    assert.equal(body.action, "proceed")
    assert.match(body.rationale, /Max rounds reached/)
    assert.equal(http.calls.length, 0)
  })

  it("past the limit, same", async () => {
    modelSays(expand)
    assert.equal((await (await call(MAX_DOCKING_ROUND + 3)).json()).action, "proceed")
    assert.equal(http.calls.length, 0)
  })

  it("below the limit it asks the model and tells it the real limit", async () => {
    modelSays(expand)
    const body = await (await call(1)).json()
    assert.equal(body.action, "expand_class")
    assert.deepEqual(body.drugNames, ["a"])
    const sent = JSON.parse(http.calls[0].init.body as string)
    assert.match(sent.messages[0].content, new RegExp(`Max rounds: ${MAX_DOCKING_ROUND}\\.`))
  })

  it("the last round that may expand still can", async () => {
    modelSays(expand)
    assert.equal((await (await call(MAX_EXPANSION_ROUNDS)).json()).action, "expand_class")
    assert.equal(http.calls.length, 1)
  })
})
