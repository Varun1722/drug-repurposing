// The real /api/review handler on a fake network: a review whose drug-map reply is cut
// off must still give every drug (lib/drug-map.ts has the unit tests). In real runs a
// cut-off reply left NSCLC and COVID-19 with no review drugs at all. The fake model
// cuts a reply off when it would name more than 66 drugs: a stand-in for any limit on
// the reply, since what cut off the real replies was hidden reasoning (see lib/openai.ts).

import assert from "node:assert/strict"
import { afterEach, beforeEach, describe, it } from "node:test"

import { POST as review } from "../app/api/review/route"
import { readSSEStream } from "../lib/sse"
import { mockFetch, res } from "./_fetch"

const CAP = 66
const section = (tag: string, n: number) =>
  Array.from({ length: n }, (_, i) => `${tag}-p${i} ${"lorem ".repeat(60)}`).join("\n\n")

let http: ReturnType<typeof mockFetch>
let mapCalls: number
beforeEach(() => {
  process.env.OPENAI_API_KEY = "test-key"
  process.env.EXA_API_KEY = "test-key"
  mapCalls = 0
})
afterEach(() => http.restore())

/** The network, with a drug-analysis section naming `nA` drugs and a repurposing section naming `nB`. */
function network(nA: number, nB: number, neverFits?: string) {
  http = mockFetch({
    "eutils.ncbi.nlm.nih.gov": res(200, { esearchresult: { idlist: [] } }),
    "api.exa.ai": res(200, { results: [] }),
    "pubchem.ncbi.nlm.nih.gov": res(404),
    "api.openai.com": ({ init }) => {
      const body = JSON.parse(String(init.body))
      const system: string = body.messages[0].content
      const user: string = body.messages[1].content
      const reply = (content: string, finish_reason = "stop") => res(200, { choices: [{ finish_reason, message: { content } }] })
      if (system.includes("data-extraction")) {
        mapCalls++
        const text = user.slice(user.indexOf("Text:\n"))
        const tags = [...text.matchAll(/([A-Z])-p(\d+) /g)]
        if (tags.length > CAP || (neverFits && text.includes(neverFits))) return reply('{"drugs":[{"drug":"cut', "length")
        const drugs = tags.map(([, t, n]) => ({
          drug: `${t}-drug${n}`,
          proteins: ["EGFR"],
          mechanism: "m",
          fda_status: "approved",
          category: "mainstream",
          trial_status_for_targets: "none",
        }))
        return reply(JSON.stringify({ drugs }))
      }
      if (system.includes("research assistant")) return reply(JSON.stringify({ proteins: ["EGFR", "KRAS"] }))
      if (system.includes("pharmacology researcher")) return reply(section("A", nA))
      if (system.includes("drug repurposing")) return reply(section("B", nB))
      return reply("First review.")
    },
  })
}

async function runReview() {
  const response = await review(new Request("http://t/api/review", { method: "POST", body: JSON.stringify({ disease: "test disease" }) }))
  let complete: { drugs: { drug: string; mechanism: string }[]; warnings: { message: string }[] } | undefined
  await readSSEStream(response, (event) => {
    if (event.type === "complete") complete = event as never
  }, "review")
  assert.ok(complete)
  return { ...complete, reviewDrugs: complete.drugs.filter((d) => !d.mechanism.startsWith("Bioactive")) }
}

describe("/api/review drug list", () => {
  it("gives every drug when each section fits in one reply", async () => {
    network(30, 20)
    const out = await runReview()
    assert.equal(out.reviewDrugs.length, 50)
    assert.equal(mapCalls, 2, "one call per section")
    assert.deepEqual(out.warnings, [])
  })

  it("gives every drug when together they are over the cap (the old single call lost all of them)", async () => {
    network(50, 40)
    const out = await runReview()
    assert.equal(out.reviewDrugs.length, 90)
    assert.deepEqual(out.warnings, [])
  })

  it("splits a section that is over the cap on its own and still gives every drug", async () => {
    network(80, 10)
    const out = await runReview()
    assert.equal(out.reviewDrugs.length, 90)
    assert.equal(mapCalls, 4, "the cut-off section, its two halves, and the other section")
    assert.deepEqual(out.warnings, [])
  })

  it("keeps what it could extract and says what it could not", async () => {
    network(80, 10, "A-p79 ")
    const out = await runReview()
    assert.ok(out.reviewDrugs.length >= 10 + 40, "the other section and the first half of this one")
    assert.ok(out.reviewDrugs.length < 90)
    assert.ok(out.warnings.some((w) => /Drug-protein map unusable for the drug analysis section \(truncated\)/.test(w.message)))
  })
})
