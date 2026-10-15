// Turning the review text into a drug -> protein list (lib/drug-map.ts).
//
// Why it exists: one extraction call over the whole review could come back cut off,
// and the route went on with no review drugs at all (NSCLC and COVID-19 in the real
// runs). Contract: each review section is extracted on its own
// and at the same time; a cut-off reply is retried on the two halves of that text,
// to a limited depth; what cannot be recovered costs a warning and only that piece,
// never the other pieces; an API failure still throws (an outage is not "no drugs").
// The model is faked: it "extracts" one drug per paragraph tag it finds in the text.

import assert from "node:assert/strict"
import { afterEach, beforeEach, describe, it } from "node:test"

import { extractDrugMap, mergeDrugs, splitInHalf } from "../lib/drug-map"
import { DrugProteinMapSchema } from "../lib/llm-schemas"
import { ExternalCallError } from "../lib/retry"
import { toStrictJsonSchema } from "../lib/strict-schema"
import type { ReviewDrug } from "../lib/types"
import { instantTimers, mockFetch, res } from "./_fetch"

const PROTEINS = ["EGFR", "KRAS"]
const options = () => ({})

/** `n` paragraphs of about 370 characters, each carrying a tag the fake model turns into a drug. */
const section = (tag: string, n: number) =>
  Array.from({ length: n }, (_, i) => `${tag}-p${i} ${"lorem ".repeat(60)}`).join("\n\n")

type Outcome = "truncate" | "refuse" | { category?: string; proteins?: string[] }
type Model = (userText: string) => Outcome | undefined

/** One drug per `TAG-pN` found in the request: what a perfect extraction of that text would give. */
function drugsIn(text: string, outcome: Outcome | undefined) {
  const drugs = [...text.matchAll(/([A-Z]+)-p(\d+) /g)].map(([, tag, n]) => ({
    drug: `${tag}-drug${n}`,
    proteins: typeof outcome === "object" && outcome.proteins ? outcome.proteins : ["EGFR"],
    mechanism: "m",
    fda_status: "approved",
    category: (typeof outcome === "object" && outcome.category) || "mainstream",
    trial_status_for_targets: "none",
  }))
  return { drugs }
}

let http: ReturnType<typeof mockFetch>
let timers: ReturnType<typeof instantTimers>
let requests: string[] // the review text of each request, in order sent

/** The fake model's reply to one request: cut off, refuse, or extract. Records the review text it was sent. */
function modelReply(init: RequestInit, model: Model): Response {
  const user: string = JSON.parse(String(init.body)).messages[1].content
  const text = user.slice(user.indexOf("Text:\n") + "Text:\n".length)
  requests.push(text)
  const outcome = model(text)
  if (outcome === "truncate") {
    return res(200, { choices: [{ finish_reason: "length", message: { content: '{"drugs":[{"drug":"cut' } }] })
  }
  if (outcome === "refuse") {
    return res(200, { choices: [{ finish_reason: "stop", message: { content: null, refusal: "no" } }] })
  }
  return res(200, { choices: [{ finish_reason: "stop", message: { content: JSON.stringify(drugsIn(text, outcome)) } }] })
}

/** Fake OpenAI: `model` decides, per request text, whether to cut off, refuse, or extract. */
function fakeModel(model: Model) {
  http.restore()
  http = mockFetch({ "api.openai.com": ({ init }) => modelReply(init, model) })
}

beforeEach(() => {
  process.env.OPENAI_API_KEY = "test-key"
  timers = instantTimers()
  http = mockFetch({})
  requests = []
})
afterEach(() => {
  http.restore()
  timers.restore()
})

const names = (drugs: ReviewDrug[]) => drugs.map((d) => d.drug).sort()

describe("splitInHalf", () => {
  it("splits at the blank line nearest the middle and loses no text", () => {
    const text = section("A", 8)
    const halves = splitInHalf(text)
    assert.ok(halves)
    const [first, second] = halves
    assert.equal(`${first}\n\n${second}`, text)
    assert.ok(first.includes("A-p3 ") && !first.includes("A-p4 "), "4 paragraphs each side")
    assert.ok(second.includes("A-p4 "))
  })

  it("falls back to a line break when there is no blank line (a long table)", () => {
    const text = Array.from({ length: 12 }, (_, i) => `| A-p${i} | ${"cell ".repeat(40)} |`).join("\n")
    const [first, second] = splitInHalf(text)!
    assert.equal(`${first}\n${second}`, text)
    assert.ok(first.length > 0 && second.length > 0)
  })

  it("gives up on text it cannot or need not split", () => {
    assert.equal(splitInHalf("x".repeat(5000)), null, "no line break at all")
    assert.equal(splitInHalf(section("A", 2)), null, "too short to have filled an 8,000-token reply")
    assert.equal(splitInHalf(""), null)
  })
})

describe("mergeDrugs", () => {
  const drug = (over: Partial<ReviewDrug>): ReviewDrug => ({
    drug: "Palbociclib",
    proteins: ["CDK4"],
    mechanism: "first",
    fdaStatus: "approved",
    category: "repurposing_candidate",
    trialStatusForTargets: "t1",
    ...over,
  })

  it("one entry per drug name, ignoring case and spacing; proteins are joined without repeats", () => {
    const merged = mergeDrugs([[drug({})], [drug({ drug: "  palbociclib ", proteins: ["CDK6", "CDK4"], mechanism: "second" })]])
    assert.equal(merged.length, 1)
    assert.deepEqual(merged[0].proteins, ["CDK4", "CDK6"])
    assert.equal(merged[0].mechanism, "first", "the first section's wording is kept")
  })

  it("'mainstream' wins when the sections disagree", () => {
    const merged = mergeDrugs([[drug({})], [drug({ category: "mainstream" })]])
    assert.equal(merged[0].category, "mainstream")
  })

  it("fills a field the first entry left empty from the second", () => {
    const merged = mergeDrugs([[drug({ mechanism: "", trialStatusForTargets: undefined })], [drug({ mechanism: "second", trialStatusForTargets: "t2" })]])
    assert.equal(merged[0].mechanism, "second")
    assert.equal(merged[0].trialStatusForTargets, "t2")
  })

  it("keeps different drugs apart and keeps first-seen order", () => {
    const merged = mergeDrugs([[drug({ drug: "B" }), drug({ drug: "A" })], [drug({ drug: "C" })]])
    assert.deepEqual(merged.map((d) => d.drug), ["B", "A", "C"])
  })
})

describe("extractDrugMap", () => {
  it("extracts each section on its own, at the same time, and merges the results", async () => {
    let inFlight = 0
    let peak = 0
    http.restore()
    http = mockFetch({
      "api.openai.com": async ({ init }) => {
        inFlight++
        peak = Math.max(peak, inFlight)
        await new Promise((resolve) => setImmediate(resolve))
        inFlight--
        return modelReply(init, () => undefined)
      },
    })

    const out = await extractDrugMap(
      [
        { label: "drug analysis", text: section("A", 2) },
        { label: "repurposing", text: section("B", 2) },
      ],
      "breast cancer",
      PROTEINS,
      options
    )
    assert.equal(requests.length, 2)
    assert.equal(peak, 2, "both calls were in flight together")
    assert.ok(requests.some((t) => t.includes("A-p0 ") && !t.includes("B-p")), "a call sees only its own section")
    assert.ok(requests.some((t) => t.includes("B-p0 ") && !t.includes("A-p")))
    assert.deepEqual(names(out.drugs), ["A-drug0", "A-drug1", "B-drug0", "B-drug1"])
    assert.deepEqual(out.warnings, [])
  })

  it("sends the disease and the protein list with every call", async () => {
    fakeModel(() => undefined)
    await extractDrugMap([{ label: "drug analysis", text: section("A", 1) }], "breast cancer", PROTEINS, options)
    const body = JSON.parse(String(http.calls[0].init.body))
    assert.ok(body.messages[1].content.includes("breast cancer"))
    assert.ok(body.messages[1].content.includes("EGFR, KRAS"))
  })

  it("a reply that was cut off is retried on the two halves of that text, and nothing is lost", async () => {
    fakeModel((text) => (text.length > 4000 ? "truncate" : undefined))
    const out = await extractDrugMap([{ label: "drug analysis", text: section("A", 16) }], "d", PROTEINS, options)
    assert.equal(requests.length, 3, "the whole text, then each half")
    assert.deepEqual(names(out.drugs), Array.from({ length: 16 }, (_, i) => `A-drug${i}`).sort())
    assert.deepEqual(out.warnings, [], "a recovered cut-off is not a warning")
  })

  it("splits again when a half is cut off too, up to a limit, then warns for that piece", async () => {
    fakeModel((text) => (text.length > 1000 ? "truncate" : undefined))
    const out = await extractDrugMap([{ label: "drug analysis", text: section("A", 16) }], "d", PROTEINS, options)
    assert.equal(requests.length, 7, "1 whole + 2 halves + 4 quarters, then it stops")
    assert.deepEqual(out.drugs, [])
    assert.equal(out.warnings.length, 4, "one warning per piece that could not be recovered")
    assert.ok(out.warnings.every((w) => w.source === "openai" && w.kind === "truncated" && w.message.includes("drug analysis")))
  })

  it("keeps the pieces that worked when one piece never fits", async () => {
    fakeModel((text) => (text.includes("A-p15 ") ? "truncate" : undefined))
    const out = await extractDrugMap([{ label: "drug analysis", text: section("A", 16) }], "d", PROTEINS, options)
    assert.deepEqual(names(out.drugs), Array.from({ length: 12 }, (_, i) => `A-drug${i}`).sort(), "p12 to p15 were the last quarter")
    assert.equal(out.warnings.length, 1)
  })

  it("a section that cannot be split is a warning, not a loop", async () => {
    fakeModel(() => "truncate")
    const oneLine = await extractDrugMap([{ label: "repurposing", text: "x".repeat(3000) }], "d", PROTEINS, options)
    assert.equal(oneLine.warnings.length, 1)
    const short = await extractDrugMap([{ label: "repurposing", text: section("B", 2) }], "d", PROTEINS, options)
    assert.equal(short.warnings.length, 1)
    assert.equal(requests.length, 2, "one request each, no re-ask")
  })

  it("one section failing keeps the other section's drugs", async () => {
    fakeModel((text) => (text.includes("A-p0 ") ? "refuse" : undefined))
    const out = await extractDrugMap(
      [
        { label: "drug analysis", text: section("A", 2) },
        { label: "repurposing", text: section("B", 2) },
      ],
      "d",
      PROTEINS,
      options
    )
    assert.deepEqual(names(out.drugs), ["B-drug0", "B-drug1"])
    assert.equal(out.warnings.length, 1)
    assert.equal(out.warnings[0].kind, "refusal")
    assert.ok(out.warnings[0].message.includes("drug analysis"))
  })

  it("an API outage throws instead of reading as 'no drugs'", async () => {
    http.restore()
    http = mockFetch({ "api.openai.com": res(503) })
    await assert.rejects(
      extractDrugMap([{ label: "drug analysis", text: section("A", 2) }], "d", PROTEINS, options),
      (e: unknown) => e instanceof ExternalCallError && e.source === "openai"
    )
  })

  it("makes no call for an empty section", async () => {
    fakeModel(() => undefined)
    const out = await extractDrugMap(
      [
        { label: "drug analysis", text: "   " },
        { label: "repurposing", text: section("B", 1) },
      ],
      "d",
      PROTEINS,
      options
    )
    assert.equal(requests.length, 1)
    assert.deepEqual(names(out.drugs), ["B-drug0"])
    const none = await extractDrugMap([{ label: "drug analysis", text: "" }], "d", PROTEINS, options)
    assert.deepEqual(none, { drugs: [], warnings: [] })
    assert.equal(requests.length, 1)
  })

  it("maps the reply to the app's drug shape", async () => {
    fakeModel(() => ({ category: "repurposing_candidate", proteins: ["KRAS"] }))
    const out = await extractDrugMap([{ label: "repurposing", text: section("B", 1) }], "d", PROTEINS, options)
    assert.deepEqual(out.drugs, [
      {
        drug: "B-drug0",
        proteins: ["KRAS"],
        mechanism: "m",
        fdaStatus: "approved",
        category: "repurposing_candidate",
        trialStatusForTargets: "none",
      },
    ])
  })
})

describe("the extraction schema asks for short text", () => {
  it("limits the three free-text fields in words, because the reply is capped at 8,000 tokens", () => {
    const json = toStrictJsonSchema(DrugProteinMapSchema) as {
      properties: { drugs: { items: { properties: Record<string, { description?: string }> } } }
    }
    const fields = json.properties.drugs.items.properties
    for (const key of ["mechanism", "fda_status", "trial_status_for_targets"]) {
      assert.match(fields[key].description ?? "", /at most \d+ words/, key)
    }
  })
})
