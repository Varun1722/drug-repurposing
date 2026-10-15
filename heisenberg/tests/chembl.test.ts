// How far a compound has got in drug development, from ChEMBL (lib/chembl.ts).
// The responses are real ones recorded from the ChEMBL API for a few drugs
// (tests/fixtures/chembl.json), so the tests run against what ChEMBL actually
// sends. fetch is mocked; no waiting.
//
// Contract: a stage comes from ChEMBL's max_phase; "withdrawn" only when the
// drug was withdrawn worldwide or in the United States (ChEMBL's withdrawn flag
// alone also covers a single country, and thalidomide is on sale today); a
// flagged drug with no withdrawal detail at all is "unknown" (trovafloxacin was
// pulled, ergotamine is still sold, and ChEMBL lists no country for either); a
// compound ChEMBL does not know is "unknown", not "preclinical"; a failed call
// throws, so an outage is not read as an answer.

import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { afterEach, beforeEach, describe, it } from "node:test"

import { lookupStage, stageFromMaxPhase, stageLookup } from "../lib/chembl"
import { ExternalCallError } from "../lib/retry"
import type { Stage } from "../lib/types"
import { instantTimers, mockFetch, res } from "./_fetch"

type Recorded = { key: string; molecule: { molecules: { molecule_chembl_id: string }[] }; warning?: unknown }
const recorded: Record<string, Recorded> = JSON.parse(
  readFileSync(new URL("./fixtures/chembl.json", import.meta.url), "utf8")
)
const keyOf = (drug: string) => recorded[drug].key
const NOTHING = { molecules: [], page_meta: { limit: 20, next: null, offset: 0, previous: null, total_count: 0 } }

let timers: ReturnType<typeof instantTimers>
let http: ReturnType<typeof mockFetch>
beforeEach(() => {
  timers = instantTimers()
  http = mockFetch({})
})
afterEach(() => {
  http.restore()
  timers.restore()
})

const route = (routes: Parameters<typeof mockFetch>[0]) => {
  http.restore()
  http = mockFetch(routes)
}

/** ChEMBL as recorded: the molecule and drug_warning endpoints, looked up by the key or id asked for. */
function serveRecorded() {
  route({
    "molecule.json": ({ url }) => {
      const key = new URL(url).searchParams.get("molecule_structures__standard_inchi_key")
      return res(200, Object.values(recorded).find((r) => r.key === key)?.molecule ?? NOTHING)
    },
    "drug_warning.json": ({ url }) => {
      const id = new URL(url).searchParams.get("molecule_chembl_id")
      const found = Object.values(recorded).find((r) => r.molecule.molecules[0]?.molecule_chembl_id === id)
      return res(200, found?.warning ?? { drug_warnings: [] })
    },
  })
}

describe("stageFromMaxPhase", () => {
  const cases: [string | number | null | undefined, Stage][] = [
    ["4.0", "approved"],
    [4, "approved"],
    ["3.0", "phase-3"],
    ["2.0", "phase-2"],
    ["1.0", "phase-1"],
    ["0.5", "phase-1"], // early phase 1 (microdose / exploratory studies)
    [null, "preclinical"], // in ChEMBL with bioactivity data, but no record of a trial
    ["-1.0", "unknown"], // a clinical candidate whose phase ChEMBL could not work out
    [undefined, "unknown"], // field missing: do not read it as preclinical
    ["not a number", "unknown"],
  ]
  for (const [input, expected] of cases) {
    it(`${JSON.stringify(input)} -> ${expected}`, () => assert.equal(stageFromMaxPhase(input), expected))
  }
})

describe("lookupStage (recorded ChEMBL responses)", () => {
  const cases: [string, Stage, string][] = [
    ["metformin", "approved", "max_phase 4"],
    ["telaglenastat", "phase-2", "max_phase 2"],
    ["LY294002", "unknown", "max_phase -1"],
    ["SB203580", "preclinical", "max_phase null"],
    ["rofecoxib", "withdrawn", "flagged, withdrawn Worldwide"],
    ["troglitazone", "withdrawn", "flagged, one warning lists 'Peru; United Kingdom; United States'"],
    ["thalidomide", "approved", "flagged, but only withdrawn in Finland (1963); on sale today"],
    ["trovafloxacin", "unknown", "flagged, but ChEMBL lists no withdrawal country: cannot tell, so not 'approved'"],
  ]
  for (const [drug, expected, why] of cases) {
    it(`${drug} -> ${expected} (${why})`, async () => {
      serveRecorded()
      assert.equal(await lookupStage(keyOf(drug)), expected)
    })
  }

  it("asks for the molecule by its full InChIKey, and checks withdrawals only for flagged drugs", async () => {
    serveRecorded()
    await lookupStage(keyOf("metformin"))
    assert.equal(http.count("molecule.json"), 1)
    assert.ok(http.calls[0].url.includes(`molecule_structures__standard_inchi_key=${keyOf("metformin")}`))
    assert.equal(http.count("drug_warning"), 0)

    await lookupStage(keyOf("rofecoxib"))
    assert.equal(http.count("drug_warning"), 1)
  })

  it("a compound ChEMBL does not have is 'unknown', not 'preclinical'", async () => {
    serveRecorded()
    assert.equal(await lookupStage("AAAAAAAAAAAAAA-BBBBBBBBBB-N"), "unknown")
  })

  it("a 404 is also 'not found'", async () => {
    route({ "molecule.json": res(404) })
    assert.equal(await lookupStage("AAAAAAAAAAAAAA-BBBBBBBBBB-N"), "unknown")
    assert.equal(http.calls.length, 1)
  })

  it("something that is not an InChIKey is 'unknown' and sends nothing", async () => {
    for (const bad of ["", "metformin", "../molecule", "AAAAAAAAAAAAAA-BBBBBBBBBB-N&limit=1000"]) {
      assert.equal(await lookupStage(bad), "unknown", JSON.stringify(bad))
    }
    assert.equal(http.calls.length, 0)
  })

  it("a ChEMBL failure throws instead of reading as 'unknown'", async () => {
    route({ "molecule.json": res(503) })
    await assert.rejects(
      lookupStage(keyOf("metformin")),
      (e: unknown) => e instanceof ExternalCallError && e.source === "chembl" && e.kind === "server"
    )
    assert.equal(http.count("molecule.json"), 3, "a transient failure is retried")
  })

  it("withdrawal records that name no country are no detail either", async () => {
    route({
      "molecule.json": res(200, recorded.rofecoxib.molecule),
      "drug_warning.json": res(200, { drug_warnings: [{ warning_country: null }, { warning_country: "" }] }),
    })
    assert.equal(await lookupStage(keyOf("rofecoxib")), "unknown")
  })

  it("a failed withdrawal check throws too, rather than calling a flagged drug approved", async () => {
    route({
      "molecule.json": res(200, recorded.rofecoxib.molecule),
      "drug_warning.json": res(500),
    })
    await assert.rejects(lookupStage(keyOf("rofecoxib")), (e: unknown) => e instanceof ExternalCallError && e.source === "chembl")
  })
})

describe("stageLookup (one per request)", () => {
  const key = (i: number) => `${String.fromCharCode(65 + i).repeat(14)}-BBBBBBBBBB-N`

  it("asks ChEMBL once per compound, however many proteins want it", async () => {
    serveRecorded()
    const look = stageLookup()
    const stages = await Promise.all([look(keyOf("metformin")), look(keyOf("metformin")), look(keyOf("metformin"))])
    assert.deepEqual(stages, ["approved", "approved", "approved"])
    assert.equal(http.count("molecule.json"), 1)
  })

  it("never has more than `concurrency` requests in flight", async () => {
    let inFlight = 0
    let peak = 0
    route({
      "molecule.json": async () => {
        inFlight++
        peak = Math.max(peak, inFlight)
        await new Promise((resolve) => setImmediate(resolve))
        inFlight--
        return res(200, NOTHING)
      },
    })
    const look = stageLookup(2)
    await Promise.all([0, 1, 2, 3, 4, 5].map((i) => look(key(i))))
    assert.equal(http.count("molecule.json"), 6)
    assert.equal(peak, 2)
  })

  it("after 3 failed lookups it stops calling ChEMBL, so an outage cannot use up the route's time", async () => {
    route({ "molecule.json": res(503) })
    const look = stageLookup(1)
    const outcomes = await Promise.allSettled([0, 1, 2, 3, 4, 5].map((i) => look(key(i))))
    assert.ok(outcomes.every((o) => o.status === "rejected"))
    assert.ok(outcomes.every((o) => o.status === "rejected" && o.reason instanceof ExternalCallError))
    assert.equal(http.count("molecule.json"), 9, "3 lookups x 3 attempts, then no more requests")
  })
})
