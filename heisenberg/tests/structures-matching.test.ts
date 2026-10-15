// /api/structures: which drugs it looks up and docks for each protein, on the real
// route handler with a mocked network (lib/protein-match.ts has the rules and the
// real-run cases). Wrong pairs cost real work: each one is a PubChem lookup here and
// a docking run later, and it shows in the ranking as a binder for the wrong protein.

import assert from "node:assert/strict"
import { afterEach, beforeEach, describe, it } from "node:test"

import { POST as structures } from "../app/api/structures/route"
import { readSSEStream } from "../lib/sse"
import type { DockingTarget, ReviewDrug } from "../lib/types"
import { instantTimers, mockFetch, res } from "./_fetch"

let timers: ReturnType<typeof instantTimers>
let http: ReturnType<typeof mockFetch>
let log: typeof console.log
beforeEach(() => {
  timers = instantTimers()
  log = console.log
  console.log = () => {}
  // No structures at RCSB; every drug name is a PubChem compound without an InChIKey (so no ChEMBL call).
  http = mockFetch({
    "search.rcsb.org": res(204),
    "compound/name/": ({ url }) => {
      const name = decodeURIComponent(url.split("compound/name/")[1].split("/")[0])
      return res(200, { PropertyTable: { Properties: [{ CID: name.length, CanonicalSMILES: `C${name}`, IUPACName: name, MolecularFormula: "C" }] } })
    },
  })
})
afterEach(() => {
  http.restore()
  timers.restore()
  console.log = log
})

const drug = (name: string, proteins: string[]): ReviewDrug => ({ drug: name, proteins, mechanism: "m", fdaStatus: "s", category: "mainstream" })

async function run(proteins: string[], drugs: ReviewDrug[]) {
  const response = await structures(
    new Request("http://test/api/structures", { method: "POST", body: JSON.stringify({ proteins, drugs }) })
  )
  const events: Record<string, unknown>[] = []
  await readSSEStream(response, (e) => void events.push(e))
  const complete = events.find((e) => e.type === "complete") as {
    targets: DockingTarget[]
    proteinsWithoutDrugs: string[]
    drugsWithoutProtein: string[]
  }
  return { events, complete, ligands: Object.fromEntries(complete.targets.map((t) => [t.protein, t.ligands.map((l) => l.name)])) }
}

describe("/api/structures drug matching", () => {
  it("docks a drug only against the protein it is listed for: a HER2 drug is not an ER drug", async () => {
    const { ligands } = await run(["ER", "HER2"], [drug("lapatinib", ["HER2"]), drug("tamoxifen", ["ER"])])
    assert.deepEqual(ligands, { ER: ["tamoxifen"], HER2: ["lapatinib"] })
    assert.equal(http.count("compound/name/lapatinib"), 1, "looked up once, not once for the wrong protein too")
    assert.equal(http.count("compound/name/tamoxifen"), 1)
  })

  it("a protein with no drug is not docked: it gets no ligands and no lookups, and is reported", async () => {
    const { events, ligands, complete } = await run(["EGFR", "BRCA2"], [drug("gefitinib", ["EGFR"])])
    assert.deepEqual(ligands, { EGFR: ["gefitinib"], BRCA2: [] })
    assert.deepEqual(complete.proteinsWithoutDrugs, ["BRCA2"])
    assert.equal(http.count("compound/name/"), 1, "the old fallback looked up every drug for it")
    const messages = events.filter((e) => e.type === "progress").map((e) => String(e.message))
    assert.ok(messages.some((m) => m.includes("BRCA2") && m.includes("not be docked")), "the progress line says so")
    assert.ok(!messages.some((m) => /for 0 drugs/.test(m)), "and does not say it is looking up 0 drugs")
  })

  it("a drug listed for none of the proteins is not docked, and is reported", async () => {
    const { ligands, complete } = await run(["EGFR"], [drug("metformin", []), drug("gefitinib", ["EGFR"])])
    assert.deepEqual(ligands, { EGFR: ["gefitinib"] })
    assert.deepEqual(complete.drugsWithoutProtein, ["metformin"])
    assert.equal(http.count("compound/name/metformin"), 0)
  })

  it("matches names whatever their case or spacing", async () => {
    const { ligands } = await run(["KRAS G12C"], [drug("sotorasib", [" kras  g12c"])])
    assert.deepEqual(ligands, { "KRAS G12C": ["sotorasib"] })
  })

  it("still reports every target and the warnings in the complete event", async () => {
    const { complete } = await run(["EGFR"], [drug("gefitinib", ["EGFR"])])
    assert.equal(complete.targets.length, 1)
    assert.deepEqual(complete.proteinsWithoutDrugs, [])
    assert.deepEqual(complete.drugsWithoutProtein, [])
  })
})
