// The routes that build ligands (/api/structures, /api/expand) put the ChEMBL
// development stage on each one, so the badge and the report do not have to
// guess it from the literature review's wording. The real route handlers run
// against mocked PubChem, RCSB and ChEMBL; ChEMBL's replies are recorded ones.

import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { afterEach, beforeEach, describe, it } from "node:test"

import { POST as expand } from "../app/api/expand/route"
import { POST as structures } from "../app/api/structures/route"
import { readSSEStream } from "../lib/sse"
import type { DockingTarget, ReviewDrug } from "../lib/types"
import { instantTimers, mockFetch, res } from "./_fetch"

type Recorded = { key: string; molecule: unknown }
const recorded: Record<string, Recorded> = JSON.parse(
  readFileSync(new URL("./fixtures/chembl.json", import.meta.url), "utf8")
)
const NOTHING = { molecules: [], page_meta: { limit: 20, next: null, offset: 0, previous: null, total_count: 0 } }

let timers: ReturnType<typeof instantTimers>
let http: ReturnType<typeof mockFetch>
let log: typeof console.log
beforeEach(() => {
  timers = instantTimers()
  http = mockFetch({})
  log = console.log
  console.log = () => {}
})
afterEach(() => {
  http.restore()
  timers.restore()
  console.log = log
})

/** PubChem's property row for a compound, with the InChIKey of a recorded drug. */
const props = (cid: number, smiles: string, key: string) => ({
  CID: cid,
  CanonicalSMILES: smiles,
  IUPACName: `compound ${cid}`,
  MolecularFormula: "C",
  InChIKey: key,
})
const table = (...rows: object[]) => res(200, { PropertyTable: { Properties: rows } })

/** Recorded ChEMBL answers by key; a key listed in `fail` gets a 503; any other key is not in ChEMBL. */
function chembl(fail: string[] = []) {
  return ({ url }: { url: string }) => {
    const key = new URL(url).searchParams.get("molecule_structures__standard_inchi_key") ?? ""
    if (fail.includes(key)) return res(503)
    return res(200, Object.values(recorded).find((r) => r.key === key)?.molecule ?? NOTHING)
  }
}

const reviewDrug = (drug: string, category: string, fdaStatus = "Phase 2 trial in the literature"): ReviewDrug => ({
  drug,
  proteins: ["EGFR"],
  mechanism: "m",
  fdaStatus,
  category,
})

async function runStructures(drugs: ReviewDrug[], proteins = ["EGFR"]) {
  const response = await structures(
    new Request("http://test/api/structures", { method: "POST", body: JSON.stringify({ proteins, drugs }) })
  )
  const events: Record<string, unknown>[] = []
  await readSSEStream(response, (e) => void events.push(e))
  return events
}
const targetsOf = (events: Record<string, unknown>[]) => events.filter((e) => e.type === "target").map((e) => e.target as DockingTarget)

describe("/api/structures", () => {
  it("puts the ChEMBL stage and the review's category on each ligand, and leaves the review's status text alone", async () => {
    http.restore()
    http = mockFetch({
      "search.rcsb.org": res(204),
      "compound/name/": table(props(4091, "CN(C)C(=N)N=C(N)N", recorded.metformin.key)),
      "molecule.json": chembl(),
    })
    const [target] = targetsOf(await runStructures([reviewDrug("metformin", "mainstream")]))
    assert.equal(target.ligands.length, 1)
    const [ligand] = target.ligands
    assert.equal(ligand.stage, "approved")
    assert.equal(ligand.category, "mainstream")
    assert.equal(ligand.fdaStatus, "Phase 2 trial in the literature")
  })

  it("a ChEMBL failure keeps the drug in the docking run as 'unknown' and warns, naming ChEMBL", async () => {
    http.restore()
    http = mockFetch({
      "search.rcsb.org": res(204),
      "compound/name/metformin": table(props(4091, "CCN", recorded.metformin.key)),
      "compound/name/ibuprofen": table(props(3672, "CCO", recorded.ibuprofen.key)),
      "molecule.json": chembl([recorded.ibuprofen.key]),
    })
    const events = await runStructures([reviewDrug("metformin", "mainstream"), reviewDrug("ibuprofen", "repurposing_candidate")])
    const stages = Object.fromEntries(targetsOf(events)[0].ligands.map((l) => [l.name, l.stage]))
    assert.deepEqual(stages, { metformin: "approved", ibuprofen: "unknown" })
    const warnings = events.filter((e) => e.type === "warning").map((e) => e.warning as { source: string })
    assert.ok(warnings.some((w) => w.source === "chembl"))
  })

  it("asks ChEMBL once for a drug that goes to two proteins", async () => {
    http.restore()
    http = mockFetch({
      "search.rcsb.org": res(204),
      "compound/name/": table(props(4091, "CCN", recorded.metformin.key)),
      "molecule.json": chembl(),
    })
    const drug = { ...reviewDrug("metformin", "mainstream"), proteins: ["EGFR", "KRAS"] }
    await runStructures([drug], ["EGFR", "KRAS"])
    assert.equal(http.count("molecule.json"), 1)
  })
})

describe("/api/expand", () => {
  const existing: DockingTarget = { protein: "EGFR", pdbId: "1ABC", pdbContentB64: "AAAA", ligands: [] }
  const run = async (decision: object) => {
    const response = await expand(
      new Request("http://test/api/expand", {
        method: "POST",
        body: JSON.stringify({ decision, proteins: ["EGFR"], existingTargets: [existing] }),
      })
    )
    return (await response.json()) as { newTargets: DockingTarget[]; warnings: { source: string }[] }
  }

  it("class expansion: the looked-up drug gets its ChEMBL stage", async () => {
    http.restore()
    http = mockFetch({
      "compound/name/": table(props(4091, "CCN", recorded.metformin.key)),
      "molecule.json": chembl(),
    })
    const { newTargets } = await run({ action: "expand_class", rationale: "", hypothesis: "", drugClass: "biguanides", drugNames: ["metformin"] })
    assert.equal(newTargets[0].ligands[0].stage, "approved")
  })

  it("3D-similar expansion: each similar compound gets its own stage, 'unknown' when ChEMBL has no record", async () => {
    http.restore()
    http = mockFetch({
      fastsimilarity_3d: res(200, { IdentifierList: { CID: [2, 3] } }),
      "compound/cid/2,3": table(
        props(2, "CCO", recorded.telaglenastat.key),
        props(3, "CCN", "AAAAAAAAAAAAAA-BBBBBBBBBB-N")
      ),
      "molecule.json": chembl(),
    })
    const { newTargets } = await run({ action: "expand_3d_similar", rationale: "", hypothesis: "", seedCids: [1] })
    assert.deepEqual(newTargets[0].ligands.map((l) => l.stage), ["phase-2", "unknown"])
  })

  it("a ChEMBL failure keeps the compound as 'unknown' and says ChEMBL failed", async () => {
    http.restore()
    http = mockFetch({
      "compound/name/": table(props(4091, "CCN", recorded.metformin.key)),
      "molecule.json": res(503),
    })
    const { newTargets, warnings } = await run({ action: "expand_class", rationale: "", hypothesis: "", drugClass: "biguanides", drugNames: ["metformin"] })
    assert.equal(newTargets[0].ligands[0].stage, "unknown")
    assert.ok(warnings.some((w) => w.source === "chembl"))
  })
})
