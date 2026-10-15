// Failure handling of the external-API clients: PubChem, RCSB PDB, Exa and the
// GNINA job API. Contract under test: "nothing found" is a normal result,
// while a call that fails even after retries throws ExternalCallError, so an
// outage is never mistaken for an empty answer. fetch is mocked; no waiting.

import assert from "node:assert/strict"
import { afterEach, beforeEach, describe, it } from "node:test"

import { pollDockJob, submitDockJob } from "../lib/dock-jobs"
import { webSearch } from "../lib/exa"
import { downloadPdb, getPdbMetadata, pickBestStructure, searchPdb } from "../lib/pdb"
import { lookupSmiles, search3dSimilar, searchCompoundsForTarget } from "../lib/pubchem"
import { ExternalCallError } from "../lib/retry"
import { describeWarnings, errorMessageFrom } from "../lib/warnings"
import { instantTimers, mockFetch, networkError, res } from "./_fetch"

let timers: ReturnType<typeof instantTimers>
let http: ReturnType<typeof mockFetch>
let logs: { log: typeof console.log; warn: typeof console.warn }

beforeEach(() => {
  timers = instantTimers()
  http = mockFetch({})
  logs = { log: console.log, warn: console.warn }
  console.log = () => {}
  console.warn = () => {}
})
afterEach(() => {
  http.restore()
  timers.restore()
  console.log = logs.log
  console.warn = logs.warn
})

const route = (routes: Parameters<typeof mockFetch>[0]) => {
  http.restore()
  http = mockFetch(routes)
}
const isExternal = (source: string, kind?: string) => (e: unknown) =>
  e instanceof ExternalCallError && e.source === source && (kind === undefined || e.kind === kind)

const props = (cid: number, smiles = "CCO") => ({ CID: cid, CanonicalSMILES: smiles, IUPACName: `c${cid}`, MolecularFormula: "C" })
const table = (...rows: object[]) => ({ PropertyTable: { Properties: rows } })

describe("pubchem.lookupSmiles", () => {
  it("finds a compound", async () => {
    route({ "compound/name/": res(200, table(props(5394, "CC1=NN=C2N1C=NC2=O"))) })
    const r = await lookupSmiles("temozolomide")
    assert.deepEqual([r?.cid, r?.smiles], [5394, "CC1=NN=C2N1C=NC2=O"])
  })

  it("404 is 'not found': null, no retry", async () => {
    route({ "compound/name/": res(404) })
    assert.equal(await lookupSmiles("zoldonrasib"), null)
    assert.equal(http.calls.length, 1)
  })

  it("400 (a name PubChem cannot parse) is also 'not found', not an outage", async () => {
    route({ "compound/name/": res(400) })
    assert.equal(await lookupSmiles("N/A"), null)
    assert.equal(http.calls.length, 1)
  })

  it("a transient 503 is retried", async () => {
    route({ "compound/name/": [res(503), res(200, table(props(1)))] })
    assert.equal((await lookupSmiles("x1234"))?.cid, 1)
    assert.equal(http.calls.length, 2)
  })

  it("a PubChem failure throws instead of returning null, and tries no aliases", async () => {
    route({ "compound/name/": res(503) })
    await assert.rejects(lookupSmiles("daraxonrasib (RMC-6236)"), isExternal("pubchem", "server"))
    assert.equal(http.calls.length, 3) // one name, three attempts
    assert.ok(http.calls.every((c) => c.url.includes("daraxonrasib%20(RMC-6236)")))
  })

  it("a network failure throws too", async () => {
    route({ "compound/name/": networkError() })
    await assert.rejects(lookupSmiles("temozolomide"), isExternal("pubchem", "connection"))
  })

  it("tries name variants only after a genuine 'not found'", async () => {
    route({
      "name/Sotorasib%20(AMG%20510)": res(404),
      "name/Sotorasib/": res(200, table(props(137278711, "CC"))),
    })
    assert.equal((await lookupSmiles("Sotorasib (AMG 510)"))?.cid, 137278711)
  })

  it("looks CID_ names up by CID", async () => {
    route({ "compound/cid/42/": res(200, table(props(42))) })
    assert.equal((await lookupSmiles("CID_42"))?.cid, 42)
    route({ "compound/cid/43/": res(404) })
    assert.equal(await lookupSmiles("CID_43"), null)
    route({ "compound/cid/44/": res(500) })
    await assert.rejects(lookupSmiles("CID_44"), isExternal("pubchem"))
  })
})

describe("pubchem.searchCompoundsForTarget", () => {
  const cids = (...c: number[]) => res(200, { InformationList: { Information: [{ CID: c }] } })

  it("returns compounds", async () => {
    route({ "assay/target": cids(100, 200), "compound/cid/": res(200, table(props(100), props(200))) })
    assert.deepEqual((await searchCompoundsForTarget("KRAS G12D")).map((c) => c.cid), [100, 200])
    assert.match(http.calls[0].url, /genesymbol\/KRAS\//)
  })

  it("no compounds anywhere is [] (404s are results)", async () => {
    route({ "assay/target": res(404), "gene/symbol": res(404) })
    assert.deepEqual(await searchCompoundsForTarget("NOPE"), [])
  })

  it("an assay-search failure falls back to the gene-symbol search", async () => {
    route({ "assay/target": res(503), "gene/symbol": cids(300), "compound/cid/": res(200, table(props(300))) })
    assert.deepEqual((await searchCompoundsForTarget("MGMT")).map((c) => c.cid), [300])
  })

  it("a failed search with nothing found throws, so it is not read as 'no compounds'", async () => {
    route({ "assay/target": res(503), "gene/symbol": res(404) })
    await assert.rejects(searchCompoundsForTarget("MGMT"), isExternal("pubchem", "server"))
  })

  it("a property-fetch failure throws", async () => {
    route({ "assay/target": cids(100), "compound/cid/": res(500) })
    await assert.rejects(searchCompoundsForTarget("KRAS"), isExternal("pubchem"))
  })
})

describe("pubchem.search3dSimilar", () => {
  it("returns similar compounds", async () => {
    route({ fastsimilarity_3d: res(200, { IdentifierList: { CID: [2, 3] } }), "compound/cid/2,3": res(200, table(props(2), props(3))) })
    assert.deepEqual((await search3dSimilar(1)).map((c) => c.cid), [2, 3])
  })

  it("no 3D conformer (404) is []", async () => {
    route({ fastsimilarity_3d: res(404) })
    assert.deepEqual(await search3dSimilar(1), [])
  })

  it("a failed search throws, not []", async () => {
    route({ fastsimilarity_3d: res(500) })
    await assert.rejects(search3dSimilar(1), isExternal("pubchem", "server"))
  })

  it("a failed property fetch throws", async () => {
    route({ fastsimilarity_3d: res(200, { IdentifierList: { CID: [2] } }), "compound/cid/": res(500) })
    await assert.rejects(search3dSimilar(1), isExternal("pubchem"))
  })
})

describe("pdb", () => {
  it("searchPdb maps hits", async () => {
    route({ "search.rcsb.org": res(200, { result_set: [{ identifier: "6GJ8", score: 1 }] }) })
    assert.deepEqual(await searchPdb("KRAS"), [{ pdbId: "6GJ8", score: 1 }])
  })

  it("searchPdb: 204 No Content means no matches", async () => {
    route({ "search.rcsb.org": res(204) })
    assert.deepEqual(await searchPdb("NOPE"), [])
    assert.equal(http.calls.length, 1)
  })

  it("searchPdb: a failure throws instead of returning []", async () => {
    route({ "search.rcsb.org": res(500) })
    await assert.rejects(searchPdb("KRAS"), isExternal("rcsb", "server"))
    assert.equal(http.calls.length, 3)
  })

  it("getPdbMetadata: parses, 404 is empty metadata, failure throws", async () => {
    route({ "/entry/6GJ8": res(200, { struct: { title: "t" }, rcsb_entry_info: { resolution_combined: [1.9], nonpolymer_entity_count: 2 } }) })
    assert.deepEqual(await getPdbMetadata("6GJ8"), { resolution: 1.9, hasLigands: true, title: "t" })
    route({ "/entry/NOPE": res(404) })
    assert.deepEqual(await getPdbMetadata("NOPE"), { hasLigands: false, title: "" })
    route({ "/entry/FAIL": res(503) })
    await assert.rejects(getPdbMetadata("FAIL"), isExternal("rcsb"))
  })

  const meta = (resolution: number, ligands: number) =>
    res(200, { struct: { title: "" }, rcsb_entry_info: { resolution_combined: [resolution], nonpolymer_entity_count: ligands } })

  it("pickBestStructure prefers a ligand, then resolution", async () => {
    route({ "/entry/AAA": meta(1.5, 0), "/entry/BBB": meta(2.0, 1) })
    assert.equal((await pickBestStructure([{ pdbId: "AAA", score: 1 }, { pdbId: "BBB", score: 1 }]))?.pdbId, "BBB")
  })

  it("pickBestStructure skips a candidate it cannot judge", async () => {
    route({ "/entry/BAD": res(500), "/entry/GOOD": meta(2.0, 1) })
    assert.equal((await pickBestStructure([{ pdbId: "BAD", score: 1 }, { pdbId: "GOOD", score: 1 }]))?.pdbId, "GOOD")
  })

  it("pickBestStructure throws when no candidate can be judged", async () => {
    route({ "/entry/": res(500) })
    await assert.rejects(pickBestStructure([{ pdbId: "A", score: 1 }, { pdbId: "B", score: 1 }]), isExternal("rcsb"))
  })

  it("downloadPdb: base64 of the .pdb file", async () => {
    route({ "6GJ8.pdb": res(200, "HEADER") })
    assert.equal(await downloadPdb("6gj8"), Buffer.from("HEADER").toString("base64"))
  })

  it("downloadPdb: falls back to .cif when there is no .pdb", async () => {
    route({ "9E3S.pdb": res(404), "9E3S.cif": res(200, "data_9E3S") })
    assert.equal(await downloadPdb("9E3S"), Buffer.from("data_9E3S").toString("base64"))
  })

  it("downloadPdb: null when neither exists; a failure throws", async () => {
    route({ "0000.pdb": res(404), "0000.cif": res(404) })
    assert.equal(await downloadPdb("0000"), null)
    route({ "1111.pdb": res(502) })
    await assert.rejects(downloadPdb("1111"), isExternal("rcsb", "server"))
  })
})

describe("exa.webSearch", () => {
  beforeEach(() => void (process.env.EXA_API_KEY = "test-key"))

  it("returns results, retrying a transient failure", async () => {
    route({ "api.exa.ai": [res(503), res(200, { results: [{ title: "t", url: "u", text: "x" }] })] })
    assert.deepEqual(await webSearch("q"), [{ title: "t", url: "u", text: "x" }])
    assert.equal(http.calls.length, 2)
  })

  it("throws ExternalCallError when Exa keeps failing, and does not retry a 4xx", async () => {
    route({ "api.exa.ai": res(500) })
    await assert.rejects(webSearch("q"), isExternal("exa", "server"))
    route({ "api.exa.ai": res(401) })
    await assert.rejects(webSearch("q"), isExternal("exa", "auth"))
    assert.equal(http.calls.length, 1)
  })

  it("sends the key in a header, never in the URL", async () => {
    route({ "api.exa.ai": res(200, { results: [] }) })
    await webSearch("q")
    assert.ok(!http.calls[0].url.includes("test-key"))
    assert.equal((http.calls[0].init.headers as Record<string, string>)["x-api-key"], "test-key")
  })
})

describe("dock-jobs", () => {
  beforeEach(() => void (process.env.GNINA_JOB_API_URL = "https://gnina.test/"))

  it("submit returns the job id", async () => {
    route({ "/submit": res(200, { jobId: "j1" }) })
    assert.deepEqual(await submitDockJob([], 1), { jobId: "j1" })
    assert.equal(http.calls[0].url, "https://gnina.test/submit")
  })

  it("submit is never repeated: a retry could start the docking twice", async () => {
    route({ "/submit": res(503) })
    await assert.rejects(submitDockJob([], 1), isExternal("gnina", "server"))
    assert.equal(http.calls.length, 1)
  })

  it("poll fails once and leaves the retry to its caller's polling loop", async () => {
    route({ "/status/j1": res(503) })
    await assert.rejects(pollDockJob("j1", 0), isExternal("gnina", "server"))
    assert.equal(http.calls.length, 1)
  })

  it("poll passes the cursor", async () => {
    route({ "/status/": res(200, { events: [], nextIndex: 7, done: false }) })
    assert.equal((await pollDockJob("a b", 5)).nextIndex, 7)
    assert.match(http.calls[0].url, /status\/a%20b\?since=5$/)
  })

  it("keeps the job API's address out of its errors (it is unauthenticated, and errors reach the browser)", async () => {
    route({ "/submit": res(400, { detail: "job requests 60 ligands, max is 50" }) })
    await assert.rejects(submitDockJob([], 1), (e: unknown) => {
      assert.ok(e instanceof ExternalCallError)
      assert.match(e.message, /HTTP 400: job requests 60 ligands, max is 50/, "the reason is kept")
      assert.match(e.message, /<docking-api>\/submit/)
      assert.doesNotMatch(e.message, /gnina\.test/)
      return true
    })
    route({ "/status/": networkError() })
    await assert.rejects(pollDockJob("j1", 0), (e: unknown) => e instanceof ExternalCallError && !/gnina\.test/.test(e.message))
  })

  it("a missing URL is a configuration error, not an upstream failure", async () => {
    delete process.env.GNINA_JOB_API_URL
    await assert.rejects(pollDockJob("j1", 0), (e: unknown) => e instanceof Error && !(e instanceof ExternalCallError))
  })
})

describe("warnings wording", () => {
  const w = (source: string, target?: string) => ({ source, kind: "server", message: "m", target })

  it("groups by source and names a few affected items", () => {
    assert.equal(
      describeWarnings([w("pubchem", "A/x"), w("pubchem", "A/y"), w("pubchem", "B/z"), w("pubchem", "C/q"), w("openai")]),
      "4 PubChem calls failed (A/x, A/y, B/z, …); 1 OpenAI call failed"
    )
  })

  it("is empty for no warnings, and falls back to the raw source name", () => {
    assert.equal(describeWarnings([]), "")
    assert.equal(describeWarnings([w("mystery", "t")]), "1 mystery call failed (t)")
  })

  it("errorMessageFrom reads the API error, else the fallback", async () => {
    assert.equal(await errorMessageFrom(res(502, { error: "openai: server" }), "HTTP 502"), "openai: server")
    assert.equal(await errorMessageFrom(res(500, "<html>"), "HTTP 500"), "HTTP 500")
    assert.equal(await errorMessageFrom(res(500, { error: "" }), "HTTP 500"), "HTTP 500")
  })
})
