// ChEMBL client: how far a compound has got in drug development.
//
// Looked up by InChIKey (from PubChem), not by name: a name search for
// "metformin" returns 32 records including salts, and misses brand names and
// code names such as "LY294002" (ChEMBL writes "LY-294002"). The key is exact.
//
//   max_phase  4 approved, 3 / 2 / 1 phase 3 / 2 / 1, 0.5 early phase 1, -1 a
//              clinical candidate whose phase ChEMBL could not work out, and
//              null "in ChEMBL, but no record of a trial" (usually a research compound)
//   Not in ChEMBL (or not a valid key): "unknown". It is not "preclinical":
//   ChEMBL not having a compound says nothing about its testing.
//   Withdrawn: ChEMBL's withdrawn flag means "withdrawn in at least one country"
//   (thalidomide carries it for a 1963 withdrawal in Finland and is on sale
//   today), so the drug_warning records decide: worldwide, or in the United
//   States. A flagged drug with no country in them at all is "unknown": it may
//   be off the market (trovafloxacin) or on it (ergotamine), and calling it
//   approved would offer a pulled drug as a repurposing candidate.
//   max_phase 4 means approved by any regulator, not only the FDA.
//
// A failed call throws ExternalCallError, so an outage is not read as an answer.

import { fetchWithRetry, responseJson } from "./retry"
import type { Stage } from "./types"

const CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
const INCHIKEY = /^[A-Z]{14}-[A-Z]{10}-[A-Z]$/
const WITHDRAWN_WHERE_IT_MATTERS = /worldwide|united states/i

type Molecule = { molecule_chembl_id: string; max_phase: string | number | null; withdrawn_flag: boolean }

async function getJson<T>(url: string): Promise<T | null> {
  const res = await fetchWithRetry(url, {}, { source: "chembl", allowStatus: [404], timeoutMs: 15_000 })
  if (res.status === 404) return null
  return responseJson<T>(res, "chembl")
}

/** ChEMBL sends max_phase as a string ("4.0") or null. */
export function stageFromMaxPhase(maxPhase: string | number | null | undefined): Stage {
  if (maxPhase === null) return "preclinical"
  switch (Number(maxPhase)) {
    case 4:
      return "approved"
    case 3:
      return "phase-3"
    case 2:
      return "phase-2"
    case 1:
    case 0.5:
      return "phase-1"
    default:
      return "unknown"
  }
}

/** The countries ChEMBL's withdrawal records name (one record can list several, "Peru; United Kingdom; United States"). */
async function withdrawalCountries(chemblId: string): Promise<string[]> {
  const data = await getJson<{ drug_warnings?: { warning_country?: string | null }[] }>(
    `${CHEMBL_BASE}/drug_warning.json?molecule_chembl_id=${encodeURIComponent(chemblId)}&warning_type=Withdrawn&limit=100&only=warning_country`
  )
  return (data?.drug_warnings ?? []).map((w) => w.warning_country ?? "").filter(Boolean)
}

/**
 * The stage of the compound with this standard InChIKey.
 * shortcuts: takes the first record for the key (a full key matched one record
 * every time in a 20-drug check), so a second record with a different phase is
 * ignored; and a withdrawal only in other markets (the EU, say: rimonabant) stays
 * "approved". Upgrade the first to the highest max_phase if it turns up, the
 * second by adding the markets that matter to WITHDRAWN_WHERE_IT_MATTERS.
 * A key ChEMBL registered under another form (a salt, a stereoisomer) misses and
 * is "unknown": trovafloxacin's PubChem key is not the one ChEMBL holds.
 */
export async function lookupStage(inchiKey: string): Promise<Stage> {
  if (!INCHIKEY.test(inchiKey)) return "unknown"
  const data = await getJson<{ molecules?: Molecule[] }>(
    `${CHEMBL_BASE}/molecule.json?molecule_structures__standard_inchi_key=${inchiKey}&only=molecule_chembl_id,max_phase,withdrawn_flag`
  )
  const molecule = data?.molecules?.[0]
  if (!molecule) return "unknown"
  if (molecule.withdrawn_flag) {
    const countries = await withdrawalCountries(molecule.molecule_chembl_id)
    if (countries.length === 0) return "unknown"
    if (countries.some((country) => WITHDRAWN_WHERE_IT_MATTERS.test(country))) return "withdrawn"
  }
  return stageFromMaxPhase(molecule.max_phase)
}

const MAX_FAILURES = 3

/**
 * A lookup for one request (make a new one per request). The same compound is
 * asked for once, however many proteins want it; at most `concurrency` calls run
 * at a time (ChEMBL publishes no rate limit, so stay polite); and after
 * MAX_FAILURES failed lookups it stops calling ChEMBL and rethrows the last
 * error, so an outage cannot use up the route's time limit.
 */
export function stageLookup(concurrency = 4): (inchiKey: string) => Promise<Stage> {
  const cache = new Map<string, Promise<Stage>>()
  const waiting: (() => void)[] = []
  let running = 0
  let failures = 0
  let lastError: unknown

  async function run(inchiKey: string): Promise<Stage> {
    if (running >= concurrency) await new Promise<void>((resolve) => waiting.push(resolve))
    else running++
    try {
      if (failures >= MAX_FAILURES) throw lastError
      return await lookupStage(inchiKey)
    } catch (error) {
      failures++
      lastError = error
      throw error
    } finally {
      // Hand the slot straight to the next waiter, so the limit holds.
      const next = waiting.shift()
      if (next) next()
      else running--
    }
  }

  return (inchiKey) => {
    let lookup = cache.get(inchiKey)
    if (!lookup) {
      lookup = run(inchiKey)
      cache.set(inchiKey, lookup)
    }
    return lookup
  }
}
