// PubChem API client — ported from agent2.py
//
// "Not found" (a 404) is a normal result: null / []. A request that fails even
// after retries throws ExternalCallError, so an outage is not mistaken for "no
// such compound" or "no similar compounds".

import { ExternalCallError, fetchWithRetry, responseJson } from "./retry"

const PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
const PUBCHEM_DELAY = 250 // ms between requests
const PROPERTIES = "IsomericSMILES,CanonicalSMILES,IUPACName,MolecularFormula"

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export interface PubChemCompound {
  cid: number
  smiles: string
  iupacName: string
  molecularFormula: string
}

type PubChemProps = Record<string, unknown>

// PubChem returns SMILES under varying keys depending on the request.
function smilesOf(props: PubChemProps | undefined): string {
  return (
    (props?.CanonicalSMILES as string) ||
    (props?.IsomericSMILES as string) ||
    (props?.ConnectivitySMILES as string) ||
    (props?.SMILES as string) ||
    ""
  )
}

function toCompound(props: PubChemProps): PubChemCompound {
  return {
    cid: props.CID as number,
    smiles: smilesOf(props),
    iupacName: (props.IUPACName as string) || "",
    molecularFormula: (props.MolecularFormula as string) || "",
  }
}

// PubChem answers 404 for "no such record" and 400 (PUGREST.BadRequest) for an
// identifier it cannot parse. For identifiers that come from an LLM both mean
// "no match", not "PubChem is down", so neither is a failure.
const NO_MATCH = [400, 404] as const

async function getJson<T>(url: string, timeoutMs: number, allowStatus: readonly number[] = [404]): Promise<T | null> {
  const res = await fetchWithRetry(url, {}, { source: "pubchem", allowStatus, timeoutMs })
  if (allowStatus.includes(res.status)) return null
  return responseJson<T>(res, "pubchem")
}

type PropertyTable = { PropertyTable?: { Properties?: PubChemProps[] } }

/** Try a single PubChem name lookup. */
async function tryLookup(name: string): Promise<PubChemCompound | null> {
  const encoded = encodeURIComponent(name.trim())
  const url = `${PUBCHEM_BASE}/compound/name/${encoded}/property/${PROPERTIES}/JSON`
  console.log(`[pubchem] tryLookup "${name}" → ${url}`)
  const data = await getJson<PropertyTable>(url, 15_000, NO_MATCH)
  const props = data?.PropertyTable?.Properties?.[0]
  if (!props) {
    console.log(`[pubchem] tryLookup "${name}" → not found`)
    return null
  }
  const smiles = smilesOf(props)
  if (!smiles) {
    console.log(`[pubchem] tryLookup "${name}" → no SMILES in props`)
    return null
  }
  console.log(`[pubchem] tryLookup "${name}" → CID ${props.CID}, SMILES ${smiles.slice(0, 40)}...`)
  return toCompound(props)
}

/**
 * Look up SMILES for a drug by name.
 * Tries multiple name variants (strip parenthetical, code names, etc.).
 * Returns null only for "not found"; a PubChem failure throws ExternalCallError
 * (trying other spellings against a service that is down would not help).
 */
export async function lookupSmiles(
  drugName: string
): Promise<PubChemCompound | null> {
  if (!drugName) return null

  // Handle CID_xxx names — look up by CID directly
  const cidMatch = drugName.match(/^CID_(\d+)$/)
  if (cidMatch) {
    const url = `${PUBCHEM_BASE}/compound/cid/${cidMatch[1]}/property/${PROPERTIES}/JSON`
    const data = await getJson<PropertyTable>(url, 15_000, NO_MATCH)
    const props = data?.PropertyTable?.Properties?.[0]
    if (!props || !smilesOf(props)) return null
    return toCompound(props)
  }

  // Build candidate name variants
  const variants: string[] = [drugName]

  // Strip parenthetical: "Sotorasib (AMG 510)" → "Sotorasib"
  const noParens = drugName.replace(/\s*\(.*?\)\s*/g, "").trim()
  if (noParens && noParens !== drugName) variants.push(noParens)

  // Extract parenthetical content: "Sotorasib (AMG 510)" → "AMG 510"
  const parenMatch = drugName.match(/\(([^)]+)\)/)
  if (parenMatch) variants.push(parenMatch[1].trim())

  // Strip trailing descriptors: "Erlotinib hydrochloride" → "Erlotinib"
  const firstWord = drugName.split(/[\s,;(/-]+/)[0]
  if (firstWord && firstWord.length > 3 && !variants.includes(firstWord))
    variants.push(firstWord)

  // Try each variant
  for (const name of variants) {
    const result = await tryLookup(name)
    if (result) return result
    await delay(PUBCHEM_DELAY)
  }

  return null
}

type CidList = { InformationList?: { Information?: { CID?: number[] }[] } }

async function collectCids(url: string): Promise<number[]> {
  const data = await getJson<CidList>(url, 30_000, NO_MATCH)
  return (data?.InformationList?.Information ?? []).flatMap((entry) => entry.CID ?? [])
}

/**
 * Search for bioactive compounds targeting a protein.
 * Port of agent2.py:search_compounds_for_target()
 *
 * Returns [] when PubChem has no active compounds for the target. If a lookup
 * failed and nothing was found, or the property fetch failed, it throws so an
 * outage is not read as "no compounds".
 */
export async function searchCompoundsForTarget(
  proteinName: string,
  maxCompounds: number = 50
): Promise<PubChemCompound[]> {
  const geneSymbol = proteinName.split(/\s+/)[0]
  let cids: number[] = []
  const lookupErrors: ExternalCallError[] = []

  // Try bioassay target search
  try {
    cids = await collectCids(
      `${PUBCHEM_BASE}/assay/target/genesymbol/${encodeURIComponent(geneSymbol)}/cids/JSON?cids_type=active`
    )
  } catch (err) {
    if (!(err instanceof ExternalCallError)) throw err
    lookupErrors.push(err)
  }

  // Fallback: gene symbol search
  if (cids.length === 0) {
    try {
      await delay(PUBCHEM_DELAY)
      cids = await collectCids(
        `${PUBCHEM_BASE}/gene/symbol/${encodeURIComponent(geneSymbol)}/cids/JSON?cids_type=pharmacologically_active`
      )
    } catch (err) {
      if (!(err instanceof ExternalCallError)) throw err
      lookupErrors.push(err)
    }
  }

  // Deduplicate and limit
  cids = [...new Set(cids)].slice(0, maxCompounds)
  if (cids.length === 0) {
    if (lookupErrors.length > 0) throw lookupErrors[lookupErrors.length - 1]
    return []
  }

  // Fetch compound properties in batches of 100
  const compounds: PubChemCompound[] = []
  for (let i = 0; i < cids.length; i += 100) {
    const batch = cids.slice(i, i + 100)
    await delay(PUBCHEM_DELAY)

    const data = await getJson<PropertyTable>(
      `${PUBCHEM_BASE}/compound/cid/${batch.join(",")}/property/${PROPERTIES}/JSON`,
      30_000
    )
    for (const p of data?.PropertyTable?.Properties ?? []) compounds.push(toCompound(p))
  }

  return compounds
}

/**
 * Search for 3D-similar compounds.
 * Port of agent2.py:search_3d_similar()
 *
 * Returns [] if the CID has no 3D conformer or no neighbours; a failed request
 * throws ExternalCallError.
 */
export async function search3dSimilar(
  cid: number,
  maxResults: number = 20
): Promise<PubChemCompound[]> {
  // Get similar CIDs
  const searchUrl = `${PUBCHEM_BASE}/compound/fastsimilarity_3d/cid/${cid}/cids/JSON`
  const found = await getJson<{ IdentifierList?: { CID?: number[] } }>(searchUrl, 60_000)
  const similarCids = (found?.IdentifierList?.CID || []).slice(0, maxResults)
  if (similarCids.length === 0) return []

  // Fetch properties
  await delay(PUBCHEM_DELAY)
  const data = await getJson<PropertyTable>(
    `${PUBCHEM_BASE}/compound/cid/${similarCids.join(",")}/property/${PROPERTIES}/JSON`,
    30_000
  )
  return (data?.PropertyTable?.Properties ?? []).map(toCompound)
}
