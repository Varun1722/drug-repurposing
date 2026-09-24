// GNINA docking client — calls a Modal serverless CPU worker (modal_app/gnina_worker.py).
// Blind docking on the whole receptor for now; pocket-targeted docking is a follow-up (see TODO.md).

export interface GninaPose {
  cnn_score: number
  cnn_affinity: number
  vina_affinity: number
}

export interface GninaDockingResult {
  name: string
  confidence_score: number
  confidence_raw: number
  all_poses?: GninaPose[]
}

interface GninaWorkerResponse {
  results: GninaDockingResult[]
  errors?: { name: string; error: string }[]
  processing_time_seconds?: number
}

/**
 * Dock a batch of ligands against a protein receptor via the GNINA Modal worker.
 * This is a single synchronous request/response — no submit/poll needed since
 * the worker runs GNINA to completion before replying.
 */
export async function dockLigands(
  proteinPdbB64: string,
  ligands: { name: string; smiles: string }[],
  samplesPerComplex: number = 10
): Promise<GninaDockingResult[]> {
  const endpoint = process.env.GNINA_ENDPOINT_URL
  if (!endpoint) throw new Error("GNINA_ENDPOINT_URL not set")

  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      protein_pdb_b64: proteinPdbB64,
      ligands,
      samples_per_complex: samplesPerComplex,
    }),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`GNINA worker error ${res.status}: ${text}`)
  }

  const data: GninaWorkerResponse = await res.json()

  if (data.errors?.length) {
    for (const e of data.errors) {
      console.error(`GNINA docking failed for ${e.name}: ${e.error}`)
    }
  }

  return data.results || []
}
