// GNINA docking job client — submits/polls the async job queue exposed by
// the Modal `job_api` endpoint (modal_app/gnina_worker.py). Docking runs in
// the background on Modal; this module never blocks on it, so callers stay
// well under Vercel's function duration limits regardless of batch size.
//
// No retries in here, on purpose. Submit creates a job, so repeating it after a
// lost response could start (and pay for) the docking twice. Poll is repeated
// by its caller anyway (the UI asks every 2s), so a failed poll is retried by
// the next tick; stacking another retry layer here would only multiply waits.
// Both throw ExternalCallError, which the routes turn into a 502.

import { fetchWithRetry, responseJson } from "@/lib/retry"
import type {
  DockingTarget,
  DockStatusResponse,
  DockSubmitResponse,
} from "@/lib/types"

const SINGLE_ATTEMPT = { maxAttempts: 1 } as const

function jobApiUrl(path: string): string {
  const base = process.env.GNINA_JOB_API_URL
  if (!base) throw new Error("GNINA_JOB_API_URL not set")
  return `${base.replace(/\/$/, "")}${path}`
}

export async function submitDockJob(
  targets: DockingTarget[],
  round: number
): Promise<DockSubmitResponse> {
  const res = await fetchWithRetry(
    jobApiUrl("/submit"),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ targets, round }),
    },
    { source: "gnina", policy: SINGLE_ATTEMPT, timeoutMs: 25_000 }
  )
  return responseJson<DockSubmitResponse>(res, "gnina")
}

export async function pollDockJob(
  jobId: string,
  since: number
): Promise<DockStatusResponse> {
  const res = await fetchWithRetry(
    jobApiUrl(
      `/status/${encodeURIComponent(jobId)}?since=${encodeURIComponent(String(since))}`
    ),
    {},
    { source: "gnina", policy: SINGLE_ATTEMPT, timeoutMs: 20_000 }
  )
  return responseJson<DockStatusResponse>(res, "gnina")
}
