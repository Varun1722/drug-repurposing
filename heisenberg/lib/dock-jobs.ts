// GNINA docking job client — submits/polls the async job queue exposed by
// the Modal `job_api` endpoint (services/docking/gnina_worker.py). Docking runs in
// the background on Modal; this module never blocks on it, so callers stay
// well under Vercel's function duration limits regardless of batch size.
//
// No retries in here, on purpose. Submit creates a job, so repeating it after a
// lost response could start (and pay for) the docking twice. Poll is repeated
// by its caller anyway (the UI asks every 2s), so a failed poll is retried by
// the next tick; stacking another retry layer here would only multiply waits.
// Both throw ExternalCallError, which the routes turn into a 502.

import { ExternalCallError, fetchWithRetry, responseJson } from "@/lib/retry"
import type {
  DockingTarget,
  DockStatusResponse,
  DockSubmitResponse,
} from "@/lib/types"

const SINGLE_ATTEMPT = { maxAttempts: 1 } as const

function jobApiBase(): string {
  const base = process.env.GNINA_JOB_API_URL
  if (!base) throw new Error("GNINA_JOB_API_URL not set")
  return base.replace(/\/$/, "")
}

/**
 * One call to the job API. The API is unauthenticated, and error messages reach
 * the browser (the page shows them), so its address is replaced by `<docking-api>`
 * in any failure.
 */
async function callJobApi<T>(path: string, init: RequestInit, timeoutMs: number): Promise<T> {
  const base = jobApiBase()
  try {
    const res = await fetchWithRetry(`${base}${path}`, init, {
      source: "gnina",
      policy: SINGLE_ATTEMPT,
      timeoutMs,
    })
    return await responseJson<T>(res, "gnina")
  } catch (error) {
    if (error instanceof ExternalCallError) {
      error.message = error.message.split(base).join("<docking-api>")
    }
    throw error
  }
}

export function submitDockJob(
  targets: DockingTarget[],
  round: number
): Promise<DockSubmitResponse> {
  return callJobApi<DockSubmitResponse>(
    "/submit",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ targets, round }),
    },
    25_000
  )
}

export function pollDockJob(
  jobId: string,
  since: number
): Promise<DockStatusResponse> {
  return callJobApi<DockStatusResponse>(
    `/status/${encodeURIComponent(jobId)}?since=${encodeURIComponent(String(since))}`,
    {},
    20_000
  )
}
