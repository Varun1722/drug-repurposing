// GNINA docking job client — submits/polls the async job queue exposed by
// the Modal `job_api` endpoint (modal_app/gnina_worker.py). Docking runs in
// the background on Modal; this module never blocks on it, so callers stay
// well under Vercel's function duration limits regardless of batch size.

import type {
  DockingTarget,
  DockStatusResponse,
  DockSubmitResponse,
} from "@/lib/types"

function jobApiUrl(path: string): string {
  const base = process.env.GNINA_JOB_API_URL
  if (!base) throw new Error("GNINA_JOB_API_URL not set")
  return `${base.replace(/\/$/, "")}${path}`
}

export async function submitDockJob(
  targets: DockingTarget[],
  round: number
): Promise<DockSubmitResponse> {
  const res = await fetch(jobApiUrl("/submit"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ targets, round }),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`GNINA submit error ${res.status}: ${text}`)
  }

  return res.json()
}

export async function pollDockJob(
  jobId: string,
  since: number
): Promise<DockStatusResponse> {
  const res = await fetch(
    jobApiUrl(
      `/status/${encodeURIComponent(jobId)}?since=${encodeURIComponent(String(since))}`
    )
  )

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`GNINA status error ${res.status}: ${text}`)
  }

  return res.json()
}
