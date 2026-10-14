// Polling loop for a GNINA docking job, kept out of the page component so its
// failure handling can be tested without a browser.
//
// Docking runs in the background on Modal; the page asks /api/dock/status for
// new events until the job reports `done`. This loop IS the retry for that
// status call (the server does not retry it): one failed poll, such as a
// network blip or a 5xx, must not abandon a job that is still running, so it is
// simply tried again on the next tick. Only a run of consecutive failures
// gives up, and a 4xx (bad or unknown job id) gives up at once because it will
// not fix itself.

import type { DockJobEvent, DockStatusResponse } from "./types"
import { errorMessageFrom } from "./warnings"

/** Consecutive failed polls (one per interval) before the page gives up on a job. */
export const MAX_FAILED_POLLS = 15 // ~30s at the default interval

export interface PollDockJobOptions {
  jobId: string
  /** One status request: GET /api/dock/status?jobId=…&since=… */
  fetchStatus: (since: number) => Promise<Response>
  /** New events, in order, from each successful poll. */
  onEvents: (events: DockJobEvent[]) => void
  /** Called after a failed poll that will be retried. */
  onRetry?: (failedPolls: number, maxFailedPolls: number, reason: string) => void
  sleep: (ms: number) => Promise<void>
  intervalMs?: number
  maxFailedPolls?: number
}

export async function pollDockJobUntilDone(options: PollDockJobOptions): Promise<void> {
  const { jobId, fetchStatus, onEvents, onRetry, sleep } = options
  const intervalMs = options.intervalMs ?? 2000
  const maxFailedPolls = options.maxFailedPolls ?? MAX_FAILED_POLLS

  let since = 0
  let failedPolls = 0

  for (let done = false; !done; ) {
    await sleep(intervalMs)

    let res: Response | null = null
    let reason = ""
    try {
      res = await fetchStatus(since)
    } catch (error) {
      reason = error instanceof Error ? error.message : "network error"
    }
    if (res && !res.ok) {
      reason = await errorMessageFrom(res, `HTTP ${res.status}`)
      // A 4xx will not fix itself (408 and 429 are the exceptions); a 5xx may.
      if (res.status < 500 && res.status !== 408 && res.status !== 429) {
        throw new Error(`Dock status error: ${reason}`)
      }
    }
    let status: DockStatusResponse | null = null
    if (res && res.ok) {
      try {
        status = (await res.json()) as DockStatusResponse
      } catch {
        reason = "unreadable status response" // a truncated body is a failed poll too
      }
    }

    if (!status) {
      failedPolls++
      if (failedPolls >= maxFailedPolls) {
        throw new Error(
          `Lost contact with docking job ${jobId} after ${failedPolls} failed status checks (${reason}). ` +
            `The job may still be running on Modal.`
        )
      }
      onRetry?.(failedPolls, maxFailedPolls, reason)
      continue
    }
    failedPolls = 0

    since = status.nextIndex
    done = status.done
    onEvents(status.events)
  }
}
