import { submitDockJob } from "@/lib/dockJobs"
import { errorPayload } from "@/lib/retry"
import type { DockingTarget } from "@/lib/types"

// Just kicks off the Modal job and returns — no need for the 600s budget
// the old blocking route needed.
export const maxDuration = 30

export async function POST(req: Request) {
  try {
    const { targets, round = 1 } = (await req.json()) as {
      targets: DockingTarget[]
      round?: number
    }

    if (!targets?.length) {
      return Response.json(
        { error: "targets array is required" },
        { status: 400 }
      )
    }

    const { jobId } = await submitDockJob(targets, round)
    return Response.json({ jobId })
  } catch (error) {
    const { status, body } = errorPayload(error)
    return Response.json(body, { status })
  }
}
