import { submitDockJob } from "@/lib/dockJobs"
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
    return Response.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    )
  }
}
