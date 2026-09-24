import { pollDockJob } from "@/lib/dockJobs"

// One quick read of the job's event log — fast regardless of how long the
// underlying Modal job has been running.
export const maxDuration = 30

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const jobId = url.searchParams.get("jobId")
    const sinceParam = url.searchParams.get("since")
    const since = sinceParam ? Number(sinceParam) : 0

    if (!jobId) {
      return Response.json(
        { error: "jobId query param is required" },
        { status: 400 }
      )
    }

    const status = await pollDockJob(jobId, Number.isFinite(since) ? since : 0)
    return Response.json(status)
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    )
  }
}
