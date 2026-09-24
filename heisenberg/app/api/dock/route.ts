import { dockLigands } from "@/lib/gnina"
import type { DockingTarget, DockingResult } from "@/lib/types"

export const maxDuration = 600 // 10 min for docking

export async function POST(req: Request) {
  try {
    const { targets, round = 1 } = (await req.json()) as {
      targets: DockingTarget[]
      round?: number
    }

    if (!targets?.length) {
      return new Response(
        JSON.stringify({ error: "targets array is required" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      )
    }

    // SSE streaming response
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      async start(controller) {
        function send(data: object) {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify(data)}\n\n`)
          )
        }

        const allResults: DockingResult[] = []

        try {
          for (const target of targets) {
            const { protein, pdbId, pdbContentB64, ligands } = target

            if (!pdbContentB64) {
              send({
                type: "error",
                message: `No PDB data for ${protein}, skipping`,
              })
              continue
            }

            const targetLigands = ligands.filter((l) => l.smiles)
            if (targetLigands.length === 0) {
              send({
                type: "error",
                message: `No valid ligands for ${protein}, skipping`,
              })
              continue
            }

            send({
              type: "progress",
              protein,
              drugIndex: 0,
              drugTotal: targetLigands.length,
              message: `Starting docking for ${protein} (${targetLigands.length} ligands)`,
            })

            // One ligand per GNINA worker call, fired concurrently — Modal
            // autoscales a CPU container per in-flight request.
            let completed = 0

            const chunkResultLists = await Promise.all(
              targetLigands.map(async (origLigand) => {
                const chunk = [{ name: origLigand.name, smiles: origLigand.smiles }]
                try {
                  const chunkResults = await dockLigands(pdbContentB64, chunk)
                  completed++
                  send({
                    type: "progress",
                    protein,
                    drugIndex: completed,
                    drugTotal: targetLigands.length,
                    currentDrug: null,
                    message: `Completed ${origLigand.name} against ${protein} (${completed}/${targetLigands.length})`,
                  })

                  return chunkResults.map(
                    (r): DockingResult => ({
                      name: r.name,
                      confidenceScore: r.confidence_score,
                      confidenceRaw: r.confidence_raw,
                      mechanism: origLigand.mechanism || "",
                      fdaStatus: origLigand.fdaStatus || "",
                      source: origLigand.source || "",
                      proteinTarget: protein,
                      pdbId: pdbId || "",
                      round,
                      allPoses: r.all_poses,
                    })
                  )
                } catch (err) {
                  send({
                    type: "error",
                    message: `Docking failed for ${origLigand.name} against ${protein}: ${err instanceof Error ? err.message : "unknown"}`,
                  })
                  return []
                }
              })
            )

            const targetResults = chunkResultLists.flat()

            // Sort by confidence
            targetResults.sort(
              (a, b) => b.confidenceScore - a.confidenceScore
            )
            allResults.push(...targetResults)

            send({
              type: "target_complete",
              protein,
              results: targetResults,
            })
          }

          // Sort all results
          allResults.sort(
            (a, b) => b.confidenceScore - a.confidenceScore
          )

          send({ type: "complete", allResults })
        } catch (err) {
          send({
            type: "error",
            message: err instanceof Error ? err.message : "Unknown error",
          })
        }

        controller.close()
      },
    })

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    })
  } catch (error) {
    return new Response(
      JSON.stringify({
        error: error instanceof Error ? error.message : "Unknown error",
      }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    )
  }
}
