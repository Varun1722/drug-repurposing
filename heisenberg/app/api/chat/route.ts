import { NextResponse } from "next/server"
import { queryPrompt } from "@/lib/openai"
import { errorPayload } from "@/lib/retry"

export const maxDuration = 60

export async function POST(req: Request) {
  try {
    const { question, context } = (await req.json()) as {
      question: string
      context: {
        disease?: string
        proteins?: string[]
        dockingResults?: {
          name: string
          confidence: number
          protein: string
          mechanism: string
        }[]
        reviewSummary?: string
      }
    }

    if (!question) {
      return NextResponse.json(
        { error: "question is required" },
        { status: 400 }
      )
    }

    let contextStr = ""
    if (context.disease) {
      contextStr += `Disease: ${context.disease}\n`
    }
    if (context.proteins?.length) {
      contextStr += `Protein targets: ${context.proteins.join(", ")}\n`
    }
    if (context.dockingResults?.length) {
      contextStr += `\nTop docking results:\n`
      for (const r of context.dockingResults) {
        contextStr += `- ${r.name}: ${Math.round(r.confidence * 100)}% confidence against ${r.protein} (${r.mechanism})\n`
      }
    }
    if (context.reviewSummary) {
      contextStr += `\nLiterature review excerpt:\n${context.reviewSummary}\n`
    }

    const answer = await queryPrompt("chat/answer", {
      context_block: contextStr ? `Context:\n${contextStr}\n\n` : "",
      question,
    }, { maxTotalMs: 50_000 })

    return NextResponse.json({ answer })
  } catch (error) {
    console.error("Chat API error:", error)
    const { status, body } = errorPayload(error)
    return NextResponse.json(body, { status })
  }
}
