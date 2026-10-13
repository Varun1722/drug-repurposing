import { NextResponse } from "next/server"
import { queryPromptStructured, LLMOutputError } from "@/lib/openai"
import { ReasoningDecisionSchema } from "@/lib/llm-schemas"
import { errorPayload } from "@/lib/retry"
import type { DockingResult, ReasoningDecision } from "@/lib/types"

export const maxDuration = 60

const MAX_EXPANSION_ROUNDS = 2

export async function POST(req: Request) {
  try {
    const {
      disease,
      dockingResults,
      round,
      hypotheses = [],
      expansionHistory = [],
    } = (await req.json()) as {
      disease: string
      dockingResults: DockingResult[]
      round: number
      hypotheses?: string[]
      expansionHistory?: { round: number; action: string; rationale: string }[]
    }

    if (!disease || !dockingResults?.length) {
      return NextResponse.json(
        { error: "disease and dockingResults are required" },
        { status: 400 }
      )
    }

    // Build top 30 summary
    const topResults = [...dockingResults]
      .sort((a, b) => b.confidenceScore - a.confidenceScore)
      .slice(0, 30)
      .map((r) => ({
        name: r.name,
        score: Math.round(r.confidenceScore * 10000) / 10000,
        mechanism: (r.mechanism || "").slice(0, 100),
        fda_status: (r.fdaStatus || "").slice(0, 60),
        protein: r.proteinTarget || "",
        source: r.source || "",
        round: r.round || 1,
      }))

    let decision: ReasoningDecision
    try {
      const parsed = await queryPromptStructured(
        "analysis/reasoning_decision",
        {
          disease,
          round,
          max_round: MAX_EXPANSION_ROUNDS + 1,
          hypotheses: JSON.stringify(hypotheses),
          expansion_history: JSON.stringify(expansionHistory),
          top_results: JSON.stringify(topResults, null, 2),
        },
        ReasoningDecisionSchema,
        // Leave headroom under maxDuration so a stuck call fails with a clean
        // error instead of the platform killing the request.
        { maxTotalMs: 50_000 }
      )
      decision = {
        action: parsed.action,
        rationale: parsed.rationale,
        hypothesis: parsed.hypothesis,
      }
      if (parsed.action === "expand_3d_similar") {
        decision.seedCids = parsed.seed_cids ?? []
      }
      if (parsed.action === "expand_class") {
        decision.drugClass = parsed.drug_class ?? ""
        decision.drugNames = parsed.drug_names ?? []
      }
    } catch (error) {
      if (!(error instanceof LLMOutputError)) throw error
      // The reply was unusable (refusal, truncation, ...). Don't expand on a
      // guess, and say why in the rationale so it shows up in the history.
      console.error("Analyze: unusable LLM output:", error.kind, error.message)
      decision = {
        action: "proceed",
        rationale: `(Reasoning step produced no usable result: ${error.kind}. Proceeding to synthesis.)`,
        hypothesis: "",
      }
    }

    // Force proceed if max rounds reached
    if (round >= MAX_EXPANSION_ROUNDS + 1) {
      decision.action = "proceed"
      delete decision.seedCids
      delete decision.drugClass
      delete decision.drugNames
      decision.rationale += " (Max rounds reached, proceeding to synthesis.)"
    }

    return NextResponse.json(decision)
  } catch (error) {
    console.error("Analyze API error:", error)
    const { status, body } = errorPayload(error)
    return NextResponse.json(body, { status })
  }
}
