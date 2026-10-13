import { NextResponse } from "next/server"
import { queryPrompt } from "@/lib/openai"
import { promptVersions } from "@/lib/prompts"
import { errorPayload, routeBudget } from "@/lib/retry"
import type { DockingResult, DockingTarget } from "@/lib/types"

export const maxDuration = 180

function classifyCompound(
  fdaStatus: string,
  disease: string
): "disease_purposed" | "repurposing" | "novel" {
  const s = fdaStatus.toLowerCase()
  const diseaseLower = disease.toLowerCase()
  if (s.includes("approved") && s.includes(diseaseLower.split(/\s+/)[0])) {
    return "disease_purposed"
  }
  if (s.includes("approved") || s.includes("fda")) {
    return "repurposing"
  }
  return "novel"
}

const CATEGORY_TABLE_HEADER =
  "| Rank | Compound | Confidence | Mechanism | FDA Status | Source |\n" +
  "|------|----------|------------|-----------|------------|--------|\n"

/**
 * Per-protein ranked tables for one compound category, in the same layout the
 * Python pipeline (results.py: build_classified_overview) feeds the shared
 * report/* prompts.
 */
function formatCategoryBlock(
  results: DockingResult[],
  targets: DockingTarget[]
): string {
  const blocks: string[] = []
  for (const target of targets) {
    const compounds = results
      .filter((r) => (r.proteinTarget || "unknown") === target.protein)
      .sort((a, b) => b.confidenceScore - a.confidenceScore)
    if (compounds.length === 0) continue
    const header =
      `### Protein: ${target.protein}  (PDB: ${target.pdbId || "N/A"})\n` +
      `- Compounds in this category: ${compounds.length}\n`
    const rows = compounds
      .map(
        (c, i) =>
          `| ${i + 1} | ${c.name.slice(0, 80)} | ${c.confidenceScore.toFixed(4)} ` +
          `| ${(c.mechanism || "").slice(0, 60)} | ${(c.fdaStatus || "").slice(0, 40)} ` +
          `| ${c.source || ""} |\n`
      )
      .join("")
    blocks.push(header + CATEGORY_TABLE_HEADER + rows)
  }
  return blocks.length > 0 ? blocks.join("\n") : "(none)"
}

function toTitleCase(text: string): string {
  return text.replace(/\b\p{L}/gu, (letter) => letter.toUpperCase())
}

export async function POST(req: Request) {
  try {
    const { disease, allResults, targets } = (await req.json()) as {
      disease: string
      allResults: DockingResult[]
      targets: DockingTarget[]
    }

    if (!disease || !allResults?.length) {
      return NextResponse.json(
        { error: "disease and allResults are required" },
        { status: 400 }
      )
    }

    // Build overview per target
    const resultsByProtein: Record<string, DockingResult[]> = {}
    for (const r of allResults) {
      const prot = r.proteinTarget || "unknown"
      if (!resultsByProtein[prot]) resultsByProtein[prot] = []
      resultsByProtein[prot].push(r)
    }

    // Sort each target's results
    for (const prot of Object.keys(resultsByProtein)) {
      resultsByProtein[prot].sort(
        (a, b) => b.confidenceScore - a.confidenceScore
      )
    }

    // Classify compounds
    const classified: Record<string, DockingResult[]> = {
      disease_purposed: [],
      repurposing: [],
      novel: [],
    }
    for (const r of allResults) {
      const cat = classifyCompound(r.fdaStatus, disease)
      classified[cat].push(r)
    }

    // Build overview text
    let overviewText = `## Docking Overview\n\n`
    overviewText += `- Disease: ${disease}\n`
    overviewText += `- Total compounds docked: ${allResults.length}\n`
    overviewText += `- Protein targets: ${targets.map((t) => t.protein).join(", ")}\n\n`

    for (const target of targets) {
      const results = resultsByProtein[target.protein] || []
      overviewText += `### ${target.protein} (PDB: ${target.pdbId || "N/A"})\n\n`
      overviewText += `| Rank | Drug | Score | FDA Status | Mechanism |\n`
      overviewText += `|------|------|-------|------------|------------|\n`
      for (const [i, r] of results.slice(0, 15).entries()) {
        overviewText += `| ${i + 1} | ${r.name} | ${r.confidenceScore.toFixed(4)} | ${r.fdaStatus} | ${r.mechanism.slice(0, 60)} |\n`
      }
      overviewText += "\n"
    }

    const disease_purposed = formatCategoryBlock(classified.disease_purposed, targets)
    const repurposing = formatCategoryBlock(classified.repurposing, targets)
    const novel = formatCategoryBlock(classified.novel, targets)

    // The three calls share one deadline, so retries cannot outlast maxDuration.
    const budget = routeBudget(170_000)
    const methodology = await queryPrompt("report/methodology", {
      disease,
      overview: overviewText,
    }, { maxTotalMs: budget() })
    const results = await queryPrompt("report/results", {
      disease,
      disease_title: toTitleCase(disease),
      overview: overviewText,
      disease_purposed,
      repurposing,
      novel,
    }, { maxTotalMs: budget() })
    const conclusion = await queryPrompt("report/conclusion", {
      disease,
      overview: overviewText,
      repurposing,
    }, { maxTotalMs: budget() })

    const resultsMd =
      `# Results: Drug Repurposing Analysis for ${disease}\n\n` +
      `## Methodology\n\n${methodology}\n\n` +
      `## Results\n\n${results}\n\n` +
      `## Conclusion\n\n${conclusion}\n`

    return NextResponse.json({
      resultsMd,
      promptVersions: promptVersions([
        "report/methodology",
        "report/results",
        "report/conclusion",
      ]),
    })
  } catch (error) {
    console.error("Report API error:", error)
    const { status, body } = errorPayload(error)
    return NextResponse.json(body, { status })
  }
}
