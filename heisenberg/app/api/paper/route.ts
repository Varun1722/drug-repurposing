import { NextResponse } from "next/server"
import { queryPrompt } from "@/lib/openai"
import { promptVersions } from "@/lib/prompts"

export const maxDuration = 300

export async function POST(req: Request) {
  try {
    const { reviewMd, resultsMd } = (await req.json()) as {
      reviewMd: string
      resultsMd: string
    }

    if (!reviewMd || !resultsMd) {
      return NextResponse.json(
        { error: "reviewMd and resultsMd are required" },
        { status: 400 }
      )
    }

    const abstract = await queryPrompt("paper/abstract", {
      review_text: reviewMd.slice(0, 3000),
      results_text: resultsMd.slice(0, 3000),
    })
    const introduction = await queryPrompt("paper/introduction", {
      sections: reviewMd.slice(0, 5000),
    })
    const mergedResults = await queryPrompt("paper/results", {
      results_text: resultsMd,
    })
    const conclusion = await queryPrompt("paper/conclusion_from_full_text", {
      review_text: reviewMd.slice(0, 2000),
      results_text: resultsMd.slice(0, 2000),
    })

    // Assemble final paper
    const paperMd =
      `# Drug Repurposing Analysis\n\n` +
      `## Abstract\n\n${abstract}\n\n` +
      `---\n\n` +
      `## 1. Introduction and Background\n\n${introduction}\n\n` +
      `---\n\n` +
      `## 2. Results\n\n${mergedResults}\n\n` +
      `---\n\n` +
      `## 3. Conclusion\n\n${conclusion}\n`

    return NextResponse.json({
      paperMd,
      promptVersions: promptVersions([
        "paper/abstract",
        "paper/introduction",
        "paper/results",
        "paper/conclusion_from_full_text",
      ]),
    })
  } catch (error) {
    console.error("Paper API error:", error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    )
  }
}
