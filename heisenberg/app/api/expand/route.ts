import { NextResponse } from "next/server"
import { queryPromptStructured, LLMOutputError } from "@/lib/openai"
import { DrugNameListSchema } from "@/lib/llm-schemas"
import { lookupSmiles, search3dSimilar } from "@/lib/pubchem"
import type { ReasoningDecision, DockingTarget, DockingResult, Ligand } from "@/lib/types"
import { shortenDrugName } from "@/lib/types"
import { errorPayload, toWarning, type FailureWarning } from "@/lib/retry"

export const maxDuration = 120

export async function POST(req: Request) {
  try {
    const {
      decision,
      proteins,
      existingTargets,
      allDockingResults = [],
    } = (await req.json()) as {
      decision: ReasoningDecision
      proteins: string[]
      existingTargets: DockingTarget[]
      allDockingResults?: DockingResult[]
    }

    if (!decision || !proteins?.length) {
      return NextResponse.json(
        { error: "decision and proteins are required" },
        { status: 400 }
      )
    }

    const newLigandsByProtein: Record<string, Ligand[]> = {}
    // PubChem calls that failed (after retries). An empty result with warnings
    // means "the search failed", not "nothing similar exists"; the UI says which.
    const warnings: FailureWarning[] = []

    if (decision.action === "expand_3d_similar") {
      let seedCids = decision.seedCids || []

      // Fallback: extract CIDs from top docking results
      if (seedCids.length === 0 && allDockingResults.length > 0) {
        const sorted = [...allDockingResults].sort(
          (a, b) => b.confidenceScore - a.confidenceScore
        )
        for (const r of sorted) {
          const src = r.source || ""
          if (src.startsWith("pubchem_cid_")) {
            const cid = parseInt(src.split("_").pop() || "", 10)
            if (!isNaN(cid)) seedCids.push(cid)
          }
          if (seedCids.length >= 5) break
        }
      }

      // Search all seed CIDs in parallel
      const similarResults = await Promise.allSettled(
        seedCids.map((cid) => search3dSimilar(cid, 20))
      )

      for (let i = 0; i < seedCids.length; i++) {
        const r = similarResults[i]
        if (r.status !== "fulfilled") {
          warnings.push(toWarning(r.reason, `CID ${seedCids[i]}`))
          continue
        }
        const cid = seedCids[i]
        for (const c of r.value) {
          if (!c.smiles) continue
          const lig: Ligand = {
            name: shortenDrugName(c.iupacName, c.cid),
            smiles: c.smiles,
            mechanism: `3D-similar to CID ${cid} (ST≥0.80, CT≥0.50)`,
            fdaStatus: "Unknown — requires verification",
            source: `pubchem_3dsim_cid_${c.cid}_from_${cid}`,
          }
          for (const protein of proteins) {
            if (!newLigandsByProtein[protein]) newLigandsByProtein[protein] = []
            newLigandsByProtein[protein].push(lig)
          }
        }
      }
    } else if (decision.action === "expand_class") {
      let drugNames = decision.drugNames || []

      // If no drug names provided, ask the LLM
      if (drugNames.length === 0 && decision.drugClass) {
        try {
          const parsed = await queryPromptStructured(
            "analysis/class_expansion",
            {
              drug_class: decision.drugClass,
              proteins: proteins.join(", "),
              exclude:
                allDockingResults
                  .slice(0, 20)
                  .map((r) => r.name)
                  .join(", ") || "(none)",
            },
            DrugNameListSchema,
            { maxTotalMs: 100_000 }
          )
          drugNames = parsed.drug_names
        } catch (error) {
          if (!(error instanceof LLMOutputError)) throw error
          // No names -> no new ligands -> the route returns an empty
          // newTargets list; the warning tells the UI why.
          console.error("Expand: unusable LLM output:", error.kind, error.message)
          warnings.push({ source: "openai", kind: error.kind, message: `Drug-name reply unusable: ${error.message}` })
        }
      }

      // Look up SMILES for all drugs in parallel
      const lookupResults = await Promise.allSettled(
        drugNames.map(async (drugName) => {
          const result = await lookupSmiles(drugName)
          return { drugName, result }
        })
      )

      for (let i = 0; i < lookupResults.length; i++) {
        const r = lookupResults[i]
        if (r.status !== "fulfilled") {
          warnings.push(toWarning(r.reason, drugNames[i]))
          continue
        }
        const { drugName, result } = r.value
        if (!result?.smiles) continue

        const lig: Ligand = {
          name: drugName,
          smiles: result.smiles,
          mechanism: `Class expansion: ${decision.drugClass || "unknown"}`,
          fdaStatus: "FDA-approved — requires indication verification",
          source: `pubchem_cid_${result.cid}`,
        }
        for (const protein of proteins) {
          if (!newLigandsByProtein[protein]) newLigandsByProtein[protein] = []
          newLigandsByProtein[protein].push(lig)
        }
      }
    }

    // Build new targets for docking (only with new ligands, reusing PDB data)
    const newTargets: DockingTarget[] = []

    for (const protein of proteins) {
      const newLigands = newLigandsByProtein[protein] || []
      if (newLigands.length === 0) continue

      // Deduplicate against existing ligands
      const existing = existingTargets.find((t) => t.protein === protein)
      const existingSmiles = new Set(
        (existing?.ligands || []).map((l) => l.smiles)
      )
      const uniqueLigands = newLigands.filter(
        (l) => !existingSmiles.has(l.smiles)
      )

      if (uniqueLigands.length === 0) continue

      newTargets.push({
        protein,
        pdbId: existing?.pdbId || null,
        pdbContentB64: existing?.pdbContentB64 || null,
        ligands: uniqueLigands,
      })
    }

    return NextResponse.json({ newTargets, warnings })
  } catch (error) {
    console.error("Expand API error:", error)
    const { status, body } = errorPayload(error)
    return NextResponse.json(body, { status })
  }
}
