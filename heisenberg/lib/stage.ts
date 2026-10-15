// What the development stage of a compound (see Stage in lib/types.ts, looked up
// in lib/chembl.ts) means for the report, and how it travels with a compound
// through docking.

import type { DockingResult, DockingTarget, Stage } from "./types"

export const STAGE_LABELS: Record<Stage, string> = {
  unknown: "Unknown",
  preclinical: "Preclinical",
  "phase-1": "Phase 1",
  "phase-2": "Phase 2",
  "phase-3": "Phase 3",
  approved: "Approved",
  withdrawn: "Withdrawn",
}

export type CompoundClass = "disease_purposed" | "repurposing" | "novel"

/**
 * The report's three sections: an approved drug the literature review calls a
 * mainstream drug for this disease; an approved drug from elsewhere (a repurposing
 * candidate); everything not approved now, from preclinical to withdrawn.
 */
export function classifyCompound(stage: Stage | undefined, category: string | undefined): CompoundClass {
  if (stage !== "approved") return "novel"
  return category === "mainstream" ? "disease_purposed" : "repurposing"
}

/**
 * The docking worker only echoes the ligand fields it knows, so its results come
 * back without the stage. Put it back from the ligands that were sent, matched by
 * protein and name (the worker keys its ligands by name too). No match: "unknown".
 */
export function attachStages(
  results: readonly DockingResult[],
  targets: readonly DockingTarget[]
): DockingResult[] {
  return results.map((result) => {
    const ligand = targets
      .find((target) => target.protein === result.proteinTarget)
      ?.ligands.find((l) => l.name === result.name)
    return { ...result, stage: ligand?.stage ?? "unknown", category: ligand?.category }
  })
}
