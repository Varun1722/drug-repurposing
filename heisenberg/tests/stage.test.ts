// Turning a development stage (from ChEMBL, lib/chembl.ts) into the report's
// three sections, and carrying the stage from the ligands sent to docking onto
// the results that come back (the Modal worker only echoes the fields it knows).

import assert from "node:assert/strict"
import { describe, it } from "node:test"

import { attachStages, classifyCompound } from "../lib/stage"
import type { DockingResult, DockingTarget, Ligand, Stage } from "../lib/types"

describe("classifyCompound", () => {
  const cases: [Stage | undefined, string | undefined, string][] = [
    // Approved, and the literature review calls it a mainstream drug for this disease.
    ["approved", "mainstream", "disease_purposed"],
    // Approved, but not a mainstream drug for this disease: a repurposing candidate.
    ["approved", "repurposing_candidate", "repurposing"],
    ["approved", undefined, "repurposing"],
    // Not approved (now): research compounds, however far along, and withdrawn drugs.
    ["phase-3", "mainstream", "novel"],
    ["phase-2", "repurposing_candidate", "novel"],
    ["phase-1", undefined, "novel"],
    ["preclinical", "repurposing_candidate", "novel"],
    ["withdrawn", "mainstream", "novel"],
    ["unknown", "mainstream", "novel"],
    // A result from a client that sent no stage.
    [undefined, "mainstream", "novel"],
  ]
  for (const [stage, category, expected] of cases) {
    it(`${stage ?? "no stage"} + ${category ?? "no category"} -> ${expected}`, () => {
      assert.equal(classifyCompound(stage, category), expected)
    })
  }
})

describe("attachStages", () => {
  const ligand = (name: string, stage: Stage, category?: string): Ligand => ({
    name,
    smiles: "CCO",
    mechanism: "",
    fdaStatus: "",
    source: "",
    stage,
    category,
  })
  const target = (protein: string, ...ligands: Ligand[]): DockingTarget => ({ protein, pdbId: "1ABC", pdbContentB64: null, ligands })
  const result = (name: string, proteinTarget: string): DockingResult => ({
    name,
    confidenceScore: 0.5,
    confidenceRaw: -7,
    mechanism: "",
    fdaStatus: "",
    source: "",
    proteinTarget,
    pdbId: "1ABC",
    round: 1,
  })

  it("copies the stage and category of the ligand with the same protein and name", () => {
    const targets = [target("EGFR", ligand("gefitinib", "approved", "mainstream"), ligand("zzz", "phase-2"))]
    const [r] = attachStages([result("gefitinib", "EGFR")], targets)
    assert.equal(r.stage, "approved")
    assert.equal(r.category, "mainstream")
  })

  it("matches per protein: the same drug can differ between two targets' ligand lists", () => {
    const targets = [target("EGFR", ligand("d", "approved")), target("KRAS", ligand("d", "phase-3"))]
    const out = attachStages([result("d", "KRAS"), result("d", "EGFR")], targets)
    assert.deepEqual(out.map((r) => r.stage), ["phase-3", "approved"])
  })

  it("a result with no matching ligand gets 'unknown', not a guess", () => {
    const [r] = attachStages([result("ghost", "EGFR")], [target("EGFR", ligand("other", "approved"))])
    assert.equal(r.stage, "unknown")
  })

  it("does not change its inputs", () => {
    const input = result("gefitinib", "EGFR")
    attachStages([input], [target("EGFR", ligand("gefitinib", "approved"))])
    assert.equal("stage" in input, false)
  })
})
