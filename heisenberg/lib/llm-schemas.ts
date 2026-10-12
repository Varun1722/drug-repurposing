// Schemas for LLM calls that must return structured JSON. Each one is sent to
// OpenAI as a strict `json_schema` response format (see lib/openai.ts) and
// then re-validated with Zod on the way back.
//
// Rules for strict mode (enforced by lib/strict-schema.ts):
//   - the root is an object, so lists are wrapped ({ proteins: [...] })
//   - no `.optional()` / `.default()`: model a missing value as `.nullable()`
//   - keys are generated in declaration order, so reasoning fields
//     (rationale, hypothesis) come before the decision they justify
//
// Python mirror: llm_schemas.py. Keep the two in sync.

import { z } from "zod/v4"

export const ProteinListSchema = z.object({
  proteins: z
    .array(z.string())
    .describe("Short protein names or gene symbols, e.g. EGFR, BRAF, KRAS G12D"),
})

export const DrugNameListSchema = z.object({
  drug_names: z
    .array(z.string())
    .describe("Generic (non-brand) drug names, e.g. atorvastatin"),
})

export const DrugProteinMapSchema = z.object({
  drugs: z.array(
    z.object({
      drug: z.string(),
      proteins: z
        .array(z.string())
        .describe("Target protein symbols this drug may bind to"),
      mechanism: z.string(),
      fda_status: z.string(),
      category: z.enum(["mainstream", "repurposing_candidate"]),
      trial_status_for_targets: z
        .string()
        .describe(
          'Clinical-trial status for the listed targets, or "No known trials for these targets"'
        ),
    })
  ),
})

export const ReasoningDecisionSchema = z.object({
  rationale: z.string().describe("Analysis of the docking patterns, 2-3 sentences"),
  hypothesis: z.string().describe("A specific scientific hypothesis, 1 sentence"),
  action: z.enum(["expand_3d_similar", "expand_class", "proceed"]),
  seed_cids: z
    .array(z.number().int())
    .nullable()
    .describe("PubChem CIDs to seed 3D similarity search; null unless action is expand_3d_similar"),
  drug_class: z
    .string()
    .nullable()
    .describe("Drug class to expand; null unless action is expand_class"),
  drug_names: z
    .array(z.string())
    .nullable()
    .describe("Drug names in that class; null unless action is expand_class"),
})

// Every schema a prompt may name in its `output.schema` front matter.
export const SCHEMAS = {
  ProteinList: ProteinListSchema,
  DrugNameList: DrugNameListSchema,
  DrugProteinMap: DrugProteinMapSchema,
  ReasoningDecision: ReasoningDecisionSchema,
} as const

export type ProteinList = z.infer<typeof ProteinListSchema>
export type DrugNameList = z.infer<typeof DrugNameListSchema>
export type DrugProteinMap = z.infer<typeof DrugProteinMapSchema>
export type ReasoningDecisionOutput = z.infer<typeof ReasoningDecisionSchema>
