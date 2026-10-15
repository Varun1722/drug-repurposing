// Pipeline types — shared between API routes and frontend

// How far a compound has got in drug development, from ChEMBL (lib/chembl.ts).
// The order is the display order, not a ranking: `withdrawn` is an outcome, not a
// step after `approved`. `unknown` means ChEMBL has no usable answer.
export const STAGES = [
  "unknown",
  "preclinical",
  "phase-1",
  "phase-2",
  "phase-3",
  "approved",
  "withdrawn",
] as const
export type Stage = (typeof STAGES)[number]

export interface ReviewDrug {
  drug: string
  proteins: string[]
  mechanism: string
  fdaStatus: string
  category: string
  trialStatusForTargets?: string
}

export interface Ligand {
  name: string
  smiles: string
  mechanism: string
  /** The literature review's wording, for display. The stage below is what code reads. */
  fdaStatus: string
  source: string
  stage: Stage
  /** "mainstream" or "repurposing_candidate", from the literature review (absent for compounds found by search). */
  category?: string
}

export interface DockingTarget {
  protein: string
  pdbId: string | null
  pdbContentB64: string | null
  ligands: Ligand[]
}

export interface DockingResult {
  name: string
  confidenceScore: number
  confidenceRaw: number
  mechanism: string
  fdaStatus: string
  source: string
  proteinTarget: string
  pdbId: string
  round: number
  allPoses?: unknown[]
  /** Not sent back by the docking worker; added from the ligand by attachStages (lib/stage.ts). */
  stage?: Stage
  category?: string
}

export interface ReasoningDecision {
  action: "expand_3d_similar" | "expand_class" | "proceed"
  rationale: string
  hypothesis: string
  seedCids?: number[]
  drugClass?: string
  drugNames?: string[]
}

// Chat types
export interface ChatMessage {
  id: string
  role: "system" | "user" | "assistant"
  content: string
  timestamp: number
  actions?: ChatAction[]
}

export interface ChatAction {
  id: string
  label: string
  value: string
  variant?: "primary" | "secondary" | "outline"
}

// Docking job event types — emitted by the Modal orchestrator
// (services/docking/gnina_worker.py:run_docking_job) into a per-job event log that
// the frontend polls incrementally via /api/dock/status.
export interface DockJobProgress {
  type: "progress"
  protein: string
  drugIndex: number
  drugTotal: number
  message: string
}

export interface DockJobTargetComplete {
  type: "target_complete"
  protein: string
  results: DockingResult[]
}

export interface DockJobComplete {
  type: "complete"
  allResults: DockingResult[]
}

export interface DockJobError {
  type: "error"
  message: string
}

export type DockJobEvent =
  | DockJobProgress
  | DockJobTargetComplete
  | DockJobComplete
  | DockJobError

export interface DockSubmitResponse {
  jobId: string
}

export interface DockStatusResponse {
  events: DockJobEvent[]
  nextIndex: number
  done: boolean
}

/**
 * Shorten a drug name — if longer than maxLen, fall back to CID_xxx format.
 * Handles IUPAC names, complex chemical names, etc.
 */
export function shortenDrugName(
  name: string | undefined | null,
  cid?: number | null,
  maxLen: number = 30
): string {
  if (!name || name.length > maxLen) {
    return cid ? `CID_${cid}` : name || "Unknown"
  }
  return name
}
