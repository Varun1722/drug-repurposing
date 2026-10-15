// Splits one docking run into jobs the Modal worker will accept, and runs them
// one after another. Kept out of the page component so it can be tested without
// a browser (like lib/dock-polling.ts).
//
// Why a run has to be split: the worker rejects a job with more than
// MAX_TARGETS_PER_JOB targets or MAX_LIGANDS_PER_JOB (target, ligand) pairs
// (services/docking/gnina_worker.py), and Vercel rejects a request body over
// 4.5 MB. A normal run (4 proteins x 15 compounds is already 60 pairs) is over
// the first, and a few large receptors are over the second.
//
// Batches run one at a time: they share the workspace's container limit (100 on
// Modal's Starter plan, and one job uses up to 50 at once).

import type { DockingResult, DockingTarget } from "./types"

export const DOCK_LIMITS = {
  /** Mirrors MAX_TARGETS_PER_JOB in services/docking/gnina_worker.py (a test checks this). */
  maxTargets: 10,
  /** Mirrors MAX_LIGANDS_PER_JOB in services/docking/gnina_worker.py (a test checks this). */
  maxPairs: 50,
  /** Receptor text per request. Under Vercel's 4.5 MB body limit, with room for the ligands. */
  maxBytes: 3_500_000,
} as const

/**
 * Groups targets into batches that each stay within DOCK_LIMITS, keeping their
 * order. A target with more ligands than one batch holds is split into pieces
 * that each carry the same receptor. A single receptor over `maxBytes` gets a
 * batch of its own (it cannot be split) rather than being dropped.
 */
export function chunkDockTargets(
  targets: readonly DockingTarget[],
  limits: typeof DOCK_LIMITS = DOCK_LIMITS
): DockingTarget[][] {
  const batches: DockingTarget[][] = []
  let batch: DockingTarget[] = []
  let pairs = 0
  let bytes = 0

  for (const target of targets) {
    const size = target.pdbContentB64?.length ?? 0
    for (let i = 0; i < target.ligands.length; i += limits.maxPairs) {
      const piece = { ...target, ligands: target.ligands.slice(i, i + limits.maxPairs) }
      const full =
        batch.length >= limits.maxTargets ||
        pairs + piece.ligands.length > limits.maxPairs ||
        bytes + size > limits.maxBytes
      if (batch.length > 0 && full) {
        batches.push(batch)
        batch = []
        pairs = 0
        bytes = 0
      }
      batch.push(piece)
      pairs += piece.ligands.length
      bytes += size
    }
  }
  if (batch.length > 0) batches.push(batch)
  return batches
}

export interface DockBatchOutcome {
  /** Results of every batch that finished, in batch order. */
  results: DockingResult[]
  /** Set when a batch failed after earlier ones finished: their results are kept. */
  stoppedAt?: { batch: number; of: number; message: string }
}

export interface DockInBatchesOptions {
  batches: readonly DockingTarget[][]
  /** Docks one batch and returns its results. Throws if the batch fails. */
  dockBatch: (batch: DockingTarget[], index: number) => Promise<DockingResult[]>
}

/**
 * Runs the batches one at a time. If the first batch fails the error is thrown,
 * since there is nothing to keep. If a later one fails, the earlier results are
 * returned together with where it stopped, and the rest are not attempted.
 */
export async function dockInBatches({ batches, dockBatch }: DockInBatchesOptions): Promise<DockBatchOutcome> {
  const results: DockingResult[] = []
  for (const [index, batch] of batches.entries()) {
    try {
      results.push(...(await dockBatch(batch, index)))
    } catch (error) {
      if (index === 0) throw error
      const message = error instanceof Error ? error.message : "unknown error"
      return { results, stoppedAt: { batch: index + 1, of: batches.length, message } }
    }
  }
  return { results }
}
