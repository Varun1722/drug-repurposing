// Splitting a docking run into jobs the Modal worker accepts (lib/dock-batches.ts).
// The worker rejects a job over 10 targets or 50 (target, ligand) pairs, and Vercel
// rejects a request body over 4.5 MB, so one big run has to become several jobs.

import assert from "node:assert/strict"
import { existsSync, readFileSync } from "node:fs"
import { describe, it } from "node:test"

import { DOCK_LIMITS, chunkDockTargets, dockInBatches } from "../lib/dock-batches"
import type { DockingResult, DockingTarget, Ligand } from "../lib/types"

const ligand = (name: string): Ligand => ({
  name,
  smiles: "CCO",
  mechanism: "",
  fdaStatus: "",
  source: "",
})

/** A target named `protein` with `n` ligands and a receptor of `pdbChars` characters. */
const target = (protein: string, n: number, pdbChars = 1000): DockingTarget => ({
  protein,
  pdbId: "1ABC",
  pdbContentB64: "A".repeat(pdbChars),
  ligands: Array.from({ length: n }, (_, i) => ligand(`${protein}-drug${i}`)),
})

const pairs = (batch: DockingTarget[]) => batch.reduce((sum, t) => sum + t.ligands.length, 0)
const bytes = (batch: DockingTarget[]) => batch.reduce((sum, t) => sum + (t.pdbContentB64?.length ?? 0), 0)
const names = (batches: DockingTarget[][]) =>
  batches.flatMap((b) => b.flatMap((t) => t.ligands.map((l) => l.name)))

function assertWithinLimits(batches: DockingTarget[][]) {
  for (const batch of batches) {
    assert.ok(batch.length >= 1, "no empty batches")
    assert.ok(batch.length <= DOCK_LIMITS.maxTargets, `${batch.length} targets in one batch`)
    assert.ok(pairs(batch) <= DOCK_LIMITS.maxPairs, `${pairs(batch)} pairs in one batch`)
  }
}

describe("chunkDockTargets", () => {
  it("returns nothing for nothing", () => {
    assert.deepEqual(chunkDockTargets([]), [])
    assert.deepEqual(chunkDockTargets([target("EGFR", 0)]), [])
  })

  it("keeps a small run as one batch", () => {
    const batches = chunkDockTargets([target("EGFR", 5), target("KRAS", 5)])
    assert.equal(batches.length, 1)
    assert.equal(batches[0].length, 2)
  })

  it("splits 4 proteins x 15 pairs (60 pairs, the case the worker rejects) into 2 batches", () => {
    const input = ["EGFR", "KRAS", "BRAF", "MET"].map((p) => target(p, 15))
    const batches = chunkDockTargets(input)
    assert.equal(batches.length, 2)
    assertWithinLimits(batches)
    assert.equal(names(batches).length, 60)
    assert.equal(new Set(names(batches)).size, 60, "every pair exactly once")
  })

  it("splits by target count: 12 targets of one ligand each is 10 + 2", () => {
    const batches = chunkDockTargets(Array.from({ length: 12 }, (_, i) => target(`P${i}`, 1)))
    assert.deepEqual(batches.map((b) => b.length), [10, 2])
  })

  it("splits one protein with 120 ligands into pieces that each carry the receptor", () => {
    const batches = chunkDockTargets([target("EGFR", 120)])
    assert.deepEqual(batches.map(pairs), [50, 50, 20])
    for (const batch of batches) {
      assert.equal(batch.length, 1)
      assert.equal(batch[0].protein, "EGFR")
      assert.equal(batch[0].pdbId, "1ABC")
      assert.equal(batch[0].pdbContentB64?.length, 1000)
    }
    assert.deepEqual(names(batches), target("EGFR", 120).ligands.map((l) => l.name), "order kept")
  })

  it("fills a batch before starting the next one", () => {
    // 30 + 30: the second does not fit beside the first, so it starts a new batch.
    assert.deepEqual(chunkDockTargets([target("A", 30), target("B", 30)]).map(pairs), [30, 30])
    // 30 + 20 fits exactly in 50.
    assert.deepEqual(chunkDockTargets([target("A", 30), target("B", 20)]).map(pairs), [50])
  })

  it("splits by receptor size so the request body stays under Vercel's limit", () => {
    const big = 1_500_000
    const batches = chunkDockTargets([target("A", 1, big), target("B", 1, big), target("C", 1, big)])
    assert.deepEqual(batches.map((b) => b.length), [2, 1])
    for (const batch of batches) assert.ok(bytes(batch) <= DOCK_LIMITS.maxBytes)
  })

  it("puts a receptor that is over the size limit by itself rather than dropping it", () => {
    const batches = chunkDockTargets([target("A", 1, 10), target("HUGE", 1, DOCK_LIMITS.maxBytes + 1), target("B", 1, 10)])
    assert.equal(batches.length, 3)
    assert.equal(batches[1][0].protein, "HUGE")
  })

  it("keeps targets without a structure (the worker reports those itself)", () => {
    const noPdb: DockingTarget = { ...target("NOPDB", 2), pdbId: null, pdbContentB64: null }
    assert.deepEqual(names(chunkDockTargets([noPdb])), ["NOPDB-drug0", "NOPDB-drug1"])
  })

  it("does not change its input", () => {
    const input = [target("EGFR", 120), target("KRAS", 10)]
    const before = JSON.stringify(input)
    chunkDockTargets(input)
    assert.equal(JSON.stringify(input), before)
  })

  it("holds the limits and loses no pair, for many random runs", () => {
    let seed = 12345
    const rand = (n: number) => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff
      return seed % n
    }
    for (let run = 0; run < 300; run++) {
      const input = Array.from({ length: rand(15) }, (_, i) => target(`P${i}`, rand(130), rand(2_000_000)))
      const batches = chunkDockTargets(input)
      assertWithinLimits(batches)
      const want = input.flatMap((t) => t.ligands.map((l) => l.name)).sort()
      assert.deepEqual(names(batches).sort(), want, `run ${run}: every pair exactly once`)
      for (const batch of batches) {
        // Over the byte limit only when a single receptor is itself too big.
        if (bytes(batch) > DOCK_LIMITS.maxBytes) assert.equal(batch.length, 1, `run ${run}`)
      }
    }
  })

  it("agrees with the worker's caps (skipped if the worker is not in this checkout)", (t) => {
    const worker = new URL("../../services/docking/gnina_worker.py", import.meta.url)
    if (!existsSync(worker)) return t.skip("services/docking/gnina_worker.py not found")
    const source = readFileSync(worker, "utf8")
    const targets = Number(/^MAX_TARGETS_PER_JOB\s*=\s*(\d+)/m.exec(source)?.[1])
    const ligands = Number(/^MAX_LIGANDS_PER_JOB\s*=\s*(\d+)/m.exec(source)?.[1])
    assert.equal(DOCK_LIMITS.maxTargets, targets, "MAX_TARGETS_PER_JOB changed: update DOCK_LIMITS")
    assert.equal(DOCK_LIMITS.maxPairs, ligands, "MAX_LIGANDS_PER_JOB changed: update DOCK_LIMITS")
  })
})

describe("dockInBatches", () => {
  const result = (name: string): DockingResult => ({
    name,
    confidenceScore: 0.5,
    confidenceRaw: -7,
    mechanism: "",
    fdaStatus: "",
    source: "",
    proteinTarget: "EGFR",
    pdbId: "1ABC",
    round: 1,
  })
  const batches = [[target("A", 1)], [target("B", 1)], [target("C", 1)]]

  it("docks the batches in order and joins their results", async () => {
    const order: number[] = []
    const out = await dockInBatches({
      batches,
      dockBatch: async (batch, index) => {
        order.push(index)
        return [result(batch[0].protein)]
      },
    })
    assert.deepEqual(order, [0, 1, 2])
    assert.deepEqual(out.results.map((r) => r.name), ["A", "B", "C"])
    assert.equal(out.stoppedAt, undefined)
  })

  it("never runs two batches at once (they share Modal's container limit)", async () => {
    let running = 0
    let peak = 0
    await dockInBatches({
      batches,
      dockBatch: async () => {
        peak = Math.max(peak, ++running)
        await new Promise((resolve) => setImmediate(resolve))
        running--
        return []
      },
    })
    assert.equal(peak, 1)
  })

  it("keeps earlier results and says where it stopped when a later batch fails", async () => {
    let calls = 0
    const out = await dockInBatches({
      batches,
      dockBatch: async (batch, index) => {
        calls++
        if (index === 1) throw new Error("Modal said no")
        return [result(batch[0].protein)]
      },
    })
    assert.equal(calls, 2, "batch 3 is not attempted")
    assert.deepEqual(out.results.map((r) => r.name), ["A"])
    assert.deepEqual(out.stoppedAt, { batch: 2, of: 3, message: "Modal said no" })
  })

  it("throws when the first batch fails, since there is nothing to keep", async () => {
    await assert.rejects(
      dockInBatches({
        batches,
        dockBatch: async () => {
          throw new Error("could not submit")
        },
      }),
      /could not submit/
    )
  })

  it("describes a non-Error failure", async () => {
    const out = await dockInBatches({
      batches,
      dockBatch: async (_batch, index) => {
        if (index === 1) throw "boom"
        return []
      },
    })
    assert.equal(out.stoppedAt?.message, "unknown error")
  })

  it("does nothing for no batches", async () => {
    assert.deepEqual(await dockInBatches({ batches: [], dockBatch: async () => [result("x")] }), { results: [] })
  })
})
