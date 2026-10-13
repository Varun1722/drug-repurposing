// The docking status poll loop (lib/dockPolling.ts): a failed poll is retried
// on the next tick, a run of failures gives up, a 4xx gives up at once.

import assert from "node:assert/strict"
import { describe, it } from "node:test"

import { MAX_FAILED_POLLS, pollDockJobUntilDone } from "../lib/dockPolling"
import type { DockJobEvent } from "../lib/types"
import { networkError, res } from "./_fetch"

const progress = (message: string): DockJobEvent => ({ type: "progress", protein: "KRAS", drugIndex: 1, drugTotal: 2, message })
const ok = (events: DockJobEvent[], nextIndex: number, done: boolean) => res(200, { events, nextIndex, done })

/** Run the loop against scripted poll outcomes; returns what the page would have seen. */
async function run(script: Array<Response | Error>, over: { maxFailedPolls?: number } = {}) {
  const queue = [...script]
  const requestedSince: number[] = []
  const delivered: DockJobEvent[][] = []
  const retries: Array<[number, number, string]> = []
  const sleeps: number[] = []

  const outcome = await pollDockJobUntilDone({
    jobId: "job-1",
    fetchStatus: async (since) => {
      requestedSince.push(since)
      const next = queue.shift()
      if (next === undefined) throw new Error("test script ran out")
      if (next instanceof Error) throw next
      return next
    },
    onEvents: (events) => void delivered.push(events),
    onRetry: (failed, max, reason) => void retries.push([failed, max, reason]),
    sleep: async (ms) => void sleeps.push(ms),
    ...over,
  }).then(
    () => ({ error: undefined as Error | undefined }),
    (error: Error) => ({ error })
  )
  return { ...outcome, requestedSince, delivered, retries, sleeps, unused: queue.length }
}

describe("pollDockJobUntilDone", () => {
  it("polls until the job is done, passing the cursor and delivering events in order", async () => {
    const r = await run([ok([progress("a")], 1, false), ok([progress("b"), progress("c")], 3, false), ok([], 3, true)])
    assert.equal(r.error, undefined)
    assert.deepEqual(r.requestedSince, [0, 1, 3])
    assert.deepEqual(r.delivered.map((e) => e.length), [1, 2, 0])
    assert.deepEqual(r.sleeps, [2000, 2000, 2000]) // waits before every poll
  })

  it("a single failed poll (5xx) is retried on the next tick and the job carries on", async () => {
    const r = await run([ok([progress("a")], 1, false), res(503, { error: "gnina: server" }), ok([progress("b")], 2, true)])
    assert.equal(r.error, undefined)
    assert.deepEqual(r.requestedSince, [0, 1, 1]) // the failed poll did not advance the cursor
    assert.equal(r.retries.length, 1)
    assert.deepEqual(r.retries[0], [1, MAX_FAILED_POLLS, "gnina: server"])
    assert.equal(r.delivered.length, 2)
  })

  it("a network error is also retried", async () => {
    const r = await run([networkError(), ok([], 0, true)])
    assert.equal(r.error, undefined)
    assert.deepEqual(r.retries.map((x) => x[0]), [1])
  })

  it("the failure count resets after a success, so scattered failures never add up", async () => {
    const script = [res(503), res(503), ok([], 0, false), res(503), res(503), ok([], 0, false), res(503), ok([], 0, true)]
    const r = await run(script, { maxFailedPolls: 3 })
    assert.equal(r.error, undefined)
    assert.equal(r.unused, 0)
  })

  it("gives up after maxFailedPolls consecutive failures, naming the job and the last reason", async () => {
    const r = await run(Array.from({ length: 10 }, () => res(502, { error: "bad gateway" })), { maxFailedPolls: 3 })
    assert.match(r.error!.message, /Lost contact with docking job job-1 after 3 failed status checks \(bad gateway\)/)
    assert.match(r.error!.message, /may still be running on Modal/)
    assert.equal(r.requestedSince.length, 3)
    assert.deepEqual(r.retries.map((x) => x[0]), [1, 2]) // told about retries 1 and 2, then gave up
  })

  it("uses the default limit when none is given", async () => {
    const r = await run(Array.from({ length: 20 }, () => networkError()))
    assert.equal(r.requestedSince.length, MAX_FAILED_POLLS)
    assert.ok(r.error)
  })

  it("a 4xx gives up at once: a bad or unknown job id will not fix itself", async () => {
    const r = await run([res(404, { error: "unknown job" }), ok([], 0, true)])
    assert.equal(r.error?.message, "Dock status error: unknown job")
    assert.equal(r.requestedSince.length, 1)
    assert.deepEqual(r.retries, [])
  })

  it("408 and 429 are transient, unlike other 4xx", async () => {
    const r = await run([res(429, { error: "slow down" }), res(408), ok([], 0, true)])
    assert.equal(r.error, undefined)
    assert.deepEqual(r.retries.map((x) => x[0]), [1, 2])
  })

  it("an unreadable 200 body counts as a failed poll instead of abandoning the job", async () => {
    const r = await run([res(200, "{not json"), ok([progress("a")], 1, true)])
    assert.equal(r.error, undefined)
    assert.equal(r.retries[0][2], "unreadable status response")
    assert.equal(r.delivered.length, 1)
  })

  it("tolerates about 30s of failures by default", () => {
    assert.ok(MAX_FAILED_POLLS * 2000 >= 30_000)
  })

  it("falls back to the HTTP status when the error body is unusable", async () => {
    const r = await run([res(400, "<html>")])
    assert.equal(r.error?.message, "Dock status error: HTTP 400")
  })

  it("a 5xx with an unusable body is retried with the status as the reason", async () => {
    const r = await run([res(500, "<html>"), ok([], 0, true)])
    assert.equal(r.error, undefined)
    assert.equal(r.retries[0][2], "HTTP 500")
  })

  it("honours a custom interval", async () => {
    const r = await run([ok([], 0, true)])
    assert.deepEqual(r.sleeps, [2000])
    const queue = [ok([], 0, true)]
    const sleeps: number[] = []
    await pollDockJobUntilDone({ jobId: "j", fetchStatus: async () => queue.shift()!, onEvents: () => {}, sleep: async (ms) => void sleeps.push(ms), intervalMs: 50 })
    assert.deepEqual(sleeps, [50])
  })
})
