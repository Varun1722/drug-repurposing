// lib/retry.ts against a real local HTTP server and the real fetch (no mocks).
//
// The mocked tests prove the logic; these prove it holds for the real thing:
// Response header parsing, per-attempt timeouts via AbortSignal, refused
// connections, request bodies resent on retry. Delays are tiny.

import assert from "node:assert/strict"
import { createServer, type IncomingMessage, type Server } from "node:http"
import { type AddressInfo, createServer as createNetServer } from "node:net"
import { afterEach, describe, it } from "node:test"

import { ExternalCallError, fetchWithRetry, type RetryPolicy } from "../lib/retry"

const FAST: Partial<RetryPolicy> = { maxAttempts: 4, baseDelayMs: 10, maxDelayMs: 20, maxTotalMs: 30_000 }

interface Reply {
  status: number
  headers?: Record<string, string>
  body?: unknown
  delayMs?: number
}

const servers: Server[] = []
afterEach(async () => {
  await Promise.all(servers.splice(0).map((s) => new Promise((resolve) => s.close(resolve))))
})

/** Serves `script` one reply per request (the last repeats) and records requests. */
async function serve(...script: Reply[]) {
  const received: { path: string; body: string; headers: IncomingMessage["headers"] }[] = []
  const queue = [...script]
  const server = createServer((req, res) => {
    const chunks: Buffer[] = []
    req.on("data", (c) => chunks.push(c))
    req.on("end", () => {
      received.push({ path: req.url ?? "", body: Buffer.concat(chunks).toString(), headers: req.headers })
      const reply = queue.length > 1 ? queue.shift()! : queue[0]
      setTimeout(() => {
        const payload = JSON.stringify(reply.body ?? {})
        res.writeHead(reply.status, { "content-type": "application/json", ...reply.headers })
        res.end(payload)
      }, reply.delayMs ?? 0)
    })
  })
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve))
  servers.push(server)
  const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`
  return { url, received }
}

const quiet = () => {
  const original = console.warn
  console.warn = () => {}
  return () => void (console.warn = original)
}

describe("fetchWithRetry over a real socket", () => {
  it("returns a real response", async () => {
    const s = await serve({ status: 200, body: { ok: true } })
    const res = await fetchWithRetry(`${s.url}/x`, {}, { source: "svc", policy: FAST })
    assert.deepEqual(await res.json(), { ok: true })
  })

  it("parses a real retry-after-ms header and waits for it", async () => {
    const restore = quiet()
    const s = await serve({ status: 429, headers: { "retry-after-ms": "150" } }, { status: 200 })
    const started = Date.now()
    const res = await fetchWithRetry(s.url, {}, { source: "svc", policy: FAST })
    restore()
    assert.equal(res.status, 200)
    assert.equal(s.received.length, 2)
    assert.ok(Date.now() - started >= 140, "should have waited what the server asked")
  })

  it("parses a real Retry-After (seconds) header", async () => {
    const restore = quiet()
    const s = await serve({ status: 503, headers: { "Retry-After": "1" } }, { status: 200 })
    const started = Date.now()
    await fetchWithRetry(s.url, {}, { source: "svc", policy: FAST })
    restore()
    assert.ok(Date.now() - started >= 950)
  })

  it("exhausts its attempts on a persistent 503, resending the body each time", async () => {
    const restore = quiet()
    const s = await serve({ status: 503 })
    await assert.rejects(
      fetchWithRetry(s.url, { method: "POST", body: JSON.stringify({ a: 1 }) }, { source: "svc", policy: FAST }),
      (e: unknown) => e instanceof ExternalCallError && e.kind === "server" && e.status === 503 && e.attempts === 4
    )
    restore()
    assert.equal(s.received.length, 4)
    assert.ok(s.received.every((r) => r.body === '{"a":1}'))
  })

  it("404 is a result when allowed and an error when not", async () => {
    const s = await serve({ status: 404 })
    assert.equal((await fetchWithRetry(s.url, {}, { source: "svc", policy: FAST, allowStatus: [404] })).status, 404)
    await assert.rejects(
      fetchWithRetry(s.url, {}, { source: "svc", policy: FAST }),
      (e: unknown) => e instanceof ExternalCallError && e.kind === "client"
    )
    assert.equal(s.received.length, 2) // one try each: 404 is not retried
  })

  it("204 No Content (RCSB's 'no matches') comes back as a response", async () => {
    const s = await serve({ status: 204 })
    const res = await fetchWithRetry(s.url, {}, { source: "rcsb", policy: FAST, allowStatus: [204] })
    assert.equal(res.status, 204)
  })

  it("retries a refused connection, then reports it as 'connection'", async () => {
    const restore = quiet()
    const probe = createNetServer()
    await new Promise<void>((resolve) => probe.listen(0, "127.0.0.1", resolve))
    const port = (probe.address() as AddressInfo).port
    await new Promise((resolve) => probe.close(resolve)) // nothing listens on it now
    await assert.rejects(
      fetchWithRetry(`http://127.0.0.1:${port}`, {}, { source: "svc", policy: FAST }),
      (e: unknown) => e instanceof ExternalCallError && e.kind === "connection" && e.attempts === 4
    )
    restore()
  })

  it("times a slow attempt out, retries, and recovers", async () => {
    const restore = quiet()
    const s = await serve({ status: 200, delayMs: 600 }, { status: 200, body: { fast: true } })
    const res = await fetchWithRetry(s.url, {}, { source: "svc", policy: FAST, timeoutMs: 200 })
    restore()
    assert.deepEqual(await res.json(), { fast: true })
    assert.equal(s.received.length, 2)
  })

  it("a call that always times out ends as a 'timeout' error", async () => {
    const restore = quiet()
    const s = await serve({ status: 200, delayMs: 500 })
    await assert.rejects(
      fetchWithRetry(s.url, {}, { source: "svc", policy: { ...FAST, maxAttempts: 2 }, timeoutMs: 100 }),
      (e: unknown) => e instanceof ExternalCallError && e.kind === "timeout" && e.attempts === 2
    )
    restore()
  })

  it("an exhausted-credit 429 is reported at once, read from the real body", async () => {
    const s = await serve({
      status: 429,
      body: { error: { code: "credit_balance_exhausted", type: "insufficient_quota" } },
    })
    await assert.rejects(
      fetchWithRetry(s.url, {}, { source: "openai", policy: FAST }),
      (e: unknown) => e instanceof ExternalCallError && e.kind === "quota" && e.attempts === 1
    )
    assert.equal(s.received.length, 1)
  })

  it("the total time budget clamps a hung attempt, so a route can still answer", async () => {
    const restore = quiet()
    const s = await serve({ status: 200, delayMs: 2000 })
    const started = Date.now()
    await assert.rejects(
      fetchWithRetry(s.url, {}, { source: "svc", policy: { ...FAST, maxTotalMs: 300 } }),
      (e: unknown) => e instanceof ExternalCallError && e.kind === "timeout"
    )
    restore()
    assert.ok(Date.now() - started < 1500, "gave up near the budget, not after the server's 2s")
  })
})
