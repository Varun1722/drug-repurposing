// Tests for lib/retry.ts: retry/backoff, failure classification, and the
// helpers routes use to report failures. fetch is mocked; no network, no waiting.

import assert from "node:assert/strict"
import { afterEach, beforeEach, describe, it } from "node:test"

import {
  BILLING_CODES,
  ExternalCallError,
  backoffDelayMs,
  classifyStatus,
  compactError,
  errorPayload,
  fetchWithRetry,
  parseRetryAfterMs,
  responseJson,
  routeBudget,
  toWarning,
  type RetryPolicy,
} from "../lib/retry"
import { instantTimers, mockFetch, networkError, res } from "./_fetch"

const URL_ = "https://example.test/api"
const policy = (over: Partial<RetryPolicy> = {}): Partial<RetryPolicy> => ({
  maxAttempts: 4,
  baseDelayMs: 1000,
  maxDelayMs: 30_000,
  maxTotalMs: 600_000,
  maxRetryAfterMs: 60_000,
  ...over,
})

let fetchMock: ReturnType<typeof mockFetch>
beforeEach(() => {
  fetchMock = mockFetch({})
})
afterEach(() => fetchMock.restore())

/** Run fetchWithRetry against `replies`, recording the waits it asks for. */
async function run(
  replies: Parameters<typeof mockFetch>[0][string],
  over: Partial<RetryPolicy> = {},
  extra: { allowStatus?: number[]; random?: () => number; now?: () => number; init?: RequestInit } = {}
) {
  fetchMock.restore()
  fetchMock = mockFetch({ "example.test": replies })
  const sleeps: number[] = []
  const outcome = await fetchWithRetry(URL_, extra.init ?? {}, {
    source: "svc",
    policy: policy(over),
    allowStatus: extra.allowStatus,
    random: extra.random ?? (() => 1),
    now: extra.now,
    sleep: async (ms) => {
      sleeps.push(ms)
    },
  }).then(
    (response) => ({ response, error: undefined as unknown }),
    (error: unknown) => ({ response: undefined, error })
  )
  return { ...outcome, sleeps, calls: fetchMock.calls.length }
}

const quiet = () => {
  const original = console.warn
  const lines: string[] = []
  console.warn = (...args: unknown[]) => void lines.push(args.join(" "))
  return { lines, restore: () => void (console.warn = original) }
}

describe("fetchWithRetry: retrying", () => {
  let log: ReturnType<typeof quiet>
  beforeEach(() => void (log = quiet()))
  afterEach(() => log.restore())

  it("returns a good response without retrying", async () => {
    const r = await run(res(200, { a: 1 }))
    assert.equal(r.response?.status, 200)
    assert.deepEqual(r.sleeps, [])
    assert.equal(r.calls, 1)
  })

  it("retries transient failures, then succeeds", async () => {
    const r = await run([res(503), res(429), res(200)])
    assert.equal(r.response?.status, 200)
    assert.equal(r.calls, 3)
    assert.equal(r.sleeps.length, 2)
  })

  it("doubles the wait each retry (full jitter at its maximum)", async () => {
    const r = await run(res(503), { maxAttempts: 4, baseDelayMs: 1000 })
    assert.deepEqual(r.sleeps, [1000, 2000, 4000])
  })

  it("scales the wait down with jitter", async () => {
    const r = await run(res(503), { maxAttempts: 4, baseDelayMs: 1000 }, { random: () => 0.5 })
    assert.deepEqual(r.sleeps, [500, 1000, 2000])
  })

  it("caps one wait at maxDelayMs", async () => {
    const r = await run(res(503), { maxAttempts: 6, baseDelayMs: 10_000, maxDelayMs: 25_000 })
    assert.deepEqual(r.sleeps, [10_000, 20_000, 25_000, 25_000, 25_000])
  })

  it("gives up after maxAttempts and says how many were made", async () => {
    const r = await run(res(503), { maxAttempts: 3 })
    const e = r.error
    assert.ok(e instanceof ExternalCallError)
    assert.deepEqual([e.source, e.kind, e.status, e.attempts, e.retryable], ["svc", "server", 503, 3, true])
    assert.match(e.message, /after 3 attempt\(s\)/)
    assert.equal(r.calls, 3)
  })

  it("stops when the total time budget would be exceeded", async () => {
    let t = 0
    const r = await run(
      () => {
        t += 10_000 // every attempt burns 10s
        return res(503)
      },
      { maxAttempts: 10, baseDelayMs: 1000, maxTotalMs: 25_000 },
      { now: () => t }
    )
    // t=10s fail, wait 1s fits (11 < 25) -> t=20s fail, wait 2s fits (22 < 25)
    // -> t=30s fail: a 4s wait would end at 34s, past the 25s budget, so stop.
    assert.ok(r.error instanceof ExternalCallError)
    assert.equal(r.calls, 3)
  })

  it("prints a retry notice without the query string (it can hold an API key)", async () => {
    await run([res(503), res(200)], {}, { init: { method: "GET" } })
    fetchMock.restore()
    fetchMock = mockFetch({ "example.test": [res(503), res(200)] })
    await fetchWithRetry(`${URL_}?api_key=SECRET123`, {}, { source: "svc", sleep: async () => {} })
    assert.ok(log.lines.some((l) => l.includes("[retry] svc GET https://example.test/api")))
    assert.ok(log.lines.every((l) => !l.includes("SECRET123")))
  })

  it("keeps secrets out of the error message too", async () => {
    fetchMock.restore()
    fetchMock = mockFetch({ "example.test": res(503) })
    await assert.rejects(
      fetchWithRetry(`${URL_}?api_key=SECRET123`, {}, { source: "svc", policy: { maxAttempts: 1 } }),
      (e: Error) => !e.message.includes("SECRET123")
    )
  })
})

describe("fetchWithRetry: classification", () => {
  beforeEach(() => void quiet())

  for (const [status, kind, retryable] of [
    [400, "client", false],
    [401, "auth", false],
    [403, "auth", false],
    [404, "client", false],
    [422, "client", false],
    [408, "timeout", true],
    [409, "conflict", true],
    [429, "rate_limited", true],
    [500, "server", true],
    [502, "server", true],
    [503, "server", true],
    [504, "server", true],
  ] as const) {
    it(`HTTP ${status} is ${kind}${retryable ? " and retried" : " and not retried"}`, async () => {
      const r = await run(res(status), { maxAttempts: 3 })
      const e = r.error
      assert.ok(e instanceof ExternalCallError)
      assert.deepEqual([e.kind, e.retryable, e.status], [kind, retryable, status])
      assert.equal(r.calls, retryable ? 3 : 1)
    })
  }

  it("returns an allowed status as a result, not a failure", async () => {
    const r = await run(res(404), {}, { allowStatus: [404] })
    assert.equal(r.response?.status, 404)
    assert.equal(r.calls, 1)
  })

  for (const code of BILLING_CODES) {
    it(`a 429 with ${code} is permanent`, async () => {
      const r = await run(res(429, { error: { code, type: "insufficient_quota" } }))
      const e = r.error
      assert.ok(e instanceof ExternalCallError)
      assert.deepEqual([e.kind, e.retryable, e.attempts], ["quota", false, 1])
      assert.equal(r.calls, 1)
    })
  }

  it("recognises a billing code in a non-JSON body", async () => {
    const r = await run(res(429, "plain text: insufficient_quota"))
    assert.ok(r.error instanceof ExternalCallError && r.error.kind === "quota")
  })

  it("an ordinary 429 is a rate limit", async () => {
    const r = await run(res(429, { error: { code: "rate_limit_exceeded" } }), { maxAttempts: 2 })
    assert.ok(r.error instanceof ExternalCallError && r.error.kind === "rate_limited" && r.calls === 2)
  })

  it("retries network failures", async () => {
    const r = await run(networkError(), { maxAttempts: 3 })
    assert.ok(r.error instanceof ExternalCallError)
    assert.deepEqual([r.error.kind, r.error.attempts], ["connection", 3])
  })

  it("treats timeouts as retryable", async () => {
    const timeout = new DOMException("The operation timed out", "TimeoutError")
    const r = await run(timeout, { maxAttempts: 2 })
    assert.ok(r.error instanceof ExternalCallError && r.error.kind === "timeout" && r.calls === 2)
  })

  it("rethrows errors that are not network failures, unretried", async () => {
    const r = await run(new Error("boom"))
    assert.ok(r.error instanceof Error && !(r.error instanceof ExternalCallError))
    assert.equal(r.calls, 1)
  })

  it("does not mistake a bug in its own post-processing for a network failure", async () => {
    // A response with no headers makes parseRetryAfterMs throw a TypeError. That is
    // a bug in the caller's mock here, and must surface as such, not be retried.
    const broken = { status: 503, text: async () => "", clone: () => broken } as unknown as Response
    const r = await run(broken)
    assert.ok(r.error instanceof TypeError)
    assert.equal(r.calls, 1)
  })

  it("never retries a caller-initiated abort", async () => {
    const controller = new AbortController()
    controller.abort()
    const abort = new DOMException("aborted", "AbortError")
    const r = await run(abort, {}, { init: { signal: controller.signal } })
    assert.ok(r.error === abort, String(r.error))
    assert.equal(r.calls, 1)
  })

  it("gives each attempt a fresh abort signal", async () => {
    const signals: AbortSignal[] = []
    const r = await run(
      (call) => {
        signals.push(call.init.signal as AbortSignal)
        return res(503)
      },
      { maxAttempts: 3 }
    )
    assert.ok(r.error instanceof ExternalCallError)
    assert.equal(signals.length, 3)
    assert.ok(signals.every((s) => s instanceof AbortSignal))
    assert.equal(new Set(signals).size, 3)
  })
})

describe("fetchWithRetry: Retry-After", () => {
  beforeEach(() => void quiet())

  it("uses it as a floor for the wait", async () => {
    const r = await run([res(429, {}, { "retry-after": "5" }), res(200)], {}, { random: () => 0 })
    assert.deepEqual(r.sleeps, [5000])
    assert.equal(r.response?.status, 200)
  })

  it("does not sleep on one longer than maxRetryAfterMs; the caller sees it at once", async () => {
    const r = await run(res(429, {}, { "retry-after": "600" }), { maxRetryAfterMs: 60_000 })
    assert.ok(r.error instanceof ExternalCallError && r.error.kind === "rate_limited")
    assert.deepEqual([r.sleeps, r.calls], [[], 1])
  })

  it("does not sleep on one that does not fit the time budget", async () => {
    const r = await run(res(429, {}, { "retry-after": "50" }), { maxTotalMs: 10_000 })
    assert.ok(r.error instanceof ExternalCallError)
    assert.deepEqual(r.sleeps, [])
  })

  it("parses seconds, milliseconds and dates", () => {
    const now = Date.parse("2015-10-21T07:28:00Z")
    const h = (o: Record<string, string>) => new Headers(o)
    assert.equal(parseRetryAfterMs(h({ "retry-after-ms": "1500" })), 1500)
    assert.equal(parseRetryAfterMs(h({ "retry-after-ms": "1500", "retry-after": "9" })), 1500)
    assert.equal(parseRetryAfterMs(h({ "retry-after": "7" })), 7000)
    assert.equal(parseRetryAfterMs(h({ "retry-after": "2.5" })), 2500)
    assert.equal(parseRetryAfterMs(h({ "retry-after": "-3" })), 0)
    assert.equal(parseRetryAfterMs(h({ "retry-after": "Wed, 21 Oct 2015 07:28:10 GMT" }), now), 10_000)
    assert.equal(parseRetryAfterMs(h({ "retry-after": "Wed, 21 Oct 2015 07:00:00 GMT" }), now), 0)
    assert.equal(parseRetryAfterMs(h({ "retry-after": "soon" })), undefined)
    assert.equal(parseRetryAfterMs(h({})), undefined)
  })
})

describe("backoff and classification helpers", () => {
  it("backoffDelayMs: floor from Retry-After, null when too long", () => {
    const p = { ...(policy() as RetryPolicy) }
    assert.equal(backoffDelayMs(1, p, undefined, () => 0.5), 500)
    assert.equal(backoffDelayMs(1, p, 7000, () => 0), 7000)
    assert.equal(backoffDelayMs(1, p, 90_000, () => 0), null)
  })

  it("classifyStatus", () => {
    assert.deepEqual(classifyStatus(429, '{"error":{"code":"insufficient_quota"}}'), { kind: "quota", retryable: false })
    assert.deepEqual(classifyStatus(429), { kind: "rate_limited", retryable: true })
    assert.deepEqual(classifyStatus(418), { kind: "client", retryable: false })
  })
})

describe("responseJson / reporting helpers", () => {
  it("responseJson wraps a non-JSON body", async () => {
    await assert.rejects(
      responseJson(res(200, "<html>"), "pubchem"),
      (e: unknown) => e instanceof ExternalCallError && e.kind === "bad_response" && e.source === "pubchem"
    )
    assert.deepEqual(await responseJson(res(200, { a: 1 }), "x"), { a: 1 })
  })

  it("compactError collapses whitespace and truncates", () => {
    assert.equal(compactError(new Error("a\n  b\t c")), "a b c")
    assert.equal(compactError("x".repeat(1000), 50).length, 50)
    assert.ok(compactError("x".repeat(1000), 50).endsWith("…"))
  })

  it("toWarning describes an upstream failure and a plain bug", () => {
    const upstream = new ExternalCallError("pubchem", "server", "pubchem: server (HTTP 503)")
    assert.deepEqual(toWarning(upstream, "KRAS/x"), {
      source: "pubchem",
      kind: "server",
      message: "pubchem: server (HTTP 503)",
      target: "KRAS/x",
    })
    assert.deepEqual(toWarning(new Error("bug")), { source: "server", kind: "unexpected", message: "bug", target: undefined })
  })

  it("errorPayload: upstream failures are 502 with source and kind, anything else 500", () => {
    const upstream = new ExternalCallError("openai", "rate_limited", "openai: rate_limited")
    assert.deepEqual(errorPayload(upstream), {
      status: 502,
      body: { error: "openai: rate_limited", source: "openai", kind: "rate_limited" },
    })
    assert.deepEqual(errorPayload(new Error("bug")), { status: 500, body: { error: "bug" } })
    assert.deepEqual(errorPayload("weird"), { status: 500, body: { error: "Unknown error" } })
  })

  it("routeBudget counts down and never reaches zero", () => {
    let t = 1000
    const budget = routeBudget(10_000, () => t)
    assert.equal(budget(), 10_000)
    t += 4000
    assert.equal(budget(), 6000)
    t += 99_999
    assert.equal(budget(), 1)
  })
})

describe("real timers are restored by the helper", () => {
  it("instantTimers records waits and restores setTimeout", async () => {
    const original = globalThis.setTimeout
    const timers = instantTimers()
    await new Promise<void>((resolve) => setTimeout(resolve, 5000))
    assert.deepEqual(timers.waits, [5000])
    timers.restore()
    assert.equal(globalThis.setTimeout, original)
  })
})

describe("error text: the service's own explanation, and no secrets", () => {
  beforeEach(() => void quiet())

  const fail = async (status: number, body: unknown) => (await run(res(status, body), { maxAttempts: 1 })).error as Error

  it("quotes OpenAI's error.message and PubChem's Fault.Message", async () => {
    assert.match((await fail(404, { error: { message: "The model `nope` does not exist" } })).message, /client \(HTTP 404: The model `nope` does not exist\)/)
    assert.match((await fail(400, { Fault: { Code: "PUGREST.BadRequest", Message: "Invalid name" } })).message, /HTTP 400: Invalid name/)
  })

  it("does not quote HTML or empty bodies", async () => {
    for (const body of ["<html>502</html>", "", {}, { error: {} }]) {
      assert.match((await fail(400, body)).message, /\(HTTP 400\)/, JSON.stringify(body))
    }
  })

  it("truncates a long explanation", async () => {
    const message = (await fail(400, { error: { message: "x".repeat(1000) } })).message
    assert.ok(message.length < 300 && message.includes("…"))
  })

  it("scrubs key-shaped text from errors and from compactError", async () => {
    for (const leak of [
      "Incorrect API key provided: sk-proj-abc***xyz1. You can find your API key",
      "bad credentials api_key=abcdef123456 rejected",
      "Authorization: Bearer abcdef123456",
    ]) {
      const shown = (await fail(401, { error: { message: leak } })).message + compactError(leak)
      assert.ok(shown.includes("[redacted]"), shown)
      assert.ok(!shown.includes("sk-proj-abc") && !shown.includes("abcdef123456"), shown)
    }
  })

  it("leaves ordinary words that contain 'sk' alone", () => {
    assert.equal(compactError("task-based risk-averse desk-top"), "task-based risk-averse desk-top")
  })
})


describe("time budget vs. the caller's own timeout", () => {
  beforeEach(() => void quiet())

  /** Record the per-attempt timeouts fetchWithRetry asks AbortSignal.timeout for. */
  const spyTimeouts = () => {
    const real = AbortSignal.timeout
    const asked: number[] = []
    AbortSignal.timeout = ((ms: number) => {
      asked.push(ms)
      return real.call(AbortSignal, ms)
    }) as typeof AbortSignal.timeout
    return { asked, restore: () => void (AbortSignal.timeout = real) }
  }

  it("does not clamp an explicit 60s attempt timeout to the 20s default budget", async () => {
    const spy = spyTimeouts()
    fetchMock.restore()
    fetchMock = mockFetch({ "example.test": res(200) })
    await fetchWithRetry(URL_, {}, { source: "svc", timeoutMs: 60_000 })
    spy.restore()
    assert.deepEqual(spy.asked, [60_000])
  })

  it("still leaves room for a retry after a fast failure", async () => {
    const spy = spyTimeouts()
    fetchMock.restore()
    fetchMock = mockFetch({ "example.test": [res(503), res(200)] })
    await fetchWithRetry(URL_, {}, { source: "svc", timeoutMs: 60_000, sleep: async () => {} })
    spy.restore()
    assert.equal(spy.asked.length, 2)
  })

  it("an explicit policy.maxTotalMs still wins", async () => {
    const spy = spyTimeouts()
    fetchMock.restore()
    fetchMock = mockFetch({ "example.test": res(200) })
    await fetchWithRetry(URL_, {}, { source: "svc", timeoutMs: 60_000, policy: { maxTotalMs: 5_000 } })
    spy.restore()
    assert.ok(spy.asked[0] <= 5_000)
  })
})
