// Retry and failure classification for every external fetch (OpenAI, Exa,
// PubChem, RCSB, PubMed, the GNINA job API, and any Python model service).
//
//   - Transient failures (timeouts, dropped connections, HTTP 408/409/429/5xx)
//     are retried with capped exponential backoff and full jitter, honouring
//     Retry-After. Attempts AND total time are bounded, because a serverless
//     route has a hard duration limit.
//   - Failures that waiting cannot fix (400/401/403/404, and a 429 that means
//     the credit balance or a spend limit is exhausted) are thrown at once.
//   - A call that still fails throws ExternalCallError. "Nothing found" (a
//     status the caller lists in `allowStatus`, e.g. 404 or 204) is returned as
//     a normal Response, so callers can tell "empty" from "failed".

export type FailureKind =
  | "timeout"
  | "connection"
  | "rate_limited"
  | "quota"
  | "auth"
  | "server"
  | "conflict"
  | "client"
  | "bad_response"
  | "not_found"

export class ExternalCallError extends Error {
  readonly source: string
  readonly kind: FailureKind
  readonly status?: number
  readonly attempts: number
  readonly retryable: boolean

  constructor(
    source: string,
    kind: FailureKind,
    message: string,
    details: { status?: number; attempts?: number; retryable?: boolean } = {}
  ) {
    super(message)
    this.name = "ExternalCallError"
    this.source = source
    this.kind = kind
    this.status = details.status
    this.attempts = details.attempts ?? 1
    this.retryable = details.retryable ?? false
  }
}

// A 429 whose body carries one of these codes means "no credit / over a spend
// limit": retrying cannot restore access (OpenAI error-codes guide).
export const BILLING_CODES: ReadonlySet<string> = new Set([
  "insufficient_quota",
  "credit_balance_exhausted",
  "organization_usage_limit_exceeded",
  "organization_spend_limit_exceeded",
  "project_spend_limit_exceeded",
])

export interface RetryPolicy {
  maxAttempts: number
  baseDelayMs: number
  maxDelayMs: number
  /** Cap on attempts + waits together; each attempt's timeout is clamped to what is left. */
  maxTotalMs: number
  /** A longer Retry-After is surfaced to the caller instead of slept on. */
  maxRetryAfterMs: number
}

// Public data APIs (PubChem, RCSB, Exa, PubMed): fail fast, the route has a clock.
export const DEFAULT_POLICY: RetryPolicy = {
  maxAttempts: 3,
  baseDelayMs: 500,
  maxDelayMs: 4_000,
  maxTotalMs: 20_000,
  maxRetryAfterMs: 10_000,
}

export interface FetchRetryOptions {
  /** Name of the dependency, e.g. "openai", "pubchem". Appears in errors and logs. */
  source: string
  /** Statuses that are results, not failures (e.g. [404] for "not found"). */
  allowStatus?: readonly number[]
  /** Per-attempt timeout; defaults to the time left in the total budget. */
  timeoutMs?: number
  policy?: Partial<RetryPolicy>
  // Injection points for tests.
  sleep?: (ms: number) => Promise<void>
  random?: () => number
  now?: () => number
}

interface Failure {
  kind: FailureKind
  retryable: boolean
  status?: number
  retryAfterMs?: number
  detail: string
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

// Anything key-shaped is scrubbed from error text: error bodies can echo a
// (masked) API key, and these messages reach logs, API responses and the UI.
const SECRET = /\b(?:sk|pk|rk)-[A-Za-z0-9_*.-]{3,}|(?:api[_-]?key|token|bearer)[=:\s]+[A-Za-z0-9_*.-]{6,}/gi

export function scrub(text: string): string {
  return text.replace(SECRET, "[redacted]")
}

/**
 * The service's own explanation from an error body (OpenAI `error.message`,
 * PubChem `Fault.Message`, FastAPI `detail` as used by the Modal job API, or
 * short plain text), on one line. Empty if there is none; HTML error pages
 * are not quoted.
 */
export function bodySnippet(body: string, limit = 160): string {
  if (!body || body.trimStart().startsWith("<")) return ""
  let text: string
  try {
    const data = JSON.parse(body) as { error?: unknown; detail?: unknown; Fault?: { Message?: unknown } }
    const err = data?.error
    const message =
      err && typeof err === "object" ? (err as { message?: unknown }).message : err
    const picked = typeof message === "string" ? message : (data?.Fault?.Message ?? data?.detail)
    text = typeof picked === "string" ? picked : ""
  } catch {
    text = body
  }
  text = text.replace(/\s+/g, " ").trim()
  return text.length <= limit ? text : `${text.slice(0, limit - 1)}…`
}

function httpDetail(status: number, body: string): string {
  const snippet = bodySnippet(body)
  return snippet ? `HTTP ${status}: ${snippet}` : `HTTP ${status}`
}

function isBilling(body: string): boolean {
  try {
    const err = (JSON.parse(body) as { error?: { code?: unknown; type?: unknown } }).error
    return [err?.code, err?.type].some((v) => typeof v === "string" && BILLING_CODES.has(v))
  } catch {
    return [...BILLING_CODES].some((code) => body.includes(code))
  }
}

export function classifyStatus(
  status: number,
  body = ""
): { kind: FailureKind; retryable: boolean } {
  if (status === 429) {
    return isBilling(body)
      ? { kind: "quota", retryable: false }
      : { kind: "rate_limited", retryable: true }
  }
  if (status === 408) return { kind: "timeout", retryable: true }
  if (status === 409) return { kind: "conflict", retryable: true }
  if (status >= 500) return { kind: "server", retryable: true }
  if (status === 401 || status === 403) return { kind: "auth", retryable: false }
  return { kind: "client", retryable: false }
}

/** Seconds-or-date `Retry-After` / `retry-after-ms` header as milliseconds. */
export function parseRetryAfterMs(headers: Headers, nowMs = Date.now()): number | undefined {
  const ms = headers.get("retry-after-ms")
  if (ms !== null && ms.trim() !== "" && Number.isFinite(Number(ms))) {
    return Math.max(0, Number(ms))
  }
  const value = headers.get("retry-after")
  if (value === null || value.trim() === "") return undefined
  if (Number.isFinite(Number(value))) return Math.max(0, Number(value) * 1000)
  const date = Date.parse(value)
  return Number.isNaN(date) ? undefined : Math.max(0, date - nowMs)
}

function classifyThrown(err: unknown): Failure | null {
  const name = err instanceof Error ? err.name : ""
  if (name === "TimeoutError" || name === "AbortError") {
    return { kind: "timeout", retryable: true, detail: "timed out" }
  }
  // undici reports every network failure (DNS, reset, refused) as TypeError: fetch failed.
  if (err instanceof TypeError) {
    const cause = (err as { cause?: { code?: string } }).cause
    return { kind: "connection", retryable: true, detail: cause?.code ?? err.message }
  }
  return null
}

// ---------------------------------------------------------------------------
// Retry loop
// ---------------------------------------------------------------------------

/**
 * Wait before the next try, or null to give up now. Full jitter: a uniform
 * draw from [0, min(maxDelay, base * 2^(attempt-1))] spreads out clients that
 * failed together; a server Retry-After is a floor, and one that is longer
 * than `maxRetryAfterMs` is not slept on.
 */
export function backoffDelayMs(
  attempt: number,
  policy: RetryPolicy,
  retryAfterMs: number | undefined,
  random: () => number
): number | null {
  const cap = Math.min(policy.maxDelayMs, policy.baseDelayMs * 2 ** (attempt - 1))
  let delay = random() * cap
  if (retryAfterMs !== undefined) {
    if (retryAfterMs > policy.maxRetryAfterMs) return null
    delay = Math.max(delay, retryAfterMs)
  }
  return delay
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

function describeRequest(url: string, init: RequestInit): string {
  // Strip the query string: it can carry API keys.
  return `${(init.method ?? "GET").toUpperCase()} ${url.split("?")[0]}`
}

/**
 * `fetch(url, init)` with retries. Returns the Response for any status below
 * 400 and for statuses in `allowStatus`; any other status is a failed attempt.
 * Throws ExternalCallError when the failure is permanent or the attempt/time
 * budget runs out. The request must be safe to repeat.
 */
export async function fetchWithRetry(
  url: string,
  init: RequestInit,
  opts: FetchRetryOptions
): Promise<Response> {
  // The default time budget must not undercut a caller's own (longer) per-attempt
  // timeout: leave room for one full-length attempt plus fast-failing retries.
  const defaultTotal = Math.max(DEFAULT_POLICY.maxTotalMs, (opts.timeoutMs ?? 0) + 10_000)
  const policy: RetryPolicy = { ...DEFAULT_POLICY, maxTotalMs: defaultTotal, ...opts.policy }
  const sleep = opts.sleep ?? defaultSleep
  const random = opts.random ?? Math.random
  const now = opts.now ?? Date.now
  const allowed = new Set(opts.allowStatus ?? [])
  const what = `${opts.source} ${describeRequest(url, init)}`
  const started = now()

  for (let attempt = 1; ; attempt++) {
    const remaining = policy.maxTotalMs - (now() - started)
    const timeoutMs = Math.max(1, Math.min(opts.timeoutMs ?? remaining, remaining))
    const timeout = AbortSignal.timeout(timeoutMs)
    const signal = init.signal ? AbortSignal.any([init.signal, timeout]) : timeout

    // Only fetch() itself is guarded: a TypeError from our own code below is a
    // bug, not a network failure, and must not be retried.
    let res: Response | undefined
    let failure: Failure | undefined
    try {
      res = await fetch(url, { ...init, signal })
    } catch (err) {
      // The caller cancelled: that is their decision, not a failure to retry.
      if (init.signal?.aborted) throw err
      failure = classifyThrown(err) ?? undefined
      if (!failure) throw err
    }
    if (res) {
      if (res.status < 400 || allowed.has(res.status)) return res
      const body = await res.text().catch(() => "")
      const { kind, retryable } = classifyStatus(res.status, body)
      failure = {
        kind,
        retryable,
        status: res.status,
        retryAfterMs: parseRetryAfterMs(res.headers, now()),
        detail: httpDetail(res.status, body),
      }
    }
    if (!failure) throw new Error("unreachable: fetch returned neither a response nor a failure")

    const error = new ExternalCallError(
      opts.source,
      failure.kind,
      scrub(`${what}: ${failure.kind} (${failure.detail}) after ${attempt} attempt(s)`),
      { status: failure.status, attempts: attempt, retryable: failure.retryable }
    )
    if (!failure.retryable || attempt >= policy.maxAttempts) throw error
    const delay = backoffDelayMs(attempt, policy, failure.retryAfterMs, random)
    if (delay === null || now() - started + delay >= policy.maxTotalMs) throw error
    console.warn(
      scrub(
        `[retry] ${what}: ${failure.kind} (${failure.detail}); attempt ${attempt}/${policy.maxAttempts}, ` +
          `retrying in ${(delay / 1000).toFixed(1)}s`
      )
    )
    await sleep(delay)
  }
}

/** `res.json()`, throwing ExternalCallError instead of SyntaxError for a non-JSON body. */
export async function responseJson<T = unknown>(res: Response, source: string): Promise<T> {
  try {
    return (await res.json()) as T
  } catch {
    throw new ExternalCallError(source, "bad_response", `${source}: response was not valid JSON`)
  }
}

/**
 * A shared deadline for a route that makes several calls in a row: `budget()`
 * returns the milliseconds left (at least 1). Pass it as each call's
 * `maxTotalMs`, so retries across all of them stay inside the route's
 * `maxDuration` instead of each call spending a full budget of its own.
 */
export function routeBudget(totalMs: number, now: () => number = Date.now): () => number {
  const deadline = now() + totalMs
  return () => Math.max(1, deadline - now())
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/** One short line for an error: no stack, whitespace collapsed, truncated. */
export function compactError(error: unknown, limit = 240): string {
  const message = error instanceof Error ? error.message : String(error)
  const text = scrub(message.replace(/\s+/g, " ").trim())
  return text.length <= limit ? text : `${text.slice(0, limit - 1)}…`
}

export interface FailureWarning {
  source: string
  kind: string
  message: string
  target?: string
}

/** A failure as a warning entry for an API response ({ source, kind, message, target? }). */
export function toWarning(error: unknown, target?: string): FailureWarning {
  if (error instanceof ExternalCallError) {
    return { source: error.source, kind: error.kind, message: compactError(error), target }
  }
  return { source: "server", kind: "unexpected", message: compactError(error), target }
}

/**
 * HTTP status + JSON body for a route's catch block. An upstream failure is a
 * 502 (the caller did nothing wrong, and the body says which dependency and
 * why); anything else is a 500.
 */
export function errorPayload(error: unknown): {
  status: number
  body: { error: string; source?: string; kind?: string }
} {
  if (error instanceof ExternalCallError) {
    return {
      status: 502,
      body: { error: compactError(error), source: error.source, kind: error.kind },
    }
  }
  return {
    status: 500,
    body: { error: error instanceof Error ? error.message : "Unknown error" },
  }
}
