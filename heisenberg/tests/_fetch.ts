// Shared helpers for tests that mock fetch. Not a test file (no .test suffix).

export interface Call {
  url: string
  init: RequestInit
}

type Reply = Response | Error | ((call: Call) => Response | Error | Promise<Response | Error>)

/** A Response with a JSON body (or `body` as given), status and optional headers. */
export function res(status: number, body: unknown = {}, headers: Record<string, string> = {}): Response {
  const text = typeof body === "string" ? body : JSON.stringify(body)
  // 204/304 may not carry a body.
  return new Response(status === 204 ? null : text, { status, headers })
}

/**
 * Replace globalThis.fetch. `routes` are matched against the request URL by
 * substring, first match wins; a route value may be one reply, an array
 * (consumed one per request, the last repeating), or a function. Returns the
 * list of calls and a restore function.
 */
export function mockFetch(routes: Record<string, Reply | Reply[]>) {
  const realFetch = globalThis.fetch
  const calls: Call[] = []
  const queues = new Map<string, Reply[]>()
  for (const [needle, value] of Object.entries(routes)) {
    queues.set(needle, Array.isArray(value) ? [...value] : [value])
  }

  globalThis.fetch = (async (input: unknown, init: RequestInit = {}) => {
    const url = String(input)
    const call = { url, init }
    calls.push(call)
    for (const [needle, queue] of queues) {
      if (!url.includes(needle)) continue
      const reply = queue.length > 1 ? queue.shift()! : queue[0]
      const out = typeof reply === "function" ? await reply(call) : reply
      if (out instanceof Error) throw out
      // A Response body can be read once; hand out a fresh copy for repeated replies.
      return out.clone()
    }
    throw new Error(`mockFetch: no route for ${url}`)
  }) as unknown as typeof fetch

  return {
    calls,
    count: (needle: string) => calls.filter((c) => c.url.includes(needle)).length,
    restore: () => {
      globalThis.fetch = realFetch
    },
  }
}

/** Make setTimeout run immediately (recording the waits), so backoff costs no time. */
export function instantTimers() {
  const realSetTimeout = globalThis.setTimeout
  const waits: number[] = []
  globalThis.setTimeout = ((fn: () => void, ms?: number) => {
    waits.push(ms ?? 0)
    fn()
    return 0
  }) as unknown as typeof setTimeout
  return {
    waits,
    restore: () => {
      globalThis.setTimeout = realSetTimeout
    },
  }
}

/** The error a network failure looks like in Node's fetch. */
export const networkError = () =>
  Object.assign(new TypeError("fetch failed"), { cause: { code: "ECONNRESET" } })
