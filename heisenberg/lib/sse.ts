// Reads the server-sent-event streams of /api/review and /api/structures. Kept
// out of the page component so it can be tested without a browser (like
// lib/dock-polling.ts).
//
// Contract: a stream ends with a `complete` event. Anything else is an error the
// caller sees, never a normal end, or the page carries on with half the data:
//   - an `error` event from the server throws StreamError("server") with its message;
//   - a stream that stops without `complete` (a connection cut at the platform's
//     time limit) throws StreamError("incomplete");
//   - an exception from the handler passes through unchanged.
// Lines that are not parseable events are skipped.

export class StreamError extends Error {
  constructor(
    readonly kind: "server" | "incomplete",
    message: string
  ) {
    super(message)
    this.name = "StreamError"
  }
}

function parseEvent(frame: string): Record<string, unknown> | null {
  if (!frame.startsWith("data: ")) return null
  try {
    const event: unknown = JSON.parse(frame.slice(6))
    return event !== null && typeof event === "object" && !Array.isArray(event)
      ? (event as Record<string, unknown>)
      : null
  } catch {
    return null
  }
}

/**
 * Passes each event to `onEvent` until the stream ends. `label` names the stream
 * in the "stopped before it finished" message, e.g. "literature review".
 */
export async function readSSEStream(
  response: Response,
  onEvent: (event: Record<string, unknown>) => void,
  label = "stream"
): Promise<void> {
  if (!response.body) throw new Error("No response body")
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let completed = false

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const frames = buffer.split("\n\n")
      buffer = frames.pop() ?? ""

      for (const frame of frames) {
        const event = parseEvent(frame)
        if (!event) continue
        if (event.type === "error") {
          const message = typeof event.message === "string" && event.message ? event.message : `The ${label} reported an error`
          throw new StreamError("server", message)
        }
        if (event.type === "complete") completed = true
        onEvent(event)
      }
    }
  } finally {
    // If we stopped early (an error, or the handler threw), release the connection.
    reader.cancel().catch(() => {})
  }

  if (!completed) {
    throw new StreamError(
      "incomplete",
      `The ${label} stopped before it finished (the connection closed without a final result, possibly the time limit).`
    )
  }
}
