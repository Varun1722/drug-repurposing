// Reading the server-sent-event streams of /api/review and /api/structures
// (lib/sse.ts). Contract: a stream must end with a `complete` event. A server
// `error` event, or a stream that just stops, is an error the caller sees. It is
// never a normal end, or the page carries on with half the data.

import assert from "node:assert/strict"
import { describe, it } from "node:test"

import { StreamError, readSSEStream } from "../lib/sse"

const enc = new TextEncoder()
const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`

/** A Response streaming `text` in chunks of `chunkSize` bytes, then closing. */
function sse(text: string, chunkSize = Infinity): Response {
  const bytes = enc.encode(text)
  return new Response(
    new ReadableStream({
      start(controller) {
        for (let i = 0; i < bytes.length; i += chunkSize) controller.enqueue(bytes.slice(i, i + chunkSize))
        controller.close()
      },
    })
  )
}
const events = (...list: object[]) => sse(list.map(frame).join(""))

async function collect(response: Response, label?: string): Promise<Record<string, unknown>[]> {
  const seen: Record<string, unknown>[] = []
  await readSSEStream(response, (event) => void seen.push(event), label)
  return seen
}

describe("readSSEStream", () => {
  it("passes every event to the handler, in order, and ends normally after `complete`", async () => {
    const seen = await collect(events({ type: "progress" }, { type: "proteins", proteins: ["EGFR"] }, { type: "complete", n: 1 }))
    assert.deepEqual(seen.map((e) => e.type), ["progress", "proteins", "complete"])
    assert.deepEqual(seen[2], { type: "complete", n: 1 })
  })

  it("reassembles events split across chunks, even one byte at a time", async () => {
    const text = [{ type: "proteins", proteins: ["EGFR", "KRAS"] }, { type: "complete" }].map(frame).join("")
    for (const size of [1, 7, 64]) {
      const seen = await collect(sse(text, size))
      assert.deepEqual(seen.map((e) => e.type), ["proteins", "complete"], `chunk size ${size}`)
      assert.deepEqual(seen[0].proteins, ["EGFR", "KRAS"])
    }
  })

  it("throws the server's message on an `error` event, after delivering the events before it", async () => {
    const seen: string[] = []
    const stream = events({ type: "proteins", proteins: ["EGFR"] }, { type: "error", message: "openai: server (HTTP 503)" }, { type: "drugs", drugs: [] })
    await assert.rejects(
      readSSEStream(stream, (e) => void seen.push(String(e.type))),
      (e: unknown) => e instanceof StreamError && e.kind === "server" && e.message === "openai: server (HTTP 503)"
    )
    assert.deepEqual(seen, ["proteins"], "nothing after the error is handled")
  })

  it("gives an `error` event with no message a readable one", async () => {
    await assert.rejects(
      collect(events({ type: "error" })),
      (e: unknown) => e instanceof StreamError && e.kind === "server" && e.message.length > 0
    )
  })

  it("throws when the stream ends without `complete` (a cut connection)", async () => {
    await assert.rejects(
      collect(events({ type: "proteins", proteins: ["EGFR"] }, { type: "progress", message: "Analyzing" }), "literature review"),
      (e: unknown) => e instanceof StreamError && e.kind === "incomplete" && /literature review/.test(e.message)
    )
  })

  it("throws for an empty stream", async () => {
    await assert.rejects(collect(sse("")), (e: unknown) => e instanceof StreamError && e.kind === "incomplete")
  })

  it("lets an exception from the handler through instead of swallowing it", async () => {
    await assert.rejects(
      readSSEStream(events({ type: "proteins" }, { type: "complete" }), () => {
        throw new Error("handler broke")
      }),
      /handler broke/
    )
  })

  it("skips lines that are not parseable events, and SSE comments", async () => {
    const text = "data: {oops\n\n" + ": keep-alive\n\n" + "event: x\n\n" + frame({ type: "progress" }) + "data: [1,2]\n\n" + frame({ type: "complete" })
    const seen = await collect(sse(text))
    assert.deepEqual(seen.map((e) => e.type), ["progress", "complete"])
  })

  it("passes a read failure (connection broken mid-stream) through", async () => {
    const broken = new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(enc.encode(frame({ type: "progress" })))
          controller.error(new TypeError("network error"))
        },
      })
    )
    await assert.rejects(collect(broken), /network error/)
  })

  it("throws for a response with no body", async () => {
    await assert.rejects(readSSEStream(new Response(null), () => {}), /No response body/)
  })

  it("stops reading, so the connection is released, when it throws", async () => {
    let cancelled = false
    const hanging = new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(enc.encode(frame({ type: "error", message: "boom" })))
          // never closes: the server would keep the connection open
        },
        cancel() {
          cancelled = true
        },
      })
    )
    await assert.rejects(collect(hanging), /boom/)
    assert.equal(cancelled, true)
  })
})
