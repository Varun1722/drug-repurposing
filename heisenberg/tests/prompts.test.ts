// Tests for the prompt loader (lib/prompts.ts) and the prompt files in prompts/.
// Run with `pnpm test`. No API key and no network.

import assert from "node:assert/strict"
import { spawnSync } from "node:child_process"
import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { describe, it } from "node:test"

import { SCHEMAS } from "../lib/llm-schemas"
import { listPromptIds, loadPrompt, parsePrompt, PromptError, renderPrompt } from "../lib/prompts"
import { PROMPT_SOURCES } from "../lib/prompts.generated"

const ROOT = join(import.meta.dirname ?? __dirname, "..")
const PROMPTS_DIR = join(ROOT, "prompts")

const GOOD = `---
name: demo
description: "A demo prompt: with a colon."
input:
  schema:
    disease: string
    round: integer
---
{{role "system"}}
You study {{disease}}.
{{role "user"}}
Round {{round}} of {{disease}}.
`

const parse = (source = GOOD) => parsePrompt("demo/demo", source)

describe("parsePrompt", () => {
  it("reads metadata and sections", () => {
    const t = parse()
    assert.deepEqual(t.variables, { disease: "string", round: "integer" })
    assert.equal(t.description, "A demo prompt: with a colon.")
    assert.equal(t.system, "You study {{disease}}.")
    assert.equal(t.user, "Round {{round}} of {{disease}}.")
    assert.equal(t.outputSchema, null)
  })

  it("hash is stable, content-sensitive and CRLF-insensitive", () => {
    assert.equal(parse().hash, parse().hash)
    assert.notEqual(parse(GOOD.replace("You study", "You research")).hash, parse().hash)
    assert.equal(parse(GOOD.replaceAll("\n", "\r\n")).hash, parse().hash)
  })

  const malformed: [string, string, RegExp][] = [
    ["no front matter", "no front matter", /must start with/],
    ["unclosed front matter", "---\nname: x\n", /not closed/],
    ["system marker replaced by user", GOOD.replace('{{role "system"}}', '{{role "user"}}'), /exactly one/],
    ["user marker replaced by system", GOOD.replace('{{role "user"}}', '{{role "system"}}'), /exactly one/],
    ["text before first marker", GOOD.replace('{{role "system"}}\n', 'stray text\n{{role "system"}}\n'), /before the first role/],
    ["undeclared placeholder", GOOD.replace("{{round}}", "{{rnd}}"), /not declared/],
    ["unused variable", GOOD.replace("Round {{round}} of ", "Round of "), /never used/],
    ["spaces in placeholder", GOOD.replace("{{round}}", "{{ round }}"), /malformed placeholder/],
    ["unknown variable type", GOOD.replace("integer", "float"), /allowed:/],
    ["empty system section", GOOD.replace("You study {{disease}}.", ""), /must not be empty/],
    ["front matter that is a list", "---\n- a list\n---\n" + GOOD.split("---\n").slice(2).join("---\n"), /mapping/],
  ]
  for (const [name, source, message] of malformed) {
    it(`rejects: ${name}`, () => {
      assert.throws(() => parse(source), (e) => e instanceof PromptError && message.test(e.message))
    })
  }
})

describe("renderPrompt (real prompts, strictness)", () => {
  const id = "analysis/class_expansion"
  const vars = { drug_class: "statins", proteins: "KRAS", exclude: "a, b" }

  it("substitutes variables", () => {
    const r = renderPrompt(id, vars)
    assert.match(r.user, /class: statins, that might bind to these proteins: KRAS\. Exclude any already in this list: a, b$/)
  })

  it("is single-pass and inserts values verbatim ($& $1 {{x}})", () => {
    const tricky = "{{proteins}} $& $1 \\n é {{ x }}"
    const r = renderPrompt(id, { ...vars, drug_class: tricky })
    assert.ok(r.user.includes(`class: ${tricky}, that`))
  })

  it("rejects missing and unexpected variables", () => {
    assert.throws(() => renderPrompt(id, { drug_class: "x" }), /missing variables \[exclude, proteins\]/)
    assert.throws(() => renderPrompt(id, { ...vars, extra: "y" }), /unexpected variables \[extra\]/)
  })

  it("rejects wrong types", () => {
    assert.throws(() => renderPrompt(id, { ...vars, drug_class: 5 }), /must be a string/)
    assert.throws(
      () => renderPrompt("analysis/reasoning_decision", { disease: "d", round: "2", max_round: 3, hypotheses: "", expansion_history: "", top_results: "" }),
      /must be a integer/
    )
    assert.throws(
      () => renderPrompt("analysis/reasoning_decision", { disease: "d", round: 2.5, max_round: 3, hypotheses: "", expansion_history: "", top_results: "" }),
      /must be a integer/
    )
    assert.throws(
      () => renderPrompt("analysis/reasoning_decision", { disease: "d", round: true, max_round: 3, hypotheses: "", expansion_history: "", top_results: "" }),
      /must be a integer/
    )
  })

  for (const bad of ["../etc/passwd", "demo", "demo/", "Demo/demo", "a/b/c", "demo/de mo"]) {
    it(`rejects invalid id ${JSON.stringify(bad)}`, () => {
      assert.throws(() => loadPrompt(bad), /invalid prompt id/)
    })
  }

  it("rejects unknown prompts (incl. prototype keys)", () => {
    assert.throws(() => loadPrompt("demo/missing"), /unknown prompt/)
    assert.throws(() => loadPrompt("constructor/constructor"), /unknown prompt/)
    assert.throws(() => loadPrompt("__proto__/x"), /unknown prompt/)
  })
})

describe("shipped prompts", () => {
  const ids = listPromptIds()

  it("the bundle matches the files on disk and is up to date", () => {
    const onDisk: string[] = []
    for (const group of readdirSync(PROMPTS_DIR, { withFileTypes: true })) {
      if (!group.isDirectory() || group.name.startsWith("_")) continue
      for (const file of readdirSync(join(PROMPTS_DIR, group.name))) {
        if (file.endsWith(".prompt")) onDisk.push(`${group.name}/${file.slice(0, -7)}`)
      }
    }
    assert.deepEqual(ids, onDisk.sort())
    const check = spawnSync("node", ["scripts/gen-prompts.mjs", "--check"], { cwd: ROOT, encoding: "utf8" })
    assert.equal(check.status, 0, check.stdout + check.stderr)
  })

  for (const id of ids) {
    it(`${id} parses and is well-formed`, () => {
      const t = loadPrompt(id)
      assert.ok(t.description, "every prompt needs a description")
      assert.equal(PROMPT_SOURCES[id].split("\nname: ")[1].split("\n")[0], id.split("/")[1])
      assert.ok(t.outputSchema === null || t.outputSchema in SCHEMAS, `unknown output.schema ${t.outputSchema}`)
    })
  }

  it("no API route calls the raw OpenAI client (prompts live in prompts/)", () => {
    const offenders: string[] = []
    const walk = (dir: string) => {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const path = join(dir, entry.name)
        if (entry.isDirectory()) walk(path)
        else if (entry.name === "route.ts" && /queryOpenAI(Structured)?\b/.test(readFileSync(path, "utf8"))) offenders.push(path)
      }
    }
    walk(join(ROOT, "app", "api"))
    assert.deepEqual(offenders, [])
  })
})
