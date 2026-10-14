// Load and render prompt files from prompts/ (bundled into
// lib/prompts.generated.ts by scripts/gen-prompts.mjs).
//
// A prompt is one prompts/<group>/<name>.prompt file (format documented in
// prompts/README.md): YAML front matter, then a {{role "system"}} section and a
// {{role "user"}} section containing {{variable}} placeholders.
//
// Rendering is deliberately strict -- a missing, extra, mistyped or unknown
// variable throws PromptError instead of silently producing a half-filled
// prompt. Values are substituted in a single pass and never re-scanned, so text
// that happens to contain {{...}} (e.g. a paper abstract) is inserted verbatim.
//

import { createHash } from "node:crypto"
import { parse as parseYaml } from "yaml"
import { PROMPT_SOURCES } from "./prompts.generated"

export class PromptError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "PromptError"
  }
}

export type VariableType = "string" | "integer" | "boolean"
export type PromptValue = string | number | boolean

export interface PromptTemplate {
  id: string
  hash: string
  description: string
  variables: Record<string, VariableType>
  outputSchema: string | null
  system: string
  user: string
}

export interface RenderedPrompt {
  id: string
  hash: string
  system: string
  user: string
  outputSchema: string | null
}

const ID_RE = /^[a-z0-9_]+\/[a-z0-9_]+$/
const NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/
const ROLE_RE = /^\{\{role "(system|user)"\}\}[ \t]*$/gm
const PLACEHOLDER_RE = /\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}/g
const VARIABLE_TYPES: readonly string[] = ["string", "integer", "boolean"]
const TRAILING_WHITESPACE_RE = /[ \t\n\r\f\v]+$/ // ASCII only: unicode whitespace in a prompt is meaningful

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** Parse the text of a .prompt file. Throws PromptError if it is malformed. */
export function parsePrompt(id: string, source: string): PromptTemplate {
  const text = source.replace(/\r\n/g, "\n")
  const hash = createHash("sha256").update(text, "utf8").digest("hex").slice(0, 12)

  if (!text.startsWith("---\n")) {
    throw new PromptError(`${id}: file must start with '---' front matter`)
  }
  const end = text.indexOf("\n---\n", 3)
  if (end === -1) {
    throw new PromptError(`${id}: front matter is not closed with '---'`)
  }
  let meta: unknown
  try {
    meta = parseYaml(text.slice(4, end)) ?? {}
  } catch (error) {
    throw new PromptError(`${id}: invalid front matter YAML: ${(error as Error).message}`)
  }
  if (!isPlainObject(meta)) {
    throw new PromptError(`${id}: front matter must be a mapping`)
  }
  const body = text.slice(end + "\n---\n".length)

  const variables = readVariables(id, meta)
  const outputSchema = readOutputSchema(id, meta)
  const [system, user] = splitRoles(id, body)

  const used = new Set([...system.matchAll(PLACEHOLDER_RE), ...user.matchAll(PLACEHOLDER_RE)].map((m) => m[1]))
  for (const [sectionName, section] of [["system", system], ["user", user]]) {
    const leftover = section.replace(PLACEHOLDER_RE, "")
    if (leftover.includes("{{") || leftover.includes("}}")) {
      throw new PromptError(
        `${id}: malformed placeholder in the ${sectionName} section (placeholders are exactly {{name}}, no spaces)`
      )
    }
  }
  const declared = new Set(Object.keys(variables))
  const undeclared = [...used].filter((name) => !declared.has(name)).sort()
  const unused = [...declared].filter((name) => !used.has(name)).sort()
  if (undeclared.length > 0) {
    throw new PromptError(`${id}: placeholders not declared in input.schema: ${undeclared.join(", ")}`)
  }
  if (unused.length > 0) {
    throw new PromptError(`${id}: input.schema variables never used in the body: ${unused.join(", ")}`)
  }

  return {
    id,
    hash,
    description: meta.description === undefined ? "" : String(meta.description),
    variables,
    outputSchema,
    system,
    user,
  }
}

function readVariables(id: string, meta: Record<string, unknown>): Record<string, VariableType> {
  const input = meta.input
  const schema = isPlainObject(input) ? input.schema ?? {} : {}
  if (!isPlainObject(schema)) {
    throw new PromptError(`${id}: input.schema must be a mapping of name -> type`)
  }
  const variables: Record<string, VariableType> = {}
  for (const [name, type] of Object.entries(schema)) {
    if (!NAME_RE.test(name)) {
      throw new PromptError(`${id}: invalid variable name '${name}'`)
    }
    if (typeof type !== "string" || !VARIABLE_TYPES.includes(type)) {
      throw new PromptError(`${id}: variable '${name}' has type '${String(type)}'; allowed: ${VARIABLE_TYPES.join(", ")}`)
    }
    variables[name] = type as VariableType
  }
  return variables
}

function readOutputSchema(id: string, meta: Record<string, unknown>): string | null {
  const output = meta.output
  const schema = isPlainObject(output) ? output.schema : undefined
  if (schema === undefined || schema === null) return null
  if (typeof schema !== "string") {
    throw new PromptError(`${id}: output.schema must be the name of a code-defined schema`)
  }
  return schema
}

function splitRoles(id: string, body: string): [string, string] {
  const markers = [...body.matchAll(ROLE_RE)]
  const roles = markers.map((m) => m[1])
  if (roles.length !== 2 || roles[0] !== "system" || roles[1] !== "user") {
    throw new PromptError(
      `${id}: body needs exactly one {{role "system"}} followed by one {{role "user"}} marker (found [${roles.join(", ")}])`
    )
  }
  const [first, second] = markers as [RegExpMatchArray, RegExpMatchArray]
  if (body.slice(0, first.index).trim() !== "") {
    throw new PromptError(`${id}: text before the first role marker`)
  }

  // Drop the newline that ends the marker line and trailing whitespace;
  // everything else is the prompt text, byte for byte.
  const section = (start: number, stop: number): string => {
    const text = body.slice(start, stop)
    return (text.startsWith("\n") ? text.slice(1) : text).replace(TRAILING_WHITESPACE_RE, "")
  }
  const system = section(first.index! + first[0].length, second.index!)
  const user = section(second.index! + second[0].length, body.length)
  if (system === "" || user === "") {
    throw new PromptError(`${id}: system and user sections must not be empty`)
  }
  return [system, user]
}

const cache = new Map<string, PromptTemplate>()

export function loadPrompt(id: string): PromptTemplate {
  if (!ID_RE.test(id)) {
    throw new PromptError(`invalid prompt id '${id}' (expected '<group>/<name>')`)
  }
  let template = cache.get(id)
  if (!template) {
    const source = Object.prototype.hasOwnProperty.call(PROMPT_SOURCES, id) ? PROMPT_SOURCES[id] : undefined
    if (source === undefined) {
      throw new PromptError(`unknown prompt '${id}' (run \`pnpm gen:prompts\` if you just added it)`)
    }
    template = parsePrompt(id, source)
    cache.set(id, template)
  }
  return template
}

/**
 * Render `id` with `variables`. Every declared variable must be supplied with
 * the declared type, and nothing else may be passed.
 */
export function renderPrompt(id: string, variables: Record<string, PromptValue>): RenderedPrompt {
  const template = loadPrompt(id)

  const supplied = Object.keys(variables)
  const missing = Object.keys(template.variables).filter((name) => !supplied.includes(name)).sort()
  const extra = supplied.filter((name) => !(name in template.variables)).sort()
  if (missing.length > 0 || extra.length > 0) {
    const parts: string[] = []
    if (missing.length > 0) parts.push(`missing variables [${missing.join(", ")}]`)
    if (extra.length > 0) parts.push(`unexpected variables [${extra.join(", ")}]`)
    throw new PromptError(`${id}: ${parts.join("; ")}`)
  }

  const values: Record<string, string> = Object.create(null)
  for (const name of supplied) {
    values[name] = toText(id, name, template.variables[name], variables[name])
  }
  // A replacer function (not a replacement string) so "$&" etc. in values are literal.
  const substitute = (section: string): string =>
    section.replace(PLACEHOLDER_RE, (_match, name: string) => values[name])

  return {
    id,
    hash: template.hash,
    system: substitute(template.system),
    user: substitute(template.user),
    outputSchema: template.outputSchema,
  }
}

function toText(id: string, name: string, type: VariableType, value: unknown): string {
  if (type === "string" && typeof value === "string") return value
  if (type === "integer" && typeof value === "number" && Number.isInteger(value)) return String(value)
  if (type === "boolean" && typeof value === "boolean") return value ? "true" : "false"
  throw new PromptError(`${id}: variable '${name}' must be a ${type}, got ${value === null ? "null" : typeof value}`)
}

/** Every prompt id bundled into this build. */
export function listPromptIds(): string[] {
  return Object.keys(PROMPT_SOURCES).sort()
}

/** `{ prompt id: content hash }` for the given prompts, e.g. to stamp a response. */
export function promptVersions(ids: string[]): Record<string, string> {
  return Object.fromEntries(ids.map((id) => [id, loadPrompt(id).hash]))
}
