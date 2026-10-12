// Zod -> JSON Schema conversion for OpenAI Structured Outputs (strict mode).
//
// Strict mode accepts only a subset of JSON Schema: the root must be an
// object, every property must be listed in `required`, every object must set
// `additionalProperties: false`, and a handful of keywords (oneOf, allOf,
// default, ...) are rejected. Zod happily emits schemas that break those
// rules (e.g. `.optional()` drops a key from `required`), and OpenAI only
// reports the problem as an opaque 400 at request time. `assertStrictSchema`
// turns that into an immediate, readable error naming the offending path.

import { z } from "zod/v4"

type JsonSchema = Record<string, unknown>

const FORBIDDEN_KEYWORDS = [
  "oneOf",
  "allOf",
  "not",
  "if",
  "then",
  "else",
  "default",
  "patternProperties",
  "propertyNames",
  "minProperties",
  "maxProperties",
]

// OpenAI allows 10 levels of object nesting in a strict schema.
const MAX_OBJECT_DEPTH = 10

function isRecord(value: unknown): value is JsonSchema {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function walk(node: unknown, path: string, objectDepth: number): void {
  if (!isRecord(node)) return

  for (const keyword of FORBIDDEN_KEYWORDS) {
    if (keyword in node) {
      throw new Error(
        `Schema at ${path} uses "${keyword}", which OpenAI strict mode does not support`
      )
    }
  }

  const properties = isRecord(node.properties) ? node.properties : undefined
  let depth = objectDepth
  if (node.type === "object" || properties) {
    depth += 1
    if (depth > MAX_OBJECT_DEPTH) {
      throw new Error(
        `Schema at ${path} nests objects more than ${MAX_OBJECT_DEPTH} levels deep`
      )
    }
    if (node.additionalProperties !== false) {
      throw new Error(`Object at ${path} must set additionalProperties: false`)
    }
    const keys = Object.keys(properties ?? {})
    const required = Array.isArray(node.required) ? (node.required as string[]) : []
    const missing = keys.filter((key) => !required.includes(key))
    if (missing.length > 0) {
      throw new Error(
        `Object at ${path} has non-required properties (${missing.join(", ")}). ` +
          `Strict mode needs every key required; use .nullable() instead of .optional()`
      )
    }
  }

  // Recurse into sub-schemas. `properties` and `$defs` are maps of name ->
  // schema, so walk their values (a field called "default" is not a keyword).
  for (const mapKey of ["properties", "$defs", "definitions"]) {
    const map = node[mapKey]
    if (isRecord(map)) {
      for (const [name, child] of Object.entries(map)) {
        walk(child, `${path}/${mapKey}/${name}`, depth)
      }
    }
  }
  walk(node.items, `${path}/items`, depth)
  for (const variantKey of ["anyOf"]) {
    const variants = node[variantKey]
    if (Array.isArray(variants)) {
      variants.forEach((variant, i) =>
        walk(variant, `${path}/${variantKey}/${i}`, depth)
      )
    }
  }
}

export function assertStrictSchema(schema: JsonSchema): void {
  if (schema.type !== "object" || "anyOf" in schema) {
    throw new Error(
      'Root schema must be a plain object (z.object); root-level unions are not supported'
    )
  }
  walk(schema, "#", 0)
}

/**
 * Convert a Zod schema to the JSON Schema sent as
 * `response_format.json_schema.schema`, verifying it is strict-mode safe.
 */
export function toStrictJsonSchema(schema: z.ZodType): JsonSchema {
  const jsonSchema = z.toJSONSchema(schema, {
    target: "draft-7",
    override: ({ jsonSchema: node }) => {
      // z.number().int() adds safe-integer bounds; plain "integer" is enough.
      if (node.type === "integer") {
        delete node.minimum
        delete node.maximum
      }
    },
  }) as JsonSchema
  assertStrictSchema(jsonSchema)
  return jsonSchema
}
