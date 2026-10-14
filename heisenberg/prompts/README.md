# Prompts

Every LLM prompt in the app lives here, as one file per call. The Next.js app
loads them with `lib/prompts.ts` and calls the model through `lib/openai.ts`,
so a prompt is written once and never inlined in a route.

```
prompts/
  analysis/      reasoning_decision, class_expansion
  chat/          answer
  extraction/    protein_list, protein_list_fallback, drug_protein_map
  literature/    review_first_pass, drug_analysis, repurposing_candidates
  paper/         abstract, introduction, results, conclusion_from_full_text
  report/        methodology, results, conclusion
```

A prompt id is `<group>/<name>`, e.g. `analysis/reasoning_decision`.

The old Python pipeline (`legacy/python-pipeline/`) has its own frozen snapshot
of this directory. Two of its prompts, `paper/therapeutic_landscape` and
`paper/conclusion`, are not in the app: porting the 6-section paper means
copying them back from that snapshot.

## File format

The shape follows [Dotprompt](https://google.github.io/dotprompt/reference/), so
the files can move to the real library later without being rewritten.

```
---
name: reasoning_decision                 # must equal the file name
description: "What this prompt is for."  # required; quote it (YAML: a bare ': ' breaks)
input:
  schema:
    disease: string                      # string | integer | boolean
    round: integer
output:
  schema: ReasoningDecision              # optional: name of a code-defined schema
---
{{role "system"}}
You are ... round {{round}} ...
{{role "user"}}
Disease: {{disease}}
```

- Exactly one `{{role "system"}}` section, then one `{{role "user"}}` section.
- A placeholder is exactly `{{name}}` (no spaces). Every declared variable must be
  used and every placeholder declared, otherwise the file fails to load.
- Text is taken **verbatim**: newlines in the file are newlines in the prompt, so
  long paragraphs stay on one line. Only the newline after a role marker and
  trailing whitespace at the end of a section are dropped.
- **No logic in templates** (no `if`/loops). Format data in code and pass the
  finished string: joined lists, title-cased names, JSON blobs, truncation.
- Values are inserted in one pass and never re-scanned, so a value containing
  `{{x}}` is not interpreted.

## Using a prompt

```ts
import { queryPrompt, queryPromptStructured } from "@/lib/openai"
import { ProteinListSchema } from "@/lib/llm-schemas"

const text = await queryPrompt("report/methodology", { disease, overview })
const result = await queryPromptStructured("extraction/protein_list", { text }, ProteinListSchema)
```

A prompt with `output.schema` must be called through `queryPromptStructured`
with that exact schema, and a prompt without one through `queryPrompt`;
anything else throws, so a prompt cannot silently drift from its schema. The
schemas are Zod objects in `lib/llm-schemas.ts` (the `SCHEMAS` registry).

## Changing a prompt

1. Edit the `.prompt` file.
2. `pnpm gen:prompts`: rebuilds `lib/prompts.generated.ts` (also runs before
   `pnpm dev` and `pnpm build`; the file is committed so a fresh checkout
   type-checks).
3. `pnpm test`.

Adding a prompt: create the file, then steps 2-3.

## Versioning

Git is the version store. Each prompt also has a content **hash** (first 12 hex
chars of the SHA-256 of the normalised file). `/api/report` and `/api/paper`
return `promptVersions`, so every generated document says which prompt versions
produced it.

## Tests

`tests/prompts.test.ts` checks the loader rules, that every shipped prompt
parses, that declared schemas exist in code, that no route calls the raw client
with an inline prompt, and that the generated bundle is up to date.
