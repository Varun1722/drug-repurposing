# Prompts

Every LLM prompt in this repo lives here, as one file per call. The Python
pipeline (`prompt_loader.py`, `llm.py`) and the Next.js app (`lib/prompts.ts`,
`lib/openai.ts`) render the same files, so a prompt is written once.

```
prompts/
  analysis/      reasoning_decision, class_expansion
  chat/          answer
  extraction/    protein_list, protein_list_fallback, drug_protein_map
  literature/    review_first_pass, drug_analysis, repurposing_candidates (+ *_cli variants)
  paper/         abstract, introduction, therapeutic_landscape, results, conclusion, ...
  report/        methodology, results, conclusion
  _parity/       test fixtures (not prompts)
```

A prompt id is `<group>/<name>`, e.g. `analysis/reasoning_decision`.

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
  finished string: joined lists, `disease.title()`, JSON blobs, truncation.
- Values are inserted in one pass and never re-scanned, so a value containing
  `{{x}}` is not interpreted.

## Using a prompt

```python
# Python
from llm import query_prompt, query_prompt_structured
from llm_schemas import ProteinList

text   = query_prompt("report/methodology", disease=disease, overview=overview)
result = query_prompt_structured("extraction/protein_list", ProteinList, text=text)
```

```ts
// TypeScript
import { queryPrompt, queryPromptStructured } from "@/lib/openai"
import { ProteinListSchema } from "@/lib/llm-schemas"

const text = await queryPrompt("report/methodology", { disease, overview })
const result = await queryPromptStructured("extraction/protein_list", { text }, ProteinListSchema)
```

A prompt with `output.schema` must be called through the `*_structured` helper
with that exact schema, and a prompt without one through the plain helper;
anything else raises, so a prompt cannot silently drift from its schema.
Schemas are defined twice, in `llm_schemas.py` and `lib/llm-schemas.ts` (keep
them in sync).

## Changing a prompt

1. Edit the `.prompt` file.
2. `pnpm gen:prompts` &mdash; rebuilds `lib/prompts.generated.ts` (also runs before
   `pnpm dev` and `pnpm build`; the file is committed so a fresh checkout type-checks).
3. `python3 scripts/update_prompt_parity.py` &mdash; refreshes `_parity/rendered.json`.
4. `pnpm test` and `pytest test_prompts.py`.

Adding a prompt: create the file, then steps 2-4. For a new variable, `vars.json`
gets a deliberately awkward default (`{{x}}`, `$&`, backslashes, CRLF, unicode);
edit it by hand if you want specific cases.

## Versioning

Git is the version store. Each prompt also has a content **hash** (first 12 hex
chars of the SHA-256 of the normalised file). The pipeline records
`prompt_versions` in `pipeline_state.json`, and `/api/report` and `/api/paper`
return `promptVersions`, so every generated document says which prompt versions
produced it.

## Tests

- `test_prompts.py` / `tests/prompts.test.ts`: loader rules, every shipped prompt
  parses, declared schemas exist, no route or module calls the raw client with an
  inline prompt, generated files are up to date.
- `_parity/`: both loaders render `vars.json` and must equal `rendered.json` byte
  for byte (text and hash), which keeps the two implementations from drifting.
