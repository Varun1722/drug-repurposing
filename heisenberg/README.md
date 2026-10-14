# Heisenberg
A drug repurposing agent which takes a disease and tries to find out other FDA approved or pre clinical drugs, which are not approved for this particular disease, that can help in cure of this disease. 

## Architecture

- **Orchestration is TypeScript.** The Next.js app owns the whole pipeline: PubMed and web search, protein and drug extraction, structure retrieval, the reasoning loop, docking submission, report and paper generation (`app/api/*`, `lib/*`). Every LLM prompt is a file in `prompts/`; all outbound calls go through `lib/retry.ts` (retries, backoff, one failure type).
- **Python provides model and chemistry services**, as separate containers behind HTTP with an async submit/poll contract. Today that is `modal_app/` (GNINA docking on Modal). Planned: RDKit/gemmi preparation, and the Stage-1 models (MATRIX, TxGNN).
- **`legacy-python-pipeline/`** is the original all-in-Python CLI. It is frozen, kept as a behavioural reference and for unattended runs until the TypeScript replacement exists. See its README.

Commands: `pnpm dev`, `pnpm test`, `pnpm build`. The plan and open work are in `TODO.md`.
