# Heisenberg

A drug repurposing agent. Given a disease, it finds existing FDA-approved or pre-clinical drugs that are not approved for that disease but may help treat it.

## Repo map

| Folder | What it is |
|---|---|
| [`heisenberg/`](heisenberg/) | The web app and the pipeline orchestrator (Next.js, TypeScript). Literature search, protein and drug extraction, structure retrieval, the reasoning loop, docking submission, report and paper. |
| [`services/docking/`](services/docking/) | GNINA molecular docking on Modal (Python). Async submit/poll HTTP API used by the web app. |
| [`services/stage1/`](services/stage1/) | Stage 1, candidate generation (Python): MATRIX score store, disease lookup, PrimeKG crosswalk, and the TxGNN scorer container. Large data lives in `services/stage1/data/` and is gitignored. |
| [`legacy/python-pipeline/`](legacy/python-pipeline/) | The original all-in-Python CLI. Frozen: kept as a behavioural reference until everything in the roadmap is ported, then deleted. |
| [`docs/`](docs/) | Problem statement, Stage 1 spec, model survey, BIOS research, and the [roadmap](docs/roadmap.md). |

## Architecture

- **TypeScript orchestrates.** The Next.js app owns the whole pipeline (`heisenberg/app/api/*`, `heisenberg/lib/*`). Every LLM prompt is a file in `heisenberg/prompts/`. All outbound calls go through `heisenberg/lib/retry.ts` (retries, backoff, one failure type).
- **Python provides model and chemistry services**, as separate containers behind HTTP with an async submit/poll contract. Today that is `services/docking/`. Planned: RDKit/gemmi preparation, and the Stage 1 models (MATRIX, TxGNN).
- **Docs** explain the direction: start with [`docs/problem-statement.md`](docs/problem-statement.md), then [`docs/stage1-spec.md`](docs/stage1-spec.md).

## Quickstart

Web app (needs Node 20 and pnpm):

```bash
cd heisenberg
cp .env.example .env      # fill in OPENAI_API_KEY, EXA_API_KEY, GNINA_JOB_API_URL
pnpm install
pnpm dev                  # http://localhost:3000
pnpm test
```

Docking service (needs a Modal account):

```bash
modal deploy services/docking/gnina_worker.py   # prints the URL for GNINA_JOB_API_URL
```

Stage 1 (needs `uv`; tests that need the downloaded data skip when it is missing, and the full data set is about 11 GB):

```bash
cd services/stage1
uv sync
uv run pytest
```

Legacy CLI: see [`legacy/python-pipeline/README.md`](legacy/python-pipeline/README.md).

## Where outputs go

Nothing generated is committed. Build output (`.next/`), caches, and run outputs are gitignored. Stage 1 downloads and results stay under `services/stage1/data/` (`raw/`, `db/`, `txgnn_work/`, `txgnn_out/`, `logs/`). The legacy CLI writes its state and reports into the directory it is run from, and that folder ignores them.
