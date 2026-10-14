# Heisenberg web app

The Next.js app that runs the drug repurposing pipeline: PubMed and web search, protein and drug extraction, structure retrieval, the reasoning loop, docking submission, report and paper generation. The repo layout and architecture are in the [root README](../README.md).

## Commands

```bash
pnpm dev      # regenerates lib/prompts.generated.ts, then starts next dev
pnpm test     # node:test over tests/*.test.ts
pnpm build
```

Copy `.env.example` to `.env` and fill it in. `GNINA_JOB_API_URL` comes from deploying [`../services/docking/`](../services/docking/).

## Layout

| Path | Contents |
|---|---|
| `app/page.tsx`, `app/deck/` | The UI and the slide deck page |
| `app/api/*/route.ts` | One route per pipeline step: `review`, `structures`, `dock/submit`, `dock/status`, `analyze`, `expand`, `report`, `paper`, `chat` |
| `lib/` | Clients and shared logic: `retry.ts` (all outbound calls), `openai.ts`, `exa.ts`, `pubchem.ts`, `pdb.ts`, `dock-jobs.ts`, `dock-polling.ts`, `prompts.ts`, `llm-schemas.ts`, `types.ts` |
| `prompts/` | Every LLM prompt as a `.prompt` file ([format](prompts/README.md)) |
| `components/` | App components; `components/ui/` is the shadcn/ui kit |
| `scripts/gen-prompts.mjs` | Bundles `prompts/` into `lib/prompts.generated.ts` (checked by a test) |
| `tests/` | Unit tests for the clients, retry layer, prompts, structured outputs and polling |
