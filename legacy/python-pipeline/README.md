# Legacy Python pipeline (frozen reference)

This folder is the original Python implementation of the Heisenberg pipeline: a CLI that ran literature review → structure retrieval → docking → LLM reasoning loop → report → paper in one process.

**It is no longer the orchestrator and is not maintained.** Orchestration now lives in TypeScript (the Next.js app in `heisenberg/`: `app/api/*`, `lib/*`). Python's role in this project is **model and chemistry services** behind HTTP (today: `services/docking/`, the GNINA docking worker). This code is kept for two reasons:

1. It is the behavioural reference for what the TypeScript side still has to port (see the table below), and its tests record the edge cases already learned.
2. It still runs, for anyone who wants an unattended end-to-end run before the TypeScript replacement exists.

Nothing outside this folder imports from it, and nothing in it imports from the app. It carries its own snapshot of `prompts/` (taken when the folder was created), so changes to the app's prompts do not affect it.

## Running it

```bash
cd legacy/python-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # needs RDKit and gemmi
python pipeline.py "pancreatic ductal adenocarcinoma" --max-rounds 3
python pipeline.py --resume
pytest -q                                  # 358 tests; offline, HTTP is mocked
```

It reads `OPENAI_API_KEY`, `EXA_API_KEY` and `GNINA_ENDPOINT_URL` (the old **synchronous** docking endpoint of `services/docking/gnina_worker.py`, used only by `agent3.py`) from the environment or the app's `.env`. It writes `pipeline_state.json`, `review.*`, `results.md`, `final_paper.md` and `structures/` into the directory you run it from (git-ignored here). `structures/` holds the sample PDB files from earlier runs.

## Where each part went

| Legacy file | What it did | Now |
|---|---|---|
| `pipeline.py` | state machine, reasoning loop, `--resume` | TS workflow with persisted run state (planned) |
| `review.py` | PubMed + Exa + literature prompts | `heisenberg/app/api/review/route.ts` |
| `agent2.py` | RCSB / PubChem, **RDKit SMILES canonicalization, gemmi CIF→PDB** | `heisenberg/lib/pdb.ts`, `heisenberg/lib/pubchem.ts`; the chemistry moves to a Python service (planned) |
| `agent3.py` | sync GNINA docking | `heisenberg/lib/dock-jobs.ts` (async job queue) |
| `results.py`, `final.py` | report and 6-section paper, compound classifier | `heisenberg/app/api/report`, `heisenberg/app/api/paper` (4 paper sections; the 6-section merge is not ported) |
| `llm.py`, `llm_schemas.py`, `prompt_loader.py`, `resilience.py` | OpenAI client, schemas, prompt loader, retry layer | `heisenberg/lib/openai.ts`, `heisenberg/lib/llm-schemas.ts`, `heisenberg/lib/prompts.ts`, `heisenberg/lib/retry.ts` |

## Known behaviour the TypeScript side does not have yet

- SMILES canonicalization and CIF→PDB conversion (RDKit / gemmi).
- The stricter `classify_compound` rule (`results.py`); the TS classifier disagrees with it on some inputs.
- A real XML parser for PubMed (`defusedxml`); the TS side parses with regexes.
- The two extra paper sections (`prompts/paper/therapeutic_landscape.prompt`, `prompts/paper/conclusion.prompt` in this folder's snapshot).
- Unattended, fully automatic runs and the saved `prompt_versions` record per run.
