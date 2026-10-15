# Roadmap and open work

Open work for the repo, and the decisions already made. Repo layout is in the [root README](../README.md). Items marked ~~struck through~~ are done and kept for the reasoning.

## 1. ~~Decide the primary pipeline implementation~~ — decided
**TypeScript orchestrates; Python provides model and chemistry services.**
- The Next.js app (`app/api/*`, `lib/*`) owns the whole pipeline: literature, structures, docking, the reasoning loop, report and paper, plus approvals and run state.
- Python is used only where Python-only libraries or models are needed (RDKit, gemmi, TxGNN, MATRIX, GNINA), as services behind HTTP with an async submit/poll contract. Today that is `services/docking/` (docking).
- The old Python orchestrator (`pipeline.py`, `review.py`, `agent2.py`, `agent3.py`, `results.py`, `final.py` and its LLM/prompt/retry twins) was moved, unchanged, to `legacy/python-pipeline/`. It is frozen: see that folder's README for how to run it and where each part went. Delete it once everything in section 2 is ported.
- Why: the models cannot live in the orchestrator's process in any design (TxGNN needs Python 3.8 + DGL 0.5.2 in its own container), and durable-execution engines exist for TypeScript, so keeping the orchestrator in TS keeps the UI, approvals and workflow in one place.

## 2. Port what only the legacy Python pipeline did
The TS pipeline dropped these. Each one is a work item, with the legacy file as the behavioural reference (its tests list the edge cases):
- [ ] **SMILES canonicalization** (RDKit): `legacy/python-pipeline/agent2.py canonicalize_smiles`. TS (`lib/pubchem.ts`) trusts PubChem's `CanonicalSMILES` as-is. Plan: a prepare endpoint in the Modal worker (it already runs RDKit), called by TS in batches.
- [ ] **CIF→PDB conversion** (gemmi): `agent2.py download_pdb`. `lib/pdb.ts downloadPdb()` sends whichever of `.pdb`/`.cif` RCSB returns straight to docking. Same Modal prepare endpoint.
- [x] ~~**Compound classifier**~~ — done (audit issue 3): the stage now comes from ChEMBL (`heisenberg/lib/chembl.ts`, `lib/stage.ts`) instead of parsing the LLM's FDA-status text; the Python `classify_compound` was not ported (it needs the literal "fda", matches any disease word over 3 letters, and still counts "Not FDA-approved" as approved).
- [ ] **PubMed XML parsing**: `review.py` uses a real XML parser (`defusedxml`); `app/api/review/route.ts` parses with regexes. Replace with a maintained TS XML library (check its entity/security behaviour first) and test against a recorded PubMed response.
- [ ] **Paper sections**: the legacy paper has 6 sections, the app 4. Decide whether to port `therapeutic_landscape` and `conclusion` (prompts are in `legacy/python-pipeline/prompts/paper/`).
- [ ] **Durable run state and workflow** (replaces `--resume`, the state file and the browser-driven control loop in `app/page.tsx`): persisted run state plus a durable workflow engine, with an approval policy (`ask` | `auto`) so a run can be unattended. The engine is not chosen yet (candidates: Vercel Workflow SDK, Inngest, DBOS).
- [ ] **Unattended runs**: a thin client over "start run / read status / export artifacts" once the workflow exists. Until then `legacy/python-pipeline/pipeline.py` is the only unattended path.
- [ ] **Run record**: persist `promptVersions` per run (the legacy state file did).
- [ ] Remove the legacy synchronous `dock` endpoint from `services/docking/gnina_worker.py` (only `legacy/python-pipeline/agent3.py` uses it; needs a Modal redeploy).
- [ ] Typed contract for Python services: generate TS types from each service's OpenAPI schema instead of writing them by hand on both sides.

## 3. ~~Fix docking endpoint ID mismatch~~ — resolved
The old GPU docking backend was replaced with GNINA on Modal (`services/docking/gnina_worker.py`). The web app reads `GNINA_JOB_API_URL` (`heisenberg/lib/dock-jobs.ts`) and the legacy `agent3.py` reads `GNINA_ENDPOINT_URL`. Neither has a hardcoded default, so the old two-different-defaults mismatch can't recur.

- [x] ~~GNINA currently docks blind...~~ — resolved. `services/docking/gnina_worker.py` now determines the box via a 3-tier fallback: co-crystallized ligand (parsed from the receptor's own `HETATM` records) → `fpocket`-predicted pocket → blind whole-receptor box, only as a last resort. `fpocket` isn't packaged for Debian/apt and its own source fails to compile under GCC 14 (an abandoned C codebase's `strcpy(&array, ...)` bug, only a warning on older GCC), so the Dockerfile fetches the prebuilt conda-forge binary via `micromamba` instead of building from source. Deployed and smoke-tested end-to-end on Modal — all three tiers confirmed live (`8A27.pdb` → `cocrystallized_ligand`, `2FMA.pdb`/`3QL9.pdb` → `pocket_prediction`, correctly excluding `3QL9`'s `M3L` modified-residue false positive). See `modal_app/README.md` for details.

## 4. ~~Remove or relabel `router_agent.py`~~ — resolved
Decided the Fetch.ai/Agentverse/ASI:One discoverability angle isn't wanted. Deleted `router_agent.py` and its unused `uagents`/`uagents-core` deps from `requirements.txt`. `test_agents/` (uAgents scaffolding examples) no longer exists in the repo, so nothing further to clean up there.

## 5. Rewrite README.md
- [ ] Current content is just 3 placeholder lines — write a real description now that this has a proper repo name and public audience.

## 6. ~~Decide fate of large generated artifacts~~ — resolved by the repo restructure
- The old CLI outputs (`old-outputs/`, `pipeline_state.json`) and build caches (`tsconfig.tsbuildinfo`, `.next/`) were deleted. They were untracked and are regenerated by running the tools.
- The 3 sample PDBs the docking worker was validated against (`8A27`, `2FMA`, `3QL9`) moved to `services/docking/tests/fixtures/`. The other 4 stay in `legacy/python-pipeline/structures/` and go when the legacy folder is deleted.
- [ ] Still local and untracked, safe to delete: `heisenberg/deprecated/` (the old blocking dock route) and `heisenberg/styles/` (a stale copy of `app/globals.css`).

## 7. Generalize from cancer-only to any disease — ✅ DONE
The pipeline mechanism is already disease-agnostic (query building just interpolates whatever string is passed in) — this is mostly renaming + prompt wording, not a rewrite. All items below are complete and verified (Python compiles, full test suite passes, TS project typechecks clean):
- [x] ~~Rename `cancer_type`/`cancerType` params throughout Python + TS to `disease`/`condition`~~ — done, repo-wide. Renamed the shared JSON contract (`review.json`/`agent2_output.json`/`agent3_output.json`/`pipeline_state.json` field, and the TS request bodies `/api/review`, `/api/analyze`, `/api/report`, `/api/chat`) consistently on both the write and read side in every file: `review.py`, `agent2.py`, `agent3.py`, `results.py`, `final.py`, `pipeline.py`, `router_agent.py`, `test_agent2.py`, `app/api/review/route.ts`, `app/api/analyze/route.ts`, `app/api/report/route.ts`, `app/api/chat/route.ts`, `app/page.tsx`. Caught and fixed a real bug introduced mid-rename (a parameter/local-`const` name collision in `app/api/report/route.ts`'s `classifyCompound` causing a TDZ `ReferenceError`). Verified: all Python files compile, full `test_agent2.py` suite passes (48/48), and the whole TS project typechecks clean.
- [x] ~~Rename classification label `"cancer_purposed"` → `"disease_purposed"`/`"approved_for_indication"`~~ — done: renamed to `disease_purposed` everywhere (code keys, docstrings, generated report/paper prose) in `results.py` and `app/api/report/route.ts`
- [x] ~~Generalize LLM prompt wording, e.g. `"Mark drugs commonly used for this cancer..."` → `"...for this disease/condition..."`~~ — done in `review.py` / `app/api/review/route.ts`'s drug-protein extraction prompt
- [x] Fixed the oncology-locked system prompt (`"You are an expert oncology researcher..."`, not originally itemized above but found during the fix) → generalized to `"expert biomedical researcher"` in `review.py`, `pipeline.py`, and `app/api/review/route.ts`
- [x] Fixed the silent fallback protein list (`["EGFR", "p53", "KRAS"]` — cancer oncogenes substituted for **any** disease on extraction failure, not originally itemized above but found during the fix): now retries with an Exa web search grounded to the actual disease (`extract_proteins_fallback` / `extractProteinsFallback`), and fails loudly with a clear error if that also comes back empty, instead of silently proceeding on wrong data. Applied in `review.py`, `pipeline.py`, `app/api/review/route.ts`.
- [x] ~~Fix fallback title default in `final.py:407` (defaults to `"Cancer"`)~~ — done, defaults to `"Disease"` now
- [x] ~~Update remaining report heading prose in `results.py`...~~ — done as part of the rename above; repo-wide sweep confirmed the only remaining "cancer" mentions anywhere are deliberate example disease names in help text/docstrings (e.g. "pancreatic cancer", "Alzheimer's disease") and one unrelated narrative slide in `app/deck/page.tsx`.
- [x] ~~**Swap literature source**: arXiv is a weak source for clinical/disease literature...~~ — done: replaced arXiv with the **PubMed E-utilities API** (`ESearch` + `EFetch`, no API key required) in `review.py`, `pipeline.py`, and `app/api/review/route.ts`. Live-verified against real queries (including a non-cancer disease, Alzheimer's) — returns genuine, recent, relevant peer-reviewed papers where arXiv returned little to nothing. Notes:
  - Depth is unchanged (abstract-only, same as the old arXiv usage) — PubMed itself doesn't index full text; that lives in the separate PMC database and only covers a fraction of records. Not pursued here; would be a separate future enhancement.
  - Handles PubMed's structured abstracts (multiple `<AbstractText Label="...">` elements per article — concatenated, not just the first) and citation-only records with no abstract (skipped).
  - Optional `NCBI_API_KEY` / `NCBI_CONTACT_EMAIL` env vars added — raise the rate limit from 3 req/sec to 10 req/sec and follow NCBI's courtesy identification policy; not required for this pipeline's low request volume.

## Completed: Replaced Perplexity with OpenAI + Exa
Swapped every LLM/search call site in both stacks — Perplexity (LLM + built-in web search) → **OpenAI** (`gpt-5.6-terra` by default, via `lib/openai.ts` / the `query_openai` helper in each Python script) for writing/extraction/reasoning, and **Exa** (`lib/exa.ts` / `web_search_exa`) for the calls that actually need live web grounding (Perplexity's browsing had no OpenAI equivalent):
- `discoverRepurposingCandidates` / `_discover_repurposing_candidates` — now searches Exa first, then writes up the findings grounded in those results with `[N]` citations
- `extract_proteins_fallback` / `extractProteinsFallback` (see item 7 above) — same pattern
- Every other call site (writing from already-provided arXiv papers/docking data, JSON extraction, reasoning over existing results) was a straight provider swap, no search needed
- `lib/perplexity.ts` deleted; Python files kept self-contained (each still defines its own client, matching the original pattern)

**Action needed before running:** swap env vars — `PERPLEXITY_API_KEY` → `OPENAI_API_KEY` + `EXA_API_KEY` (optionally `OPENAI_MODEL` to override the default).

## Already checked — no action needed
- [x] `.gitignore` already excludes `.env`, `node_modules/`, `.next/`, `__pycache__/`
- [x] No hardcoded API keys found anywhere in the codebase
