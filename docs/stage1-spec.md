# Heisenberg v1, stage 1: candidate generation (spec)

Drafted 2026-10-07. Part of [problem-statement.md](problem-statement.md); model background is in [kg-model-survey.md](kg-model-survey.md). Source tags: **[verified]** I checked it directly this session (a file, an API, or the code), **[me]** my inference or general knowledge, **[to check]** an open item.

---

## 1. Purpose and boundaries

**Purpose.** Given a disease, return a ranked list of existing drugs that could be worth investigating, with one score per model, built from existing models. Stage 1 answers "which drugs does the evidence suggest?" and nothing more.

**Out of scope for stage 1** (specified later): checks for stopped trials, contraindications and conflicting evidence; mechanism paths; LLM summaries; the user interface; docking and target-first search (v2).

**Inherited decisions:** researchers are the users; reuse existing graphs and models; no training from scratch; no GPU for now; the benchmark decides which models stay.

---

## 2. Constraints that shape the design

| Constraint | What I found | Consequence |
|---|---|---|
| No GPU for now | You said so | Inference only. No training or fine-tuning of TxGNN. MATRIX is a lookup. |
| Your Mac is an Apple M4 Max (arm64), 36 GB RAM, 398 GB free, Docker installed, `uv` installed, Python 3.9 system | [verified] | Enough memory and disk. Python 3.9 is too old for new work, so use `uv` environments. |
| TxGNN installs on Python 3.8 with DGL 0.5.2 | DGL 0.5.2 has builds for linux-64, osx-64 (Intel) and win-64 only. No Apple-Silicon build. [verified] | Run TxGNN in a `linux/amd64` container (local Docker, or Modal CPU). Not natively on the Mac. |
| TxGNN code defaults to `cuda:0` | But it loads weights with `map_location=cpu` and I found no hard-coded `.cuda()` call. [verified in source] | CPU inference looks possible. **Unproven until the smoke test (task T3).** |
| TxGNN's pretrained weights are an "example" file on Google Drive | **Opened in T3 prep (2026-10-07).** The file is a 1.58 GB zip (SHA-256 `0e3e9d3c…ccf8`, 6.0 GB unpacked) of the TxGNN Explorer bundle, not just a checkpoint. It holds `model.pt` (105 MB) and `config.pkl` (hidden size 512, 3 prototypes, `all_nodes_profile`), plus precomputed explanation outputs for indications (GraphMask, attention, GNNExplainer), `paths.csv`, node embeddings, and a 3.8 GB `full_graph_split1_eval.pkl`. The file name suggests the model was trained on the **full graph** (no diseases held out), seed 1. This is a guess until T3 checks it. [verified: listing only] | If it was trained on all known indications, there is no held-out set to flag, and the post-cutoff trials remain a fair frozen test. The precomputed explanation files may supply explanations later without running the explainer ourselves. `config.pkl` was checked with a pickle disassembler (plain values, no code) before loading; everything else is loaded only inside an offline container. |
| TxGNN's knowledge stops around 2021–2022 | The PrimeKG v2.1 release on Dataverse is dated May 2022. TxGNN's own downloader uses different file IDs, so its exact snapshot is **[to check]**. | A long post-cutoff window exists for evaluating it. |
| MATRIX has a single data upload | Hugging Face history shows one score upload on 2026-02-26 and only README edits since. No older versions. [verified] | MATRIX cannot be evaluated retrospectively against its own history. Evaluate it frozen, on data after its cutoff. |
| MATRIX's file holds two kinds of row | Of 39,508,102 rows, 39,444,648 are scored pairs and 63,454 are label-only rows (25,046 known positives, 38,408 known negatives) with no scores. **A labelled pair is not scored:** a disease's known training positives and negatives are absent from its drug list. `y` is NULL on scored rows (the README says 2). [verified in T2] | Known training pairs are looked up in a separate `labels` table. Any pair that appears there cannot count as a new test positive. A model's score for its own training positives cannot be inspected. |
| MATRIX column meanings changed between README versions | **Resolved in T2.** `rank_drug` is the rank among all drugs for the same disease, by `untransformed_treat_score` (100% exact match on 6 diseases). `rank_disease` is the rank among diseases for the same drug, by the same score (100% on one drug). `rank` is global, by `transformed_treat_score` (all but 2 of 39.4M rows match; the 2 are ties). The parquet columns are `not treat score` and `unknown score` (with spaces). [verified] | The store renames them to underscores. Ranks are recomputed in tests, not trusted. |
| The two MATRIX scores order drugs differently within a disease | For 5 test diseases, the top 10 by `transformed_treat_score` and by `untransformed_treat_score` share 6 to 9 drugs, and the top 100 share 65 to 85. [verified] | Benchmark both as separate variants (MATRIX-transformed, MATRIX-raw). Don't assume the "final" score is better. |
| The TxGNN web explorer was down | `txgnn.org` returned HTTP 503 when I checked. [verified] | Don't depend on it. Run the model ourselves. |

---

## 3. Components

### 3.1 Reference data layer

| Item | Detail |
|---|---|
| Every Cure drug list | 1,934 rows (includes deleted entries). Fields include EC id, name, synonyms, `drugbank_id`, `translator_id`, ATC levels, approval status, a `deleted` flag. [verified] |
| Every Cure disease list | 23,224 rows. Fields include MONDO `id`, `name`, `synonyms`, `definition`, many classification flags, `official_matrix_filter`, and **`txgnn` and `mondo_txgnn` columns**. [verified] |
| Crosswalk | **Checked in T1 (2026-10-07): the `txgnn` and `mondo_txgnn` columns are not IDs.** They are TxGNN's disease-area labels (for example `metabolic_disorder\|autoimmune_diseases`, or `other`). The real crosswalk is PrimeKG's `node.csv`: a disease node's `node_id` is one MONDO number (`5180` is `MONDO:0005180`) or several joined by `_` when PrimeKG merged diseases; a drug node's `node_id` is a DrugBank id. Built in `stage1/crosswalk.py`. Coverage: 20,218 of 23,224 Every Cure diseases (87.1%) map to a PrimeKG node, and 5,765 of them share a node with another disease; 1,643 of 1,926 live drugs (85%) map to a PrimeKG drug node. [verified] |
| Disease lookup | User text matched against `name` and `synonyms`; returns MONDO id, TxGNN id when present, and close alternatives. |
| Storage | One DuckDB file plus Parquet copies. |

### 3.2 MATRIX store

| Item | Detail |
|---|---|
| Source | `everycure/matrix-scores` on Hugging Face: 11 Parquet shards, 3.8 GB download, 39,508,102 rows. [verified] |
| Key columns | `source` (drug CURIE, mixed prefixes such as CHEBI, PUBCHEM, DRUGBANK, UNII), `target` (MONDO), `source_ec_id`, three class scores, `transformed_treat_score`, ranks, `y`, known-positive and known-negative flags, trial-outcome flags (`trial_sig_better`, `trial_non_sig_better`, `trial_sig_worse`, `trial_non_sig_worse`), `off_label`, indications-list flags. The flags are sparse: 55 pairs with `trial_sig_better`, 41 `trial_non_sig_better`, 11 `trial_non_sig_worse`, none `trial_sig_worse`; 1,625 pairs on the indications list (1,132 on-label, 493 off-label); 1,384 `off_label`. [verified in T2] |
| Ingest | Load into DuckDB, index by `target`, keep an EC-id join. About 5 GB uncompressed. |
| Query | Given a MONDO id, return the top N drugs with all scores and flags. |

### 3.3 TxGNN scorer (runs in a container)

| Item | Detail |
|---|---|
| Environment | `linux/amd64`, Python 3.8, DGL 0.5.2, a CPU PyTorch build that DGL 0.5.2 supports (the 0.5 line targeted PyTorch 1.5–1.7 **[me, unverified]**), `pip install TxGNN`. |
| Data | PrimeKG files downloaded by TxGNN's own `TxData` (`kg.csv` about 1 GB, plus node and edge files). The Dataverse listing marks the dataset CC0 1.0. [verified] |
| Weights | The example checkpoint from the repository's Google Drive link. Record its hash. |
| Scoring | `TxGNN.predict(df)` takes drug index, relation and disease index and runs the **whole graph forward each call**. [verified in source] So score in chunks of diseases, not pair by pair. |
| Relations | `indication`, `contraindication`, `off-label`. Score all drugs for each requested disease. |
| Output | Parquet of disease, drug, relation, score. |
| Gate | If CPU time or memory is unworkable, fall back to Modal (CPU, or GPU on demand) or ship MATRIX-only first. |

### 3.4 Baselines (CPU only)

| Baseline | Notes |
|---|---|
| Random | Control. |
| Popularity | Rank drugs by number of known indications. The key control for popularity bias. |
| Similar-disease borrowing | Rank drugs approved for the most similar diseases. |
| Network proximity | Drug targets near disease genes in a protein network. Second phase. |
| Matrix completion (BNNR, OMC) | Second phase. Fits a 1,800 × 22,000 matrix on CPU. [me] |

**Rule:** baselines train only on data from before the evaluation cutoff, using the same snapshot as the model they are compared with. For MATRIX comparisons, use the `y = 1` rows of its `labels` table as the interaction matrix.

---

## 4. Interface and output contract

```
generate_candidates(disease: str | MONDO id, top_k=100, models=["matrix", "txgnn", ...]) -> table
```

| Field | Meaning |
|---|---|
| `disease_query`, `disease_mondo`, `disease_name`, `disease_txgnn_id` | What was asked, resolved IDs (TxGNN id nullable) |
| `drug_ec_id`, `drug_name`, `drug_curie`, `drugbank_id` | Normalised drug identity |
| `model` | `matrix`, `txgnn`, or a baseline |
| `score_raw`, `score_transformed` | MATRIX transformed score where relevant |
| `rank_in_model`, `percentile_in_model` | Computed per model |
| `p_treat`, `p_not_treat`, `p_unknown` | MATRIX only |
| `txgnn_indication`, `txgnn_contraindication` | TxGNN only |
| `known_flags` | MATRIX known-positive, known-negative, trial-outcome and off-label flags; TxGNN known indication edges |
| `already_known` | True if any source lists the pair as an existing indication |
| `drug_in_curated_list` | Whether the drug is on the Every Cure list (TxGNN rows only; MATRIX rows are always true) |
| `disease_n_known_indications` | Known indications for this disease in each model's graph, and a low-confidence flag at 0 (TxGNN) |
| `n_models` | How many models scored this drug |
| `snapshot_ids` | MATRIX upload hash; PrimeKG version; weights hash |
| `run_id`, `created_at` | Provenance |

**Behaviour rules**
- **Never average raw scores across models.** Their scales differ. Report each model's rank; offer reciprocal-rank fusion as one option and let the benchmark judge it. [me]
- **Show known indications separately.** The default view hides `already_known` pairs, since a repurposing user wants new candidates. They stay available as a sanity check.
- A drug scored by only one model is kept and flagged through `n_models`.
- A disease that does not resolve returns close matches, not an error.
- A disease with no TxGNN mapping returns MATRIX results only, and says so.

---

## 5. Benchmark design

**Principles:** frozen models (no retraining, which needs GPU); ground truth dated after every model's cutoff; per-disease reporting; comparison against baselines. The first four rows below fit our no-GPU constraint. Retraining-based time splits wait for GPU.

### 5.1 What counts as a positive

A (drug, disease) pair whose **phase 2 or 3 drug trial started after the cutoff**, taken from ClinicalTrials.gov. This measures whether the model ranks what researchers go on to test, which matches stage 1's job. It is not proof the drug works.

Availability check: the ClinicalTrials.gov API counts **4,831 phase 2/3 drug trials started since 2026-03-01**, with conditions, interventions, phase, start date and status per trial. [verified]

### 5.2 Evaluation windows

| Window | Cutoff | Models compared | Purpose |
|---|---|---|---|
| **W2 (primary)** | 2026-03-01 (just after MATRIX's upload; confirm MATRIX's own graph cutoff with Every Cure) **[to check]** | MATRIX, TxGNN, baselines | Head-to-head. Neither model has seen these trials. |
| W1 (secondary) | About mid-2022 (TxGNN's data cutoff **[to check]**) to 2026-02 | TxGNN and baselines only | A larger sample for TxGNN. MATRIX excluded, since it trained later. |
| W3 (later, needs GPU) | Several cutoffs | Retrained models | Proper time-split training. |

### 5.3 Building the test pairs

1. Pull trials from the API for the window.
2. Map conditions to MONDO (name and synonym match on the EC disease list, with a reviewed fuzzy fallback).
3. Map interventions to EC drug ids (name, synonym, DrugBank id).
4. Keep single-agent trials, and flag combination arms.
5. Keep only drugs and diseases present in the models' universes.
6. **Remove pairs the models already know:** any pair in MATRIX's `labels` table (known positive or negative), plus its indications-list and off-label flags, and TxGNN indication edges. Pairs in the `labels` table are not in the score table at all, so for MATRIX this step mostly means "not scored, so not testable"; count them and report them.
7. Audit 100 random pairs by hand and report mapping precision.

### 5.4 Metrics

Per disease, then averaged across diseases: recall at 10, 50 and 100; median percentile of the positives; mean reciprocal rank. Report bootstrap confidence intervals over diseases.

Always stratify by:
- **Diseases with no known drugs vs some vs many** (the cold-start question).
- **Rare vs common** (EC prevalence flags).
- **Drug popularity** (low, medium, high number of known indications), to expose popularity shortcuts.

If a stratum has too few positives, say so rather than reporting a noisy number.

### 5.5 Sanity checks

- **Positive control:** MATRIX does not score its own training positives, so use the 1,625 pairs on the Every Cure indications list that it does score: they should rank high. For TxGNN use its indication edges. If they don't, the pipeline is wrong.
- **Date check:** every trial start date is on or after the cutoff.
- **TxGNN split check:** find which diseases the example weights held out, and flag them in results.

### 5.6 Success for stage 1

I have set no numeric thresholds. Measure the baselines first. The aim is to show, with intervals, whether MATRIX and TxGNN beat popularity and similar-disease borrowing, and where (especially cold-start and rare diseases), and which combination becomes the default.

---

## 6. Environments and layout (proposal) [me]

- A Python project managed with `uv` (Python 3.12) for the data layer, benchmark and API: DuckDB, pandas or polars, `huggingface_hub`, `requests`.
- A separate `linux/amd64` Docker image for TxGNN, with a mounted data volume.
- Keep this separate from the existing Next.js app. The app calls the Python service later.
- Disk: about 4 GB for the MATRIX download, 5 GB for DuckDB, and a few GB for PrimeKG. Fine for your machine.

---

## 7. Tasks, in order, with acceptance checks

| Task | What | Done when |
|---|---|---|
| **T1** Reference data **(done 2026-10-07)** | Download the EC drug and disease lists. Build the disease lookup and the MONDO/DrugBank to PrimeKG crosswalk. | 20 test diseases (mix of common and rare) resolve to MONDO and to TxGNN ids. Failures listed. **Result:** all 20 resolve to MONDO; 19 map to a PrimeKG node; idiopathic pulmonary fibrosis has none, and MATRIX does not score it either (T2), so it is kept on purpose as the "no model covers this disease" case. The MATRIX-only path is tested with another disease from the 2,336 that MATRIX scores and PrimeKG lacks. List in `heisenberg-stage1/config/test_diseases.csv`. |
| **T2** MATRIX store **(done 2026-10-07)** | Download the shards. Build DuckDB. | Row count is 39,508,102. Rank columns recomputed and understood. Top results for 5 diseases look sensible. **Result:** 39,508,102 rows (39,444,648 scored + 63,454 label-only). Rank columns understood (see section 2). Top 10 for Parkinson disease, type 2 diabetes, rheumatoid arthritis, ALS and Huntington disease are clinically plausible (for example selegiline and amantadine for Parkinson; riluzole first for ALS). 19 of the 20 test diseases are scored with 1,588 to 1,825 drugs each; **idiopathic pulmonary fibrosis is not scored**, although the Every Cure list marks it as in the MATRIX filter. Coverage is not the same as `official_matrix_filter`: Alzheimer disease and asthma are scored though outside it, and 563 diseases inside it are not scored. Repeatable check: `python -m stage1.verify_matrix`. |
| **T3** TxGNN smoke test **(done 2026-10-07; gate passed)** | Build the container. Load weights on CPU. Score one disease against all drugs. | It runs. Time and peak memory are recorded. Known indications of sample diseases rank high. The weights' split is identified. **Decision gate.** **Result: GO.** The image builds and runs on the Mac under x86 emulation (Python 3.8.20, torch 1.7.1 CPU, DGL 0.5.2, TxGNN pinned to commit `0e0203c`), offline. First run 456 s end to end (219 s of that is one-time graph preprocessing, cached afterwards); a cached start takes about 2 minutes. Scoring one disease against all 7,957 drugs takes **46 s**; 19 diseases in one batch take **64 s** (about 1 s per extra disease); peak memory **7.9 GB** of the 15.7 GB given to Docker. Batch and single-disease scores are identical. Scoring about 1,000 diseases should take roughly 20 minutes (extrapolated, not yet run). The weights match the `full_graph` split: TxGNN builds it as a random 95/5 edge split (the 5% is used as both validation and test), so about 5% of known indications were probably not used in training. The seed (1) is a guess from the file name `full_graph_split1_eval.pkl`. Known indications rank high for diseases the model trained on (Parkinson: median 19 of 7,957, best 2; Alzheimer: median 4.5; ALS: riluzole second), but that is partly memorisation, not accuracy. Details below. Repeatable: `docker/txgnn/smoke_test.py`. |
| **T4** Candidate table | The function and output contract. | For 5 diseases, a merged table with all fields and flags. |
| **T5** Ground truth (W2) | Pull trials, map, filter known pairs, audit. | Numbers of diseases and pairs reported by stratum. Mapping precision reported from the 100-pair audit. |
| **T6** First baselines | Random, popularity, similar-disease borrowing. | Same-format candidate tables. |
| **T7** Benchmark and report | Run everything on W2 (and W1 for TxGNN). | Tables and intervals by stratum. A written decision on the default model combination. |
| **T8** Later | Network proximity, matrix completion, anything needing GPU. | |

Order: T1 then T2 then T4. T3 runs in parallel. T5 then T6 then T7.

---

## 8. Risks and open questions

- **TxGNN on CPU is unproven.** The old dependency stack and full-graph forward pass are the risks. T3 is the gate.
- **Example weights:** unknown training split, and hosted on Google Drive.
- **MATRIX's graph cutoff is unconfirmed.** It sets the W2 cutoff. Ask Every Cure, or read their release notes closely.
- **Trials are a weak proxy,** and mapping errors and combination trials add noise. Many new trials involve drugs outside the lists.
- **ID mapping:** PrimeKG merged some diseases that MATRIX keeps separate (Every Cure says so in its v0.12.0 notes). Measured in T1: 5,765 Every Cure diseases share a PrimeKG node. For example, TxGNN's Parkinson disease node stands for 18 MONDO diseases, and its Alzheimer disease node for 21. A TxGNN score for such a node applies to the whole group, so benchmark rows on merged nodes must be flagged, and a hit on a merged node cannot be credited to one specific subtype.
- **Rare vs common labels are patchy.** Every Cure's `prevalence_world` band is filled for only 7,228 of 23,224 diseases, and some values are odd (essential hypertension shows `27583.33 in 100,000`). The rare/common stratum needs a fallback such as the Orphanet flags.
- **Different drug universes:** MATRIX covers about 1,800 curated drugs; TxGNN about 7,957 candidates. Only 1,643 of Every Cure's 1,926 live drugs have a PrimeKG drug node, so TxGNN cannot score about 15% of the MATRIX drug list. In the other direction, only 20.6% of TxGNN's 7,957 "drugs" are on the Every Cure list; the rest are largely experimental compounds and biologics from DrugBank. [verified]
- **TxGNN's raw ranking has a hub problem. [verified in T3, 19 diseases]** Harmaline is in the top 50 for 13 of 19 test diseases, and 25 drugs are in the top 50 for 5 or more diseases. Most of those are experimental ligands, not medicines. For ulcerative colitis, psoriasis, tuberculosis and pancreatic cancer the raw top 8 is almost entirely such compounds. Restricting TxGNN to the Every Cure drug list removes most of this: Parkinson disease gives pimavanserin, entacapone, rasagiline, safinamide and amantadine; ulcerative colitis gives mesalazine, sulfasalazine, infliximab and guselkumab. So the default TxGNN view is restricted to the curated list, and the benchmark compares raw, restricted, and restricted with a per-drug bias correction.
- **TxGNN does not generalise to diseases with no known indications in its graph. [verified in T3]** Tuberculosis, pancreatic ductal adenocarcinoma, MASH, Friedreich ataxia and Rett syndrome have 0 known indication edges in PrimeKG, and their restricted top lists are unrelated drugs (inclisiran, cisatracurium, plecanatide). The output must carry the number of known indications in TxGNN's graph, and a low-confidence flag when it is 0.
- **MATRIX and TxGNN barely agree on the top of the list.** With TxGNN restricted to the curated list, the median overlap of the two top-20 lists across the 18 diseases both cover is 2 of 20. This is partly an artefact: MATRIX's list omits its training positives, while TxGNN's list is full of known indications. The benchmark has to compare them on pairs neither model trained on.
- **Licences:** PrimeKG on Dataverse is CC0 1.0 [verified], though underlying sources may carry their own terms [me]. MATRIX and Every Cure lists are CC-BY with upstream licences. ClinicalTrials.gov terms **[to check]**.
- **What GPU would unlock:** retraining on fresher graphs, time-split retraining (W3), and trying newer models such as CellAwareGNN if their code is released.

---

## 9. Data available for the later verification stage (not part of stage 1)

Noted because it affects planning, not because stage 1 uses it:
- MATRIX's file carries trial-outcome flags and known-negative labels, but only for a small set: about 107 pairs with trial-outcome flags and 38,408 known-negative label rows, none of which are scored. [verified in T2]
- ClinicalTrials.gov lists 4,253 phase 2/3 trials that started 2020–2024 and were terminated, withdrawn or suspended. [verified]

---

## 10. How TxGNN's explanation works, and what it showed on a real query (2026-10-09)

Run for Parkinson disease with three drugs: riluzole and tetrabenazine (candidates) and entacapone (a known drug, as a control). Everything below ran offline in the container. Scripts: `docker/txgnn/explain_paths.py`, `docker/txgnn/similar_diseases.py`.

**What the explainer is.** GraphMask is trained once, on top of the frozen model, for the *indication* relation. It learns a small gate for every edge in the graph, in each of the model's 2 message-passing layers (`layer1_att`, `layer2_att`, each 0 to 1), that switches the edge off if the prediction does not change. The result is one score per edge for the whole graph (7,676,670 edges): median 0.03, 14% above 1.0 (sum of the two layers, range 0 to 2). It is **not** computed per drug-disease pair. [verified]

**How a path is built (this part is the Explorer notebook, not GraphMask).** For the disease and the drug, keep the top 10 neighbours per relation type, then the top 10 neighbours of those; take the subgraph on them; list every simple path of up to 4 edges from disease to drug; score a path as the mean edge score; group by meta-path; drop meta-paths with contraindication, drug-drug, off-label or `anatomy_protein_absent` edges. [verified from `demo.ipynb`]

**Findings**
- **The notebook's neighbour selection looks inverted.** It pushes negated scores onto a heap and then calls `heapq.nlargest`, which picks the *lowest*-scoring neighbours. Measured: the neighbours it picks have a mean edge score of 0.31 to 0.37, against 0.64 to 0.71 when picking the highest, and 0.35 across all edges. The shipped `paths.csv` has a mean path score of 0.31, consistent with this. We re-implement the selection with the highest scores. [verified by running both]
- **Shipped explanations cover 193 pairs and 42 diseases, none of our test diseases.** We have to generate paths ourselves from `graphmask_output_indication.pkl`, which is plain CPU work (about 1 to 2 minutes per pair in an unoptimised first version).
- **Only the indication head has explanation data.** No GraphMask output for contraindication ships, so a "why contraindicated" explanation would need GraphMask trained on that relation.
- **Explanations describe resemblance, not mechanism.** For riluzole the strongest paths go Parkinson disease, to a similar disease, to a known drug, then to riluzole through a shared transporter (ABCG2) or a shared side effect (nausea). For tetrabenazine they go through DRD2, a target it shares with real Parkinson drugs. The graph does not say whether a drug activates or blocks a target.
- **Strong edges are rare.** If every edge on the path must have a summed gate of at least 1.0, there is no path at all for riluzole or tetrabenazine, but 103 paths for entacapone.
- **Known drugs explain themselves strongly (best path 1.78 of 2.0), mostly through training indications.** Candidates peak around 1.2 to 1.3.
- **The contraindication score does not separate bad from good.** Tetrabenazine scores 0.989, but approved Parkinson drugs do too: entacapone 0.93, amantadine 0.94, carbidopa 0.98. It cannot be shown to users as a warning until the benchmark has tested it.
- **How much a disease "borrows" from similar diseases.** The weight is 0.7 * exp(-0.7 * n) + 0.2 for n known indications: Parkinson (27) borrows 0.20; ALS (2) 0.37; a disease with none borrows 0.90. Borrowed neighbours can be very weak: for tuberculosis the top similarity is 0.22 (ornithosis, miliary tuberculosis, leprosy), for pancreatic cancer 0.14 (astrocytomas), for Rett syndrome 0.12. This is the likely cause of the unrelated drugs seen for those diseases. [verified]

**Proposed output fields from this** (added to the section 4 contract): `n_known_indications`, `borrow_weight`, `similar_diseases_used` with their similarities, `explanation_best_path_score`, `explanation_path_mix`, and a low-confidence flag when `borrow_weight` is above 0.5 and the best similarity is below 0.3.
