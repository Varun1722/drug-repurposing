# Heisenberg: problem statement and scope

Drafted 2026-10-07. Revised 2026-10-07 after the door order changed. Built from the BIOS research ([bios-research.md](bios-research.md)), a code audit of the current app, a literature review of repurposing approaches, and a survey of knowledge-graph models ([kg-model-survey.md](kg-model-survey.md)). Source tags: **[verified]** I read it in a source this session, **[audit]** from the code audit, **[me]** my inference or general knowledge.

---

## 0. What changed

The first draft made target-first (protein in, drugs out) the first version and disease-first the second. That order is now reversed, because the project already started on knowledge graphs, which do well on drug–disease pairs.

- **v1 is disease-first:** existing knowledge graphs and GNN models, used to do the best possible job on drug–disease pairs.
- **v2 is target-first.** Its full scope from the first draft is preserved in Part B, with only the version label changed.

---

# Part A: v1, disease-first

## 1. Problem statements

### 1.1 The shared problem

Most diseases, especially rare and newly emerging ones, have no effective treatment, and new-drug development takes over a decade with very high failure rates. Thousands of approved and investigational drugs already have human safety data, but the evidence that one might work elsewhere is scattered across knowledge graphs, literature, trials, protein structures and clinical records. Existing computational methods each see one slice, report scores that are hard to trust, and rarely attach the mechanism, safety and exposure evidence a researcher needs to decide what to test.

### 1.2 v1: disease-first

Researchers deciding which existing drugs to investigate for a disease face tens of thousands of possible drug–disease pairs. Published models give a score from one graph, benchmarks overstate their accuracy, and the explanations they offer are rarely tested or paired with evidence against.

**Heisenberg v1** takes a disease and returns a short ranked list of existing drugs worth investigating. Each comes with:

- which models and knowledge graphs support it, and where they disagree;
- a mechanism path, with the source of every link;
- what argues against it: contraindications, stopped trials, conflicting evidence;
- a confidence grade calibrated on time-split benchmarks.

### 1.3 v2: target-first

Preserved in Part B, section 12.

## 2. Decisions made

| Decision | Choice |
|---|---|
| Primary user | Researchers working on drug discovery |
| First door (v1) | Disease-first: drug–disease pairs from existing knowledge graphs and GNN models. Changed from target-first. |
| Second door (v2) | Target-first (protein in, drugs out). Full scope preserved in Part B. |
| Models | Reuse existing graphs and models; no training from scratch in v1. Use the best available, combining multiple databases. The benchmark decides which. |
| v1 focus | Candidate generation first: which drugs could treat the disease. GNN-based models lead, with MATRIX and TxGNN the favourites. Verification (stopped trials, contraindications, conflicting evidence) is specified later. |
| v2 target scope | Viral proteins first; design stays generic |
| v2 input | Structure file (PDB/mmCIF) or sequence |
| v2 compound library | Approved plus clinical-stage (about 5,400 to about 12,000 compounds) |
| Project type | Open research tool |

## 3. Scope

### 3.1 v1 scope (disease-first)

**In v1**
- Disease input by name or ontology identifier.
- Approved and investigational drugs available in the knowledge graphs.
- Ranked candidates with mechanism paths and counter-evidence.
- Reuse of existing scores and pretrained models.
- Retrospective validation on time-split data.

**Not in v1**
- Protein-first search and docking (v2).
- De novo molecule design.
- Training a new knowledge-graph model from scratch.
- Patient-level personalisation.
- Any clinical or efficacy claim.

### 3.2 v2 scope

Preserved in Part B, section 13.

## 4. How the approaches fit

| Approach | Role in v1 | Role in v2 |
|---|---|---|
| Knowledge-graph scores and GNNs (MATRIX, TxGNN) | Main engine | Optional context for human targets |
| Path explanations (KGML-xDTD and similar) | Explanation layer | Not used |
| Literature / LLM | Evidence summaries constrained to retrieved, cited sources. Do not generate candidates or approval status. | Same |
| Trial and label evidence | Support and counter-evidence | Not used |
| Structure-based screening | Not used (existing code kept) | Main engine |
| Known drug–target data | Mechanism check | Evidence layer |
| Phenotypic screening | Wet-lab validation, outside the software | Same |
| Generative design | Later fallback when no existing drug fits | Same |

## 5. v1 pipeline

Current focus: stage 1 (candidate generation). Stages 2 to 6 will be specified later. Checks for stopped trials and contraindications belong to those later stages.

| # | Stage | What happens | Sources and controls |
|---|---|---|---|
| 0 | Disease intake | Normalise the input to the ontology identifiers the graphs use, including synonyms. | Handle diseases missing from a graph explicitly |
| 1 | Candidate generation | Score drugs with an ensemble: MATRIX scores plus TxGNN inference. Add other models only if the benchmark shows a gain. | Record which models scored each drug and where they disagree |
| 2 | Mechanism paths | Retrieve multi-hop paths from a unified graph, with the source of every edge. Check against known curated mechanisms. | EC-KG, RTX-KG2, PrimeKG; DrugMechDB |
| 3 | Evidence and counter-evidence | Find trials (including stopped ones), label contraindications, and summarise literature with an LLM limited to retrieved, cited text. | ClinicalTrials.gov, openFDA, PubMed |
| 4 | Triage | Filter by approval status and route, flag known safety issues, and compare against what is already an approved indication. | Authoritative status sources, not LLM text |
| 5 | Dossier and grade | Per candidate: support, disagreement, path, counter-evidence, evidence tier. | See section 6 |
| 6 | Handoff and feedback | Export the ranked list. Record outcomes for later calibration. | |

### Stage 1 output contract (draft; full spec in [stage1-spec.md](stage1-spec.md))

Each candidate row carries: disease and drug identifiers normalised to the graphs' identifiers; one score per model, with its rank and percentile within that model; class probabilities where a model provides them (MATRIX: treat, not treat, unknown); model and data-snapshot versions; and explanation edges where a model provides them. The later stages read these fields, so nothing has to be recomputed.

## 6. v1 evidence tiers

| Tier | Meaning |
|---|---|
| A | Independent evidence of a different type (trial evidence in this or a closely related disease, or human-data support) plus model agreement |
| B | Multi-model agreement plus a coherent, curated mechanism path |
| C | A single model's score only |
| Contradicted | A contraindication, stopped trial or conflicting evidence was found. Shown regardless of tier. |

MATRIX and TxGNN share data through PrimeKG, so their agreement counts as consistency, not independent confirmation. [me]

## 7. v1 benchmark and success criteria

- **Time split:** use a graph snapshot from before a cutoff, and test against drug–disease links added afterwards. Open Targets provides time-stamped releases, and Every Cure documents a clinical-trials ground-truth set dated after its graph cutoff. [verified]
- **Reporting:** per disease, not only pooled. Include cold-start diseases (no known drugs) and degree-matched negatives, to defeat popularity shortcuts. [verified: KG-Bench, CellAwareGNN]
- **Baselines:** popularity, network proximity, matrix completion (OMC, BNNR), TxGNN alone, MATRIX alone, and the ensemble.
- **Explanation quality:** recovery of curated mechanism paths (DrugMechDB), faithfulness (deletion and insertion), stability under graph perturbation, and a small expert review.
- **Calibration:** do confidence grades predict outcomes?
- **Success:** the ensemble beats the best single model and the popularity baseline on cold-start and degree-matched tests, explanations recover curated mechanisms at a reported rate, and grades are calibrated. I have not set numeric thresholds. Measure baselines first.

## 8. v1 data sources

| Source | Gives | Terms |
|---|---|---|
| [MATRIX scores](https://huggingface.co/datasets/everycure/matrix-scores) | 39.4M precomputed scores, three-class, plus 63,454 training-label rows and about 107 pairs with trial-outcome flags | CC-BY 4.0; upstream licences apply |
| [EC-KG and drug/disease lists](https://docs.dev.everycure.org/releases/public_data_releases/) | Unified graph (7.35M nodes, 81.1M edges), curated lists | CC-BY 4.0; upstream licences apply |
| PrimeKG and TxGNN | TxGNN's graph, code and example weights | Code MIT; Dataverse lists the PrimeKG dataset as CC0 1.0 |
| RTX-KG2 | Large Biolink graph; KGML-xDTD predictions served in ARAX | Code MIT, artifacts CC-BY; content licences restrict redistribution |
| DrugMechDB | Curated mechanism paths | Not checked |
| Open Targets | Time-stamped drug–disease data for benchmarks | Not checked |
| ClinicalTrials.gov, openFDA | Trial status and outcomes; labels and contraindications | Not checked |
| LINCS | Perturbation transcriptomics (optional) | Not checked |

## 9. v1: what changes in the current code

From the audit [audit]:

| Today | v1 needs |
|---|---|
| Candidate drugs come from an LLM reading web search results; FDA status is LLM text | Candidates from knowledge-graph scores and models; status from authoritative sources |
| Target proteins are extracted by an LLM from abstracts | Not needed in v1; the graph supplies targets and paths |
| "Confidence %" is a docking pose score | A calibrated ensemble score and an evidence grade, named for what they are |
| No citations or provenance per candidate | Provenance on every edge and evidence item |
| Docking and structure pipeline is the core | Keep the code for v2; take it off the v1 path |
| Free-text disease input only | Normalise to ontology identifiers |
| No persistence | Stored runs: inputs, data and model versions, outputs |
| No auth or rate limiting | Add before going public |
| Duplicate Python and TypeScript pipelines | Python core, since the models are PyTorch-based; TypeScript as the UI [me] |

## 10. v1 build order

1. **Benchmark harness:** time-split, per-disease, with the baselines, MATRIX and TxGNN. This shows what each can do before anything is built around it.
2. **Data layer:** disease normalisation, unified drug and disease identifiers, loading MATRIX scores and the graph.
3. **Model layer:** MATRIX lookup, TxGNN inference with explanations, path retrieval.
4. **Evidence layer:** trials, labels, constrained literature summaries.
5. **Dossier, interface and visualisation, persistence and guardrails.**
6. **Then v2,** using the build order in Part B.

## 11. v1 open questions and risks

- **Compute:** whether TxGNN inference and its explainer need a GPU is not measured.
- **Different drug universes:** MATRIX covers about 1,800 curated, mostly approved drugs, while TxGNN covers about 7,957 candidates including experimental ones. The ensemble must handle drugs only one model scores.
- **Reproducibility of newer models:** several 2026 models have no code I could confirm.
- **Diseases missing from the graphs:** needs an explicit fallback.
- **Licences:** fine for an open research tool, but record each source's terms.
- **Counter-evidence is mostly not in the models:** MATRIX's file carries known-negative labels (38,408 pairs, unscored) and trial-outcome flags on only about 107 pairs, so stopped trials and conflicting edges in general must be built, not borrowed.

---

# Part B: v2, target-first (preserved from the first draft)

## 12. v2 problem statement

When a new pathogen emerges, researchers have a protein structure (or just a sequence) within weeks, but no drug and almost no literature about that protein. Thousands of approved and clinical-stage compounds already have human safety data, and some may inhibit the new protein. There is no cheap, trustworthy way to shortlist which to test first. Docking-only shortlists have low hit rates and are rarely validated. Literature-based and graph-based tools cannot help, because the target is new and has no links in any database yet.

**Heisenberg v2** takes a viral protein (structure or sequence) and returns a ranked shortlist of approved and clinical-stage compounds to test first. Each comes with an evidence dossier covering:

- structural plausibility, checked against controls;
- knowledge transferred from homologous proteins that already have known inhibitors;
- exposure and safety flags;
- a confidence grade calibrated against retrospective benchmarks.

It is open, reproducible, and built to hand off to a wet lab.

## 13. v2 scope

**In v2**
- Viral enzymes with a defined active site (proteases, polymerases and similar).
- Direct-acting candidates: compounds that bind the viral protein.
- Approved and clinical-stage compounds only.
- Retrospective validation against public data.

**Not in v2**
- De novo molecule design (the BIOS-style approach).
- Host-directed drugs, which target the human side and need a different pipeline.
- Protein–protein interfaces such as spike–ACE2, which are poor targets for small-molecule docking [me].
- Any clinical or efficacy claim.

(Disease-first search is no longer deferred. It is v1.)

## 14. v2 pipeline

| # | Stage | What happens | Data and controls |
|---|---|---|---|
| 0 | Target intake | Accept a structure, or predict one from a sequence. Clean the file and record provenance. | Report model confidence around the pocket. Predicted structures need care. [verified: JCAMD 2026 review] |
| 1 | Site definition | Find drugged homologs (Foldseek or BLAST against the PDB) and transfer the known ligand site. Otherwise use a pocket predictor, or the user's site. | Priority: homolog ligand site, then predicted pocket, then blind box |
| 2 | Compound universe | Merge sources, deduplicate by parent InChIKey, standardise with RDKit, and annotate status, targets and indication. | DrugCentral, Broad Repurposing Hub, ReFRAME |
| 3 | Known-activity lookup | Find measured activity on this target or close homologs. Rank library compounds by similarity to known homolog inhibitors. | ChEMBL, BindingDB, PubChem BioAssay, DrugCentral targets |
| 4 | Structure screen | Dock the library with at least two independent methods. Score interaction similarity to known inhibitors. Check pose validity. | Controls: redocking, known actives vs property-matched decoys, ligand-only baseline |
| 5 | Triage | Flag assay interferers, known toxicity, and whether achievable blood concentration is plausible. Check availability. | Potency can't be predicted, so the flag is "needs EC50 below Cmax" |
| 6 | Dossier and grade | Per compound: sources, scores, controls, caveats, evidence tier. | See section 15 |
| 7 | Handoff | Ranked list and plate-ready export. Record assay outcomes for feedback. | |

New cold-start drug–target models (HierHGT-DTI, ColdstartMHDTI) are candidates for stage 3 in this version; see the survey. [verified]

## 15. v2 evidence tiers and dossier

| Tier | Meaning |
|---|---|
| A | Measured activity on this target or a close homolog |
| B | Consensus structural support plus similarity to known inhibitors |
| C | Single-method docking only |

Each dossier shows which sources supported the compound, what the controls showed, and what could make it wrong. The confidence grade is calibrated on the benchmark, not asserted.

## 16. v2: why this design (evidence)

- The COVID record says docking alone is a weak shortlist: a 2021 PNAS screen of 6,218 drugs found most docking predictions were false positives. Pre- and post-docking filtering raised it to 7 of 38 compounds inhibiting the virus in cells. [verified]
- Enzyme hits do not guarantee antiviral activity. One protease screen found two compounds that blocked the enzyme at 1–4 µM, and neither stopped the virus in cells. [verified]
- Direct-acting repurposed antivirals mostly failed clinically in WHO Solidarity. What worked clinically was host-directed (corticosteroids, IL-6 blockers, baricitinib). [verified] So v2 should promise a test shortlist, not a treatment.
- Docking scores are not comparable across targets, but ranking many compounds against one pocket is the legitimate use. [verified: JCAMD 2026 review] [me]
- A preprint reported that a ligand-only baseline matched Boltz-2 on a standard affinity benchmark, which is why a ligand-only control is in stage 4. [verified]

## 17. v2 data sources

| Source | Gives | Terms |
|---|---|---|
| [DrugCentral](https://drugcentral.org/download) | Approved drugs (FDA, EMA, PMDA), structures, drug–target data | Not checked |
| [Broad Repurposing Hub](https://repo-hub.broadinstitute.org/repurposing) | 5,400+ approved and clinical compounds, annotated | Annotations CC-BY 4.0, research use only, no commercial redistribution |
| [ReFRAMEdb](https://reframedb.org/) | About 12,000 compounds, screening data downloads | Not checked |
| [Every Cure public data](https://docs.dev.everycure.org/releases/public_data_releases/) | Drug list, graph nodes and edges, MATRIX scores | CC-BY 4.0; upstream licences still apply |
| ChEMBL, BindingDB, PubChem BioAssay | Measured drug–target activity | Check current terms [me] |
| PDB, UniProt, AlphaFold DB | Structures and sequences | Check current terms [me] |

Because this is an open research tool, the non-commercial terms are acceptable now, but record each source's licence so it stays auditable.

## 18. v2 benchmark and success criteria

- **First target:** SARS-CoV-2 main protease, using only information available at an early-2020 cutoff.
- **Ground truth:** enzymatic activity first. Public sources include ChEMBL, PubChem BioAssay and COVID Moonshot, plus curated sets of 150 known inhibitors and 81 inhibitors with 113 inactives from the "Haste makes waste" review. [verified]
- **Secondary truth:** cell-based antiviral results (NCATS screen of 8,810 compounds, ReFRAME screens). [verified]
- **Baselines to beat:** random, similarity to known inhibitors of related proteases, raw GNINA CNN score (today's Heisenberg), Vina score, and a ligand-only model.
- **Metrics:** enrichment at the top 1% and 5%, AUPRC, hit recovery in the top 50, and calibration of the confidence grade.
- **Second target:** a structurally different viral enzyme, to show it generalises. To be chosen.
- **Success:** beats all baselines on the first target, holds on the second, and the grades are calibrated.

## 19. v2: what changes in the current code

From the audit [audit]:

| Today | Needed |
|---|---|
| Input is a disease name only | Target intake: structure upload or sequence |
| PDB found by name search sorted by resolution, no protein mapping | Homology-based site definition |
| "Confidence %" is the best pose's CNN score (pose quality) | Per-target calibrated score from controls, using the affinity values already computed |
| Worker returns scores only, no poses | Return poses for validity checks, interaction fingerprints and the viewer |
| One container per ligand, cap of 50 ligands per job | A bulk mode for thousands of ligands, with a cost estimate I haven't made |
| Compounds from LLM suggestions and PubChem, FDA status is LLM text | Curated library; status from DrugCentral or the Hub |
| No persistence | Stored runs: inputs, data versions, seeds, outputs |
| No auth or rate limiting | Add before going public |
| Duplicate Python and TypeScript pipelines | Make Python the scientific core, since RDKit and gemmi have no JavaScript equivalent. TypeScript becomes the UI. [me] |

## 20. v2 build order

1. **Benchmark harness on the main protease, using the current GNINA worker.** This measures how good today's approach actually is, before changing anything.
2. **Data layer:** merged compound library and known-activity lookup.
3. **Target intake and site definition.**
4. **Controls, consensus scoring and calibration.**
5. **Dossier, UI, persistence and guardrails.**

## 21. v2 open questions and risks

- **Compute cost and time** for 5,000–12,000 ligands per target is unmeasured.
- **Site choice** for viral proteins with allosteric or cryptic pockets.
- **ReFRAME terms:** structure data downloads exist, but I have not checked the licence.
- **Scope assumptions I made** (change them if wrong): active-site enzymes only, direct-acting only.
- **Time-to-shortlist target** is not yet set. Measure first, then decide.

---

## Sources

- bios-research.md and kg-model-survey.md (this repo)
- Computational drug repurposing review, Nature Reviews Drug Discovery 2025
- Reproducibility, validation and failure modes in docking, J Comput Aided Mol Des 2026
- PNAS 2021 virtual screening of 6,218 drugs; Frontiers in Drug Discovery 2022 protease screen
- WHO Solidarity trial; NCATS SARS-CoV-2 screen; ReFRAME screens (Nature Communications 2021)
- "Haste makes waste" docking review (2021)
- Every Cure public data releases; Broad Repurposing Hub; DrugCentral; ReFRAMEdb
- KG-Bench; knowledge-graph embedding leakage study; CellAwareGNN
- Pat Walters and Björn Mattsson preprint on ligand-only baselines (via a LinkedIn post)
