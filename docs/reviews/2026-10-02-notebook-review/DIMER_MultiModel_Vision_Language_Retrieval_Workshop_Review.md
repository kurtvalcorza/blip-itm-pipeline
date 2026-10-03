# DIMER Vision-Language Retrieval Notebook — Review

**Verdict: Needs revision**
**Review date:** 2 October 2026
**Repository:** `kurtvalcorza/blip-itm-pipeline`
**Notebook:** `tutorials/DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb`
**Reviewed commit:** `a25fdf6d2e1c8b5109300185ad21dac2e57c9723` (`main`, confirmed against the GitHub API at review time)
**Notebook Git blob:** `e5651bb143d8576d289bacad53976b474c52bb6f`
**Finding prefix:** `BVR`

## Executive assessment

The notebook runs a real and well-guarded experiment: three frozen first-stage retrievers (SigLIP 2, SigLIP v1, BLIP ITC) on one pinned VizWiz-Captions gallery of 391 photographs and 1,737 captions, a common NumPy evaluator, analytical and colour/keyword baselines, BLIP ITM reranking of each shortlist, freeze-before-test, digest-verified index export with full ranking-parity reload, and an optional `FULL` BLIP adaptation with fresh adapter reload. A maintainer-supplied Colab T4 run of the identical code cells completed the `STANDARD` path on 2026-09-26.

Three things stop it from being ready for its intended learner. The principal reranking table compares coarse text→image Recall@1 over **all 1,737 captions** with reranked text→image Recall@1 over **391 first captions**, so the apparent reranking gain mixes a query-set change with the reranker's effect (the repository's own design specification asks for "coarse canonical T2I R@1"). The category diagnostic scores each category in its own sub-gallery of a different size, which the notebook itself teaches is not comparable. And the optional BYOD branch ends by printing a directory path: the learner never sees the retrieval or reranking results it computed, and is not told how to get a ZIP into the runtime.

This review does **not** establish that any recorded score is wrong, that the hosted default run failed, or that the reranker does not help. It establishes that two displayed comparisons are not like-for-like and that the BYOD journey does not reach a visible result.

## 1. Review contract and evidence

| Item | Scope |
|---|---|
| Declared profile and mode | `MULTI-CAPABILITY` / `WORKSHOP`; notebook specification `2.1` (`metadata.dimer`) |
| Specification baseline | Declared 2.1; requirements checked against fleet `NOTEBOOK_SPEC.md` 2.2 at `ml-worker` `origin/main` `b1cfe13` (blob `7428d5be`) |
| Design specification | `docs/vision-language-retrieval-workshop-spec.md` (status "Proposed") |
| Stated learner | Can run Python cells in Colab/Jupyter; new to multimodal embeddings, retrieval or reranking |
| Supported runtime | NVIDIA Tesla T4 or equivalent; `STANDARD` default tier |
| Promised outcomes | Capability A dual-encoder retrieval; B cross-modal reranking; C (`FULL`) bounded BLIP adaptation; 12 learning objectives; BYOD; exported indexes and provenance |
| Optional paths | `WORKSHOP_TIER="FULL"`; `USE_BYOD` + `BYOD_ZIP_PATH`; scratch-cell candidate-oracle activity |
| Shared unit | The same notebook (identical code cells) ships in `siglip2-vision-language-pipeline`; fixes here diverge that copy until it is updated |

### Evidence actually obtained

- **Source inspection:** all 72 cells at the reviewed commit, the design specification, `docs/release-verification.md`, `tutorials/README.md`, the workshop tests and the validator.
- **Documented execution evidence:** `docs/execution-evidence/2026-09-26/DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb` (Colab, Tesla T4, Python 3.13.15, torch 2.14.0+cu130). Its code cells match commit `89eea02`; the only later notebook change on `main` (`f474a75`) is markdown, so the run covers the reviewed code. Scope: `STANDARD` default only. `FULL` and BYOD were not run. Saved outputs were inspected; execution was not repeated.
- **Direct execution (this review):** CPU only, Windows, Python 3.13 (anaconda base; the packet's `eo-notebook-test` env crashes on any NumPy matmul, `0xc06d007f`). The notebook's own helpers (`retrieval_metrics`, `candidate_oracle`, `rerank_with_blip`, `load_byod`) were executed on synthetic NumPy data with inert stand-in models. No model weights were loaded; no GPU was used. Probes are in `DIMER_MultiModel_Vision_Language_Retrieval_Workshop_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
- **Learner observation:** none. Instructional findings are judgments from source inspection, not measured learning outcomes.

### Journeys

| Journey | Basis | Outcome |
|---|---|---|
| First-time learner | Source inspection | Mostly well oriented (how-to-use, roadmap, glossary, interpretation boundaries). Gaps: BVR-M1, M2, m4, m5, m6, m9 |
| Clean default (`STANDARD`) | Documented execution evidence, 2026-09-26 | Completed to the summary with zero saved errors. Two displayed comparisons are not like-for-like (M1, M2) |
| Active learning | Direct execution of `candidate_oracle` (synthetic); `FULL` not verified | The scratch-cell activity uses only saved validation scores and works. `FULL` adaptation has no hosted run |
| Reuse and recovery | Direct execution of `load_byod` / stand-in BYOD orchestration; index reload parity in hosted run | Invalid archives are refused, but messages do not name the offender (m1); the BYOD result is never shown (M3). BYOD hosted run: not verified |

## 2. Separate judgments

- **Technical correctness:** strong. Digest-pinned models and corpus, `trust_remote_code=False`, `weights_only=True` for the BLIP pickle, decoded-pixel leakage checks, finite/range checks on ITM output, full ranking-parity index reload, safe ZIP extraction. Defect: the report bundle archives every file under `OUTPUT_DIR`, including earlier runs (m7).
- **Promise fulfilment:** Capabilities A and B run on the default path; C is honestly labelled `FULL`. Section 25 promises caption perturbation that is not implemented (m3). The BYOD promise stops short of a visible result (M3).
- **Scientific validity:** the reranking and category comparisons are not like-for-like (M1, M2). Model differences of 0.003–0.008 R@1 on 391 images carry no dispersion estimate (S1).
- **Learner experience:** good scaffolding overall; several reported quantities are unexplained or unanswered (m4, m5, m6, m9).
- **Spec conformance:** unresolved `MUST`s: **DAT19** (BYOD messages, m1), **DAT9** (pretraining overlap, m2), **EVAL3** for ITM pair accuracy (m4), **REL1/REL12** for the optional paths (no `FULL` or BYOD hosted run). RUN8 partially (summary without results, m8).

## 3. Findings

### BVR-M1 — Major: the reranking table compares different query sets

- **Cell/section:** §18 "BLIP ITM reranks all three first-stage systems", cell `e5928ea5`; helper `rerank_with_blip` in cell `8477ecb0`; the same table in BYOD `reranking.csv` (cell `05f2bcd3`) and the `FULL` rows (cell `aaf0e9d5`).
- **Observed issue:** `coarse_t2i_r1` is `retrieval_metrics(...)["t2i_recall_at_1"]`, averaged over all 1,737 caption queries. `itm_t2i_recall_at_1` is computed only for the canonical first caption of each photograph (391 queries). The two sit side by side with no Δ and no statement that the query sets differ. The design specification (§36) asks for "Coarse canonical T2I R@1 | BLIP-reranked T2I R@1 | Δ". `pair_evaluations` and `seconds` are not split by direction.
- **Consequence:** the learner (and the repository's own release record: "BLIP ITM reranking of the top 5 raises these to … 0.731") reads the change in text→image R@1 as the reranker's effect. Part or all of it can be the change from 1,737 to 391 queries. Objective 8 and the reranking conclusion rest on this table.
- **Evidence:** direct execution (P01): with an *order-preserving* stand-in reranker, which cannot change any ranking, the notebook's row shows coarse T2I R@1 0.400 vs reranked 0.133 on a synthetic grid; the coarse value on the same 30 canonical queries is 0.133. Documented execution evidence (P13): saved table, SigLIP 2 `coarse_t2i_r1` 0.7214 (1,737 queries) beside `itm_t2i_recall_at_1` 0.7315 (391 queries).
- **Recommended correction:** compute the coarse baseline inside `rerank_with_blip` on exactly the queries it reranks (I2T: every photograph; T2I: canonical captions), return `coarse_i2t_r1`, `coarse_t2i_r1_canonical`, `delta_i2t_r1`, `delta_t2i_r1`, and use those in every reranking table. Rename the cell-`e5928ea5` all-caption column so it cannot be mistaken for the baseline, and add a "What to notice" note naming both query sets.
- **Acceptance check:** with an order-preserving stand-in reranker, `delta_i2t_r1 == 0` and `delta_t2i_r1 == 0`, and `coarse_t2i_r1_canonical` equals the argmax-over-images R@1 of the canonical captions; every reranking table (default, `FULL`, BYOD) carries the same-query coarse columns; the markdown before the table states the query set of each column.

### BVR-M2 — Major: category diagnostics compare sub-galleries of different sizes

- **Cell/section:** §21 "Category and caption-length diagnostics", cell `58332fc5`.
- **Observed issue:** each category (`no-text`, `text`) is scored in its own sub-gallery (`scores[np.ix_(image_indices,text_indices)]`): 160 vs 231 photographs in the hosted run. Recall@1 depends on gallery size, which §20 and §27 of the notebook teach. No chance reference per category and no prose. The caption-length table in the same cell ranks against the full gallery, so the two tables in one cell use different protocols.
- **Consequence:** the learner is invited to conclude that one category is harder (hosted run: BLIP ITC `no-text` I2T R@1 0.8125 vs `text` 0.7316) when part of the gap is that the `no-text` gallery is 30 % smaller. This is a confounded comparison presented as a category effect.
- **Evidence:** source inspection; direct execution (P02): identical-quality synthetic scores give T2I R@1 0.250 in a 160-image sub-gallery and 0.225 in a 231-image one; documented execution evidence of the category sizes.
- **Recommended correction:** keep the sub-gallery columns (labelled as such) but add full-gallery Recall@1 for each category's queries (ranked against all 391 photographs / 1,737 captions, as the length table does) and the sub-gallery chance R@1; add a "What to notice" note that only the full-gallery columns compare categories on equal terms.
- **Acceptance check:** the category table has `full_gallery_i2t_r1` and `full_gallery_t2i_r1` equal to the full-gallery per-query hits restricted to that category (test on synthetic scores); a sub-gallery chance column is present; the markdown directly above the cell states which columns are comparable across categories.

### BVR-M3 — Major: the BYOD branch never shows its results

- **Cell/section:** §26 BYOD contract (`07a272b2`) and the BYOD cell `05f2bcd3`.
- **Observed issue:** after a successful BYOD run the cell prints only `BYOD retrieval/index exports: <dir>`. Retrieval metrics, reranking results and adapter parity are written to `results.json` / CSV but never displayed. The markdown gives the ZIP schema but not how to place the ZIP in a hosted runtime or what to set; the empty-path error says "Set BYOD_ZIP_PATH for noninteractive execution", implying an interactive mode that does not exist.
- **Consequence:** the BYOD journey (DAT10, DAT15, objective transfer) ends without an interpretable result in the notebook; a learner has to find and open JSON files in the file browser. Completion/transfer is not delivered for the promised user-data path.
- **Evidence:** source inspection (P03); direct execution of the BYOD orchestration with stand-ins (existing `test_standard_actual_orchestration`) shows results only on disk.
- **Recommended correction:** after `run_byod_retrieval`, display the retrieval and reranking tables (and adapter parity in `FULL`) from the run's `results.json`; add step-by-step instructions (upload `dataset.zip` in the Colab Files pane or mount Drive, copy its path into `BYOD_ZIP_PATH`, tick `USE_BYOD`, re-run the controls cell and the BYOD cell); replace the empty-path message with one naming the field and the steps.
- **Acceptance check:** executing the BYOD cell with stand-in models and a valid two-image archive displays a retrieval table with three model rows and a reranking table with three candidate-source rows from that run's directory; `USE_BYOD=True` with an empty path raises an error that names `BYOD_ZIP_PATH` and the upload step; the BYOD markdown contains the placement steps.

### BVR-m1 — Minor (DAT19 MUST): BYOD validation errors do not name the offending record

- **Cell/section:** `load_byod`, cell `05f2bcd3`.
- **Observed issue:** messages such as "Each image needs 1..5 captions", "Unknown split role", "Captions must be strings", "Missing/unsafe/duplicate image file", "Invalid/duplicate image ID" and "Every archive image must be declared" do not say which line, id, value or file failed. Two messages have missing spaces ("outside16..4096", "At most100 million").
- **Consequence:** a learner with a 100-row archive cannot find the bad row without bisecting. DAT19 requires the failed contract to be identified.
- **Evidence:** direct execution (P04): six crafted archives, all refused, none naming the offender.
- **Recommended correction:** prefix each per-row message with the JSONL line number and id, include the offending value (role, caption count, file name, duplicate id), list undeclared/missing files, fix the spacing.
- **Acceptance check:** each of the six P04 archives raises `ValueError` whose message contains the offending id/value/file; the existing invalid-BYOD tests still raise.

### BVR-m2 — Minor (DAT9 MUST): no pretraining-overlap statement; corpus licence not stated

- **Cell/section:** §4 corpus (`472e7ac5`), §27 interpretation boundaries (`63ebfe7c`).
- **Observed issue:** VizWiz-Captions is a public dataset; overlap with the pretraining data of SigLIP/SigLIP 2 (WebLI) or BLIP cannot be ruled out, and the notebook does not say so. The corpus licence (CC BY 4.0, stated in the primary tutorial) is not stated here.
- **Consequence:** frozen zero-shot scores may be read as clean out-of-distribution evidence.
- **Evidence:** source inspection (P06).
- **Recommended correction:** add an overlap limitation and the licence in §4 and §27.
- **Acceptance check:** the notebook markdown contains a pretraining-overlap statement and "CC BY 4.0".

### BVR-m3 — Minor: "Caption perturbation" is promised but not run

- **Cell/section:** §25 heading (`73e42590`), cell `797b6acd`.
- **Observed issue:** the heading reads "Caption perturbation and no-match behavior"; only no-match is implemented. The design specification §68 defines a validation-only perturbation (original / lowercase / shortened) and a `validation/caption_perturbation.csv` output.
- **Consequence:** a promised activity is absent; the learner is not shown that rankings depend on how a query is phrased.
- **Evidence:** source inspection (P05).
- **Recommended correction:** implement the bounded validation-only comparison for SigLIP 2 (original, lowercase, first five words), write `outputs/validation/caption_perturbation.csv`, and explain that the SigLIP wrapper already lowercases (so lowercase is expected to change nothing).
- **Acceptance check:** the cell computes validation T2I R@1 for the three caption variants from the notebook's own `retrieval_metrics`, writes `caption_perturbation.csv`, touches no test data, and the markdown explains the expected lowercase result.

### BVR-m4 — Minor (EVAL3): ITM pair accuracy and cost columns are unexplained

- **Cell/section:** reranking tables (`8477ecb0`, `e5928ea5`); glossary (`a46a54b6`).
- **Observed issue:** `itm_pair_accuracy`, `pair_evaluations` and `seconds` appear in the principal table with no definition in prose or glossary.
- **Consequence:** objective 10 (reranking cost) and the hard-negative lesson cannot be read from the table.
- **Evidence:** source inspection (P07).
- **Recommended correction:** define ITM pair accuracy (canonical caption vs the coarse retriever's highest-scoring wrong caption, chance 0.5) and the cost columns in the §18 note and the glossary.
- **Acceptance check:** markdown and glossary define "ITM pair accuracy" (including its chance level) and `pair_evaluations`.

### BVR-m5 — Minor (GDL9): exercises have no worked answers; heading typo

- **Cell/section:** §28 "Try it yourselfs" (`3b012ccb`).
- **Observed issue:** six exercises, no collapsible sample answers; heading typo.
- **Consequence:** a self-paced learner cannot check an answer (e.g. exercise B's ceiling).
- **Evidence:** source inspection (P08).
- **Recommended correction:** add a collapsible sample answer per exercise; fix the heading.
- **Acceptance check:** each exercise A–F is followed by a `<details>` sample answer; heading reads "Try it yourself".

### BVR-m6 — Minor (GDL8): gallery-size section has no guidance

- **Cell/section:** §20 (`5333fe70`, `6ae280da`).
- **Observed issue:** a heading only. In the hosted run the model ordering flips with gallery size (70 images: SigLIP v1 I2T R@1 0.943 > SigLIP 2 0.900; 391: SigLIP 2 0.790 > v1 0.783), and no chance reference per size is shown.
- **Consequence:** objective 6 is exercised only through exercise A; the flip, the most instructive result, is unremarked.
- **Evidence:** source inspection and documented execution evidence (P09).
- **Recommended correction:** add a question/prediction and "What to notice" note, and a chance T2I R@1 column per size.
- **Acceptance check:** §20 markdown has a prediction prompt and a note on reading the table; the table has a chance column equal to `1/n_images`.

### BVR-m7 — Minor: the report bundle can include files from earlier runs

- **Cell/section:** §29 export (`005e710a`).
- **Observed issue:** `shutil.make_archive` zips the whole `OUTPUT_DIR`. After a `FULL` or BYOD run, a later `STANDARD` Run all in the same runtime bundles the earlier adapter and BYOD runs beside a manifest that says `STANDARD`.
- **Consequence:** exports do not provably belong to the run their manifest describes.
- **Evidence:** source inspection (P10).
- **Recommended correction:** record a session start time in the controls cell (kept across re-runs) and bundle only files written since then; list excluded files.
- **Acceptance check:** a file older than the session start is excluded from the bundle and reported; files written in the session are included.

### BVR-m8 — Minor (RUN8): the completion summary reports no results

- **Cell/section:** completion summary (`76b36372`).
- **Observed issue:** lists spec, tier, counts and paths, but no metric, whether adaptation ran, or the BYOD run directory.
- **Evidence:** source inspection (P11).
- **Recommended correction:** add best coarse and reranked R@1 per direction, `FULL` adaptation status and BYOD run directory.
- **Acceptance check:** the summary includes per-model test I2T/T2I R@1, reranked R@1 with Δ, adaptation status and BYOD status.

### BVR-m9 — Minor (UX1): objectives 11–12 are `FULL`-only but not marked

- **Cell/section:** §0 objectives (`e4c7f831`).
- **Observed issue:** adaptation and adapter export only execute in `FULL`; the default learner cannot meet them, and the list does not say so.
- **Evidence:** source inspection (P12).
- **Recommended correction:** mark objectives 11 and the adapter part of 12 "(`FULL` tier)".
- **Acceptance check:** objectives 11–12 state the tier that exercises them.

### Suggestions (optional, not release requirements)

- **BVR-S1:** report bootstrap intervals (over photographs) for the test Recall@1 differences; SigLIP 2 vs v1 differ by 3 photographs in I2T.
- **BVR-S2:** add the validation `RERANK_TOP_K` sweep (`rerank_k_sweep.csv`) and median-rank table named in the design specification's output tree.
- **BVR-S3:** label disagreement examples "no advantage" when the selected rank difference is ≤ 0.
- **BVR-S4:** run the no-match queries against the validation gallery and also show their BLIP ITM scores (design specification §69).

## 4. Positive findings and non-findings

- Freeze-before-test is real: validation selects the `FULL` epoch, the freeze record is written before any test embedding is computed.
- Index reload verifies digest, shape, finiteness and full rankings in both axes; the hosted run printed PASS.
- The candidate-oracle activity (guided-01) uses saved validation scores only and cannot disturb the canonical path.
- ITM output cardinality/range is checked before use; ZIP extraction rejects traversal, symlinks and duplicates.
- Not a finding: `RERANK_TOP_K` is not a form field; the notebook calls it the fixed canonical value and the troubleshooting table explains changing it needs a new validation/freeze experiment.

## 5. Promise-to-evidence matrix

| Claim | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Compare three frozen dual encoders on one gallery | `41f88c25`, `035b1b5a`, `624390ea`, `bc902f7e` | Hosted table, 391/1,737 | §17 note; baselines first | Delivered |
| BLIP ITM reranks each shortlist | `8477ecb0`, `e5928ea5` | Hosted rerank table | Coarse vs reranked T2I not like-for-like | **M1** |
| Reranker cannot recover omitted candidates | `candidate_oracle` | Oracle columns | §27, exercise B | Delivered |
| Gallery size changes difficulty | `6ae280da` | Hosted table (ordering flips) | No note | m6 |
| Category behaviour | `58332fc5` | Sub-gallery table | Confounded | **M2** |
| Caption perturbation | none | none | heading only | m3 |
| `FULL` adaptation + fresh reload | `1ae95e90`, `fa83aad2`, `aaf0e9d5` | Not run on hosted runtime | — | Not verified |
| Export and reload indexes | `1705c558` | PASS in hosted run | — | Delivered |
| BYOD runs the same contracts | `05f2bcd3` | Files on disk only | Path printed | **M3**, m1 |

| Objective | Learner activity | Evidence exercised |
|---|---|---|
| 1–2 retrieval vs classification; dual vs cross encoder | Opening diagram, glossary, exercise C | Reading only |
| 3–5 index, Recall@K, median rank, rsum | Tables §9, §17 | Reading tables |
| 6 gallery size | §20 table, exercise A | Table without guidance (m6) |
| 7 candidate oracle | guided-01 scratch-cell activity | Predict → change K → observe |
| 8 ITM reranking | §18 table | Misleading comparison (M1) |
| 9 hard negatives / disagreements | §22 gallery | Figures shown |
| 10 storage/latency/rerank cost | §23 table; rerank `seconds` | Cost columns undefined (m4) |
| 11–12 adaptation, adapter export | `FULL` only | Not exercised by default (m9) |

## 6. Readiness

**Needs revision.** Open Majors BVR-M1, M2, M3; unresolved `MUST`s DAT19, DAT9, EVAL3 (pair accuracy). After fixes, readiness becomes **Verification pending** until a hosted T4 run of the fixed notebook covers: `STANDARD` Run all; `FULL` Run all (adaptation, fresh reload, adapted reranking); BYOD with a valid archive plus one rejected archive (REL12). Status stays Candidate; promotion is a human decision.

## 7. Verified versus inferred

- Verified by direct execution: M1 mechanism (P01), M2 mechanism (P02), m1 messages (P04), positive controls (P14).
- Verified from documented execution: hosted `STANDARD` numbers quoted above.
- Inferred from source: M3 learner consequence, m3–m9.
- Not verified: `FULL` tier and BYOD on a real runtime; any real-model numbers after fixes.
- Finding most likely to be wrong: **BVR-M2's severity.** The table does print `n_images`, and a careful learner could discount the size difference; if the learner is expected to cross-check sizes, this is Minor.
