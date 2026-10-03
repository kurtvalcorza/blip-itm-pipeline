# Release verification

`tutorials/blip_itm_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned
  VizWiz-Captions dataset revision is the one other 40-hex string allowed);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `BlipItmPipeline.from_pretrained(weights_dir=...)`, `fetch_annotations` and `fetch_images` from the pinned
  cache path, `build_sample_dataset(seed=SPLIT_SEED, image_paths=...)` / `load_byod_dataset`, `validate_dataset` per
  split, `check_split_disjoint`, `gallery`, `write_dataset_jsonl`, the ceiling print, `validate_inputs` with the
  duplicate-caption refusal probe, `pipe.score` with the sanity checks and the per-grid `evaluation_report` on the
  drawn scenes, `chance_baseline`, `colour_keyword_baseline`, `pipe.evaluate` on the frozen model with
  `rerank_top_k` and on the validation and test splits after adaptation with the rsum assertions, the per-category
  breakdown, `pipe.adapt` with its explicit hyperparameters, `evaluation_report` on the grid after adaptation,
  `pipe.save_artifact`, `BlipItmPipeline.from_artifact` and the reload-parity assertion, and the provenance fields
  `weight_format`, `weight_sha256`, `hosted_tf_weight_file_not_loaded` and the `corpus` block with its gallery row
  group), the six expected `outputs/` paths, the learner-facing statements (BSD-3-Clause weights, neither score
  calibrated nor abstaining, adaptation with matching captions, the CC BY 4.0 corpus, the first two row groups, the
  test gallery, the two non-neural baselines, recall@1/5/10, ITM pair accuracy, no dispersion estimate, the OCR
  exclusion, the weight-format note) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any
  `git clone` / `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `from transformers import` / `BlipForImageTextRetrieval` / `BlipProcessor` / `text_encoder(` / `itm_head(` /
  `from huggingface_hub import` / `get_hf_file_metadata` / `urllib.request` / `pyarrow` / `safetensors` /
  `torch.optim` / `.backward(` / `pipe._model` / `extractall(` use **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy`, `pillow`,
`huggingface-hub` and `pyarrow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the
offline unit suite (`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected runner, annotation and image fetchers,
tiny PIL drawings, temporary manifests, no weights — `tests/test_model_backed.py` is skipped without the snapshot).
These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/blip-itm-base-coco/` or the corpus cache `weights/vizwiz-captions/` (the standalone
   path writes the manifest itself, stages the missing files from the Hub, reads the pinned VizWiz-Captions text
   columns and the pinned row group of photographs from the Hub, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `RERANK_TOP_K = 5`, `EPOCHS = 4`, `LEARNING_RATE = 2e-5`,
   `BATCH_SIZE = 16`, `TRAINABLE_TEXT_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`,
   `pillow==11.3.0`, `huggingface-hub==0.36.2`, `pyarrow==25.0.1` (an interpreter restart after the install is
   expected where the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `BlipItmPipeline`, `verify_snapshot`, `stage_missing_files`,
     `validate_inputs`, `evaluation_report`, `format_texts`, `recall_at_1`, `retrieval_metrics`, `gallery`,
     `itm_pair_accuracy`, `chance_baseline`, `colour_keyword_baseline`, `fetch_annotations`, `fetch_images`,
     `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `split_dataset`, `load_byod_dataset`,
     `write_dataset_jsonl`, `query_caption` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 8 manifest entries fetched from `Salesforce/blip-itm-base-coco` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (8 files, the 895 MB pickle re-hashed
     before `torch` is imported), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory
     with `source` `local-snapshot`;
   - Section 4: `fetch_annotations` checking the shard's declared size and SHA-256 against the pins, reading only
     its four text columns (1,550 rows) and matching the pinned text digest `9799ebb1…`; `fetch_images` reading the
     672 photographs of row groups 0 and 1 with every size and SHA-256 matching; the seeded split of the 318 captioned
     row-group-0 photographs into 208 / 40 / 70 plus row group 1's 321 in the test split (391 records, 1,737 gallery
     captions) with `check_split_disjoint` reporting no shared image, the category mix printed and the three
     dataset digests; `outputs/…_train.jsonl` written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `IMAGE_SIZE` 384, `MAX_IMAGES` 16,
     `MAX_TEXTS` 16, `MAX_TEXT_CHARS` 256, `MIN_RECORDS` 8, `MAX_RECORDS` 5000, `MIN_CAPTIONS` 1,
     `MAX_CAPTION_CHARS` 256) surfaced; the three scenes drawn; `validate_inputs` writing
     `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the duplicate-caption
     probe); `pipe.score` on the 3×3 grid with every sanity check `True` and the per-grid `evaluation_report`
     verdict `sample-sanity` (recall@1 1.0 both ways in the card-pass smoke; a different ranking on another runtime is a
     finding to record, not a failure);
   - Section 6: the chance baseline (rsum ≈ 0.08), the colour-keyword baseline (≈ 0.13) and the frozen model's
     gallery score (i2t R@1 ≈ 0.74, t2i R@1 ≈ 0.65, rsum ≈ 5.02, ITM-reranked i2t ≈ 0.80, pair accuracy ≈ 0.68 in
     the build record on the RTX 5070 Ti) with the per-category breakdown, and the cell's assertion that the frozen
     rsum is above both baselines;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 19,298,818 trainable of 223,744,258 parameters,
     208 training photographs and 888 (photograph, caption) pairs, and a four-epoch history with validation rsum
     moving by hundredths on a saturated 40-photograph gallery (5.816 frozen → 5.855 → 5.886 → 5.861 → 5.861 in the
     build record; `best_epoch` 2);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison on the six recalls
     and rsum, the ITM re-ranking, the per-category breakdown and `outputs/…_evaluation_report.json` written (the
     cell asserts the adapted test rsum exceeds the frozen one — on the build record 5.158 versus 5.023: i2t R@1
     0.742 → 0.790, t2i R@1 0.651 → 0.672, ITM-reranked i2t 0.803 → 0.841, pair accuracy unchanged at 0.683);
   - Section 9: the 3×3 grid re-scored by the adapted model with the `sample-sanity` report,
     `outputs/…_scores.csv` written; `pipe.save_artifact` writing
     `outputs/…_adapter/{adapter.safetensors,manifest.json}` (58 tensors, 77,202,384 bytes) and
     `BlipItmPipeline.from_artifact` reloading it with 64/64 identical cosine scores on eight test photographs
     against their query captions (the cell asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the
     model identity and licence, the snapshot block (`weight_format`, `weight_sha256`,
     `hosted_tf_weight_file_not_loaded`), the `corpus` block, the inference-contract grids, the comparison, the
     artifact digest, the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `blip_itm_colab.ipynb` (`E2E`) | `1f24a1d` / `6f1e02a3` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-blip-itm` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 691 files, 1063 MB staged from the Hub into a clean cache; comparison {i2t_recall_at_1: {chance: 0.003, neighbour: 0.008, frozen: 0.742, adapted: 0.79}, i2t_recall_at_5: {chance: 0.013, neighbour: 0.026, frozen: 0.905, adapted: 0.939}, i2t_recall_at_10: {chance: 0.025, neighbour: 0.043, frozen: 0.944, adapted: 0.962}, t2i_recall_at_1: {chance: 0.003, neighbour: 0.008, frozen: 0.652, adapted: 0.672}, t2i_recall_at_5: {chance: 0.013, neighbour: 0.017, frozen: 0.863, adapted: 0.877}, t2i_recall_at_10: {chance: 0.026, neighbour: 0.031, frozen: 0.918, adapted: 0.918}, rsum: {chance: 0.082, neighbour: 0.132, frozen: 5.024, adapted: 5.157}, itm: {itm_i2t_recall_at_1: {frozen: 0.803, adapted: 0.841}, itm_t2i_recall_at_1: {frozen: 0.708, adapted: 0.701}, itm_pair_accuracy: {frozen: 0.683, adapted: 0.683}}, median_rank: {frozen: [1, 1], adapted: [1, 1]}, delta_vs_frozen: {i2t_recall_at_1: 0.049, i2t_recall_at_5: 0.033, i2t_recall_at_10: 0.018, t2i_recall_at_1: 0.02, t2i_recall_at_5: 0.014, t2i_recall_at_10: 0.001, rsum: 0.134, itm_i2t_recall_at_1: 0.038, itm_t2i_recall_at_1: -0.008, itm_pair_accuracy: 0}, by_category: {no-text: {n: 160, frozen: [0.812, 0.775], adapted: [0.812, 0.786]}, text: {n: 231, frozen: [0.732, 0.628], adapted: [0.801, 0.643]}}}; reload parity {identical_scores: 64, of: 64}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-blip-itm/v2/evidence/` in the workspace |
| `blip_itm_colab.ipynb` (`TASK-INFERENCE`, superseded) | `2ccdea7` / `90b0268a87f0` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-blip-itm` v1) | PASSED — 8/8 code cells, 265.2 s, 18 files, 896 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/blip_itm_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/blip_itm_colab.ipynb`). Wall times, when recorded, are the sum of per-cell times
reported by the executor and include installs and the model download; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-19 | `1f24a1d` / `6f1e02a3` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-blip-itm` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 596.5 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 691 files, 1063 MB staged from the Hub into a clean cache; comparison {i2t_recall_at_1: {chance: 0.003, neighbour: 0.008, frozen: 0.742, adapted: 0.79}, i2t_recall_at_5: {chance: 0.013, neighbour: 0.026, frozen: 0.905, adapted: 0.939}, i2t_recall_at_10: {chance: 0.025, neighbour: 0.043, frozen: 0.944, adapted: 0.962}, t2i_recall_at_1: {chance: 0.003, neighbour: 0.008, frozen: 0.652, adapted: 0.672}, t2i_recall_at_5: {chance: 0.013, neighbour: 0.017, frozen: 0.863, adapted: 0.877}, t2i_recall_at_10: {chance: 0.026, neighbour: 0.031, frozen: 0.918, adapted: 0.918}, rsum: {chance: 0.082, neighbour: 0.132, frozen: 5.024, adapted: 5.157}, itm: {itm_i2t_recall_at_1: {frozen: 0.803, adapted: 0.841}, itm_t2i_recall_at_1: {frozen: 0.708, adapted: 0.701}, itm_pair_accuracy: {frozen: 0.683, adapted: 0.683}}, median_rank: {frozen: [1, 1], adapted: [1, 1]}, delta_vs_frozen: {i2t_recall_at_1: 0.049, i2t_recall_at_5: 0.033, i2t_recall_at_10: 0.018, t2i_recall_at_1: 0.02, t2i_recall_at_5: 0.014, t2i_recall_at_10: 0.001, rsum: 0.134, itm_i2t_recall_at_1: 0.038, itm_t2i_recall_at_1: -0.008, itm_pair_accuracy: 0}, by_category: {no-text: {n: 160, frozen: [0.812, 0.775], adapted: [0.812, 0.786]}, text: {n: 231, frozen: [0.732, 0.628], adapted: [0.801, 0.643]}}}; reload parity {identical_scores: 64, of: 64}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-blip-itm/v2/evidence/` in the workspace |
| 2026-09-19 | generated at `e38e7a8` / blob `567665a3046a` | Local Windows-venv harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6, snapshot, annotation cache and the 672 photographs pre-staged | Default sample path, all 11 code cells: pinned install skipped (pre-installed), `stage_missing_files` reported nothing to fetch, `verify_snapshot` PASS (8 files), annotations read from the pre-staged cache and 672 photographs re-hashed, split 208 / 40 / 391 by image (gallery 1,737 captions), the 3×3 grid scored (recall@1 1.0 both ways), baselines rsum 0.082 / 0.132, frozen gallery rsum 5.024 in 236.4 s (i2t R@1 0.742, t2i 0.652, ITM-reranked i2t 0.803, pair accuracy 0.683; `text` 0.732 / 0.628, `no-text` 0.812 / 0.775), four epochs 561.1 s over 888 pairs (validation rsum 5.816 → 5.855 → 5.886 → 5.861 → 5.861, epoch 2 kept), adapted gallery rsum 5.162 (i2t R@1 0.791, t2i 0.673, ITM-reranked i2t 0.844 / t2i 0.703, pair accuracy 0.688; `text` 0.797 / 0.644, `no-text` 0.812 / 0.789), grid unchanged after adaptation, adapter 77,202,384 B / 58 tensors, reload parity 64/64, 6 outputs written; the committed blob differs from the executed one in markdown prose only (CPU figures filled in after this run) | 1512.3 s | PASS — pre-flight only; not promotion evidence |

## Current status

**Release-grade.** The `E2E` notebook blob `6f1e02a3` (committed at `1f24a1d`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 596.5 s, 691 files, 1063 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

Facts a reviewer should still weigh: the frozen model is already a competent retriever on VizWiz photographs — at the ceiling of a 70-photograph gallery (rsum 5.72 of 6), which is why the test gallery is 391 photographs — so the adaptation gain is a few recall points carried by the image → text direction (rsum 5.024 → 5.157 on the T4 run, i2t R@1 0.742 → 0.790, on the `text` photographs 0.732 → 0.801) while the ITM pair accuracy does not move (0.683); the T4 run reproduced the CPU pre-flight within 0.005 rsum (same seed; the hard-negative sampling is seeded too); the 40-photograph validation split saturates and selects the epoch weakly; four text-encoder blocks at 5e-5 for six epochs gained no more than the default in the build record; and the drawn grid re-scored after adaptation is one 3×3 grid of evidence about behaviour outside the corpus, not a measurement.


## Shared retrieval supplemental remediation evidence — 2026-09-26

Applies only to `DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb`, identical executable cells in both repository entry points. Candidate remains; no real-model or hosted execution was performed here. Baseline BLIP:6 workshop+5 primary parity tests and validator passed. Baseline SigLIP2:5 primary parity tests passed; validator rejected the extra named supplemental notebook. The SigLIP2 validator now explicitly checks the named supplemental profile, standalone/candidate metadata, source syntax, saved errors and carried runtime policy while preserving strict primary parity and refusing unrelated extra notebooks.

Confirmed gaps repaired: BYOD was loader-only; archive extraction replaced a shared directory; caption inputs coerced arbitrary values and lacked bounds; ITM output zip truncation hid missing results; index parity covered only a subset without prior digest verification; FULL reload reported finite outputs without comparison to live adapted weights. The shared notebook now runs bounded optional retrieval/reranking/index exports and FULL train→validation→freeze→fresh-reload→test, with live-to-fresh validation embedding checks. NumPy retains a loaded2.x ABI rather than replacing observed Colab2.1.3 with2.5.3; fallback is2.1.3. Incompatible loaded packages retain an explicit restart guard.

Lightweight tests execute the actual loader, evaluator, reranker, index verifier, adapter export/reload and STANDARD/FULL orchestration with tiny fixtures/model doubles. They establish control flow and failure boundaries, not GPU fit, model quality or real safetensors serialization. A fake adapter that loses learned values is rejected by the actual parity comparison. Model identities and data pins are unchanged.

Pending: fresh STANDARD and FULL T4 runs with exact commit/blob, complete saved outputs, versions, restart count and wall/VRAM; valid STANDARD BYOD and FULL split BYOD through real models and adapter/index exports; invalid-caption/duplicate-image runs rejected before optional model loading. Preserve each `outputs/byod/run-*` result independently. These are open hosted/REL12 evidence gates, not presumed complete from local tests.

Local remediation checks:17 focused tests passed, including STANDARD/FULL orchestration, adapter-loss rejection and deterministic cross-image batch completion for small FULL datasets. BLIP combined suite28 passed (17 focused+6 workshop+5 primary parity). Safetensors IO and model weights are doubled in focused tests; no real model execution is claimed.


### Maintainer-supplied successful Colab run — 2026-09-26

The maintainer supplied the [executed notebook](execution-evidence/2026-09-26/DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb) and authorized merging PR #7 (merge commit `380be23`). The file is archived byte-for-byte, SHA-256 `b76acfad5db8d02119b412160164e61bdbb0722c6d09fb053f3039cd0dfccd42`. All 33 code cells have execution counts, 59 saved outputs and zero saved errors. Code-cell sources match commit `89eea028600220148b437417fb05b98606682807`, tutorial blob `3cdeee63722219eb9a06da8e81cf2985acb03850`, apart from Colab-inserted `# @title` lines. One code cell differs from the committed blob only by trailing whitespace that Colab strips. Later commits on `main` that touch the notebook (`f474a75` (AI Use Disclosure)) change only markdown cells; its code cells are identical to the executed revision. This evidence commit does not change tutorial code.

Scope: STANDARD tier: pinned VizWiz-derived parquet shard, test gallery of 391 images and 1,737 captions; SigLIP v1, SigLIP 2 and BLIP ITC embeddings with BLIP ITM reranking of the top 5. FULL adaptation and BYOD were not exercised. The same executed notebook (blob `3cdeee63`) also covers `siglip2-vision-language-pipeline` PR #13.

Saved runtime: Python 3.13.15, torch 2.14.0+cu130, Transformers 4.57.6, pyarrow 25.0.1, CUDA Tesla T4. Execution reaches the final completion summary. The separate exported files were not supplied, so their bytes/digests were not independently inspected. Saved counts run sequentially from 1 to 33; runtime freshness and absence of manual restarts/reruns are not independently established by the artifact.

Results (sample-sanity measures on the built-in data, not general model rankings): Test recall@1 image→text / text→image: SigLIP 2 0.790 / 0.721, SigLIP v1 0.783 / 0.716, BLIP ITC 0.742 / 0.652; BLIP ITM reranking of the top 5 raises these to 0.839 / 0.731 (SigLIP 2 candidates), 0.834 / 0.734 (SigLIP v1) and 0.803 / 0.708 (BLIP ITC).

Status remains **Candidate**. Merge approval and this successful default-path run do not close the optional-path (FULL/BYOD) or REL12 qualification gates, and `metadata.dimer.clean_runtime_evidence` in the notebook stays `pending` as authored (editing it would change the verified blob).

## Notebook review fixes, revision 0.2.0-candidate — 2026-10-02

Applies only to `DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb`. A Notebook Review Framework v1 review of `a25fdf6` (`docs/reviews/2026-10-02-notebook-review/`) found 3 Major and 9 Minor issues; all 12 are fixed in this revision by anchor-asserting edits to the notebook (there is no in-repo generator), recorded in `metadata.dimer.review_revisions`. Cell ids and order are unchanged.

- **Reading the 2026-09-26 result above.** Its text→image comparison is not like-for-like: the coarse values (0.721 / 0.716 / 0.652) are over all 1,737 captions, the reranked values (0.731 / 0.734 / 0.708) over the 391 first captions. The image→text comparison uses the same 391 queries on both sides. Revision 0.2.0 reports coarse Recall@1 on exactly the reranked queries (`coarse_i2t_r1`, `coarse_t2i_r1_canonical`) and the deltas.
- Other changes: category diagnostics add full-gallery columns; the BYOD branch displays its retrieval and reranking tables and names the failing `records.jsonl` line/id/field; caption perturbation (validation only) is implemented; the report bundle holds only files written since the current Run all started; licence and pretraining-overlap statements, sample answers, glossary entries and a results summary are added.
- Verification so far: CPU-only unit tests that execute the notebook's own cells with NumPy data and stand-in models (`tests/test_bvr_review_fixes.py`, 21 tests, all failing on `a25fdf6`). This is **not** clean-runtime evidence. The 2026-09-26 run covers the previous code cells only.
- Required before release: a fresh Colab T4 `STANDARD` Run all of this revision; a `FULL` Run all (adaptation, fresh reload, adapted reranking); BYOD with a valid archive and one rejected archive (REL12). Status remains **Candidate**.

### Colab CLI execution of revision 0.2.0-candidate (`04664ca`, blob `571cb2f2`) — 2026-10-03

- **File:** [`execution-evidence/2026-10-03/DIMER_MultiModel_Vision_Language_Retrieval_Workshop_04664ca_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_MultiModel_Vision_Language_Retrieval_Workshop_04664ca_colab-cli-t4.ipynb), SHA-256 `88230408683946ed2b4cbac68f8c8713547feb777b07ddabb4253ece5131ac89`, a byte-for-byte copy of the CLI's output notebook. The same notebook bytes are the PR head of both `blip-itm-pipeline` #9 (`04664ca`) and `siglip2-vision-language-pipeline` #15 (`13040f4`), so this one run covers both.
- **Executor:** Google Colab CLI 0.7.4 on a fresh Colab Tesla T4 session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`). The notebook was downloaded from GitHub at the PR head and its git blob `571cb2f2d3dc` checked before the session was created. Code cells ran in order in one kernel; this is not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/33` log.
- **Result:** **PASSED** (STANDARD, default controls): 33/33 code cells, no error output, 589.1 s wall; the completion summary reports `full_adaptation: not run (STANDARD)`, `byod: not run`, and "Vision-language retrieval workshop complete".
- **Results:** coarse test recall@1 image→text / text→image over all captions equal the 2026-09-26 run exactly: SigLIP 2 0.790 / 0.721, SigLIP v1 0.783 / 0.716, BLIP ITC 0.742 / 0.652. BLIP ITM reranking of the top 5 on the 391 canonical queries also equals it (same model order: SigLIP 2 / SigLIP v1 / BLIP ITC candidates): image→text 0.839 / 0.834 / 0.803 and text→image 0.731 / 0.734 / 0.708. New in 0.2.0, the like-for-like coarse text→image recall@1 on the same 391 queries is 0.688 / 0.662 / 0.627, so reranking gains +0.044 / +0.072 / +0.082.
- **Observation:** cell "Fetch and verify the full pinned parquet" printed `Error while fetching HF_TOKEN secret value from your vault ... timed out`. Under the CLI the Colab secrets prompt cannot be answered; `huggingface_hub` then downloaded anonymously and the pinned parquet verified. This is cosmetic, CLI-specific, and not a failure.
- **Boundary:** saved outputs were inspected. FULL adaptation, BYOD and the rejected-archive path (REL12) were not exercised and remain required before release. Status remains **Candidate**.


## 2026-10-03 uv isolated environment (revision 0.3.0-candidate)

Applies only to `DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb`: notebook blob `571cb2f2d3dcd58d04afc1dd273742a68264c9be` (revision 0.2.0-candidate, the blob the 2026-10-03 Colab CLI run above covers) → `954cf7bc208ce10e5f41cf2d1beeec0ff5761bd1` (revision 0.3.0-candidate).

- **Change.** The in-kernel `pip install` and its "Restart session, then Run all" guard are removed. The notebook carries `tools/retrieval_workshop.py` (every model stage; the former cell functions moved unchanged) and `tools/retrieval-workshop-requirements.lock` (`uv pip compile --generate-hashes` for CPython 3.12 on `x86_64-manylinux_2_28`) in carrier cell `uvcarrier`, written by `tools/build_retrieval_workshop_carrier.py` in pieces of at most 1,000 characters (longest notebook line well under 2,000). The runtime cell verifies a pinned uv 0.12.15 wheel by size and SHA-256, builds a CPython 3.12.12 venv (`--managed-python`) and installs with `--require-hashes --only-binary :all:`; every model stage runs as a separate process with that venv's interpreter, `MPLBACKEND=Agg`, and `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`/HF tokens dropped. Display cells read the stage JSON/CSV/PNG outputs. A warm runtime reuses the venv when its recorded lock hash matches.
- **Pins.** Same direct versions as before (torch 2.14.0, torchvision 0.29.0, torchaudio 2.11.0, transformers 4.57.6, huggingface-hub 0.36.2, safetensors 0.8.0, numpy 2.1.3, pillow 11.3.0, pyarrow 25.0.1); the former ranges resolve to pandas 2.2.3 and matplotlib 3.10.8; sentencepiece 0.2.1 and protobuf 6.33.5 are added (the hosted kernel used to supply them for the SigLIP v1 tokenizer).
- **User-visible.** Linux x86_64 only (Google Colab, Kaggle, Linux Jupyter); Windows/macOS kernels are refused. The experiment manifest gains an `environment` record. No metric, split, seed, model or output file changes.
- **Local verification (not clean-runtime evidence).** A CPU harness on Windows executed every code cell in order with real model weights and the real pinned corpus; stand-ins: the platform check was patched, and this host's interpreter (torch 2.9.1, not the lock) replaced the uv venv. STANDARD reproduced the 2026-10-03 Colab values exactly: role digest `3cb7cbd2…`, test R@1 i2t/t2i SigLIP 2 0.790/0.721, SigLIP v1 0.783/0.716, BLIP ITC 0.742/0.652, reranked I2T 0.839/0.834/0.803 and T2I 0.731/0.734/0.708 and the 27-file bundle (ITM pair accuracy SigLIP 2/v1/BLIP 0.780/0.767/0.683, the same three values as the archived 0.2.0 Colab output, which shows 0.7801/0.7673/0.6829). A second Run all in the same work directory with `FULL` (BLIP_EPOCHS reduced to 1 for CPU time) and a FULL BYOD archive passed adaptation, fresh-reload parity, adapted reranking and BYOD; a rejected archive stopped with `records.jsonl line 6 (id='img5'): split 'holdout' …`. `pytest` 89 passed / 1 skipped before, 103 passed / 1 skipped after.
- **Pending.** The hosted Colab T4 re-run of this blob passed on 2026-10-03 (see "Colab CLI execution of revision 0.3.0-candidate" below, the first hosted run of the uv bootstrap and the hash-locked install); FULL, BYOD and REL12 remain pending as before. Status remains **Candidate**.

### Colab CLI execution of revision 0.3.0-candidate (`3d7dbc3`, blob `954cf7bc`) — 2026-10-03

- **File:** [`execution-evidence/2026-10-03/DIMER_MultiModel_Vision_Language_Retrieval_Workshop_3d7dbc3_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_MultiModel_Vision_Language_Retrieval_Workshop_3d7dbc3_colab-cli-t4.ipynb), SHA-256 `29db8b37b2dde817fc471d14bfac1978baa332db9fae650a5759107dfd3ff466`, a byte-for-byte copy of the CLI's output notebook. The same notebook bytes are the PR head of both `blip-itm-pipeline` #9 (`3d7dbc3`) and `siglip2-vision-language-pipeline` #15 (`4542d3a`), so this one run is recorded in both repositories.
- **Executor:** Google Colab CLI 0.7.4 on a fresh Colab Tesla T4 session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`). The notebook was downloaded from GitHub at the PR head and its git blob `954cf7bc208c` checked before the session was created. Code cells ran in order in one kernel; this is not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/34` log.
- **Result:** **PASSED** (STANDARD, default controls): 34/34 code cells, no error output, 536.8 s wall for the whole session (589.1 s for the 0.2.0 run above); the completion summary reports `full_adaptation: not run (STANDARD)`, `byod: not run`, `release_status: candidate`, and "Vision-language retrieval workshop complete".
- **uv bootstrap (first hosted run).** The runtime cell built the isolated environment and installed the hash lock without error, then reported `{'python': '3.12.12', 'torch': '2.14.0+cu130', 'device': 'cuda:0', 'gpu': 'Tesla T4', 'lock_pins_checked': 61}`; the carrier cell verified `requirements.lock.txt` and `retrieval_workshop.py`. No restart was needed. The CLI records no per-cell times, so setup time on its own is not measured; it is contained in the 536.8 s total, which is 52.3 s shorter than the 0.2.0 run with its in-kernel `pip install`.
- **Results vs the 0.2.0 run above (same 391-image / 1,737-caption test gallery):** every metric is equal. Parquet SHA-256 `4492465a…` and role digest `3cb7cbd2…` are unchanged. Coarse test recall@1 image→text / text→image over all captions: SigLIP 2 0.790 / 0.721, SigLIP v1 0.783 / 0.716, BLIP ITC 0.742 / 0.652. BLIP ITM reranking of the top 5 on the 391 canonical queries (SigLIP 2 / SigLIP v1 / BLIP ITC candidates): image→text 0.839 / 0.834 / 0.803, like-for-like coarse text→image 0.688 / 0.662 / 0.627 → reranked 0.731 / 0.734 / 0.708 (+0.044 / +0.072 / +0.082), ITM pair accuracy 0.780 / 0.767 / 0.683, 4,692 pair evaluations each. The validation reranking table (40 queries), the nested-gallery table, the caption-perturbation table and the no-match nearest-neighbour scores are also unchanged. Timing columns differ, as expected (reranking 35.6–36.8 s per source vs 42.0–43.6 s). The 27-file report bundle has a different SHA-256 (`b6500aac…` vs `eef23d5e…`); its contents include the new `environment` record and run-specific timings, so a different digest is expected and is not a result difference.
- **Observation:** unlike the 0.2.0 run, no `HF_TOKEN secret ... timed out` line appears (inferred cause: the parquet fetch and model stages now run as separate processes in the uv venv with HF tokens dropped, so the kernel never queries the Colab secrets vault).
- **Boundary:** saved outputs were inspected. FULL adaptation, BYOD and the rejected-archive path (REL12) were not exercised and remain required before release. Status remains **Candidate**; `metadata.dimer.clean_runtime_evidence` is unchanged.
