# BLIP ITM-base Image-Text Retrieval E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/blip-itm-pipeline`  
**Notebook:** `tutorials/blip_itm_colab.ipynb`  
**Reviewed commit:** `a25fdf6d2e1c8b5109300185ad21dac2e57c9723` (`main`, confirmed with `gh api repos/kurtvalcorza/blip-itm-pipeline/commits/main`)  
**Notebook Git blob:** `6f1e02a3380e36cd2452e25470287349973cfac9`. This is the blob committed at `07fc3a6` (`1f24a1d` on the PR branch) and executed in the recorded Kaggle run of 2026-09-19. Later commits on `main` did not touch the notebook, the carried modules or the generator; generator `--check` exits 0 at the reviewed commit.  
**Finding prefix:** `ITM`

## Executive assessment

The default path is careful engineering, and it reproduces. The notebook carries its three modules byte for byte (generator `--check` and `tools/validate_release_assets.py` both exit 0). It digest-verifies the pinned pickle before `torch.load(weights_only=True)`, reads a digest-pinned real captioning corpus column-only, splits it by image, and frames the model with two non-neural baselines and a full retrieval protocol (recall@1/5/10 both ways, median rank, rsum, ITM re-ranking, ITM pair accuracy, per category). It says plainly that neither score is calibrated and neither abstains, and it reloads its safetensors adapter with asserted parity.

A direct CPU execution of all 11 code cells, verbatim, with the real weights and the exact pins, reproduced the recorded numbers:

| Measure | Value |
|---|---|
| Baseline rsum (chance / colour-keyword) | 0.082 / 0.132 |
| Frozen model rsum | 5.024 |
| Validation rsum by epoch | 5.816 → 5.855 → 5.886 → 5.861 → 5.861 (epoch 2 kept) |
| Adapted rsum | 5.162 |
| i2t R@1 | 0.742 → 0.790 |
| Reload parity | 64/64 |

The figures match the CPU pre-flight row in `docs/release-verification.md` exactly. They match the Kaggle T4 row to 0.005 rsum.

Five problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (ITM-M1).** In a fresh hosted runtime, `Run all` does not finish in one pass. The recorded qualification run's first pass stopped in cell 3 with `RuntimeError: … cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime`. Even so, `README.md`, `STATUS.md`, `tutorials/README.md` and `docs/release-verification.md` record a Release-grade PASS.
2. **Result assertions crash honest outcomes (ITM-M2).** Sections 6 and 8 **assert** results: `assert adapted_test['rsum'] > frozen_test['rsum']`. When the epoch selector keeps epoch 0, the adapted model *is* the frozen model and the two rsums are equal. Any configuration whose fine-tuning does not raise validation rsum therefore ends in a bare `AssertionError`, before the adapter is exported or `result.json` is written. This includes the documented `RERANK_TOP_K = 10` rerun and BYOD corpora where adaptation does not help.
3. **Reruns reuse the adapted model (ITM-M3).** Rerunning Section 7 after a field change starts from the already adapted model. In P2 the notebook's own documented experiment (`TRAINABLE_TEXT_LAYERS = 4`, `LEARNING_RATE = 5e-5`) printed the previous adaptation as `'note': 'frozen model'` (validation rsum 5.886 instead of 5.816) and kept "epoch 0". Section 8 then reprinted the old comparison unchanged.
4. **BYOD contract does not match enforcement (ITM-M4).** The stated minimum is 8 records; the enforced minimum is **50**. A second upload silently trains on the first upload's records. Images in a zip subfolder cannot be referenced. A cancelled upload or a zip without a records file raises a bare `StopIteration`.
5. **Guided layer missing (ITM-M5).** The notebook is declared `GUIDED`, but most of the guided layer is absent. The 2,387 lines of carried code are not labelled or collapsed as infrastructure.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, and the opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** (2026-09-26), `ml-worker` `origin/main` `b1cfe13` |
| Intended audience | Not stated. The Prerequisites ask for: basic Python and PIL; dual-encoder versus fused-encoder scores; the risk of pickle checkpoints; recall@k, median rank and rsum and why a small gallery inflates them |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU float32, CUDA when present; ~896 MB model plus ~166 MB of photographs |
| Promised outcomes | One-pass `Run all` with no configuration edit; a pinned install; three carried modules; a digest-verified pickle snapshot; digest-pinned VizWiz-Captions text and 672 photographs; a 208/40/70 split by image plus a 391-photograph / 1,737-caption test gallery; four refusals; the inference contract on a drawn 3×3 grid; two baselines and the frozen model on the full retrieval protocol, per category; bounded fine-tuning of the text encoder's last two blocks, the ITC projections and the ITM head, with validation-rsum epoch selection; held-out evaluation; the grid re-scored; a safetensors adapter with reload parity; six `outputs/` entries; BYOD "through the same validation, … fine-tuning, held-out evaluation, artifact export and reload-parity cells"; three optional experiments |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; carried modules from `src/blip_itm_pipeline/` @ `e38e7a8` |

### Evidence actually obtained

- **Source inspection:** all 25 cells (11 code). From the carried `pipeline.py`: `adapt`, `evaluate`, `save_artifact` and `from_artifact`. From `samples.py`: `split_dataset`, `validate_dataset`, `load_byod_dataset` and `build_sample_dataset`. Also the generator and template, `README.md`, `STATUS.md`, `MODEL_CARD.md`, `tutorials/README.md` and `docs/release-verification.md`.
- **Documented execution evidence:**
  - Sources: `docs/release-verification.md` and the archived executor summary `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-blip-itm/v2/evidence/run_summary.json` (workspace).
  - The run: Kaggle Tesla T4, 2026-09-19, on **blob `6f1e02a3`, the reviewed blob**.
  - Attempt 1 failed in cell 3 with the restart `RuntimeError` after 209.3 s.
  - Attempt 2, after an interpreter restart, ran 11/11 cells in 387.2 s (`restarted_after_install_cell: true`).
  - No Colab run and no BYOD run of this blob are recorded.
  - The 2026-09-26 Colab evidence in `docs/execution-evidence/` covers the supplemental workshop notebook, not this one.
- **Direct execution (this review):**
  - **Environment:** `run_probes.py` on Windows, CPython 3.12, in an existing workspace venv that holds the notebook's exact pins with CPU torch: `torch 2.14.0+cpu`, `transformers 4.57.6`, `safetensors 0.8.0`, `numpy 2.5.3`, `pillow 11.3.0`, `huggingface-hub 0.36.2`, `pyarrow 25.0.1`. CPU only, `HF_HUB_OFFLINE=1`.
  - **Not a clean runtime:** the **real weights** and the pinned VizWiz cache (annotations and 672 photographs) were hard-linked into a scratch working directory. The notebook wrote its own manifest, `stage_missing_files` fetched nothing (`fetched: []`) and every file was re-hashed. Nothing was downloaded.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`.
  - **Execution method:** every code cell ran **verbatim from the notebook JSON** in one namespace. Only form-field literals were substituted, and `google.colab.files.upload` was replaced by a fake.
  - **Stand-in data:** BYOD images are **stand-ins** (flat coloured shapes drawn with Pillow, two captions each).
  - **Timing:** the default path took 1,240 s of cell time.
  - **Static checks:** JSON parse, compile of all 11 code cells, blob id, generator `--check` (exit 0), `tools/validate_release_assets.py` (exit 0).
- **Probes stopped for time:** P3 (a fresh base with an in-range learning rate that does not improve validation, through Section 8) and P5 (a stand-in BYOD set through Sections 5–9) were queued but stopped at the review's time cap. They are **not verified**. ITM-M2's crash claim and the BYOD downstream path therefore rest on source inspection, and on P2's direct observation that `best_epoch = 0` is reachable with a documented setting.
- **Not verified:** a Colab run of any kind; a one-pass hosted `Run all`; the CUDA path in this review; BYOD past Section 4; BYOD through the real upload widget or with real photographs; learner understanding.

## 2. Separate judgments

- **Technical correctness:** sound on the default path. Identity, digests, the image-disjoint split, train-only baselines, export and reload parity all hold, and direct execution matched the record. Defects:
  - the restart-dependent install (ITM-M1);
  - result assertions that stop the notebook on a legitimate outcome (ITM-M2);
  - state that survives a rerun and is mislabelled (ITM-M3);
  - the BYOD loader and contract (ITM-M4).
- **Scientific validity:**
  - **Sound:** the protocol, the baselines, the gallery-size argument, the per-category breakdown and the "a few recall points, carried by image → text" reading are honest and well supported.
  - **Minor caveat:** recipe comparisons are reported on the test gallery, and that gallery's size was chosen after the frozen model's test ceiling was observed. Both choices are disclosed, but nothing tells the learner that this makes the test delta less independent (ITM-m3). The 0.138 rsum gain has no dispersion estimate (stated by the notebook).
- **Promise fulfilment:**
  - **Delivered:** the default capability list.
  - **Not delivered:** one-pass `Run all` (ITM-M1).
  - **Not deliverable as written:** the optional experiments (ITM-M2, ITM-M3).
  - **BYOD:** delivered through Section 4 for one zip layout of 50+ records uploaded once. Not for the stated contract (ITM-M4); downstream not verified.
- **Learner experience:** dense, careful stage prose, two "Look for" notes, explicit limits and three optional experiments. There is no prediction, checkpoint, worked answer, troubleshooting section or conclusion template, and there are 2,387 lines of unlabelled infrastructure (ITM-M5).
- **Spec conformance (2.2):** these applicable `MUST`s fail:

  | Finding | Failed `MUST`s |
  |---|---|
  | ITM-M1 | RUN1, RUN10, ENV6, REL2, REL11 |
  | ITM-M2 | RUN9, UX7 |
  | ITM-M3 | UX7, OUT8, ART8 |
  | ITM-M4 | DAT12, DAT19, REL12 |
  | ITM-m2 | UX12 |

  GDL1–GDL15 are largely unmet (SHOULD). The declared spec is 2.0 (ITM-S1).

## 3. Promise and objective tracing

| Claim (where) | Implementation | Observable result (this review) | Learner interpretation |
|---|---|---|---|
| One-pass `Run all`, no intervention (cell 0) | Cell 3: in-kernel `pip install` + stale-import guard | Kaggle record: attempt 1 `RuntimeError` (numpy 2.0.2 → 2.5.3, cuda-bindings 12.9.4 → 13.4.2), restart, attempt 2 ok. Local: install skipped | **Not delivered** (ITM-M1) |
| Pinned, digest-verified pickle snapshot (cells 10–11) | `stage_missing_files`, `verify_snapshot`, `from_pretrained` | 8 files verified, `fetched: []` (pre-staged), `local-snapshot`, cpu | Delivered |
| Pinned VizWiz text + 672 photographs; 208/40/70 by image + gallery; four refusals (cells 12–13) | `fetch_annotations`, `fetch_images`, `build_sample_dataset`, `validate_dataset`, `check_split_disjoint` | 1,550 rows, 672 photographs, 208/40/391, 1,737 gallery captions; `text` 110/25/231; four probes rejected with the rule named | Delivered |
| Inference contract on the drawn 3×3 grid (cells 14–15) | `validate_inputs`, `score`, `evaluation_report` | ITM diagonal 0.998 / 0.634 / 0.278, equal to the card; duplicate-caption probe rejected; `sample-sanity` | Delivered, well explained |
| Baselines + frozen model, per category (cells 16–17) | `chance_baseline`, `colour_keyword_baseline`, `evaluate(rerank_top_k=5)` | rsum 0.082 / 0.132 / 5.024; i2t R@1 `no-text` 0.81, `text` 0.73; 275 s | Delivered |
| Bounded fine-tuning with validation epoch selection (cells 18–19) | `adapt(lr=2e-5, epochs=4, layers=2)` | 19,298,818 trainable, 888 pairs, val rsum 5.816 → 5.886, best epoch 2, 603 s | Delivered |
| Held-out evaluation (cells 20–21) | `evaluate` + `assert adapted rsum > frozen rsum` | rsum 5.024 → 5.162, i2t R@1 0.742 → 0.790, ITM pair accuracy 0.683 → 0.688 | Delivered on the default; **assertion gates non-improving runs** (ITM-M2) |
| Re-score the grid, export, reload parity (cells 22–23) | `save_artifact`, `from_artifact` | fruit ITM 0.278 → 0.350; 58 tensors, 77,202,384 B; parity 64/64; 6 outputs | Delivered |
| "set `TRAINABLE_TEXT_LAYERS = 4` and `LEARNING_RATE = 5e-5` … against your own run" (cell 24) | Rerun of cell 19 → 21 | Epoch 0 "frozen model" = previous adaptation (5.886); new epoch 5.849; best epoch 0; comparison unchanged | **Stale, mislabelled** (ITM-M3) |
| "set `RERANK_TOP_K = 10` and compare" (cell 24) | Rerun of cells 17 → 21 | Source: Section 6 rescored on the adapted `pipe` → "frozen" = adapted → equal rsum → `AssertionError` | **Crashes** (ITM-M2, ITM-M3; inferred) |
| BYOD "8..5,000 records", "re-run from that cell" (cells 0, 1, 13) | Cell 13: zip flatten + `load_byod_dataset` + `split_dataset` + per-split `validate_dataset` | 8–49 records fail in Section 4; 50 and 60 pass | Partly delivered (ITM-M3, ITM-M4) |

| Objective (cell 0) | Learner activity | Evidence it was exercised |
|---|---|---|
| Install; read what the carried modules guarantee; stage and digest-verify | Run cells | Procedural only |
| Read the captions/photographs of a pinned corpus, validate, split by image | Read printed digests and refusals | Shown, not practised |
| Score drawn scenes and read the two scores correctly | Read the grids | Shown; no check of the reading |
| Measure the frozen model's retrieval vs baselines, read per category | Read tables | Shown; no prediction or question |
| Bounded fine-tuning with validation selection; evaluate on a disjoint gallery | Run cells | Runs on the default (ITM-M2 gates other outcomes) |
| Re-score drawings; export and reload with parity | Run cell | Shown |
| (Optional) re-ranking depth, layers/learning rate, epochs, BYOD | Edit a field, no rerun instructions | Reruns are stale (ITM-M3) or crash (ITM-M2) |

## 4. Prioritized findings

### ITM-M1 — Major: fresh-runtime `Run all` needs a manual restart after the install cell, yet the release records call it a PASS

- **Cell/section:** Section 1 (cell 3); generator `tools/build_notebook.py:47-70` (`_INSTALL_GUARD`) and `:470-471`.
- **Observed issue:** cell 3 pip-installs 9 pins into the running kernel. When a pin replaces an imported distribution, it raises `RuntimeError(… 'Restart the runtime, then rerun from the top.')`.
  - The recorded qualification run of this exact blob did exactly that. `run_summary.json` attempt 1 failed with `cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`, and attempt 2 ran after a restart.
  - `README.md`, `STATUS.md`, `tutorials/README.md` and `docs/release-verification.md` record this as "PASSED — 11/11 code cells ok (1 restart after install cell)" and "Release-grade".
  - The release procedure (step 4) says the restart "is expected".
  - The opening cell promises a single `Run all` with "no configuration edit".
- **Consequence:** a learner's first `Run all` stops in cell 3. The release status rests on a two-pass run that the spec defines as non-conformant (RUN10, REL2).
- **Evidence:** documented execution (archived executor summary, release record); source inspection. Whether Colab's preloaded packages trigger the failure identically is **not verified**. The Kaggle image's NumPy 2.0.2 is typical of hosted images, and the pin is 2.5.3.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass.
  - The setup cell bootstraps uv and creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`).
  - It installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`) and runs the pinned stages in that environment. The kernel's preloaded NumPy/torch are never replaced, so no restart can be required.
  - Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`.
  - Do not add another in-kernel install guard or loosen pins to dodge the restart.
  - Implement it in the repository's notebook generator (`tools/build_notebook.py` / `tools/notebook_template.py`), regenerate, and re-qualify with a one-pass hosted Run all.
  - Correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:**
  - A fresh Colab (or Kaggle) runtime runs **Run all** once, with no restart and no intervention, through cell 23, and the executor summary shows one pass.
  - README, STATUS, `tutorials/README.md` and `docs/release-verification.md` no longer call the 2026-09-19 two-pass run a Run-all PASS.
  - The procedure no longer calls a restart "expected".
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11 (MUST).

### ITM-M2 — Major: result assertions turn an honest non-improving outcome into a crash before export

- **Cell/section:**
  - Section 6 (cell 17), Section 8 (cell 21), optional experiments (cell 24).
  - Template `tools/notebook_template.py:337`, `:424`, `:528-531`.
- **Observed issue:**
  1. Cell 21 ends with `assert adapted_test['rsum'] > frozen_test['rsum']`, and cell 17 with `assert frozen_test['rsum'] > baseline_chance['rsum'] and … > baseline_neighbour['rsum']`.
  2. `adapt` keeps epoch 0 whenever no epoch raises validation rsum (`if current > best_score`). The restored weights are then exactly the starting weights, so the adapted gallery rsum *equals* the frozen one and the strict `>` fails.
  3. The 40-photograph validation gallery saturates (5.82–5.89 of 6), so "no epoch beats epoch 0" is an ordinary outcome. P2 hit it with the notebook's own documented setting (`TRAINABLE_TEXT_LAYERS = 4`, `LEARNING_RATE = 5e-5`): epoch 1 scored 5.849 < 5.886 and `best_epoch` was 0.
  4. The `RERANK_TOP_K = 10` experiment reaches the same `assert` by another route (ITM-M3: Section 6 rerun on the adapted model).
- **Consequence:**
  - **Who hits it:** a learner whose experiment does not improve validation, and a BYOD user whose corpus does not reward adaptation on 20 % of their data.
  - **What they see:** a bare `AssertionError` with no explanation.
  - **What is lost:** `Run all` stops there, so there is no scores CSV, no adapter, no reload and no `result.json`.
  - **Spec reference:** the spec's reference practice is that "a negative held-out result is preserved rather than replaced" (§25.13). RUN9 and UX7 require optional experiments and BYOD never to block the path.
- **Evidence:**
  - Source inspection of cells 17 and 21 and of `adapt` (`pipeline.py`, selection loop).
  - Direct execution P2: `best_epoch = 0` on a documented setting. In P2 the run did not crash only because the old adaptation was still loaded (ITM-M3).
  - The fresh-base crash (P3) was **not verified**; the probe was stopped at the time cap.
- **Recommended correction:**
  - Replace both result assertions with reporting. Print the delta, and when the selector keeps epoch 0 or the delta is ≤ 0, say so and explain it. Export still proceeds, and the artifact manifest already records `best_epoch`.
  - Keep `assert` only for invariants such as reload parity.
  - Rewrite cells 20 and 24 so learners expect that outcome.
- **Acceptance check:**
  1. With a configuration that keeps epoch 0 (for example `EPOCHS = 1`, `LEARNING_RATE = 1e-3` from a fresh base), cells 19→23 complete. Cell 21 prints that epoch 0 was kept and the delta is 0.000, and `result.json` records it.
  2. `grep -n "^assert .*rsum" tutorials/blip_itm_colab.ipynb` returns nothing.
- **Spec:** RUN9, UX7 (MUST); §25.13.

### ITM-M3 — Major: rerunning a section after changing a field reuses the adapted model and labels it "frozen"

- **Cell/section:**
  - Section 7 (cell 19), Section 6 (cell 17), Section 4 / BYOD instruction (cells 0, 13).
  - Carried `pipeline.adapt`: it starts from the current weights, and `history[0]` is always noted "frozen model".
  - Template `tools/notebook_template.py:44`, `:528-531`.
- **Observed issue:** `adapt` trains from whatever weights `pipe` currently holds, and Section 3 is the only place a base model is loaded. The optional experiments give no rerun instruction.
  - **Section 7 rerun (P2, direct execution).** After the default run, the documented experiment (`TRAINABLE_TEXT_LAYERS = 4`, `LEARNING_RATE = 5e-5`, `EPOCHS = 1`) was applied and cells 19 and 21 were rerun.
    - Epoch 0 was printed as `'note': 'frozen model'` with validation rsum **5.886**, the previous adaptation's score. The true frozen model scores 5.816.
    - The new epoch scored 5.849, so the selector kept "epoch 0".
    - Section 8 reprinted the old comparison unchanged (adapted rsum 5.162), and the learner would conclude that the four-layer recipe behaves exactly like the default.
    - `adapt_result` now reports 38,202,370 trainable parameters, `lr = 5e-5` and `best_epoch = 0`. Rerunning Section 9 would write an artifact whose manifest records that configuration around tensors trained by the 2-layer, 2e-5 run.
  - **`RERANK_TOP_K = 10` (source).** Changing the field and rerunning Section 6 scores the *adapted* `pipe` and stores it as `frozen_test`. Section 8 then compares the adapted model with itself, and the `assert` from ITM-M2 fires.
  - **BYOD ("set `USE_BYOD = True` in Section 4 and re-run from that cell") (source).** The same mechanism applies: Section 6's "frozen model" is the VizWiz-adapted model, and Section 7 adapts on top of it.
- **Consequence:** the documented experiments and the BYOD comparison report a baseline that is not the base model, under the base model's name. The artifact's provenance would misdescribe its tensors.
- **Evidence:** direct execution P2; source inspection of `adapt`, `evaluate` and `save_artifact`. The `RERANK_TOP_K` and BYOD routes are inferred from source.
- **Recommended correction:**
  - Make Section 7 start from the verified base every time. Either reload in the cell, or have `adapt` refuse to start when `self.adapter is not None` unless told to continue, with a message naming the cell to rerun.
  - Make Section 6 refuse (or reload) when `pipe.adapter is not None`.
  - Record in `history[0]` whether epoch 0 is the base.
  - State the rerun range for each optional experiment and for BYOD ("rerun from Section 3", or a reset helper), following GDL10's Predict → Change → Run → Observe → Explain.
- **Acceptance check:**
  - After a completed default run, changing `TRAINABLE_TEXT_LAYERS`/`LEARNING_RATE` and rerunning Section 7 either reloads the base (epoch-0 validation rsum 5.816 on the sample) or stops with an actionable message.
  - Rerunning Section 6 after adaptation either scores the base or stops.
  - On the BYOD path, Section 6's "frozen" rsum equals the base model's rsum on the BYOD test split.
- **Spec:** UX7 (MUST), OUT8 / ART8 (MUST), GDL10, UX10.

### ITM-M4 — Major: the BYOD contract's stated limits and layout are not the enforced ones; bad or repeated uploads fail late, silently or opaquely

- **Cell/section:** Prerequisites (cell 1), BYOD paragraph (cell 0), Section 4 (cell 13); carried `samples.py` `split_dataset` and `validate_dataset`; template `tools/notebook_template.py:44-48`, `:132`, `:165-183`.
- **Observed issue** (direct execution P4 on stand-in zips, cell 13 verbatim):
  1. **Wrong minimum.** The contract says "a dataset needs 8..5,000 records". `split_dataset` takes 20 % for test and 15 % for validation, and cell 13 then validates **each split** with the default `min_records = 8`.
     - 8 records fail with "split leaves 5 training records".
     - 12, 20, 40 and 49 records fail with "2 / 4 / 6 / 7 records; 8..5000 are required". That message names neither the split nor the real rule.
     - The smallest passing dataset is **50** records (direct API scan, 8..80).
  2. **Silent stale data.** `work/byod` is never cleared. Uploading `b.zip` (with `records.json`) after `a.zip` (with `records.jsonl`) printed `data_source: 'BYOD (b.zip)'`, yet all 60 records came from upload A (`ids_from_upload_a: 60`, `ids_from_upload_b: 0`), because `records.jsonl` is searched first.
  3. **Subfolders.** Members are flattened to their base names, but `image` is resolved as written. A zip with `photos/img_000.jpg` referenced as `photos/img_000.jpg` fails with `image file not found: work\byod\photos\img_000.jpg`. Two images with the same name in different folders would overwrite each other (source inspection).
  4. **Opaque failures.** A cancelled upload and a zip without `records.jsonl` / `records.json` both raise a bare `StopIteration`.
  5. **Provenance under BYOD.** Cell 13 prints the VizWiz `text_sha256` and `pinned_photographs: 672` even for BYOD (observed). From source, cell 23 writes the VizWiz `corpus` block into `result.json`. Under BYOD the test gallery is also just the 20 % split (12 photographs for 60 records), so `evaluate` labels it `measured-small-sample`. The notebook does not tell the user that their gallery is this small, which matters given its own "Gallery first" advice.
- **Consequence:** a user with a modest labelled set, the likely BYOD user, meets the documented contract and is rejected with a message about "6 records" when they uploaded 40. A second attempt in the same session trains and evaluates on the wrong data under the new file name.
- **Evidence:** direct execution P4 (cell 13 verbatim, fake upload, stand-in images). The empty-caption rejection is clear (`records[0]: every reference caption must be a non-empty string`). The downstream BYOD path (Sections 5–9) is **not verified**: the P5 probe was stopped at the time cap.
- **Recommended correction:**
  - State the true minimum (derived from the fractions and `MIN_RECORDS`) before upload, or validate the splits with a smaller per-split minimum and name the split in the message.
  - Clear `work/byod` before extracting.
  - Preserve member paths under a guarded root, or reject subfolders with a message.
  - Turn an empty upload or a missing records file into "no records.jsonl/records.json found in <zip>; …".
  - Make the provenance print and `result.json` describe the BYOD source, and print the BYOD gallery size beside the chance baseline.
  - Add a location field (EXE2).
  - Implement all of this in `samples.py` and the template, regenerate, and record a BYOD run (positive and negative) in `docs/release-verification.md`.
- **Acceptance check:**
  1. The documented minimum equals the smallest set that runs cells 13→23. A smaller set is rejected in Section 4 with a message naming the split and the count required.
  2. A second upload's `data_source` and records are both upload B's.
  3. A zip whose records reference `photos/x.jpg` loads, or is rejected with a layout message.
  4. A cancelled upload prints an actionable message.
  5. Under BYOD, `result.json` carries no VizWiz corpus block.
- **Spec:** DAT12, DAT19, REL12 (MUST); VAL1, UX10, EXE2, OUT7.

### ITM-M5 — Major: declared `GUIDED`, but most of the guided layer and any structured learner activity are absent; infrastructure is not labelled

- **Cell/section:** whole notebook; generator `tools/build_notebook.py` (`render`) and `tools/notebook_template.py`.
- **Observed issue:**
  - **Absent:** an intended-learner statement, **How to use this notebook**, a roadmap, an Input → Model → Output contract, a glossary, a prediction prompt, an interpretation checkpoint, a worked answer, a troubleshooting section and a conclusion template. The P0 marker counts are all 0.
  - **Objectives are procedural** ("install the pinned runtime; read what the carried … modules guarantee; stage and digest-verify").
  - **Infrastructure is not labelled.** The three carried module cells (220 + 984 + 1,183 lines) sit between Section 1 and Section 3. They are not titled as **Infrastructure**, not collapsed (`cellView: form` count 0) and not marked safe to skip.
  - **Present:** two "Look for" notes (Sections 1 and 4), clear stage prose with consequences, a strong limits section with three "things to carry to real data", and three optional experiments without rerun instructions (ITM-M3).
- **Consequence:** a self-paced learner new to retrieval metrics must work out several things alone, and nothing checks their understanding:
  - what to notice;
  - why recall@k depends on gallery size;
  - why ITM re-ranking helps image → text but not text → image;
  - how to read the per-category split.

  The first screens after the install are 2,387 lines of code that look like prerequisite reading.
- **Evidence:** source inspection; P0 marker counts.
- **Recommended correction:** add the guided layer in the template, following the spec's 2.2 reference notebook (§25.13):
  - audience and prerequisites;
  - how to use the notebook;
  - a roadmap;
  - the Input → Model → Output contract;
  - a short glossary: ITC, ITM, gallery, recall@k, median rank, rsum, hard negative, epoch selection;
  - a prediction before Section 6 and before Section 8;
  - "What to notice" after each principal stage;
  - collapsible "Check your reasoning" answers (for example: why does rsum drop from 5.72 to 5.02 when the gallery grows, with no change to the model?);
  - one **Predict → Change one thing → Run → Observe → Explain** activity with its rerun range;
  - troubleshooting (install, download, memory, BYOD layout);
  - an evidence-based conclusion template.

  Title the carried cells `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:**
  - Each of GDL1–GDL14 maps to a named cell.
  - The three carried cells are titled Infrastructure and collapsed.
  - At least one activity asks for a prediction before a result and gives a worked answer after it.
- **Spec:** GDL1–GDL15, UX5, UX8, UX9 (SHOULD).

### ITM-m1 — Minor: doubled braces in the data contract, including the id pattern

- **Cell/section:** cell 0 (BYOD paragraph) and cell 1 (Data contract); template `tools/notebook_template.py:45`, `:132` (strings that are not passed through `.format`).
- **Observed issue:** `{{id, image, captions}}` and the id pattern `[A-Za-z0-9_.:-]{{1,64}}` render with doubled braces. The regex as shown is not the enforced `{1,64}`.
- **Consequence:** a user who copies the contract or the pattern gets a wrong schema string and a wrong regex.
- **Evidence:** source inspection; P0 `doubled_braces`.
- **Recommended correction:** use single braces in strings that are not formatted.
- **Acceptance check:** no `{{` or `}}` in any rendered markdown cell.
- **Spec:** DAT12.

### ITM-m2 — Minor: runtime figures and "build record" numbers do not name their environment, and two cells disagree

- **Cell/section:** cell 0 ("about forty minutes"), cell 1 ("the build record measured about 236 s … 561 s … 1,512 s … about four minutes on an RTX 5070 Ti"), cells 16, 18, 20 and 24; template `:40`, `:130`, `:386-389`, `:502-503`.
- **Observed issue:**
  - **Unnamed environments.** The CPU figures and the quoted metrics come from the local Windows CPU pre-flight row of the release record. `MODEL_CARD.md` and the Kaggle record give slightly different numbers under the same name "build record" (for example ITM pair accuracy 0.683 → 0.683, i2t R@1 0.790, ITM-reranked 0.841, against the notebook's 0.688, 0.791 and 0.844).
  - **Two cells disagree.** Cell 20 says the ITM pair accuracy "barely moved (0.683 → 0.688)", while cell 24 says it "did not move".
  - **Runtime expectations.** "About forty minutes" is not labelled as an estimate for a stated machine. This review's CPU run took 1,240 s of cell time, and the Kaggle T4 run took 387 s after the restart.
- **Consequence:** a learner on CUDA who sees 0.683 → 0.683 cannot tell whether that is the expected variation, and a Colab CPU learner has no basis for the time expectation.
- **Evidence:** source inspection; release record; MODEL_CARD; P1.
- **Recommended correction:**
  - Name the environment for each measured figure, and label estimates.
  - Quote one run consistently, and note that CPU and CUDA differ in the third decimal.
  - Make cells 20 and 24 agree.
- **Acceptance check:**
  - Every runtime figure and quoted metric in the notebook names its run or environment, or says "estimate".
  - Cells 20 and 24 describe the pair-accuracy change identically.
- **Spec:** UX12 (MUST).

### ITM-m3 — Minor: recipe comparisons and the gallery design were read on the test split, without saying so where the test split is described as untouched

- **Cell/section:** Section 7 prose (cell 18: "four blocks at 5e-5 for six epochs gained no more than two blocks at 2e-5 for four"), Section 8 (cell 20: "never used for training or epoch selection"), opening cell (the gallery was enlarged because the 70-photograph test split saturates); `MODEL_CARD.md` §Variability; `tutorials/README.md` ("the test split is used for nothing but the final evaluation").
- **Observed issue:**
  - **Recipe comparison on test.** The card's sweep compares recipes by their test-gallery rsum (5.158 vs 5.157; 5.757 vs 5.721 on the 70-photograph split).
  - **Gallery design on test.** The gallery size was chosen after the frozen model's test score was seen to hit the ceiling.
  - **Undisclosed at the point of use.** Both choices are disclosed in the card and the opening cell, but cell 20 and `tutorials/README.md` describe the test split as untouched by selection.
  - **Default not chosen on test.** Unlike its captioning sibling, the notebook does not say the default was chosen *because* of its test result.
- **Consequence:** small. The headline delta is still read honestly as sample-sanity evidence, but a learner is not told that the comparison design saw the test split.
- **Evidence:** source inspection.
- **Recommended correction:** in cell 20 and `tutorials/README.md`, say that the recipe comparisons in the card and the gallery size were decided with test-gallery numbers, and that the delta is therefore not fully independent. Preferably carve a development split from training for future sweeps.
- **Acceptance check:** no learner-facing text claims that the test split influenced nothing beyond the final evaluation, unless the documented sweep used only training/validation data.
- **Spec:** SPL6, SPL7, EVAL14.

### ITM-m4 — Minor: the BYOD zip handler has no expanded-size or member-count limit

- **Cell/section:** Section 4 (cell 13); template `:174-179`.
- **Observed issue:** each member is read fully into memory and written out with no total-size or count ceiling. Flattening to base names does keep writes inside `work/byod` (no traversal), which is good.
- **Consequence:** a large or hostile archive can exhaust the runtime's disk or memory before validation names a limit.
- **Evidence:** source inspection.
- **Recommended correction:** enforce an expanded-size and member-count limit before writing, and state the limit in the Prerequisites.
- **Acceptance check:** a zip whose declared uncompressed size exceeds the stated limit is rejected before any member is written.
- **Spec:** §20 (SHOULD), VAL6.

### Suggestions

- **ITM-S1:** update the declared notebook spec from 2.0 to 2.2 (metadata, opening cell, `NOTEBOOK_SOURCE`, References).
- **ITM-S2:** report a bootstrap interval over the 391 gallery photographs for the rsum and R@1 deltas, so that "a few recall points" carries its own uncertainty.
- **ITM-S3:** extend reload parity from an 8×8 grid to the whole gallery's ITC scores (VER4), which costs one more `evaluate` on the reloaded pipeline.
- **ITM-S4:** record per-stage wall times and the device in `result.json`, so that hosted runs can be compared with the runtime figures the notebook quotes.

## 5. Readiness

**Needs revision.** Remaining gates:

1. ITM-M1: a one-pass hosted `Run all` of a revised blob (uv isolated environment), and corrected release records.
2. ITM-M2 and ITM-M3:
   - no result assertions on held-out metrics;
   - experiments and BYOD that start from the base model, with stated rerun ranges.
3. ITM-M4: a BYOD contract that matches enforcement, recorded with a positive and a negative BYOD run that reaches export (REL12).
4. ITM-M5: the guided layer.

After those, record a fresh clean-runtime run of the new blob in `docs/release-verification.md`.

## 6. Verified versus inferred

- **Verified by direct execution** (CPU, real weights, exact pins, not a clean runtime):
  - the default numbers and parity (P1);
  - the stale-rerun mislabel with the documented four-layer experiment, including `best_epoch = 0` (P2);
  - the BYOD minimum, the stale second upload, the subfolder failure and the `StopIteration` cases (P4).
- **Verified from documented evidence:** the restart in the Kaggle qualification run of this blob.
- **Inferred from source:**
  - the `AssertionError` when epoch 0 is kept from a fresh base (P3 stopped at the time cap);
  - the `RERANK_TOP_K = 10` rerun crash;
  - the BYOD "re-run from that cell" stale baseline;
  - the BYOD downstream path (P5 stopped at the time cap);
  - basename collisions and the zip size risk;
  - Colab behaving like Kaggle at the install.
- **Not verified:** Colab, CUDA in this review, real-photograph BYOD, learner understanding.
- **Most likely to be wrong:** ITM-M2's claim that a fresh-base non-improving run *crashes*. It rests on source reading: with `best_epoch = 0` the restored weights equal the base, so the two `evaluate` calls should give identical rsum and the strict `>` should fail. A CPU/CUDA nondeterminism in `evaluate` could in principle make the two rsums differ by a hair, and then the assertion would pass or fail at random rather than always fail. Either way the gate is wrong.

Probes: `blip_itm_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
