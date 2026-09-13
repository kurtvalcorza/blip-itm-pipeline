# Release verification

`tutorials/blip_itm_colab.ipynb` (`TASK-INFERENCE`, **standalone** carrier) is a
**release candidate** until the exact notebook revision has executed top-to-bottom in a clean
supported runtime. Unit tests, JSON validation, code-cell compilation, the generator parity checks
and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under
DIMER Notebook Specification 2.0. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that
  profile, spec `2.0`, a pedagogical mode, `standalone: true` and `generated_from` (repository, revision, module
  SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on
  the primary path; exactly one cell tagged `embedded_module` equal to
  `src/blip_itm_pipeline/pipeline.py` after the generator's documented rewrites; the
  inline `MANIFEST` equal to the committed snapshot manifest and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to `tools/build_notebook.py`
  output for its recorded revision; the pinned-install cell with its restart-on-stale-import guard;
  `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline
  manifest, which the notebook asserts against the module before fetching), the revision is a 40-hex
  immutable commit, and the same identity string appears in `README.md`, `MODEL_CARD.md`, and
  `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `BlipItmPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `score`, `evaluation_report`),
  the ceiling print (`MIN_IMAGE_SIDE`, `MAX_IMAGE_SIDE`, `IMAGE_SIZE`, `MAX_IMAGES`, `MAX_TEXTS`,
  `MAX_TEXT_CHARS`), the exports (including the executed-vs-hosted weight-file names), the learner-facing
  statements (caller-owned caption set, neither score calibrated nor abstaining, the weight-format note,
  recall@k needs a captioned set, `not-measurable` on BYOD, the caption set is part of the request,
  capability exclusions) and the gated-off BYOD default listed in the validator; forbidden patterns
  (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a mutable
  `revision='main'`, direct `from transformers import` / `BlipForImageTextRetrieval` / `BlipProcessor` /
  `use_itm_head` / `weights_only=` / `from huggingface_hub import` use **outside the carried module
  cell**, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy` and
`pillow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit
suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`;
injected runner, no weights). These are source/provenance and unit checks. They are **not** execution
evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (**no repository checkout is needed — the notebook is standalone**) |
| Local Windows-venv harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded
   in `metadata.dimer.generated_from` and that the installed core package versions equal the inline
   `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines `BlipItmPipeline`, `validate_inputs`,
     `evaluation_report`, `recall_at_1`, `format_texts`, `verify_snapshot`, `stage_missing_files`) with no
     import of the repository package;
   - three synthetic cartoon scenes drawn in code with their RGB SHA-256 printed, three authored captions,
     and the ceilings (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `IMAGE_SIZE` 384, `MAX_IMAGES` 16,
     `MAX_TEXTS` 16, `MAX_TEXT_CHARS` 256) surfaced;
   - pinned `Salesforce/blip-itm-base-coco` acquisition at the immutable revision through the carried
     module: the inline `MANIFEST` is asserted against the module identity and written to
     `weights/blip-itm-base-coco/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all 8
     manifest entries on a clean runtime (`pytorch_model.bin` among them; `tf_model.h5` must not be
     fetched), `verify_snapshot` returns its summary dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)`
     loads from the verified directory with no further Hub access (any download in the logs after
     staging is a finding);
   - `validate_inputs` writes `outputs/blip_itm_input_manifest.json` (verdict `accepted`, three inputs,
     three captions, nine pairs, one recorded rejection finding from the duplicate-caption probe);
   - one `score` call over the 3×3 grid; record the ITM probability and cosine grids (the card-pass CPU
     smoke gave ITM diagonal 0.998 / 0.634 / 0.278 with off-diagonal ≤ 0.001 and cosine diagonal 0.495 /
     0.422 / 0.441; a materially different result is a finding to record, not a failure by itself, because
     no metric is asserted — kernels differ across devices);
   - `evaluation_report` writes `outputs/blip_itm_evaluation_report.json` with verdict `sample-sanity`,
     four `recall_at_1` entries (both directions × both scores) and the chance baselines on the synthetic
     grid (`not-measurable` on BYOD), stated as such;
   - `outputs/blip_itm_result.json`, `outputs/blip_itm_scores.csv` and `outputs/blip_itm_annotated.png`
     written with `NOTEBOOK_SOURCE`, model revision, model licence, the executed and hosted weight-file
     names, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/blip_itm_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/blip_itm_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Local pre-flight evidence (not a supported runtime)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
__LOCAL_ROW__

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| | | | Default sample path | | pending — no Colab/Kaggle run yet |

## Current status

No clean-runtime execution in a **supported** runtime (Colab or Kaggle) has been recorded yet; the run is
**pending**. What exists: static validation (`tools/validate_release_assets.py`), the generator parity
checks (`--check` OK), the offline unit suite, and one **local fresh-kernel execution** of the generated
notebook (table above) that exercised the standalone carrier, the real `hf_hub_download` staging path
into an empty `weights/` directory, verification, scoring, the evaluation report and every export —
which is necessary but not promotion evidence because the workstation is not a supported runtime. The
registry status remains **Candidate** until a reviewer confirms a recorded supported-runtime run against
the notebook blob under review and an integrator promotes it. Facts a reviewer should weigh: the CUDA
path has not been executed; the executed weight file is a pickle (`pytorch_model.bin`, digest-checked before
`torch.load(weights_only=True)`) while the DIMER-hosted blob is the upstream `tf_model.h5`, which this
repository never loads and whose equivalence to the pickle it has not verified; the tutorial grid is three
flat cartoons and three authored captions, so a perfect recall@1 on nine pairs is plumbing evidence only, and
the fruit scene's own caption scored just 0.278 — an uncalibrated score that no fixed threshold would treat
consistently; blank and noise images still receive rankings; and every pair costs two forward passes
(~0.35 s on the reference CPU), so a 16×16 grid is 256 pairs and scales as images × captions, not
linearly.
