"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E image-text retrieval workflow: the pinned Salesforce/blip-itm-base-coco
snapshot is digest-verified and loaded, the captions of a digest-pinned VizWiz-Captions shard are read
column-only and the 672 photographs of its first two row groups are read the same way, validated and split
by image (row group 0 into train / validation / core test, row group 1 into the test gallery), three drawn
scenes are scored through the inference contract, the frozen model's retrieval over the 391-photograph gallery
is measured beside two non-neural baselines, a bounded fine-tuning of the text encoder's last blocks, both
projections and the ITM head runs in the kernel, the gallery is scored again per category, the adapted model
re-scores the drawn scenes, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "blip_itm_pipeline",
    "repo_name": "blip-itm-pipeline",
    "stem": "blip_itm",
    "notebook_name": "blip_itm_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `Salesforce/blip-itm-base-coco` snapshot (an 895 MB `pytorch_model.bin` pickle, opened only after its "
        "SHA-256 matches and with `weights_only=True`), reads the four text columns of one digest-pinned VizWiz-Captions shard "
        "from the Hugging Face Hub (about 0.5 MB over HTTP range requests, no credential) and the image column of its first "
        "two row groups (672 photographs, about 166 MB in two range reads, each photograph refused on any size or SHA-256 "
        "mismatch), cuts row group 0 by image into 208 / 40 / 70 training, validation and core-test photographs and appends "
        "row group 1's 321 captioned photographs to the test gallery (391 photographs, 1,737 captions), scores three drawn "
        "scenes through the inference contract with an input manifest and a rejection probe, measures the frozen model's "
        "retrieval over the gallery (recall@1/5/10 in both directions, ITM re-ranking, ITM pair accuracy) beside the chance "
        "and colour-keyword baselines, runs a bounded fine-tuning of the text encoder's last two blocks, the two ITC "
        "projections and the ITM head with BLIP's contrastive and matching objectives and validation-rsum epoch selection, "
        "scores the gallery again per category, re-scores the drawn scenes with the adapted model, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify score parity. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration "
        "edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about forty minutes after the downloads; a CUDA runtime is "
        "used automatically when present and finishes in a few minutes."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "holding a `records.jsonl` (or `records.json`) of `{{id, image, captions}}` objects — `image` a file name inside the "
        "zip, `captions` one or more distinct matching captions (the first is the retrieval query), optional `category` — "
        "beside the image files. They pass through the same validation, seeded image-disjoint split, baselines, fine-tuning, "
        "held-out evaluation, artifact export and reload-parity cells as the VizWiz sample. The expected schema and the "
        "ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is "
        "optional and never part of the default path."
    ),
    "pipeline_class": "BlipItmPipeline",
    "weights_key": "blip-itm-base-coco",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers"],
    "title": "BLIP ITM-base — DIMER E2E image-text retrieval fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/blip-itm-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/blip-itm-pipeline/blob/main/tutorials/blip_itm_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Salesforce%2Fblip--itm--base--coco-ffcc4d?style=flat",
            "https://huggingface.co/Salesforce/blip-itm-base-coco",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-salesforce%2FBLIP-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/salesforce/BLIP",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2201.12086-b31b1b.svg", "https://arxiv.org/abs/2201.12086"),
    ],
    "capability": "image-text matching / retrieval and bounded supervised fine-tuning of the text encoder's last blocks, the ITC projections and the ITM head on a photograph/matching-captions dataset, using the pinned `Salesforce/blip-itm-base-coco` weights",
    "intro": (
        "`Salesforce/blip-itm-base-coco` is the BLIP model of Li et al. (2022) fine-tuned for image-text retrieval on COCO — "
        "a ViT-B/16 image encoder at 384×384 and a 12-layer BERT-style text encoder whose blocks also cross-attend to the "
        "image; 223,744,258 parameters, published under the **BSD-3-Clause** licence. It gives two scores per image–caption "
        "pair: the **ITC cosine** between the projected image and text embeddings (a dual-encoder score that can rank a "
        "gallery cheaply) and the **ITM probability** from a classifier head over the fused encoding (a per-pair score that "
        "is more accurate and far more expensive). **Neither score is calibrated and neither abstains**: a caption set with "
        "nothing that fits an image still yields a highest-scoring caption.\n\n"
        "What this notebook adds to inference is **adaptation with matching captions**. The dataset is real and out of the "
        "model's distribution: VizWiz-Captions (Gurari et al., ECCV 2020; **CC BY 4.0**) — photographs taken by blind "
        "people, each with up to five crowd-written captions that name what is held and what a label says — a population "
        "the model never saw. The notebook reads only the four text columns of one pinned Hub shard (about 0.5 MB over "
        "HTTPS range requests) and the image column of its **first two row groups** (672 photographs in two range reads, "
        "each pinned by size and SHA-256 in the carried module). Row group 0 is split by image into 208 / 40 / 70; row "
        "group 1's 321 captioned photographs join the **test gallery** so that retrieval is scored over 391 photographs "
        "and 1,737 captions — on a 70-photograph gallery the frozen model is already at the ceiling (the build record "
        "measured rsum 5.72 of 6), and a gallery that small says nothing about adaptation. The honest question is narrow: "
        "does a bounded fine-tuning of the text encoder's last blocks, the projections and the ITM head on 208 photographs "
        "move recall over a real gallery, in which direction, and does the ITM re-ranking move with it? The retrieval "
        "protocol is implemented in the carried modules (**recall@1/5/10** in both directions with every caption in the "
        "gallery, **median rank**, **rsum**, **ITM pair accuracy** against the hardest wrong caption), and two **non-neural "
        "baselines** — **chance** and the **colour-keyword nearest neighbour** — show where a system with no model sits. "
        "Nothing here is a quality claim about your photographs: it is one seeded split of one corpus.\n\n"
        "**Weight-format note:** upstream hosts no SafeTensors at the pinned revision; the carried module executes the "
        "digest-pinned `pytorch_model.bin` (a pickle, deserialised with `weights_only=True` after its SHA-256 is checked), "
        "while the `tf_model.h5` upstream also hosts is DIMER's upload artifact and is never loaded here. Section 3 stages "
        "and digest-verifies the snapshot before the processor or the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot (a pickle checkpoint, and why that matters); read the captions of a "
        "digest-pinned corpus without downloading its shard and its photographs from two pinned row groups with per-file "
        "digests, validate them and split by image without leakage; score drawn scenes through the public API and read the "
        "two scores correctly (uncalibrated, no abstention); measure the frozen model's retrieval over a real gallery with "
        "recall@k in both directions and ITM re-ranking beside two non-neural baselines and read the per-category "
        "breakdown; run a bounded fine-tuning with BLIP's two objectives, explicit hyperparameters and validation-based "
        "epoch selection; evaluate on an image-disjoint gallery; re-score drawings from a different image family with the "
        "adapted model; and export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "captioning or question answering (separate checkpoints), retrieval over thousands of candidates with a precomputed "
        "index (the gallery here is 391 photographs, embedded in the kernel), reading text in the image (BLIP is not an OCR "
        "model, even though VizWiz captions transcribe labels), non-English captions, batch throughput, evaluation on COCO or "
        "Flickr30k (not bundled), fine-tuning of the vision encoder or the text embeddings, training on images that are not "
        "the pinned sample or your own uploads, and any claim that a VizWiz split stands in for your photographs. The "
        "repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is slow but adequate: the build record measured about 236 s to embed and score the 391-photograph gallery with top-5 ITM re-ranking and 561 s for the four epochs of fine-tuning (888 photograph–caption pairs per epoch, three fused passes per pair for the matching loss), including the one-off encoding of the 208 training photographs and the per-epoch validation scoring; the whole default path took 1,512 s with the snapshot and photographs already cached, and about four minutes on an RTX 5070 Ti. The pinned `torch==2.14.0` install and the 895 MB checkpoint are the large downloads of the run; the two row groups of photographs add about 166 MB.",
        "- **Knowledge:** basic Python and PIL; what a dual-encoder (contrastive) score and a fused-encoder (matching) score are and why one is cheap and the other accurate; why a pickle checkpoint needs a digest check before `torch.load`; what recall@k, median rank and rsum measure over a gallery and why a small gallery inflates them; why a high match probability is not a correct match.",
        "- **Data contract:** records are `{{id, image, captions}}` — an image file decodable by Pillow with sides between `MIN_IMAGE_SIDE` (16) and `MAX_IMAGE_SIDE` (4096) px and one or more distinct, non-empty matching captions of at most `MAX_TEXT_CHARS` (256) characters (`MIN_CAPTIONS` = 1; VizWiz supplies up to five; the first caption is the retrieval query); optional `image_id` (defaults to the id) groups records on the same image and optional `category` labels the breakdown (`text` / `no-text` in the sample, from the corpus's text-detected flag). Ids match `[A-Za-z0-9_.:-]{{1,64}}` and are unique; a dataset needs 8..5,000 records; every record on the same image lands in the same split so a test image is never trained on; every (image, caption) pair is one positive training sample and the batch supplies the negatives. BYOD accepts one zip of images plus a `records.jsonl` / `records.json` in that shape.",
        "- **Validation is structural, not semantic:** every image is opened and decoded and every caption checked, but nothing checks that a caption matches its image — a mislabelled corpus is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of people, documents or homes are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path reads three parts of one object in the Hub dataset repository `mm-eval/VizWiz-Captions` at the immutable revision `c4a6d897…` (`data/val-00004-of-00005.parquet`, 392,245,504 bytes, SHA-256 `4492465a…`): the declared size and SHA-256 are checked against the pins before any byte is read; the four text columns of all 1,550 rows are fetched over HTTPS range requests through `pyarrow` (only the parquet footer and those column chunks) and refused unless their decoded SHA-256 matches; then the image column of row groups 0 and 1 only (672 JPEG files, about 166 MB) is read the same way, each photograph pinned by size and SHA-256 in the carried module and refused on any mismatch. The corpus is CC BY 4.0 (Gurari et al., 2020).",
    ],
    "cells": [
        {
            "md": (
                "## 4. VizWiz photographs, captions and split\n\n"
                "`fetch_annotations` reads the pinned shard's four text columns (or the cache under `weights/vizwiz-captions/`): "
                "it first checks the byte size and SHA-256 the Hub declares for the file against the pins, then reads only the "
                "parquet footer and those column chunks through `pyarrow` over HTTPS range requests, and refuses the decoded "
                "columns unless their SHA-256 matches. `fetch_images` stages the 672 pinned photographs of row groups 0 and 1 "
                "— each cached file is re-hashed; anything missing is read from its row group's image column in one range "
                "read and refused on any size or SHA-256 mismatch. `build_sample_dataset` keeps the photographs that carry at "
                "least one caption within the scoring ceiling (one caption of 1,391 in row group 0 is longer than 256 "
                "characters and is dropped; duplicates are removed), shuffles row group 0's 318 with `SPLIT_SEED` and cuts "
                "them **by image** into 208 / 40 / 70 training, validation and core-test records, then appends row group 1's "
                "321 captioned photographs to the test split as the gallery. Each record is labelled `text` or `no-text` "
                "from the corpus's text-detected flag. `validate_dataset` then opens and decodes every image and checks every "
                "record against the contract, `check_split_disjoint` asserts no image is shared, and the training split is "
                "written to `outputs/{stem}_train.jsonl` in the shape BYOD expects.\n\n"
                "Look for: 1,550 annotation rows, 672 photographs, three digests, the category mix per split (a little over "
                "half the photographs contain text), captions per image between 1 and 5, a 391-record test split whose "
                "gallery holds 1,737 captions, and four refusal probes — a duplicate id, a missing image file, a duplicated "
                "caption and a dataset too small to split — each rejected before `torch` does anything."
            ),
            "code": (
                "import collections\n"
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import zipfile\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_dir = Path('work') / 'byod'\n"
                "    byod_dir.mkdir(parents=True, exist_ok=True)\n"
                "    with zipfile.ZipFile(io.BytesIO(payload)) as archive:\n"
                "        for member in archive.infolist():\n"
                "            name = Path(member.filename).name\n"
                "            if member.is_dir() or not name or name.startswith('.'):\n"
                "                continue\n"
                "            (byod_dir / name).write_bytes(archive.read(member))\n"
                "    records_file = next(p for p in (byod_dir / 'records.jsonl', byod_dir / 'records.json') if p.is_file())\n"
                "    records = load_byod_dataset(records_file)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED, base_dir=byod_dir)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    annotations = fetch_annotations(cache_dir='weights/vizwiz-captions')\n"
                "    image_paths = fetch_images(sorted(IMAGE_PINS), cache_dir='weights/vizwiz-captions')\n"
                "    raw_rows = {{'annotations': len(annotations), 'photographs': len(image_paths), 'row_groups': CORPUS_FILE['row_groups']}}\n"
                "    splits = build_sample_dataset(annotations, seed=SPLIT_SEED, image_paths=image_paths)\n"
                "    data_source = f'{{CORPUS_NAME}} {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "categories = {{name: manifest['categories'] for name, manifest in dataset_manifests.items()}}\n"
                "gallery_texts, gallery_owners = gallery(test_records)\n"
                "write_dataset_jsonl(splits['train'], 'outputs/{stem}_train.jsonl')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'gallery_captions': len(gallery_texts), 'text_sha256': CORPUS_FILE['text_sha256'][:16] + '...', 'pinned_photographs': len(IMAGE_PINS)}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_images': manifest['unique_images'], 'categories': manifest['categories'], 'captions_per_image': manifest['captions_per_image'], 'caption_words': manifest['caption_words'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = splits['train'][0]\n"
                "print({{'example': {{'id': example['id'], 'image': Path(example['image']).name, 'size': example['image_size'], 'category': example['category'], 'query': query_caption(example), 'captions': len(example['captions'])}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in splits['train'][:8]],\n"
                "    'missing image file': [{{**splits['train'][0], 'image': 'work/does-not-exist.jpg'}}, *splits['train'][1:8]],\n"
                "    'duplicated caption': [{{**splits['train'][0], 'captions': [splits['train'][0]['captions'][0]] * 2}}, *splits['train'][1:8]],\n"
                "    'too small': splits['train'][:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Score through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: three flat cartoon scenes — "
                "a red house with a tree, a beach with a red sailboat, an apple and an orange on a table — are drawn in code "
                "with Pillow (no text rendering, so their digests are stable across builds) with one authored caption each, "
                "a different image family from the photographs, and a grid the model will score again after adaptation. "
                "`validate_inputs` applies exactly the checks `score` applies (image sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE`, "
                "at most `MAX_IMAGES` × `MAX_TEXTS` distinct captions of up to `MAX_TEXT_CHARS`) and returns an input "
                "manifest; a duplicated caption is validated too and its rejection recorded as a finding. `score` returns the "
                "`itm_probability`, `itm_logit_match` and `cosine` grids plus per-image rankings. **Neither score is "
                "calibrated and neither abstains** — the ITM probability is a per-pair classifier output, the cosine is "
                "comparable only within a row or column, and a caption set with nothing that fits still produces a "
                "highest-scoring caption. As recorded in the model card, the repository's CPU smoke on this same grid put "
                "every image's own caption first by both scores (ITM 0.998 for the house, 0.634 for the beach, 0.278 for the "
                "fruit scene). The per-image `evaluation_report` on the 3×3 grid is `sample-sanity` — three authored pairs "
                "carry no dispersion; whether the model retrieves *well* is what Section 6 measures over 391 photographs."
            ),
            "code": (
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n\n"
                "def synthetic_scenes():\n"
                "    \"\"\"Three flat cartoon scenes drawn with Pillow (no text); returns [(name, image, authored caption)].\"\"\"\n"
                "    house = Image.new('RGB', (640, 480), (135, 206, 235))  # sky\n"
                "    d = ImageDraw.Draw(house)\n"
                "    d.rectangle([0, 300, 640, 480], fill=(60, 179, 75))  # grass\n"
                "    d.ellipse([500, 40, 600, 140], fill=(255, 215, 0))  # sun\n"
                "    d.rectangle([120, 180, 320, 330], fill=(200, 40, 40))  # red house\n"
                "    d.polygon([(100, 180), (220, 90), (340, 180)], fill=(90, 50, 20))  # brown roof\n"
                "    d.rectangle([200, 260, 240, 330], fill=(70, 40, 20))  # brown door\n"
                "    d.ellipse([420, 260, 520, 360], fill=(40, 100, 40))  # tree crown\n"
                "    d.rectangle([460, 350, 480, 420], fill=(90, 60, 30))  # trunk\n"
                "    d.ellipse([60, 380, 140, 440], fill=(255, 255, 255))  # white ball\n"
                "    beach = Image.new('RGB', (640, 480), (120, 190, 240))  # sky\n"
                "    d = ImageDraw.Draw(beach)\n"
                "    d.rectangle([0, 220, 640, 330], fill=(30, 110, 200))  # sea\n"
                "    d.rectangle([0, 330, 640, 480], fill=(238, 214, 150))  # sand\n"
                "    d.ellipse([60, 40, 150, 130], fill=(255, 230, 80))  # sun\n"
                "    d.polygon([(400, 330), (470, 330), (435, 210)], fill=(230, 40, 40))  # red sail\n"
                "    d.rectangle([432, 210, 438, 330], fill=(90, 60, 30))  # mast\n"
                "    d.ellipse([200, 370, 260, 430], fill=(255, 120, 40))  # beach ball\n"
                "    fruit = Image.new('RGB', (480, 480), (250, 250, 245))\n"
                "    d = ImageDraw.Draw(fruit)\n"
                "    d.ellipse([60, 120, 220, 280], fill=(220, 30, 30))  # red apple\n"
                "    d.rectangle([135, 95, 145, 125], fill=(80, 50, 20))  # stalk\n"
                "    d.ellipse([250, 140, 430, 300], fill=(255, 170, 20))  # orange\n"
                "    d.polygon([(90, 400), (400, 400), (360, 330), (130, 330)], fill=(180, 120, 60))  # table\n"
                "    return [\n"
                "        ('synthetic_house_640x480.png', house, 'a red house with a tree under a blue sky'),\n"
                "        ('synthetic_beach_640x480.png', beach, 'a sailboat on the sea next to a beach'),\n"
                "        ('synthetic_fruit_480x480.png', fruit, 'an apple and an orange on a wooden table'),\n"
                "    ]\n\n\n"
                "scenes = synthetic_scenes()\n"
                "scene_names = [name for name, _, _ in scenes]\n"
                "scene_images = [image for _, image, _ in scenes]\n"
                "scene_texts = [caption for _, _, caption in scenes]\n"
                "correct_text_per_image = list(range(len(scenes)))  # caption i describes image i\n"
                "scene_digests = {{name: hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() for name, image in zip(scene_names, scene_images)}}\n"
                "ceilings = {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'IMAGE_SIZE': IMAGE_SIZE, 'MAX_IMAGES': MAX_IMAGES, 'MAX_TEXTS': MAX_TEXTS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'MIN_CAPTIONS': MIN_CAPTIONS, 'MAX_CAPTION_CHARS': MAX_CAPTION_CHARS}}\n"
                "print(ceilings)\n"
                "input_manifest = validate_inputs(scene_images, scene_texts, names=scene_names)\n"
                "try:\n"
                "    validate_inputs(scene_images, [scene_texts[0], '  ' + scene_texts[0] + ' '])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'duplicate-caption-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'scenes': scene_names, 'rgb_sha256': {{k: v[:16] + '...' for k, v in scene_digests.items()}}, 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n"
                "t0 = time.perf_counter()\n"
                "result = pipe.score(scene_images, scene_texts)\n"
                "np.set_printoptions(precision=3, suppress=True)\n"
                "print({{'seconds': round(time.perf_counter() - t0, 2), 'pairs': result['n_images'] * len(result['texts'])}})\n"
                "print('itm_probability [image][text]:')\n"
                "print(result['itm_probability'])\n"
                "print('cosine [image][text]:')\n"
                "print(result['cosine'])\n"
                "checks = {{\n"
                "    'grid_shape': result['itm_probability'].shape == (len(scenes), len(scene_texts)),\n"
                "    'probabilities_in_range': bool(np.all((result['itm_probability'] >= 0) & (result['itm_probability'] <= 1))),\n"
                "    'cosines_finite': bool(np.all(np.isfinite(result['cosine']))),\n"
                "    'one_ranking_per_image': len(result['rankings']) == len(scenes),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'score output failed a sanity check: {{checks}}')\n"
                "frozen_scene = evaluation_report(result, correct_text_per_image, sample_kind='synthetic')\n"
                "print({{'checks': checks, 'frozen_scene': {{m['id']: round(m['value'], 3) for m in frozen_scene['metrics']}}, 'verdict': frozen_scene['verdict']}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's retrieval over the gallery\n\n"
                "Three systems frame the adaptation. **Chance** is the analytical expectation of a random ranking over the "
                "same gallery (recall@k ≈ k / 391 in the text-to-image direction). The **colour-keyword nearest neighbour** "
                "retrieves with no model: text → image through the query's closest training caption by bag-of-words F1 and "
                "that photograph's 3×3 mean-colour grid; image → text through the colour-nearest training photograph's "
                "captions. The **frozen model** is scored by `pipe.evaluate`: every caption of every test record is the "
                "gallery (1,737 captions for 391 photographs), the ITC cosine grid gives **recall@1/5/10** in both directions "
                "(image → text counts a hit when any of the photograph's own captions is in the top k), **median rank** and "
                "**rsum** (the sum of the six recalls, 0..6); then the ITM head re-ranks each photograph's top-5 captions and "
                "each query caption's top-5 photographs (`itm_i2t_recall_at_1`, `itm_t2i_recall_at_1`) and is scored on "
                "**ITM pair accuracy** — each photograph's query caption against the hardest wrong caption by ITC. Expect the "
                "frozen model far above both baselines — it is a trained retriever — and read the per-category breakdown: the "
                "build record measured i2t R@1 0.742 / t2i R@1 0.652 / rsum 5.02 frozen, with the `text` photographs (labels, "
                "screens, packaging: i2t R@1 0.73, t2i 0.63) harder than the `no-text` ones (0.81 / 0.78)."
            ),
            "code": (
                "RERANK_TOP_K = 5  # @param {{type:\"integer\"}}\n\n"
                "baseline_chance = chance_baseline(test_records)\n"
                "baseline_neighbour = colour_keyword_baseline(train_records, test_records)\n"
                "METRICS = ('i2t_recall_at_1', 'i2t_recall_at_5', 'i2t_recall_at_10', 't2i_recall_at_1', 't2i_recall_at_5', 't2i_recall_at_10', 'rsum')\n"
                "ITM_METRICS = ('itm_i2t_recall_at_1', 'itm_t2i_recall_at_1', 'itm_pair_accuracy')\n"
                "print({{'chance_baseline': {{k: round(baseline_chance[k], 3) for k in METRICS}}, 'n': baseline_chance['n_images'], 'note': baseline_chance['baseline']}})\n"
                "print({{'colour_keyword_baseline': {{k: round(baseline_neighbour[k], 3) for k in METRICS}}, 'note': baseline_neighbour['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, rerank_top_k=RERANK_TOP_K)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'itm': {{k: round(frozen_test[k], 3) for k in ITM_METRICS}}, 'median_rank': [frozen_test['i2t_median_rank'], frozen_test['t2i_median_rank']], 'n': [frozen_test['n_images'], frozen_test['n_captions']], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': METRIC_DEFINITIONS}})\n"
                "frozen_fields = frozen_test['by_category']  # each category scored as its own sub-gallery of the same grid\n"
                "print({{'by_category_frozen': frozen_fields}})\n"
                "assert frozen_test['rsum'] > baseline_chance['rsum'] and frozen_test['rsum'] > baseline_neighbour['rsum']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the text encoder's last blocks, the projections and the ITM head\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_TEXT_LAYERS` blocks of the fused text encoder (self-attention, "
                "cross-attention to the image and feed-forward), the two ITC projections and the ITM head — two blocks by "
                "default, 19,298,818 of 223,744,258 parameters; the vision encoder and the text embeddings stay frozen. The "
                "frozen vision encoder's output is computed **once** per training photograph and reused across epochs. Every "
                "(photograph, caption) pair is one sample, 888 per epoch here, and a batch of 16 trains BLIP's two "
                "objectives: the **image-text contrastive loss** (symmetric cross-entropy over the in-batch cosine "
                "similarities at temperature 0.07, pairs of the same photograph counted as positives) and the **image-text "
                "matching loss** (the ITM head on the batch's positives plus one hard negative caption per photograph and one "
                "hard negative photograph per caption, sampled in proportion to their ITC similarity, never from the same "
                "photograph). AdamW at a fixed learning rate, gradient clipping at 1.0, seeded shuffling and no scheduler. "
                "Epoch 0 records the frozen model's validation retrieval; every epoch is scored on the 40 validation "
                "photographs by ITC rsum, and the epoch with the highest validation rsum is kept — a 40-photograph gallery "
                "saturates near 5.9 of 6, so that selection is weakly informed, which is why the 391-photograph gallery in "
                "Section 8 is what the numbers are read from.\n\n"
                "Watch the training loss fall from about 0.62 while validation rsum moves by hundredths: the frozen model "
                "already retrieves well, and what adapts is the alignment of VizWiz's caption style with its photographs. The "
                "build record's counter-examples — four blocks at 5e-5 for six epochs gained no more than two blocks at 2e-5 for four — are in the model card."
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 2e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 16  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_TEXT_LAYERS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in ('i2t_recall_at_1', 't2i_recall_at_1', 'rsum')}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_text_layers=TRAINABLE_TEXT_LAYERS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'training_pairs': adapt_result['n_pairs'], 'itc_temperature': adapt_result['itc_temperature'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation over the gallery\n\n"
                "The test photographs were never used for training or epoch selection, and no test image appears in the "
                "training or validation splits. The adapted model is scored exactly as the frozen model was in Section 6, the "
                "four systems are put side by side on the six recalls and rsum, the ITM re-ranking and pair accuracy are "
                "repeated, and the per-category recall@1 is repeated. Read it in this order: **rsum** first (the number the "
                "epoch was selected on), then **image → text R@1** (where the build record measured the gain: 0.742 → 0.791, "
                "and 0.803 → 0.844 after ITM re-ranking — carried by the `text` photographs, 0.73 → 0.80, while the `no-text` ones "
                "stay at 0.81) and **text → image R@1** (0.652 → 0.673), then the ITM pair accuracy, which barely moved in the "
                "build record (0.683 → 0.688) — the ITM head was already the better judge of hard pairs, and 208 "
                "photographs from one seeded split of one corpus gives **no dispersion estimate**; the deltas are sample-sanity "
                "evidence that the adaptation contract works, not a benchmark, and a gain on VizWiz says nothing about your "
                "photographs until you measure it there."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, rerank_top_k=RERANK_TOP_K)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "adapted_fields = adapted_test['by_category']\n"
                "comparison = {{metric: {{'chance': round(baseline_chance[metric], 3), 'neighbour': round(baseline_neighbour[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['itm'] = {{metric: {{'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in ITM_METRICS}}\n"
                "comparison['median_rank'] = {{'frozen': [frozen_test['i2t_median_rank'], frozen_test['t2i_median_rank']], 'adapted': [adapted_test['i2t_median_rank'], adapted_test['t2i_median_rank']]}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS + ITM_METRICS}}\n"
                "comparison['by_category'] = {{category: {{'n': frozen_fields[category]['n'], 'frozen': [round(frozen_fields[category]['i2t_recall_at_1'], 3), round(frozen_fields[category]['t2i_recall_at_1'], 3)], 'adapted': [round(adapted_fields[category]['i2t_recall_at_1'], 3), round(adapted_fields[category]['t2i_recall_at_1'], 3)]}} for category in frozen_fields}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'categories': categories,\n"
                "    'gallery': {{'photographs': len(test_records), 'captions': len(gallery_texts)}},\n"
                "    'rerank_top_k': RERANK_TOP_K,\n"
                "    'baselines': {{'chance': baseline_chance, 'colour_keyword': baseline_neighbour}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['rsum'] > frozen_test['rsum']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Re-score the drawn scenes, export the adapter and reload it\n\n"
                "The 3×3 grid from Section 5 is scored again by the adapted model — drawings, a different image family from the "
                "photographs it was tuned on, so this is a small look at what the adaptation did *outside* its corpus (the "
                "build record's grids are in the model card; a changed ranking here is a finding to record, not a failure) — "
                "and reported with the per-image `evaluation_report` (`sample-sanity`). Both grids are written as CSV.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the text encoder's last two blocks, the two projections and "
                "the ITM head, about 77 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, "
                "the base model id and revision, the digest of the base `pytorch_model.bin`, the tensor names, the file size "
                "and SHA-256, the training configuration and the epoch history (OUT8). `BlipItmPipeline.from_artifact` "
                "re-verifies the base snapshot, checks the artifact manifest, its digest and its exact tensor set **before** "
                "deserialising, refuses any tensor outside the text encoder blocks, the projections and the ITM head, and "
                "overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). "
                "The cell asserts identical cosine grids on eight test photographs against their query captions (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "adapted_result = pipe.score(scene_images, scene_texts)\n"
                "adapted_scene = evaluation_report(adapted_result, correct_text_per_image, sample_kind='synthetic')\n"
                "print('adapted itm_probability [image][text]:')\n"
                "print(adapted_result['itm_probability'])\n"
                "print({{'scene_after_adaptation': {{m['id']: round(m['value'], 3) for m in adapted_scene['metrics']}}, 'verdict': adapted_scene['verdict']}})\n"
                "with open('outputs/{stem}_scores.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'text', 'frozen_itm_probability', 'frozen_cosine', 'adapted_itm_probability', 'adapted_cosine'])\n"
                "    for i, name in enumerate(scene_names):\n"
                "        for j, text in enumerate(scene_texts):\n"
                "            writer.writerow([name, text, f\"{{result['itm_probability'][i, j]:.4f}}\", f\"{{result['cosine'][i, j]:.4f}}\", f\"{{adapted_result['itm_probability'][i, j]:.4f}}\", f\"{{adapted_result['cosine'][i, j]:.4f}}\"])\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = BlipItmPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "parity_images = []\n"
                "for record in test_records[:8]:\n"
                "    with Image.open(record['image']) as photo:\n"
                "        photo.load()\n"
                "        parity_images.append(photo.copy())\n"
                "parity_texts = [query_caption(r) for r in test_records[:8]]\n"
                "before = pipe.score(parity_images, parity_texts)['cosine'].round(4)\n"
                "after = reloaded.score(parity_images, parity_texts)['cosine'].round(4)\n"
                "parity = {{'identical_scores': int((before == after).sum()), 'of': int(before.size)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_scores'] == parity['of']\n\n"
                "weight_entry = next(entry for entry in MANIFEST['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'pytorch_model.bin pickle, digest-verified, weights_only=True', 'weight_sha256': weight_entry['sha256'], 'hosted_tf_weight_file_not_loaded': HOSTED_TF_WEIGHT_FILE}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'text_columns': list(CORPUS_TEXT_COLUMNS), 'file': CORPUS_FILE, 'pinned_images': len(IMAGE_PINS), 'gallery_row_group': GALLERY_ROW_GROUP}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'scenes': {{'names': scene_names, 'sizes': [list(image.size) for image in scene_images], 'rgb_sha256': scene_digests, 'texts': scene_texts, 'correct_text_per_image': correct_text_per_image}}, 'frozen_grid': {{'itm_probability': result['itm_probability'].tolist(), 'cosine': result['cosine'].tolist()}}, 'adapted_grid': {{'itm_probability': adapted_result['itm_probability'].tolist(), 'cosine': adapted_result['cosine'].tolist()}}, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene}},\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model is already a competent retriever on photographs it never saw — far above the two non-neural "
        "baselines, and at the ceiling of a 70-photograph gallery — and a bounded fine-tuning of the text encoder's last "
        "two blocks, the ITC projections and the ITM head on 208 VizWiz photographs moves recall over a 391-photograph "
        "gallery by a few points (rsum 5.02 → 5.16 in the build record: image → text R@1 0.742 → 0.791, text → image R@1 "
        "0.652 → 0.673, ITM-reranked image → text 0.803 → 0.844) with a 77 MB adapter that reloads to identical scores. That "
        "is the claim: the adaptation contract works end to end on a real out-of-distribution retrieval corpus, and the "
        "numbers it produces are read on six recalls, the ITM re-ranking and per category against two non-neural baselines "
        "and the frozen model rather than in isolation. The gallery size matters more than the adaptation does — the same "
        "frozen model scores rsum 5.72 on 70 photographs and 5.02 on 391 — and the notebook says so.\n\n"
        "The gallery is 391 photographs from two row groups of one shard of one corpus, the validation split that picks "
        "the epoch is 40 photographs whose rsum saturates, the metrics are recall@k over a gallery (own implementations of "
        "the COCO retrieval protocol; none a human judgement), and the ITM pair accuracy did not move. So a gain here says "
        "the contract works, not that the adapted model is better on your photographs, that it reads the labels VizWiz "
        "captions transcribe, or that a match probability is trustworthy — it still ranks something first for every "
        "query, and it can be wrong confidently. Fine-tuning on a narrow corpus can also erode the model elsewhere; the "
        "drawn scenes re-scored in Section 9 are one 3×3 grid of evidence about that, not a measurement.\n\n"
        "Three things to carry to real data. **Gallery first:** recall@k is only meaningful against a gallery of the "
        "deployment's own size — measure the frozen model over your gallery before any adapted number. **Leakage:** keep "
        "every record on an image in one split (the contract does this) and split by photographer or session when your "
        "images come from few sources, never at random over near-duplicate frames. **Negatives:** the in-batch negatives "
        "are what teach the ITM head; a corpus of near-identical photographs gives it hard negatives, a corpus of "
        "unrelated ones gives it easy ones and little to learn.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify the captions and "
        "photographs of a real captioning corpus, validate the demonstrated dataset contract without leakage, execute the "
        "inference contract and a bounded fine-tuning, evaluate against two trivial baselines and the frozen model on an "
        "image-disjoint gallery, and emit the shown machine-readable artifacts — without the repository being reachable. It "
        "does **not** establish benchmark superiority, retrieval quality on any other population or gallery, or production "
        "fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `RERANK_TOP_K = 10` and compare the re-ranked "
        "recalls with the cost; set `TRAINABLE_TEXT_LAYERS = 4` and `LEARNING_RATE = 5e-5` and read the build record's "
        "counter-example against your own run; raise `EPOCHS` and watch the saturated validation rsum pick the epoch; or "
        "bring your own photographs through BYOD and read the chance baseline over *your* gallery before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/blip-itm-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/blip-itm-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/blip-itm-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model (Salesforce, BSD-3-Clause): https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/salesforce/BLIP\n"
        "- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Li et al., 2022): https://arxiv.org/abs/2201.12086\n"
        "- Captioning Images Taken by People Who Are Blind (Gurari et al., ECCV 2020; VizWiz-Captions, CC BY 4.0): https://arxiv.org/abs/2002.08565 — data: https://vizwiz.org/tasks-and-datasets/image-captioning/\n"
        "- Align before Fuse: Vision and Language Representation Learning with Momentum Distillation — the ITC + ITM objectives with hard-negative mining (Li et al., 2021): https://arxiv.org/abs/2107.07651\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
