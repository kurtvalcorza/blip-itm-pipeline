"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "blip_itm_pipeline",
    "repo_name": "blip-itm-pipeline",
    "stem": "blip_itm",
    "notebook_name": "blip_itm_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "BlipItmPipeline",
    "weights_key": "blip-itm-base-coco",
    "runtime_imports": ["torch", "transformers"],
    "title": "BLIP ITM-base COCO — DIMER image-text matching and retrieval tutorial (standalone)",
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
    "capability": "Image-text matching and retrieval — a grid of 1–16 images × 1–16 captions → an ITM match probability and an ITC cosine similarity per pair, and the captions ranked per image — using the pinned `Salesforce/blip-itm-base-coco` weights",
    "intro": (
        "At inference the BLIP retrieval model (a ViT-B/16 image encoder at 384×384, a BERT-style text encoder, and an "
        "image-grounded text encoder whose cross-attention fuses each caption with the image; about 224M parameters, "
        "pretrained on 129M image–text pairs with captioning-and-filtering bootstrapping and fine-tuned on COCO for "
        "retrieval) scores every image–caption pair twice: the **ITC head** compares the two projected embeddings by cosine "
        "similarity (the fast dual-encoder score used for retrieval), and the **ITM head** classifies the fused pair as "
        "match/no-match (the slower cross-attention score used for re-ranking); the carried module softmaxes the ITM logits "
        "into a match probability and ranks the captions per image by it. **No adaptation occurs:** no training, "
        "fine-tuning, in-context conditioning, or preprocessing fitting happens in this notebook — the upstream checkpoint "
        "supplies the weights, processor and tokenizer, and the carried module adds snapshot verification, the input "
        "contract (image side ceilings, 1–16 images, 1–16 distinct captions up to 256 characters), a fixed output contract "
        "(grids indexed `[image][text]`), and the `recall_at_1`, `validate_inputs` and `evaluation_report` helpers. "
        "**Weight-format note:** upstream hosts no SafeTensors at the pinned revision; the carried module executes the "
        "digest-pinned `pytorch_model.bin` (a pickle, deserialised with `weights_only=True` after its SHA-256 is checked), "
        "while the `tf_model.h5` upstream also hosts is DIMER's upload artifact and is never loaded here. The default "
        "sample is a 3×3 grid of cartoon scenes drawn in code and three authored captions, so recall@1 in both directions "
        "is demonstration (plumbing) evidence for one tiny grid, not a COCO retrieval benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision (a pickle checkpoint, and why that matters), draw three synthetic scenes and "
        "write their captions (or upload your own images and type captions) and validate them into an input manifest, run "
        "the supported task over the whole grid, read the two scores correctly (an uncalibrated match probability and a "
        "cosine similarity, neither an abstention), exercise an optional BYOD path, produce an evaluation report that is "
        "`sample-sanity` with recall@1 in both directions only when the image–caption correspondence is known and "
        "`not-measurable` otherwise, and export the grids, a ranked contact sheet and provenance."
    ),
    "exclusions": (
        "Captioning or question answering (separate checkpoints), retrieval over a corpus of thousands (the grid is capped "
        "at 16×16 and every pair is a full forward pass — an index of precomputed embeddings is a deployment's own build), "
        "recall@5/@10 or median rank (a 3×3 grid cannot express them), batch throughput, evaluation on COCO or Flickr30k "
        "(not bundled; only drawn scenes are scored here), and any training. The model was fine-tuned on COCO photographs "
        "with human captions; flat drawings, documents, non-English captions and captions that describe attributes the "
        "model ignores are outside what this notebook measures, and a high match probability carries no signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 3.9 s to load and 3.1 s for the nine pairs of the 3×3 grid in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the 895 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python, NumPy and PIL; what a dual-encoder cosine score and a cross-attention match score are and why they differ; why a pickle checkpoint needs a digest check before `torch.load`; what recall@1 on a 3×3 grid does and does not show.",
        "- **Data:** the default sample is three deterministic cartoon scenes drawn in code with Pillow (a house with a tree and the sun; a beach with a sailboat; two fruits on a table — no text rendering, so their digests are stable across Pillow builds) and three authored captions, one per scene, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one or more images decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, sides between 16 and 4096 px, plus your own captions typed into the form field; the correspondence is unknown for uploads, so their report is `not-measurable`. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic scenes or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own references: three flat cartoon scenes — a red "
                "house with a tree, a white ball and the sun on grass under a blue sky; a beach with sea, sand, a red "
                "sailboat and the sun; a red apple and an orange on a wooden table — are drawn with Pillow (640×480, "
                "640×480, 480×480), the same drawings the repository's smoke run used, and three captions are authored, one "
                "per scene, in that order. The diagonal of the grid is therefore the known correspondence and the reference "
                "for the recall@1 sanity check later. They are not a labelled dataset, so nothing here is a COCO retrieval "
                "measurement; the smoke run recorded a weak diagonal for the fruit scene (ITM 0.28 against its own caption) "
                "even though it still ranked first. The image digests are printed for the record. BYOD is optional and "
                "disabled by default; when enabled, upload one or more images and type your captions (one per line) — no "
                "correspondence is known for them, so the evaluation report will be `not-measurable`.\n\n"
                "The caption set is a **caller-owned request parameter**: both scores rank only what you supply, and a "
                "caption set with nothing that fits an image still yields a highest-scoring caption. Nothing is validated "
                "in this cell — the next section hands the images and captions to the pipeline's own validation stage, "
                "which is the only checker. Look for one dictionary per image naming the sample kind, size and digest, plus "
                "the captions and the grid size."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "byod_captions = 'a dog on a beach\\na plate of food'  # @param {{type:\"string\"}}\n\n\n"
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
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    names, images = [], []\n"
                "    for name, data in uploaded.items():\n"
                "        image = Image.open(io.BytesIO(data))\n"
                "        image.load()\n"
                "        names.append(name)\n"
                "        images.append(image)\n"
                "    texts = [line.strip() for line in byod_captions.splitlines() if line.strip()]\n"
                "    correct_text_per_image = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic drawings: no randomness and no text rendering, so no seed is needed and the digests are stable.\n"
                "    samples = synthetic_scenes()\n"
                "    names = [name for name, _, _ in samples]\n"
                "    images = [image for _, image, _ in samples]\n"
                "    texts = [caption for _, _, caption in samples]\n"
                "    correct_text_per_image = list(range(len(samples)))  # caption i describes image i\n"
                "    sample_kind = 'synthetic'\n\n"
                "digests = {{name: hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() for name, image in zip(names, images)}}\n"
                "for name, image in zip(names, images):\n"
                "    print({{'sample_kind': sample_kind, 'name': name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': digests[name]}})\n"
                "print({{'texts': texts, 'grid': [len(images), len(texts)], 'has_correspondence': correct_text_per_image is not None}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `score` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, at most `MAX_IMAGES` images, 1..`MAX_TEXTS` "
                "distinct non-empty captions of at most `MAX_TEXT_CHARS` characters (whitespace collapsed) — and returns an "
                "**input manifest** naming the schema (including the 384×384 resize that does not preserve aspect ratio and "
                "the two scores), each input's observed mode and size, the checked captions, the number of pairs and the "
                "verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks "
                "like, the cell also validates a duplicated caption and records the pipeline's own error message as a finding. "
                "Inside the pipeline each image is converted to RGB and resized to `IMAGE_SIZE`×`IMAGE_SIZE`; nothing else is "
                "dropped or altered. The pipeline cannot tell whether any caption describes any image: that contract is the "
                "caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'IMAGE_SIZE': IMAGE_SIZE, 'MAX_IMAGES': MAX_IMAGES, 'MAX_TEXTS': MAX_TEXTS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS}}}})\n"
                "input_manifest = validate_inputs(images, texts, names=names)\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(images, [texts[0], '  ' + texts[0] + ' '])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'duplicate-caption-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Score the grid and read the output correctly\n\n"
                "`score` returns three `[image][text]` grids — `itm_probability` (the softmax of the ITM head's match/no-match "
                "logits), `itm_logit_match` (the raw match logit) and `cosine` (the ITC similarity of the projected "
                "embeddings) — plus `rankings` (per image, the captions ordered by ITM probability), the checked captions, "
                "the image sizes and the model identity. **Neither score is calibrated and neither abstains**: the ITM "
                "probability is a per-pair classifier output that is not comparable to a human match rate, the cosine is "
                "comparable only within a row or a column, and a caption set with nothing that fits still produces a "
                "highest-scoring caption. Both are deterministic on a fixed device and dtype; CUDA kernels can shift them "
                "slightly, so GPU and CPU rankings need not agree on close pairs. Every pair costs one fused forward pass and "
                "one dual-encoder pass (about 0.35 s per pair on the reference CPU), so the grid scales as images × captions. "
                "As recorded in the model card, the repository's CPU smoke on this same grid put every image's own caption "
                "first by both scores — ITM 0.998 for the house, 0.634 for the beach and only 0.278 for the fruit scene — and "
                "gave a blank white image ITM 0.04–0.07 against all three captions: a low match probability is not evidence "
                "of an empty image, and a high one is not evidence of a correct description."
            ),
            "code": (
                "import time\n\n"
                "t0 = time.time()\n"
                "result = pipe.score(images, texts)\n"
                "elapsed = round(time.time() - t0, 2)\n"
                "print({{'device': pipe.device, 'dtype': pipe.dtype, 'seconds': elapsed, 'pairs': result['n_images'] * len(result['texts'])}})\n"
                "np.set_printoptions(precision=3, suppress=True)\n"
                "print('itm_probability [image][text]:')\n"
                "print(result['itm_probability'])\n"
                "print('cosine [image][text]:')\n"
                "print(result['cosine'])\n"
                "for name, ranking in zip(names, result['rankings']):\n"
                "    print(f\"{{name}} -> \" + '; '.join(f\"{{entry['text']!r}} itm={{entry['itm_probability']:.3f}} cos={{entry['cosine']:.3f}}\" for entry in ranking))"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No retrieval "
                "quality is reported by default: recall@k needs a captioned image set from the deployment domain with "
                "thousands of candidates, and this repository ships none (COCO and Flickr30k are not bundled). When the "
                "image–caption correspondence is supplied the report carries image-to-text and text-to-image `recall_at_1` "
                "for both the ITM probability and the cosine grid (the text-to-image direction only when the grid is square "
                "and one-to-one), the chance baselines, and the verdict `sample-sanity`. On the synthetic path the "
                "correspondence is the diagonal **you drew and wrote yourself**, so a perfect recall@1 on nine pairs proves "
                "only that the input contract, preprocessing, both heads and the ranking round-trip. On BYOD no "
                "correspondence is known, the verdict is `not-measurable`, and the report states what would make the task "
                "measurable. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, correct_text_per_image, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k not in ('metrics', 'baselines')}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"{{metric['id']:44}} {{metric['value']:.3f}}  ({{metric['estimation']}})\")\n"
                "for baseline in report['baselines']:\n"
                "    print(f\"{{baseline['id']:44}} {{baseline['value']:.3f}}  ({{baseline['note']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No image-caption correspondence is known for this grid, so nothing is scored; read the rankings against the images yourself.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the three grids, the rankings, the captions, the evaluation report, the input "
                "manifest, the sample identities, digests and correspondence, the notebook's source (repository, revision, "
                "embedded module digest, generator), the model identifier, the immutable model revision, the model licence, "
                "the executed weight file and the hosted TensorFlow file it is not, and the runtime identity (Python, `torch`, "
                "`transformers`, device). The pair scores are also written as CSV with explicit `image`, `text`, "
                "`itm_probability`, `itm_logit_match`, `cosine` columns, and a contact-sheet PNG shows each image with its "
                "top-ranked caption and ITM probability for visual inspection — a supplement to, not a replacement for, the "
                "machine-readable files. No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "thumb_w, thumb_h, panel_h = 320, 240, 44\n"
                "sheet = Image.new('RGB', (thumb_w * len(images), thumb_h + panel_h), 'white')\n"
                "draw = ImageDraw.Draw(sheet)\n"
                "panel_font = ImageFont.load_default(size=13)\n"
                "for index, (image, ranking) in enumerate(zip(images, result['rankings'])):\n"
                "    thumb = image.convert('RGB').copy()\n"
                "    thumb.thumbnail((thumb_w, thumb_h))\n"
                "    sheet.paste(thumb, (index * thumb_w + (thumb_w - thumb.width) // 2, (thumb_h - thumb.height) // 2))\n"
                "    top = ranking[0]\n"
                "    draw.text((index * thumb_w + 6, thumb_h + 6), f\"itm {{top['itm_probability']:.2f}}: {{top['text'][:44]}}\", fill=(40, 90, 220), font=panel_font)\n"
                "sheet.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'itm_probability': result['itm_probability'].tolist(),\n"
                "    'itm_logit_match': result['itm_logit_match'].tolist(),\n"
                "    'cosine': result['cosine'].tolist(),\n"
                "    'rankings': result['rankings'],\n"
                "    'texts': result['texts'],\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'names': names, 'sizes': [list(image.size) for image in images], 'rgb_sha256': digests, 'correct_text_per_image': correct_text_per_image}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'executed_weight_file': WEIGHT_FILE,\n"
                "    'hosted_tf_weight_file_not_loaded': HOSTED_TF_WEIGHT_FILE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_scores.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'text', 'itm_probability', 'itm_logit_match', 'cosine'])\n"
                "    for i, name in enumerate(names):\n"
                "        for j, text in enumerate(result['texts']):\n"
                "            writer.writerow([name, text, f\"{{result['itm_probability'][i, j]:.6f}}\", f\"{{result['itm_logit_match'][i, j]:.4f}}\", f\"{{result['cosine'][i, j]:.6f}}\"])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The scores are a match classifier's probability and an embedding cosine for pairs you supplied; nothing in the "
        "output says whether any caption truly describes any image, neither score is calibrated, and the model ranks "
        "every grid — including a blank image — without abstaining. On the drawn scenes the recall@1 values in the "
        "evaluation report compare the rankings with a correspondence you drew and wrote yourself and the verdict is "
        "`sample-sanity`, which proves only that the input contract, preprocessing, both heads and the ranking work (the "
        "repository's smoke run scored 1.0 in both directions by both scores on nine pairs, with the fruit scene's own "
        "caption winning at only 0.28); it says nothing about photographs, near-duplicate captions, attribute-level "
        "distinctions (`a red house` 0.64 vs `a blue house` 0.005 in the smoke run, but `a house` 0.30), non-English "
        "captions, or retrieval over thousands of candidates, and a BYOD result is a single-grid observation with the "
        "verdict `not-measurable`. **The caption set is part of the request**: the highest-scoring caption for an image is "
        "the best of what you offered, not a description. The pipeline provides no captioning, no corpus index, no "
        "abstention, no benchmark evaluation and no training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model (a pickle checkpoint loaded with `weights_only=True` only after its digest "
        "matched), validate the demonstrated request, execute the public pipeline path, and emit the shown machine-readable "
        "outputs in the tested runtime — without the repository being reachable. It does **not** establish benchmark "
        "superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen "
        "domain.\n\n"
        "**Next experiments:** add a fourth caption that fits none of the scenes and watch which image claims it; replace the "
        "fruit caption with `an apple and a banana` and compare the ITM probability with the cosine; enable `USE_BYOD` with "
        "photographs you know, type their captions, then pass the true correspondence to `evaluation_report` to see the "
        "verdict switch to `sample-sanity`.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/blip-itm-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/blip-itm-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/blip-itm-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/salesforce/BLIP\n"
        "- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Li et al., 2022): https://arxiv.org/abs/2201.12086\n"
        "- Microsoft COCO Captions: Data Collection and Evaluation Server (Chen et al., 2015): https://arxiv.org/abs/1504.00325"
    ),
}
