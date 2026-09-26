# DIMER Workshop Specification: Vision-Language Retrieval and Cross-Modal Reranking

**Status:** Proposed  
**Notebook specification:** DIMER `NOTEBOOK_SPEC` **2.1**  
**Notebook profile:** `MULTI-CAPABILITY`  
**Pedagogical mode:** `WORKSHOP`  
**Proposed filename:** `DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb`  
**Canonical runtime:** NVIDIA Tesla T4 or equivalent  
**Canonical execution:** standalone, credential-free, top-to-bottom `Run all`

---

# 1. Purpose

This workshop demonstrates how modern vision-language systems retrieve images from text and retrieve text from images.

It compares three first-stage retrieval representations:

1. **SigLIP 2 Base P16-224**
2. **SigLIP v1 Base P16-256**
3. **BLIP ITC**

and then adds:

4. **BLIP ITM cross-modal reranking**

The principal application workflow is:

```text
captioned image corpus
→ compute image embeddings once
→ compute text embeddings once
→ cosine-similarity retrieval
→ top-K shortlist
→ BLIP ITM pairwise reranking
→ retrieval evaluation
```

The workshop asks:

> How much retrieval quality can we obtain from scalable dual-encoder embeddings, and when is it worth paying the additional computational cost of a fused image-text reranker?

---

# 2. Core distinction

The workshop MUST distinguish:

### Retrieval

```text
query
→ rank existing candidate items
```

from:

### Classification

```text
image
→ choose from predefined labels
```

and:

### Caption generation

```text
image
→ generate new text
```

No text generation occurs in this notebook.

---

# 3. Notebook profile

Declare:

**Profile:** `MULTI-CAPABILITY`  
**Mode:** `WORKSHOP`

Capabilities:

### Capability A

Dual-encoder image-text retrieval.

### Capability B

Cross-modal BLIP ITM reranking.

### Capability C

Optional retrieval-domain adaptation of BLIP.

This is more accurate than `E2E` because the SigLIP models' current DIMER adaptation contracts are class-prompt adaptation workflows rather than general image-caption retrieval fine-tuning.

The notebook MUST NOT silently repurpose those adapters.

---

# 4. Execution tiers

```python
WORKSHOP_TIER = "STANDARD"  # @param ["STANDARD", "FULL"]
```

## STANDARD

Runs:

- SigLIP 2 frozen retrieval
- SigLIP v1 frozen retrieval
- BLIP ITC frozen retrieval
- common retrieval metrics
- BLIP ITM reranking
- gallery-size sensitivity
- hard-negative analysis
- index/latency/storage comparison
- BYOD retrieval

## FULL

Additionally runs:

- BLIP ITC + ITM bounded adaptation
- validation-rsum selection
- fresh adapter export/reload
- adapted BLIP coarse retrieval
- adapted BLIP reranking
- adapted BLIP reranking of SigLIP 2 candidates

`STANDARD` SHOULD be the default release path.

---

# 5. Model A — SigLIP 2

**Model:**

`google/siglip2-base-patch16-224`

Revision:

```text
5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2
```

License:

**Apache-2.0**

Weights:

```text
model.safetensors
1,500,800,904 bytes
SHA-256:
612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b
```

Parameters:

```text
375,187,970
```

---

# 6. SigLIP 2 representation

Architecture:

```text
image
→ ViT P16 / 224
→ image projection
→ 768-d embedding

text
→ multilingual text Transformer
→ text projection
→ 768-d embedding
```

Both vectors are L2-normalized.

Retrieval score:

\[
s(i,t)=\hat v_i^\top \hat t
\]

which is cosine similarity.

---

# 7. SigLIP 2 role

Use SigLIP 2 as the principal **scalable dual-encoder retriever**.

It demonstrates:

- precomputed image embeddings;
- precomputed caption embeddings;
- text → image retrieval;
- image → text retrieval;
- vector-space nearest-neighbor search.

It does **not** perform fused pairwise reasoning after the embeddings have been produced.

---

# 8. Model B — SigLIP v1

**Model:**

`google/siglip-base-patch16-256`

Revision:

```text
b078df89e446d623010d890864d4207fe6399f61
```

License:

**Apache-2.0**

Weights:

```text
model.safetensors
812,856,640 bytes
SHA-256:
f0cee7c815135c44a515eff72ab3040499744920442bc25567cd04efc93f8f65
```

Parameters:

```text
203,202,050
```

---

# 9. SigLIP v1 representation

Input resolution:

```text
256 × 256
```

Embedding dimension:

```text
768
```

English SentencePiece vocabulary:

approximately:

```text
32k tokens
```

The model is included primarily as the **v1 reference** beside SigLIP 2.

---

# 10. Why include both SigLIP generations

The workshop can ask:

> Does the newer, substantially larger SigLIP 2 produce better retrieval rankings on this real caption corpus than the smaller original SigLIP?

The notebook MUST NOT assume that model generation or parameter count determines the result.

Both receive exactly the same:

- photographs;
- captions;
- retrieval gallery;
- ranking evaluator.

---

# 11. Model C — BLIP ITM Base COCO

**Model:**

`Salesforce/blip-itm-base-coco`

Revision:

```text
bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c
```

Parameters:

```text
223,744,258
```

---

# 12. BLIP weights

Upstream PyTorch checkpoint:

```text
pytorch_model.bin
895,139,697 bytes
SHA-256:
017fb3e7f4e125f13a8a4717f1402dbe0d0bb877474b4a203db13a4447b0227f
```

The upstream revision provides no SafeTensors base checkpoint.

Therefore the DIMER trust boundary remains:

```text
download pinned bytes
→ verify size + SHA-256
→ load with weights_only=True
→ trust_remote_code=False
```

The TensorFlow `.h5` port is not used.

---

# 13. BLIP architecture

BLIP provides two useful scoring modes.

### ITC — Image-Text Contrastive representation

```text
ViT-B/16 image encoder
→ vision projection

BERT-style text encoder
→ text projection
```

Projection dimension:

```text
256
```

Score:

```text
cosine(image_embedding, text_embedding)
```

This behaves like a dual encoder.

### ITM — Image-Text Matching

```text
image features
+
text tokens
→ cross-attending text encoder
→ fused CLS representation
→ binary match head
```

This requires one joint image-text forward pass per pair.

---

# 14. Why BLIP is especially useful here

BLIP allows us to compare:

```text
BLIP ITC
```

against:

```text
BLIP ITM
```

inside one model family.

That isolates the conceptual difference between:

> **independent embeddings**

and:

> **pairwise cross-modal reasoning**

better than comparing unrelated models alone.

---

# 15. Retrieval architecture taxonomy

The workshop SHOULD introduce:

## Dual encoder

```text
image → embedding
text  → embedding

similarity = dot product
```

Advantages:

- embeddings computed once;
- cheap matrix search;
- indexable;
- practical for large corpora.

Examples:

- SigLIP v1
- SigLIP 2
- BLIP ITC

## Cross encoder / fused matcher

```text
(image, text)
→ joint multimodal model
→ pair score
```

Advantages:

- richer pairwise reasoning.

Disadvantages:

- cannot precompute one universal pair score;
- computational cost grows with number of pairs.

Example:

- BLIP ITM

---

# 16. Canonical dataset

Use the existing digest-pinned:

**VizWiz-Captions**

Source:

`mm-eval/VizWiz-Captions`

Revision:

```text
c4a6d897836e7885d0095134f92d392e4e770539
```

License:

**CC BY 4.0**

---

# 17. Source shard

Pinned shard:

```text
data/val-00004-of-00005.parquet
```

Bytes:

```text
392,245,504
```

SHA-256:

```text
4492465a41d32b3c12b8b7b6a0cf7e0a0e202a5b825b006ca0c85dcdf24efd3e
```

Caption columns SHOULD be read without downloading unnecessary image bytes where possible.

Image bytes remain digest-pinned individually through the existing BLIP sample contract.

---

# 18. Why VizWiz-Captions

This is much better than the iNaturalist class-prompt sample for a retrieval workshop because each photograph has **natural human-written descriptions**.

The task becomes:

```text
photograph
↔
one or more genuine captions
```

rather than:

```text
bird photograph
↔
species-name prompt
```

That exercises actual cross-modal retrieval.

---

# 19. Canonical split

Reuse the existing BLIP retrieval split.

### Training

```text
208 photographs
```

### Validation

```text
40 photographs
```

### Test core

```text
70 photographs
```

### Additional gallery-only photographs

```text
321
```

### Full independent test gallery

```text
391 photographs
1,737 captions
```

The additional 321 gallery photographs increase retrieval difficulty without entering training or validation.

---

# 20. Why the large test gallery matters

Retrieval metrics depend strongly on gallery size.

The same model will generally appear stronger when choosing among:

```text
70 candidates
```

than:

```text
391 candidates
```

Therefore the workshop MUST explicitly measure gallery-size sensitivity.

---

# 21. Canonical record

```text
{
  "id": str,
  "image": PIL.Image,
  "captions": [
    "...",
    "...",
    ...
  ],
  "category": "text" | "no-text"
}
```

The first caption SHALL additionally serve as the canonical:

```text
query_caption
```

for bounded ITM text→image reranking.

All captions remain valid positives for normal retrieval metrics.

---

# 22. Split safety

The split MUST operate by photograph.

All captions belonging to one photograph stay in the same split.

The notebook MUST check:

- unique photograph IDs;
- decoded-image digest disjointness;
- no photograph shared across train/validation/test;
- no exact image bytes shared across roles.

Caption duplication across unrelated photographs SHOULD be reported because generic descriptions may legitimately repeat.

---

# 23. Retrieval direction A — text to image

Query:

```text
caption
```

Gallery:

```text
all photographs
```

Success at rank K:

> the caption's owning photograph occurs in the first K returned images.

Metrics:

```text
T2I R@1
T2I R@5
T2I R@10
median rank
```

---

# 24. Retrieval direction B — image to text

Query:

```text
photograph
```

Gallery:

```text
all captions
```

A photograph may have several correct captions.

Success at K:

> at least one caption belonging to the query photograph appears in the first K results.

Metrics:

```text
I2T R@1
I2T R@5
I2T R@10
median rank
```

---

# 25. Primary summary — rsum

Define:

\[
R_{sum}
=
I2T@1+I2T@5+I2T@10+
T2I@1+T2I@5+T2I@10
\]

Range:

```text
0–6
```

`rsum` is a convenient summary.

It MUST NOT replace the directional recalls in the principal result table.

---

# 26. Optional MRR

The notebook MAY additionally report:

**mean reciprocal rank**

for each direction.

This is useful pedagogically but is not required for parity with the current BLIP carrier.

Canonical headline metrics remain Recall@K and median rank.

---

# 27. Common evaluator

Use one notebook-owned evaluator for every coarse retriever.

Input:

```text
similarity matrix
caption owner IDs
image IDs
```

The evaluator MUST NOT depend on:

- SigLIP logits;
- BLIP-specific probability values;
- model-specific calibration.

Only ranking matters.

---

# 28. Coarse retriever A — SigLIP 2

Compute:

```text
391 image embeddings
1,737 text embeddings
```

L2-normalize.

Then:

\[
S = V T^\top
\]

where:

```text
V ∈ R^(391×768)
T ∈ R^(1737×768)
```

No full model pass is required after embeddings exist.

---

# 29. Coarse retriever B — SigLIP v1

Same procedure:

```text
V ∈ R^(391×768)
T ∈ R^(1737×768)
```

with its own processor and encoders.

Embeddings from SigLIP v1 and SigLIP 2 MUST NOT be mixed.

Their coordinate systems are unrelated.

---

# 30. Coarse retriever C — BLIP ITC

Compute BLIP's projected features:

```text
V ∈ R^(391×256)
T ∈ R^(1737×256)
```

then cosine similarity.

This allows a fair common ranking evaluation of BLIP as a first-stage dual encoder.

---

# 31. BLIP ITM reranking

A full cross-product would require roughly:

\[
391 × 1737
\]

fused image-text evaluations.

That is unnecessary and defeats the purpose of a two-stage search system.

Instead:

```text
dual-encoder retrieval
→ top K candidates
→ BLIP ITM scores only those pairs
→ re-sort shortlist
```

---

# 32. Canonical reranking depth

Default:

```text
RERANK_TOP_K = 5
```

because the existing qualified BLIP carrier already evaluates this depth.

A validation-only experiment MAY compare:

```text
5
10
20
```

The final K must be frozen before test.

---

# 33. Image→text reranking

For each image:

1. retrieve top K captions from the coarse model;
2. compute BLIP ITM score for those K image-caption pairs;
3. rerank those candidates;
4. measure whether an owned caption is ranked first.

Headline:

```text
ITM-reranked I2T R@1
```

---

# 34. Text→image reranking

Reranking all 1,737 caption queries would be unnecessarily expensive.

Use the canonical first caption from every photograph:

```text
391 query captions
```

For each:

1. retrieve top K photographs;
2. apply BLIP ITM;
3. rerank;
4. check whether its owning photograph is first.

Headline:

```text
ITM-reranked T2I R@1
```

The normal dual-encoder T2I Recall@K still uses **all 1,737 captions**.

---

# 35. Cross-model reranking experiment

Apply the same BLIP ITM reranker to shortlists produced by:

1. SigLIP 2
2. SigLIP v1
3. BLIP ITC

This yields a particularly useful application question:

> Does the quality of the first-stage retriever still matter after the same reranker is applied?

---

# 36. Principal reranking table

| First stage | Coarse I2T R@1 | BLIP-reranked I2T R@1 | Δ | Pair evaluations |
|---|---:|---:|---:|---:|
| SigLIP 2 | | | | |
| SigLIP v1 | | | | |
| BLIP ITC | | | | |

And separately:

| First stage | Coarse canonical T2I R@1 | BLIP-reranked T2I R@1 | Δ |
|---|---:|---:|---:|
| SigLIP 2 | | | |
| SigLIP v1 | | | |
| BLIP ITC | | | |

Do not combine these into a winner score.

---

# 37. BLIP ITM pair accuracy

For every photograph:

### Positive

its canonical query caption.

### Negative

highest-scoring wrong caption according to the selected coarse retriever.

Ask BLIP ITM:

> Which pair looks like a match?

Report:

```text
ITM pair accuracy
```

This tests the reranker's ability to resolve **hard negatives**, not random negatives.

---

# 38. Non-neural baseline A — chance

Calculate analytical random-retrieval expectations for the exact gallery.

For T2I:

\[
R@K \approx K/N_{images}
\]

For I2T, account for the number of valid captions belonging to each image.

The exact baseline SHOULD be computed from gallery ownership, not hard-coded.

---

# 39. Non-neural baseline B — colour + keyword

Reuse the current BLIP carrier's concept:

- 3×3 mean-colour image descriptor;
- simple caption token overlap/keyword representation.

This baseline is intentionally weak but data-aware.

It should pass through the same retrieval evaluator.

---

# 40. Gallery-size sensitivity

After embeddings are computed once, evaluate deterministic nested galleries:

```text
70 images
128 images
256 images
391 images
```
with their associated captions.

At minimum compare:

- SigLIP 2
- BLIP ITC

Optionally SigLIP v1 as well.

---

# 41. Gallery-size table

| Gallery images | Captions | SigLIP2 rsum | BLIP ITC rsum |
|---:|---:|---:|---:|
| 70 | | | |
| 128 | | | |
| 256 | | | |
| 391 | 1,737 | | |

Key lesson:

> Retrieval quality is a property of the model **and the candidate population**.

---

# 42. Why this matters

A claim such as:

> “R@1 = 90%”

has little meaning without knowing whether the model chose among:

```text
10
100
1,000
1,000,000
```

candidates.

The notebook MUST reinforce this.

---

# 43. Category analysis

The existing sample identifies:

```text
text
no-text
```

photograph categories.

Report R@1 by category.

This asks whether retrieval behaves differently when an image visibly contains text that captions may transcribe.

---

# 44. Caption-length analysis

Group test captions by token/word count:

```text
short
medium
long
```

Thresholds determined from the **training caption distribution**.

Report:

```text
T2I R@1
T2I median rank
```

for each group.

---

# 45. Generic-caption analysis

Some captions may be highly generic, e.g.:

```text
a blurry image
a bottle
a room
```

A caption shared semantically by many photographs is intrinsically harder to retrieve uniquely.

The workshop SHOULD calculate a simple caption-specificity statistic, such as inverse token frequency over training captions, and compare retrieval rank against it.

Diagnostic only.

---

# 46. Hard-negative analysis

For every query retrieve:

- best correct item;
- highest-ranked incorrect item.

Display examples where:

```text
wrong similarity > correct similarity
```

or where the correct item ranks unusually low.

Show the actual competing images/captions.

---

# 47. Model-disagreement gallery

Select deterministic test queries with:

1. largest SigLIP2 rank advantage over SigLIP v1;
2. largest SigLIP v1 advantage;
3. largest SigLIP2 advantage over BLIP ITC;
4. largest BLIP ITC advantage;
5. BLIP ITM fixes first-stage error;
6. BLIP ITM makes a correct top-1 worse.

Do not hand-pick examples.

---

# 48. No overall model winner

The workshop MUST NOT compute a composite model score.

Users should inspect:

- I2T recall;
- T2I recall;
- gallery-size behavior;
- embedding size;
- indexing cost;
- query latency;
- reranking gain;
- model weight footprint.

Different applications optimize different dimensions.

---

# 49. Embedding storage comparison

Report raw float32 storage.

### SigLIP

```text
768 × 4 = 3,072 bytes / item
```

### BLIP ITC

```text
256 × 4 = 1,024 bytes / item
```

For the 391-image gallery:

- both are trivial.

For an illustrative one-million-image index:

### SigLIP

approximately:

```text
3.07 GB
```

### BLIP ITC

approximately:

```text
1.02 GB
```

before index overhead.

These are arithmetic projections, not measured index sizes.

---

# 50. Retrieval computation

Measure separately:

### Corpus indexing

```text
image-embedding seconds
caption-embedding seconds
```

### Search

```text
matrix-similarity milliseconds/query
```

### Reranking

```text
BLIP ITM milliseconds/pair
top-K rerank seconds/query
```

This distinction is central to production retrieval design.

---

# 51. Query latency versus indexing latency

The notebook SHOULD explain:

> An expensive image encoder may be acceptable if every corpus image is encoded once offline.

At query time:

```text
text encoder
+
vector search
```

may be all that runs.

BLIP ITM reranking adds pairwise online cost.

---

# 52. Runtime memory strategy

Load models sequentially:

```text
SigLIP 2
→ embed corpus
→ save embeddings
→ unload

SigLIP v1
→ embed corpus
→ save embeddings
→ unload

BLIP
→ embed corpus
→ rerank
→ optional adaptation
→ unload
```

Do not keep all ~3.2 GB of base model weights resident simultaneously.

---

# 53. Embedding artifacts

Export:

```text
outputs/index/siglip2_images.npy
outputs/index/siglip2_texts.npy

outputs/index/siglip1_images.npy
outputs/index/siglip1_texts.npy

outputs/index/blip_itc_images.npy
outputs/index/blip_itc_texts.npy
```

Plus:

```text
image_ids.json
caption_ids.json
caption_owners.json
index_manifest.json
```

---

# 54. Index manifest

Record:

```text
model ID
model revision
weight digest
embedding dimension
normalization
dtype
item order
image IDs
caption IDs
embedding file bytes
embedding SHA-256
```

This prevents vectors from silently being paired with another model or item ordering.

---

# 55. Fresh index reconstruction check

Reload exported embeddings.

Verify:

```text
same matrix shape
same item IDs
same SHA-256
same top-K ranking
```

on several deterministic queries.

The embedding matrix itself is an artifact deserving provenance.

---

# 56. FULL — BLIP retrieval adaptation

BLIP already has a release-grade retrieval adaptation contract.

Train:

- last two fused text-encoder blocks;
- `vision_proj`;
- `text_proj`;
- `itm_head`.

Freeze:

- vision encoder;
- text embeddings;
- remaining text blocks.

---

# 57. BLIP trainable surface

Default:

```text
19,298,818
of
223,744,258 parameters
```

Adapter:

```text
58 tensors
approximately 77.2 MB
```

---

# 58. BLIP adaptation objective

Joint objective:

### ITC

image-text contrastive loss with:

```text
temperature = 0.07
```

Same-photograph captions are positives.

### ITM

binary image-text matching objective.

Hard negative examples are sampled according to ITC similarity.

Negatives MUST NOT come from the same photograph.

---

# 59. BLIP adaptation recipe

Canonical:

```text
epochs = 4
batch size = 16
learning rate = 2e-5
trainable text layers = 2
seed = 0
```

Selection:

> highest validation `rsum`

Epoch 0 / frozen BLIP remains the baseline.

---

# 60. Why SigLIP is not adapted here

The live SigLIP v1/v2 E2E carriers fine-tune on:

```text
image
+
class prompt
```

using class-labelled datasets.

That is a valid capability but is not equivalent to learning from:

```text
image
+
multiple natural captions
```

in a retrieval corpus.

The workshop MUST NOT reuse that class-prompt adaptation and describe it as caption-retrieval adaptation.

A future retrieval-specific SigLIP adapter would require a separately qualified image-caption sigmoid/contrastive training contract.

---

# 61. Adapted BLIP evaluation

Evaluate adapted BLIP in two roles:

### Adapted coarse retriever

Use the adapted:

```text
vision_proj
text_proj
```

for BLIP ITC retrieval.

### Adapted reranker

Use its adapted:

```text
text blocks
ITM head
```

for top-K reranking.

Report both separately.

---

# 62. Adapted reranker on SigLIP candidates

This is the most application-oriented `FULL` experiment:

```text
SigLIP 2 frozen index
→ top-K shortlist
→ adapted BLIP ITM reranker
```

Compare against:

```text
SigLIP 2
→ frozen BLIP ITM
```

This isolates whether domain adaptation of the expensive second stage improves a scalable retrieval architecture.

---

# 63. Freeze-before-test

Before evaluating the full 391-image test gallery, freeze:

```text
dataset roles
model revisions
embedding normalization
gallery membership
retrieval K values
rerank K
query caption rule
BLIP adaptation epoch
BLIP artifact digest
metric implementation
```

Write:

```text
outputs/frozen/frozen_experiment.json
```

No choices may change after final test ranks are viewed.

---

# 64. Principal frozen retrieval table

| Model | Dim | I2T R@1 | R@5 | R@10 | T2I R@1 | R@5 | R@10 | rsum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chance | — | | | | | | | |
| Colour/keyword | — | | | | | | | |
| SigLIP v1 | 768 | | | | | | | |
| SigLIP 2 | 768 | | | | | | | |
| BLIP ITC | 256 | | | | | | | |

No ranking label such as “best model” should be generated.

---

# 65. Reranking table

| Candidate source | Reranker | I2T R@1 | canonical T2I R@1 | Pair evaluations |
|---|---|---:|---:|---:|
| SigLIP v1 | none | | | 0 |
| SigLIP v1 | BLIP ITM | | | |
| SigLIP 2 | none | | | 0 |
| SigLIP 2 | BLIP ITM | | | |
| BLIP ITC | none | | | 0 |
| BLIP ITC | BLIP ITM | | | |

`FULL` adds:

```text
adapted BLIP ITM
```

rows.

---

# 66. Reranking interpretation

A reranker can only reorder the candidate set it receives.

Therefore:

> If the correct item is not in the first-stage top K, BLIP ITM cannot recover it.

Report:

```text
candidate recall@K
```

before reranking.

This is critical when interpreting reranker performance.

---

# 67. Retrieval upper bound for reranker

For each first-stage system calculate:

### I2T candidate oracle

Fraction of photographs having at least one valid caption somewhere in its top-K shortlist.

### T2I candidate oracle

Fraction of canonical captions whose owning photograph is in the top-K shortlist.

These form the maximum possible reranked R@1 at that K.

---

# 68. Caption perturbation exercise

On validation only, compare a bounded set of caption transformations:

### Original

human caption.

### Lowercase

all lowercase.

### Shortened

first N content words or a deterministic rule.

The purpose is not prompt engineering for the final test.

It is to demonstrate:

> Retrieval rankings depend on how the text query is expressed.

---

# 69. No-match behavior

All coarse retrievers return a nearest item even when the query is poor.

BLIP ITM also assigns every examined pair a score.

Include several deliberately irrelevant queries such as:

```text
a satellite orbiting mars
an underwater coral reef
```

against a bounded validation gallery.

Do not report the returned item as a true match.

Label the evaluation:

```text
no ground-truth match — nearest-neighbor behavior only
```

---

# 70. Score semantics

### SigLIP embedding similarity

cosine similarity.

### BLIP ITC

cosine similarity in BLIP's 256-dimensional projection.

### BLIP ITM

binary match-head softmax output.

None is a calibrated probability that the image and caption are semantically correct.

In particular:

```text
0.7 SigLIP cosine
```

cannot be numerically compared to:

```text
0.7 BLIP ITM probability
```

---

# 71. BYOD contract

Preferred archive:

```text
dataset.zip
├── records.jsonl
└── images/
```

Each JSONL row:

```json
{
  "id": "img001",
  "file": "images/img001.jpg",
  "captions": [
    "a red cup on a table",
    "a cup beside a plate"
  ],
  "split": "train",
  "category": "optional"
}
```

---

# 72. BYOD requirements

Validate:

- unique IDs;
- safe relative paths;
- no symlinks/path traversal;
- images decode;
- image dimensions within model ceilings;
- at least one caption per image;
- caption strings non-empty;
- captions within 256 characters;
- no duplicate captions within a record;
- no identical image digest across split roles.

---

# 73. BYOD role behavior

### STANDARD

Only a test/evaluation gallery is required.

### FULL

Require:

```text
train
validation
test
```

if BLIP adaptation is requested.

Do not adapt on the test gallery.

---

# 74. BYOD scalability boundary

This notebook is an **in-memory reference implementation**.

It is not a production vector database.

For large corpora, export embeddings to a suitable ANN/vector index.

The notebook MAY mention examples conceptually but SHOULD NOT add a new database dependency to the canonical path.

---

# 75. BYOD privacy guidance

The notebook MUST state:

> User-supplied photographs and captions are processed inside the selected notebook runtime and are not sent to DIMER workers or APIs. A hosted notebook remains an external compute environment. Do not upload confidential, personal, restricted, security-sensitive, or proprietary image/text data unless authorized.

VizWiz itself can contain personal spaces and everyday belongings; that should be acknowledged when discussing the sample corpus.

---

# 76. Search bias

Semantic retrieval can expose associations encoded in pretrained vision-language data.

Queries about:

- people;
- occupations;
- demographic attributes;
- cultures;
- places;

can produce stereotypical or uneven rankings.

The built-in workshop does not establish fairness for such queries.

---

# 77. Output structure

```text
outputs/
├── data/
│   ├── dataset_manifest.json
│   ├── train.json
│   ├── validation.json
│   └── test_gallery.json
│
├── index/
│   ├── siglip2_images.npy
│   ├── siglip2_texts.npy
│   ├── siglip1_images.npy
│   ├── siglip1_texts.npy
│   ├── blip_itc_images.npy
│   ├── blip_itc_texts.npy
│   ├── image_ids.json
│   ├── caption_ids.json
│   └── index_manifest.json
│
├── validation/
│   ├── retrieval_metrics.csv
│   ├── rerank_k_sweep.csv
│   └── caption_perturbation.csv
│
├── frozen/
│   └── frozen_experiment.json
│
├── test/
│   ├── retrieval_metrics.csv
│   ├── median_ranks.csv
│   ├── reranking_metrics.csv
│   ├── gallery_size_metrics.csv
│   ├── category_metrics.csv
│   └── hard_negatives.csv
│
├── adaptation/
│   └── blip/
│       └── training_history.json
│
├── artifacts/
│   └── blip/
│       ├── adapter.safetensors
│       └── manifest.json
│
├── figures/
├── provenance/
│   └── experiment_manifest.json
└── workshop_summary.json
```

---

# 78. Provenance

Record:

```text
notebook_spec
notebook_profile
notebook_mode
workshop_revision
execution_tier

dataset:
  source
  revision
  shard SHA-256
  train IDs
  validation IDs
  test/gallery IDs
  image/caption counts

siglip2:
  model ID
  revision
  weight SHA-256
  embedding dimension
  preprocessing

siglip1:
  model ID
  revision
  weight SHA-256
  embedding dimension
  preprocessing

blip:
  model ID
  revision
  base weight SHA-256
  ITC dimension
  preprocessing

retrieval:
  K values
  rerank K
  ownership convention
  canonical query-caption rule
  metric implementation

adaptation:
  training recipe
  selected epoch
  adapter digest
```

---

# 79. Notebook metadata

```json
{
  "dimer": {
    "notebook_spec": "2.1",
    "notebook_profile": "MULTI-CAPABILITY",
    "notebook_mode": "WORKSHOP",
    "standalone": true,
    "capability": "multi-model-vision-language-retrieval",
    "carrier": "dual-encoder image-text retrieval plus BLIP ITM reranking",
    "dataset": "VizWiz-Captions 208/40 train-validation plus 391-image 1737-caption independent test gallery",
    "default_tier": "STANDARD",
    "canonical_runtime": "NVIDIA Tesla T4",
    "worker_required": false,
    "credentials_required": false,
    "clean_runtime_evidence": "pending"
  }
}
```

---

# 80. STANDARD release acceptance

| Requirement | Required |
|---|---:|
| Notebook Spec 2.1 | PASS |
| `MULTI-CAPABILITY` / `WORKSHOP` | PASS |
| Fresh T4 `Run all` | PASS |
| No Git clone | PASS |
| No DIMER runtime source fetch | PASS |
| No DIMER services | PASS |
| No credentials | PASS |
| VizWiz digest verification | PASS |
| Image-disjoint split | PASS |
| 391-image / 1,737-caption gallery | PASS |
| Chance baseline | PASS |
| Colour-keyword baseline | PASS |
| SigLIP v1 embeddings | PASS |
| SigLIP 2 embeddings | PASS |
| BLIP ITC embeddings | PASS |
| Common I2T/T2I evaluator | PASS |
| R@1 / R@5 / R@10 | PASS |
| Median rank | PASS |
| rsum | PASS |
| BLIP ITM reranking | PASS |
| Candidate oracle@K | PASS |
| Gallery-size experiment | PASS |
| Hard-negative gallery | PASS |
| Category breakdown | PASS |
| Embedding artifact export | PASS |
| BYOD positive case | PASS |
| BYOD refusal probes | PASS |
| Provenance export | PASS |

---

# 81. FULL release acceptance

Additionally:

| Requirement | Required |
|---|---:|
| BLIP retrieval adaptation | PASS |
| Validation-rsum epoch selection | PASS |
| Freeze-before-test | PASS |
| SafeTensors adapter export | PASS |
| Fresh adapter reload | PASS |
| Adapted BLIP ITC evaluation | PASS |
| Adapted BLIP ITM reranking | PASS |
| Adapted BLIP reranks SigLIP2 candidates | PASS |
| Runtime / VRAM record | PASS |

---

# 82. Suggested registry entry

```markdown
| Notebook | Profile | Mode | Capability | Runtime | Sample | BYOD | Run-all | Status |
|---|---|---|---|---|---|---|---|---|
| `DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb` | `MULTI-CAPABILITY` | `WORKSHOP` | SigLIP v1/v2 + BLIP ITC retrieval and BLIP ITM reranking | T4 | VizWiz-Captions, 391-image / 1,737-caption held-out gallery | yes | pending | candidate |
```

---

# 83. Implementation principle

The notebook should center on the architecture actually used by practical multimodal search systems:

```text
                    ┌─ SigLIP 2 ─┐
caption/image query ├─ SigLIP v1 ├─→ vector similarity → top-K
                    └─ BLIP ITC ─┘                    │
                                                      ↓
                                              BLIP ITM reranker
                                                      │
                                                      ↓
                                                final ranking
```

The central lesson is:
> **Dual encoders make large-scale vision-language retrieval practical because image and text representations can be computed independently and indexed. Cross-modal rerankers are more expensive but can reconsider a small shortlist with richer pairwise reasoning. Retrieval system design is therefore not simply “which model scores highest,” but how candidate generation, gallery size, representation size, reranking depth, and domain adaptation fit together.**