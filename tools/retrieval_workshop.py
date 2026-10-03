"""Stages of DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb, run in its isolated environment.

The notebook carries this file byte-for-byte, writes it next to the hash-locked requirements, and runs
every model stage as ``python retrieval_workshop.py --config <json> --stage <name>`` with the
interpreter of a uv-built Python 3.12 virtual environment. Nothing is installed into the notebook
kernel, which only reads the JSON/CSV/PNG files written here.

Function bodies are the notebook's former cell code (revision 0.2.0-candidate), moved unchanged
except where noted: the BLIP wrapper resolves its default snapshot root and device when it is
constructed, and stage outputs that used to be kernel objects are written under ``WORK_ROOT/state``.
"""
# ruff: noqa: E401,E402,E501,E701,E702,E731,F841,I001,UP017
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import platform
import random
import re
import sys
import tempfile
import time
import traceback
import zipfile
from math import comb
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image

STAGE_FORMAT = "dimer.retrieval-workshop.stages.v1"

# ---- Configuration (overwritten by configure() from the notebook controls) --------------------------
WORKSHOP_TIER = "STANDARD"
RERANK_TOP_K = 5
RECALL_KS = (1, 5, 10)
GALLERY_SIZES = (70, 128, 256, 391)
BLIP_EPOCHS = 4
BLIP_LEARNING_RATE = 2e-5
BLIP_BATCH_SIZE = 16
BLIP_TRAINABLE_TEXT_LAYERS = 2
SEED = 0
SPLIT_SEED = 42
OUTPUT_DIR = "outputs"
OUTPUT_ROOT = Path(OUTPUT_DIR)
WORK_ROOT = Path("work")
BYOD_ZIP_PATH = ""
RUN_STARTED = 0.0

# Heavy libraries load only in the stages that use them (see load_torch / load_transformers).
torch = None
F = None
DEVICE = "cpu"
save_file = None
load_file = None
AutoModel = AutoProcessor = BlipForImageTextRetrieval = BlipProcessor = None
hf_hub_download = None
SIGLIP2_ROOT = WORK_ROOT / "models" / "siglip2"
SIGLIP1_ROOT = WORK_ROOT / "models" / "siglip1"
BLIP_ROOT = WORK_ROOT / "models" / "blip"

# ---- Immutable model identities ---------------------------------------------------------------------
SIGLIP2 = {
    "model_id":"google/siglip2-base-patch16-224",
    "revision":"5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2",
    "files":[
        ("README.md",3367,"77fd3e4cc34abaa0cf891f03fc73be58cd0e36ce09741cb0fe096342e4d8867f"),
        ("config.json",253,"fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7"),
        ("model.safetensors",1500800904,"612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"),
        ("preprocessor_config.json",394,"9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc"),
        ("special_tokens_map.json",636,"baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351"),
        ("tokenizer.json",34363039,"cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"),
        ("tokenizer.model",4241003,"61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"),
        ("tokenizer_config.json",47164,"14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8"),
    ],
    "dim":768,
}
SIGLIP1 = {
    "model_id":"google/siglip-base-patch16-256",
    "revision":"b078df89e446d623010d890864d4207fe6399f61",
    "files":[
        ("README.md",4116,"b105a3cfa97df9f082c42817c3caf61bfd7d360386fed4bdebc0c7fff0acba01"),
        ("config.json",322,"acc261689fe8d29cf8ceacd5dc5d05fd53d2a43f3f6c57e6c16112a14a618dfa"),
        ("model.safetensors",812856640,"f0cee7c815135c44a515eff72ab3040499744920442bc25567cd04efc93f8f65"),
        ("preprocessor_config.json",368,"e12b577bd0da1f9bf6a0b8d3d78c8f9b55b9d99dec25556954e19cfc7b8ecf28"),
        ("special_tokens_map.json",409,"2b6a1ff67a27e0df9ac0c7d93250fc0d87431c7b366b3d5669217104f9088a26"),
        ("spiece.model",798330,"1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec"),
        ("tokenizer.json",2399357,"c6e405cb7c670d56636a9402c81023a55bc6c3c53d89cf02b92f5c5005bfe920"),
        ("tokenizer_config.json",711,"d6423dae508cc3a129d22ea443841c111832a1a73125b8f25ea8736951698bcb"),
    ],
    "dim":768,
}
BLIP = {
    "model_id":"Salesforce/blip-itm-base-coco",
    "revision":"bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c",
    "files":[
        ("README.md",5492,"db2e7ff1e647bc42d8b0d4cd3653ad65c1d24b5559a153e6e6ea2f3c12edd978"),
        ("config.json",4560,"3e6464c2ce7c54512ddb101c5e9a8e77f4c2d637be9e3d005667ccd4a34c6ef2"),
        ("preprocessor_config.json",445,"0aa66e2e9ac3ea3b5cd4388c35072e22db4e1cc1f96c7872bed07749c712ade1"),
        ("pytorch_model.bin",895139697,"017fb3e7f4e125f13a8a4717f1402dbe0d0bb877474b4a203db13a4447b0227f"),
        ("special_tokens_map.json",125,"b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"),
        ("tokenizer.json",711396,"d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
        ("tokenizer_config.json",456,"86da6fdb761b02f73a05561aba71711c2d7c205fe1fd1744046173a410263925"),
        ("vocab.txt",231508,"07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
    ],
    "dim":256,
}

# ---- Canonical corpus -------------------------------------------------------------------------------
CORPUS_REPO="mm-eval/VizWiz-Captions"
CORPUS_REVISION="c4a6d897836e7885d0095134f92d392e4e770539"
CORPUS_PATH="data/val-00004-of-00005.parquet"
CORPUS_BYTES=392_245_504
CORPUS_SHA256="4492465a41d32b3c12b8b7b6a0cf7e0a0e202a5b825b006ca0c85dcdf24efd3e"

ITC_TEMPERATURE=0.07


def configure(config):
    """Apply the notebook controls (a JSON object written by the notebook's run_stage)."""
    global WORKSHOP_TIER, RERANK_TOP_K, RECALL_KS, GALLERY_SIZES, BLIP_EPOCHS, BLIP_LEARNING_RATE
    global BLIP_BATCH_SIZE, BLIP_TRAINABLE_TEXT_LAYERS, SEED, SPLIT_SEED, OUTPUT_DIR, OUTPUT_ROOT
    global WORK_ROOT, BYOD_ZIP_PATH, RUN_STARTED, SIGLIP2_ROOT, SIGLIP1_ROOT, BLIP_ROOT
    WORKSHOP_TIER = config["tier"]
    RERANK_TOP_K = int(config["rerank_top_k"])
    RECALL_KS = tuple(config["recall_ks"])
    GALLERY_SIZES = tuple(config["gallery_sizes"])
    BLIP_EPOCHS = int(config["blip_epochs"])
    BLIP_LEARNING_RATE = float(config["blip_learning_rate"])
    BLIP_BATCH_SIZE = int(config["blip_batch_size"])
    BLIP_TRAINABLE_TEXT_LAYERS = int(config["blip_trainable_text_layers"])
    SEED = int(config["seed"])
    SPLIT_SEED = int(config["split_seed"])
    OUTPUT_DIR = config["output_dir"]
    OUTPUT_ROOT = Path(OUTPUT_DIR)
    WORK_ROOT = Path(config["work_dir"])
    BYOD_ZIP_PATH = config.get("byod_zip_path", "")
    RUN_STARTED = float(config["run_started"])
    SIGLIP2_ROOT = WORK_ROOT / "models" / "siglip2"
    SIGLIP1_ROOT = WORK_ROOT / "models" / "siglip1"
    BLIP_ROOT = WORK_ROOT / "models" / "blip"
    if WORKSHOP_TIER not in {"STANDARD", "FULL"}:
        raise ValueError("WORKSHOP_TIER must be STANDARD or FULL")
    if RERANK_TOP_K not in (5, 10, 20):
        raise ValueError("RERANK_TOP_K must be one of 5, 10, 20")


def load_torch():
    global torch, F, DEVICE, save_file, load_file
    import torch as _torch
    import torch.nn.functional as _F
    from safetensors.torch import save_file as _save, load_file as _load
    torch, F, save_file, load_file = _torch, _F, _save, _load
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_transformers():
    global AutoModel, AutoProcessor, BlipForImageTextRetrieval, BlipProcessor
    load_torch()
    from transformers import AutoModel as _AM, AutoProcessor as _AP
    from transformers import BlipForImageTextRetrieval as _BR, BlipProcessor as _BP
    AutoModel, AutoProcessor, BlipForImageTextRetrieval, BlipProcessor = _AM, _AP, _BR, _BP


def model_root(name):
    return {"siglip2":SIGLIP2_ROOT,"siglip1":SIGLIP1_ROOT,"blip":BLIP_ROOT}[name]


def state_path(name):
    path=WORK_ROOT/"state"/name
    path.parent.mkdir(parents=True,exist_ok=True)
    return path


def write_state(name,value):
    state_path(name).write_text(json.dumps(value,indent=2,default=str),encoding="utf-8")


def read_state(name):
    path=WORK_ROOT/"state"/name
    if not path.is_file():
        raise RuntimeError(f"{name} is missing: run the earlier notebook cells first (Run all).")
    return json.loads(path.read_text(encoding="utf-8"))


def save_scores(name,scores):
    np.save(state_path(f"{name}_scores.npy"),scores)


def load_scores(name):
    path=WORK_ROOT/"state"/f"{name}_scores.npy"
    if not path.is_file():
        raise RuntimeError(f"{name} scores are missing: run the earlier notebook cells first (Run all).")
    return np.load(path,allow_pickle=False)


def records():
    return read_state("records.json")


# ---- Former notebook functions (verbatim) -----------------------------------------------------------
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def clean_captions(answer):
    out=[]
    for c in answer or []:
        text=" ".join(str(c).split())
        if text and len(text)<=256 and text not in out:
            out.append(text)
    return out

def decoded_rgb_digest(payload):
    with Image.open(io.BytesIO(payload)) as im:
        rgb=im.convert("RGB")
        h=hashlib.sha256()
        h.update(f"{rgb.width}x{rgb.height}".encode())
        h.update(rgb.tobytes())
        return h.hexdigest()

def read_group(group):
    table=pf.read_row_group(
        group,
        columns=["id","answer","question_type","text_detected","media"],
    )
    rows=[]
    for row in table.to_pylist():
        captions=clean_captions(row["answer"])
        media=row["media"]
        if not isinstance(media,list) or len(media)!=1:
            raise ValueError(f"{row['id']}: expected exactly one image")
        payload=bytes(media[0]["bytes"])
        image_id=str(row["id"])
        dest=WORK_ROOT/"vizwiz"/"images"/f"{image_id}.jpg"
        dest.write_bytes(payload)
        # Decode now so corrupt images fail before model acquisition.
        with Image.open(dest) as im:
            im.verify()
        rows.append({
            "image_id":image_id,
            "path":str(dest),
            "captions":captions,
            "category":"text" if bool(row["text_detected"]) else "no-text",
            "question_type":str(row["question_type"]),
            "file_sha256":hashlib.sha256(payload).hexdigest(),
            "pixel_sha256":decoded_rgb_digest(payload),
            "row_group":group,
        })
    return rows

def gallery(records):
    texts=[];owners=[]
    for i,r in enumerate(records):
        for caption in r["captions"]:
            texts.append(str(caption));owners.append(i)
    return texts,owners

def retrieval_metrics(scores,owners):
    grid=np.asarray(scores,dtype=np.float64)
    owners=np.asarray(owners,dtype=int)
    if grid.ndim!=2 or grid.shape[1]!=len(owners):
        raise ValueError("score grid/owner mismatch")
    if set(owners.tolist())!=set(range(grid.shape[0])):
        raise ValueError("every image must own at least one caption")
    if not np.isfinite(grid).all():
        raise ValueError("scores must be finite")

    i2t_ranks=[]
    for i,row in enumerate(grid):
        order=np.argsort(-row,kind="stable")
        positions=np.flatnonzero(owners[order]==i)
        i2t_ranks.append(int(positions[0])+1)

    t2i_ranks=[]
    for j,owner in enumerate(owners):
        order=np.argsort(-grid[:,j],kind="stable")
        position=int(np.flatnonzero(order==owner)[0])+1
        t2i_ranks.append(position)

    out={"n_images":grid.shape[0],"n_captions":grid.shape[1]}
    for k in RECALL_KS:
        out[f"i2t_recall_at_{k}"]=float(np.mean(np.asarray(i2t_ranks)<=k))
        out[f"t2i_recall_at_{k}"]=float(np.mean(np.asarray(t2i_ranks)<=k))
    out["i2t_median_rank"]=float(np.median(i2t_ranks))
    out["t2i_median_rank"]=float(np.median(t2i_ranks))
    out["rsum"]=sum(out[f"{d}_recall_at_{k}"] for d in ("i2t","t2i") for k in RECALL_KS)
    return out

def canonical_caption_indices(owners,n_images):
    owners=np.asarray(owners)
    return [int(np.flatnonzero(owners==i)[0]) for i in range(n_images)]

def candidate_oracle(scores,owners,k):
    grid=np.asarray(scores);owners=np.asarray(owners)
    i2t=[]
    for i,row in enumerate(grid):
        top=np.argsort(-row,kind="stable")[:min(k,grid.shape[1])]
        i2t.append(bool(np.any(owners[top]==i)))
    queries=canonical_caption_indices(owners,grid.shape[0])
    t2i=[]
    for j in queries:
        top=np.argsort(-grid[:,j],kind="stable")[:min(k,grid.shape[0])]
        t2i.append(bool(owners[j] in top))
    return {
        "i2t_candidate_oracle":float(np.mean(i2t)),
        "t2i_candidate_oracle":float(np.mean(t2i)),
    }

def expected_recall(n,total_positives,k):
    if k>=n:return 1.0
    if total_positives<=0:return 0.0
    return 1.0 - comb(n-total_positives,k)/comb(n,k)

def chance_baseline(records):
    texts,owners=gallery(records)
    n_images=len(records);n_captions=len(texts)
    out={"n_images":n_images,"n_captions":n_captions}
    for k in RECALL_KS:
        out[f"i2t_recall_at_{k}"]=sum(
            expected_recall(n_captions,len(r["captions"]),min(k,n_captions))
            for r in records
        )/n_images
        out[f"t2i_recall_at_{k}"]=min(k,n_images)/n_images
    out["i2t_median_rank"]=None
    out["t2i_median_rank"]=(n_images+1)/2
    out["rsum"]=sum(out[f"{d}_recall_at_{k}"] for d in ("i2t","t2i") for k in RECALL_KS)
    return out

def colour_signature(path,grid=3):
    with Image.open(path) as im:
        small=im.convert("RGB").resize((grid*8,grid*8),Image.BILINEAR)
        arr=np.asarray(small,dtype=np.float32)/255.0
    out=[]
    for y in range(grid):
        for x in range(grid):
            cell=arr[y*8:(y+1)*8,x*8:(x+1)*8]
            out.extend(cell.mean(axis=(0,1)).tolist())
    return np.asarray(out,dtype=np.float32)

def tokens(text):
    return set(re.findall(r"[a-z0-9]+",text.lower()))

def unigram_f1(a,b):
    aa=tokens(a);bb=tokens(b)
    if not aa or not bb:return 0.0
    overlap=len(aa&bb)
    if not overlap:return 0.0
    p=overlap/len(aa);r=overlap/len(bb)
    return 2*p*r/(p+r)

def colour_keyword_baseline(train,records):
    train_sig=[colour_signature(r["path"]) for r in train]
    train_caps=[(c,i) for i,r in enumerate(train) for c in r["captions"]]
    texts,owners=gallery(records)
    test_sig=[colour_signature(r["path"]) for r in records]

    # T2I score grid.
    t2i=np.zeros((len(records),len(texts)),dtype=np.float32)
    for j,text in enumerate(texts):
        best=max(train_caps,key=lambda x:unigram_f1(text,x[0]))[1]
        for i,sig in enumerate(test_sig):
            t2i[i,j]=-float(np.linalg.norm(sig-train_sig[best]))

    # I2T score grid.
    i2t=np.zeros_like(t2i)
    for i,sig in enumerate(test_sig):
        twin=min(range(len(train)),key=lambda t:float(np.linalg.norm(sig-train_sig[t])))
        twin_caps=train[twin]["captions"]
        for j,text in enumerate(texts):
            i2t[i,j]=max(unigram_f1(text,c) for c in twin_caps)

    mt=retrieval_metrics(t2i,owners)
    mi=retrieval_metrics(i2t,owners)
    out={"n_images":len(records),"n_captions":len(texts)}
    for k in RECALL_KS:
        out[f"i2t_recall_at_{k}"]=mi[f"i2t_recall_at_{k}"]
        out[f"t2i_recall_at_{k}"]=mt[f"t2i_recall_at_{k}"]
    out["i2t_median_rank"]=mi["i2t_median_rank"]
    out["t2i_median_rank"]=mt["t2i_median_rank"]
    out["rsum"]=sum(out[f"{d}_recall_at_{k}"] for d in ("i2t","t2i") for k in RECALL_KS)
    return out

def stage_snapshot(spec,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for name,size,digest in spec["files"]:
        path=Path(hf_hub_download(
            repo_id=spec["model_id"],
            filename=name,
            revision=spec["revision"],
            local_dir=str(root),
        ))
        if path.stat().st_size!=size or sha256_file(path)!=digest:
            raise ValueError(f"{spec['model_id']} {name}: digest/size mismatch")
    return root

def siglip_embed(spec,root,records,texts,image_batch=16,text_batch=64):
    import traceback
    model=None;processor=None
    try:
        processor=AutoProcessor.from_pretrained(str(root),local_files_only=True,trust_remote_code=False)
        model=AutoModel.from_pretrained(
            str(root),local_files_only=True,trust_remote_code=False
        ).to(DEVICE).eval()

        image_chunks=[]
        t0=time.perf_counter()
        for start in range(0,len(records),image_batch):
            images=[]
            for r in records[start:start+image_batch]:
                with Image.open(r["path"]) as im:
                    images.append(im.convert("RGB"))
            batch=processor(images=images,return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                feat=model.get_image_features(**batch)
                feat=F.normalize(feat,p=2,dim=-1)
            image_chunks.append(feat.cpu())
        image_seconds=time.perf_counter()-t0

        text_chunks=[]
        t0=time.perf_counter()
        for start in range(0,len(texts),text_batch):
            chunk=[str(x).lower() for x in texts[start:start+text_batch]]
            batch=processor(
                text=chunk,padding="max_length",max_length=64,return_tensors="pt"
            ).to(DEVICE)
            with torch.inference_mode():
                feat=model.get_text_features(**batch)
                feat=F.normalize(feat,p=2,dim=-1)
            text_chunks.append(feat.cpu())
        text_seconds=time.perf_counter()-t0

        images=torch.cat(image_chunks).numpy().astype(np.float32,copy=False)
        text=torch.cat(text_chunks).numpy().astype(np.float32,copy=False)
        model=None;processor=None
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

        return images,text,{
            "image_embedding_seconds":image_seconds,
            "text_embedding_seconds":text_seconds,
        }
    except Exception as exc:
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        model=None;processor=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

class BlipWorkshop:
    def __init__(self,root=None,device=None):
        # Moved from a notebook cell: resolve the defaults at construction, after configure()/load_torch().
        root=model_root("blip") if root is None else root
        device=DEVICE if device is None else device
        self.root=Path(root)
        self.device=device
        self.processor=BlipProcessor.from_pretrained(str(root),local_files_only=True)
        self.model=BlipForImageTextRetrieval.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=False,
            weights_only=True,
            dtype=torch.float32,
        ).to(device).eval()

    def raw_image_embeds(self,records,batch_size=8):
        out=[]
        t0=time.perf_counter()
        for start in range(0,len(records),batch_size):
            images=[]
            for r in records[start:start+batch_size]:
                with Image.open(r["path"]) as im:
                    images.append(im.convert("RGB"))
            pixels=self.processor(images=images,return_tensors="pt")["pixel_values"].to(self.device)
            with torch.inference_mode():
                embeds=self.model.vision_model(pixel_values=pixels)[0]
            out.extend(e.detach().clone() for e in embeds)
        return out,time.perf_counter()-t0

    def image_features(self,raw_embeds,batch_size=64):
        out=[]
        with torch.inference_mode():
            for start in range(0,len(raw_embeds),batch_size):
                stacked=torch.stack([x[0] for x in raw_embeds[start:start+batch_size]])
                out.append(F.normalize(self.model.vision_proj(stacked),dim=-1).cpu())
        return torch.cat(out).numpy().astype(np.float32,copy=False)

    def text_features(self,texts,batch_size=64):
        out=[];t0=time.perf_counter()
        for start in range(0,len(texts),batch_size):
            tokens=self.processor.tokenizer(
                list(texts[start:start+batch_size]),padding=True,return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                hidden=self.model.text_encoder(
                    input_ids=tokens["input_ids"],attention_mask=tokens["attention_mask"]
                )[0]
                out.append(F.normalize(self.model.text_proj(hidden[:,0,:]),dim=-1).cpu())
        return torch.cat(out).numpy().astype(np.float32,copy=False),time.perf_counter()-t0

    def itm_probabilities(self,pairs,batch_size=16):
        values=[]
        t0=time.perf_counter()
        for start in range(0,len(pairs),batch_size):
            chunk=pairs[start:start+batch_size]
            tokens=self.processor.tokenizer(
                [text for _embed,text in chunk],padding=True,return_tensors="pt"
            ).to(self.device)
            image_embeds=torch.stack([embed for embed,_text in chunk]).to(self.device)
            image_atts=torch.ones(image_embeds.shape[:2],dtype=torch.long,device=self.device)
            with torch.inference_mode():
                hidden=self.model.text_encoder(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                )[0]
                logits=self.model.itm_head(hidden[:,0,:])
                values.extend(torch.softmax(logits.float(),dim=-1)[:,1].cpu().tolist())
        return values,time.perf_counter()-t0

def rerank_with_blip(blip,raw_image_embeds,texts,owners,coarse,k):
    retrieval_metrics(coarse,owners)
    if coarse.shape[0]<2 or len(raw_image_embeds)!=coarse.shape[0] or len(texts)!=coarse.shape[1] or k<1:
        raise ValueError("Reranking needs aligned inputs, at least two images and positive k")
    owners=np.asarray(owners,dtype=int)
    n_images=coarse.shape[0]

    # Image -> text (every photograph is a query).
    pairs=[];pair_index=[];coarse_i2t_hits=[]
    for i in range(n_images):
        top=np.argsort(-coarse[i],kind="stable")[:min(k,coarse.shape[1])]
        coarse_i2t_hits.append(bool(owners[int(top[0])]==i))
        for j in top:
            pairs.append((raw_image_embeds[i],texts[int(j)]))
            pair_index.append((i,int(j)))
    probs,seconds_i2t=blip.itm_probabilities(pairs)
    if len(probs)!=len(pair_index) or not np.isfinite(probs).all() or any(p<0 or p>1 for p in probs):
        raise ValueError("BLIP ITM returned invalid probability count/range")
    best={}
    for (i,j),p in zip(pair_index,probs,strict=True):
        if i not in best or p>best[i][0]:
            best[i]=(p,j)
    i2t_r1=float(np.mean([owners[j]==i for i,(_p,j) in best.items()]))

    # Canonical caption -> image (the first caption of every photograph is a query).
    queries=canonical_caption_indices(owners,n_images)
    pairs=[];pair_index=[];coarse_t2i_hits=[]
    for j in queries:
        top=np.argsort(-coarse[:,j],kind="stable")[:min(k,n_images)]
        coarse_t2i_hits.append(bool(int(top[0])==owners[j]))
        for i in top:
            pairs.append((raw_image_embeds[int(i)],texts[j]))
            pair_index.append((int(i),j))
    probs2,seconds_t2i=blip.itm_probabilities(pairs)
    if len(probs2)!=len(pair_index) or not np.isfinite(probs2).all() or any(p<0 or p>1 for p in probs2):
        raise ValueError("BLIP ITM returned invalid probability count/range")
    best2={}
    for (i,j),p in zip(pair_index,probs2,strict=True):
        if j not in best2 or p>best2[j][0]:
            best2[j]=(p,i)
    t2i_r1=float(np.mean([owners[j]==i for j,(_p,i) in best2.items()]))

    # Hard-negative pair accuracy.
    pairs=[]
    for i,j_pos in enumerate(queries):
        row=coarse[i].copy()
        row[owners==i]=-np.inf
        j_neg=int(np.argmax(row))
        pairs.append((raw_image_embeds[i],texts[j_pos]))
        pairs.append((raw_image_embeds[i],texts[j_neg]))
    probs3,seconds_pair=blip.itm_probabilities(pairs)
    if len(probs3)!=len(pairs) or not np.isfinite(probs3).all() or any(p<0 or p>1 for p in probs3):
        raise ValueError("BLIP ITM returned invalid probability count/range")
    pair_accuracy=float(np.mean(np.asarray(probs3[0::2])>np.asarray(probs3[1::2])))

    # Coarse R@1 on exactly the queries that were reranked, so each delta isolates the reranker.
    coarse_i2t_r1=float(np.mean(coarse_i2t_hits))
    coarse_t2i_r1_canonical=float(np.mean(coarse_t2i_hits))
    return {
        "rerank_top_k":k,
        "i2t_queries":n_images,
        "coarse_i2t_r1":coarse_i2t_r1,
        "itm_i2t_recall_at_1":i2t_r1,
        "delta_i2t_r1":i2t_r1-coarse_i2t_r1,
        "t2i_queries":len(queries),
        "coarse_t2i_r1_canonical":coarse_t2i_r1_canonical,
        "itm_t2i_recall_at_1":t2i_r1,
        "delta_t2i_r1":t2i_r1-coarse_t2i_r1_canonical,
        "itm_pair_accuracy":pair_accuracy,
        "pair_evaluations":len(probs)+len(probs2)+len(probs3),
        "seconds":seconds_i2t+seconds_t2i+seconds_pair,
        **candidate_oracle(coarse,owners,k),
    }

def blip_trainable_names(model,n_layers=2):
    total=len(model.text_encoder.encoder.layer)
    first=total-n_layers
    prefixes=tuple(f"text_encoder.encoder.layer.{k}." for k in range(first,total))+(
        "vision_proj.","text_proj.","itm_head.",
    )
    return [name for name,_ in model.named_parameters() if name.startswith(prefixes)]

def blip_score_from_raw(blip,records,raw_embeds):
    texts,owners=gallery(records)
    image_feat=blip.image_features(raw_embeds)
    text_feat,_=blip.text_features(texts)
    scores=image_feat@text_feat.T
    return retrieval_metrics(scores,owners),scores,texts,owners

def ensure_cross_image_batch(chosen,pairs):
    chosen=list(chosen)
    if not chosen:raise ValueError("Empty BLIP training batch")
    owners={owner for owner,_ in chosen}
    if len(owners)<2:
        other=next((pair for pair in pairs if pair[0] not in owners),None)
        if other is None:raise ValueError("BLIP adaptation needs at least two distinct image owners")
        chosen.append(other)
    return chosen

def adapt_blip(blip,train,val,epochs=4,lr=2e-5,batch_size=16,n_layers=2,seed=0):
    model=blip.model
    names=blip_trainable_names(model,n_layers)
    wanted=set(names)
    for name,param in model.named_parameters():
        param.requires_grad_(name in wanted)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=0.01)

    train_raw,_=blip.raw_image_embeds(train)
    val_raw,_=blip.raw_image_embeds(val)

    pairs=[(i,str(c)) for i,r in enumerate(train) for c in r["captions"]]
    generator=torch.Generator().manual_seed(seed)

    def score_val():
        model.eval()
        metrics,_,_,_=blip_score_from_raw(blip,val,val_raw)
        return metrics

    initial={k:v.detach().clone() for k,v in model.state_dict().items() if k in wanted}
    best={k:v.clone() for k,v in initial.items()}
    history=[{"epoch":0,"train_loss":None,"val":score_val(),"note":"frozen model"}]
    best_epoch=0
    best_score=history[0]["val"]["rsum"]

    t0=time.perf_counter()
    try:
        for epoch in range(1,epochs+1):
            model.train()
            order=torch.randperm(len(pairs),generator=generator).tolist()
            losses=[]
            for start in range(0,len(order),batch_size):
                chosen=[pairs[j] for j in order[start:start+batch_size]]
                chosen=ensure_cross_image_batch(chosen,pairs)

                image_index=torch.tensor([i for i,_ in chosen],device=DEVICE)
                image_embeds=torch.stack([train_raw[i] for i,_ in chosen]).to(DEVICE)
                tokens=blip.processor.tokenizer(
                    [c for _,c in chosen],padding=True,return_tensors="pt"
                ).to(DEVICE)

                text_hidden=model.text_encoder(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )[0]
                text_feat=F.normalize(model.text_proj(text_hidden[:,0,:]),dim=-1)
                image_feat=F.normalize(model.vision_proj(image_embeds[:,0,:]),dim=-1)
                sim=image_feat@text_feat.T/ITC_TEMPERATURE

                same=image_index[:,None].eq(image_index[None,:])
                targets=same.float()
                targets=targets/targets.sum(dim=1,keepdim=True)
                log_i=torch.log_softmax(sim,dim=1)
                log_t=torch.log_softmax(sim.T,dim=1)
                loss_itc=-((targets*log_i).sum(1).mean()+(targets.T*log_t).sum(1).mean())/2

                with torch.no_grad():
                    w_text=torch.softmax(sim.detach().float(),dim=1)
                    w_text=w_text.masked_fill(same,0.0)
                    w_text=w_text/(w_text.sum(dim=1,keepdim=True)+1e-12)

                    w_image=torch.softmax(sim.detach().float().T,dim=1)
                    w_image=w_image.masked_fill(same.T,0.0)
                    w_image=w_image/(w_image.sum(dim=1,keepdim=True)+1e-12)

                    # Every normal batch contains at least two photographs. Refuse pathological batches.
                    if torch.any(w_text.sum(dim=1)<=0) or torch.any(w_image.sum(dim=1)<=0):
                        raise RuntimeError("BLIP batch lacks a valid cross-photograph hard negative")
                    neg_text=torch.multinomial(w_text,1).squeeze(1)
                    neg_image=torch.multinomial(w_image,1).squeeze(1)

                itm_images=torch.cat([image_embeds,image_embeds,image_embeds[neg_image]])
                itm_ids=torch.cat([
                    tokens["input_ids"],
                    tokens["input_ids"][neg_text],
                    tokens["input_ids"],
                ])
                itm_mask=torch.cat([
                    tokens["attention_mask"],
                    tokens["attention_mask"][neg_text],
                    tokens["attention_mask"],
                ])
                image_atts=torch.ones(itm_images.shape[:2],dtype=torch.long,device=DEVICE)

                fused=model.text_encoder(
                    input_ids=itm_ids,
                    attention_mask=itm_mask,
                    encoder_hidden_states=itm_images,
                    encoder_attention_mask=image_atts,
                )[0]
                logits=model.itm_head(fused[:,0,:])
                labels=torch.cat([
                    torch.ones(len(chosen),dtype=torch.long,device=DEVICE),
                    torch.zeros(2*len(chosen),dtype=torch.long,device=DEVICE),
                ])
                loss_itm=F.cross_entropy(logits,labels)
                loss=loss_itc+loss_itm

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params,1.0)
                optimizer.step()
                losses.append(float(loss.detach()))

            model.eval()
            row={
                "epoch":epoch,
                "train_loss":float(np.mean(losses)),
                "val":score_val(),
            }
            history.append(row)
            print(row)
            if row["val"]["rsum"]>best_score:
                best_score=row["val"]["rsum"]
                best_epoch=epoch
                best={k:v.detach().clone() for k,v in model.state_dict().items() if k in wanted}
    except BaseException:
        merged=dict(model.state_dict());merged.update(initial)
        model.load_state_dict(merged,strict=True)
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        raise

    merged=dict(model.state_dict());merged.update(best)
    model.load_state_dict(merged,strict=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)

    return {
        "trainable_names":names,
        "n_trainable":sum(v.numel() for n,v in model.named_parameters() if n in wanted),
        "n_total":sum(p.numel() for p in model.parameters()),
        "epochs":epochs,
        "best_epoch":best_epoch,
        "selection":"highest validation rsum",
        "lr":lr,
        "batch_size":batch_size,
        "itc_temperature":ITC_TEMPERATURE,
        "history":history,
        "seconds":time.perf_counter()-t0,
    }

def save_verified_blip_adapter(blip,report,folder,validation):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    names=report['trainable_names'];state=blip.model.state_dict()
    tensors={name:state[name].detach().cpu().contiguous() for name in names}
    path=folder/'adapter.safetensors';save_file(tensors,str(path),metadata={'format':'pt'})
    records=validation[:2];texts,_=gallery(records)
    raw,_=blip.raw_image_embeds(records);images=blip.image_features(raw);text,_=blip.text_features(texts)
    np.save(folder/'validation_images.npy',images);np.save(folder/'validation_texts.npy',text)
    files=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in [path,folder/'validation_images.npy',folder/'validation_texts.npy']]
    manifest={'base':{'model_id':BLIP['model_id'],'revision':BLIP['revision']},'tensors':names,'files':files,
              'validation_ids':[r['image_id'] for r in records],'history':report['history'],
              'selection':report['selection'],'format':'org.valcorza.blip-itm-base-coco.adapter.v1'}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

def load_verified_blip_adapter(folder,validation):
    import traceback
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['base']!={'model_id':BLIP['model_id'],'revision':BLIP['revision']}:
        raise ValueError('BLIP adapter base identity mismatch')
    expected=['adapter.safetensors','validation_images.npy','validation_texts.npy']
    if [f['path'] for f in manifest['files']]!=expected:raise ValueError('Unexpected adapter files')
    for entry in manifest['files']:
        path=folder/entry['path']
        if path.stat().st_size!=entry['bytes'] or sha256_file(path)!=entry['sha256']:
            raise ValueError('BLIP adapter/reference digest mismatch')
    fresh=None
    try:
        fresh=BlipWorkshop();tensors=load_file(str(folder/'adapter.safetensors'));state=fresh.model.state_dict()
        if sorted(tensors)!=sorted(manifest['tensors']):raise ValueError('BLIP adapter tensor list mismatch')
        merged=dict(state)
        for name,value in tensors.items():
            if name not in state or tuple(value.shape)!=tuple(state[name].shape):raise ValueError('BLIP adapter tensor shape mismatch')
            merged[name]=value.to(state[name].dtype)
        fresh.model.load_state_dict(merged,strict=True);fresh.model.eval()
        records=validation[:2]
        if [r['image_id'] for r in records]!=manifest['validation_ids']:raise ValueError('Adapter validation identity mismatch')
        texts,_=gallery(records);raw,_=fresh.raw_image_embeds(records)
        images=fresh.image_features(raw);text,_=fresh.text_features(texts)
        for actual,name in [(images,'validation_images.npy'),(text,'validation_texts.npy')]:
            expected_array=np.load(folder/name,allow_pickle=False)
            if actual.shape!=expected_array.shape or not np.isfinite(actual).all() or not np.allclose(actual,expected_array,rtol=1e-5,atol=1e-5):
                raise RuntimeError('Live adapted to fresh BLIP embedding parity failed')
        return fresh,{'live_to_fresh_validation_parity':'PASS','adapter_sha256':manifest['files'][0]['sha256']}
    except Exception as exc:
        traceback.clear_frames(exc.__traceback__);fresh=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        raise

def subset_gallery(records,scores,n_images):
    sub_records=records[:n_images]
    texts,owners=gallery(sub_records)
    caption_count=len(texts)
    # Because captions are flattened in record order, the first N images' captions form a prefix.
    return scores[:n_images,:caption_count],owners,caption_count

def full_gallery_top1_hits(scores,owners):
    owners=np.asarray(owners,dtype=int)
    i2t=np.asarray([owners[int(np.argsort(-scores[i],kind="stable")[0])]==i for i in range(scores.shape[0])])
    t2i=np.asarray([int(np.argsort(-scores[:,j],kind="stable")[0])==owners[j] for j in range(scores.shape[1])])
    return i2t,t2i

def image_best_correct_and_wrong(scores,records,texts,owners):
    owners=np.asarray(owners)
    rows=[]
    for i,row in enumerate(scores):
        correct=np.flatnonzero(owners==i)
        wrong=np.flatnonzero(owners!=i)
        best_correct=int(correct[np.argmax(row[correct])])
        best_wrong=int(wrong[np.argmax(row[wrong])])
        order=np.argsort(-row,kind="stable")
        correct_rank=int(np.flatnonzero(order==best_correct)[0])+1
        rows.append({
            "image_index":i,
            "image_id":records[i]["image_id"],
            "correct_rank":correct_rank,
            "best_correct_caption":texts[best_correct],
            "best_wrong_caption":texts[best_wrong],
            "correct_score":float(row[best_correct]),
            "wrong_score":float(row[best_wrong]),
            "margin":float(row[best_correct]-row[best_wrong]),
        })
    return pd.DataFrame(rows)

def verify_index_pair(root,image_entry,text_entry,reference):
    for entry in (image_entry,text_entry):
        path=Path(root)/entry["path"]
        if path.stat().st_size!=entry["bytes"] or sha256_file(path)!=entry["sha256"]:
            raise ValueError("Index digest/size mismatch")
    img=np.load(Path(root)/image_entry["path"],allow_pickle=False)
    txt=np.load(Path(root)/text_entry["path"],allow_pickle=False)
    if img.ndim!=2 or txt.ndim!=2 or img.shape[1]!=txt.shape[1] or not np.isfinite(img).all() or not np.isfinite(txt).all():
        raise ValueError("Index shape or finite-value mismatch")
    replay=img@txt.T
    if replay.shape!=reference.shape or not np.allclose(replay,reference,atol=1e-6,rtol=1e-6):
        raise RuntimeError("Index reload score parity failed")
    for axis in (0,1):
        if not np.array_equal(np.argsort(-replay,axis=axis,kind="stable"),np.argsort(-reference,axis=axis,kind="stable")):
            raise RuntimeError("Index reload ranking parity failed")
    return img,txt

def first_words(text,n=5):
    return " ".join(str(text).split()[:n])

def safe_extract_zip(path,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist();names=set()
        if len(infos)>1000 or sum(i.file_size for i in infos)>2*1024**3:raise ValueError('Archive exceeds bounds')
        for info in infos:
            rel=PurePosixPath(info.filename)
            if rel.is_absolute() or '..' in rel.parts or '\\' in info.filename or ':' in info.filename:raise ValueError(f'Unsafe ZIP member path {info.filename!r}; use relative paths inside the archive')
            if str(rel).casefold() in names:raise ValueError(f'Duplicate ZIP member {info.filename!r}')
            names.add(str(rel).casefold())
            if (info.external_attr>>16)&0o170000==0o120000:raise ValueError(f'ZIP symlink {info.filename!r} refused; store the file itself')
        root=Path(tempfile.mkdtemp(prefix='dataset-',dir=destination));archive.extractall(root)
    return root

def load_byod(path):
    root=safe_extract_zip(path,WORK_ROOT/'byod');files=list(root.rglob('records.jsonl'))
    if len(files)!=1:raise ValueError(f'The archive must contain exactly one records.jsonl (found {len(files)})')
    base=files[0].parent.resolve();roles={};ids=set();paths=set();pixels=set();total_pixels=0;count=0
    for lineno,line in enumerate(files[0].read_text(encoding='utf-8').splitlines(),start=1):
        if not line.strip():continue
        count+=1
        if count>128:raise ValueError(f'records.jsonl line {lineno}: at most 128 BYOD images are supported; remove rows')
        try:row=json.loads(line)
        except json.JSONDecodeError as exc:raise ValueError(f'records.jsonl line {lineno}: not valid JSON ({exc.msg})') from None
        if not isinstance(row,dict):raise ValueError(f'records.jsonl line {lineno}: each line must be one JSON object')
        rid=row.get('id');filename=row.get('file');role=row.get('split','test')
        where=f'records.jsonl line {lineno} (id={rid!r})'
        if not isinstance(rid,str) or not rid.strip():raise ValueError(f'{where}: "id" must be a non-empty string')
        if rid in ids:raise ValueError(f'{where}: duplicate id; every image needs a unique id')
        if not isinstance(filename,str) or not filename or '\\' in filename or ':' in filename:
            raise ValueError(f'{where}: "file" must be a relative path such as "images/{rid}.jpg" (no backslashes or drive letters)')
        path=(base/filename).resolve()
        if base not in path.parents:raise ValueError(f'{where}: file {filename!r} points outside the archive')
        if not path.is_file():raise ValueError(f'{where}: file {filename!r} is not in the archive')
        if path in paths:raise ValueError(f'{where}: file {filename!r} is already used by an earlier row')
        if role not in ('train','validation','test'):raise ValueError(f'{where}: split {role!r} must be train, validation or test')
        captions=row.get('captions')
        if not isinstance(captions,list) or not 1<=len(captions)<=5:
            got=f'{len(captions)} captions' if isinstance(captions,list) else type(captions).__name__
            raise ValueError(f'{where}: "captions" must be a list of 1..5 strings (got {got})')
        cleaned=[]
        for k,text in enumerate(captions):
            if not isinstance(text,str):raise ValueError(f'{where}: caption {k} is {type(text).__name__}, not a string')
            text=' '.join(text.split())
            if not text:raise ValueError(f'{where}: caption {k} is empty')
            if len(text)>256:raise ValueError(f'{where}: caption {k} has {len(text)} characters (limit 256)')
            if text in cleaned:raise ValueError(f'{where}: caption {k} repeats an earlier caption of the same image')
            cleaned.append(text)
        with Image.open(path) as original:
            if min(original.size)<16 or max(original.size)>4096:
                raise ValueError(f'{where}: image {filename!r} is {original.width}x{original.height}; each side must be 16..4096 px')
            total_pixels+=original.width*original.height
            if total_pixels>100_000_000:raise ValueError(f'{where}: total decoded pixels exceed 100 million; use fewer or smaller images')
            image=original.convert('RGB');digest=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()
        if digest in pixels:raise ValueError(f'{where}: image {filename!r} has the same decoded pixels as an earlier row; remove the duplicate')
        ids.add(rid);paths.add(path);pixels.add(digest)
        roles.setdefault(role,[]).append({'id':rid,'image_id':rid,'path':str(path),'captions':cleaned,'split':role,
            'category':str(row.get('category','unknown')),'pixel_sha256':digest,'sha256':sha256_file(path)})
    available={p.resolve() for p in root.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}}
    undeclared=sorted(p.relative_to(base).as_posix() if base in p.parents else p.name for p in available-paths)
    if undeclared:raise ValueError(f'Every archive image must be listed in records.jsonl; undeclared: {undeclared[:5]}')
    not_images=sorted(p.name for p in paths-available)
    if not_images:raise ValueError(f'Declared files must be .png/.jpg/.jpeg/.webp/.bmp/.tif images: {not_images[:5]}')
    counts={role:len(roles.get(role,[])) for role in ('train','validation','test')}
    if counts['test']<2:raise ValueError(f'The test/evaluation gallery needs at least two images (found {counts["test"]})')
    if WORKSHOP_TIER=='FULL' and any(n<2 for n in counts.values()):
        raise ValueError(f'FULL needs at least two images in every train/validation/test role (found {counts}); use STANDARD for an evaluation-only gallery')
    return roles

def export_byod_index(folder,name,images,texts,spec,reference):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    entries=[]
    for suffix,values in [('images',images),('texts',texts)]:
        path=folder/f'{name}_{suffix}.npy';np.save(path,values)
        entries.append({'path':path.name,'bytes':path.stat().st_size,'sha256':sha256_file(path)})
    verify_index_pair(folder,entries[0],entries[1],reference)
    return {'model_id':spec['model_id'],'revision':spec['revision'],'images':entries[0],'texts':entries[1]}

def run_byod_retrieval(roles):
    parent=OUTPUT_ROOT/'byod';parent.mkdir(exist_ok=True);out=Path(tempfile.mkdtemp(prefix='run-',dir=parent))
    records=roles['test'];texts,owners=gallery(records);blip=None;raw=None
    models={'siglip2':SIGLIP2,'siglip1':SIGLIP1,'blip_itc':BLIP};indexes={};scores={};metrics=[];reranks=[]
    try:
        adapter=None
        if WORKSHOP_TIER=='FULL':
            blip=BlipWorkshop()
            report=adapt_blip(blip,roles['train'],roles['validation'],epochs=BLIP_EPOCHS,lr=BLIP_LEARNING_RATE,
                batch_size=BLIP_BATCH_SIZE,n_layers=BLIP_TRAINABLE_TEXT_LAYERS,seed=SEED)
            adapter=save_verified_blip_adapter(blip,report,out/'adapter',roles['validation'])
            (out/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            blip=None;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        frozen={'models':models,'tier':WORKSHOP_TIER,'roles':roles,'adapter':adapter,'rerank_top_k':RERANK_TOP_K,
                'recall_ks':list(RECALL_KS),'zip_sha256':sha256_file(BYOD_ZIP_PATH),
                'runtime':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':DEVICE}}
        (out/'frozen_experiment.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
        # First access to evaluation embeddings follows the freeze.
        for name,spec,root in [('siglip2',SIGLIP2,SIGLIP2_ROOT),('siglip1',SIGLIP1,SIGLIP1_ROOT)]:
            images,text,timing=siglip_embed(spec,root,records,texts);score=images@text.T;scores[name]=score
            indexes[name]=export_byod_index(out/'index',name,images,text,spec,score)
            metrics.append({'model':name,**retrieval_metrics(score,owners),**timing})
        blip=BlipWorkshop();raw,image_seconds=blip.raw_image_embeds(records)
        images=blip.image_features(raw);text,text_seconds=blip.text_features(texts);scores['blip_itc']=images@text.T
        indexes['blip_itc']=export_byod_index(out/'index','blip_itc',images,text,BLIP,scores['blip_itc'])
        metrics.append({'model':'blip_itc',**retrieval_metrics(scores['blip_itc'],owners)})
        for name,score in scores.items():reranks.append({'candidate_source':name,**rerank_with_blip(blip,raw,texts,owners,score,RERANK_TOP_K)})
        blip=None;raw=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        parity=None
        if WORKSHOP_TIER=='FULL':
            blip,parity=load_verified_blip_adapter(out/'adapter',roles['validation'])
            raw,_=blip.raw_image_embeds(records);images=blip.image_features(raw);text,_=blip.text_features(texts)
            score=images@text.T;indexes['adapted_blip']=export_byod_index(out/'index','adapted_blip',images,text,BLIP,score)
            metrics.append({'model':'adapted_blip',**retrieval_metrics(score,owners)})
            for name,coarse in [('adapted_blip',score),('siglip2',scores['siglip2'])]:
                reranks.append({'candidate_source':name,'reranker':'adapted_blip',**rerank_with_blip(blip,raw,texts,owners,coarse,RERANK_TOP_K)})
        manifest={'models':indexes,'image_ids':[r['image_id'] for r in records],'captions':texts,'caption_owners':owners,'adapter_parity':parity}
        (out/'index'/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        (out/'results.json').write_text(json.dumps({'retrieval':metrics,'reranking':reranks,'adapter_parity':parity},indent=2),encoding='utf-8')
        pd.DataFrame(metrics).to_csv(out/'retrieval.csv',index=False);pd.DataFrame(reranks).to_csv(out/'reranking.csv',index=False)
        return out
    except Exception as exc:
        traceback.clear_frames(exc.__traceback__);raise
    finally:
        blip=None;raw=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

def files_for_bundle(root,started):
    current=[];stale=[]
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            (current if p.stat().st_mtime>=started-1.0 else stale).append(p)
    return current,stale


# ---- Stages ------------------------------------------------------------------------------------------
STAGES = {}


def stage(name):
    def register(function):
        STAGES[name] = function
        return function
    return register


def show(frame):
    # The notebook displays the written table itself; the stage log records only its extent.
    print(f"table: {frame.shape[0]} rows x {frame.shape[1]} columns")


@stage("environment")
def stage_environment(args):
    load_transformers()
    import importlib.metadata as metadata
    versions={dist:metadata.version(dist) for dist in (
        "torch","torchvision","torchaudio","transformers","huggingface-hub","safetensors","numpy",
        "pillow","pyarrow","pandas","matplotlib","sentencepiece","protobuf")}
    info={
        "python":platform.python_version(),
        "executable":sys.executable,
        "versions":versions,
        "device":DEVICE,
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    pins=dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)",(Path(__file__).parent/"requirements.lock.txt").read_text(encoding="utf-8"),re.M))
    drift={name:(metadata.version(name),pin) for name,pin in pins.items() if metadata.version(name)!=pin}
    if drift:
        raise RuntimeError(f"Installed versions differ from the hash lock: {drift}")
    info["lock_pins_checked"]=len(pins)
    write_state("environment.json",info)
    print({"python":info["python"],"torch":torch.__version__,"device":DEVICE,"gpu":info["gpu"],
           "lock_pins_checked":len(pins)})


@stage("manifests")
def stage_manifests(args):
    rows=[]
    for name,spec in [("siglip2",SIGLIP2),("siglip1",SIGLIP1),("blip",BLIP)]:
        primary=max(spec["files"],key=lambda f:f[1])
        rows.append({"model":name,"model_id":spec["model_id"],"revision":spec["revision"],
                     "files":len(spec["files"]),"primary_weight":primary[0],"primary_bytes":primary[1],
                     "primary_sha256":primary[2],"embedding_dim":spec["dim"]})
    write_state("model_manifests.json",rows)
    print("Model identities pinned:",[r["model_id"] for r in rows])


@stage("corpus")
def stage_corpus(args):
    from huggingface_hub import hf_hub_download
    shard_path=Path(hf_hub_download(
        repo_id=CORPUS_REPO,
        repo_type="dataset",
        filename=CORPUS_PATH,
        revision=CORPUS_REVISION,
        local_dir=str(WORK_ROOT/"vizwiz"),
    ))
    if shard_path.stat().st_size!=CORPUS_BYTES or sha256_file(shard_path)!=CORPUS_SHA256:
        raise ValueError("VizWiz parquet digest/size mismatch")
    write_state("corpus.json",{"path":str(shard_path)})
    print({
        "path":str(shard_path),
        "bytes":shard_path.stat().st_size,
        "sha256":sha256_file(shard_path),
    })


@stage("split")
def stage_split(args):
    global pf
    import pyarrow.parquet as pq
    shard_path=Path(read_state("corpus.json")["path"])
    (WORK_ROOT/"vizwiz"/"images").mkdir(parents=True,exist_ok=True)
    pf=pq.ParquetFile(shard_path)

    group0=[r for r in read_group(0) if r["captions"]]
    group1=[r for r in read_group(1) if r["captions"]]

    if len(group0)!=318 or len(group1)!=321:
        raise RuntimeError(f"captioned-row counts drifted: {len(group0)} / {len(group1)}")

    pool=sorted(group0,key=lambda r:r["image_id"])
    random.Random(SPLIT_SEED).shuffle(pool)

    train_records=pool[:208]
    validation_records=pool[208:248]
    core_test=pool[248:318]
    test_records=core_test+sorted(group1,key=lambda r:r["image_id"])

    for split,records in [
        ("train",train_records),
        ("validation",validation_records),
        ("test",test_records),
    ]:
        for i,r in enumerate(records):
            r["id"]=f"{split}-{i:04d}"
            r["split"]=split

    if len(test_records)!=391:
        raise RuntimeError("test gallery must contain exactly 391 photographs")

    all_seen={}
    for split,records in [("train",train_records),("validation",validation_records),("test",test_records)]:
        for r in records:
            prior=all_seen.get(r["pixel_sha256"])
            if prior and prior!=split:
                raise RuntimeError(f"decoded-pixel leakage across {prior}/{split}")
            all_seen[r["pixel_sha256"]]=split

    test_caption_count=sum(len(r["captions"]) for r in test_records)
    if test_caption_count!=1737:
        raise RuntimeError(f"expected 1737 test captions, got {test_caption_count}")

    print({
        "train":[len(train_records),sum(len(r["captions"]) for r in train_records)],
        "validation":[len(validation_records),sum(len(r["captions"]) for r in validation_records)],
        "test":[len(test_records),test_caption_count],
    })

    # Dataset provenance (former cell 8f39c22b).
    dataset_manifest={
        "source":CORPUS_REPO,
        "revision":CORPUS_REVISION,
        "shard":CORPUS_PATH,
        "shard_bytes":CORPUS_BYTES,
        "shard_sha256":CORPUS_SHA256,
        "split_seed":SPLIT_SEED,
        "roles":{
            "train":[r["image_id"] for r in train_records],
            "validation":[r["image_id"] for r in validation_records],
            "test":[r["image_id"] for r in test_records],
        },
        "counts":{
            "train_images":len(train_records),
            "validation_images":len(validation_records),
            "test_images":len(test_records),
            "test_captions":test_caption_count,
        },
    }
    dataset_manifest["role_digest"]=hashlib.sha256(
        json.dumps(dataset_manifest["roles"],sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()

    (OUTPUT_ROOT/"data"/"dataset_manifest.json").write_text(
        json.dumps(dataset_manifest,indent=2),encoding="utf-8"
    )
    write_state("records.json",{"train":train_records,"validation":validation_records,"test":test_records})
    print("role_digest",dataset_manifest["role_digest"])


@stage("models")
def stage_models(args):
    global hf_hub_download
    from huggingface_hub import hf_hub_download
    for name,spec in [("siglip2",SIGLIP2),("siglip1",SIGLIP1),("blip",BLIP)]:
        stage_snapshot(spec,model_root(name))
    print("All model snapshots verified.")


def dual_encoder(name):
    return {"siglip2":SIGLIP2,"siglip1":SIGLIP1}[name]


@stage("validation")
def stage_validation(args):
    load_transformers()
    validation_records=records()["validation"]
    validation_texts,validation_owners=gallery(validation_records)

    validation_scores={}
    validation_runtime={}

    for name in ("siglip2","siglip1"):
        v_img,v_txt,timing=siglip_embed(
            dual_encoder(name),model_root(name),validation_records,validation_texts
        )
        validation_scores[name]=v_img@v_txt.T
        validation_runtime[name]=timing
        del v_img,v_txt

    blip_val=BlipWorkshop()
    val_raw,val_image_seconds=blip_val.raw_image_embeds(validation_records)
    val_img=blip_val.image_features(val_raw)
    val_txt,val_text_seconds=blip_val.text_features(validation_texts)
    validation_scores["blip_itc"]=val_img@val_txt.T
    validation_runtime["blip_itc"]={
        "image_embedding_seconds":val_image_seconds,
        "text_embedding_seconds":val_text_seconds,
    }

    validation_rows=[]
    for name,scores in validation_scores.items():
        validation_rows.append({"model":name,**retrieval_metrics(scores,validation_owners)})
        save_scores(f"validation_{name}",scores)
    validation_table=pd.DataFrame(validation_rows)
    validation_table.to_csv(OUTPUT_ROOT/"validation"/"retrieval_metrics.csv",index=False)
    write_state("validation_runtime.json",validation_runtime)
    show(validation_table)


@stage("validation-rerank")
def stage_validation_rerank(args):
    load_transformers()
    validation_records=records()["validation"]
    validation_texts,validation_owners=gallery(validation_records)
    blip_val=BlipWorkshop()
    val_raw,_seconds=blip_val.raw_image_embeds(validation_records)
    validation_rerank=[]
    for name in ("siglip2","siglip1","blip_itc"):
        scores=load_scores(f"validation_{name}")
        result=rerank_with_blip(
            blip_val,val_raw,validation_texts,validation_owners,scores,RERANK_TOP_K
        )
        validation_rerank.append({"candidate_source":name,**result})
    validation_rerank_table=pd.DataFrame(validation_rerank)
    validation_rerank_table.to_csv(OUTPUT_ROOT/"validation"/"reranking.csv",index=False)
    show(validation_rerank_table)


@stage("candidate-oracle")
def stage_candidate_oracle(args):
    validation_owners=gallery(records()["validation"])[1]
    scores=load_scores("validation_siglip2")
    rows=[{"k":k,**candidate_oracle(scores,validation_owners,k)} for k in sorted({5,int(args.k)})]
    show(pd.DataFrame(rows))
    print("Only the candidate-pool size changed; no model was loaded and no export was written.")


@stage("adapt")
def stage_adapt(args):
    if WORKSHOP_TIER!="FULL":
        print("STANDARD: BLIP adaptation skipped.")
        return
    load_transformers()
    data=records()
    blip_val=BlipWorkshop()
    blip_adapter_report=adapt_blip(
        blip_val,
        data["train"],
        data["validation"],
        epochs=BLIP_EPOCHS,
        lr=BLIP_LEARNING_RATE,
        batch_size=BLIP_BATCH_SIZE,
        n_layers=BLIP_TRAINABLE_TEXT_LAYERS,
        seed=SEED,
    )

    artifact_dir=OUTPUT_ROOT/"artifacts"/"blip"
    save_verified_blip_adapter(blip_val,blip_adapter_report,artifact_dir,data["validation"])
    weights_path=artifact_dir/"adapter.safetensors"
    (OUTPUT_ROOT/"adaptation"/"blip"/"training_history.json").write_text(
        json.dumps(blip_adapter_report,indent=2,default=str),encoding="utf-8"
    )
    print({
        "best_epoch":blip_adapter_report["best_epoch"],
        "adapter_bytes":weights_path.stat().st_size,
    })


def full_tier_file(path):
    """A FULL-tier file written during this Run all, or None (STANDARD, or a file from an earlier run)."""
    path=Path(path)
    if WORKSHOP_TIER!="FULL" or not path.is_file() or path.stat().st_mtime<RUN_STARTED-1.0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@stage("freeze")
def stage_freeze(args):
    dataset_manifest=json.loads((OUTPUT_ROOT/"data"/"dataset_manifest.json").read_text(encoding="utf-8"))
    blip_artifact_manifest=full_tier_file(OUTPUT_ROOT/"artifacts"/"blip"/"manifest.json")
    if WORKSHOP_TIER=="FULL" and blip_artifact_manifest is None:
        raise RuntimeError("FULL: the BLIP adapter of this Run all is missing; run the adaptation cell first.")
    frozen={
        "notebook_spec":"2.1",
        "profile":"MULTI-CAPABILITY",
        "mode":"WORKSHOP",
        "tier":WORKSHOP_TIER,
        "dataset":dataset_manifest,
        "models":{
            "siglip2":{
                "id":SIGLIP2["model_id"],"revision":SIGLIP2["revision"],
                "weight_sha256":"612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
                "embedding_dim":768,
            },
            "siglip1":{
                "id":SIGLIP1["model_id"],"revision":SIGLIP1["revision"],
                "weight_sha256":"f0cee7c815135c44a515eff72ab3040499744920442bc25567cd04efc93f8f65",
                "embedding_dim":768,
            },
            "blip":{
                "id":BLIP["model_id"],"revision":BLIP["revision"],
                "weight_sha256":"017fb3e7f4e125f13a8a4717f1402dbe0d0bb877474b4a203db13a4447b0227f",
                "embedding_dim":256,
            },
        },
        "evaluation":{
            "recall_k":list(RECALL_KS),
            "rerank_top_k":RERANK_TOP_K,
            "query_caption_rule":"first caption per photograph",
            "gallery_sizes":list(GALLERY_SIZES),
            "normalization":"L2 unit norm; cosine via dot product",
        },
        "blip_adapter":blip_artifact_manifest,
    }
    freeze_path=OUTPUT_ROOT/"frozen"/"frozen_experiment.json"
    freeze_path.write_text(json.dumps(frozen,indent=2,default=str),encoding="utf-8")
    print("Frozen:",freeze_path)


def require_frozen():
    path=OUTPUT_ROOT/"frozen"/"frozen_experiment.json"
    if not path.is_file() or path.stat().st_mtime<RUN_STARTED-1.0:
        raise RuntimeError("Freeze the experiment (section 12) before computing test results.")
    return json.loads(path.read_text(encoding="utf-8"))


@stage("test-baselines")
def stage_test_baselines(args):
    require_frozen()
    data=records()
    test_records=data["test"]
    chance=chance_baseline(test_records)
    colour=colour_keyword_baseline(data["train"],test_records)
    write_state("test_baselines.json",{"chance":chance,"colour_keyword":colour})
    show(pd.DataFrame([
        {"model":"chance",**chance},
        {"model":"colour_keyword",**colour},
    ]))


INDEX_FILES={"siglip2":"siglip2","siglip1":"siglip1","blip_itc":"blip_itc"}


@stage("test-index")
def stage_test_index(args):
    require_frozen()
    load_transformers()
    name=args.model
    test_records=records()["test"]
    test_texts,test_owners=gallery(test_records)
    if name in ("siglip2","siglip1"):
        img,txt,timing=siglip_embed(
            dual_encoder(name),model_root(name),test_records,test_texts
        )
    elif name=="blip_itc":
        blip_frozen=BlipWorkshop()
        blip_raw,blip_image_seconds=blip_frozen.raw_image_embeds(test_records)
        img=blip_frozen.image_features(blip_raw)
        txt,blip_text_seconds=blip_frozen.text_features(test_texts)
        timing={"image_embedding_seconds":blip_image_seconds,"text_embedding_seconds":blip_text_seconds}
    else:
        raise ValueError(f"unknown index {name!r}")
    scores=img@txt.T
    metrics=retrieval_metrics(scores,test_owners)

    np.save(OUTPUT_ROOT/"index"/f"{INDEX_FILES[name]}_images.npy",img)
    np.save(OUTPUT_ROOT/"index"/f"{INDEX_FILES[name]}_texts.npy",txt)
    save_scores(f"test_{name}",scores)
    write_state(f"test_{name}.json",{"metrics":metrics,"timing":timing})

    print(metrics)


@stage("retrieval-table")
def stage_retrieval_table(args):
    baselines=read_state("test_baselines.json")
    metrics={name:read_state(f"test_{name}.json")["metrics"] for name in ("siglip2","siglip1","blip_itc")}
    retrieval_rows=[
        {"model":"chance","dim":None,**baselines["chance"]},
        {"model":"colour_keyword","dim":None,**baselines["colour_keyword"]},
        {"model":"siglip_v1","dim":768,**metrics["siglip1"]},
        {"model":"siglip2","dim":768,**metrics["siglip2"]},
        {"model":"blip_itc","dim":256,**metrics["blip_itc"]},
    ]
    retrieval_table=pd.DataFrame(retrieval_rows)
    retrieval_table.to_csv(OUTPUT_ROOT/"test"/"retrieval_metrics.csv",index=False)
    write_state("retrieval_rows.json",retrieval_rows)
    show(retrieval_table)


RERANK_VIEW=[
    "candidate_source","coarse_i2t_r1","itm_i2t_recall_at_1","delta_i2t_r1",
    "coarse_t2i_r1_canonical","itm_t2i_recall_at_1","delta_t2i_r1",
    "i2t_candidate_oracle","t2i_candidate_oracle","itm_pair_accuracy","pair_evaluations","seconds",
]


@stage("rerank")
def stage_rerank(args):
    require_frozen()
    load_transformers()
    test_records=records()["test"]
    test_texts,test_owners=gallery(test_records)
    blip_frozen=BlipWorkshop()
    blip_raw,_seconds=blip_frozen.raw_image_embeds(test_records)
    rerank_rows=[]
    for name,key in [
        ("siglip_v1","siglip1"),
        ("siglip2","siglip2"),
        ("blip_itc","blip_itc"),
    ]:
        scores=load_scores(f"test_{key}")
        result=rerank_with_blip(
            blip_frozen,blip_raw,test_texts,test_owners,scores,RERANK_TOP_K
        )
        coarse=retrieval_metrics(scores,test_owners)
        rerank_rows.append({
            "candidate_source":name,
            **result,
            # All 1,737 caption queries (the section 17 value); not the baseline for itm_t2i_recall_at_1.
            "coarse_t2i_r1_all_captions":coarse["t2i_recall_at_1"],
        })

    rerank_table=pd.DataFrame(rerank_rows)
    rerank_table.to_csv(OUTPUT_ROOT/"test"/"reranking_metrics.csv",index=False)
    write_state("rerank_rows.json",rerank_rows)
    show(rerank_table[RERANK_VIEW].round(4))


@stage("adapted-test")
def stage_adapted_test(args):
    result={"adapted_blip_metrics":None,"adapted_rerank_rows":[],"blip_reload_parity":None}
    if WORKSHOP_TIER=="FULL":
        require_frozen()
        load_transformers()
        data=records()
        test_records=data["test"]
        test_texts,test_owners=gallery(test_records)
        fresh,blip_reload_parity=load_verified_blip_adapter(OUTPUT_ROOT/"artifacts"/"blip",data["validation"])
        adapted_raw,adapted_image_seconds=fresh.raw_image_embeds(test_records)
        adapted_img=fresh.image_features(adapted_raw)
        adapted_txt,adapted_text_seconds=fresh.text_features(test_texts)
        adapted_scores=adapted_img@adapted_txt.T
        adapted_blip_metrics=retrieval_metrics(adapted_scores,test_owners)

        adapted_rerank_rows=[]
        for source,scores in [
            ("adapted_blip_itc",adapted_scores),
            ("frozen_siglip2",load_scores("test_siglip2")),
        ]:
            rerank=rerank_with_blip(
                fresh,adapted_raw,test_texts,test_owners,scores,RERANK_TOP_K
            )
            adapted_rerank_rows.append({"candidate_source":source,**rerank})
        result={"adapted_blip_metrics":adapted_blip_metrics,"adapted_rerank_rows":adapted_rerank_rows,
                "blip_reload_parity":blip_reload_parity}
        print("Adapted BLIP:",adapted_blip_metrics)
        show(pd.DataFrame(adapted_rerank_rows))
    else:
        print("STANDARD: adapted BLIP test skipped.")
    write_state("adapted_test.json",result)


def test_score_grids():
    return [
        ("siglip_v1",load_scores("test_siglip1")),
        ("siglip2",load_scores("test_siglip2")),
        ("blip_itc",load_scores("test_blip_itc")),
    ]


@stage("gallery-size")
def stage_gallery_size(args):
    test_records=records()["test"]
    gallery_rows=[]
    grids=test_score_grids()
    for n in GALLERY_SIZES:
        for name,scores in grids:
            sub,owners,n_caps=subset_gallery(test_records,scores,n)
            m=retrieval_metrics(sub,owners)
            gallery_rows.append({
                "gallery_images":n,"captions":n_caps,"model":name,
                "chance_t2i_r1":1/n,
                "i2t_r1":m["i2t_recall_at_1"],
                "t2i_r1":m["t2i_recall_at_1"],
                "rsum":m["rsum"],
            })
    gallery_table=pd.DataFrame(gallery_rows)
    gallery_table.to_csv(OUTPUT_ROOT/"test"/"gallery_size_metrics.csv",index=False)
    write_state("gallery_rows.json",gallery_rows)
    show(gallery_table)


def category_tables(test_records,test_owners,test_texts,train_records,grids):
    category_rows=[]
    for model,scores in grids:
        full_i2t,full_t2i=full_gallery_top1_hits(scores,test_owners)
        for category in sorted({r["category"] for r in test_records}):
            image_indices=[i for i,r in enumerate(test_records) if r["category"]==category]
            members=set(image_indices)
            text_indices=[j for j,o in enumerate(test_owners) if o in members]
            remap={old:new for new,old in enumerate(image_indices)}
            sub=scores[np.ix_(image_indices,text_indices)]
            owners=[remap[test_owners[j]] for j in text_indices]
            m=retrieval_metrics(sub,owners)
            category_rows.append({
                "model":model,"category":category,"n_images":len(image_indices),"n_captions":len(text_indices),
                "full_gallery_i2t_r1":float(np.mean(full_i2t[image_indices])),
                "full_gallery_t2i_r1":float(np.mean(full_t2i[text_indices])),
                "subgallery_i2t_r1":m["i2t_recall_at_1"],"subgallery_t2i_r1":m["t2i_recall_at_1"],
                "subgallery_rsum":m["rsum"],"subgallery_chance_t2i_r1":1/len(image_indices),
            })
    category_table=pd.DataFrame(category_rows)

    train_lengths=[len(c.split()) for r in train_records for c in r["captions"]]
    q1,q2=np.quantile(train_lengths,[1/3,2/3])

    def length_group(text):
        n=len(text.split())
        return "short" if n<=q1 else "medium" if n<=q2 else "long"

    length_rows=[]
    owners_arr=np.asarray(test_owners)
    for model,scores in grids:
        for group in ("short","medium","long"):
            indices=[j for j,t in enumerate(test_texts) if length_group(t)==group]
            ranks=[]
            for j in indices:
                order=np.argsort(-scores[:,j],kind="stable")
                ranks.append(int(np.flatnonzero(order==owners_arr[j])[0])+1)
            length_rows.append({
                "model":model,"caption_length":group,"n":len(indices),
                "t2i_r1":float(np.mean(np.asarray(ranks)==1)),
                "t2i_median_rank":float(np.median(ranks)),
            })
    length_table=pd.DataFrame(length_rows)
    return category_table,length_table


@stage("categories")
def stage_categories(args):
    data=records()
    test_records=data["test"]
    test_texts,test_owners=gallery(test_records)
    category_table,length_table=category_tables(test_records,test_owners,test_texts,data["train"],test_score_grids())
    category_table.to_csv(OUTPUT_ROOT/"test"/"category_metrics.csv",index=False)
    length_table.to_csv(OUTPUT_ROOT/"test"/"caption_length_metrics.csv",index=False)
    show(category_table)
    show(length_table)


def hard_negative_tables(test_records,test_texts,test_owners,grids):
    scores=dict(grids)
    hard_tables={
        "siglip_v1":image_best_correct_and_wrong(scores["siglip_v1"],test_records,test_texts,test_owners),
        "siglip2":image_best_correct_and_wrong(scores["siglip2"],test_records,test_texts,test_owners),
        "blip_itc":image_best_correct_and_wrong(scores["blip_itc"],test_records,test_texts,test_owners),
    }
    rank_compare=hard_tables["siglip2"][["image_index","correct_rank"]].rename(
        columns={"correct_rank":"siglip2_rank"}
    ).merge(
        hard_tables["siglip_v1"][["image_index","correct_rank"]].rename(
            columns={"correct_rank":"siglip1_rank"}
        ),
        on="image_index"
    ).merge(
        hard_tables["blip_itc"][["image_index","correct_rank"]].rename(
            columns={"correct_rank":"blip_rank"}
        ),
        on="image_index"
    )

    rank_compare["siglip2_vs_v1"]=rank_compare["siglip1_rank"]-rank_compare["siglip2_rank"]
    rank_compare["siglip2_vs_blip"]=rank_compare["blip_rank"]-rank_compare["siglip2_rank"]
    return hard_tables,rank_compare


@stage("hard-negatives")
def stage_hard_negatives(args):
    test_records=records()["test"]
    test_texts,test_owners=gallery(test_records)
    hard_tables,rank_compare=hard_negative_tables(test_records,test_texts,test_owners,test_score_grids())
    hard_tables["siglip2"].to_csv(OUTPUT_ROOT/"test"/"hard_negatives.csv",index=False)
    rank_compare.to_csv(state_path("rank_compare.csv"),index=False)
    show(rank_compare.sort_values("siglip2_vs_v1",ascending=False).head())


@stage("disagreement-figures")
def stage_disagreement_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    test_records=records()["test"]
    test_texts,test_owners=gallery(test_records)
    hard_tables,rank_compare=hard_negative_tables(test_records,test_texts,test_owners,test_score_grids())
    chosen=[
        ("SigLIP2 advantage over v1",int(rank_compare.loc[rank_compare["siglip2_vs_v1"].idxmax(),"image_index"])),
        ("SigLIP v1 advantage over SigLIP2",int(rank_compare.loc[rank_compare["siglip2_vs_v1"].idxmin(),"image_index"])),
        ("SigLIP2 advantage over BLIP ITC",int(rank_compare.loc[rank_compare["siglip2_vs_blip"].idxmax(),"image_index"])),
    ]

    written=[]
    for title,i in chosen:
        r=test_records[i]
        with Image.open(r["path"]) as im:
            fig,ax=plt.subplots(figsize=(7,5))
            ax.imshow(im.convert("RGB"))
            ax.axis("off")
            ax.set_title(title)
            text=(
                f"Gold: {r['captions'][0]}\n\n"
                f"SigLIP2 wrong: {hard_tables['siglip2'].iloc[i]['best_wrong_caption']}\n"
                f"SigLIP v1 wrong: {hard_tables['siglip_v1'].iloc[i]['best_wrong_caption']}\n"
                f"BLIP ITC wrong: {hard_tables['blip_itc'].iloc[i]['best_wrong_caption']}"
            )
            fig.text(0.02,0.01,text,fontsize=8,va="bottom")
            plt.tight_layout(rect=[0,0.17,1,1])
            safe=title.lower().replace(" ","_")
            plt.savefig(OUTPUT_ROOT/"figures"/f"{safe}.png",dpi=150,bbox_inches="tight")
            plt.close(fig)
        written.append({"title":title,"image_index":i,"path":str(OUTPUT_ROOT/"figures"/f"{safe}.png")})
    write_state("disagreement_figures.json",written)
    print("Figures:",[w["path"] for w in written])


def load_index(name):
    return (np.load(OUTPUT_ROOT/"index"/f"{name}_images.npy",allow_pickle=False),
            np.load(OUTPUT_ROOT/"index"/f"{name}_texts.npy",allow_pickle=False))


@stage("runtime-storage")
def stage_runtime_storage(args):
    timing={name:read_state(f"test_{name}.json")["timing"] for name in ("siglip2","siglip1","blip_itc")}
    storage_rows=[
        {
            "model":"siglip_v1","dim":768,
            "bytes_per_float32_vector":768*4,
            "one_million_vectors_GB":768*4*1_000_000/1e9,
            **timing["siglip1"],
        },
        {
            "model":"siglip2","dim":768,
            "bytes_per_float32_vector":768*4,
            "one_million_vectors_GB":768*4*1_000_000/1e9,
            **timing["siglip2"],
        },
        {
            "model":"blip_itc","dim":256,
            "bytes_per_float32_vector":256*4,
            "one_million_vectors_GB":256*4*1_000_000/1e9,
            "image_embedding_seconds":timing["blip_itc"]["image_embedding_seconds"],
            "text_embedding_seconds":timing["blip_itc"]["text_embedding_seconds"],
        },
    ]
    storage_table=pd.DataFrame(storage_rows)
    storage_table.to_csv(OUTPUT_ROOT/"test"/"runtime_storage.csv",index=False)
    write_state("storage_rows.json",storage_rows)
    show(storage_table)

    # Dense similarity search cost for one text query across the 391 image vectors.
    search=[]
    for name,(img,txt) in [
        ("siglip_v1",load_index("siglip1")),
        ("siglip2",load_index("siglip2")),
        ("blip_itc",load_index("blip_itc")),
    ]:
        q=txt[0]
        trials=[]
        for _ in range(100):
            t0=time.perf_counter()
            _=img@q
            trials.append(time.perf_counter()-t0)
        print(name,"median matrix-search ms/query",1000*np.median(trials))
        search.append({"model":name,"median_matrix_search_ms_per_query":1000*float(np.median(trials))})
    write_state("matrix_search.json",search)


@stage("index-manifest")
def stage_index_manifest(args):
    test_records=records()["test"]
    image_ids=[r["image_id"] for r in test_records]
    caption_ids=[]
    caption_owners=[]
    for i,r in enumerate(test_records):
        for j,_caption in enumerate(r["captions"]):
            caption_ids.append(f"{r['image_id']}:{j}")
            caption_owners.append(i)

    (OUTPUT_ROOT/"index"/"image_ids.json").write_text(json.dumps(image_ids,indent=2),encoding="utf-8")
    (OUTPUT_ROOT/"index"/"caption_ids.json").write_text(json.dumps(caption_ids,indent=2),encoding="utf-8")
    (OUTPUT_ROOT/"index"/"caption_owners.json").write_text(json.dumps(caption_owners),encoding="utf-8")

    index_manifest={
        "item_order":{
            "images":"image_ids.json",
            "captions":"caption_ids.json",
            "owners":"caption_owners.json",
        },
        "models":{},
    }
    for name,spec,img_file,txt_file,dim in [
        ("siglip2",SIGLIP2,"siglip2_images.npy","siglip2_texts.npy",768),
        ("siglip1",SIGLIP1,"siglip1_images.npy","siglip1_texts.npy",768),
        ("blip_itc",BLIP,"blip_itc_images.npy","blip_itc_texts.npy",256),
    ]:
        index_manifest["models"][name]={
            "model_id":spec["model_id"],
            "revision":spec["revision"],
            "embedding_dim":dim,
            "normalization":"L2",
            "dtype":"float32",
            "image_file":{
                "path":img_file,
                "bytes":(OUTPUT_ROOT/"index"/img_file).stat().st_size,
                "sha256":sha256_file(OUTPUT_ROOT/"index"/img_file),
            },
            "text_file":{
                "path":txt_file,
                "bytes":(OUTPUT_ROOT/"index"/txt_file).stat().st_size,
                "sha256":sha256_file(OUTPUT_ROOT/"index"/txt_file),
            },
        }

    (OUTPUT_ROOT/"index"/"index_manifest.json").write_text(
        json.dumps(index_manifest,indent=2),encoding="utf-8"
    )

    # The reference scores were computed from the in-memory embeddings by the stage that exported them.
    for name in ("siglip2","siglip1","blip_itc"):
        entry=index_manifest["models"][name]
        verify_index_pair(OUTPUT_ROOT/"index",entry["image_file"],entry["text_file"],load_scores(f"test_{name}"))
    print("Full index digest/shape/reload ranking parity: PASS")


@stage("perturbation")
def stage_perturbation(args):
    load_transformers()
    data=records()
    validation_records=data["validation"]
    test_records=data["test"]
    validation_texts,validation_owners=gallery(validation_records)
    siglip2_img=np.load(OUTPUT_ROOT/"index"/"siglip2_images.npy",allow_pickle=False)
    perturbation_table=caption_perturbation(validation_records,validation_texts,validation_owners,
                                            siglip2_img,test_records)
    show(perturbation_table.round(4))


def caption_perturbation(validation_records,validation_texts,validation_owners,siglip2_img,test_records):
    perturbations={
        "original":list(validation_texts),
        "lowercase":[str(t).lower() for t in validation_texts],
        "first_five_words":[first_words(t) for t in validation_texts],
    }
    irrelevant_queries=[
        "a satellite orbiting mars",
        "an underwater coral reef",
    ]

    # One SigLIP 2 pass embeds the validation photographs, every caption variant and the no-match queries.
    all_texts=[t for variant in perturbations.values() for t in variant]+irrelevant_queries
    p_img,p_txt,_p_timing=siglip_embed(SIGLIP2,model_root("siglip2"),validation_records,all_texts)
    n_caps=len(validation_texts)

    perturbation_rows=[]
    for k,(variant,texts) in enumerate(perturbations.items()):
        m=retrieval_metrics(p_img@p_txt[k*n_caps:(k+1)*n_caps].T,validation_owners)
        perturbation_rows.append({
            "split":"validation","model":"siglip2","variant":variant,
            "changed_captions":sum(a!=b for a,b in zip(texts,validation_texts,strict=True)),
            "t2i_recall_at_1":m["t2i_recall_at_1"],"t2i_recall_at_5":m["t2i_recall_at_5"],
            "i2t_recall_at_1":m["i2t_recall_at_1"],"rsum":m["rsum"],
        })
    perturbation_table=pd.DataFrame(perturbation_rows)
    perturbation_table.to_csv(OUTPUT_ROOT/"validation"/"caption_perturbation.csv",index=False)

    # No-match queries against the SigLIP 2 test index.
    qfeat=p_txt[len(all_texts)-len(irrelevant_queries):]
    del p_img,p_txt

    for query,feat in zip(irrelevant_queries,qfeat,strict=True):
        scores=siglip2_img@feat
        top=np.argsort(-scores)[:3]
        print("\nQuery:",query)
        print("Evaluation status: no ground-truth match — nearest-neighbor behavior only")
        for i in top:
            print(float(scores[i]),test_records[int(i)]["captions"][0])
    return perturbation_table


@stage("byod")
def stage_byod(args):
    if not BYOD_ZIP_PATH or not Path(BYOD_ZIP_PATH).is_file():
        raise ValueError(f"BYOD_ZIP_PATH {BYOD_ZIP_PATH!r} is not a file in this runtime; check the path with the Files pane")
    load_transformers()
    byod_roles=load_byod(BYOD_ZIP_PATH)
    print("BYOD roles:",{role:len(rows) for role,rows in byod_roles.items()},"| tier:",WORKSHOP_TIER)
    byod_run_dir=run_byod_retrieval(byod_roles)
    write_state("byod_run.json",{"run_dir":str(byod_run_dir),
                                 "roles":{role:len(rows) for role,rows in byod_roles.items()}})
    print("BYOD files (CSV/JSON, index, adapter):",byod_run_dir)


def byod_run_dir_of_this_run():
    path=WORK_ROOT/"state"/"byod_run.json"
    if not path.is_file() or path.stat().st_mtime<RUN_STARTED-1.0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))["run_dir"]


def completion_summary(tier,test_records,test_texts,retrieval_rows,rerank_rows,blip_adapter_report,
                       blip_reload_parity,byod_run_dir):
    return {
        "notebook_spec":"2.1",
        "profile":"MULTI-CAPABILITY",
        "mode":"WORKSHOP",
        "tier":tier,
        "models":["SigLIP v1","SigLIP 2","BLIP ITC/ITM"],
        "test_gallery_images":len(test_records),
        "test_gallery_captions":len(test_texts),
        "rerank_top_k":RERANK_TOP_K,
        "test_r1_i2t / t2i (all captions)":{
            r["model"]:f'{r["i2t_recall_at_1"]:.3f} / {r["t2i_recall_at_1"]:.3f}' for r in retrieval_rows
        },
        "reranked R@1 (same queries: coarse -> ITM)":{
            r["candidate_source"]:(
                f'I2T {r["coarse_i2t_r1"]:.3f} -> {r["itm_i2t_recall_at_1"]:.3f}; '
                f'T2I first caption {r["coarse_t2i_r1_canonical"]:.3f} -> {r["itm_t2i_recall_at_1"]:.3f}'
            ) for r in rerank_rows
        },
        "full_adaptation":(
            f'ran; best epoch {blip_adapter_report["best_epoch"]}; reload parity {blip_reload_parity}'
            if blip_adapter_report else "not run (STANDARD)"
        ),
        "byod":str(byod_run_dir) if byod_run_dir else "not run",
        "index_directory":str((OUTPUT_ROOT/"index").resolve()),
        "output_directory":str(OUTPUT_ROOT.resolve()),
        "release_status":"candidate",
    }


@stage("report")
def stage_report(args):
    import datetime
    frozen=require_frozen()
    data=records()
    test_records=data["test"]
    test_texts,_owners=gallery(test_records)
    dataset_manifest=json.loads((OUTPUT_ROOT/"data"/"dataset_manifest.json").read_text(encoding="utf-8"))
    index_manifest=json.loads((OUTPUT_ROOT/"index"/"index_manifest.json").read_text(encoding="utf-8"))
    retrieval_table=pd.DataFrame(read_state("retrieval_rows.json"))
    rerank_table=pd.DataFrame(read_state("rerank_rows.json"))
    gallery_table=pd.DataFrame(read_state("gallery_rows.json"))
    storage_table=pd.DataFrame(read_state("storage_rows.json"))
    adapted=read_state("adapted_test.json")
    blip_adapter_report=full_tier_file(OUTPUT_ROOT/"adaptation"/"blip"/"training_history.json")
    adapted_blip_metrics=adapted["adapted_blip_metrics"]
    adapted_rerank_rows=adapted["adapted_rerank_rows"]
    blip_reload_parity=adapted["blip_reload_parity"]
    byod_run_dir=byod_run_dir_of_this_run()

    experiment_manifest={
        "notebook_spec":"2.1",
        "notebook_profile":"MULTI-CAPABILITY",
        "notebook_mode":"WORKSHOP",
        "workshop_revision":"0.3.0-candidate",
        "timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tier":WORKSHOP_TIER,
        "dataset":dataset_manifest,
        "models":frozen["models"],
        "evaluation":frozen["evaluation"],
        "retrieval_test":retrieval_table.to_dict("records"),
        "reranking_test":rerank_table.to_dict("records"),
        "gallery_size":gallery_table.to_dict("records"),
        "runtime_storage":storage_table.to_dict("records"),
        "blip_adaptation":blip_adapter_report,
        "adapted_blip_test":adapted_blip_metrics,
        "adapted_reranking":adapted_rerank_rows,
        "blip_reload":blip_reload_parity,
        "index_manifest":index_manifest,
        "environment":read_state("environment.json"),
        "evidence_scope":"one seeded VizWiz-Captions sample; 391-image / 1,737-caption held-out gallery",
        "byod_run_directory":str(byod_run_dir) if byod_run_dir else None,
        "bundle_policy":"files under OUTPUT_DIR written since this Run all started (RUN_STARTED)",
        "release_gates":[
            "fresh T4 STANDARD Run all",
            "fresh T4 FULL Run all",
            "record peak VRAM and exact released-notebook wall times",
        ],
    }

    (OUTPUT_ROOT/"provenance"/"experiment_manifest.json").write_text(
        json.dumps(experiment_manifest,indent=2,default=str),encoding="utf-8"
    )
    (OUTPUT_ROOT/"workshop_summary.json").write_text(
        json.dumps({
            "tier":WORKSHOP_TIER,
            "dataset_role_digest":dataset_manifest["role_digest"],
            "retrieval":retrieval_table.to_dict("records"),
            "reranking":rerank_table.to_dict("records"),
            "adapted_blip":adapted_blip_metrics,
        },indent=2,default=str),encoding="utf-8"
    )

    bundle_root=Path(OUTPUT_DIR).resolve()
    bundle_files,stale_files=files_for_bundle(bundle_root,RUN_STARTED)
    bundle=str(bundle_root)+"_DIMER_Vision_Language_Retrieval_Report.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as archive:
        for p in bundle_files:
            archive.write(p,p.relative_to(bundle_root).as_posix())
    if stale_files:
        print(f"Left out {len(stale_files)} file(s) written before this Run all started, e.g.",
              [p.relative_to(bundle_root).as_posix() for p in stale_files[:5]])
    print({"bundle":bundle,"files":len(bundle_files),"sha256":sha256_file(bundle)})

    write_state("completion_summary.json",completion_summary(
        WORKSHOP_TIER,test_records,test_texts,read_state("retrieval_rows.json"),read_state("rerank_rows.json"),
        blip_adapter_report,blip_reload_parity,byod_run_dir))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config",required=True)
    parser.add_argument("--stage",required=True,choices=sorted(STAGES))
    parser.add_argument("--model",default=None)
    parser.add_argument("--k",default=5,type=int)
    args=parser.parse_args(argv)
    configure(json.loads(Path(args.config).read_text(encoding="utf-8")))
    for p in [
        OUTPUT_ROOT / "data",
        OUTPUT_ROOT / "index",
        OUTPUT_ROOT / "validation",
        OUTPUT_ROOT / "frozen",
        OUTPUT_ROOT / "test",
        OUTPUT_ROOT / "adaptation" / "blip",
        OUTPUT_ROOT / "artifacts" / "blip",
        OUTPUT_ROOT / "figures",
        OUTPUT_ROOT / "provenance",
        WORK_ROOT / "state",
    ]:
        p.mkdir(parents=True, exist_ok=True)
    STAGES[args.stage](args)
    return 0


if __name__=="__main__":
    sys.exit(main())
