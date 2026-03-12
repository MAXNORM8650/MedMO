# medmo_loader.py
import os, io
import inspect
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, interleave_datasets
from glob import glob
from PIL import Image
import json
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional
from typing import Optional, List, Dict, Any
import os, hashlib
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image, ImageOps
from typing import Sequence
import os, json, hashlib
from pathlib import Path
from typing import Any, List, Tuple
import pandas as pd
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage, IterableDataset, Sequence as HFSequence
from io import BytesIO
import gc
import ast
from functools import lru_cache
import time


import os, torch.distributed as dist
# from trl.rewards.all_rewards import get_reward_funcs

def barrier_if_needed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

rank = int(os.environ.get("RANK", 0))


DATA_ROOT = "./Document/MedEvalKit/datas"
DATA_ROOT_MEDMO = "./Medmo_Dataset_1/Medmo_Dataset"
IMG_CACHE_ROOT = "./Document/medmo/trl/.cache/medevalkit_images"


# --------------------------
# Prompting (kept from you)
# --------------------------
REPORT_PROMPTS = [
    "Give a detailed medical description of the image. Identify the imaging modality and describe the anatomical region visible.",
    "Based on the image, identify the modality and provide a comprehensive analysis of what is visible.",
    "Generate a detailed radiology report based on the given medical image.",
    "Analyze the medical image and provide a structured report describing any findings, abnormalities, and impressions.",
    "Write a comprehensive medical report, including anatomical observations and pathological findings visible in the image.",
    "Please describe everything clinically relevant visible in this scan, including abnormal signals or lesions.",
    "Write a detailed radiology report based on this scan.",
]

def _det_prompt(uid_val, idx):
    key = str(uid_val) if uid_val is not None else str(idx)
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return REPORT_PROMPTS[h % len(REPORT_PROMPTS)]

def _format_batch_indexed_no_io(samples):
    """
    Build chat messages with an image placeholder in the first user turn.
    DO NOT load images — 'images' is a list[str] of paths.

    Expected input columns (after standardization):
      - full_path (str): path to image
      - caption   (str): ground-truth report
      - uid       (optional): for deterministic prompt selection

    Output columns:
      - messages: list of role/content dicts; image placeholder references index 0
      - images:   list[str] paths (index-aligned with any 'image' content entries)
    """
    out_messages, out_images = [], []
    n = len(samples["full_path"])
    captions = samples.get("caption", [""] * n)
    uids     = samples.get("uid", [None] * n)

    for i in range(n):
        img_path = samples["full_path"][i]
        caption  = captions[i] if i < len(captions) else ""
        uid_val  = uids[i] if i < len(uids) else None

        imgs = [img_path] if isinstance(img_path, str) and img_path else []

        question = _det_prompt(uid_val, i)
        user_content = (
            [{"type": "text", "text": question, "index": None},
             {"type": "image", "text": None, "index": 0}]
            if imgs else
            [{"type": "text", "text": question, "index": None}]
        )

        msg = [
            {"role": "user",     "content": user_content},
            {"role": "assistant","content": [{"type": "text", "text": caption or "", "index": None}]},
        ]

        out_messages.append(msg)
        out_images.append(imgs)

    return {"messages": out_messages, "images": out_images}


# --------------------------
# Helpers
# --------------------------
def _alias_and_standardize(
    df: pd.DataFrame,
    column_map: Dict[str, str],
    image_root: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ensure the dataframe has columns: full_path, caption, (optional) uid.
    `column_map` maps your CSV columns -> ['full_path','caption','uid'].
    """
    colmap = {"filename": "filename", "full_path": "full_path", "caption": "caption", "uid": "uid"}
    # First apply user-provided aliases (e.g., {"image_path": "full_path", "report": "caption"})
    for src, dst in (column_map or {}).items():
        if src in df.columns:
            df = df.rename(columns={src: dst})

    # Minimal required columns
    # if "full_path" not in df.columns:
    #     raise ValueError("CSV must contain (or map to) a 'full_path' column.")
    if "caption" not in df.columns:
        # If not present, create empty caption column
        df["caption"] = ""

    # Make image paths absolute when an image_root is provided and paths are relative
    if image_root:
        df["full_path"] = [
            p if os.path.isabs(str(p)) else os.path.join(image_root, str(p))
            for p in df["filename"].tolist()
        ]

    return df[["full_path", "caption"] + (["uid"] if "uid" in df.columns else [])]

def _to_hfds(
    df: pd.DataFrame,
    num_proc: int = 16,
    batch_size: int = 128,
    desc_prefix: str = "",
) -> Dataset:
    hf_ds = Dataset.from_pandas(df, preserve_index=False)
    hf_ds = hf_ds.map(
        _format_batch_indexed_no_io,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc=f"{desc_prefix} Formatting (indexed, no I/O)",
    )
    keep = {"messages", "images"}
    drop_cols = [c for c in hf_ds.column_names if c not in keep]
    if drop_cols:
        hf_ds = hf_ds.remove_columns(drop_cols)
    return hf_ds


def _split_frame_tail_disjoint(df_all: pd.DataFrame, eval_count: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into disjoint train/validation/test.

    - `eval_count` rows from the tail are reserved for evaluation.
    - evaluation rows are split into validation/test halves (deterministic order).
    - train excludes evaluation rows.
    """
    n = len(df_all)
    eval_n = max(0, min(int(eval_count), n))

    if eval_n == 0:
        empty = df_all.iloc[0:0].reset_index(drop=True)
        return df_all.reset_index(drop=True), empty.copy(), empty.copy()

    train_end = n - eval_n
    df_train = df_all.iloc[:train_end].reset_index(drop=True)
    df_eval = df_all.iloc[train_end:].reset_index(drop=True)

    val_n = (len(df_eval) + 1) // 2
    df_val = df_eval.iloc[:val_n].reset_index(drop=True)
    df_test = df_eval.iloc[val_n:].reset_index(drop=True)
    return df_train, df_val, df_test


def _split_frame_ratio_disjoint(df_all: pd.DataFrame, eval_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ratio wrapper over `_split_frame_tail_disjoint` for disjoint train/val/test."""
    n = len(df_all)
    if n == 0:
        empty = df_all.iloc[0:0].reset_index(drop=True)
        return empty.copy(), empty.copy(), empty.copy()
    eval_n = max(1, int(round(float(eval_ratio) * n)))
    return _split_frame_tail_disjoint(df_all, eval_n)


def _split_dataset_tail_disjoint(ds_all: Dataset, eval_count: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a HuggingFace Dataset into disjoint train/validation/test.

    - `eval_count` rows from the tail are reserved for eval.
    - eval rows are split into validation/test halves.
    - train excludes eval rows.
    """
    n = len(ds_all)
    eval_n = max(0, min(int(eval_count), n))

    if eval_n == 0:
        empty = ds_all.select([])
        return ds_all, empty, empty

    train_end = n - eval_n
    ds_train = ds_all.select(range(0, train_end))
    ds_eval = ds_all.select(range(train_end, n))

    val_n = (len(ds_eval) + 1) // 2
    ds_val = ds_eval.select(range(0, val_n))
    ds_test = ds_eval.select(range(val_n, len(ds_eval)))
    return ds_train, ds_val, ds_test


# --------------------------
# Merge/export helpers (non-invasive)
# --------------------------
def load_and_merge_datasets(
    dataset_names: Sequence[str],
    split: str = "train",
    strategy: str = "interleave",  # or "concatenate"
    seed: int = 42,
    probabilities: Optional[Sequence[float]] = None,
    per_dataset_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dataset:
    """
    Load several registered datasets and merge them into a single Dataset.

    - Adds 'source_dataset' column for debugging/weighting.
    - Uses existing loader functions; does not touch image bytes.
    """
    per_dataset_kwargs = per_dataset_kwargs or {}
    parts: List[Dataset] = []

    for name in dataset_names:
        if name not in _LOADER_REGISTRY:
            raise ValueError(f"Dataset '{name}' is not registered. Available: {available_datasets()}")

        loader_fn = _LOADER_REGISTRY[name]
        kwargs = per_dataset_kwargs.get(name, {})
        ds_dict = loader_fn(**kwargs)

        if split not in ds_dict:
            continue

        ds = ds_dict[split]
        is_stream = getattr(ds, "is_streaming", False) or isinstance(ds, IterableDataset)
        if is_stream:
            ds = ds.map(lambda _: {"source_dataset": name})
        else:
            try:
                ds = ds.add_column("source_dataset", [name] * len(ds))
            except TypeError:
                # Fallback if length is undefined but is_streaming flag is missing
                ds = ds.map(lambda _: {"source_dataset": name})
                is_stream = True
        parts.append(ds)

    if not parts:
        raise ValueError(f"No datasets found for split '{split}' in {dataset_names}")

    any_stream = any(getattr(p, "is_streaming", False) or isinstance(p, IterableDataset) for p in parts)
    if strategy == "concatenate" and any_stream:
        # concatenate_datasets requires known lengths; fall back to interleave for streaming sources
        strategy = "interleave"

    if strategy == "concatenate":
        merged = concatenate_datasets(parts)
    else:
        merged = interleave_datasets(parts, probabilities=probabilities, seed=seed)
    return merged


def export_manifest_jsonl(
    dataset: Dataset,
    jsonl_path: str,
    split: str,
) -> str:
    """
    Emit a JSONL manifest with image paths and text only (no image I/O).
    Fields: image_path, text, source_dataset, split
    """
    def _simplify(example: Dict[str, Any]) -> Dict[str, Any]:
        text = ""
        msgs = example.get("messages") or []
        if msgs and isinstance(msgs, list):
            assistant = msgs[-1] if len(msgs) > 1 else msgs[0]
            for piece in assistant.get("content", []):
                if piece.get("type") == "text":
                    text = piece.get("text", "")
                    break

        imgs = example.get("images")
        if isinstance(imgs, list) and imgs:
            first = imgs[0]
            if isinstance(first, list) and first:
                image_path = first[0]
            else:
                image_path = first
        else:
            image_path = None

        return {
            "image_path": image_path,
            "text": text,
            "source_dataset": example.get("source_dataset"),
            "split": split,
        }

    if getattr(dataset, "is_streaming", False):
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for ex in dataset:
                f.write(json.dumps(_simplify(ex)) + "\n")
    else:
        simplified = dataset.map(
            _simplify,
            remove_columns=dataset.column_names,
            desc="Flattening to manifest",
        )
        simplified.to_json(jsonl_path, orient="records", lines=True)

    return jsonl_path

# --------------------------
# Registry
# --------------------------
_LOADER_REGISTRY: Dict[str, Callable[..., DatasetDict]] = {}

def register_dataset(name: str):
    def _wrap(fn: Callable[..., DatasetDict]):
        _LOADER_REGISTRY[name] = fn
        return fn
    return _wrap

def available_datasets() -> List[str]:
    return sorted(_LOADER_REGISTRY.keys())

def _call_loader(name: str, num_proc: int, batch_size: int) -> DatasetDict:
    fn = _LOADER_REGISTRY[name]
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(num_proc=num_proc, batch_size=batch_size)

    kwargs = {}
    if "num_proc" in sig.parameters:
        kwargs["num_proc"] = num_proc
    if "batch_size" in sig.parameters:
        kwargs["batch_size"] = batch_size
    return fn(**kwargs)

# --------------------------
# Generic CSV-based loader
# --------------------------
@dataclass
class CsvDatasetSpec:
    name: str
    splits: Dict[str, str]                      # {"train": "/path/train.csv", "validation": "...", "test": "..."}
    column_map: Dict[str, str] = field(default_factory=dict)  # e.g. {"image_path": "full_path", "report": "caption", "study_id":"uid"}
    image_root: Optional[str] = None

def load_from_csv_spec(
    spec: CsvDatasetSpec,
    num_proc: int = 8,
    batch_size: int = 1,
) -> DatasetDict:
    out = {}
    for split, csv_path in spec.splits.items():
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[{spec.name}] Missing CSV for split '{split}': {csv_path}")
        df = pd.read_csv(csv_path)
        df = _alias_and_standardize(df, spec.column_map, spec.image_root)
        out[split] = _to_hfds(df, num_proc=num_proc, batch_size=batch_size, desc_prefix=f"[{spec.name}:{split}]")
    return DatasetDict(out)



@register_dataset("medbullets_op5")
def load_medbullets_op5(
    num_proc: int = 1,
    batch_size: int = 2048,
    parquet_paths: dict | None = None,
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Text-only MedBullets op5 loader:
      - reads parquet shards per split
      - builds `messages` (user -> assistant)
      - keeps `images` as [None] per sample for schema consistency
      - merges answer_idx with answer (e.g., "C: <answer>")
    """
    import pandas as pd
    from datasets import Dataset, DatasetDict

    if parquet_paths is None:
        parquet_paths = {
            "train": "./Document/MedEvalKit/datas/medbullets_op5/data/train-00000-of-00001.parquet",
            "validation": "./Document/MedEvalKit/datas/medbullets_op5/data/validation-00000-of-00001.parquet",
            "test": "./Document/MedEvalKit/datas/medbullets_op5/data/test-00000-of-00001.parquet",
        }

    def _norm_idx(idx):
        if idx is None:
            return ""
        if isinstance(idx, bool):
            return ""
        if isinstance(idx, (int, float)):
            i = int(idx)
            if 1 <= i <= 26:
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i - 1]
            if 0 <= i < 26:
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
            return str(i)
        s = str(idx).strip()
        if not s:
            return ""
        if s.isdigit():
            i = int(s)
            if 1 <= i <= 26:
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i - 1]
        return s.upper()

    def _read_split(split: str) -> Dataset:
        if split not in parquet_paths:
            return Dataset.from_dict({"messages": [], "images": []})
        df = pd.read_parquet(parquet_paths[split], engine="pyarrow")
        if debug_limit:
            df = df.head(debug_limit)

        # required columns
        required = ("question", "opa", "opb", "opc", "opd", "ope", "answer_idx", "answer")
        for col in required:
            if col not in df.columns:
                raise ValueError(f"[medbullets_op5:{split}] missing required column '{col}'")

        ds = Dataset.from_pandas(df, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            qs = batch["question"]
            opa = batch["opa"]
            opb = batch["opb"]
            opc = batch["opc"]
            opd = batch["opd"]
            ope = batch["ope"]
            idxs = batch["answer_idx"]
            ans = batch["answer"]
            n = len(qs)

            for i in range(n):
                question = (qs[i] or "").strip()
                options = {
                    "A": (opa[i] or "").strip(),
                    "B": (opb[i] or "").strip(),
                    "C": (opc[i] or "").strip(),
                    "D": (opd[i] or "").strip(),
                    "E": (ope[i] or "").strip(),
                }
                options = {k: v for k, v in options.items() if v}
                options_text = "\n".join(f"{k}. {v}" for k, v in options.items())

                user_q = (
                    "## Instruction: Choose the correct option based on the question below.\n\n"
                    f"{question}\n\n## Options:\n{options_text}"
                )

                idx = _norm_idx(idxs[i])
                answer_text = (ans[i] or "").strip()
                if not answer_text and idx:
                    answer_text = (options.get(idx, "") or "").strip()

                if idx:
                    final_answer = f"{idx}: {answer_text}".strip()
                else:
                    key = next((k for k, v in options.items() if (v or "").strip() == answer_text), "")
                    final_answer = f"{key}: {answer_text}".strip() if key else answer_text

                user = {"role": "user", "content": [{"type": "text", "text": user_q, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": final_answer, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[medbullets_op5:{split}] format")

        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    out = {sp: _read_split(sp) for sp in ("train", "validation")}
    return DatasetDict(out)

# --------------------------
# MedXpertQA loaders
# --------------------------
@register_dataset("medxpertqa_text")
def load_medxpertqa_text(
    num_proc: int = 1,
    batch_size: int = 2048,
    jsonl_paths: dict | None = None,
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Text-only MedXpertQA loader:
      - reads jsonl per split (Text)
      - builds `messages` (user -> assistant)
      - keeps `images` as [None] per sample for schema consistency
      - uses `label` as the answer
    """
    import json
    import os
    from datasets import Dataset, DatasetDict

    if jsonl_paths is None:
        base = "./Document/MedEvalKit/datas/MedXpertQA/Text"
        jsonl_paths = {
            "train": os.path.join(base, "dev.jsonl"),
            "validation": os.path.join(base, "validation.jsonl"),
            "test": os.path.join(base, "test.jsonl"),
        }

    def _read_split(split: str) -> Dataset:
        path = jsonl_paths.get(split)
        if not path or not os.path.exists(path):
            return Dataset.from_dict({"messages": [], "images": []})

        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if debug_limit and len(rows) >= debug_limit:
                    break

        if not rows:
            return Dataset.from_dict({"messages": [], "images": []})

        ds = Dataset.from_list(rows)

        def _format(batch):
            msgs, imgs = [], []
            qs = batch.get("question", [])
            opts = batch.get("options", [])
            labels = batch.get("label", [])
            n = len(qs)

            for i in range(n):
                question = (qs[i] or "").strip()
                options = opts[i] or {}
                options_text = "\n".join(f"{k}. {v}" for k, v in options.items())

                user_q = (
                    "## Instruction: Choose the correct option based on the question below.\n\n"
                    f"{question}\n\n## Options:\n{options_text}"
                )

                answer = (labels[i] or "").strip()

                user = {"role": "user", "content": [{"type": "text", "text": user_q, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": answer, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[medxpertqa_text:{split}] format")

        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    out = {sp: _read_split(sp) for sp in ("train", "validation", "test")}
    return DatasetDict(out)

@register_dataset("medxpertqa_mm")
def load_medxpertqa_mm(
    num_proc: int = 1,
    batch_size: int = 1024,
    jsonl_paths: Optional[dict] = None,
    image_root: str = "./Document/MedEvalKit/datas/MedXpertQA/images",
    drop_missing_images: bool = True,
    debug_limit: Optional[int] = None,
) -> DatasetDict:
    """
    Multimodal MedXpertQA loader:
      - reads jsonl per split (MM)
      - builds `messages` (user -> assistant)
      - keeps `images` as list of image paths
      - uses `label` as the answer
    """
    import json
    import os
    from datasets import Dataset, DatasetDict
    
    if jsonl_paths is None:
        base = "./Document/MedEvalKit/datas/MedXpertQA/MM"
        jsonl_paths = {
            "train": os.path.join(base, "dev.jsonl"),
            "validation": os.path.join(base, "validation.jsonl"),
            "test": os.path.join(base, "test.jsonl"),
        }
    
    # Verify image root exists
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"[medxpertqa_mm] Image directory not found: {image_root}")
    
    def _resolve_image_path(img_name: str) -> str:
        """Resolve image path relative to image_root."""
        if not img_name:
            return ""
        img_name = img_name.strip()
        if os.path.isabs(img_name):
            return os.path.normpath(img_name)
        return os.path.normpath(os.path.join(image_root, img_name))
    
    def _read_split(split: str) -> Dataset:
        path = jsonl_paths.get(split)
        
        # Check if path exists
        if not path or not os.path.exists(path):
            if rank == 0:
                print(f"[medxpertqa_mm] Warning: {split} split not found at {path}, returning empty dataset")
            return Dataset.from_dict({"uid": [], "messages": [], "images": []})
        
        records: List[Dict[str, Any]] = []
        missing_images = 0
        limit = int(debug_limit) if (debug_limit is not None and debug_limit >= 0) else None
        
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    if rank == 0:
                        print(f"[medxpertqa_mm] Warning: Failed to parse line {idx} in {split}: {e}")
                    continue
                
                # Extract fields
                question = (row.get("question") or "").strip()
                options = row.get("options") or {}
                label = (row.get("label") or "").strip()
                img_names = row.get("images") or []
                
                # Handle single image string
                if isinstance(img_names, str):
                    img_names = [img_names]
                
                # Resolve image paths
                img_paths = [_resolve_image_path(img) for img in img_names]
                
                # Check if images exist
                valid_images = [p for p in img_paths if p and os.path.isfile(p)]
                if len(valid_images) != len(img_paths):
                    missing_images += 1
                    if drop_missing_images:
                        continue
                    # Use valid images only
                    img_paths = valid_images
                
                # Skip if no valid data
                if not question:
                    question = "Choose the correct option based on the image."
                if not label:
                    label = "No answer provided."
                
                # Format options text
                options_text = "\n".join(f"{k}. {v}" for k, v in options.items()) if options else ""
                
                # Build user question
                if options_text:
                    user_text = (
                        "## Instruction: Choose the correct option based on the question below.\n\n"
                        f"{question}\n\n## Options:\n{options_text}"
                    )
                else:
                    user_text = question
                
                # Build messages with image content
                user_content = []
                # Add images first
                for img_idx in range(len(img_paths)):
                    user_content.append({"type": "image", "index": img_idx, "text": None})
                # Add text
                user_content.append({"type": "text", "text": user_text, "index": None})
                
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": label, "index": None}]},
                ]
                
                # Create unique ID
                uid = row.get("id") or f"medxpertqa_mm_{split}_{idx}"
                
                records.append({
                    "uid": str(uid),
                    "messages": messages,
                    "images": img_paths
                })
                
                if limit is not None and len(records) >= limit:
                    break
        
        if not records:
            if rank == 0:
                print(f"[medxpertqa_mm] Warning: No usable samples in {split} split")
            return Dataset.from_dict({"uid": [], "messages": [], "images": []})
        
        if rank == 0 and missing_images > 0:
            if drop_missing_images:
                print(f"[medxpertqa_mm:{split}] Dropped {missing_images} samples with missing images.")
            else:
                print(f"[medxpertqa_mm:{split}] Warning: {missing_images} samples had missing images (kept anyway).")
        
        if rank == 0:
            print(f"[medxpertqa_mm:{split}] Loaded {len(records)} samples")
        
        return Dataset.from_list(records)
    
    # Load all splits
    splits_dict = {}
    for split_name in ("train", "validation", "test"):
        splits_dict[split_name] = _read_split(split_name)
    
    # Ensure at least train split has data
    if len(splits_dict["train"]) == 0:
        raise ValueError("[medxpertqa_mm] No usable samples in train split. Check JSONL files and image paths.")
    
    return DatasetDict(splits_dict)

# --------------------------
# SuperGPQA loader (arrow)
# --------------------------
@register_dataset("super_gpqa")
def load_super_gpqa(
    num_proc: int = 1,
    batch_size: int = 1024,
    file_paths: Optional[dict] = None,
    debug_limit: Optional[int] = None,
) -> DatasetDict:
    """
    Text-only SuperGPQA loader:
      - reads Arrow files
      - builds `messages` (user -> assistant)
      - keeps `images` as empty list per sample for schema consistency
      - uses `answer_letter` (if present) to format answer
    """
    import os
    import pyarrow as pa
    from datasets import Dataset, DatasetDict

    if file_paths is None:
        base = "./Document/MedEvalKit/datas/m-a-p___super_gpqa/default/0.0.0/4430d4458112c7d4497fdcf94d7cc223313d6acf"
        file_paths = {
            "train": os.path.join(base, "super_gpqa-train.arrow"),
        }

    def _read_split(split: str) -> Dataset:
        path = file_paths.get(split)
        if not path or not os.path.exists(path):
            return Dataset.from_dict({"uid": [], "messages": [], "images": []})

        try:
            # Try different methods to read the arrow file
            table = None
            
            # Method 1: Try as streaming format
            try:
                with pa.memory_map(path, 'r') as source:
                    reader = pa.ipc.open_stream(source)
                    table = reader.read_all()
            except:
                pass
            
            # Method 2: Try as file format (if method 1 failed)
            if table is None:
                try:
                    with pa.memory_map(path, 'r') as source:
                        reader = pa.ipc.open_file(source)
                        table = reader.read_all()
                except:
                    pass
            
            # Method 3: Use datasets library directly (fallback)
            if table is None:
                from datasets import load_from_disk
                try:
                    # Try loading as a datasets arrow file
                    import pyarrow.feather as feather
                    table = feather.read_table(path)
                except:
                    pass
            
            if table is None:
                raise ValueError(f"Could not read arrow file with any method")
            
            # Convert to list of dicts
            data_list = table.to_pylist()
            
            # Apply debug limit if specified
            if debug_limit and debug_limit > 0:
                data_list = data_list[:debug_limit]
            
        except Exception as e:
            print(f"[super_gpqa:{split}] Error: {e}")
            return Dataset.from_dict({"uid": [], "messages": [], "images": []})

        if len(data_list) == 0:
            return Dataset.from_dict({"uid": [], "messages": [], "images": []})

        records: List[Dict[str, Any]] = []
        
        for idx, row in enumerate(data_list):
            question = str(row.get("question", "")).strip()
            options = row.get("options", [])
            answer = str(row.get("answer", "")).strip()
            answer_letter = str(row.get("answer_letter", "")).strip()
            uuid = str(row.get("uuid", f"super_gpqa_{idx}"))
            
            # Handle options formatting
            if isinstance(options, list) and len(options) > 0:
                options_text = "\n".join(f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options))
            else:
                options_text = ""
            
            # Build user question
            if options_text:
                user_text = (
                    "## Instruction: Choose the correct option based on the question below.\n\n"
                    f"{question}\n\n## Options:\n{options_text}"
                )
            else:
                user_text = question if question else "Answer the question."
            
            # Format answer
            if answer_letter:
                final_answer = f"{answer_letter}: {answer}" if answer else answer_letter
            else:
                final_answer = answer if answer else "No answer provided."
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text, "index": None}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": final_answer, "index": None}]
                }
            ]
            
            records.append({
                "uid": uuid,
                "messages": messages,
                "images": [None]
            })
        
        if not records:
            return Dataset.from_dict({"uid": [], "messages": [], "images": []})
        
        return Dataset.from_list(records)

    splits_dict = {"train": _read_split("train")}
    
    if len(splits_dict["train"]) == 0:
        raise ValueError("[super_gpqa] No usable samples in train split. Check Arrow file path.")
    
    return DatasetDict(splits_dict)



# --------------------------
# MedQA USMLE 4-options (phrases_no_exclude) loader
# --------------------------
@register_dataset("medqa_usmle")
def load_medqa_usmle_4opt_phrases(
    num_proc: int = 1,
    batch_size: int = 2048,
    jsonl_paths: dict | None = None,
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Text-only MedQA-USMLE-4-options loader:
      - reads jsonl for train/test and merges BOTH into train
      - builds `messages` (user -> assistant)
      - keeps `images` as [None] per sample for schema consistency
      - uses answer_idx to select option text when available, else falls back to answer
    """
    import json
    import os
    from datasets import Dataset, DatasetDict, concatenate_datasets

    if jsonl_paths is None:
        base = "./Document/MedEvalKit/datas/MedQA-USMLE-4-options"
        jsonl_paths = {
            "train": os.path.join(base, "phrases_no_exclude_train.jsonl"),
            "test": os.path.join(base, "phrases_no_exclude_test.jsonl"),
        }

    def _load_jsonl(path: str) -> Dataset:
        if not path or not os.path.exists(path):
            return Dataset.from_dict({"messages": [], "images": []})
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if debug_limit and len(rows) >= debug_limit:
                    break
        if not rows:
            return Dataset.from_dict({"messages": [], "images": []})
        ds = Dataset.from_list(rows)

        def _format(batch):
            msgs, imgs = [], []
            qs = batch.get("question", [])
            opts = batch.get("options", [])
            ans = batch.get("answer", [])
            idxs = batch.get("answer_idx", [])
            n = len(qs)

            for i in range(n):
                question = (qs[i] or "").strip()
                options = opts[i] or {}
                options_text = "\n".join(f"{k}. {v}" for k, v in options.items())
                user_q = (
                    "## Instruction: Choose the correct option based on the question below.\n\n"
                    f"{question}\n\n## Options:\n{options_text}"
                )

                idx = (idxs[i] or "").strip() if i < len(idxs) else ""
                answer_text = (ans[i] or "").strip() if i < len(ans) else ""
                if idx:
                    final_answer = f"{idx}. {options.get(idx, answer_text)}".strip()
                else:
                    final_answer = answer_text

                user = {"role": "user", "content": [{"type": "text", "text": user_q, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": final_answer, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc="[medqa_usmle_4opt_phrases] format")
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    ds_train = _load_jsonl(jsonl_paths.get("train"))
    ds_test = _load_jsonl(jsonl_paths.get("test"))

   

    return DatasetDict({"train": ds_train})


# --------------------------
# MedMCQA loader
# --------------------------
@register_dataset("medmcqa")
def load_medmcqa(
    num_proc: int = 1,
    batch_size: int = 2048,
    parquet_paths: dict | None = None,
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Text-only MedMCQA loader:
      - reads parquet per split
      - builds `messages` (user -> assistant)
      - keeps `images` as [None] per sample for schema consistency
      - uses `cop` (0-based index) to select correct option
    """
    import os
    import pandas as pd
    from datasets import Dataset, DatasetDict

    if parquet_paths is None:
        base = "./Document/MedEvalKit/datas/medmcqa/data"
        parquet_paths = {
            "train": f"{base}/train-00000-of-00001.parquet",
            "validation": f"{base}/validation-00000-of-00001.parquet",
            "test": f"{base}/test-00000-of-00001.parquet",
        }


    def _read_split(split: str) -> Dataset:
        path = parquet_paths.get(split)
        if not path or not os.path.exists(path):
            return Dataset.from_dict({"messages": [], "images": []})
        df = pd.read_parquet(path, engine="pyarrow")
        if debug_limit:
            df = df.head(debug_limit)

        required = ("question", "opa", "opb", "opc", "opd", "cop")
        for col in required:
            if col not in df.columns:
                raise ValueError(f"[medmcqa:{split}] missing required column '{col}'")

        ds = Dataset.from_pandas(df, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            qs = batch["question"]
            opa = batch["opa"]
            opb = batch["opb"]
            opc = batch["opc"]
            opd = batch["opd"]
            cop = batch["cop"]
            n = len(qs)

            for i in range(n):
                question = (qs[i] or "").strip()
                options = {
                    "A": (opa[i] or "").strip(),
                    "B": (opb[i] or "").strip(),
                    "C": (opc[i] or "").strip(),
                    "D": (opd[i] or "").strip(),
                }
                options = {k: v for k, v in options.items() if v}
                options_text = "\n".join(f"{k}. {v}" for k, v in options.items())

                user_q = (
                    "## Instruction: Choose the correct option based on the question below.\n\n"
                    f"{question}\n\n## Options:\n{options_text}"
                )

                letter = ""
                idx = cop[i]
                if idx is not None:
                    try:
                        idx_int = int(idx)
                        if 0 <= idx_int < 26:
                            letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[idx_int]
                    except Exception:
                        letter = ""
                answer_text = options.get(letter, "")
                final_answer = f"{letter}. {answer_text}".strip() if letter else answer_text

                user = {"role": "user", "content": [{"type": "text", "text": user_q, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": final_answer, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[medmcqa:{split}] format")
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _read_split("train"),
        "validation": _read_split("validation"),
        "test": _read_split("test"),
    })


# --------------------------
# PubMedQA loader
# --------------------------
@register_dataset("pubmedqa")
def load_pubmedqa_local(
    num_proc: int = 1,
    batch_size: int = 1024,
    parquet_paths: Optional[dict] = None,
    debug_limit: Optional[int] = None,
    use_splits: Optional[List[str]] = None,  # NEW: Control which splits to use for training
) -> DatasetDict:
    """
    Text-only PubMedQA loader:
      - reads parquet per split (id, data)
      - builds `messages` (user -> assistant)
      - keeps `images` as empty list per sample for schema consistency
      - uses `Correct Option` / `Correct Answer` for the answer
    
    Args:
        use_splits: List of splits to load (e.g., ["train", "validation", "test"]). 
                    If None, loads all available splits.
    """
    import os
    from datasets import Dataset, DatasetDict

    if parquet_paths is None:
        base = "./Document/MedEvalKit/datas/pubmedqa/data"
        parquet_paths = {
            "train": os.path.join(base, "train-00000-of-00001.parquet"),
            "validation": os.path.join(base, "validation-00000-of-00001.parquet"),
            "test": os.path.join(base, "test-00000-of-00001.parquet"),
        }
    
    # Determine which splits to load
    if use_splits is None:
        use_splits = ["train", "validation", "test"]
    
    def _read_split(split: str) -> Dataset:
        path = parquet_paths.get(split)
        if not path or not os.path.exists(path):
            print(f"[pubmedqa_local:{split}] Warning: file not found at {path}")
            return Dataset.from_dict({"uid": [], "messages": [], "images": []})
        
        print(f"[pubmedqa_local:{split}] Loading from {path}")
        
        # Load parquet using datasets library
        ds = Dataset.from_parquet(path)
        
        if debug_limit and debug_limit > 0:
            ds = ds.select(range(min(debug_limit, len(ds))))

        if "data" not in ds.column_names:
            raise ValueError(f"[pubmedqa_local:{split}] missing required column 'data'")

        # Convert to list for processing
        data_list = [ds[i] for i in range(len(ds))]
        records: List[Dict[str, Any]] = []

        for idx, item in enumerate(data_list):
            data = item.get("data") or {}
            item_id = item.get("id", f"pubmedqa_{split}_{idx}")
            
            question = (data.get("Question") or "").strip()
            context = data.get("Context")
            options = data.get("Options") or {}
            corr_opt = (data.get("Correct Option") or "").strip()
            corr_ans = (data.get("Correct Answer") or "").strip()

            # Context can be list/array; join into a single passage
            if context is None:
                context_text = ""
            elif isinstance(context, (list, tuple)):
                context_text = "\n".join([str(x) for x in context if str(x).strip()])
            else:
                try:
                    context_text = "\n".join([str(x) for x in list(context) if str(x).strip()])
                except Exception:
                    context_text = str(context)

            # Format options
            options_text = "\n".join(f"{k}. {v}" for k, v in options.items())

            # Build user question
            user_text = (
                "## Instruction: Answer the question using the context below.\n\n"
                f"## Context:\n{context_text}\n\n"
                f"## Question:\n{question}\n\n"
                f"## Options:\n{options_text}"
            )

            # Build answer
            if corr_opt:
                ans_text = options.get(corr_opt, corr_ans)
                final_answer = f"{corr_opt}. {ans_text}".strip()
            else:
                final_answer = corr_ans if corr_ans else "No answer provided."

            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text, "index": None}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": final_answer, "index": None}]
                }
            ]

            records.append({
                "uid": str(item_id),
                "messages": messages,
                "images": [None]
            })

        if not records:
            return Dataset.from_dict({"uid": [], "messages": [], "images": []})
        
        print(f"[pubmedqa_local:{split}] Loaded {len(records)} samples")
        return Dataset.from_list(records)

    # Load requested splits
    loaded_splits = {}
    for split in use_splits:
        loaded_splits[split] = _read_split(split)
    
    
    if "train" in loaded_splits and len(loaded_splits["train"]) == 0:
        raise ValueError("[pubmedqa_local] No usable samples in train split. Check parquet files.")
        
        return DatasetDict(loaded_splits)
    


# --------------------------
# IU-XRAY (local test.json) loader
# --------------------------
@register_dataset("iu_xray_test")
def load_iu_xray_test(
    num_proc: int = 1,
    batch_size: int = 1024,
    json_paths: dict | None = None,
    image_root: str = "./Document/MedEvalKit/datas/IU_XRAY/Images",
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    IU-XRAY loader for local JSON files:
      - reads json per split (expects fields: id, image[list], findings, impression)
      - builds `messages` (user -> assistant)
      - keeps `images` as list of image paths (multi-image)
    """
    import json
    import os
    from datasets import Dataset, DatasetDict

    if json_paths is None:
        base = "./Document/MedEvalKit/datas/IU_XRAY"
        json_paths = {
            "train": os.path.join(base, "train.json"),
            "validation": os.path.join(base, "validation.json"),
            "test": os.path.join(base, "test.json"),
        }

    def _read_split(split: str) -> Dataset:
        path = json_paths.get(split)
        if not path or not os.path.exists(path):
            return Dataset.from_dict({"messages": [], "images": []})

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if debug_limit:
            raw = raw[:debug_limit]

        def _format(batch):
            msgs, imgs = [], []
            findings_list = batch.get("findings", [])
            impression_list = batch.get("impression", [])
            images_list = batch.get("image", [])
            n = len(findings_list)

            for i in range(n):
                findings = (findings_list[i] or "").strip()
                impression = (impression_list[i] or "").strip()
                images = images_list[i] if i < len(images_list) else []
                if isinstance(images, str):
                    images = [images]

                img_paths = [os.path.join(image_root, p) if not os.path.isabs(p) else p for p in (images or [])]

                user_text = "Describe the findings and impression for this chest X-ray."
                user_content = [{"type": "image", "index": j, "text": None} for j in range(len(img_paths))]
                user_content.append({"type": "text", "text": user_text, "index": None})

                if findings and impression:
                    answer = f"{findings}\n\nImpression: {impression}"
                elif impression:
                    answer = impression
                else:
                    answer = findings

                user = {"role": "user", "content": user_content}
                asst = {"role": "assistant", "content": [{"type": "text", "text": answer, "index": None}]}
                msgs.append([user, asst])
                imgs.append(img_paths)

            return {"messages": msgs, "images": imgs}

        ds = Dataset.from_list(raw)
        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[iu_xray_test:{split}] format")
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _read_split("train"),
        "validation": _read_split("validation"),
        "test": _read_split("test"),
    })


# --------------------------
# IU-XRay loader (kept)
# --------------------------
@register_dataset("iuxray_report")
def load_iuxray_report(num_proc: int = 8, batch_size: int = 128) -> DatasetDict:
    spec = CsvDatasetSpec(
        name="iuxray",
        splits={
            "train":      "./Medmo_Dataset_1/Medmo_Dataset/IU-XRay/clean_report.csv",
            "validation": "./Medmo_Dataset_1/Medmo_Dataset/IU-XRay/clean_report_test.csv",
            "test":       "./Medmo_Dataset_1/Medmo_Dataset/IU-XRay/clean_report_test_10.csv",
        },
        # If your CSV already has columns 'full_path' and 'caption', you can leave column_map empty.
        column_map={"filename": "filename"},         # e.g., {"image_path": "full_path", "report": "caption", "uid":"uid"}
        image_root='./Medmo_Dataset_1/Medmo_Dataset/IU-XRay/images/images_normalized',       # set to the images root if 'full_path' is relative in CSV
    )
    return load_from_csv_spec(spec, num_proc=num_proc, batch_size=batch_size)



# Assumes you already have a working @register_dataset decorator in your project.
@register_dataset("mimiccxr_report")
def load_mimiccxr_report(
    num_proc: int = 1,
    batch_size: int = 128,
    image_root: str = "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.1.0/files",
    train_csv: str = "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-Report/Reports/mimic_reports_with_images_clean.csv",
    val_csv: str   = "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-Report/Reports/mimic_reports_with_images_clean_val.csv",
    debug_limit: Optional[int] = None,
    check_files: bool = False,
    report_prompts: Optional[List[str]] = None,  # optional deterministic prompt bank
) -> DatasetDict:
    """
    Build a DatasetDict with splits {train, validation}.

    Output schema per example:
      {
        "images":   [ "<ABSOLUTE/IMAGE/PATH>" ],
        "messages": [
          { "role":"user",
            "content":[
              {"type":"text",  "text": "<prompt>",              "index": None},
              {"type":"image", "text":  None,                   "index": 0}
            ]
          },
          { "role":"assistant",
            "content":[
              {"type":"text",  "text": "<findings+impression>", "index": None}
            ]
          }
        ]
      }
    """

    # Default prompt bank (deterministic selection via md5(uid or row index))
    if not report_prompts:
        report_prompts = [
            "Give a detailed medical description of the image. Identify the imaging modality and describe the anatomical region visible.",
            "Based on the image, identify the modality and provide a comprehensive analysis of what is visible.",
            "Generate a detailed radiology report based on the given medical image.",
            "Analyze the medical image and provide a structured report describing any findings, abnormalities, and impressions.",
            "Write a comprehensive medical report, including anatomical observations and pathological findings visible in the image.",
            "Please describe everything clinically relevant visible in this scan, including abnormal signals or lesions.",
            "Write a detailed radiology report based on this scan.",
        ]

    def _read_one_split(csv_path: str, split_name: str) -> Dataset:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[mimiccxr:{split_name}] Missing CSV: {csv_path}")

        # Load directly with HF Datasets (faster than pandas -> from_pandas)
        ds = load_dataset("csv", data_files=csv_path, split="train")

        # Optionally shrink for quick iterations
        if debug_limit and len(ds) > debug_limit:
            ds = ds.select(range(debug_limit))

        def _build_batch(batch: Dict[str, List[Any]], indices: List[int]) -> Dict[str, Any]:
            n = len(indices)
            colnames = set(batch.keys())

            # Findings + Impression -> caption (robust to missing columns)
            findings   = batch["findings"]   if "findings"   in colnames else [""] * n
            impression = batch["impression"] if "impression" in colnames else [""] * n
            captions = []
            for f, imp in zip(findings, impression):
                f = ("" if f is None else str(f)).strip()
                imp = ("" if imp is None else str(imp)).strip()
                captions.append((f + ("\n\n" if f and imp else "") + imp).strip())

            # Image path (CSV must provide 'image_path' relative to image_root)
            if "image_path" not in colnames:
                raise ValueError("[mimiccxr] CSV must contain 'image_path' column (relative paths to JPGs).")
            rels = [("" if p is None else str(p)) for p in batch["image_path"]]
            full_paths = [
                p if os.path.isabs(p) else os.path.normpath(os.path.join(image_root, p))
                for p in rels
            ]

            # Optional UID to stabilize prompt selection; fallback to row index
            uids = None
            if "file_path" in colnames:
                uids = [("" if u is None else str(u)) for u in batch["file_path"]]

            # Deterministic prompt per row (md5(uid or index) % len(report_prompts))
            prompts = []
            L = len(report_prompts)
            for i, idx in enumerate(indices):
                key = (uids[i] if uids is not None and len(uids) > i and uids[i] else str(idx))
                h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
                prompts.append(report_prompts[h % L])

            # Build messages & images exactly as requested
            messages_out, images_out = [], []
            for i in range(n):
                img_path = full_paths[i]
                # user: text prompt + image placeholder (index=0)
                user_msg = {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text": prompts[i],  "index": None},
                        {"type": "image", "text": None,        "index": 0},
                    ],
                }
                # assistant: ground-truth caption (findings+impression)
                asst_msg = {
                    "role": "assistant",
                    "content": [
                        {"type": "text",  "text": captions[i], "index": None},
                    ],
                }
                messages_out.append([user_msg, asst_msg])
                images_out.append([img_path])  # list with a single full path (string)

            # Optional: verify files exist (slower; network FS hits)
            if check_files:
                keep = []
                for p in images_out:
                    ok = bool(p and isinstance(p[0], str) and os.path.exists(p[0]))
                    keep.append(ok)
                messages_out = [m for m, k in zip(messages_out, keep) if k]
                images_out   = [p for p, k in zip(images_out, keep)   if k]

            return {"messages": messages_out, "images": images_out}

        ds = ds.map(
            _build_batch,
            with_indices=True,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=list(ds.column_names),  # keep only images/messages
            desc=f"[mimiccxr:{split_name}] build messages + paths",
        )
        return ds

    return DatasetDict({
        "train": _read_one_split(train_csv, "train"),
        "validation": _read_one_split(val_csv, "validation"),
    })




# --- MedTrinity-25M report loader (same schema as IU-XRay/MIMIC-CXR) ---
# ───────────────────────── constants ─────────────────────────
MEDTRINITY_IMAGE_ROOT = "./Medmo_Dataset_1/Medmo_Dataset/MedTrinity-25M/25M_clean/"

# ───────────────────────── helpers ─────────────────────────
def _prefix_root_if_relative(df: pd.DataFrame, root: str | None) -> pd.DataFrame:
    """Join root to any non-empty, non-absolute path in df['full_path']."""
    if not root or "full_path" not in df.columns:
        return df
    fp = df["full_path"].astype(str)
    # leave absolute paths or ones already starting with the root
    def _join(p: str) -> str:
        if not p or p.startswith(root) or os.path.isabs(p) or p.startswith("/"):
            return p
        return os.path.normpath(os.path.join(root, p.lstrip("/")))
    df["full_path"] = [ _join(p) for p in fp.tolist() ]
    return df







@register_dataset("medtrinity_report")
def load_medtrinity_report(
    num_proc: int = 1,
    batch_size: int = 32,
    jsonl_paths: dict | None = None,
    image_root: str | None = MEDTRINITY_IMAGE_ROOT,
    debug_sample: int | None = None,
    streaming: bool = False,
) -> DatasetDict:
    """Memory-optimized MedTrinity-25M JSONL loader with streaming support."""
    import os
    import hashlib
    import gc
    
    # Get distributed training info
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0
    except:
        world_size = 1
        rank = 0
    
    REPORT_PROMPTS = [
        "Provide a detailed clinical interpretation of the medical image.",
        "Describe all visible anatomical structures and any abnormalities present in the image.",
        "Summarize the relevant medical findings shown in this image.",
        "Generate a comprehensive report based on the content of the medical image.",
        "Identify the anatomical region and discuss any pathological signs present.",
        "Write a detailed medical description highlighting both normal and abnormal observations.",
        "What are the key diagnostic features visible in this image?",
        "Based on this image, what can you infer about the patient's condition?",
        "List and describe any clinical observations supported by the image.",
        "Explain what this image reveals about the underlying disease or treatment status.",
    ]
    
    SPLIT_PATHS = {
        "train": os.path.join(MEDTRINITY_IMAGE_ROOT, "split_98.jsonl"),
        "validation": os.path.join(MEDTRINITY_IMAGE_ROOT, "split_2.jsonl"),
    }
    if jsonl_paths is not None:
        SPLIT_PATHS = jsonl_paths
    
    out = {}
    
    for split, path in SPLIT_PATHS.items():
        if not os.path.isfile(path):
            if rank == 0:
                print(f"[medtrinity] Skipping missing split '{split}': {path}")
            continue
        
        if rank == 0:
            print(f"[medtrinity:{split}] Loading {path} (streaming={streaming}, world_size={world_size})...")
        
        if streaming:
            # ✅ Create a custom generator that reads JSONL directly
            def jsonl_generator_for_rank(filepath, current_rank, total_ranks, is_train):
                """Read JSONL file and yield only samples for this rank."""
                import json
                
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    for idx, line in enumerate(f):
                        # For training: partition data across ranks
                        if is_train and total_ranks > 1:
                            if idx % total_ranks != current_rank:
                                continue  # Skip samples not assigned to this rank
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            example = json.loads(line, strict=False)
                            yield example
                        except Exception as e:
                            # Skip malformed lines
                            continue
            
            # Create generator with closures capturing current values
            from datasets import IterableDataset
            
            is_train_split = (split == "train")
            
            hf_ds = IterableDataset.from_generator(
                jsonl_generator_for_rank,
                gen_kwargs={
                    'filepath': path,
                    'current_rank': rank,
                    'total_ranks': world_size,
                    'is_train': is_train_split
                }
            )
            
            if rank == 0:
                if is_train_split and world_size > 1:
                    print(f"[medtrinity:{split}] Each rank processes ~{18_525_997 // world_size:,} samples")
                elif not is_train_split:
                    print(f"[medtrinity:{split}] All ranks evaluate on full validation set")
            
            # Shuffle training data
            if is_train_split:
                hf_ds = hf_ds.shuffle(seed=42, buffer_size=10_000)
            
        else:
            # Non-streaming mode with inline processing to avoid map() multiprocessing issues
            import json
            from datasets import Dataset
            import hashlib
            
            if rank == 0:
                print(f"[medtrinity:{split}] Reading and processing JSONL file inline...")
            
            processed_examples = []
            skipped = 0
            
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        example = json.loads(line, strict=False)
                        
                        # Extract fields
                        img_path = (example.get("full_path") or example.get("rel_path") or 
                                   example.get("image_path") or example.get("path"))
                        caption = (example.get("caption") or example.get("report") or 
                                  example.get("text") or example.get("answer") or "")
                        uid_val = str(example.get("uid") or example.get("id") or 
                                     example.get("image_id") or idx)
                        
                        # Build full path
                        if img_path and image_root and not os.path.isabs(img_path):
                            img_path = os.path.join(image_root, img_path)
                        
                        # Select prompt based on hash
                        h = int(hashlib.md5(uid_val.encode("utf-8")).hexdigest(), 16)
                        question = REPORT_PROMPTS[h % len(REPORT_PROMPTS)]
                        
                        # Build message structure
                        imgs = [img_path] if img_path else []
                        
                        user_content = [
                            {"type": "text", "text": question, "index": None},
                            {"type": "image", "text": None, "index": 0}
                        ] if imgs else [{"type": "text", "text": question, "index": None}]
                        
                        messages = [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": [{"type": "text", "text": caption, "index": None}]}
                        ]
                        
                        # Create processed example
                        processed_examples.append({
                            "messages": messages,
                            "images": imgs
                        })
                        
                        # Debug sampling
                        if debug_sample and len(processed_examples) >= debug_sample:
                            break
                        
                        # Progress logging every 1M examples
                        if rank == 0 and (idx + 1) % 1_000_000 == 0:
                            print(f"[medtrinity:{split}] Processed {len(processed_examples):,} examples...")
                            
                    except Exception as e:
                        skipped += 1
                        if skipped < 10 and rank == 0:  # Log first few errors
                            print(f"[medtrinity:{split}] Skipped malformed line {idx}: {str(e)[:100]}")
                        continue
            
            if rank == 0:
                print(f"[medtrinity:{split}] Loaded and processed {len(processed_examples):,} examples (skipped {skipped} malformed lines)")
            
            # Create dataset from processed list - no map() needed!
            hf_ds = Dataset.from_list(processed_examples)
            
        # For streaming mode, we still need map() since data isn't preloaded
        if streaming:
            # Process with map - make function self-contained for multiprocessing
            def _process_streaming(batch, img_root, prompts):
                """Self-contained processing function for streaming."""
                import hashlib
                import os
                
                paths = batch.get("full_path") or batch.get("rel_path") or batch.get("image_path") or batch.get("path")
                captions = batch.get("caption") or batch.get("report") or batch.get("text") or batch.get("answer")
                uids = batch.get("uid") or batch.get("id") or batch.get("image_id")
                
                if paths is None:
                    raise ValueError(f"No image path column found in batch. Available columns: {list(batch.keys())}")
                
                n = len(paths)
                if captions is None:
                    captions = [""] * n
                if uids is None:
                    uids = [str(i) for i in range(n)]
                
                messages_out = []
                images_out = []
                
                for i in range(n):
                    img_path = paths[i]
                    caption = captions[i] if i < len(captions) else ""
                    uid_val = uids[i] if i < len(uids) else str(i)
                    
                    if img_path and img_root and not os.path.isabs(img_path):
                        img_path = os.path.join(img_root, img_path)
                    
                    h = int(hashlib.md5(str(uid_val).encode("utf-8")).hexdigest(), 16)
                    question = prompts[h % len(prompts)]
                    
                    imgs = [img_path] if img_path else []
                    
                    user_content = [
                        {"type": "text", "text": question, "index": None},
                        {"type": "image", "text": None, "index": 0}
                    ] if imgs else [{"type": "text", "text": question, "index": None}]
                    
                    messages = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": [{"type": "text", "text": caption or "", "index": None}]}
                    ]
                    
                    messages_out.append(messages)
                    images_out.append(imgs)
                
                return {"messages": messages_out, "images": images_out}
            
            hf_ds = hf_ds.map(
                _process_streaming,
                batched=True,
                batch_size=batch_size,
                remove_columns=None,
                fn_kwargs={"img_root": image_root, "prompts": REPORT_PROMPTS},
            )
        
        gc.collect()
        
        out[split] = hf_ds
        if rank == 0:
            print(f"[medtrinity:{split}] Ready" + (f" ({len(hf_ds):,} examples)" if not streaming else " (streaming)"))
    
    # Sync all ranks
    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            print(f"[medtrinity] All {world_size} ranks synchronized")
    
    if not out:
        raise FileNotFoundError("[medtrinity] No valid splits found")
    
    return DatasetDict(out)



# # --- MedTrinity-25M report loader (same schema as IU-XRay/MIMIC-CXR) ---

import os, json, ast
import re
import pandas as pd
from typing import List, Optional
# (keep your other imports the same)

def _safe_read_jsonl(
    path: str,
    on_bad_lines: str = "skip",              # "skip" | "error"
    max_bad_lines_to_log: int = 10
) -> pd.DataFrame:
    """
    Robust JSONL reader:
      * utf-8 with replacement for bad bytes
      * json.loads(strict=False)
      * fallback to ast.literal_eval for single-quoted dicts
      * skip lines we can't parse (unless on_bad_lines="error")
    """
    rows, bad = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, raw in enumerate(f, 1):
            s = raw.strip()
            if not s:
                continue
            # Fast path
            try:
                rows.append(json.loads(s))
                continue
            except json.JSONDecodeError as e1:
                pass
            # Slightly more permissive (control chars)
            try:
                rows.append(json.loads(s, strict=False))
                continue
            except Exception as e2:
                pass
            # Single-quoted Python-literal style lines (some exporters do this)
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, dict):
                    rows.append(obj)
                    continue
            except Exception:
                pass

            # Couldn’t parse this line
            if on_bad_lines == "error":
                raise ValueError(f"{path}:{ln}: malformed JSONL line")  # keep concise
            if len(bad) < max_bad_lines_to_log:
                # Log a short preview of the offending line
                bad.append((ln, s[:200]))

    if bad:
        print(f"[medtrinity] WARNING: skipped {len(bad)} malformed lines in {path} (showing up to {max_bad_lines_to_log})")
        for ln, preview in bad:
            print(f"  line {ln}: {preview!r}")

    return pd.DataFrame(rows)



@register_dataset("medpix_cliqa_report")
def load_medpix_cliqa_report(
    num_proc: int = 1,
    batch_size: int = 2048,
    csv_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/MEDPIX-ClinQA/data_csv",
    user_prompt: str = "Write a detailed radiology report based on this scan.",
    debug_limit: Optional[int] = None,
) -> DatasetDict:
    
    REPORT_PROMPTS = [
        "Give a detailed medical description of the image. Identify the imaging modality and describe the anatomical region visible.",
        "Based on the image, identify the modality and provide a comprehensive analysis of what is visible.",
        "Generate a detailed radiology report based on the given medical image.",
        "Analyze the medical image and provide a structured report describing any findings, abnormalities, and impressions.",
        "Write a comprehensive medical report, including anatomical observations and pathological findings visible in the image.",
        "Please describe everything clinically relevant visible in this scan, including abnormal signals or lesions.",
        "Write a detailed radiology report based on this scan.",
    ]
    
    def _load_split(split: str):
        df = pd.read_csv(os.path.join(csv_dir, f"{split}.csv"))
        
        if debug_limit:
            df = df.head(debug_limit)
        
        # Convert to dataset format
        import random
        data = {"images": [], "messages": []}
        
        for _, row in df.iterrows():
            selected_prompt = random.choice(REPORT_PROMPTS)
            data["images"].append([row['image_path']])
            data["messages"].append([
                {"role": "user", "content": [
                    {"type": "text", "text": selected_prompt, "index": None},
                    {"type": "image", "text": None, "index": 0}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": row['answer'], "index": None}
                ]}
            ])
        
        return Dataset.from_dict(data)
    
    return DatasetDict({
        "train": _load_split("train"),
        "validation": _load_split("validation")
    })


@register_dataset("roco_report")
def load_roco_report_from_csv(
    csv_path: str = "./Medmo_Dataset_1/Medmo_Dataset/ROCO-Radiology/data/data_with_image_paths.csv",
    num_proc: int = 1,
    batch_size: int = 1024,
    n_validation: int = 100,   # sample count for validation
    n_test: int = 0,           # sample count for test (0 = none)
    random_state: int = 42,
    cache_to: Optional[str] = None,
    debug_limit: Optional[int] = None,
) -> DatasetDict:
    """
    Build HF DatasetDict from a CSV with columns:
      - image (absolute path string)
      - image_id (str)
      - caption (str) -> used as assistant answer
    Output columns:
      - messages: [[{user with image token & prompt}, {assistant with caption}]]
      - images: [[<path>]]   (PATHS ONLY; no PIL, no bytes)
    """

    REPORT_PROMPTS = [
        "Provide a detailed clinical interpretation of the medical image.",
        "Describe all visible anatomical structures and any abnormalities present in the image.",
        "Summarize the relevant medical findings shown in this image.",
        "Generate a comprehensive report based on the content of the medical image.",
        "Identify the anatomical region and discuss any pathological signs present.",
        "Write a detailed medical description highlighting both normal and abnormal observations.",
        "What are the key diagnostic features visible in this image?",
        "Based on this image, what can you infer about the patient's condition?",
        "List and describe any clinical observations supported by the image.",
        "Explain what this image reveals about the underlying disease or treatment status.",
    ]

    def _stable_pick(key: str, k: int) -> int:
        h = hashlib.sha256((key or "").encode("utf-8")).digest()
        return h[0] % k

    # Load CSV
    df = pd.read_csv(csv_path)
    if debug_limit is not None and debug_limit > 0 and len(df) > debug_limit:
        df = df.sample(debug_limit, random_state=random_state).reset_index(drop=True)

    # Ensure required columns exist
    for col in ["image", "image_id", "caption"]:
        if col not in df.columns:
            df[col] = ""

    # Deterministic split indices
    rng = random.Random(random_state)
    idx = list(range(len(df)))
    rng.shuffle(idx)

    n_val = min(max(n_validation, 0), len(idx))
    n_tst = min(max(n_test, 0), max(0, len(idx) - n_val))
    n_trn = max(0, len(idx) - n_val - n_tst)

    val_idx = set(idx[:n_val])
    tst_idx = set(idx[n_val:n_val + n_tst])
    trn_idx = set(idx[n_val + n_tst:])

    def _frame_to_dataset(frame: pd.DataFrame) -> Dataset:
        # Build prompts deterministically from image_id (fallback to image path)
        keys = frame["image_id"].astype(str).where(frame["image_id"].astype(str) != "", frame["image"].astype(str))
        prompts = [REPORT_PROMPTS[_stable_pick(k, len(REPORT_PROMPTS))] for k in keys]

        answers = frame["caption"].astype(str).tolist()
        paths = [[p] for p in frame["image"].astype(str).tolist()]  # [[path]]

        messages = []
        for pr, an in zip(prompts, answers):
            user_turn = {"role": "user", "content": [
                {"type": "image", "index": 0, "text": None},
                {"type": "text", "text": pr, "index": None},
            ]}
            asst_turn = {"role": "assistant", "content": [
                {"type": "text", "text": an or "", "index": None},
            ]}
            messages.append([user_turn, asst_turn])

        return Dataset.from_dict({"messages": messages, "images": paths})

    d_train = _frame_to_dataset(df.loc[sorted(trn_idx)].reset_index(drop=True)) if n_trn else Dataset.from_dict({"messages": [], "images": []})
    d_val   = _frame_to_dataset(df.loc[sorted(val_idx)].reset_index(drop=True)) if n_val else Dataset.from_dict({"messages": [], "images": []})
    d_test  = _frame_to_dataset(df.loc[sorted(tst_idx)].reset_index(drop=True)) if n_tst else Dataset.from_dict({"messages": [], "images": []})

    dsd = DatasetDict({"train": d_train, "validation": d_val, "test": d_test})

    # Keep only required columns
    for sp in dsd:
        keep = {"messages", "images"}
        drop = [c for c in dsd[sp].column_names if c not in keep]
        if drop:
            dsd[sp] = dsd[sp].remove_columns(drop)

    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        dsd.save_to_disk(cache_to)
        dsd = DatasetDict.load_from_disk(cache_to)

    return dsd




from typing import Optional, Dict, Any, List
from datasets import Dataset, DatasetDict
from pathlib import Path
import pandas as pd
import os, hashlib, json, math, random

@register_dataset("rocov2_report")
def load_roco_report(
    csv_path: str = "./Medmo_Dataset_1/Medmo_Dataset/ROCOv2-Radiology/data/data_with_image_paths.csv",
    batch_size: int = 2048,
    num_proc: int = 1,
    n_validation: int = 100,
    n_test: int = 0,                 # set >0 if you want a test split sampled too
    random_state: int = 42,
    cache_to: Optional[str] = None,
    debug_limit: Optional[int] = None,
) -> DatasetDict:
    """
    Build a DatasetDict from a CSV with columns:
      - image (absolute path string)
      - image_id (str)
      - caption (str)  -> used as ground-truth answer
      - cui (list-like or str)
    Outputs:
      - messages: [[{user with image token & prompt}, {assistant with caption}]]
      - images: [[<path>]]  (PATHS ONLY; no PIL, no bytes)
    """
    REPORT_PROMPTS = [
        "Provide a detailed clinical interpretation of the medical image.",
        "Describe all visible anatomical structures and any abnormalities present in the image.",
        "Summarize the relevant medical findings shown in this image.",
        "Generate a comprehensive report based on the content of the medical image.",
        "Identify the anatomical region and discuss any pathological signs present.",
        "Write a detailed medical description highlighting both normal and abnormal observations.",
        "What are the key diagnostic features visible in this image?",
        "Based on this image, what can you infer about the patient's condition?",
        "List and describe any clinical observations supported by the image.",
        "Explain what this image reveals about the underlying disease or treatment status.",
    ]

    def _stable_pick(key: str, k: int) -> int:
        h = hashlib.sha256((key or "").encode("utf-8")).digest()
        return h[0] % k

    # --- Load CSV ---
    df = pd.read_csv(csv_path)
    if debug_limit is not None and debug_limit > 0 and len(df) > debug_limit:
        df = df.sample(debug_limit, random_state=random_state).reset_index(drop=True)

    # Coerce required columns
    for col in ["image", "image_id", "caption"]:
        if col not in df.columns:
            df[col] = ""

    # Optional: ensure cui is a stringified list
    if "cui" in df.columns:
        df["cui"] = df["cui"].astype(str)

    # --- Split: train / validation (/ test optional) ---
    rng = random.Random(random_state)
    idx = list(range(len(df)))
    rng.shuffle(idx)

    n_val = min(n_validation, len(idx)) if n_validation and n_validation > 0 else 0
    n_tst = min(n_test, max(0, len(idx) - n_val)) if n_test and n_test > 0 else 0
    n_trn = max(0, len(idx) - n_val - n_tst)

    val_idx = set(idx[:n_val])
    tst_idx = set(idx[n_val:n_val + n_tst])
    trn_idx = set(idx[n_val + n_tst:])

    def _df_to_hfds(sub: pd.DataFrame) -> Dataset:
        # Build prompts deterministically from image_id (fallback to path)
        prompts = []
        for uid, p in zip(sub.get("image_id", [""] * len(sub)), sub["image"]):
            key = str(uid) if isinstance(uid, str) and uid else str(p)
            prompts.append(REPORT_PROMPTS[_stable_pick(key, len(REPORT_PROMPTS))])

        answers = list(sub.get("caption", [""] * len(sub)))
        paths = [[p] for p in sub["image"].astype(str).tolist()]  # [[path]]

        # Construct chat messages
        messages = []
        for prompt, answer in zip(prompts, answers):
            user_turn = {"role": "user", "content": [
                {"type": "image", "index": 0, "text": None},
                {"type": "text", "text": prompt, "index": None},
            ]}
            assistant_turn = {"role": "assistant", "content": [
                {"type": "text", "text": answer or "", "index": None}
            ]}
            messages.append([user_turn, assistant_turn])

        data = {"messages": messages, "images": paths}
        return Dataset.from_dict(data)

    d_train = _df_to_hfds(df.loc[sorted(trn_idx)].reset_index(drop=True)) if n_trn else Dataset.from_dict({"messages": [], "images": []})
    d_val   = _df_to_hfds(df.loc[sorted(val_idx)].reset_index(drop=True)) if n_val else Dataset.from_dict({"messages": [], "images": []})
    d_test  = _df_to_hfds(df.loc[sorted(tst_idx)].reset_index(drop=True)) if n_tst else Dataset.from_dict({"messages": [], "images": []})

    dsd = DatasetDict({"train": d_train, "validation": d_val, "test": d_test})

    # Keep only required columns
    for sp in dsd:
        keep = {"messages", "images"}
        drop = [c for c in dsd[sp].column_names if c not in keep]
        if drop:
            dsd[sp] = dsd[sp].remove_columns(drop)

    # Optional cache
    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        dsd.save_to_disk(cache_to)
        dsd = DatasetDict.load_from_disk(cache_to)

    return dsd




from typing import Optional, List, Dict, Any
import os, hashlib
import pandas as pd
from datasets import Dataset, DatasetDict


@register_dataset("chexpert_plus_report")
def load_chexpert_plus_report(
    num_proc: int = 8,
    batch_size: int = 256,
    image_root: str = "./Medmo_Dataset_1/Medmo_Dataset/CheXpert-Plus/images/PNG",
    train_csv: str = "./Medmo_Dataset_1/Medmo_Dataset/CheXpert-Plus/df_train_good.csv",
    val_csv: str   = "./Medmo_Dataset_1/Medmo_Dataset/CheXpert-Plus/df_val_good.csv",
    test_csv: Optional[str] = None,
    include_context: bool = True,
    debug_limit: Optional[int] = None,
    check_files: bool = False,
    cache_to: Optional[str] = None,
) -> DatasetDict:
    """
    CheXpert-Plus CSV → HF Dataset (optimized version):
      - Single-pass processing: builds everything in one map operation
      - No repeated conversions or intermediate steps
      - Deterministic prompt per UID with optional context
    """

    # Check if cached version exists
    if cache_to and os.path.exists(cache_to):
        try:
            print(f"[chexpert+] Loading from cache: {cache_to}")
            return DatasetDict.load_from_disk(cache_to)
        except Exception as e:
            print(f"[chexpert+] Cache load failed: {e}, rebuilding...")

    REPORT_PROMPTS = [
        "Generate a comprehensive radiology report based on this chest X-ray.",
        "Describe anatomical structures and any visible pathological findings in the image.",
        "Write a structured radiology report including findings and impressions.",
        "Provide a detailed clinical analysis based on this medical scan.",
        "Summarize all abnormalities or clinical signals seen in the image.",
        "Analyze this scan and provide a detailed diagnostic report.",
        "Give a radiological interpretation of the image including summary if any.",
    ]

    def _stable_pick(key: str, k: int) -> int:
        h = hashlib.sha256((key or "").encode("utf-8")).digest()
        return h[0] % k

    def _read_csv_to_df(csv_path: str) -> pd.DataFrame:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[chexpert+] Missing CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        def _get_str_col(name: str, default_len: int) -> pd.Series:
            if name in df.columns:
                s = df[name]
                if not isinstance(s, pd.Series):
                    s = pd.Series(s)
            else:
                s = pd.Series([""] * default_len)
            return s.astype(str).fillna("")

        n = len(df)

        # Build ground-truth answer
        f = _get_str_col("findings", n)
        i = _get_str_col("impression", n)
        s = _get_str_col("summary", n)

        answers = []
        for ff, ii, ss in zip(f, i, s):
            parts = []
            if ff.strip() and ff.strip().lower() != "nan":
                parts.append(ff.strip())
            if ii.strip() and ii.strip().lower() != "nan":
                parts.append(ii.strip())
            if ss.strip() and ss.strip().lower() != "nan":
                parts.append(f"Finally: {ss.strip()}")
            answers.append(" ".join(parts).strip())
        df["answer"] = answers

        # Full path to image
        img_col = "path_to_image_png" if "path_to_image_png" in df.columns else (
                "path_to_image" if "path_to_image" in df.columns else None)
        if img_col is None:
            raise ValueError("[chexpert+] CSV must contain 'path_to_image_png' or 'path_to_image'.")

        full_paths = df[img_col].astype(str).fillna("").tolist()
        full_paths = [
            p if os.path.isabs(p) else os.path.normpath(os.path.join(image_root, p))
            for p in full_paths
        ]
        df["full_path"] = full_paths
        df = df[df["full_path"].str.len() > 0].copy()

        # UID for deterministic prompt
        uid_col = "study_id" if "study_id" in df.columns else img_col
        df["uid"] = df[uid_col].astype(str)

        # Optional clinical context for prompt
        if include_context:
            hist = _get_str_col("history", n)
            age  = _get_str_col("age", n)
            sex  = _get_str_col("sex", n)
            race = _get_str_col("race", n)

            ctx_prefix = []
            for h, a, s_, r in zip(hist, age, sex, race):
                ctx = ""
                h = h.strip()
                if h and h.lower() != "nan":
                    ctx += f"Patient history: {h}. "
                if any([a, s_, r]):
                    ctx += f"Demographics - Age: {a}, Sex: {s_}, Race: {r}. "
                ctx_prefix.append(ctx)
        else:
            ctx_prefix = [""] * len(df)

        # Build caption with deterministic prompt
        captions = []
        for uid, ctx in zip(df["uid"].tolist(), ctx_prefix):
            pidx = _stable_pick(uid, len(REPORT_PROMPTS))
            captions.append(f"{ctx}{REPORT_PROMPTS[pidx]}".strip())
        df["caption"] = captions

        return df[["full_path", "caption", "answer"]]

    def _df_to_hfds(df: pd.DataFrame, split: str) -> Dataset:
        ds = Dataset.from_pandas(df, preserve_index=False)

        # Early debug cap
        if debug_limit is not None and debug_limit > 0 and len(ds) > debug_limit:
            ds = ds.select(range(debug_limit))

        # Optional existence check
        if check_files:
            ds = ds.filter(
                lambda p: bool(p) and os.path.exists(p),
                input_columns="full_path",
                desc=f"[chexpert+:{split}] exists(full_path)"
            )

        # SINGLE-PASS PROCESSING: build messages and images in one operation
        def _format_all(batch):
            full_paths = batch["full_path"]
            captions = batch["caption"]
            answers = batch["answer"]
            
            messages_out = []
            images_out = []
            
            for path, caption, answer in zip(full_paths, captions, answers):
                # Build images list
                images_out.append([path])
                
                # Build messages with both user and assistant turns
                messages_out.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": caption, "index": None},
                            {"type": "image", "text": None, "index": 0},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer or "", "index": None},
                        ],
                    },
                ])
            
            return {"messages": messages_out, "images": images_out}

        ds = ds.map(
            _format_all,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=ds.column_names,  # Remove all original columns
            desc=f"[chexpert+:{split}] format messages & images",
        )

        return ds

    # Process all splits
    splits_df = {
        "train": _read_csv_to_df(train_csv),
        "validation": _read_csv_to_df(val_csv),
    }
    if test_csv:
        splits_df["test"] = _read_csv_to_df(test_csv)

    out = {sp: _df_to_hfds(df, sp) for sp, df in splits_df.items()}
    d = DatasetDict(out)

    # Cache if requested
    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        print(f"[chexpert+] Saving to cache: {cache_to}")
        d.save_to_disk(cache_to)

    return d

import os
import pandas as pd
from typing import Optional, Dict, List
from datasets import Dataset, DatasetDict

@register_dataset("vqa_med_2019")
def load_vqa_med_2019(
    images_root: str = "./Medmo_Dataset_1/Medmo_Dataset/VQA-Med-2019/images",
    combined_csv: str = "./Medmo_Dataset_1/Medmo_Dataset/VQA-Med-2019/data/all.csv",
    train_csv: Optional[str] = "./Medmo_Dataset_1/Medmo_Dataset/VQA-Med-2019/data/all.csv",
    validation_csv: Optional[str] = "./Medmo_Dataset_1/Medmo_Dataset/VQA-Med-2019/data/validation.csv",
    test_csv: Optional[str] = "./Medmo_Dataset_1/Medmo_Dataset/VQA-Med-2019/data/test.csv",
    num_proc: int = 1,
    batch_size: int = 1024,
    drop_missing_images: bool = True,
    debug_limit: Optional[int] = None,
    cache_to: Optional[str] = None,
) -> DatasetDict:
    """
    VQA-Med-2019 loader (image paths only; no bytes).
    - Preferred input is `combined_csv` (must contain a 'split' column with values train/validation/test).
    - Falls back to per-split CSVs if `combined_csv` is missing or lacks 'split'.

    Expected CSV columns: image (filename), question, answer. (category is optional)
    Output schema: {'messages': [[user, assistant]], 'images': [[<abs_path>]]}
    """

    def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df

    def _abs_paths(names: List[str]) -> List[List[str]]:
        out = []
        for n in names:
            p = n if os.path.isabs(n) else os.path.join(images_root, str(n))
            out.append([p])
        return out

    def _frame_to_dataset(frame: pd.DataFrame) -> Dataset:
        if debug_limit and len(frame) > debug_limit:
            frame = frame.sample(debug_limit, random_state=42).reset_index(drop=True)

        frame = _ensure_cols(frame, ["image", "question", "answer"])

        # Build images as list-of-one absolute path
        images = _abs_paths(frame["image"].astype(str).tolist())

        # Optionally drop rows with missing files
        if drop_missing_images:
            keep = [i for i, (lst,) in enumerate([(x,) for x in images]) if os.path.exists(lst[0])]
            if len(keep) < len(images):
                frame = frame.iloc[keep].reset_index(drop=True)
                images = [images[i] for i in keep]

        # Build messages
        qs = frame["question"].astype(str).tolist()
        ans = frame["answer"].astype(str).tolist()
        messages = []
        for q, a in zip(qs, ans):
            user = {"role": "user", "content": [
                {"type": "image", "index": 0, "text": None},
                {"type": "text", "text": q, "index": None},
            ]}
            asst = {"role": "assistant", "content": [
                {"type": "text", "text": a or "", "index": None},
            ]}
            messages.append([user, asst])

        return Dataset.from_dict({"messages": messages, "images": images})

    def _from_combined(csv_path: str) -> Optional[DatasetDict]:
        if not (csv_path and os.path.isfile(csv_path)):
            return None
        df = pd.read_csv(csv_path)
        if "split" not in df.columns:
            return None
        df["split"] = df["split"].astype(str).str.lower()

        out: Dict[str, Dataset] = {}
        for sp in ["train", "validation", "test"]:
            sub = df[df["split"] == sp].reset_index(drop=True)
            out[sp] = _frame_to_dataset(sub) if len(sub) else Dataset.from_dict({"messages": [], "images": []})
        return DatasetDict(out)

    # Prefer combined CSV if valid; else fall back to per-split CSVs
    dsd = _from_combined(combined_csv)
    if dsd is None:
        # train_csv may point to all.csv; if so and no 'split', we treat it as train-only
        train_df = pd.read_csv(train_csv) if (train_csv and os.path.isfile(train_csv)) else pd.DataFrame()
        val_df   = pd.read_csv(validation_csv) if (validation_csv and os.path.isfile(validation_csv)) else pd.DataFrame()
        test_df  = pd.read_csv(test_csv) if (test_csv and os.path.isfile(test_csv)) else pd.DataFrame()

        d_train = _frame_to_dataset(train_df) if len(train_df) else Dataset.from_dict({"messages": [], "images": []})
        d_val   = _frame_to_dataset(val_df)   if len(val_df)   else Dataset.from_dict({"messages": [], "images": []})
        d_test  = _frame_to_dataset(test_df)  if len(test_df)  else Dataset.from_dict({"messages": [], "images": []})
        dsd = DatasetDict({"train": d_train, "validation": d_val, "test": d_test})

    # Keep only required columns (defensive – our dicts already match)
    for sp in list(dsd.keys()):
        keep = {"messages", "images"}
        drop = [c for c in dsd[sp].column_names if c not in keep]
        if drop:
            dsd[sp] = dsd[sp].remove_columns(drop)

    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        dsd.save_to_disk(cache_to)
        dsd = DatasetDict.load_from_disk(cache_to)

    return dsd




@register_dataset("pubmed_vision")
def load_pubmed_vision(
    num_proc: int = 8,
    batch_size: int = 128,
    json_paths: dict | None = None,
    image_root: str = "./Medmo_Dataset_1/Medmo_Dataset/PubMedVision",
) -> DatasetDict:
    """
    PubMedVision Alignment VQA JSON loader.
    Produces unified schema ['messages','images'].
    Keeps train/validation/test as separate disjoint sources.
    """

    # Default split paths
    SPLIT_PATHS = {
        "train":      os.path.join(image_root, "PubMedVision_Alignment_VQA_clean_ad.json"),
        "validation": os.path.join(image_root, "PubMedVision_Alignment_VQA_clean_val_ad.json"),
        "test":       os.path.join(image_root, "PubMedVision_Alignment_VQA_clean_val.json"),
    }
    if json_paths is not None:
        SPLIT_PATHS = json_paths

    # Process all splits first
    all_splits = {}
    for split, path in SPLIT_PATHS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[pubmed_vision] Missing JSON for split '{split}': {path}")

        # 1) Load JSON → DataFrame
        try:
            df = pd.read_json(path)
        except Exception:
            if "_safe_read_jsonl" in globals():
                df = _safe_read_jsonl(path, on_bad_lines="skip")
            else:
                raise ValueError(f"[pubmed_vision:{split}] Failed to parse {path}")

        # 2) Standardize image paths
        if "image" not in df.columns:
            raise ValueError(f"[pubmed_vision:{split}] No 'image' column found in {path}")
        df["full_path"] = df["image"].apply(lambda x: os.path.join(image_root, x))

        # 3) Build unified schema
        def _row_to_messages(row):
            return [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ]

        df["messages"] = df.apply(_row_to_messages, axis=1)
        df["images"] = df["full_path"].apply(lambda x: [x])

        # Optional UID
        if "uid" not in df.columns:
            df["uid"] = df.index.astype(str)

        # Guard empty
        n_rows = len(df)
        if n_rows == 0:
            raise ValueError(f"[pubmed_vision:{split}] 0 rows after parsing {path}")

        all_splits[split] = df

    train_df = all_splits["train"].reset_index(drop=True) if "train" in all_splits else None
    val_df   = all_splits["validation"].reset_index(drop=True) if "validation" in all_splits else None
    test_df  = all_splits["test"].reset_index(drop=True) if "test" in all_splits else None
    # 4) Convert → HF dataset
    out = {}
    out["train"] = _to_hfds(
        train_df, 
        num_proc=num_proc, 
        batch_size=batch_size, 
        desc_prefix=f"[pubmed_vision:train]"
    )
    
    # Keep validation separate (using the validation split)
    if "validation" in all_splits:
        out["validation"] = _to_hfds(
            val_df, 
            num_proc=num_proc, 
            batch_size=batch_size, 
            desc_prefix=f"[pubmed_vision:validation]"
        )

    return DatasetDict(out)
 


@register_dataset("omnimed_vqa")
def load_omnimed_vqa(
    num_proc: int = 1,                   # kept for interface parity with other loaders
    batch_size: int = 1024,              # unused – data is pre-tokenized into chat format
    dataset_root: str = "./Medmo_Dataset_1/Medmo_Dataset/OmniMedVQA/OmniMedVQA",
    combined_json_path: Optional[str] = None,
    drop_missing_images: bool = True,
    debug_limit: Optional[int] = None,
) -> DatasetDict:
    """Load OmniMedVQA from the pre-combined Open-access JSON file."""

    dataset_root = os.path.abspath(dataset_root)
    combined_json_path = combined_json_path or os.path.join(
        dataset_root, "QA_information", "combined_open_access.json"
    )
    image_root = os.path.join(dataset_root, "Images_resize")

    if not os.path.isfile(combined_json_path):
        raise FileNotFoundError(
            f"[omnimed_vqa] Combined JSON file not found: {combined_json_path}"
        )
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"[omnimed_vqa] Image directory not found: {image_root}")

    def _format_question(row: Dict[str, Any]) -> str:
        question = (row.get("question") or "").strip()
        option_lines = []
        for label in ["A", "B", "C", "D"]:
            opt = row.get(f"option_{label}")
            if isinstance(opt, str) and opt.strip():
                option_lines.append(f"{label}. {opt.strip()}")

        meta_bits = []
        q_type = (row.get("question_type") or "").strip()
        modality = (row.get("modality_type") or row.get("modality") or "").strip()
        if q_type:
            meta_bits.append(f"Type: {q_type}")
        if modality:
            meta_bits.append(f"Modality: {modality}")

        parts = [question]
        if option_lines:
            parts.append("Options:\n" + "\n".join(option_lines))
        if meta_bits:
            parts.append(" | ".join(meta_bits))

        text = "\n".join([p for p in parts if p]).strip()
        return text or "Answer the medical question about the attached image."

    def _resolve_path(rel_path: str) -> str:
        rel_path = (rel_path or "").strip()
        if not rel_path:
            return ""
        norm = rel_path.replace("\\", "/")
        if os.path.isabs(norm):
            return os.path.normpath(norm)
        return os.path.normpath(os.path.join(dataset_root, norm))

    records: List[Dict[str, Any]] = []
    missing_images = 0
    limit = int(debug_limit) if (debug_limit is not None and debug_limit >= 0) else None

    try:
        with open(combined_json_path, "r") as f:
            entries = json.load(f)
    except Exception as exc:
        raise ValueError(f"[omnimed_vqa] Failed to parse {combined_json_path}: {exc}") from exc

    if not isinstance(entries, list):
        raise ValueError(f"[omnimed_vqa] Expected a list in {combined_json_path}")

    for idx, row in enumerate(entries):
        img_path = _resolve_path(row.get("image_path", ""))
        if not img_path:
            continue
        if drop_missing_images and not os.path.isfile(img_path):
            missing_images += 1
            continue

        question_text = _format_question(row)
        answer_text = (row.get("gt_answer") or "").strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0, "text": None},
                    {"type": "text", "text": question_text, "index": None},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer_text, "index": None},
                ],
            },
        ]

        dataset_name = row.get("dataset") or "omnimed"
        uid = row.get("question_id") or f"{dataset_name}_{idx}"
        records.append({"uid": str(uid), "messages": messages, "images": [img_path]})

        if limit is not None and len(records) >= limit:
            break

    if not records:
        raise ValueError("[omnimed_vqa] No usable samples were created. Check paths and JSON files.")

    if missing_images and rank == 0 and drop_missing_images:
        print(f"[omnimed_vqa] Skipped {missing_images} samples with missing images.")

    dataset = Dataset.from_list(records)
    return DatasetDict({"train": dataset})



@register_dataset("mmmu_med")
def load_mmmu_med(
    num_proc: int = 1,
    batch_size: int = 1024,
    dataset_json_path: str = "./Medmo_Dataset_1/Medmo_Dataset/MMMU/Medical_Dataset/medical_dataset.json",
    image_root: Optional[str] = "./Medmo_Dataset_1/Medmo_Dataset/MMMU/Medical_Dataset/images",
    drop_missing_images: bool = True,
    debug_limit: Optional[int] = None,
) -> DatasetDict:
    """Load MMMU-Med QA pairs from the consolidated JSON manifest."""

    dataset_json_path = os.path.abspath(dataset_json_path)
    image_root = os.path.abspath(image_root) if image_root else None
    dataset_dir = os.path.dirname(dataset_json_path)

    if not os.path.isfile(dataset_json_path):
        raise FileNotFoundError(
            f"[mmmu_med] JSON manifest not found: {dataset_json_path}"
        )
    if image_root and not os.path.isdir(image_root):
        raise FileNotFoundError(f"[mmmu_med] Image directory not found: {image_root}")

    try:
        with open(dataset_json_path, "r") as f:
            entries = json.load(f)
    except Exception as exc:
        raise ValueError(f"[mmmu_med] Failed to parse {dataset_json_path}: {exc}") from exc

    if not isinstance(entries, list):
        raise ValueError(f"[mmmu_med] Expected a list in {dataset_json_path}")

    def _resolve_img_path(rel_path: str) -> str:
        rel_path = (rel_path or "").strip()
        if not rel_path:
            return ""
        norm = rel_path.replace("\\", "/").lstrip("./")
        if os.path.isabs(norm):
            return os.path.normpath(norm)
        if image_root:
            if norm.lower().startswith("images/") and "/" in norm:
                norm = norm.split("/", 1)[1]
            return os.path.normpath(os.path.join(image_root, norm))
        return os.path.normpath(os.path.join(dataset_dir, norm))

    def _normalize_options(row: Dict[str, Any]) -> List[str]:
        options = row.get("options") or []
        if isinstance(options, str):
            options = [options]
        normalized = []
        for opt in options:
            if isinstance(opt, str):
                cleaned = opt.strip()
                if cleaned:
                    normalized.append(cleaned)
        return normalized

    def _format_question(row: Dict[str, Any]) -> str:
        question = (row.get("question") or "").strip()
        options = _normalize_options(row)

        option_lines = []
        for idx, opt in enumerate(options):
            label = chr(ord("A") + idx)
            option_lines.append(f"{label}. {opt}")

        parts = [question]
        if option_lines:
            parts.append("Options:\n" + "\n".join(option_lines))

        text = "\n\n".join([p for p in parts if p]).strip()
        return text or "Answer the multiple-choice medical question about the attached image(s)."

    def _format_answer(row: Dict[str, Any]) -> str:
        answer = (row.get("answer") or "").strip()
        options = _normalize_options(row)

        if len(answer) == 1 and answer.isalpha():
            idx = ord(answer.upper()) - ord("A")
            if 0 <= idx < len(options):
                return f"{answer.upper()}: {options[idx]}"
            return answer.upper()

        if answer:
            for idx, opt in enumerate(options):
                if opt.lower() == answer.lower():
                    label = chr(ord("A") + idx)
                    return f"{label}: {opt}"
            return answer

        return "No answer provided."

    records: List[Dict[str, Any]] = []
    missing_images = 0
    dropped_samples = 0
    limit = int(debug_limit) if (debug_limit is not None and debug_limit >= 0) else None

    for idx, row in enumerate(entries):
        raw_images = row.get("images") or []
        if isinstance(raw_images, str):
            raw_images = [raw_images]

        resolved_images: List[str] = []
        for rel_path in raw_images:
            img_path = _resolve_img_path(str(rel_path))
            if not img_path:
                continue
            if os.path.isfile(img_path):
                resolved_images.append(img_path)
            else:
                missing_images += 1

        if not resolved_images:
            if drop_missing_images:
                dropped_samples += 1
                continue

        question_text = _format_question(row)
        answer_text = _format_answer(row)
        explanation = (row.get("explanation") or "").strip()
        assistant_text = answer_text or "No answer provided."
        if explanation:
            assistant_text = f"{assistant_text}\n\nExplanation: {explanation}".strip()

        user_content = []
        for img_idx in range(len(resolved_images)):
            user_content.append({"type": "image", "index": img_idx, "text": None})
        user_content.append({"type": "text", "text": question_text, "index": None})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text, "index": None}]},
        ]

        uid = row.get("id") or f"mmmu_med_{idx}"
        records.append({"uid": str(uid), "messages": messages, "images": resolved_images})

        if limit is not None and len(records) >= limit:
            break

    if not records:
        raise ValueError("[mmmu_med] No usable samples were created. Check paths and JSON manifest.")

    if rank == 0:
        if drop_missing_images and (missing_images or dropped_samples):
            print(
                f"[mmmu_med] Dropped {dropped_samples} samples; "
                f"missing image files encountered: {missing_images}."
            )
        elif missing_images:
            print(f"[mmmu_med] Found {missing_images} missing image files (kept samples).")

    dataset = Dataset.from_list(records)
    return DatasetDict({"train": dataset})



@register_dataset("vqa_rad")
def load_vqa_rad(
    num_proc: int = 1,
    batch_size: int = 1024,
    dataset_json_path: str = "./Medmo_Dataset_1/Medmo_Dataset/VQA_RAD/data/VQA_RAD_Merged/vqa_rad_dataset.json",
    image_root: Optional[str] = "./Medmo_Dataset_1/Medmo_Dataset/VQA_RAD/data/VQA_RAD_Merged/images",
    drop_missing_images: bool = True,
    debug_limit: Optional[int] = None,
) -> DatasetDict:
    """
    Loader for the merged VQA-RAD dataset exported by build_vqa_rad_dataset.py.
    Each JSON record is expected to include:
        {"image": "images/<relative-path>.jpg", "question": "...", "answer": "..."}
    """

    dataset_json_path = os.path.abspath(dataset_json_path)
    if not os.path.isfile(dataset_json_path):
        raise FileNotFoundError(f"[vqa_rad] JSON file not found: {dataset_json_path}")

    base_dir = os.path.dirname(dataset_json_path)
    image_root = image_root or os.path.join(base_dir, "images")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"[vqa_rad] Image directory not found: {image_root}")

    try:
        with open(dataset_json_path, "r", encoding="utf-8") as handle:
            entries = json.load(handle)
    except Exception as exc:
        raise ValueError(f"[vqa_rad] Failed to parse {dataset_json_path}: {exc}") from exc

    if not isinstance(entries, list):
        raise ValueError(f"[vqa_rad] Expected a list in {dataset_json_path}")

    def _resolve_path(rel_path: str) -> str:
        rel_path = (rel_path or "").strip()
        if not rel_path:
            return ""
        if os.path.isabs(rel_path):
            return os.path.normpath(rel_path)
        candidate = os.path.normpath(os.path.join(base_dir, rel_path))
        if os.path.isfile(candidate):
            return candidate
        rel_tail = rel_path
        if rel_tail.startswith("images/"):
            rel_tail = rel_tail[len("images/") :]
        return os.path.normpath(os.path.join(image_root, rel_tail))

    limit = int(debug_limit) if (debug_limit is not None and debug_limit >= 0) else None
    records: List[Dict[str, Any]] = []
    missing_images = 0

    for idx, row in enumerate(entries):
        image_path = _resolve_path(row.get("image", ""))
        if not image_path or not os.path.isfile(image_path):
            missing_images += 1
            if drop_missing_images:
                continue

        question_text = (row.get("question") or "").strip()
        answer_text = (row.get("answer") or "").strip()
        if not question_text:
            question_text = "Answer the clinical question for the attached radiology image."
        if not answer_text:
            answer_text = "No answer provided."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0, "text": None},
                    {"type": "text", "text": question_text, "index": None},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text, "index": None}],
            },
        ]
        uid = row.get("id") or f"vqa_rad_{idx}"
        img_list = [image_path] if image_path else []
        records.append({"uid": str(uid), "messages": messages, "images": img_list})

        if limit is not None and len(records) >= limit:
            break

    if not records:
        raise ValueError("[vqa_rad] No usable samples were created. Check JSON manifest and image paths.")

    if rank == 0 and missing_images and drop_missing_images:
        print(f"[vqa_rad] Dropped {missing_images} samples with missing images.")

    dataset = Dataset.from_list(records)
    return DatasetDict({"train": dataset})



@register_dataset("slake")
def load_slake(
    num_proc: int = 1,
    batch_size: int = 1024,
    json_paths: Optional[List[str]] = None,
    image_root: str = "./Medmo_Dataset_1/Medmo_Dataset/SLAKE/imgs",
    drop_missing_images: bool = True,
    debug_limit: Optional[int] = None,
) -> DatasetDict:
    """
    Load the SLAKE VQA dataset by merging train/test/validation JSON files into a single split.
    Each JSON file must be a list of dicts containing at least 'img_name', 'question', and 'answer'.
    """

    default_jsons = [
        "./Medmo_Dataset_1/Medmo_Dataset/SLAKE/train.json",
        "./Medmo_Dataset_1/Medmo_Dataset/SLAKE/test.json",
        "./Medmo_Dataset_1/Medmo_Dataset/SLAKE/validation.json",
    ]
    json_paths = json_paths or default_jsons

    image_root = os.path.abspath(image_root)
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"[slake] Image root not found: {image_root}")

    entries: List[Dict[str, Any]] = []
    for path in json_paths:
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"[slake] Missing JSON file: {abs_path}")
        try:
            with open(abs_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            raise ValueError(f"[slake] Failed to parse {abs_path}: {exc}") from exc
        if not isinstance(data, list):
            raise ValueError(f"[slake] Expected a list in {abs_path}")
        entries.extend(data)

    def _resolve_image(rel_path: str) -> str:
        rel_path = (rel_path or "").strip()
        if not rel_path:
            return ""
        rel_path = rel_path.replace("\\", "/").lstrip("./")
        return os.path.normpath(os.path.join(image_root, rel_path))

    limit = int(debug_limit) if (debug_limit is not None and debug_limit >= 0) else None
    records: List[Dict[str, Any]] = []
    missing_images = 0

    for idx, row in enumerate(entries):
        image_path = _resolve_image(str(row.get("img_name", "")))
        if not image_path or not os.path.isfile(image_path):
            missing_images += 1
            if drop_missing_images:
                continue

        question_text = (row.get("question") or "").strip()
        answer_text = (row.get("answer") or "").strip()
        if not question_text:
            question_text = "Answer the medical question for the attached image."
        if not answer_text:
            answer_text = "No answer provided."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0, "text": None},
                    {"type": "text", "text": question_text, "index": None},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text, "index": None}],
            },
        ]
        uid = row.get("qid") or row.get("id") or f"slake_{idx}"
        records.append({"uid": str(uid), "messages": messages, "images": [image_path]})

        if limit is not None and len(records) >= limit:
            break

    if not records:
        raise ValueError("[slake] No usable samples were created. Check JSON manifests and image paths.")

    if rank == 0 and missing_images and drop_missing_images:
        print(f"[slake] Dropped {missing_images} samples with missing images.")

    dataset = Dataset.from_list(records)
    return DatasetDict({"train": dataset})



@register_dataset("slake_bbox")
def load_slake_bbox(
    num_proc: int = 1,
    batch_size: int = 1024,
    dataset_json_path: str = "./Medmo_Dataset_1/Medmo_Dataset/SLAKE/slake_bbox_dataset_1.json",
    drop_missing_images: bool = True,
    debug_limit: Optional[int] = None,
) -> DatasetDict:
    """
    Loader for SLAKE bounding-box instructions generated by build_slake_bbox_dataset.py.
    """

    dataset_json_path = os.path.abspath(dataset_json_path)
    if not os.path.isfile(dataset_json_path):
        raise FileNotFoundError(f"[slake_bbox] JSON file not found: {dataset_json_path}")

    try:
        with open(dataset_json_path, "r", encoding="utf-8") as handle:
            entries = json.load(handle)
    except Exception as exc:
        raise ValueError(f"[slake_bbox] Failed to parse {dataset_json_path}: {exc}") from exc

    if not isinstance(entries, list):
        raise ValueError(f"[slake_bbox] Expected a list in {dataset_json_path}")

    def _normalize_labels(raw_labels: Any) -> List[str]:
        labels: List[str] = []
        if isinstance(raw_labels, (list, tuple)):
            for item in raw_labels:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    labels.append(text)
        elif isinstance(raw_labels, str):
            text = raw_labels.strip()
            if text:
                labels.append(text)
        return labels

    def _normalize_boxes(raw_boxes: Any) -> List[List[float]]:
        boxes: List[List[float]] = []
        if isinstance(raw_boxes, (list, tuple)):
            for box in raw_boxes:
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                try:
                    boxes.append([float(v) for v in box])
                except (TypeError, ValueError):
                    continue
        return boxes

    def _format_question(labels: List[str]) -> str:
        if labels:
            targets = ", ".join(labels)
            return (
                "You are a medical imaging expert. Detect the following anatomical regions or findings in this image: "
                f"{targets}. Provide the disease name(s), if any, and bounding box coordinates as "
                "[[x_min, y_min, x_max, y_max]] in pixel values."
            )
        return (
            "You are a medical imaging expert. Detect the annotated anatomical regions or findings in this image and "
            "return bounding boxes as [[x_min, y_min, x_max, y_max]] in pixel values."
        )

    def _format_answer(labels: List[str], boxes: List[List[float]]) -> str:
        detections = [
            {"label": label, "bbox": [round(float(coord), 4) for coord in box]}
            for label, box in zip(labels, boxes)
        ]
        payload = {
            "box_format": "[x_min, y_min, x_max, y_max] (pixel)",
            "detections": detections,
        }
        return json.dumps(payload, ensure_ascii=False)

    limit = int(debug_limit) if (debug_limit is not None and debug_limit >= 0) else None
    records: List[Dict[str, Any]] = []
    missing_images = 0

    for idx, row in enumerate(entries):
        image_path = (row.get("image") or "").strip()
        if not image_path or not os.path.isfile(image_path):
            missing_images += 1
            if drop_missing_images:
                continue

        labels = _normalize_labels(row.get("labels"))
        boxes = _normalize_boxes(row.get("boxes"))
        if not labels or not boxes:
            continue

        question_text = _format_question(labels)
        answer_text = _format_answer(labels, boxes)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0, "text": None},
                    {"type": "text", "text": question_text, "index": None},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text, "index": None}],
            },
        ]
        uid = row.get("id") or f"slake_bbox_{idx}"
        records.append({"uid": str(uid), "messages": messages, "images": [image_path]})

        if limit is not None and len(records) >= limit:
            break

    if not records:
        raise ValueError("[slake_bbox] No usable samples were created. Check JSON manifest and image paths.")

    if rank == 0 and missing_images and drop_missing_images:
        print(f"[slake_bbox] Dropped {missing_images} samples with missing images.")

    dataset = Dataset.from_list(records)
    return DatasetDict({"train": dataset})



@register_dataset("path_vqa")
def load_path_vqa(
    num_proc: int = 1,
    batch_size: int = 1024,
    dataset_json_path: str = "./Medmo_Dataset_1/Medmo_Dataset/PATH_VQA/data/PATH_VQA_Merged/path_vqa_dataset.json",
    image_root: Optional[str] = "./Medmo_Dataset_1/Medmo_Dataset/PATH_VQA/data/PATH_VQA_Merged/images",
    drop_missing_images: bool = True,
    debug_limit: Optional[int] = None,
) -> DatasetDict:
    """
    Loader for the merged PATH-VQA dataset exported by build_path_vqa_dataset.py.
    Each entry is expected to contain:
        {"image": "images/<relative-path>.jpg", "question": "...", "answer": "..."}
    """

    dataset_json_path = os.path.abspath(dataset_json_path)
    if not os.path.isfile(dataset_json_path):
        raise FileNotFoundError(f"[path_vqa] JSON file not found: {dataset_json_path}")

    base_dir = os.path.dirname(dataset_json_path)
    image_root = image_root or os.path.join(base_dir, "images")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"[path_vqa] Image directory not found: {image_root}")

    try:
        with open(dataset_json_path, "r", encoding="utf-8") as handle:
            entries = json.load(handle)
    except Exception as exc:
        raise ValueError(f"[path_vqa] Failed to parse {dataset_json_path}: {exc}") from exc

    if not isinstance(entries, list):
        raise ValueError(f"[path_vqa] Expected a list in {dataset_json_path}")

    def _resolve_path(rel_path: str) -> str:
        rel_path = (rel_path or "").strip()
        if not rel_path:
            return ""
        if os.path.isabs(rel_path):
            return os.path.normpath(rel_path)
        candidate = os.path.normpath(os.path.join(base_dir, rel_path))
        if os.path.isfile(candidate):
            return candidate
        tail = rel_path
        if tail.startswith("images/"):
            tail = tail[len("images/") :]
        return os.path.normpath(os.path.join(image_root, tail))

    limit = int(debug_limit) if (debug_limit is not None and debug_limit >= 0) else None
    records: List[Dict[str, Any]] = []
    missing_images = 0

    for idx, row in enumerate(entries):
        image_path = _resolve_path(row.get("image", ""))
        if not image_path or not os.path.isfile(image_path):
            missing_images += 1
            if drop_missing_images:
                continue

        question_text = (row.get("question") or "").strip()
        answer_text = (row.get("answer") or "").strip()
        if not question_text:
            question_text = "Answer the medical question for the attached pathology image."
        if not answer_text:
            answer_text = "No answer provided."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0, "text": None},
                    {"type": "text", "text": question_text, "index": None},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text, "index": None}],
            },
        ]

        uid = row.get("id") or f"path_vqa_{idx}"
        records.append({"uid": str(uid), "messages": messages, "images": [image_path]})

        if limit is not None and len(records) >= limit:
            break

    if not records:
        raise ValueError("[path_vqa] No usable samples were created. Check JSON manifest and image paths.")

    if rank == 0 and missing_images and drop_missing_images:
        print(f"[path_vqa] Dropped {missing_images} samples with missing images.")

    dataset = Dataset.from_list(records)
    return DatasetDict({"train": dataset})



@register_dataset("pmc_vqa")
def load_pmc_vqa(
    num_proc: int = 1,
    batch_size: int = 1024,
    csv_specs: Optional[List[Dict[str, str]]] = None,
    drop_missing_images: bool = True,
    debug_limit: Optional[int] = None,
) -> DatasetDict:
    """
    Loader for the PMC-VQA dataset. Supports two image roots (figures/ and images/)
    and their corresponding CSVs by default. All entries are merged into a single split.
    """

    default_specs = [
        {
            "csv": "./Medmo_Dataset_1/Medmo_Dataset/PMC_VQA/train_2.csv",
            "image_root": "./Medmo_Dataset_1/Medmo_Dataset/PMC_VQA/figures",
        },
        {
            "csv": "./Medmo_Dataset_1/Medmo_Dataset/PMC_VQA/train.csv",
            "image_root": "./Medmo_Dataset_1/Medmo_Dataset/PMC_VQA/images",
        },
    ]
    csv_specs = csv_specs or default_specs

    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        return str(value).strip()

    def _strip_option_label(text: str, label: str) -> str:
        pattern = re.compile(rf"^{re.escape(label)}\s*[:\.\-\)\]]\s*", re.IGNORECASE)
        stripped = pattern.sub("", text, count=1)
        stripped = stripped.strip()
        return stripped or text

    def _normalize_options(row: Dict[str, Any]) -> Dict[str, str]:
        options = {}
        for label in ["A", "B", "C", "D"]:
            value = row.get(f"Choice {label}") or row.get(f"Choice_{label}")
            text = _clean_text(value)
            if text:
                options[label] = _strip_option_label(text, label)
        return options

    def _format_question(row: Dict[str, Any], options: Dict[str, str]) -> str:
        question = _clean_text(row.get("Question"))
        option_lines = [f"{label}. {text}" for label, text in options.items()]
        parts = [question]
        if option_lines:
            parts.append("Options:\n" + "\n".join(option_lines))
        text = "\n\n".join([p for p in parts if p]).strip()
        return text or "Answer the medical question for the attached image."

    def _format_answer(row: Dict[str, Any], options: Dict[str, str]) -> str:
        ans = _clean_text(row.get("Answer"))
        if ans:
            key = ans.upper()
            if key in options:
                return f"{key}: {options[key]}"
            for label, text in options.items():
                if text.lower() == ans.lower():
                    return f"{label}: {text}"
            return ans
        return "No answer provided."

    def _resolve_image_path(figure_path: str, image_root: str) -> str:
        figure_path = _clean_text(figure_path)
        if not figure_path:
            return ""
        figure_path = figure_path.replace("\\", "/").lstrip("./")
        return os.path.normpath(os.path.join(image_root, figure_path))

    limit = int(debug_limit) if (debug_limit is not None and debug_limit >= 0) else None
    records: List[Dict[str, Any]] = []
    missing_images = 0

    for spec in csv_specs:
        csv_path = os.path.abspath(spec["csv"])
        image_root = os.path.abspath(spec["image_root"])

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[pmc_vqa] Missing CSV file: {csv_path}")
        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"[pmc_vqa] Missing image root: {image_root}")

        df = pd.read_csv(csv_path)
        for row in df.to_dict(orient="records"):
            options = _normalize_options(row)
            question_text = _format_question(row, options)
            answer_text = _format_answer(row, options)
            caption = _clean_text(row.get("Caption"))
            assistant_text = answer_text
            if caption:
                assistant_text = f"{assistant_text}\n\nReason : {caption}".strip()

            image_path = _resolve_image_path(row.get("Figure_path"), image_root)
            if not image_path or not os.path.isfile(image_path):
                missing_images += 1
                if drop_missing_images:
                    continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": 0, "text": None},
                        {"type": "text", "text": question_text, "index": None},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text, "index": None}],
                },
            ]

            uid = row.get("index") or row.get("qid") or f"pmc_vqa_{len(records)}"
            records.append({"uid": str(uid), "messages": messages, "images": [image_path]})

            if limit is not None and len(records) >= limit:
                break
        if limit is not None and len(records) >= limit:
            break

    if not records:
        raise ValueError("[pmc_vqa] No usable samples were created. Check CSVs and image paths.")

    if rank == 0 and missing_images and drop_missing_images:
        print(f"[pmc_vqa] Dropped {missing_images} samples with missing images.")

    dataset = Dataset.from_list(records)
    return DatasetDict({"train": dataset})



@register_dataset("nih_vqa")
def load_nih_vqa(num_proc: int = 8, batch_size: int = 128) -> DatasetDict:
    """
    NIH ChestX-ray14 -> HF DatasetDict with:
      - messages: chat-style [{role, content:[{type:'text'|'image', ...}]}]
      - images:   list[str] file paths aligned with 'image' entries in messages
    Uses dataset-local prompts (NIH_PROMPTS) to avoid global collisions.
    """
    # --------------------------
    # Dataset spec (paths + column aliases)
    # --------------------------
    spec = CsvDatasetSpec(
        name="nih_vqa",
        splits={
            "train":      "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/Data_Entry_2017_clean.csv",
            "validation": "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/Data_Entry_2017_clean_val.csv",
            "test":       "/l/PathoGen/Imran/Ankan_Backup/Medmo_Dataset/NIH-Data/Data_Entry_2017_clean_val.csv",
        },
        # Map CSV → standardized columns (keep 'uid' optional; here we use Patient ID to stabilize prompt choice)
        column_map={
            "Image Index": "filename",        # image filename in CSV
            "Finding Labels": "caption",      # we store the NIH label(s) as the assistant's target text
            "Patient ID": "uid",              # deterministic prompt selection anchor (optional)
        },
        image_root="./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/images",
    )

    # --------------------------
    # Local (namespaced) prompts for NIH-VQA
    # --------------------------
    NIH_PROMPTS = [
        "Examine this chest X-ray and list all thoracic diseases or abnormalities present. If there are no findings, state 'No Finding'. Else give the abormality.",
        "Based on the image, identify all relevant diagnostic labels. If nothing abnormal is detected, return 'No Finding'. Else give the abormality.",
        "From this chest radiograph, determine whether any of the NIH-labeled conditions are visible. Include 'No Finding' if appropriate. Else give the abormality.",
        "Classify the image into one or more disease categories or label it as 'No Finding' if no pathology is present. Else give the abormality.",
        "Detect and list all medical findings visible in the image. If the image appears normal, indicate 'No Finding'. Else give the abormality.",
        "Look at the X-ray and return all applicable labels from the NIH dataset. If no disease is visible, return 'No Finding'. Else give the abormality.",
        "Provide all relevant disease labels for this X-ray image, or specify 'No Finding' if no abnormalities are present. Else give the abormality.",
        "Based on the chest X-ray, determine if the patient shows signs of any thoracic diseases or if the scan is normal. Else give the abormality.",
        "Classify this chest radiograph using NIH disease labels. Ensure to return 'No Finding' if no pathological signs are observed. Else give the abormality.",
    ]

    # Deterministic per-row prompt selector (scoped to this loader)
    import hashlib
    def _nih_det_prompt(uid_val, idx):
        key = str(uid_val) if uid_val is not None else str(idx)
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        return NIH_PROMPTS[h % len(NIH_PROMPTS)]

    # Batch formatter (scoped) — mirrors _format_batch_indexed_no_io but uses _nih_det_prompt
    def _nih_format_batch_indexed_no_io(samples):
        out_messages, out_images = [], []
        n = len(samples["full_path"])
        captions = samples.get("caption", [""] * n)
        uids     = samples.get("uid", [None] * n)

        for i in range(n):
            img_path = samples["full_path"][i]
            caption  = captions[i] if i < len(captions) else ""
            uid_val  = uids[i] if i < len(uids) else None

            imgs = [img_path] if isinstance(img_path, str) and img_path else []
            question = _nih_det_prompt(uid_val, i)

            user_content = (
                [{"type": "text", "text": question, "index": None},
                 {"type": "image", "text": None, "index": 0}]
                if imgs else
                [{"type": "text", "text": question, "index": None}]
            )
            msg = [
                {"role": "user",     "content": user_content},
                {"role": "assistant","content": [{"type": "text", "text": caption or "", "index": None}]},
            ]

            out_messages.append(msg)
            out_images.append(imgs)

        return {"messages": out_messages, "images": out_images}

    # --------------------------
    # Build each split → HF dataset using local formatter
    # --------------------------
    out = {}
    for split, csv_path in spec.splits.items():
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[{spec.name}] Missing CSV for split '{split}': {csv_path}")

        df = pd.read_csv(csv_path)
        # Standardize to ['full_path','caption', optional 'uid'] using your shared helper
        df = _alias_and_standardize(df, spec.column_map, spec.image_root)

        # To HF
        hf_ds = Dataset.from_pandas(df, preserve_index=False)
        hf_ds = hf_ds.map(
            _nih_format_batch_indexed_no_io,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"[{spec.name}:{split}] Formatting (indexed, no I/O)",
        )
        keep = {"messages", "images"}
        drop_cols = [c for c in hf_ds.column_names if c not in keep]
        if drop_cols:
            hf_ds = hf_ds.remove_columns(drop_cols)

        out[split] = hf_ds

    return DatasetDict(out)




# @register_dataset("quilt_llava_pretrain")
# def load_quilt_llava_pretrain(
#     num_proc: int = 8,
#     batch_size: int = 128,
#     json_paths: dict | None = None,
#     image_root: str = "/l/PathoGen/Imran/Ankan_Backup/Medmo_Dataset/Quilt-LLaVA-Pretrain/images/quilt_1m",
#     debug_limit: int | dict | None = 20,   # << debug sampler (None = full dataset)
# ) -> DatasetDict:
#     """
#     Quilt-LLaVA-Pretrain JSON loader (no image I/O at load).
#     Produces unified schema ['messages', 'images'] with one user Q and one assistant A.

#     Args:
#         json_paths: Optional mapping {split: json_path}
#         image_root: Base dir for images (prefixed to relative paths in 'image')
#         debug_limit: None for full data; int for per-split sample size; or dict per split
#                      (e.g., {"train": 1000, "validation": 200})
#     """

#     # Default split → file mapping (override via json_paths)
#     SPLIT_PATHS = {
#         "train":      "./Medmo_Dataset_1/Medmo_Dataset/Quilt-LLaVA-Pretrain/quilt_pretrain.json",
#         "validation": "./Medmo_Dataset_1/Medmo_Dataset/Quilt-LLaVA-Pretrain/quilt_pretrain_val.json",
#         # "test":       "./Medmo_Dataset_1/Medmo_Dataset/Quilt-LLaVA-Pretrain/quilt_pretrain_val.json",
#     }
#     if json_paths is not None:
#         SPLIT_PATHS = json_paths

#     def _extract_qa(conversations: list[dict]) -> tuple[str, str]:
#         """Find first human question and first gpt answer."""
#         q = None
#         a = None
#         for c in conversations:
#             frm = (c.get("from") or "").lower()
#             val = c.get("value", "")
#             if q is None and frm == "human":
#                 q = val
#             elif a is None and frm == "gpt":
#                 a = val
#             if q is not None and a is not None:
#                 break
#         q = (q or "").replace("\n<image>", "").replace("<image>", "").strip()
#         a = (a or "").strip()
#         return q, a

#     def _maybe_apply_debug(df: pd.DataFrame, split: str) -> pd.DataFrame:
#         """Apply debug sampling if requested (deterministic)."""
#         if debug_limit is None:
#             return df
#         # allow dict per split or single int for all
#         limit = debug_limit.get(split) if isinstance(debug_limit, dict) else debug_limit
#         if limit is None:
#             return df
#         try:
#             n = max(0, min(int(limit), len(df)))
#         except Exception:
#             return df  # ignore malformed debug_limit
#         if n == 0:
#             # Return empty but valid df (rarely useful). Safer: keep at least 1 if limit>0.
#             return df.head(0)
#         # deterministic random sample
#         return df.sample(n=n, random_state=1337).reset_index(drop=True)

#     out: dict[str, Dataset] = {}
#     for split, path in SPLIT_PATHS.items():
#         if not os.path.isfile(path):
#             raise FileNotFoundError(f"[quilt_llava_pretrain] Missing JSON for split '{split}': {path}")

#         # 1) Load JSON → DataFrame (robust to list-of-dicts)
#         try:
#             df = pd.read_json(path)
#         except Exception:
#             with open(path, "r") as f:
#                 data = json.load(f)  # fallback for large/irregular files
#             df = pd.DataFrame(data)

#         # 2) Basic column checks
#         if "image" not in df.columns or "conversations" not in df.columns:
#             raise ValueError(f"[quilt_llava_pretrain:{split}] Expected columns ['image','conversations'] in {path}")

#         # 3) Build absolute image path list
#         df["full_path"] = df["image"].apply(lambda rel: os.path.join(image_root, str(rel)))
#         df["images"] = df["full_path"].apply(lambda p: [p])

#         # 4) Build messages: user(question) -> assistant(answer)
#         qa = df["conversations"].apply(_extract_qa)
#         df["messages"] = qa.apply(lambda t: [
#             {"role": "user", "content": t[0]},
#             {"role": "assistant", "content": t[1]},
#         ])

#         # 5) Optional uid
#         if "uid" not in df.columns:
#             if "id" in df.columns:
#                 df["uid"] = df["id"].astype(str)
#             else:
#                 df["uid"] = df.index.astype(str)

#         # 6) Apply debug sampling if requested
#         df = _maybe_apply_debug(df, split)

#         # 7) Guard empty
#         n_rows = int(len(df))
#         if n_rows == 0:
#             raise ValueError(f"[quilt_llava_pretrain:{split}] 0 rows after parsing/sampling {path}")

#         # 8) Convert → HF dataset
#         out[split] = _to_hfds(df, num_proc=num_proc, batch_size=batch_size, desc_prefix=f"[quilt:{split}]")

#         # Optional: quick sanity
#         # print(f"[quilt_llava_pretrain:{split}] rows={n_rows} sample_image={df['full_path'].iloc[0]}")

#     return DatasetDict(out)



@register_dataset("quilt_llava_pretrain")
def load_quilt_llava_pretrain(
    num_proc: int = 8,            # kept for signature compatibility (unused in this self-contained version)
    batch_size: int = 128,        # kept for signature compatibility (unused here)
    json_paths: dict | None = None,
    image_root: str = "./Medmo_Dataset_1/Medmo_Dataset/Quilt-LLaVA-Pretrain/images/quilt_1m",
    debug_limit: int | dict | None = None,  # None = full; int = first N; dict per split, e.g. {"train":100}
):
    """
    Single-function, self-contained Quilt-LLaVA-Pretrain loader.
    Returns a HuggingFace DatasetDict with columns: ['uid', 'messages', 'images'].

    messages schema (segmented):
      [
        {"role":"user","content":[{"type":"text","text":<question>,"index":None},
                                  {"type":"image","index":0,"text":None}]},
        {"role":"assistant","content":[{"type":"text","text":<answer>,"index":None}]}
      ]

    images: [<absolute_image_path>]
    """
    import os, json, re
    import pandas as pd
    from datasets import Dataset, DatasetDict


    
    # --- local helpers (no globals) ---
    def _strip_image_tokens(s: str) -> str:
        if not isinstance(s, str):
            return ""
        return re.sub(r"\s*<\s*image\s*>\s*", " ", s, flags=re.IGNORECASE).strip()

    def _extract_qa(conversations):
        q, a = "", ""
        for c in (conversations or []):
            who = (c.get("from") or "").lower()
            val = c.get("value", "")
            if not q and who == "human":
                q = val
            elif not a and who == "gpt":
                a = val
            if q and a:
                break
        return _strip_image_tokens(q), (a or "").strip()

    def _apply_debug(df: pd.DataFrame, split: str) -> pd.DataFrame:
        if debug_limit is None:
            return df
        n = debug_limit.get(split) if isinstance(debug_limit, dict) else debug_limit
        if n is None:
            return df
        try:
            n = int(n)
        except Exception:
            return df
        if n <= 0:
            return df.head(0).reset_index(drop=True)
        return df.head(min(n, len(df))).reset_index(drop=True)

    # --- default split paths (override via json_paths) ---
    SPLIT_PATHS = {
        "train":      "./Medmo_Dataset_1/Medmo_Dataset/Quilt-LLaVA-Pretrain/quilt_pretrain.json",
        "validation": "./Medmo_Dataset_1/Medmo_Dataset/Quilt-LLaVA-Pretrain/quilt_pretrain_val.json",
        # add "test" here if/when available
    }
    if json_paths is not None:
        SPLIT_PATHS = json_paths

    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"[quilt_llava_pretrain] Image root not found: {image_root}")

    out = {}
    for split, path in SPLIT_PATHS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[quilt_llava_pretrain] Missing JSON for split '{split}': {path}")

        # Robust JSON load (list-of-dicts)
        try:
            df = pd.read_json(path)
        except Exception:
            with open(path, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)

        # Required fields
        if "image" not in df.columns or "conversations" not in df.columns:
            raise ValueError(f"[quilt_llava_pretrain:{split}] Expected columns ['image','conversations'] in {path}")

        # Absolute image paths and images column
        df["full_path"] = df["image"].apply(lambda rel: os.path.join(image_root, str(rel)))
        df["images"] = df["full_path"].apply(lambda p: [p])

        # Build segmented messages from conversations
        qa = df["conversations"].apply(_extract_qa)
        df["messages"] = qa.apply(
            lambda t: [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": t[0], "index": None},
                     {"type": "image", "index": 0, "text": None},
                 ]},
                {"role": "assistant",
                 "content": [
                     {"type": "text", "text": t[1], "index": None},
                 ]},
            ]
        )

        # uid
        df["uid"] = (df["id"].astype(str) if "id" in df.columns else df.index.astype(str))

        # Debug slice (head-based)
        df = _apply_debug(df, split)

        if len(df) == 0:
            raise ValueError(f"[quilt_llava_pretrain:{split}] 0 rows after parsing/sampling {path}")

        # Keep only required columns
        df = df[["uid", "messages", "images"]]

        # Pandas -> HF Dataset
        out[split] = Dataset.from_pandas(df, preserve_index=False)

    return DatasetDict(out)





@register_dataset("mimic_cxr_vqa")
def load_mimic_cxr_vqa(
    num_proc: int = 8,             # kept for signature compatibility (unused here)
    batch_size: int = 128,         # kept for signature compatibility (unused here)
    json_paths: dict | None = None,
    image_root: str = "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.1.0/files",
    debug_limit: int | dict | None = None,   # None = full; int = first N; dict per split, e.g. {"train":100}
):
    """
    Single-function, self-contained loader for MIMIC-CXR-VQA.
    Returns a HuggingFace DatasetDict with columns: ['uid', 'messages', 'images'].

    messages schema (segmented):
      [
        {"role":"user","content":[{"type":"text","text":<question>,"index":None},
                                  {"type":"image","index":0,"text":None}]},
        {"role":"assistant","content":[{"type":"text","text":<answer>,"index":None}]}
      ]

    images: [<absolute_image_path>]
    """
    import os, json, re
    import pandas as pd
    from datasets import Dataset, DatasetDict

    # --- local helpers (no globals) ---
    def _strip_image_tokens(s: str) -> str:
        if not isinstance(s, str):
            return ""
        # Remove variants like "<image>", "\n<image>", case-insensitive
        return re.sub(r"\s*<\s*image\s*>\s*", " ", s, flags=re.IGNORECASE).strip()

    def _normalize_answer(ans) -> str:
        if ans is None:
            return ""
        if isinstance(ans, list):
            return ", ".join(map(str, ans))
        return str(ans)

    def _extract_qa(row) -> tuple[str, str]:
        q = _strip_image_tokens(row.get("question", "") or "")
        a = _normalize_answer(row.get("answer"))
        return q, a

    def _apply_debug(df: pd.DataFrame, split: str) -> pd.DataFrame:
        if debug_limit is None:
            return df
        n = debug_limit.get(split) if isinstance(debug_limit, dict) else debug_limit
        if n is None:
            return df
        try:
            n = int(n)
        except Exception:
            return df
        if n <= 0:
            return df.head(0).reset_index(drop=True)
        return df.head(min(n, len(df))).reset_index(drop=True)

    # --- default split paths (override via json_paths) ---
    SPLIT_PATHS = {
        "train":      "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-VQA/train_clean_full.json",
        "validation": "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-VQA/valid_clean.json",
        "test":       "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-VQA/test_clean.json",
    }
    if json_paths is not None:
        SPLIT_PATHS = json_paths

    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"[mimic_cxr_vqa] Image root not found: {image_root}")

    out = {}
    for split, path in SPLIT_PATHS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[mimic_cxr_vqa] Missing JSON for split '{split}': {path}")

        # Robust JSON load: file may be a list OR {"data": [...]}
        with open(path, "r") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "data" in obj:
            records = obj["data"]
        elif isinstance(obj, list):
            records = obj
        else:
            raise ValueError(f"[mimic_cxr_vqa:{split}] Unexpected JSON structure in {path}")

        df = pd.DataFrame(records)

        # Required columns
        if "image_path" not in df.columns or "question" not in df.columns:
            raise ValueError(f"[mimic_cxr_vqa:{split}] Expected 'image_path' and 'question' columns in {path}")

        # Absolute image paths and images column
        df["full_path"] = df["image_path"].apply(lambda rel: os.path.join(image_root, str(rel)))
        df["images"] = df["full_path"].apply(lambda p: [p])

        # Build segmented messages from question/answer
        qa = df.apply(_extract_qa, axis=1)
        df["messages"] = qa.apply(
            lambda t: [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": t[0], "index": None},
                     {"type": "image", "index": 0, "text": None},
                 ]},
                {"role": "assistant",
                 "content": [
                     {"type": "text", "text": t[1], "index": None},
                 ]},
            ]
        )

        # uid: prefer 'id', then 'image_path', else index
        if "id" in df.columns:
            df["uid"] = df["id"].astype(str)
        elif "image_path" in df.columns:
            df["uid"] = df["image_path"].astype(str)
        else:
            df["uid"] = df.index.astype(str)

        # Debug slice (head-based)
        df = _apply_debug(df, split)

        if len(df) == 0:
            raise ValueError(f"[mimic_cxr_vqa:{split}] 0 rows after parsing/sampling {path}")

        # Keep only required columns
        df = df[["uid", "messages", "images"]]

        # Pandas -> HF Dataset
        out[split] = Dataset.from_pandas(df, preserve_index=False)

    return DatasetDict(out)




@register_dataset("nih_bbox")
def load_nih_bbox(
    num_proc: int = 1,
    batch_size: int = 2048,
    image_root: str = "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/images",
    split_csv: dict | None = None,   # {"train": "...csv", "validation": "...csv", "test": "...csv"}
    eager_image_io: bool = False,    # False => paths in `images`; True => bytes in `_bytes` (if available)
    check_files: bool = False,       # slow on network FS
    debug_limit: int | None = None,    # early trim per split
    verbose: bool = False,
    original_size: int | None = 1024,  # Original image size (if None, no rescaling) | Default original_size = 1024
    target_size: int | None = 768,    # Target size for bbox rescaling (if None, no rescaling)
) -> "DatasetDict":
    """
    NIH ChestX-ray14 BBox -> HF Dataset (alignment-style schema, no image decoding).
    Stable output columns: {'messages', 'images', '_bytes'}.

    Notes:
    - Converts CSV bbox columns to integers and produces answers as:
        "<Label>: [ [x1, y1, x2, y2] ]"
    - If both original_size and target_size are provided, bounding boxes are rescaled
    - If either is None, bounding boxes remain in original coordinates
    """
    import os
    import numpy as np
    from datasets import load_dataset, DatasetDict

    # Single deterministic instruction (extendable to multiple prompts if needed)
    PROMPTS_NIH = np.array([
        "You are a medical imaging expert. Detect any visible diseases or abnormalities in this X-ray image. "
        "For each detected finding, provide: the disease name/names, and bounding box coordinates as: "
        "[ [x_min, y_min, x_max, y_max] ] in pixel values."
    ], dtype=object)

    # Defaults to your given CSVs if not provided
    if split_csv is None:
        split_csv = {
            "train": "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/BBox_List_2017.csv",
            "validation": "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/BBox_List_2017_val.csv",
            "test": "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/BBox_List_2017_val1.csv",
        }

    # Determine if we need to rescale bounding boxes
    rescale_boxes = (original_size is not None and target_size is not None)
    scale_factor = target_size / original_size if rescale_boxes else 1.0

    def _read_split(split: str):
        csv_path = split_csv.get(split)
        if not csv_path:
            raise FileNotFoundError(f"[nih_bbox:{split}] missing csv path")
        ds = load_dataset("csv", data_files=csv_path, split="train")

        # Early dev trim
        if debug_limit and debug_limit > 0 and len(ds) > debug_limit:
            ds = ds.select(range(debug_limit))

        # Normalize/rename bbox columns if needed
        # Expected final names: x, y, w, h
        rename_map = {}
        cols = set(ds.column_names)
        if {"Bbox [x", "y", "w", "h]"}.issubset(cols):
            rename_map.update({
                "Bbox [x": "x",
                "y": "y",
                "w": "w",
                "h]": "h",
            })
        if rename_map:
            ds = ds.rename_columns(rename_map)
            cols = set(ds.column_names)

        # Ensure required columns exist
        required = {"Image Index"}
        if split in ("validation", "test"):
            required |= {"x", "y", "w", "h", "Finding Label"}
        missing = required - cols
        if missing:
            raise ValueError(f"[nih_bbox:{split}] missing required columns: {missing}")

        # Build absolute image paths (no FS touch)
        def _mk_paths(batch):
            names = batch["Image Index"]
            fulls = [os.path.normpath(os.path.join(image_root, str(n).lstrip("/"))) for n in names]
            return {"full_path": fulls, "_bytes": [[] for _ in fulls]}

        ds = ds.map(
            _mk_paths,
            batched=True, batch_size=batch_size, num_proc=num_proc,
            desc=f"[nih_bbox:{split}] prefix root" if verbose else None
        )

        # Create uid if absent
        if "uid" not in ds.column_names:
            ds = ds.map(
                lambda b: {"uid": [str(x) for x in b["full_path"]]},
                batched=True, batch_size=batch_size, num_proc=num_proc
            )

        # QUESTION: deterministic prompt per row (uses absolute row indices)
        def _mk_questions_bbox(batch, indices):
            caps = [PROMPTS_NIH[i % len(PROMPTS_NIH)] for i in indices]
            return {"caption": caps}

        ds = ds.map(
            _mk_questions_bbox,
            with_indices=True,
            batched=True, batch_size=batch_size, num_proc=num_proc,
            desc=f"[nih_bbox:{split}] build questions" if verbose else None,
        )

        # ANSWER: for train split we may not have boxes/labels.
        # For val/test we do; convert (x, y, w, h) -> (x1, y1, x2, y2) and optionally rescale.
        def _mk_answers(batch):
            n = len(batch["uid"])
            if all(k in batch for k in ("x", "y", "w", "h", "Finding Label")):
                xs = batch["x"]; ys = batch["y"]; ws = batch["w"]; hs = batch["h"]
                labels = batch["Finding Label"]
                out = []
                
                for i in range(n):
                    try:
                        x = int(float(xs[i])); y = int(float(ys[i]))
                        w = int(float(ws[i])); h = int(float(hs[i]))
                        
                        # Convert to x1, y1, x2, y2
                        x1, y1 = x, y
                        x2, y2 = x + w, y + h
                        
                        # Apply rescaling if needed
                        if rescale_boxes:
                            x1 = int(x1 * scale_factor)
                            y1 = int(y1 * scale_factor)
                            x2 = int(x2 * scale_factor)
                            y2 = int(y2 * scale_factor)
                            
                            # Clip to valid range
                            x1 = max(0, min(x1, target_size))
                            y1 = max(0, min(y1, target_size))
                            x2 = max(0, min(x2, target_size))
                            y2 = max(0, min(y2, target_size))
                        
                        lab = str(labels[i])
                        out.append(f"{lab}: [ [{x1}, {y1}, {x2}, {y2}] ]")
                    except Exception:
                        out.append("")  # fallback empty
                return {"answer": out}
            else:
                # No bbox/label columns -> empty targets (e.g., train CSV)
                return {"answer": [""] * n}

        ds = ds.map(
            _mk_answers,
            batched=True, batch_size=batch_size, num_proc=num_proc,
            desc=f"[nih_bbox:{split}] build answers" if verbose else None
        )

        # Keep only minimal columns prior to formatting
        keep = {"full_path", "caption", "uid", "_bytes", "answer"}
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)

        # Build messages/images (NO I/O) — self-contained formatter using *caption* as the user text.
        def _format_batch_indexed_no_io(samples):
            """
            Build chat messages with an image placeholder in the first user turn.
            DO NOT load images — 'images' is a list[str] of paths.

            Expected input columns (after standardization):
            - full_path (str): path to image
            - caption   (str): instruction/prompt
            - uid       (optional): unused here, but preserved earlier
            """
            out_messages, out_images = [], []
            n = len(samples["full_path"])
            captions = samples.get("caption", [""] * n)

            for i in range(n):
                img_path = samples["full_path"][i]
                question = captions[i] if i < len(captions) else ""
                imgs = [img_path] if isinstance(img_path, str) and img_path else []

                user_content = (
                    [{"type": "text", "text": question, "index": None},
                     {"type": "image", "text": None, "index": 0}]
                    if imgs else
                    [{"type": "text", "text": question, "index": None}]
                )

                msg = [
                    {"role": "user", "content": user_content},
                    # assistant text will be filled with ground truth answer below
                    {"role": "assistant", "content": [{"type": "text", "text": "", "index": None}]},
                ]

                out_messages.append(msg)
                out_images.append(imgs)

            return {"messages": out_messages, "images": out_images}
        
        ds = ds.map(
            _format_batch_indexed_no_io,
            batched=True, batch_size=batch_size, num_proc=num_proc,
            desc=f"[nih_bbox:{split}] format (no I/O)" if verbose else None,
        )

        # Inject answer into assistant turn
        def _inject_ans(batch):
            msgs, ans = batch["messages"], batch["answer"]
            for i in range(len(msgs)):
                if len(msgs[i]) >= 2:
                    msgs[i][1]["content"] = [{"type": "text", "text": (ans[i] or ""), "index": None}]
            return {"messages": msgs}

        ds = ds.map(
            _inject_ans,
            batched=True, batch_size=batch_size, num_proc=num_proc,
            desc=f"[nih_bbox:{split}] inject answers" if verbose else None
        )

        # Decide where images go
        if eager_image_io:
            # keep `images` as empty bytes placeholder; paths mirrored in `_bytes`
            def _bytes_images(batch):
                return {
                    "images": [[] for _ in batch["full_path"]],
                    "_bytes": [[p] if p else [] for p in batch["full_path"]],
                }
            ds = ds.map(_bytes_images, batched=True, batch_size=batch_size, num_proc=num_proc)
        else:
            ds = ds.map(
                lambda b: {"images": [[p] if p else [] for p in b["full_path"]]},
                batched=True, batch_size=batch_size, num_proc=num_proc
            )

        # Final prune
        keep_cols = {"messages", "images", "_bytes"}
        drop2 = [c for c in ds.column_names if c not in keep_cols]
        if drop2:
            ds = ds.remove_columns(drop2)

        # Optional path existence check
        if check_files and not eager_image_io:
            def _has_file(ex):
                imgs = ex["images"]
                return bool(imgs and imgs[0] and os.path.exists(imgs[0][0]))
            ds = ds.filter(_has_file, desc=f"[nih_bbox:{split}] drop missing files")

        return ds

    return DatasetDict({
        "train": _read_split("train"),
        "validation": _read_split("validation"),
        "test": _read_split("test"),
    })
    
    
    
    
    
    
    
    
    
    
    
    
@register_dataset("deeplesion_bbox")
def load_deeplesion_bbox(
    num_proc: int = 1,
    batch_size: int = 2048,
    image_root: str = "./Medmo_Dataset_1/Medmo_Dataset/DeepLesion/Images_png",
    split_csv: dict | None = None,   # {"train": "...csv", "validation": "...csv", "test": "...csv"}
    eager_image_io: bool = False,    # False -> paths in `images`; True -> bytes in `images` (if available)
    check_files: bool = False,       # potentially slow on network FS
    debug_limit: int | None = None,  # early trim per split
    verbose: bool = False,
    original_size: int | None = 512,  # Original image size (if None, no rescaling) | Default = 512
    target_size: int | None = 768,    # Target size for bbox rescaling (if None, no rescaling)
    cache_to: str | None = None,       # Optional cache directory
) -> DatasetDict:
    """
    DeepLesion CSV -> HF Dataset (alignment-style schema, no image decoding).
    Stable output: {'messages', 'images'}.

    Notes:
    - Expects columns: 'File_name', 'Bounding_boxes' (as "x_min,y_min,x_max,y_max").
    - Builds absolute image paths from the DeepLesion naming convention:
        Correct_Image_Path = image_root / "_".join(File_name.split("_")[:3]) / File_name.split("_")[3]
    - If both original_size and target_size are provided, bounding boxes are rescaled
    - If either is None, bounding boxes remain in original coordinates
    - Produces answers as "[ x1, y1, x2, y2 ]"
    """
    import os
    import numpy as np
    from datasets import load_dataset, DatasetDict

    # Check if cached version exists
    if cache_to and os.path.exists(cache_to):
        try:
            print(f"[deeplesion_bbox] Loading from cache: {cache_to}")
            return DatasetDict.load_from_disk(cache_to)
        except Exception as e:
            print(f"[deeplesion_bbox] Cache load failed: {e}, rebuilding...")

    PROMPTS = np.array([
        "Identify the lesion in the image and provide its location using [ x_min, y_min, x_max, y_max ].",
        "Mark the region of abnormality in this scan in [ x_min, y_min, x_max, y_max ] format.",
        "Locate the visible lesion and respond with bounding box [ x_min, y_min, x_max, y_max ].",
        "Find and return the bounding box for the abnormal region in [ x_min, y_min, x_max, y_max ] format.",
        "Examine the scan and indicate the region of interest using [ x_min, y_min, x_max, y_max ]."
    ], dtype=object)

    # Default CSVs
    if split_csv is None:
        split_csv = {
            "train": "./Medmo_Dataset_1/Medmo_Dataset/DeepLesion/DL_info_train.csv",
            "validation": "./Medmo_Dataset_1/Medmo_Dataset/DeepLesion/DL_info_val.csv",
        }

    # Determine if we need to rescale bounding boxes
    rescale_boxes = (original_size is not None and target_size is not None)
    scale_factor = target_size / original_size if rescale_boxes else 1.0

    def _read_split(split: str):
        csv_path = split_csv.get(split)
        if not csv_path:
            raise FileNotFoundError(f"[deeplesion_bbox:{split}] missing csv path")
        ds = load_dataset("csv", data_files=csv_path, split="train")

        # Early dev trim
        if debug_limit and debug_limit > 0 and len(ds) > debug_limit:
            ds = ds.select(range(debug_limit))

        # Sanity on required columns
        cols = set(ds.column_names)
        required = {"File_name", "Bounding_boxes"}
        missing = required - cols
        if missing:
            raise ValueError(f"[deeplesion_bbox:{split}] missing required columns: {missing}")

        # SINGLE-PASS PROCESSING: build everything at once
        def _process_all(batch):
            files = batch["File_name"]
            boxes = batch["Bounding_boxes"]
            n = len(files)
            
            # Build paths
            fulls = []
            for fn in files:
                s = str(fn).split("_")
                if len(s) < 4:
                    fulls.append(os.path.normpath(os.path.join(image_root, str(fn))))
                else:
                    dir_part = "_".join(s[:3])
                    leaf = s[3]
                    fulls.append(os.path.normpath(os.path.join(image_root, dir_part, leaf)))
            
            # Generate random prompts
            prompts = PROMPTS[np.random.randint(0, len(PROMPTS), size=n)]
            
            # Parse and optionally rescale bounding boxes
            answers = []
            for b in boxes:
                try:
                    x_min, y_min, x_max, y_max = [float(t) for t in str(b).split(",")]
                    
                    # Apply rescaling if needed
                    if rescale_boxes:
                        x_min = int(x_min * scale_factor)
                        y_min = int(y_min * scale_factor)
                        x_max = int(x_max * scale_factor)
                        y_max = int(y_max * scale_factor)
                        
                        # Clip to valid range
                        x_min = max(0, min(x_min, target_size))
                        y_min = max(0, min(y_min, target_size))
                        x_max = max(0, min(x_max, target_size))
                        y_max = max(0, min(y_max, target_size))
                    else:
                        # No rescaling, just convert to int
                        x_min = int(x_min)
                        y_min = int(y_min)
                        x_max = int(x_max)
                        y_max = int(y_max)
                    
                    answers.append(f"[ {x_min}, {y_min}, {x_max}, {y_max} ]")
                except Exception:
                    answers.append("")
            
            # Build messages and images
            messages_out = []
            images_out = []
            
            for path, prompt, answer in zip(fulls, prompts, answers):
                images_out.append([path])
                
                messages_out.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt, "index": None},
                            {"type": "image", "text": None, "index": 0},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer, "index": None},
                        ],
                    },
                ])
            
            return {"messages": messages_out, "images": images_out}

        map_kwargs = dict(
            function=_process_all,
            batched=True,
            batch_size=batch_size,
            remove_columns=ds.column_names,  # Remove all original columns
            desc=f"[deeplesion_bbox:{split}] process & format" if verbose else None,
        )
        map_num_proc = num_proc if num_proc and num_proc > 1 else None
        try:
            ds = ds.map(num_proc=map_num_proc, **map_kwargs)
        except RuntimeError as e:
            if "subprocesses has abruptly died" in str(e) and map_num_proc is not None:
                print(f"[deeplesion_bbox:{split}] map failed with num_proc={map_num_proc}; retrying single-process.")
                ds = ds.map(num_proc=None, **map_kwargs)
            else:
                raise

        # Optional existence check
        if check_files:
            def _has_file(ex):
                imgs = ex["images"]
                return bool(imgs and imgs[0] and os.path.exists(imgs[0]))
            ds = ds.filter(_has_file, desc=f"[deeplesion_bbox:{split}] drop missing files" if verbose else None)

        return ds

    # Process all splits
    out = {}
    for split in ["train", "validation"]:
        if split in split_csv:
            out[split] = _read_split(split)
    
    d = DatasetDict(out)

    # Cache if requested
    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        print(f"[deeplesion_bbox] Saving to cache: {cache_to}")
        d.save_to_disk(cache_to)

    return d




@register_dataset("grazpedwri_dx_bbox")
def load_wrist_fracture_bbox_supervisely(
    num_proc: int = 1,
    batch_size: int = 1024,
    ann_dirs: dict | None = None,
    image_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Wrist/images",
    original_size: int | None = None,  # If both None, DO NOT rescale boxes
    target_size: int | None = None,    # If both None, DO NOT rescale boxes
    debug_limit: int | None = None,
    cache_to: str | None = None,
) -> DatasetDict:
    from datasets import Dataset, DatasetDict
    from pathlib import Path
    import json, os, random

    # Cache load (if available)
    if cache_to and os.path.exists(cache_to):
        try:
            print(f"[grazpedwri_dx_bbox] Loading from cache: {cache_to}")
            return DatasetDict.load_from_disk(cache_to)
        except Exception as e:
            print(f"[grazpedwri_dx_bbox] Cache load failed: {e}, rebuilding...")

    PROMPTS = [
    "Examine this wrist X-ray and identify all fractures or abnormalities. For each finding, provide bounding box coordinates as: Finding: [ [x_min, y_min, x_max, y_max] ]. Multiple findings: Finding: [ [x1, y1, x2, y2], [x3, y3, x4, y4], ... ].",
    "Analyze this wrist radiograph. Detect all fractures, dislocations, or abnormal regions. Return coordinates: Finding: [ [x_min, y_min, x_max, y_max] ] for single detection, or Finding: [ [x1, y1, x2, y2], ... ] for multiple findings.",
    "Identify all pathological findings in this wrist X-ray. Report each as: Finding: [ [x_min, y_min, x_max, y_max] ]. For multiple findings: Finding: [ [x1, y1, x2, y2], [x3, y3, x4, y4], ... ]. Use precise pixel coordinates.",
    "Detect fractures and abnormalities in this wrist image. Provide bounding boxes: Finding: [ [x_min, y_min, x_max, y_max] ]. List all findings as: Finding: [ [x1, y1, x2, y2], ... ].",
    "Locate all fractures or injuries visible in this wrist radiograph. Output format: Finding: [ [x_min, y_min, x_max, y_max] ] for each detected region. Multiple findings: Finding: [ [x1, y1, x2, y2], [x3, y3, x4, y4], ... ].",
]

    if ann_dirs is None:
        ann_dirs = {
            "train": "./Medmo_Dataset_1/Medmo_Dataset/Wrist/folder_structure/supervisely/wrist/ann",
            "validation": "./Medmo_Dataset_1/Medmo_Dataset/Wrist/ann_test",
            "test": "./Medmo_Dataset_1/Medmo_Dataset/Wrist/ann_val",
        }
    image_dir = Path(image_dir)

    # Rescaling flag: only if BOTH sizes provided
    rescale_boxes = (original_size is not None and target_size is not None)
    scale_factor = (target_size / float(original_size)) if rescale_boxes else 1.0

    def _collect(split: str) -> Dataset:
        ann_dir = Path(ann_dirs[split])
        anns = sorted(ann_dir.glob("*.json"))
        if debug_limit:
            anns = anns[:debug_limit]
        rows = []
        for ap in anns:
            name = ap.stem
            rows.append({"uid": name, "ann_path": str(ap), "img_path": str(image_dir / f"{name}.png")})
        return Dataset.from_list(rows)

    def _load_and_process(batch):
        msgs, imgs = [], []
        for uid, ap, ip in zip(batch["uid"], batch["ann_path"], batch["img_path"]):
            with open(ap, "r") as f:
                data = json.load(f)

            # Extract bounding boxes from annotation
            boxes = []
            for obj in data.get("objects", []):
                pts = (obj.get("points") or {}).get("exterior") or []
                if len(pts) >= 2:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                    if rescale_boxes:
                        x1 = int(round(x1 * scale_factor))
                        y1 = int(round(y1 * scale_factor))
                        x2 = int(round(x2 * scale_factor))
                        y2 = int(round(y2 * scale_factor))
                        # Optional clip to target_size canvas
                        x1 = max(0, min(x1, target_size))
                        y1 = max(0, min(y1, target_size))
                        x2 = max(0, min(x2, target_size))
                        y2 = max(0, min(y2, target_size))
                    else:
                        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

                    boxes.append((obj.get("classTitle", "region"), x1, y1, x2, y2))

            # Format answer
            if boxes:
                ans = "\n".join([f"{cls}: [ {x1}, {y1}, {x2}, {y2} ]"
                                 for cls, x1, y1, x2, y2 in boxes])
            else:
                ans = "No annotated region found."

            q = PROMPTS[random.randrange(len(PROMPTS))]
            msg = [
                {"role": "user", "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": q, "index": None}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": ans, "index": None}
                ]},
            ]
            msgs.append(msg)
            imgs.append([ip])  # PATH ONLY (no PIL)

        return {"messages": msgs, "images": imgs}

    def _read_split(split: str) -> Dataset:
        ds = _collect(split)
        ds = ds.map(
            _load_and_process,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"[wrist:{split}] load+process bboxes (paths only; no resize if sizes=None)"
        )
        drop = [c for c in ds.column_names if c not in {"messages", "images"}]
        if drop:
            ds = ds.remove_columns(drop)
        return ds

    d = DatasetDict({s: _read_split(s) for s in ["train", "validation", "test"]})

    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        print(f"[grazpedwri_dx_bbox] Saving to cache: {cache_to}")
        d.save_to_disk(cache_to)

    return d



@register_dataset("grazpedwri_dx_bbox_resize")
def load_wrist_fracture_bbox_supervisely(
    num_proc: int = 1,
    batch_size: int = 1024,
    ann_dirs: dict | None = None,
    original_image_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Wrist/images",
    resized_image_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Wrist/images_resize_768",
    target_size: int = 768,  # Target size for resized images
    debug_limit: int | None = None,
    cache_to: str | None = None,
) -> DatasetDict:
    from datasets import Dataset, DatasetDict
    from pathlib import Path
    from PIL import Image
    import json, os, random

    # Define pathological findings (diseases to detect)
    PATHOLOGICAL_CLASSES = {
        "fracture", "boneanomaly", "bonelesion", "foreignbody", 
        "periostealreaction", "softtissue", "metal", "pronatorsign"
    }
    
    # Define auxiliary annotations (non-pathological, to include in output)
    AUXILIARY_CLASSES = {"text"}
    
    # Define annotations to completely exclude from output
    EXCLUDE_CLASSES = {"axis"}

    # Cache load (if available)
    if cache_to and os.path.exists(cache_to):
        try:
            print(f"[grazpedwri_dx_bbox] Loading from cache: {cache_to}")
            return DatasetDict.load_from_disk(cache_to)
        except Exception as e:
            print(f"[grazpedwri_dx_bbox] Cache load failed: {e}, rebuilding...")

    PROMPTS = [
        "Examine this wrist X-ray and identify all fractures or abnormalities. For each finding, provide bounding box coordinates as: Finding: [ [x_min, y_min, x_max, y_max] ]. Multiple findings: Finding: [ [x1, y1, x2, y2], [x3, y3, x4, y4], ... ].",
        "Analyze this wrist radiograph. Detect all fractures, dislocations, or abnormal regions. Return coordinates: Finding: [ [x_min, y_min, x_max, y_max] ] for single detection, or Finding: [ [x1, y1, x2, y2], ... ] for multiple findings.",
        "Identify all pathological findings in this wrist X-ray. Report each as: Finding: [ [x_min, y_min, x_max, y_max] ]. For multiple findings: Finding: [ [x1, y1, x2, y2], [x3, y3, x4, y4], ... ]. Use precise pixel coordinates.",
        "Detect fractures and abnormalities in this wrist image. Provide bounding boxes: Finding: [ [x_min, y_min, x_max, y_max] ]. List all findings as: Finding: [ [x1, y1, x2, y2], ... ].",
        "Locate all fractures or injuries visible in this wrist radiograph. Output format: Finding: [ [x_min, y_min, x_max, y_max] ] for each detected region. Multiple findings: Finding: [ [x1, y1, x2, y2], [x3, y3, x4, y4], ... ].",
    ]

    if ann_dirs is None:
        ann_dirs = {
            "train": "./Medmo_Dataset_1/Medmo_Dataset/Wrist/folder_structure/supervisely/wrist/ann",
            "validation": "./Medmo_Dataset_1/Medmo_Dataset/Wrist/ann_test",
            "test": "./Medmo_Dataset_1/Medmo_Dataset/Wrist/ann_val",
        }
    
    original_image_dir = Path(original_image_dir)
    resized_image_dir = Path(resized_image_dir)

    def _collect(split: str) -> Dataset:
        ann_dir = Path(ann_dirs[split])
        anns = sorted(ann_dir.glob("*.json"))
        if debug_limit:
            anns = anns[:debug_limit]
        rows = []
        for ap in anns:
            name = ap.stem
            rows.append({
                "uid": name,
                "ann_path": str(ap),
                "original_img_path": str(original_image_dir / f"{name}.png"),
                "resized_img_path": str(resized_image_dir / f"{name}.png")
            })
        return Dataset.from_list(rows)

    def _load_and_process(batch):
        msgs, imgs = [], []
        for uid, ap, orig_ip, resize_ip in zip(
            batch["uid"], 
            batch["ann_path"], 
            batch["original_img_path"],
            batch["resized_img_path"]
        ):
            # Read original image to get dimensions
            with Image.open(orig_ip) as img:
                orig_width, orig_height = img.size
            
            # Calculate scale factors for this specific image
            scale_x = target_size / float(orig_width)
            scale_y = target_size / float(orig_height)
            
            # Load annotation
            with open(ap, "r") as f:
                data = json.load(f)

            # Separate pathological findings from auxiliary annotations
            pathological_boxes = []
            auxiliary_boxes = []
            
            for obj in data.get("objects", []):
                class_title = obj.get("classTitle", "region")
                class_title_lower = class_title.lower()
                
                # Skip excluded classes entirely
                if class_title_lower in EXCLUDE_CLASSES:
                    continue
                
                pts = (obj.get("points") or {}).get("exterior") or []
                if len(pts) >= 2:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                    # Resize bounding boxes based on per-image scale factors
                    x1_resized = int(round(x1 * scale_x))
                    y1_resized = int(round(y1 * scale_y))
                    x2_resized = int(round(x2 * scale_x))
                    y2_resized = int(round(y2 * scale_y))
                    
                    # Clip to target canvas size
                    x1_resized = max(0, min(x1_resized, target_size))
                    y1_resized = max(0, min(y1_resized, target_size))
                    x2_resized = max(0, min(x2_resized, target_size))
                    y2_resized = max(0, min(y2_resized, target_size))

                    box_data = (class_title, x1_resized, y1_resized, x2_resized, y2_resized)
                    
                    # Classify as pathological or auxiliary
                    if class_title_lower in PATHOLOGICAL_CLASSES:
                        pathological_boxes.append(box_data)
                    elif class_title_lower in AUXILIARY_CLASSES:
                        auxiliary_boxes.append(box_data)
                    else:
                        # Unknown class - treat as pathological to be safe
                        pathological_boxes.append(box_data)

            # Format answer based on whether pathological findings exist
            answer_parts = []
            
            if pathological_boxes:
                # Disease found - list all pathological findings
                for cls, x1, y1, x2, y2 in pathological_boxes:
                    answer_parts.append(f"{cls}: [ {x1}, {y1}, {x2}, {y2} ]")
            else:
                # No disease found
                answer_parts.append("No disease found")
            
            # Always append auxiliary annotations if they exist
            for cls, x1, y1, x2, y2 in auxiliary_boxes:
                answer_parts.append(f"{cls}: [ {x1}, {y1}, {x2}, {y2} ]")
            
            # Join all parts
            if answer_parts:
                ans = "\n".join(answer_parts)
            else:
                ans = "No annotated region found."

            q = PROMPTS[random.randrange(len(PROMPTS))]
            msg = [
                {"role": "user", "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": q, "index": None}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": ans, "index": None}
                ]},
            ]
            msgs.append(msg)
            imgs.append([resize_ip])  # Use RESIZED image path

        return {"messages": msgs, "images": imgs}

    def _read_split(split: str) -> Dataset:
        ds = _collect(split)
        ds = ds.map(
            _load_and_process,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"[wrist:{split}] load+process bboxes with disease classification"
        )
        drop = [c for c in ds.column_names if c not in {"messages", "images"}]
        if drop:
            ds = ds.remove_columns(drop)
        return ds

    # d = DatasetDict({s: _read_split(s) for s in ["train", "validation", "test"]})
    d = DatasetDict({s: _read_split(s) for s in ["train"]})

    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        print(f"[grazpedwri_dx_bbox] Saving to cache: {cache_to}")
        d.save_to_disk(cache_to)

    return d






@register_dataset("bacteria_bbox_resize")
def load_bacteria_bbox_yolo(
    num_proc: int = 1,
    batch_size: int = 1024,
    original_image_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Cell_Data/preproceed/Bacteria/images",
    resized_image_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Cell_Data/preproceed/Bacteria/images_768",
    label_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Cell_Data/preproceed/Bacteria/labels",
    target_size: int = 768,           # Target size for resized images
    max_boxes: int = 100,              # Max boxes threshold
    debug_limit: int | None = None,
    cache_to: str | None = None,
) -> DatasetDict:
    """
    YOLO txt labels -> VLM alignment samples with per-image bbox resizing.
    - Loads JPG paths + matching .txt labels (YOLO x_c,y_c,w,h normalized).
    - Reads original image dimensions to calculate per-image scale factors.
    - Resizes bounding boxes according to each image's original dimensions.
    - Returns resized image paths and rescaled [ x_min, y_min, x_max, y_max ] answers.
    - Skips any sample with > max_boxes boxes.
    """
    from datasets import Dataset, DatasetDict
    from pathlib import Path
    from PIL import Image
    import os

    # Cache load (if available)
    if cache_to and os.path.exists(cache_to):
        try:
            print(f"[bacteria_bbox] Loading from cache: {cache_to}")
            return DatasetDict.load_from_disk(cache_to)
        except Exception as e:
            print(f"[bacteria_bbox] Cache load failed: {e}, rebuilding...")

    PROMPTS = [
        "You are an expert in microscopy image analysis. Examine this microscopy image carefully and identify all visible bacteria cells. For each bacterial cell detected, provide its bounding box coordinates in the format: Bacteria: [ [x_min, y_min, x_max, y_max] ]. If multiple cells are present, list all bounding boxes as: Bacteria: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Ensure all coordinates are in pixel values.",
        "Analyze this microscopy image to detect and localize all bacterial cells present. Return the bounding box coordinates for each detected bacterium using the format: Bacteria: [ [x_min, y_min, x_max, y_max] ]. When multiple bacteria are visible, provide all bounding boxes in a single list: Bacteria: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Use precise pixel coordinates.",
        "Your task is to identify and localize every bacterial cell visible in this microscopy image. For each bacterium, determine its bounding box and report the coordinates as: Bacteria: [ [x_min, y_min, x_max, y_max] ]. If the image contains multiple bacterial cells, enumerate all their bounding boxes: Bacteria: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Provide accurate pixel-level coordinates.",
        "Perform bacterial cell detection on this microscopy image. Locate each bacterium and specify its bounding box using [ x_min, y_min, x_max, y_max ] format. Output format: Bacteria: [ [x_min, y_min, x_max, y_max] ] for single detection, or Bacteria: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ] for multiple detections. Report all visible bacteria with precise coordinates.",
        "As a microscopy analysis system, detect all bacterial cells in this image. For each identified bacterium, provide bounding box coordinates in pixel values using the format: Bacteria: [ [x_min, y_min, x_max, y_max] ]. When multiple bacteria are present, list all detections: Bacteria: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Be thorough and accurate in your detections.",
    ]

    original_image_dir = Path(original_image_dir)
    resized_image_dir = Path(resized_image_dir)
    label_dir = Path(label_dir)

    def _gather() -> Dataset:
        rows = []
        for root, _, files in os.walk(original_image_dir):
            for fn in sorted(files):
                if not fn.lower().endswith(".jpg"):
                    continue
                img_path = Path(root) / fn
                rel = img_path.relative_to(original_image_dir)
                
                # Corresponding resized image and label paths
                resized_img_path = resized_image_dir / rel
                lbl_path = label_dir / rel.with_suffix(".txt")
                
                if lbl_path.exists() and resized_img_path.exists():
                    rows.append({
                        "uid": rel.as_posix(),
                        "original_img_path": str(img_path),
                        "resized_img_path": str(resized_img_path),
                        "lbl_path": str(lbl_path)
                    })
        if debug_limit:
            rows = rows[:debug_limit]
        return Dataset.from_list(rows)

    # Count boxes
    def _count_boxes(batch):
        counts = []
        for lp in batch["lbl_path"]:
            n = 0
            try:
                with open(lp, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            n += 1
            except Exception:
                n = 0
            counts.append(n)
        return {"box_count": counts}

    def _process_boxes(batch):
        out_msgs, out_imgs = [], []
        for uid, orig_ip, resize_ip, lp in zip(
            batch["uid"],
            batch["original_img_path"],
            batch["resized_img_path"],
            batch["lbl_path"]
        ):
            # Read original image to get dimensions
            with Image.open(orig_ip) as img:
                orig_width, orig_height = img.size

            # Calculate per-image scale factors
            scale_x = target_size / float(orig_width)
            scale_y = target_size / float(orig_height)

            # Parse YOLO labels -> pixel boxes (in original image space)
            boxes = []
            with open(lp, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, x_c, y_c, w, h = parts
                    x_c, y_c, w, h = map(float, (x_c, y_c, w, h))

                    # Convert normalized YOLO coords to pixel coords in original image
                    x1_orig = (x_c - w / 2) * orig_width
                    y1_orig = (y_c - h / 2) * orig_height
                    x2_orig = (x_c + w / 2) * orig_width
                    y2_orig = (y_c + h / 2) * orig_height

                    # Scale to resized image dimensions
                    x1 = int(round(x1_orig * scale_x))
                    y1 = int(round(y1_orig * scale_y))
                    x2 = int(round(x2_orig * scale_x))
                    y2 = int(round(y2_orig * scale_y))

                    # Clip to target canvas size
                    x1 = max(0, min(x1, target_size))
                    y1 = max(0, min(y1, target_size))
                    x2 = max(0, min(x2, target_size))
                    y2 = max(0, min(y2, target_size))

                    boxes.append([x1, y1, x2, y2])

            # Format answer
            if boxes:
                box_str = ", ".join([f"[{x1}, {y1}, {x2}, {y2}]" for x1, y1, x2, y2 in boxes])
                ans = f"Bacteria: [ {box_str} ]"
            else:
                ans = "No bacteria found."

            q = PROMPTS[hash(uid) % len(PROMPTS)]
            msgs = [
                {"role": "user", "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": q, "index": None}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": ans, "index": None}
                ]},
            ]
            out_msgs.append(msgs)
            out_imgs.append([resize_ip])  # Use RESIZED image path

        return {"messages": out_msgs, "images": out_imgs}

    def _read_split() -> Dataset:
        ds = _gather()

        # Count boxes then filter out samples with > max_boxes
        ds = ds.map(
            _count_boxes,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="[bacteria] count boxes"
        )
        ds = ds.filter(
            lambda n: n <= max_boxes,
            input_columns=["box_count"],
            desc=f"[bacteria] filter (> {max_boxes} boxes)"
        )
        ds = ds.remove_columns(["box_count"])

        # Build messages/images for remaining samples with per-image bbox resize
        ds = ds.map(
            _process_boxes,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="[bacteria] process boxes with per-image resize"
        )
        drop = [c for c in ds.column_names if c not in {"messages", "images"}]
        if drop:
            ds = ds.remove_columns(drop)
        return ds

    d = _read_split()
    d_train, d_val, d_test = _split_dataset_tail_disjoint(d, min(200, len(d)))
    dataset_dict = DatasetDict({
        "train": d_train,
        "validation": d_val,
        "test": d_test,
    })

    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        print(f"[bacteria_bbox] Saving to cache: {cache_to}")
        dataset_dict.save_to_disk(cache_to)

    return dataset_dict


@register_dataset("bacteria_bbox")
def load_bacteria_bbox_yolo(
    num_proc: int = 1,
    batch_size: int = 1024,
    image_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Cell_Data/preproceed/Bacteria/images",
    label_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Cell_Data/preproceed/Bacteria/labels",
    max_boxes: int = 100,              # <-- make threshold configurable
    original_size: int | None = None,  # Original image size (if None, no rescaling)
    target_size: int | None = None,    # Target size for bbox rescaling (if None, no rescaling)
    debug_limit: int | None = None,
    cache_to: str | None = None,
) -> DatasetDict:
    """
    YOLO txt labels -> VLM alignment samples.
    - Loads JPG paths + matching .txt labels (YOLO x_c,y_c,w,h normalized).
    - Skips any sample with > max_boxes boxes.
    - If both original_size and target_size are provided, rescales boxes accordingly.
    - Returns image paths (not PIL images) and [ x_min, y_min, x_max, y_max ] answers in messages.
    """
    

    # Cache load (if available)
    if cache_to and os.path.exists(cache_to):
        try:
            print(f"[bacteria_bbox] Loading from cache: {cache_to}")
            return DatasetDict.load_from_disk(cache_to)
        except Exception as e:
            print(f"[bacteria_bbox] Cache load failed: {e}, rebuilding...")

    PROMPTS = [
        "You are an expert in microscopy image analysis. Examine this microscopy image carefully and identify all visible bacteria cells. For each bacterial cell detected, provide its bounding box coordinates in the format: Bacteria: [ [x_min, y_min, x_max, y_max] ]. If multiple cells are present, list all bounding boxes as: Bacteria: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Ensure all coordinates are in pixel values.",
        "Analyze this microscopy image to detect and localize all bacterial cells present. Return the bounding box coordinates for each detected bacterium using the format: Bacteria: [ [x_min, y_min, x_max, y_max] ]. When multiple bacteria are visible, provide all bounding boxes in a single list: Bacteria: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Use precise pixel coordinates.",
        "Your task is to identify and localize every bacterial cell visible in this microscopy image. For each bacterium, determine its bounding box and report the coordinates as: Bacteria: [ [x_min, y_min, x_max, y_max] ]. If the image contains multiple bacterial cells, enumerate all their bounding boxes: Bacteria: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Provide accurate pixel-level coordinates.",
        "Perform bacterial cell detection on this microscopy image. Locate each bacterium and specify its bounding box using [ x_min, y_min, x_max, y_max ] format. Output format: Bacteria: [ [x_min, y_min, x_max, y_max] ] for single detection, or Bacteria: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ] for multiple detections. Report all visible bacteria with precise coordinates.",
        "As a microscopy analysis system, detect all bacterial cells in this image. For each identified bacterium, provide bounding box coordinates in pixel values using the format: Bacteria: [ [x_min, y_min, x_max, y_max] ]. When multiple bacteria are present, list all detections: Bacteria: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Be thorough and accurate in your detections.",
    ]

    image_dir, label_dir = Path(image_dir), Path(label_dir)

    # Rescaling setup
    rescale_boxes = (original_size is not None and target_size is not None)
    scale_factor = target_size / original_size if rescale_boxes else 1.0

    def _gather() -> Dataset:
        rows = []
        for root, _, files in os.walk(image_dir):
            for fn in sorted(files):
                if not fn.lower().endswith(".jpg"):
                    continue
                img_path = Path(root) / fn
                rel = img_path.relative_to(image_dir)
                lbl_path = label_dir / rel.with_suffix(".txt")
                if lbl_path.exists():
                    rows.append({"uid": rel.as_posix(), "img_path": str(img_path), "lbl_path": str(lbl_path)})
        if debug_limit:
            rows = rows[:debug_limit]
        return Dataset.from_list(rows)

    # Count boxes
    def _count_boxes(batch):
        counts = []
        for lp in batch["lbl_path"]:
            n = 0
            try:
                with open(lp, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            n += 1
            except Exception:
                n = 0
            counts.append(n)
        return {"box_count": counts}

    def _process_boxes(batch):
        out_msgs, out_imgs = [], []
        for uid, ip, lp in zip(batch["uid"], batch["img_path"], batch["lbl_path"]):
            # image size
            with Image.open(ip) as img:
                w0, h0 = img.size

            sx = sy = scale_factor if rescale_boxes else 1.0

            # Parse YOLO labels -> pixel boxes
            boxes = []
            with open(lp, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, x_c, y_c, w, h = parts
                    x_c, y_c, w, h = map(float, (x_c, y_c, w, h))

                    x1o = (x_c - w / 2) * w0
                    y1o = (y_c - h / 2) * h0
                    x2o = (x_c + w / 2) * w0
                    y2o = (y_c + h / 2) * h0

                    x1 = int(round(x1o * sx)); y1 = int(round(y1o * sy))
                    x2 = int(round(x2o * sx)); y2 = int(round(y2o * sy))

                    if rescale_boxes:
                        x1 = max(0, min(x1, target_size))
                        y1 = max(0, min(y1, target_size))
                        x2 = max(0, min(x2, target_size))
                        y2 = max(0, min(y2, target_size))

                    boxes.append([x1, y1, x2, y2])

            if boxes:
                box_str = ", ".join([f"[{x1}, {y1}, {x2}, {y2}]" for x1, y1, x2, y2 in boxes])
                ans = f"Bacteria: [ {box_str} ]"
            else:
                ans = "No bacteria found."

            q = PROMPTS[hash(uid) % len(PROMPTS)]
            msgs = [
                {"role": "user", "content": [{"type": "image", "index": 0},
                                             {"type": "text", "text": q, "index": None}]},
                {"role": "assistant", "content": [{"type": "text", "text": ans, "index": None}]},
            ]
            out_msgs.append(msgs)
            out_imgs.append([ip])  # path list

        return {"messages": out_msgs, "images": out_imgs}

    def _read_split() -> Dataset:
        ds = _gather()

        # Count boxes then filter out samples with > max_boxes
        ds = ds.map(_count_boxes, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc="[bacteria] count boxes")
        ds = ds.filter(lambda n: n <= max_boxes, input_columns=["box_count"],
                       desc=f"[bacteria] filter (> {max_boxes} boxes)")
        ds = ds.remove_columns(["box_count"])

        # Build messages/images for remaining samples
        ds = ds.map(
            _process_boxes,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="[bacteria] process boxes"
        )
        drop = [c for c in ds.column_names if c not in {"messages", "images"}]
        if drop:
            ds = ds.remove_columns(drop)
        return ds

    d = _read_split()
    d_train, d_val, d_test = _split_dataset_tail_disjoint(d, min(200, len(d)))
    dataset_dict = DatasetDict({
        "train": d_train,
        "validation": d_val,
        "test": d_test,
    })

    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        print(f"[bacteria_bbox] Saving to cache: {cache_to}")
        dataset_dict.save_to_disk(cache_to)

    return dataset_dict






@register_dataset("ctc_bbox")
def load_ctc_bbox_yolo(
    num_proc: int = 1,
    batch_size: int = 1024,
    base_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Cell_Data/preproceed/CTCDataset",
    max_boxes: int = 100,              # Skip samples with more than this many boxes
    original_size: int | None = None,  # Original image size (if None, no rescaling)
    target_size: int | None = None,    # Target size for bbox rescaling (if None, no rescaling)
    debug_limit: int | None = None,
    cache_to: str | None = None,
) -> DatasetDict:
    """
    CTC YOLO labels -> VLM alignment samples:
      - Scans */images/{train,val} with matching */labels/{train,val}
      - If both original_size and target_size are provided, rescales bboxes accordingly
      - If either is None, keeps original bounding box coordinates
      - Skips any sample with > max_boxes boxes
      - Returns image paths (not PIL images) and [ [x1, y1, x2, y2], ... ] answers in messages
    """
    from datasets import Dataset, DatasetDict
    from pathlib import Path
    from PIL import Image
    import os

    # Check if cached version exists
    if cache_to and os.path.exists(cache_to):
        try:
            print(f"[ctc_bbox] Loading from cache: {cache_to}")
            return DatasetDict.load_from_disk(cache_to)
        except Exception as e:
            print(f"[ctc_bbox] Cache load failed: {e}, rebuilding...")

    PROMPTS = [
        "You are an expert in cell microscopy image analysis. Examine this microscopy image and identify all visible cells. For each cell detected, provide its bounding box coordinates in the format: Cell: [ [x_min, y_min, x_max, y_max] ]. If multiple cells are present, list all bounding boxes as: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Use precise pixel coordinates.",
        "Analyze this cell microscopy image to detect and localize all cells present. Return the bounding box coordinates for each detected cell using the format: Cell: [ [x_min, y_min, x_max, y_max] ]. When multiple cells are visible, provide all bounding boxes in a single list: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Provide accurate pixel-level coordinates.",
        "Your task is to identify and localize every cell visible in this microscopy image. For each cell, determine its bounding box and report the coordinates as: Cell: [ [x_min, y_min, x_max, y_max] ]. If the image contains multiple cells, enumerate all their bounding boxes: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Be thorough and precise in your detections.",
        "Perform cell detection on this microscopy image. Locate each cell and specify its bounding box using [ x_min, y_min, x_max, y_max ] format. Output format: Cell: [ [x_min, y_min, x_max, y_max] ] for single detection, or Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ] for multiple detections. Report all visible cells with precise pixel coordinates.",
        "As a microscopy analysis system, detect all cells in this image. For each identified cell, provide bounding box coordinates in pixel values using the format: Cell: [ [x_min, y_min, x_max, y_max] ]. When multiple cells are present, list all detections: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Ensure comprehensive detection of all visible cells.",
    ]

    base = Path(base_dir)

    # Determine if we need to rescale bounding boxes
    rescale_boxes = (original_size is not None and target_size is not None)
    scale_factor = target_size / original_size if rescale_boxes else 1.0

    def _gather() -> Dataset:
        rows = []
        for ds_name in sorted(os.listdir(base)):
            ds_path = base / ds_name
            if not ds_path.is_dir():
                continue
            img_root, lbl_root = ds_path / "images", ds_path / "labels"
            for split in ("train", "val"):
                img_dir, lbl_dir = img_root / split, lbl_root / split
                if not img_dir.is_dir():
                    continue
                for fn in sorted(os.listdir(img_dir)):
                    if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    img_path = img_dir / fn
                    lbl_path = lbl_dir / (Path(fn).stem + ".txt")
                    if lbl_path.exists():
                        rows.append({
                            "uid": f"{ds_name}/{split}/{fn}",
                            "img_path": str(img_path),
                            "lbl_path": str(lbl_path),
                        })
        if debug_limit:
            rows = rows[:debug_limit]
        return Dataset.from_list(rows)

    # Count boxes then attach count
    def _count_boxes(batch):
        counts = []
        for lp in batch["lbl_path"]:
            n = 0
            try:
                with open(lp, "r") as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) == 5:
                            n += 1
            except Exception:
                n = 0
            counts.append(n)
        return {"box_count": counts}

    def _process_boxes(batch):
        out_msgs, out_imgs = [], []
        for uid, ip, lp in zip(batch["uid"], batch["img_path"], batch["lbl_path"]):
            # Get original image dimensions without fully loading
            with Image.open(ip) as img:
                w0, h0 = img.size

            # Determine scale factors
            sx, sy = (scale_factor, scale_factor) if rescale_boxes else (1.0, 1.0)

            boxes = []
            with open(lp, "r") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) != 5:
                        continue
                    # YOLO: class cx cy w h (normalized)
                    _, cx, cy, w, h = p
                    cx, cy, w, h = map(float, (cx, cy, w, h))
                    # denormalize to original pixels
                    x1o = (cx - w/2) * w0
                    y1o = (cy - h/2) * h0
                    x2o = (cx + w/2) * w0
                    y2o = (cy + h/2) * h0
                    # scale for target size (or keep original if no rescaling)
                    x1 = int(round(x1o * sx)); y1 = int(round(y1o * sy))
                    x2 = int(round(x2o * sx)); y2 = int(round(y2o * sy))

                    if rescale_boxes:
                        x1 = max(0, min(x1, target_size))
                        y1 = max(0, min(y1, target_size))
                        x2 = max(0, min(x2, target_size))
                        y2 = max(0, min(y2, target_size))

                    boxes.append([x1, y1, x2, y2])

            if boxes:
                box_str = ", ".join(f"[{x1}, {y1}, {x2}, {y2}]" for x1, y1, x2, y2 in boxes)
                ans = f"Cell: [ {box_str} ]"
            else:
                ans = "No cells found."

            q = PROMPTS[hash(uid) % len(PROMPTS)]
            msgs = [
                {"role": "user", "content": [{"type": "image", "index": 0},
                                             {"type": "text", "text": q, "index": None}]},
                {"role": "assistant", "content": [{"type": "text", "text": ans, "index": None}]},
            ]
            out_msgs.append(msgs)
            out_imgs.append([ip])  # Store path instead of PIL image

        return {"messages": out_msgs, "images": out_imgs}

    def _read() -> Dataset:
        ds = _gather()

        # Count boxes and filter out samples with > max_boxes
        ds = ds.map(_count_boxes, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc="[ctc] count boxes")
        ds = ds.filter(lambda n: n <= max_boxes, input_columns=["box_count"],
                       desc=f"[ctc] filter (> {max_boxes} boxes skipped)")
        ds = ds.remove_columns(["box_count"])

        ds = ds.map(
            _process_boxes,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="[ctc] process boxes"
        )
        drop = [c for c in ds.column_names if c not in {"messages", "images"}]
        if drop:
            ds = ds.remove_columns(drop)
        return ds

    d = _read()
    d_train, d_val, d_test = _split_dataset_tail_disjoint(d, min(200, len(d)))
    dataset_dict = DatasetDict({
        "train": d_train,
        "validation": d_val,
        "test": d_test,
    })

    # Cache if requested
    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        print(f"[ctc_bbox] Saving to cache: {cache_to}")
        dataset_dict.save_to_disk(cache_to)

    return dataset_dict



@register_dataset("ctc_bbox_resize")
def load_ctc_bbox_yolo(
    num_proc: int = 1,
    batch_size: int = 1024,
    base_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Cell_Data/preproceed/CTCDataset",
    target_size: int = 768,           # Target size for resized images
    max_boxes: int = 30,              # Skip samples with more than this many boxes
    debug_limit: int | None = None,
    cache_to: str | None = None,
) -> DatasetDict:
    """
    CTC YOLO labels -> VLM alignment samples with per-image bbox resizing:
      - Scans */images/{train,val} and */images_resize_768/{train,val}
      - Reads original image dimensions to calculate per-image scale factors
      - Rescales bboxes according to each image's original dimensions
      - Returns resized image paths and rescaled [ [x1, y1, x2, y2], ... ] answers
      - Skips any sample with > max_boxes boxes
    """
    from datasets import Dataset, DatasetDict
    from pathlib import Path
    from PIL import Image
    import os

    # Check if cached version exists
    if cache_to and os.path.exists(cache_to):
        try:
            print(f"[ctc_bbox] Loading from cache: {cache_to}")
            return DatasetDict.load_from_disk(cache_to)
        except Exception as e:
            print(f"[ctc_bbox] Cache load failed: {e}, rebuilding...")

    PROMPTS = [
        "You are an expert in cell microscopy image analysis. Examine this microscopy image and identify all visible cells. For each cell detected, provide its bounding box coordinates in the format: Cell: [ [x_min, y_min, x_max, y_max] ]. If multiple cells are present, list all bounding boxes as: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Use precise pixel coordinates.",
        "Analyze this cell microscopy image to detect and localize all cells present. Return the bounding box coordinates for each detected cell using the format: Cell: [ [x_min, y_min, x_max, y_max] ]. When multiple cells are visible, provide all bounding boxes in a single list: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Provide accurate pixel-level coordinates.",
        "Your task is to identify and localize every cell visible in this microscopy image. For each cell, determine its bounding box and report the coordinates as: Cell: [ [x_min, y_min, x_max, y_max] ]. If the image contains multiple cells, enumerate all their bounding boxes: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Be thorough and precise in your detections.",
        "Perform cell detection on this microscopy image. Locate each cell and specify its bounding box using [ x_min, y_min, x_max, y_max ] format. Output format: Cell: [ [x_min, y_min, x_max, y_max] ] for single detection, or Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ] for multiple detections. Report all visible cells with precise pixel coordinates.",
        "As a microscopy analysis system, detect all cells in this image. For each identified cell, provide bounding box coordinates in pixel values using the format: Cell: [ [x_min, y_min, x_max, y_max] ]. When multiple cells are present, list all detections: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Ensure comprehensive detection of all visible cells.",
    ]

    base = Path(base_dir)

    def _gather() -> Dataset:
        rows = []
        for ds_name in sorted(os.listdir(base)):
            ds_path = base / ds_name
            if not ds_path.is_dir():
                continue
            
            # Original and resized image directories
            orig_img_root = ds_path / "images"
            resize_img_root = ds_path / "images_resize_768"
            lbl_root = ds_path / "labels"
            
            for split in ("train", "val"):
                orig_img_dir = orig_img_root / split
                resize_img_dir = resize_img_root / split
                lbl_dir = lbl_root / split
                
                if not orig_img_dir.is_dir() or not resize_img_dir.is_dir():
                    continue
                    
                for fn in sorted(os.listdir(orig_img_dir)):
                    if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    
                    orig_img_path = orig_img_dir / fn
                    resize_img_path = resize_img_dir / fn
                    lbl_path = lbl_dir / (Path(fn).stem + ".txt")
                    
                    if lbl_path.exists() and resize_img_path.exists():
                        rows.append({
                            "uid": f"{ds_name}/{split}/{fn}",
                            "original_img_path": str(orig_img_path),
                            "resized_img_path": str(resize_img_path),
                            "lbl_path": str(lbl_path),
                        })
        if debug_limit:
            rows = rows[:debug_limit]
        return Dataset.from_list(rows)

    # Count boxes then attach count
    def _count_boxes(batch):
        counts = []
        for lp in batch["lbl_path"]:
            n = 0
            try:
                with open(lp, "r") as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) == 5:
                            n += 1
            except Exception:
                n = 0
            counts.append(n)
        return {"box_count": counts}

    def _process_boxes(batch):
        out_msgs, out_imgs = [], []
        for uid, orig_ip, resize_ip, lp in zip(
            batch["uid"],
            batch["original_img_path"],
            batch["resized_img_path"],
            batch["lbl_path"]
        ):
            # Read original image to get dimensions
            with Image.open(orig_ip) as img:
                orig_width, orig_height = img.size

            # Calculate per-image scale factors
            scale_x = target_size / float(orig_width)
            scale_y = target_size / float(orig_height)

            # Parse YOLO labels and resize boxes
            boxes = []
            with open(lp, "r") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) != 5:
                        continue
                    # YOLO: class cx cy w h (normalized)
                    _, cx, cy, w, h = p
                    cx, cy, w, h = map(float, (cx, cy, w, h))
                    
                    # Denormalize to original pixel coordinates
                    x1_orig = (cx - w/2) * orig_width
                    y1_orig = (cy - h/2) * orig_height
                    x2_orig = (cx + w/2) * orig_width
                    y2_orig = (cy + h/2) * orig_height
                    
                    # Scale to resized image dimensions
                    x1 = int(round(x1_orig * scale_x))
                    y1 = int(round(y1_orig * scale_y))
                    x2 = int(round(x2_orig * scale_x))
                    y2 = int(round(y2_orig * scale_y))

                    # Clip to target canvas size
                    x1 = max(0, min(x1, target_size))
                    y1 = max(0, min(y1, target_size))
                    x2 = max(0, min(x2, target_size))
                    y2 = max(0, min(y2, target_size))

                    boxes.append([x1, y1, x2, y2])

            # Format answer
            if boxes:
                box_str = ", ".join(f"[{x1}, {y1}, {x2}, {y2}]" for x1, y1, x2, y2 in boxes)
                ans = f"Cell: [ {box_str} ]"
            else:
                ans = "No cells found."

            q = PROMPTS[hash(uid) % len(PROMPTS)]
            msgs = [
                {"role": "user", "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": q, "index": None}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": ans, "index": None}
                ]},
            ]
            out_msgs.append(msgs)
            out_imgs.append([resize_ip])  # Use RESIZED image path

        return {"messages": out_msgs, "images": out_imgs}

    def _read() -> Dataset:
        ds = _gather()

        # Count boxes and filter out samples with > max_boxes
        ds = ds.map(
            _count_boxes,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="[ctc] count boxes"
        )
        ds = ds.filter(
            lambda n: n <= max_boxes,
            input_columns=["box_count"],
            desc=f"[ctc] filter (> {max_boxes} boxes skipped)"
        )
        ds = ds.remove_columns(["box_count"])

        # Process boxes with per-image resize
        ds = ds.map(
            _process_boxes,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="[ctc] process boxes with per-image resize"
        )
        drop = [c for c in ds.column_names if c not in {"messages", "images"}]
        if drop:
            ds = ds.remove_columns(drop)
        return ds

    d = _read()
    d_train, d_val, d_test = _split_dataset_tail_disjoint(d, min(200, len(d)))
    dataset_dict = DatasetDict({
        "train": d_train,
        "validation": d_val,
        "test": d_test,
    })

    # Cache if requested
    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        print(f"[ctc_bbox] Saving to cache: {cache_to}")
        dataset_dict.save_to_disk(cache_to)

    return dataset_dict


@register_dataset("deepcell_bbox")
def load_deepcell_bbox_yolo(
    num_proc: int = 1,
    batch_size: int = 1024,
    base_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Cell_Data/preproceed/DEEPCELL",
    original_size: int | None = None,  # Original image size (if None, no rescaling)
    target_size: int | None = None,    # Target size for bbox rescaling (if None, no rescaling)
    max_boxes: int = 30,               # Skip samples with more than this many boxes
    split: str = "train",              # "train"->00, "validation"->01, "test"->02
    debug_limit: int | None = None,
    cache_to: str | None = None,
    # If True and filtering removes everything, keep the K smallest-box samples instead of returning empty
    auto_relax_if_empty: bool = True,
    auto_relax_keep_k: int = 200,
) -> DatasetDict:
    """
    DeepCell YOLO labels -> VLM alignment samples:
      - Scans images/<split_id> with matching labels/<split_id>
      - If both original_size and target_size are provided, rescales bboxes accordingly
      - If either is None, keeps original bounding box coordinates
      - Skips samples with more than `max_boxes` boxes (configurable)
      - Returns image paths (not PIL images) and [ [x1, y1, x2, y2], ... ] answers in messages
      - Prevents empty datasets (StopIteration) by optionally relaxing the filter if needed.
    """
    from datasets import Dataset, DatasetDict
    from pathlib import Path
    from PIL import Image
    import os

    if cache_to and os.path.exists(cache_to):
        try:
            print(f"[deepcell_bbox] Loading from cache: {cache_to}")
            return DatasetDict.load_from_disk(cache_to)
        except Exception as e:
            print(f"[deepcell_bbox] Cache load failed: {e}, rebuilding...")

    SPLIT_MAP = {"train": "00", "validation": "01", "test": "02"}
    assert split in SPLIT_MAP, f"split must be one of {list(SPLIT_MAP)}"
    sid = SPLIT_MAP[split]

    PROMPTS = [
        "You are an expert in cell microscopy image analysis. Examine this microscopy image carefully and identify all visible cells. For each cell detected, provide its bounding box coordinates in the format: Cell: [ [x_min, y_min, x_max, y_max] ]. If multiple cells are present, list all bounding boxes as: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Use precise pixel coordinates for accurate localization.",
        
        "Analyze this cell microscopy image to detect and localize all cells present. Return the bounding box coordinates for each detected cell using the format: Cell: [ [x_min, y_min, x_max, y_max] ]. When multiple cells are visible, provide all bounding boxes in a single list: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Ensure comprehensive detection of all visible cells with accurate pixel-level coordinates.",
        
        "Your task is to identify and localize every cell visible in this microscopy image. For each cell, determine its bounding box and report the coordinates as: Cell: [ [x_min, y_min, x_max, y_max] ]. If the image contains multiple cells, enumerate all their bounding boxes: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Be thorough and precise in your detections, covering all cells present in the field of view.",
        
        "Perform comprehensive cell detection on this microscopy image. Locate each cell and specify its bounding box using [ x_min, y_min, x_max, y_max ] format. Output format: Cell: [ [x_min, y_min, x_max, y_max] ] for single detection, or Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ] for multiple detections. Report all visible cells with precise pixel coordinates, ensuring no cells are missed.",
        
        "As a microscopy analysis system, detect and localize all cells in this image. For each identified cell, provide bounding box coordinates in pixel values using the format: Cell: [ [x_min, y_min, x_max, y_max] ]. When multiple cells are present, list all detections: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Ensure systematic and complete detection across the entire image with accurate spatial localization.",
        
        "Examine this microscopy image and perform detailed cell detection. Identify the location of each cell by providing bounding box coordinates in the format: Cell: [ [x_min, y_min, x_max, y_max] ]. For images containing multiple cells, report all detections as: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Use precise pixel-level measurements and ensure all visible cells are captured in your analysis.",
        
        "Analyze this cell microscopy image systematically. Detect all cells present and report their spatial locations using bounding box coordinates in the format: Cell: [ [x_min, y_min, x_max, y_max] ]. When encountering multiple cells, provide a comprehensive list: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Maintain high precision in coordinate specification and ensure complete coverage of all cellular structures.",
        
        "You are tasked with cell detection in this microscopy image. Carefully examine the image and identify all cells, providing their bounding box coordinates as: Cell: [ [x_min, y_min, x_max, y_max] ]. For samples with multiple cells, list all bounding boxes sequentially: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Focus on accuracy and completeness, ensuring every visible cell is detected and properly localized with precise pixel coordinates.",
    ]

    base = Path(base_dir)
    img_dir = base / "images" / sid
    lbl_dir = base / "labels" / sid

    rescale_boxes = (original_size is not None and target_size is not None)
    scale_factor = target_size / original_size if rescale_boxes else 1.0

    def _gather() -> Dataset:
        rows = []
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            ip = img_dir / fn
            lp = lbl_dir / (Path(fn).stem + ".txt")
            if lp.exists():
                rows.append({"uid": f"{split}/{fn}", "img_path": str(ip), "lbl_path": str(lp)})
        if debug_limit:
            rows = rows[:debug_limit]
        return Dataset.from_list(rows)

    def _count_boxes(batch):
        counts = []
        for lp in batch["lbl_path"]:
            n = 0
            try:
                with open(lp, "r") as f:
                    for line in f:
                        if len(line.strip().split()) == 5:
                            n += 1
            except Exception:
                n = 0
            counts.append(n)
        return {"box_count": counts}

    def _process_boxes(batch):
        out_msgs, out_imgs = [], []
        for uid, ip, lp in zip(batch["uid"], batch["img_path"], batch["lbl_path"]):
            with Image.open(ip) as img:
                w0, h0 = img.size
            sx, sy = (scale_factor, scale_factor) if rescale_boxes else (1.0, 1.0)

            boxes = []
            with open(lp, "r") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) != 5:
                        continue
                    _, cx, cy, w, h = p
                    cx, cy, w, h = map(float, (cx, cy, w, h))
                    x1o = (cx - w/2) * w0
                    y1o = (cy - h/2) * h0
                    x2o = (cx + w/2) * w0
                    y2o = (cy + h/2) * h0
                    x1 = int(round(x1o * sx)); y1 = int(round(y1o * sy))
                    x2 = int(round(x2o * sx)); y2 = int(round(y2o * sy))
                    if rescale_boxes:
                        x1 = max(0, min(x1, target_size))
                        y1 = max(0, min(y1, target_size))
                        x2 = max(0, min(x2, target_size))
                        y2 = max(0, min(y2, target_size))
                    boxes.append([x1, y1, x2, y2])

            if boxes:
                box_str = ", ".join(f"[{x1}, {y1}, {x2}, {y2}]" for x1, y1, x2, y2 in boxes)
                ans = f"Cell: [ {box_str} ]"
            else:
                ans = "No cells found."

            q = PROMPTS[hash(uid) % len(PROMPTS)]
            msgs = [
                {"role": "user", "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": q, "index": None},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": ans, "index": None}]},
            ]
            out_msgs.append(msgs)
            out_imgs.append([ip])  # path only

        return {"messages": out_msgs, "images": out_imgs}

    def _read() -> Dataset:
        ds = _gather()
        if len(ds) == 0:
            raise ValueError(f"[deepcell_bbox] No image/label pairs found under {img_dir} / {lbl_dir}")

        # Count boxes (batched for speed)
        ds = ds.map(_count_boxes, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[deepcell:{split}] count boxes")

        # Primary filter: keep <= max_boxes
        kept = ds.filter(lambda n: n <= max_boxes, input_columns=["box_count"],
                         desc=f"[deepcell:{split}] filter (<= {max_boxes} boxes)")

        if len(kept) == 0:
            if not auto_relax_if_empty:
                raise ValueError(
                    f"[deepcell_bbox] 0 samples after filtering with max_boxes={max_boxes}. "
                    f"Increase `max_boxes` or check labels in {lbl_dir}."
                )
            # Relax: keep the K smallest box-count samples to avoid StopIteration
            print(f"[deepcell_bbox] WARNING: 0 samples after filtering (max_boxes={max_boxes}). "
                  f"Keeping {min(auto_relax_keep_k, len(ds))} samples with the fewest boxes to avoid an empty dataset.")
            # Sort by box_count ascending and take top-K
            df_sorted = ds.sort("box_count")
            kept = df_sorted.select(range(min(auto_relax_keep_k, len(df_sorted))))

        # Drop helper column
        kept = kept.remove_columns(["box_count"])

        # Build messages/images
        kept = kept.map(
            _process_boxes,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"[deepcell:{split}] process boxes"
        )
        drop = [c for c in kept.column_names if c not in {"messages", "images"}]
        if drop:
            kept = kept.remove_columns(drop)
        return kept

    d = _read()

    dataset_dict = DatasetDict({
        "train": d if split == "train" else Dataset.from_dict({"messages": [], "images": []}),
        "validation": d if split == "validation" else Dataset.from_dict({"messages": [], "images": []}),
        "test": Dataset.from_dict({"messages": [], "images": []}),  # keep key present if your pipeline expects it
    })

    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        print(f"[deepcell_bbox] Saving to cache: {cache_to}")
        dataset_dict.save_to_disk(cache_to)

    return dataset_dict



@register_dataset("deepcell_bbox_resize")
def load_deepcell_bbox_yolo(
    num_proc: int = 1,
    batch_size: int = 1024,
    base_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Cell_Data/preproceed/DEEPCELL",
    target_size: int = 768,            # Target size for resized images
    max_boxes: int = 100,               # Skip samples with more than this many boxes
    split: str = "train",              # "train"->00, "validation"->01, "test"->02
    debug_limit: int | None = None,
    cache_to: str | None = None,
    # If True and filtering removes everything, keep the K smallest-box samples instead of returning empty
    auto_relax_if_empty: bool = True,
    auto_relax_keep_k: int = 200,
) -> DatasetDict:
    """
    DeepCell YOLO labels -> VLM alignment samples with per-image bbox resizing:
      - Scans images/<split_id> and images_resize_768/<split_id>
      - Reads original image dimensions to calculate per-image scale factors
      - Rescales bboxes according to each image's original dimensions
      - Returns resized image paths and rescaled [ [x1, y1, x2, y2], ... ] answers
      - Skips samples with more than `max_boxes` boxes (configurable)
      - Prevents empty datasets (StopIteration) by optionally relaxing the filter if needed.
    """
    from datasets import Dataset, DatasetDict
    from pathlib import Path
    from PIL import Image
    import os

    if cache_to and os.path.exists(cache_to):
        try:
            print(f"[deepcell_bbox] Loading from cache: {cache_to}")
            return DatasetDict.load_from_disk(cache_to)
        except Exception as e:
            print(f"[deepcell_bbox] Cache load failed: {e}, rebuilding...")

    SPLIT_MAP = {"train": "00", "validation": "01", "test": "02"}
    assert split in SPLIT_MAP, f"split must be one of {list(SPLIT_MAP)}"
    sid = SPLIT_MAP[split]

    PROMPTS = [
        "You are an expert in cell microscopy image analysis. Examine this microscopy image carefully and identify all visible cells. For each cell detected, provide its bounding box coordinates in the format: Cell: [ [x_min, y_min, x_max, y_max] ]. If multiple cells are present, list all bounding boxes as: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Use precise pixel coordinates for accurate localization.",
        
        "Analyze this cell microscopy image to detect and localize all cells present. Return the bounding box coordinates for each detected cell using the format: Cell: [ [x_min, y_min, x_max, y_max] ]. When multiple cells are visible, provide all bounding boxes in a single list: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Ensure comprehensive detection of all visible cells with accurate pixel-level coordinates.",
        
        "Your task is to identify and localize every cell visible in this microscopy image. For each cell, determine its bounding box and report the coordinates as: Cell: [ [x_min, y_min, x_max, y_max] ]. If the image contains multiple cells, enumerate all their bounding boxes: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Be thorough and precise in your detections, covering all cells present in the field of view.",
        
        "Perform comprehensive cell detection on this microscopy image. Locate each cell and specify its bounding box using [ x_min, y_min, x_max, y_max ] format. Output format: Cell: [ [x_min, y_min, x_max, y_max] ] for single detection, or Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ] for multiple detections. Report all visible cells with precise pixel coordinates, ensuring no cells are missed.",
        
        "As a microscopy analysis system, detect and localize all cells in this image. For each identified cell, provide bounding box coordinates in pixel values using the format: Cell: [ [x_min, y_min, x_max, y_max] ]. When multiple cells are present, list all detections: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Ensure systematic and complete detection across the entire image with accurate spatial localization.",
        
        "Examine this microscopy image and perform detailed cell detection. Identify the location of each cell by providing bounding box coordinates in the format: Cell: [ [x_min, y_min, x_max, y_max] ]. For images containing multiple cells, report all detections as: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Use precise pixel-level measurements and ensure all visible cells are captured in your analysis.",
        
        "Analyze this cell microscopy image systematically. Detect all cells present and report their spatial locations using bounding box coordinates in the format: Cell: [ [x_min, y_min, x_max, y_max] ]. When encountering multiple cells, provide a comprehensive list: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Maintain high precision in coordinate specification and ensure complete coverage of all cellular structures.",
        
        "You are tasked with cell detection in this microscopy image. Carefully examine the image and identify all cells, providing their bounding box coordinates as: Cell: [ [x_min, y_min, x_max, y_max] ]. For samples with multiple cells, list all bounding boxes sequentially: Cell: [ [x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ... ]. Focus on accuracy and completeness, ensuring every visible cell is detected and properly localized with precise pixel coordinates.",
    ]

    base = Path(base_dir)
    
    # Original and resized image directories
    orig_img_dir = base / "images" / sid
    resize_img_dir = base / "images_resize_768" / sid
    lbl_dir = base / "labels" / sid

    def _gather() -> Dataset:
        rows = []
        for fn in sorted(os.listdir(orig_img_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            orig_ip = orig_img_dir / fn
            resize_ip = resize_img_dir / fn
            lp = lbl_dir / (Path(fn).stem + ".txt")
            
            if lp.exists() and resize_ip.exists():
                rows.append({
                    "uid": f"{split}/{fn}",
                    "original_img_path": str(orig_ip),
                    "resized_img_path": str(resize_ip),
                    "lbl_path": str(lp)
                })
        if debug_limit:
            rows = rows[:debug_limit]
        return Dataset.from_list(rows)

    def _count_boxes(batch):
        counts = []
        for lp in batch["lbl_path"]:
            n = 0
            try:
                with open(lp, "r") as f:
                    for line in f:
                        if len(line.strip().split()) == 5:
                            n += 1
            except Exception:
                n = 0
            counts.append(n)
        return {"box_count": counts}

    def _process_boxes(batch):
        out_msgs, out_imgs = [], []
        for uid, orig_ip, resize_ip, lp in zip(
            batch["uid"],
            batch["original_img_path"],
            batch["resized_img_path"],
            batch["lbl_path"]
        ):
            # Read original image to get dimensions
            with Image.open(orig_ip) as img:
                orig_width, orig_height = img.size

            # Calculate per-image scale factors
            scale_x = target_size / float(orig_width)
            scale_y = target_size / float(orig_height)

            # Parse YOLO labels and resize boxes
            boxes = []
            with open(lp, "r") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) != 5:
                        continue
                    _, cx, cy, w, h = p
                    cx, cy, w, h = map(float, (cx, cy, w, h))
                    
                    # Denormalize to original pixel coordinates
                    x1_orig = (cx - w/2) * orig_width
                    y1_orig = (cy - h/2) * orig_height
                    x2_orig = (cx + w/2) * orig_width
                    y2_orig = (cy + h/2) * orig_height
                    
                    # Scale to resized image dimensions
                    x1 = int(round(x1_orig * scale_x))
                    y1 = int(round(y1_orig * scale_y))
                    x2 = int(round(x2_orig * scale_x))
                    y2 = int(round(y2_orig * scale_y))
                    
                    # Clip to target canvas size
                    x1 = max(0, min(x1, target_size))
                    y1 = max(0, min(y1, target_size))
                    x2 = max(0, min(x2, target_size))
                    y2 = max(0, min(y2, target_size))
                    
                    boxes.append([x1, y1, x2, y2])

            # Format answer
            if boxes:
                box_str = ", ".join(f"[{x1}, {y1}, {x2}, {y2}]" for x1, y1, x2, y2 in boxes)
                ans = f"Cell: [ {box_str} ]"
            else:
                ans = "No cells found."

            q = PROMPTS[hash(uid) % len(PROMPTS)]
            msgs = [
                {"role": "user", "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": q, "index": None},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": ans, "index": None}]},
            ]
            out_msgs.append(msgs)
            out_imgs.append([resize_ip])  # Use RESIZED image path

        return {"messages": out_msgs, "images": out_imgs}

    def _read() -> Dataset:
        ds = _gather()
        if len(ds) == 0:
            raise ValueError(f"[deepcell_bbox] No image/label pairs found under {orig_img_dir} / {lbl_dir}")

        # Count boxes (batched for speed)
        ds = ds.map(
            _count_boxes,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"[deepcell:{split}] count boxes"
        )

        # Primary filter: keep <= max_boxes
        kept = ds.filter(
            lambda n: n <= max_boxes,
            input_columns=["box_count"],
            desc=f"[deepcell:{split}] filter (<= {max_boxes} boxes)"
        )

        if len(kept) == 0:
            if not auto_relax_if_empty:
                raise ValueError(
                    f"[deepcell_bbox] 0 samples after filtering with max_boxes={max_boxes}. "
                    f"Increase `max_boxes` or check labels in {lbl_dir}."
                )
            # Relax: keep the K smallest box-count samples to avoid StopIteration
            print(f"[deepcell_bbox] WARNING: 0 samples after filtering (max_boxes={max_boxes}). "
                  f"Keeping {min(auto_relax_keep_k, len(ds))} samples with the fewest boxes to avoid an empty dataset.")
            # Sort by box_count ascending and take top-K
            df_sorted = ds.sort("box_count")
            kept = df_sorted.select(range(min(auto_relax_keep_k, len(df_sorted))))

        # Drop helper column
        kept = kept.remove_columns(["box_count"])

        # Build messages/images with per-image bbox resize
        kept = kept.map(
            _process_boxes,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"[deepcell:{split}] process boxes with per-image resize"
        )
        drop = [c for c in kept.column_names if c not in {"messages", "images"}]
        if drop:
            kept = kept.remove_columns(drop)
        return kept

    d = _read()

    dataset_dict = DatasetDict({
        "train": d if split == "train" else Dataset.from_dict({"messages": [], "images": []}),
        "validation": d if split == "validation" else Dataset.from_dict({"messages": [], "images": []}),
        "test": Dataset.from_dict({"messages": [], "images": []}),  # keep key present if your pipeline expects it
    })

    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        print(f"[deepcell_bbox] Saving to cache: {cache_to}")
        dataset_dict.save_to_disk(cache_to)

    return dataset_dict




@register_dataset("pmc_instruct_qa")
def load_pmc_instruct_qa(
    num_proc: int = 1,
    batch_size: int = 2048,
    json_paths: dict | None = None,
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Text-only loader in the SAME structure as prior loaders:
      - returns a DatasetDict with splits
      - builds `messages` (user -> assistant)
      - provides an `images` column (empty list per sample to keep schema stable)
    """
    import json, os, re
    from datasets import Dataset, DatasetDict

    if json_paths is None:
        json_paths = {
            "train":       "./Medmo_Dataset_1/Medmo_Dataset/PMC-LLAMA-Instructions/release.json",
            "validation":  "./Medmo_Dataset_1/Medmo_Dataset/PMC-LLAMA-Instructions/test_data.json",
            "test":        "./Medmo_Dataset_1/Medmo_Dataset/PMC-LLAMA-Instructions/test_data.json",
        }

    def _read_split(split: str) -> Dataset:
        path = json_paths[split]
        with open(path, "r") as f:
            raw = json.load(f)

        if debug_limit:
            raw = raw[:debug_limit]

        ds = Dataset.from_list(raw)

        # parse rationale/answer/options; build messages; keep images=[] (text-only)
        def _format(batch):
            out_msgs, out_imgs = [], []
            for sample in batch["__index_level_0__"] if "__index_level_0__" in batch else range(len(next(iter(batch.values())))):
                instruction = batch["instruction"][sample]
                input_text  = batch["input"][sample]
                output_text = batch["output"][sample]
                source      = batch.get("source", [""] * len(batch["input"]))[sample]

                # 1) rationale & answer
                rationale = ""
                if "###Rationale:" in output_text:
                    # safe split
                    try:
                        rationale = output_text.split("###Rationale:")[1].split("###Answer:")[0].strip()
                    except Exception:
                        rationale = ""

                if "###Answer:" in output_text:
                    answer_part = output_text.split("###Answer:")[1].strip()
                else:
                    answer_part = output_text.strip()

                # 2) selected option (A-F)
                m = re.search(r"OPTION\s+([A-F])", answer_part, re.IGNORECASE)
                opt_char = m.group(1).upper() if m else None

                # 3) question/options parsing
                parts = input_text.split("###Options:")
                question_part = parts[0].replace("###Question:", "").strip()
                options_part  = parts[1].strip() if len(parts) > 1 else ""

                options = {}
                if options_part:
                    for line in options_part.split("\n"):
                        line = line.strip()
                        mm = re.match(r"([A-F])\.\s*(.*)", line)
                        if mm:
                            k, v = mm.group(1), mm.group(2)
                            options[k] = f"{k}. {v}"

                selected_option = options.get(opt_char) if opt_char else None

                # 4) build final user question text
                question = f"##Instruction:\n{instruction}\n\n##Question:\n{question_part}\n\n##Options:\n{options_part}"

                # build messages (no image node)
                user = {"role": "user", "content": [{"type": "text", "text": question, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": (selected_option or answer_part), "index": None}]}
                out_msgs.append([user, asst])

                # text-only: keep images as empty list to match schema
                out_imgs.append([None])

            return {"messages": out_msgs, "images": out_imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[pmc_instruct_qa:{split}] format")

        # keep only messages/images like other loaders
        keep = {"messages", "images"}
        # keep = {"messages"}
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)

        return ds

    out = {sp: _read_split(sp) for sp in ("train", "validation")}
    return DatasetDict(out)




@register_dataset("medquad_qa")
def load_medquad_qa(
    num_proc: int = 1,
    batch_size: int = 2048,
    csv_paths: dict | None = None,
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Text-only MedQuAD loader in the SAME structure as prior loaders:
      - returns DatasetDict with splits
      - builds `messages` (user -> assistant)
      - keeps an `images` column with a single None per sample: [None]
    """
    import pandas as pd
    from datasets import Dataset, DatasetDict

    if csv_paths is None:
        csv_paths = {
            "train":      "./Medmo_Dataset_1/Medmo_Dataset/MedQuAD/medDataset_processed.csv",
            "validation": "./Medmo_Dataset_1/Medmo_Dataset/MedQuAD/medDataset_processed_val.csv",
            "test":       "./Medmo_Dataset_1/Medmo_Dataset/MedQuAD/medDataset_processed_val.csv",
        }

    def _read_split(split: str) -> Dataset:
        df = pd.read_csv(csv_paths[split])
        if debug_limit:
            df = df.head(debug_limit)
        # normalize columns
        if "Question" not in df.columns or "Answer" not in df.columns:
            raise ValueError(f"[medquad_qa] CSV for split '{split}' must contain 'Question' and 'Answer' columns.")
        if "qtype" not in df.columns:
            df["qtype"] = "unknown"

        ds = Dataset.from_pandas(df, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            qs, ans, qt = batch["Question"], batch["Answer"], batch["qtype"]
            n = len(qs)
            for i in range(n):
                qtxt = qs[i]
                atxt = ans[i] if ans[i] is not None else ""
                # messages: user(text) -> assistant(text)
                user = {"role": "user", "content": [{"type": "text", "text": qtxt, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": atxt, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema consistency
            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[medquad_qa:{split}] format")

        # Keep only messages/images
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    out = {sp: _read_split(sp) for sp in ("train", "validation", "test")}
    # out = {sp: _read_split(sp) for sp in ("train", "validation")}
    return DatasetDict(out)



@register_dataset("medqa")
def load_medqa(
    num_proc: int = 1,
    batch_size: int = 2048,
    file_paths: dict | None = None,
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Text-only MedQA loader in the SAME structure as your other loaders:
      - Merges multiple .jsonl shards per split
      - Builds `messages` (user -> assistant)
      - Keeps an `images` column with a single None per sample: [None]
    """
    import json
    from datasets import Dataset, DatasetDict

    if file_paths is None:
        file_paths = {
            "train": [
                "./Medmo_Dataset_1/Medmo_Dataset/MedQA/data_clean/questions/Taiwan/tw_translated_jsonl/en/train-2en.jsonl",
                "./Medmo_Dataset_1/Medmo_Dataset/MedQA/data_clean/questions/US/train.jsonl",
                "./Medmo_Dataset_1/Medmo_Dataset/MedQA/data_clean/questions/US/US_qbank.jsonl",
            ],
            "validation": [
                "./Medmo_Dataset_1/Medmo_Dataset/MedQA/data_clean/questions/Taiwan/tw_translated_jsonl/en/dev-2en.jsonl",
                "./Medmo_Dataset_1/Medmo_Dataset/MedQA/data_clean/questions/US/dev.jsonl",
            ],
            "test": [
                "./Medmo_Dataset_1/Medmo_Dataset/MedQA/data_clean/questions/Taiwan/tw_translated_jsonl/en/test-2en.jsonl",
                "./Medmo_Dataset_1/Medmo_Dataset/MedQA/data_clean/questions/US/test.jsonl",
            ],
        }

    def _read_split(split: str) -> Dataset:
        rows = []
        for fp in file_paths[split]:
            with open(fp, "r", encoding="utf-8") as f:
                for ln, line in enumerate(f):
                    if debug_limit and len(rows) >= debug_limit:
                        break
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    rows.append(d)
            if debug_limit and len(rows) >= debug_limit:
                break

        if not rows:
            return Dataset.from_dict({"messages": [], "images": []})

        ds = Dataset.from_list(rows)

        def _format(batch):
            msgs, imgs = [], []
            q_list, opts_list = batch["question"], batch["options"]
            has_answer_idx = "answer_idx" in batch
            has_answer_val = "answer" in batch
            n = len(q_list)

            for i in range(n):
                # Build user prompt
                question_text = "## Instruction: Choose the correct option based on the question below.\n\n" + q_list[i]
                options = opts_list[i] or {}
                options_text = "\n".join(f"{k}. {v}" for k, v in options.items())
                user_q = f"{question_text}\n\n## Options:\n{options_text}"

                # Resolve answer (prefer answer_idx if present)
                if has_answer_idx and batch["answer_idx"][i] is not None:
                    k = batch["answer_idx"][i]
                    ans = f"{k}. {options.get(k, '')}".strip()
                else:
                    ans_val = (batch["answer"][i] if has_answer_val else "") or ""
                    # try to map value back to key (exact, trimmed)
                    key = next((k for k, v in options.items() if (v or "").strip() == ans_val.strip()), None)
                    ans = f"{key}. {ans_val}".strip() if key else ans_val.strip()

                user = {"role": "user", "content": [{"type": "text", "text": user_q, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": ans, "index": None}]}
                msgs.append([user, asst])

                # text-only schema consistency
                imgs.append([None])

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[medqa:{split}] format")

        # Keep only messages/images
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    out = {sp: _read_split(sp) for sp in ("train", "validation", "test")}
    # out = {sp: _read_split(sp) for sp in ("train", "validation")}
    return DatasetDict(out)



@register_dataset("medical_meadow_medqa")
def load_medical_meadow_medqa(
    num_proc: int = 1,
    batch_size: int = 2048,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/chatdoctor_healthcaremagic",
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Text-only loader (same structure as prior loaders):
      - reads ALL *.parquet from data_dir
      - splits disjointly: tail window reserved for validation/test
      - builds `messages` (user -> assistant)
      - keeps `images` as [None] per sample for schema consistency
    """
    import os, re, ast
    import pandas as pd
    from glob import glob
    from datasets import Dataset, DatasetDict

    # ---------- read & split ----------
    parquet_files = sorted(glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"[medical_meadow_medqa] No parquet files found under {data_dir}")

    df_all = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in parquet_files], ignore_index=True)
    eval_count = min(500, len(df_all))
    df_train, df_val, df_test = _split_frame_tail_disjoint(df_all, eval_count)

    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val   = df_val.head(min(debug_limit, len(df_val)))
        df_test  = df_test.head(min(debug_limit, len(df_test)))

    def _df_to_ds(df: pd.DataFrame, split: str) -> Dataset:
        # normalize/require columns
        for col in ("instruction", "input", "output"):
            if col not in df.columns:
                raise ValueError(f"[medical_meadow_medqa:{split}] missing required column '{col}'")

        ds = Dataset.from_pandas(df, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            instrs = batch["instruction"]
            inputs = batch["input"]
            outs   = batch["output"]
            n = len(inputs)

            for i in range(n):
                instruction = (instrs[i] or "").strip()
                input_text  = (inputs[i] or "")
                output      = (outs[i] or "").strip()

                # parse question + options from input via regex
                m = re.search(r"(.*?)(\{.*\})", input_text, re.DOTALL)
                if m:
                    question_part = (m.group(1) or "").strip()
                    options_str   = (m.group(2) or "").strip().rstrip(",")
                    try:
                        options_dict = ast.literal_eval(options_str)
                        if not isinstance(options_dict, dict):
                            options_dict = {}
                    except Exception:
                        options_dict = {}
                else:
                    question_part = input_text.strip()
                    options_dict  = {}

                # format options
                options_block = "\n".join(f"{k}. {v}" for k, v in options_dict.items())

                # build full question
                user_q = f"##Instruction: {instruction}\n\nQuestion: {question_part}\n\n##Options:\n{options_block}"

                # map answer key -> full text if applicable
                final_answer = output
                if output in options_dict:
                    final_answer = f"{output}. {options_dict[output]}"

                user = {"role": "user", "content": [{"type": "text", "text": user_q, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": final_answer, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[medical_meadow_medqa:{split}] format")

        # keep only the standard columns
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _df_to_ds(df_train, "train"),
        "validation": _df_to_ds(df_val, "validation"),
        "test": _df_to_ds(df_test, "test"),
    })



@register_dataset("alphacare_qa")
def load_alphacare_qa(
    num_proc: int = 1,
    batch_size: int = 2048,
    data_path: str = "./Medmo_Dataset_1/Medmo_Dataset/AlpaCare-MedInstruct-52k/train-00000-of-00001-297892d5d4e8a0ac.parquet",
    train_cut: int | None = 50000,   # first N → train, rest → val/test
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Text-only AlpaCare-MedInstruct-52k loader (same schema as previous loaders):
      - returns DatasetDict with splits
      - builds `messages` (user -> assistant)
      - keeps `images` as [None] per sample for schema consistency
    """
    import pandas as pd
    from datasets import Dataset, DatasetDict

    df = pd.read_parquet(data_path, engine="pyarrow")
    n = len(df)
    cut = min(train_cut, n) if train_cut else n

    df_train = df.iloc[:cut].reset_index(drop=True)
    df_val   = df.iloc[cut:].reset_index(drop=True)
    df_test  = df_val.copy()

    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val   = df_val.head(min(debug_limit, len(df_val)))
        df_test  = df_test.head(min(debug_limit, len(df_test)))

    def _df_to_ds(df_split: pd.DataFrame, split: str) -> Dataset:
        # ensure required columns
        for c in ("instruction", "input", "output"):
            if c not in df_split.columns:
                raise ValueError(f"[alphacare_qa:{split}] missing column '{c}'")
        ds = Dataset.from_pandas(df_split, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            instrs, inputs, outs = batch["instruction"], batch["input"], batch["output"]
            n = len(instrs)
            for i in range(n):
                instruction = (instrs[i] or "").strip()
                inp         = (inputs[i] or "").strip()
                out         = (outs[i] or "").strip()

                # Build question text
                if inp == "<noinput>" or inp == "":
                    qtxt = f"##Instruction: {instruction}"
                else:
                    qtxt = f"##Instruction: {instruction}\n\n##Input: {inp}"

                user = {"role": "user", "content": [{"type": "text", "text": qtxt, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": out, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only, keep schema

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[alphacare_qa:{split}] format")

        # keep only standard columns
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _df_to_ds(df_train, "train"),
        "validation": _df_to_ds(df_val, "validation"),
        "test": _df_to_ds(df_test, "test"),
    })



@register_dataset("chatdoctor_healthcaremagic")
def load_chatdoctor_healthcaremagic(
    num_proc: int = 1,
    batch_size: int = 2048,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/chatdoctor_healthcaremagic",
    val_ratio: float = 0.05,          # 95/5 split (val & test share the same 5%)
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Text-only loader (same structure as prior loaders):
      - reads ALL *.parquet from data_dir
      - split disjointly by ratio: train first part, eval tail split into validation/test
      - builds `messages` (user -> assistant)
      - keeps `images` as [None] per sample for schema consistency
    """
    import os
    import pandas as pd
    from glob import glob
    from datasets import Dataset, DatasetDict

    # -------- read all parquet --------
    files = sorted(glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"[chatdoctor_healthcaremagic] No parquet files in {data_dir}")
    df_all = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in files], ignore_index=True)

    # -------- split 95/5 --------
    df_train, df_val, df_test = _split_frame_ratio_disjoint(df_all, val_ratio)

    # -------- debug truncation --------
    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val   = df_val.head(min(debug_limit, len(df_val)))
        df_test  = df_test.head(min(debug_limit, len(df_test)))

    # -------- to HF Dataset + format --------
    def _df_to_ds(df_split: pd.DataFrame, split: str) -> Dataset:
        # ensure required columns exist
        for col in ("instruction", "input", "output"):
            if col not in df_split.columns:
                df_split[col] = ""

        ds = Dataset.from_pandas(df_split, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            instrs, ctxs, outs = batch["instruction"], batch["input"], batch["output"]
            n = len(instrs)
            for i in range(n):
                instruction = (instrs[i] or "").strip()
                context     = (ctxs[i] or "").strip()
                answer      = (outs[i] or "").strip()

                qtxt = f"##Instruction: {instruction}"
                if context:
                    qtxt += f"\n\n##Context: {context}"

                user = {"role": "user", "content": [{"type": "text", "text": qtxt, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": answer, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[chatdoctor_healthcaremagic:{split}] format")

        # keep only standard columns
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _df_to_ds(df_train, "train"),
        "validation": _df_to_ds(df_val, "validation"),
        "test": _df_to_ds(df_test, "test"),
    })


@register_dataset("chatdoctor_icliniq")
def load_chatdoctor_icliniq(
    num_proc: int = 1,
    batch_size: int = 2048,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/chatdoctor-icliniq",
    answer_key: str = "answer_chatdoctor",   # or: answer_chatgpt, answer_icliniq
    val_ratio: float = 0.05,                  # 95/5 split (val & test share the same tail)
    debug_limit: int | None = None,
    cache_to: str | None = None,
) -> DatasetDict:
    """
    Text-only loader (same structure as previous loaders):
      - reads ALL *.parquet from data_dir
      - split disjointly by ratio: train first part, eval tail split into validation/test
      - builds `messages` (user -> assistant)
      - keeps `images` as [None] per sample for schema consistency
    """
    import os
    import pandas as pd
    from glob import glob
    from datasets import Dataset, DatasetDict

    # Check if cached version exists
    if cache_to and os.path.exists(cache_to):
        try:
            print(f"[chatdoctor_icliniq] Loading from cache: {cache_to}")
            return DatasetDict.load_from_disk(cache_to)
        except Exception as e:
            print(f"[chatdoctor_icliniq] Cache load failed: {e}, rebuilding...")

    assert answer_key in {"answer_chatdoctor", "answer_chatgpt", "answer_icliniq"}, \
        f"Invalid answer_key: {answer_key}"

    files = sorted(glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"[chatdoctor_icliniq] No parquet files in {data_dir}")

    df_all = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in files], ignore_index=True)

    # ensure columns exist
    for col in ["input", "answer_chatdoctor", "answer_chatgpt", "answer_icliniq"]:
        if col not in df_all.columns:
            df_all[col] = ""

    # Calculate split indices correctly
    N = len(df_all)
    if N == 0:
        raise ValueError(f"[chatdoctor_icliniq] No data found in {data_dir}")
    
    # Build disjoint train/validation/test from ratio.
    df_train, df_val, df_test = _split_frame_ratio_disjoint(df_all, val_ratio)

    # Debug truncation
    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val = df_val.head(min(debug_limit, len(df_val)))
        df_test = df_test.head(min(debug_limit, len(df_test)))

    INSTRUCTION = "If you are a doctor, please answer the medical questions based on the patient's description."

    def _df_to_ds(df_split: pd.DataFrame, split: str) -> Dataset:
        if len(df_split) == 0:
            # Return empty dataset with correct schema
            return Dataset.from_dict({"messages": [], "images": []})
        
        ds = Dataset.from_pandas(df_split, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            qs = batch["input"]
            a_ic = batch["answer_icliniq"]
            a_cd = batch["answer_chatdoctor"]
            a_gpt = batch["answer_chatgpt"]
            n = len(qs)

            # Select target column once
            ans_col = {"answer_icliniq": a_ic, "answer_chatdoctor": a_cd, "answer_chatgpt": a_gpt}[answer_key]

            for i in range(n):
                qtxt = (qs[i] or "").strip()
                atxt = (ans_col[i] or "").strip()

                prompt = f"##Instruction: {INSTRUCTION}\n\n##Patient Query: {qtxt}"
                user = {"role": "user", "content": [{"type": "text", "text": prompt, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": atxt, "index": None}]}
                msgs.append([user, asst])

                # Text-only schema
                imgs.append([None])

            return {"messages": msgs, "images": imgs}

        ds = ds.map(
            _format,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"[chatdoctor_icliniq:{split}] format"
        )

        # Keep only standard columns
        keep = {"messages", "images"}
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)
        return ds

    dataset_dict = DatasetDict({
        "train": _df_to_ds(df_train, "train"),
        "validation": _df_to_ds(df_val, "validation"),
        "test": _df_to_ds(df_test, "test"),
    })

    # Cache if requested
    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        print(f"[chatdoctor_icliniq] Saving to cache: {cache_to}")
        dataset_dict.save_to_disk(cache_to)

    return dataset_dict


@register_dataset("chatdoc_medqa_4option")
@register_dataset("chatdoc_medqa_5option")
def load_chatdoc_medqa(
    num_proc: int = 1,
    batch_size: int = 2048,
    base_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor",
    dataset_name: str = "chatdoc_medqa_4option",   # or "chatdoc_medqa_5option"
    val_ratio: float = 0.10,                       # 90/10 split; val & test share tail
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Text-only ChatDoctor MedQA-style loader (same schema as previous loaders):
      - Reads all *.parquet from:
          base_dir/med-qa-en-4options-source   (for chatdoc_medqa_4option)
          base_dir/med-qa-en-5options-source   (for chatdoc_medqa_5option)
      - Builds `messages` (user -> assistant)
      - Keeps `images` as [None] per row for schema consistency
    """
    import os, json
    import pandas as pd
    from glob import glob
    from datasets import Dataset, DatasetDict

    assert dataset_name in {"chatdoc_medqa_4option", "chatdoc_medqa_5option"}
    sub = "med-qa-en-4options-source" if dataset_name == "chatdoc_medqa_4option" else "med-qa-en-5options-source"
    folder = os.path.join(base_dir, sub)

    files = sorted(glob(os.path.join(folder, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"[{dataset_name}] No parquet files in {folder}")

    df_all = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in files], ignore_index=True)

    # Ensure columns exist
    for col in ("question", "options", "answer_idx", "answer"):
        if col not in df_all.columns:
            df_all[col] = None

    # 90/10 split with disjoint validation/test from the eval tail.
    df_train, df_val, df_test = _split_frame_ratio_disjoint(df_all, val_ratio)

    # Debug truncation
    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val   = df_val.head(min(debug_limit, len(df_val)))
        df_test  = df_test.head(min(debug_limit, len(df_test)))

    def _coerce_options(opts):
        """
        Accepts either:
          - list of {"key": "A", "value": "..."} dicts
          - dict {"A": "...", "B": "..."}
        Returns list[{"key": k, "value": v}] for uniform handling.
        """
        if isinstance(opts, list):
            # validate shape
            out = []
            for x in opts:
                if isinstance(x, dict) and "key" in x and "value" in x:
                    out.append({"key": str(x["key"]), "value": str(x["value"])})
            return out
        if isinstance(opts, dict):
            return [{"key": str(k), "value": str(v)} for k, v in opts.items()]
        return []

    def _format_batch(batch):
        msgs, imgs = [], []
        qs = batch["question"]
        os_ = batch["options"]
        aidx = batch.get("answer_idx", [None]*len(qs))
        aval = batch.get("answer", [None]*len(qs))
        n = len(qs)

        for i in range(n):
            qtext = (qs[i] or "").strip()

            # normalize options
            opt_list = _coerce_options(os_[i])
            option_text = "\n".join([f"{o['key']}. {o['value']}" for o in opt_list])

            user_q = f"##Question: {qtext}\n\n##Options:\n{option_text}"

            # resolve answer
            ans = (aval[i] or "") if aval is not None else ""
            idx = aidx[i] if aidx is not None else None
            if isinstance(idx, str):
                chosen = next((o for o in opt_list if o["key"] == idx), None)
                if chosen:
                    ans = f"{idx}. {chosen['value']}".strip()

            user = {"role": "user", "content": [{"type": "text", "text": user_q, "index": None}]}
            asst = {"role": "assistant", "content": [{"type": "text", "text": (ans or ""), "index": None}]}
            msgs.append([user, asst])

            # text-only schema
            imgs.append([None])

        return {"messages": msgs, "images": imgs}

    def _df_to_ds(df_split: pd.DataFrame, split: str) -> Dataset:
        ds = Dataset.from_pandas(df_split, preserve_index=False)
        ds = ds.map(_format_batch, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[{dataset_name}:{split}] format")
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _df_to_ds(df_train, "train"),
        "validation": _df_to_ds(df_val, "validation"),
        "test": _df_to_ds(df_test, "test"),
    })



# Common loader for both 4-option and 5-option variants
def _load_chatdoc_medqa_generic(
    folder: str,
    num_proc: int = 1,
    batch_size: int = 2048,
    val_ratio: float = 0.10,              # 90/10 split (val & test share the same tail)
    debug_limit: int | None = None,
):
    import os, json
    import pandas as pd
    from glob import glob
    from datasets import Dataset, DatasetDict

    parquet_files = sorted(glob(os.path.join(folder, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"[chatdoc_medqa] No parquet files found under {folder}")

    df_all = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in parquet_files], ignore_index=True)

    # split 90/10 with disjoint validation/test from the eval tail.
    df_train, df_val, df_test = _split_frame_ratio_disjoint(df_all, val_ratio)

    # debug limit
    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val   = df_val.head(min(debug_limit, len(df_val)))
        df_test  = df_test.head(min(debug_limit, len(df_test)))

    # normalize required columns
    for df in (df_train, df_val, df_test):
        for col in ("question", "options", "answer_idx", "answer"):
            if col not in df.columns:
                df[col] = None

    def _df_to_ds(df_split: pd.DataFrame, split: str) -> Dataset:
        ds = Dataset.from_pandas(df_split, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            qs   = batch["question"]
            opts = batch["options"]
            aidx = batch["answer_idx"] if "answer_idx" in batch else [None] * len(qs)
            atxt = batch["answer"]     if "answer"     in batch else [None] * len(qs)
            n = len(qs)

            for i in range(n):
                q = (qs[i] or "").strip()
                o = opts[i]

                # options can be list[{"key": "A", "value": "..."}, ...] or dict {"A":"..."}
                if isinstance(o, list):
                    opt_pairs = [(d.get("key"), d.get("value")) for d in o if isinstance(d, dict)]
                elif isinstance(o, dict):
                    opt_pairs = list(o.items())
                else:
                    opt_pairs = []

                options_block = "\n".join(f"{k}. {v}" for k, v in opt_pairs if k is not None)

                # resolve answer
                idx = aidx[i]
                ans_text = (atxt[i] or "").strip()

                if isinstance(idx, str):
                    # find selected option text
                    val = None
                    for k, v in opt_pairs:
                        if k == idx:
                            val = v
                            break
                    ans = f"{idx}. {val}" if val is not None else (ans_text or idx)
                else:
                    # try to map free-text answer back to a key
                    key = None
                    for k, v in opt_pairs:
                        if isinstance(v, str) and v.strip() == ans_text:
                            key = k
                            break
                    ans = f"{key}. {ans_text}" if key else ans_text

                user_q = f"##Question: {q}\n\n##Options:\n{options_block}"
                user = {"role": "user", "content": [{"type": "text", "text": user_q, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": ans, "index": None}]}
                msgs.append([user, asst])

                # text-only schema: keep images as [None]
                imgs.append([None])

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[chatdoc_medqa:{split}] format")
        # keep only messages/images
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _df_to_ds(df_train, "train"),
        "validation": _df_to_ds(df_val, "validation"),
        "test": _df_to_ds(df_test, "test"),
    })


# Register BOTH dataset names, each pointing to the same implementation
@register_dataset("chatdoc_medqa_4option")
def load_chatdoc_medqa_4option(
    num_proc: int = 1,
    batch_size: int = 2048,
    base_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor",
    val_ratio: float = 0.10,
    debug_limit: int | None = None,
) -> DatasetDict:
    folder = os.path.join(base_dir, "med-qa-en-4options-source")
    return _load_chatdoc_medqa_generic(folder, num_proc, batch_size, val_ratio, debug_limit)


@register_dataset("chatdoc_medqa_5option")
def load_chatdoc_medqa_5option(
    num_proc: int = 1,
    batch_size: int = 2048,
    base_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor",
    val_ratio: float = 0.10,
    debug_limit: int | None = None,
) -> DatasetDict:
    folder = os.path.join(base_dir, "med-qa-en-5options-source")
    return _load_chatdoc_medqa_generic(folder, num_proc, batch_size, val_ratio, debug_limit)



@register_dataset("medical_meadow_pubmed_causal")
def load_medical_meadow_pubmed_causal(
    num_proc: int = 1,
    batch_size: int = 2048,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/medical_meadow_pubmed_causal",
    tail_for_eval: int = 300,            # last N rows reserved for validation/test
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Text-only loader (same structure as prior loaders):
      - reads ALL *.parquet from data_dir
      - split disjointly: train excludes the eval tail; eval tail is split into validation/test
      - builds `messages` (user -> assistant)
      - keeps `images` as [None] per sample for schema consistency
    """
    import os
    import pandas as pd
    from glob import glob
    from datasets import Dataset, DatasetDict

    OPTIONS = {
        "A": "This is a directly correlative relationship",
        "B": "This is a conditionally causative relationship",
        "C": "This is a causative relationship",
        "D": "This no relationship.",
    }

    # ---------- read ----------
    files = sorted(glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"[medical_meadow_pubmed_causal] No parquet files in {data_dir}")
    df_all = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in files], ignore_index=True)

    # ---------- splits ----------
    n = len(df_all)
    eval_count = min(tail_for_eval, n)
    df_train, df_val, df_test = _split_frame_tail_disjoint(df_all, eval_count)

    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val   = df_val.head(min(debug_limit, len(df_val)))
        df_test  = df_test.head(min(debug_limit, len(df_test)))

    def _df_to_ds(df_split: pd.DataFrame, split: str) -> Dataset:
        # ensure columns
        for c in ("input", "output"):
            if c not in df_split.columns:
                df_split[c] = ""

        ds = Dataset.from_pandas(df_split, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            subj = batch["input"]
            outs = batch["output"]
            n = len(subj)

            options_blk = "\n".join(f"{k}. {v}" for k, v in OPTIONS.items())

            for i in range(n):
                subject = (subj[i] or "").strip()
                out_txt = (outs[i] or "").strip()

                # find key for provided output text (exact match); fall back to raw text
                key = next((k for k, v in OPTIONS.items() if v == out_txt), None)
                final_ans = f"{key}. {out_txt}" if key else out_txt

                qtxt = (
                    f"##Subject: {subject}\n\n"
                    f"Question: Is this describing a :\n\n"
                    f"##Options:\n{options_blk}"
                )

                user = {"role": "user", "content": [{"type": "text", "text": qtxt, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": final_ans, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[pubmed_causal:{split}] format")

        # keep only standard columns
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _df_to_ds(df_train, "train"),
        "validation": _df_to_ds(df_val, "validation"),
        "test": _df_to_ds(df_test, "test"),
    })




# One generic implementation for all four ChatDoctor MedicalMeadow text-only sets
def _load_chatdoctor_qna_generic(
    base_dir: str,
    num_proc: int = 1,
    batch_size: int = 2048,
    val_tail: int = 500,            # last N rows → validation & test
    debug_limit: int | None = None,
):
    import os
    import pandas as pd
    from glob import glob
    from datasets import Dataset, DatasetDict

    files = sorted(glob(os.path.join(base_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"[chatdoctor_qna] No parquet files in {base_dir}")
    df_all = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in files], ignore_index=True)

    # split: disjoint train + val/test from tail window
    eval_count = min(val_tail, len(df_all))
    df_train, df_val, df_test = _split_frame_tail_disjoint(df_all, eval_count)

    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val   = df_val.head(min(debug_limit, len(df_val)))
        df_test  = df_test.head(min(debug_limit, len(df_test)))

    # ensure columns
    for df in (df_train, df_val, df_test):
        for c in ("instruction", "input", "output"):
            if c not in df.columns:
                df[c] = ""

    def _df_to_ds(df_split: pd.DataFrame, split: str) -> Dataset:
        ds = Dataset.from_pandas(df_split, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            instrs, inputs, outs = batch["instruction"], batch["input"], batch["output"]
            n = len(instrs)
            for i in range(n):
                instruction = (instrs[i] or "").strip()
                inp         = (inputs[i] or "").strip()
                out         = (outs[i] or "").strip()

                qtxt = f"##Instruction:  {instruction}\n\nQuestion: {inp}"
                user = {"role": "user", "content": [{"type": "text", "text": qtxt, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": out, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema consistency

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[chatdoctor_qna:{split}] format")
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _df_to_ds(df_train, "train"),
        "validation": _df_to_ds(df_val, "validation"),
        "test": _df_to_ds(df_test, "test"),
    })


# Register four dataset names that share the same loader
@register_dataset("medical_meadow_flashcard")
def load_medical_meadow_flashcard(
    num_proc: int = 1,
    batch_size: int = 2048,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/medical_meadow_medical_flashcards",
    val_tail: int = 500,
    debug_limit: int | None = None,
) -> DatasetDict:
    return _load_chatdoctor_qna_generic(data_dir, num_proc, batch_size, val_tail, debug_limit)


@register_dataset("medical_meadow_mediqa")
def load_medical_meadow_mediqa(
    num_proc: int = 1,
    batch_size: int = 2048,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/medical_meadow_mediqa",
    val_tail: int = 500,
    debug_limit: int | None = None,
) -> DatasetDict:
    return _load_chatdoctor_qna_generic(data_dir, num_proc, batch_size, val_tail, debug_limit)


@register_dataset("medical_meadow_wikidoc")
def load_medical_meadow_wikidoc(
    num_proc: int = 1,
    batch_size: int = 2048,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/medical_meadow_wikidoc",
    val_tail: int = 500,
    debug_limit: int | None = None,
) -> DatasetDict:
    return _load_chatdoctor_qna_generic(data_dir, num_proc, batch_size, val_tail, debug_limit)


@register_dataset("medical_meadow_wikidoc_patient_information")
def load_medical_meadow_wikidoc_patient_information(
    num_proc: int = 1,
    batch_size: int = 2048,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/medical_meadow_wikidoc_patient_information",
    val_tail: int = 500,
    debug_limit: int | None = None,
) -> DatasetDict:
    return _load_chatdoctor_qna_generic(data_dir, num_proc, batch_size, val_tail, debug_limit)



# Generic MMMLU-style loader (text-only, shared by all subjects)
def _load_chatdoctor_mmmlu_generic(
    data_dir: str,
    num_proc: int = 1,
    batch_size: int = 2048,
    val_tail: int = 50,                # last N rows → validation & test
    debug_limit: int | None = None,
):
    """
    Reads all *.parquet in `data_dir` with columns: input, A, B, C, D, target.
    Produces a DatasetDict with `messages` and `images` ([None]) matching prior loaders.
    """
    import os
    import pandas as pd
    from glob import glob
    from datasets import Dataset, DatasetDict

    files = sorted(glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"[mmmlu] No parquet files in {data_dir}")

    df_all = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in files], ignore_index=True)

    # ensure required columns
    req = ["input", "A", "B", "C", "D", "target"]
    for c in req:
        if c not in df_all.columns:
            raise ValueError(f"[mmmlu] Missing required column '{c}' in {data_dir}")

    # splits: disjoint train + val/test from tail window
    eval_count = min(val_tail, len(df_all))
    df_train, df_val, df_test = _split_frame_tail_disjoint(df_all, eval_count)

    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val   = df_val.head(min(debug_limit, len(df_val)))
        df_test  = df_test.head(min(debug_limit, len(df_test)))

    def _df_to_ds(df_split: pd.DataFrame, split: str) -> Dataset:
        ds = Dataset.from_pandas(df_split, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            q  = batch["input"]
            A, B, C, D = batch["A"], batch["B"], batch["C"], batch["D"]
            tgt = batch["target"]
            n = len(q)

            for i in range(n):
                question_text = (q[i] or "").strip()
                opts = {
                    "A": (A[i] or "").strip(),
                    "B": (B[i] or "").strip(),
                    "C": (C[i] or "").strip(),
                    "D": (D[i] or "").strip(),
                }
                target = (tgt[i] or "").strip()

                options_block = "\n".join(f"{k}. {v}" for k, v in opts.items())
                user_q = f"Question: {question_text}\n\n##Options:\n{options_block}"

                ans = f"{target}. {opts.get(target, 'Unknown')}".strip()

                user = {"role": "user", "content": [{"type": "text", "text": user_q, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": ans, "index": None}]}
                msgs.append([user, asst])

                # text-only schema consistency
                imgs.append([None])

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[mmmlu:{split}] format")
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _df_to_ds(df_train, "train"),
        "validation": _df_to_ds(df_val, "validation"),
        "test": _df_to_ds(df_test, "test"),
    })


# --- Register each MMMLU subject to the same generic implementation ---

@register_dataset("mmmlu_anatomy")
def load_mmmlu_anatomy(
    num_proc: int = 1, batch_size: int = 2048, val_tail: int = 50, debug_limit: int | None = None,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/mmmlu-anatomy",
) -> DatasetDict:
    return _load_chatdoctor_mmmlu_generic(data_dir, num_proc, batch_size, val_tail, debug_limit)

@register_dataset("mmmlu_clinical_knowledge")
def load_mmmlu_clinical_knowledge(
    num_proc: int = 1, batch_size: int = 2048, val_tail: int = 50, debug_limit: int | None = None,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/mmmlu-clinical-knowledge",
) -> DatasetDict:
    return _load_chatdoctor_mmmlu_generic(data_dir, num_proc, batch_size, val_tail, debug_limit)

@register_dataset("mmmlu_college_biology")
def load_mmmlu_college_biology(
    num_proc: int = 1, batch_size: int = 2048, val_tail: int = 50, debug_limit: int | None = None,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/mmmlu-college-biology",
) -> DatasetDict:
    return _load_chatdoctor_mmmlu_generic(data_dir, num_proc, batch_size, val_tail, debug_limit)

@register_dataset("mmmlu_college_medicine")
def load_mmmlu_college_medicine(
    num_proc: int = 1, batch_size: int = 2048, val_tail: int = 50, debug_limit: int | None = None,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/mmmlu-college-medicine",
) -> DatasetDict:
    return _load_chatdoctor_mmmlu_generic(data_dir, num_proc, batch_size, val_tail, debug_limit)

@register_dataset("mmmlu_medical_genetics")
def load_mmmlu_medical_genetics(
    num_proc: int = 1, batch_size: int = 2048, val_tail: int = 50, debug_limit: int | None = None,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/mmmlu-medical-genetics",
) -> DatasetDict:
    return _load_chatdoctor_mmmlu_generic(data_dir, num_proc, batch_size, val_tail, debug_limit)

@register_dataset("mmmlu_professional_medicine")
def load_mmmlu_professional_medicine(
    num_proc: int = 1, batch_size: int = 2048, val_tail: int = 50, debug_limit: int | None = None,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/mmmlu-professional-medicine",
) -> DatasetDict:
    return _load_chatdoctor_mmmlu_generic(data_dir, num_proc, batch_size, val_tail, debug_limit)



# Generic text-only loader for ChatDoctor MedQA (both 4- and 5-option variants)
# Generic text-only loader for ChatDoctor MedQA (both 4- and 5-option variants)
def _load_chatdoc_medqa_generic(
    data_dir: str,
    num_proc: int = 1,
    batch_size: int = 2048,
    val_tail: int = 500,              # last N rows → validation & test
    debug_limit: int | None = None,
    cache_to: str | None = None,
):
    """
    Reads all *.parquet under `data_dir` with columns:
      - question: str
      - options: list[{"key": "A", "value": "..."}]  (or dict {"A": "...", ...})
      - answer_idx: str key (e.g., "B") [optional]
      - answer: str free-text [optional]

    Returns a DatasetDict with fields:
      - messages: [ {user}, {assistant} ]
      - images: [None]  (schema compatibility with multimodal loaders)
    """
    import os
    import pandas as pd
    from glob import glob
    from datasets import Dataset, DatasetDict

    # Check if cached version exists
    if cache_to and os.path.exists(cache_to):
        try:
            print(f"[chatdoc_medqa] Loading from cache: {cache_to}")
            return DatasetDict.load_from_disk(cache_to)
        except Exception as e:
            print(f"[chatdoc_medqa] Cache load failed: {e}, rebuilding...")

    files = sorted(glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"[chatdoc_medqa] No parquet files in {data_dir}")

    df_all = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in files], ignore_index=True)

    # Normalize required columns
    for col in ("question", "options"):
        if col not in df_all.columns:
            raise ValueError(f"[chatdoc_medqa] Missing required column '{col}' in {data_dir}")
    for col in ("answer_idx", "answer"):
        if col not in df_all.columns:
            df_all[col] = ""

    # Ensure val_tail is valid
    N = len(df_all)
    if N == 0:
        raise ValueError(f"[chatdoc_medqa] No data found in {data_dir}")
    
    # Ensure val_tail is at least 1 and not more than total samples
    val_tail = max(1, min(val_tail, N))
    
    # Split: disjoint train + val/test from tail window.
    df_train, df_val, df_test = _split_frame_tail_disjoint(df_all, val_tail)

    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val = df_val.head(min(debug_limit, len(df_val)))
        df_test = df_test.head(min(debug_limit, len(df_test)))

    def _df_to_ds(df_split: pd.DataFrame, split: str) -> Dataset:
        if len(df_split) == 0:
            # Return empty dataset with correct schema
            return Dataset.from_dict({"messages": [], "images": []})
        
        ds = Dataset.from_pandas(df_split, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            qs = batch["question"]
            opts = batch["options"]
            aidx = batch["answer_idx"]
            atxt = batch["answer"]
            n = len(qs)

            for i in range(n):
                q = (qs[i] or "").strip()
                o = opts[i]

                # options may be list[{"key","value"}] or dict
                if isinstance(o, list):
                    pairs = [(d.get("key"), d.get("value")) for d in o if isinstance(d, dict)]
                elif isinstance(o, dict):
                    pairs = list(o.items())
                else:
                    pairs = []

                options_block = "\n".join(f"{k}. {v}" for k, v in pairs if k is not None)

                idx = (aidx[i] or "").strip()
                aval = (atxt[i] or "").strip()

                # Resolve final answer text
                if idx:
                    picked = next((v for k, v in pairs if k == idx), None)
                    ans = f"{idx}. {picked}" if picked is not None else (aval or idx)
                else:
                    # Try to map free text back to a key
                    key = next((k for k, v in pairs if isinstance(v, str) and v.strip() == aval), None)
                    ans = f"{key}. {aval}" if key else aval

                user_q = f"##Question: {q}\n\n##Options:\n{options_block}"
                user = {"role": "user", "content": [{"type": "text", "text": user_q, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": ans, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema

            return {"messages": msgs, "images": imgs}

        ds = ds.map(
            _format,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"[chatdoc_medqa:{split}] format"
        )

        keep = {"messages", "images"}
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)
        return ds

    dataset_dict = DatasetDict({
        "train": _df_to_ds(df_train, "train"),
        "validation": _df_to_ds(df_val, "validation"),
        "test": _df_to_ds(df_test, "test"),
    })

    # Cache if requested
    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        print(f"[chatdoc_medqa] Saving to cache: {cache_to}")
        dataset_dict.save_to_disk(cache_to)

    return dataset_dict



# Register both dataset names to the same implementation
@register_dataset("chatdoc_4option")
def load_chatdoc_4option(
    num_proc: int = 1, batch_size: int = 2048, val_tail: int = 200, debug_limit: int | None = None,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/med-qa-en-4options-source",
) -> DatasetDict:
    return _load_chatdoc_medqa_generic(data_dir, num_proc, batch_size, val_tail, debug_limit)


@register_dataset("chatdoc_5option")
def load_chatdoc_5option(
    num_proc: int = 1, batch_size: int = 2048, val_tail: int = 200, debug_limit: int | None = None,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/med-qa-en-5options-source",
) -> DatasetDict:
    return _load_chatdoc_medqa_generic(data_dir, num_proc, batch_size, val_tail, debug_limit)




@register_dataset("medical_meadow_mmmlu")
def load_medical_meadow_mmmlu(
    num_proc: int = 1, batch_size: int = 2048, val_tail: int = 200, debug_limit: int | None = None,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/medical_meadow_mmmlu",
) -> DatasetDict:
    """
    Medical Meadow MMMLU (text-only) in the same standardized structure:
      - reads ALL *.parquet from `data_dir`
      - splits disjointly: train excludes eval tail; eval tail is split into validation/test
      - returns `messages` (user->assistant) and `images` = [None] per row
    Expected columns: instruction, input, output
    'input' contains question + options A-D (e.g., '... A: ... B: ... C: ... D: ...').
    """
    import os, re
    import pandas as pd
    from glob import glob
    from datasets import Dataset, DatasetDict

    files = sorted(glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"[medical_meadow_mmmlu] No parquet files in {data_dir}")

    df_all = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in files], ignore_index=True)

    # Ensure required columns exist
    for col in ("instruction", "input", "output"):
        if col not in df_all.columns:
            df_all[col] = ""

    # Split: disjoint train + val/test from tail window.
    eval_count = min(val_tail, len(df_all))
    df_train, df_val, df_test = _split_frame_tail_disjoint(df_all, eval_count)

    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val   = df_val.head(min(debug_limit, len(df_val)))
        df_test  = df_test.head(min(debug_limit, len(df_test)))

    # Robust A–D option extractor:
    # - handles 'A: text', 'A. text', 'A ) text', etc.
    opt_pattern = re.compile(r'([A-D])\s*[:\.\)]\s*(.*?)(?=(?:\n|\r|\s)[A-D]\s*[:\.\)]|$)', re.S)

    def _df_to_ds(df_split: pd.DataFrame, split: str) -> Dataset:
        ds = Dataset.from_pandas(df_split, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            instrs, inputs, outs = batch["instruction"], batch["input"], batch["output"]
            n = len(inputs)

            for i in range(n):
                instruction = (instrs[i] or "").strip()
                inp         = (inputs[i] or "")
                out_key     = (outs[i] or "").strip()

                # Extract question part before first option marker (A/B/C/D)
                first_opt = re.search(r'\b[A-D]\s*[:\.\)]', inp)
                if first_opt:
                    question_part = inp[:first_opt.start()].strip()
                    opt_blob      = inp[first_opt.start():]
                else:
                    # Fallback: try split on 'A:' specifically
                    chunks = re.split(r'\bA\s*[:\.\)]', inp, maxsplit=1)
                    question_part = chunks[0].strip()
                    opt_blob = 'A: ' + chunks[1] if len(chunks) > 1 else ""

                pairs = opt_pattern.findall(opt_blob)
                options_dict = {k: v.strip() for k, v in pairs}

                # Build user prompt
                options_block = "\n".join(f"{k}. {v}" for k, v in options_dict.items())
                user_q = f"{instruction}\n\nQuestion: {question_part}\n\nOptions:\n{options_block}"

                # Resolve final answer
                if out_key in options_dict:
                    ans = f"{out_key}. {options_dict[out_key]}"
                else:
                    # Try to map free-text back to a key if present in dict
                    key = next((k for k, v in options_dict.items() if v.strip() == out_key), None)
                    ans = f"{key}. {out_key}" if key else out_key

                user = {"role": "user", "content": [{"type": "text", "text": user_q, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": ans, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[medical_meadow_mmmlu:{split}] format")
        # Keep only the standardized columns
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _df_to_ds(df_train, "train"),
        "validation": _df_to_ds(df_val, "validation"),
        "test": _df_to_ds(df_test, "test"),
    })



@register_dataset("medical_meadow_cord19")
def load_medical_meadow_cord19(
    num_proc: int = 1,
    batch_size: int = 2048,
    data_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/ChatDoctor/medical_meadow_cord19",
    val_tail: int = 200,             # last N rows → validation & test
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    ChatDoctor medical_meadow_cord19 summarization (text-only), standardized structure:
      - reads ALL *.parquet from `data_dir`
      - splits disjointly: train excludes eval tail; eval tail is split into validation/test
      - returns `messages` (user -> assistant) and `images` = [None] per row
    Requires columns: instruction, input, output
    """
    import os
    import pandas as pd
    from glob import glob
    from datasets import Dataset, DatasetDict

    files = sorted(glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"[medical_meadow_cord19] No parquet files in {data_dir}")

    df_all = pd.concat([pd.read_parquet(f, engine="pyarrow") for f in files], ignore_index=True)

    # Ensure required columns
    for col in ("instruction", "input", "output"):
        if col not in df_all.columns:
            df_all[col] = ""

    # Split: disjoint train + val/test from tail window.
    eval_count = min(val_tail, len(df_all))
    df_train, df_val, df_test = _split_frame_tail_disjoint(df_all, eval_count)

    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val   = df_val.head(min(debug_limit, len(df_val)))
        df_test  = df_test.head(min(debug_limit, len(df_test)))

    def _df_to_ds(df_split: pd.DataFrame, split: str) -> Dataset:
        ds = Dataset.from_pandas(df_split, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            instrs, inputs, outs = batch["instruction"], batch["input"], batch["output"]
            n = len(instrs)
            for i in range(n):
                instruction = (instrs[i] or "").strip()
                inp         = (inputs[i] or "").strip()
                out         = (outs[i] or "").strip()

                # User prompt: instruction + abstract/input text
                qtxt = f"{instruction}\n\n{inp}".strip()

                user = {"role": "user", "content": [{"type": "text", "text": qtxt, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": out, "index": None}]}
                msgs.append([user, asst])

                # text-only schema consistency
                imgs.append([None])

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[medical_meadow_cord19:{split}] format")

        # Keep only standardized columns
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _df_to_ds(df_train, "train"),
        "validation": _df_to_ds(df_val, "validation"),
        "test": _df_to_ds(df_test, "test"),
    })



@register_dataset("mimic_ext_bhc")
def load_mimic_ext_bhc(
    num_proc: int = 1,
    batch_size: int = 2048,
    csv_path: str = "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-IV-Ext-BHC/processed_summaries.csv",
    val_tail: int = 300,              # last N rows → validation & test
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    MIMIC-IV-Ext-BHC summarization (text-only) in the standardized structure:
      - reads a single CSV with columns: note_id, input, target, input_tokens, target_tokens
      - splits disjointly: train excludes eval tail; eval tail is split into validation/test
      - returns `messages` (user -> assistant) and `images` = [None] per row
    """
    import pandas as pd
    from datasets import Dataset, DatasetDict

    INSTRUCTION = "##Instruction: Please summarize the following medical document into a concise, clinically relevant summary."

    df_all = pd.read_csv(csv_path)

    # ensure required columns exist
    for col in ("input", "target"):
        if col not in df_all.columns:
            raise ValueError(f"[mimic_ext_bhc] Missing required column '{col}' in {csv_path}")
    # optional columns
    for col in ("note_id", "input_tokens", "target_tokens"):
        if col not in df_all.columns:
            df_all[col] = None

    # split: disjoint train + val/test from tail window.
    eval_count = min(val_tail, len(df_all))
    df_train, df_val, df_test = _split_frame_tail_disjoint(df_all, eval_count)

    if debug_limit:
        df_train = df_train.head(debug_limit)
        df_val   = df_val.head(min(debug_limit, len(df_val)))
        df_test  = df_test.head(min(debug_limit, len(df_test)))

    def _df_to_ds(df_split: pd.DataFrame, split: str) -> Dataset:
        ds = Dataset.from_pandas(df_split, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            inputs, targets = batch["input"], batch["target"]
            note_ids  = batch.get("note_id", [None] * len(inputs))
            in_toks   = batch.get("input_tokens", [None] * len(inputs))
            out_toks  = batch.get("target_tokens", [None] * len(inputs))
            n = len(inputs)

            for i in range(n):
                qtxt = f"{INSTRUCTION}\n\n##Input: { (inputs[i] or '').strip() }"
                atxt = (targets[i] or "").strip()

                user = {"role": "user", "content": [{"type": "text", "text": qtxt, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": atxt, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[mimic_ext_bhc:{split}] format")

        # Keep only standardized columns
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _df_to_ds(df_train, "train"),
        "validation": _df_to_ds(df_val, "validation"),
        "test": _df_to_ds(df_test, "test"),
    })



@register_dataset("medical_o1_sft_mix")
def load_medical_o1_sft_mix(
    num_proc: int = 1,
    batch_size: int = 2048,
    json_path: str = "./Medmo_Dataset_1/Medmo_Dataset/Reasoning_Dataset/medical-o1-reasoning-SFT/medical_o1_sft_mix.json",
    val_tail: int = 200,            # last N rows → validation & test
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Medical O1 Reasoning SFT (text-only), standardized like previous loaders:
      - reads a single JSON list with keys: Question, Complex_CoT, Response
      - splits disjointly: train excludes eval tail; eval tail is split into validation/test
      - returns `messages` (user -> assistant) and `images` = [None] per row
    """
    import json
    from datasets import Dataset, DatasetDict

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("[medical_o1_sft_mix] JSON must contain a list of samples")

    # basic normalization
    rows = []
    for d in data[: (debug_limit or len(data))]:
        rows.append({
            "Question": d.get("Question", "") or "",
            "Complex_CoT": d.get("Complex_CoT", "") or "",
            "Response": d.get("Response", "") or "",
        })

    if not rows:
        empty = Dataset.from_dict({"messages": [], "images": []})
        return DatasetDict({
            "train": empty,
            "validation": empty.select([]),
            "test": empty.select([]),
        })

    ds_all = Dataset.from_list(rows)

    # disjoint train + val/test from tail window
    ds_train, ds_val, ds_test = _split_dataset_tail_disjoint(ds_all, val_tail)

    def _format(ds: Dataset, split: str) -> Dataset:
        def _map(batch):
            msgs, imgs = [], []
            Q, A = batch["Question"], batch["Response"]
            n = len(Q)
            for i in range(n):
                qtxt = f"##Question: { (Q[i] or '').strip() }"
                atxt = (A[i] or "").strip()
                user = {"role": "user", "content": [{"type": "text", "text": qtxt, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": atxt, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema
            return {"messages": msgs, "images": imgs}

        ds = ds.map(_map, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[medical_o1_sft_mix:{split}] format")
        # Keep only standardized columns
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _format(ds_train, "train"),
        "validation": _format(ds_val, "validation"),
        "test": _format(ds_test, "test"),
    })



@register_dataset("medical_o1_verifiable_problem")
def load_medical_o1_verifiable_problem(
    num_proc: int = 1,
    batch_size: int = 2048,
    json_path: str = "./Medmo_Dataset_1/Medmo_Dataset/Reasoning_Dataset/medical-o1-verifiable-problem/medical_o1_verifiable_problem.json",
    val_tail: int = 200,            # last N rows → validation & test
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Medical O1 Verifiable Problem (text-only) in the standardized structure:
      - reads JSON list with keys: 'Open-ended Verifiable Question', 'Ground-True Answer'
      - splits disjointly: train excludes eval tail; eval tail is split into validation/test
      - returns `messages` (user -> assistant) and `images` = [None] per row
    """
    import json
    from datasets import Dataset, DatasetDict

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("[medical_o1_verifiable_problem] JSON must be a list of samples")

    # normalize + optional debug cap
    rows = []
    for d in data[: (debug_limit or len(data))]:
        rows.append({
            "question": d.get("Open-ended Verifiable Question", "") or "",
            "answer":   d.get("Ground-True Answer", "") or "",
        })

    if not rows:
        empty = Dataset.from_dict({"messages": [], "images": []})
        return DatasetDict({"train": empty, "validation": empty, "test": empty})

    ds_all = Dataset.from_list(rows)

    # disjoint train + val/test from tail window
    ds_train, ds_val, ds_test = _split_dataset_tail_disjoint(ds_all, val_tail)

    def _format(ds: Dataset, split: str) -> Dataset:
        def _map(batch):
            msgs, imgs = [], []
            q, a = batch["question"], batch["answer"]
            n = len(q)
            for i in range(n):
                qtxt = f"##Question: { (q[i] or '').strip() }"
                atxt = (a[i] or "").strip()
                user = {"role": "user", "content": [{"type": "text", "text": qtxt, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": atxt, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema
            return {"messages": msgs, "images": imgs}

        ds = ds.map(_map, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[medical_o1_verifiable_problem:{split}] format")
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _format(ds_train, "train"),
        "validation": _format(ds_val, "validation"),
        "test": _format(ds_test, "test"),
    })



@register_dataset("medical_r1_distill")
def load_medical_r1_distill(
    num_proc: int = 1,
    batch_size: int = 2048,
    json_path: str = "./Medmo_Dataset_1/Medmo_Dataset/Reasoning_Dataset/Medical-R1-Distill-Data/medical_r1_distill_sft.json",
    val_tail: int = 200,            # last N rows → validation & test
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Medical R1 Distill (text-only) in the standardized structure:
      - reads a JSON list with keys: 'question', 'response (content)'
      - splits disjointly: train excludes eval tail; eval tail is split into validation/test
      - returns `messages` (user -> assistant) and `images` = [None] per row
    """
    import json
    from datasets import Dataset, DatasetDict

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("[medical_r1_distill] JSON must be a list of samples")

    # normalize + optional debug cap
    rows = []
    for d in data[: (debug_limit or len(data))]:
        rows.append({
            "question": d.get("question", "") or "",
            "answer":   d.get("response (content)", "") or "",
        })

    if not rows:
        empty = Dataset.from_dict({"messages": [], "images": []})
        return DatasetDict({"train": empty, "validation": empty, "test": empty})

    ds_all = Dataset.from_list(rows)

    # disjoint train + val/test from tail window
    ds_train, ds_val, ds_test = _split_dataset_tail_disjoint(ds_all, val_tail)

    def _format(ds: Dataset, split: str) -> Dataset:
        def _map(batch):
            msgs, imgs = [], []
            q, a = batch["question"], batch["answer"]
            n = len(q)
            for i in range(n):
                qtxt = f"##Question: { (q[i] or '').strip() }"
                atxt = (a[i] or "").strip()
                user = {"role": "user", "content": [{"type": "text", "text": qtxt, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": atxt, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema
            return {"messages": msgs, "images": imgs}

        ds = ds.map(_map, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[medical_r1_distill:{split}] format")
        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    return DatasetDict({
        "train": _format(ds_train, "train"),
        "validation": _format(ds_val, "validation"),
        "test": _format(ds_test, "test"),
    })




@register_dataset("medreason")
def load_medreason(
    num_proc: int = 1,
    batch_size: int = 2048,
    jsonl_path: str = "./Medmo_Dataset_1/Medmo_Dataset/Reasoning_Dataset/MedReason/ours_quality_33000.jsonl",
    debug_limit: int | None = None,
    cache_to: str | None = None,
) -> DatasetDict:
    """
    MedReason (text-only) in the standardized structure:
      - reads a JSONL with fields: question (str), options (str like "A. ...\nB. ..."), answer (str)
      - returns only train split
      - returns `messages` (user -> assistant) and `images` = [None] per row
    """
    import json, re
    from datasets import Dataset, DatasetDict

    # Check if cached version exists
    if cache_to and os.path.exists(cache_to):
        try:
            print(f"[medreason] Loading from cache: {cache_to}")
            return DatasetDict.load_from_disk(cache_to)
        except Exception as e:
            print(f"[medreason] Cache load failed: {e}, rebuilding...")

    # Read JSONL
    rows_raw = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows_raw.append(json.loads(line))

    if debug_limit:
        rows_raw = rows_raw[:debug_limit]

    if not rows_raw:
        empty = Dataset.from_dict({"messages": [], "images": []})
        return DatasetDict({"train": empty})

    # To HF Dataset
    ds_train = Dataset.from_list(rows_raw)

    # Robust option pattern: "A. text", "B) text", "C : text"
    opt_pat = re.compile(r'([A-D])\s*[\.\):]\s*(.+?)(?=(?:\n|^)[A-D]\s*[\.\):]|\Z)', re.S)

    def _format(batch):
        msgs, imgs = [], []
        q_list = batch.get("question", [])
        o_list = batch.get("options", [])
        a_list = batch.get("answer", [])
        n = len(q_list)

        for i in range(n):
            qtxt = (q_list[i] or "").strip()
            opts_raw = (o_list[i] or "")
            ans_raw = (a_list[i] or "").strip()

            # Parse options into ordered pairs
            pairs = opt_pat.findall(opts_raw)  # list of (letter, text)
            option_lines = [f"{k}. {v.strip().rstrip('.')}" for k, v in pairs]
            options_block = "\n".join(option_lines)

            # Best-effort answer extraction:
            # 1) drop explanation tail if present
            ans_main = ans_raw.split("Explanation:", 1)[0].strip().rstrip(".")
            # 2) handle "The final decision is: yes/no"
            m = re.search(r"The final decision is:\s*([Yy]es|[Nn]o)", ans_main)
            if m:
                ans_main = m.group(1).lower()

            # 3) try to map to an option by containment (case-insensitive)
            final = None
            for k, v in pairs:
                if ans_main and ans_main.lower() in (v or "").lower():
                    final = f"{k}. {v.strip().rstrip('.')}"
                    break
            # 4) fallback to raw answer if no match
            if final is None:
                final = ans_raw

            user_q = f"##Question: {qtxt}\n\n##Options:\n{options_block}"
            user = {"role": "user", "content": [{"type": "text", "text": user_q, "index": None}]}
            asst = {"role": "assistant", "content": [{"type": "text", "text": final, "index": None}]}
            msgs.append([user, asst])
            imgs.append([None])  # text-only schema

        return {"messages": msgs, "images": imgs}

    ds_train = ds_train.map(
        _format,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc=f"[medreason:train] format"
    )
    
    keep = {"messages", "images"}
    drop = [c for c in ds_train.column_names if c not in keep]
    if drop:
        ds_train = ds_train.remove_columns(drop)

    dataset_dict = DatasetDict({"train": ds_train})

    # Cache if requested
    if cache_to:
        os.makedirs(cache_to, exist_ok=True)
        print(f"[medreason] Saving to cache: {cache_to}")
        dataset_dict.save_to_disk(cache_to)

    return dataset_dict




@register_dataset("medsg_bbox")
def load_medsg_bbox(
    num_proc: int = 1,
    batch_size: int = 2048,
    root_path: str = "./Medmo_Dataset_1/Medmo_Dataset",
    train_root: str = "MedSG-Bench/MedSG-Train",
    bench_root: str = "MedSG-Bench/MedSG-Bench",
    task_files: list | None = None,  # ["Task1.json", ...]
    debug_limit: int | None = None,
    check_files: bool = False,
) -> "DatasetDict":
    """
    Combine all 8 tasks from both MedSG-Train and MedSG-Bench JSONs
    into one unified Hugging Face dataset with absolute image paths.
    Produces only one split: 'train'.
    """
    import os, json
    from datasets import Dataset, DatasetDict

    # Default: 8 task JSONs
    if task_files is None:
        task_files = [f"Task{i}.json" for i in range(1, 9)]

    def load_jsons(root_subdir):
        """Load all task JSONs from a given subfolder."""
        base = os.path.join(root_path, root_subdir)
        samples = []
        for tfile in task_files:
            path = os.path.join(base, tfile)
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples.extend(data)
        return samples

    # --- Load and merge all samples ---
    all_samples = load_jsons(train_root) + load_jsons(bench_root)
    if not all_samples:
        raise ValueError("No samples found in either train or bench root.")

    if debug_limit:
        all_samples = all_samples[:debug_limit]

    messages_list, images_list = [], []

    for sample in all_samples:
        # build absolute image paths
        img_paths = []
        for ip in sample.get("images", []):
            full_path = os.path.normpath(os.path.join(root_path, ip))
            if check_files and not os.path.exists(full_path):
                continue
            img_paths.append(full_path)
        if not img_paths:
            continue

        # Parse question / answer based on format type
        if "conversations" in sample:  # Train format
            question, answer = "", ""
            for conv in sample["conversations"]:
                if conv.get("from") == "human":
                    question = conv.get("value", "").replace("<image>\n", "").replace("<image>", "").strip()
                elif conv.get("from") == "gpt":
                    val = conv.get("value", "")
                    if "<|box_start|>" in val and "<|box_end|>" in val:
                        bbox_str = val.split("<|box_start|>")[1].split("<|box_end|>")[0]
                        try:
                            coords = bbox_str.strip("()").split("),(")
                            x1, y1 = map(int, coords[0].split(","))
                            x2, y2 = map(int, coords[1].split(","))
                            answer = f"Difference detected: [ [{x1}, {y1}, {x2}, {y2}] ]"
                        except:
                            answer = val
                    else:
                        answer = val
        else:  # Bench format
            question = sample.get("question", "Compare these two images carefully.")
            ans = sample.get("answer")
            if isinstance(ans, list) and len(ans) == 4:
                x1, y1, x2, y2 = ans
                answer = f"Difference detected: [ [{x1}, {y1}, {x2}, {y2}] ]"
            else:
                answer = str(ans)

        # Prepare message schema
        user_content = [{"type": "text", "text": question, "index": None}]
        for idx in range(len(img_paths)):
            user_content.append({"type": "image", "text": None, "index": idx})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer, "index": None}]},
        ]

        messages_list.append(messages)
        images_list.append(img_paths)

    ds = Dataset.from_dict({"messages": messages_list, "images": images_list})
    return DatasetDict({"train": ds})





@register_dataset("medsg_bbox_grpo")
def load_medsg_bbox_grpo(
    num_proc: int = 1,
    batch_size: int = 2048,
    root_path: str = "./Medmo_Dataset_1/Medmo_Dataset",
    train_root: str = "MedSG-Bench/MedSG-Train",
    bench_root: str = "MedSG-Bench/MedSG-Bench",
    task_files: list | None = None,
    debug_limit: int | None = 20,
    check_files: bool = False,
    verbose: bool = False,
) -> DatasetDict:
    """
    MedSG dataset → GRPO-style DatasetDict.
    Combines all 8 tasks from both MedSG-Train and MedSG-Bench.
    Returns image paths as lists (already multiple images per sample).
    NO resizing of images or boxes. Boxes are kept in original pixel coordinates.

    Final schema: ['image','problem','solution','original_question','original_answer']
    """

    import os, json, hashlib
    from typing import List, Any
    from datasets import Dataset, DatasetDict, Features, Value

    # ---------------- prompts ----------------
    PROMPTS: List[str] = [
        "Compare these two images carefully and identify the coordinates of their difference.",
        "Analyze both images and locate the region where they differ. Provide bounding box coordinates.",
        "Examine these images side-by-side and detect any differences. Return the bounding box of the changed region.",
        "Find the difference between these two medical images and provide its location as bounding box coordinates.",
        "Compare these images and identify where they differ. Provide coordinates as: [x_min, y_min, x_max, y_max].",
    ]

    def _det_prompt(uid_val: Any, idx: int) -> str:
        key = str(uid_val) if uid_val is not None else str(idx)
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        return PROMPTS[h % len(PROMPTS)]

    # ------------- formatting -------------
    def _format_solution_bbox(x1: int, y1: int, x2: int, y2: int) -> str:
        return f"<answer>\nDifference detected: <box> {x1}, {y1}, {x2}, {y2} </box>\n</answer>"

    def _format_solution_text(text: str) -> str:
        return f"<answer>\n{text}\n</answer>"

    def _orig_answer_compact(label: str, box: List[int] | None) -> str:
        if box:
            return str({"label": label, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
        else:
            return str({"label": label, "box": None})

    # ---------------- default task files ----------------
    if task_files is None:
        task_files = [f"Task{i}.json" for i in range(1, 9)]

    # ------------- loading helpers -------------
    def _load_jsons(root_subdir: str) -> List[dict]:
        base = os.path.join(root_path, root_subdir)
        samples = []
        for tfile in task_files:
            path = os.path.join(base, tfile)
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples.extend(data)
        return samples

    # ------------- build split -------------
    def _build_split() -> Dataset:
        all_samples = _load_jsons(train_root) + _load_jsons(bench_root)
        if not all_samples:
            raise ValueError("No samples found in either train or bench root.")

        if debug_limit:
            all_samples = all_samples[:debug_limit]

        data = {
            "image": [],
            "problem": [],
            "solution": [],
            "original_question": [],
            "original_answer": []
        }

        for i, sample in enumerate(all_samples):
            # Build absolute image paths
            img_paths = []
            for ip in sample.get("images", []):
                full_path = os.path.normpath(os.path.join(root_path, ip))
                if check_files and not os.path.exists(full_path):
                    continue
                img_paths.append(full_path)
            if not img_paths:
                continue

            # Parse question / answer
            question = ""
            answer_text = ""
            bbox = None

            if "conversations" in sample:  # Train format
                for conv in sample["conversations"]:
                    if conv.get("from") == "human":
                        # Strip any existing <image> tokens in source to avoid duplication
                        question = conv.get("value", "").replace("<image>\n", "").replace("<image>", "").strip()
                    elif conv.get("from") == "gpt":
                        val = conv.get("value", "")
                        if "<|box_start|>" in val and "<|box_end|>" in val:
                            bbox_str = val.split("<|box_start|>")[1].split("<|box_end|>")[0]
                            try:
                                coords = bbox_str.strip("()").split("),(")
                                x1, y1 = map(int, coords[0].split(","))
                                x2, y2 = map(int, coords[1].split(","))
                                bbox = [x1, y1, x2, y2]
                                answer_text = "Difference detected"
                            except Exception:
                                answer_text = val
                        else:
                            answer_text = val
            else:  # Bench format
                question = sample.get("question", "Compare these two images carefully.")
                ans = sample.get("answer")
                if isinstance(ans, list) and len(ans) == 4:
                    bbox = ans  # [x1, y1, x2, y2]
                    answer_text = "Difference detected"
                else:
                    answer_text = str(ans)

            if not question:
                question = _det_prompt(img_paths[0] if img_paths else "", i)

            # === NEW: ensure the number of <image> tokens matches len(img_paths) ===
            # Qwen-3-VL expects one <image> token per image fed to the processor.
            image_prefix = ("<image>\n" * len(img_paths)).rstrip("\n")
            problem = f"{image_prefix}\n{question}\n" if image_prefix else f"{question}\n"
            # =====================================================================

            if bbox:
                solution = _format_solution_bbox(*bbox)
                original_answer = _orig_answer_compact(answer_text, bbox)
            else:
                solution = _format_solution_text(answer_text)
                original_answer = _orig_answer_compact(answer_text, None)

            data["image"].append(img_paths)     # list[str]
            data["problem"].append(problem)     # string with N <image> lines + question
            data["solution"].append(solution)
            data["original_question"].append(question)
            data["original_answer"].append(original_answer)

        from datasets import Sequence

        features = Features({
            "image": Sequence(Value("string")),
            "problem": Value("string"),
            "solution": Value("string"),
            "original_question": Value("string"),
            "original_answer": Value("string"),
        })

        ds = Dataset.from_dict(data, features=features)

        if verbose:
            print(f"[medsg_bbox_grpo] Loaded {len(ds)} samples")

        return ds

    ds = _build_split()
    ds_train, ds_val, ds_test = _split_dataset_tail_disjoint(ds, min(2000, len(ds)))

    dsd = DatasetDict({
        "train": ds_train,
        "validation": ds_val,
        "test": ds_test,
    })

    if verbose:
        for split, s in dsd.items():
            print(f"[medsg_bbox_grpo:{split}] rows={len(s)}, cols={s.column_names}")

    for split, s in dsd.items():
        assert s.column_names == ["image", "problem", "solution", "original_question", "original_answer"], \
            f"{split} columns are {s.column_names}, expected ['image', 'problem', 'solution', 'original_question', 'original_answer']"

    return dsd



# --------------------------
# Text QA loaders → GRPO adapters
# --------------------------
def _coalesce_text_content(content: Any) -> str:
    """Normalize arbitrary `content` payloads from chat messages into plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        txt = content.get("text")
        return txt.strip() if isinstance(txt, str) else ""
    texts: List[str] = []
    if isinstance(content, Sequence):
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    texts.append(txt.strip())
            elif isinstance(item, str) and item.strip():
                texts.append(item.strip())
    return "\n".join(texts).strip()


def _extract_role_text(conversation: Any, role: str) -> str:
    """Return the first non-empty text chunk for a given role within a chat conversation."""
    if not isinstance(conversation, list):
        return ""
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        if turn.get("role") != role:
            continue
        # Prefer structured content payloads, but fall back to direct text fields when present.
        text = _coalesce_text_content(turn.get("content"))
        if text:
            return text
        direct = turn.get("text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
    return ""


_IMAGE_TAG_RE = re.compile(r"<\s*image\s*[_\-]?\s*(\d+)?\s*>", re.IGNORECASE)


def _strip_inline_image_tokens(text: str) -> str:
    """Replace inline <image> placeholders with plain text to avoid double-counting image tokens."""
    if not isinstance(text, str):
        return ""
    def _repl(match: re.Match) -> str:
        num = match.group(1)
        return f"Image {num}" if num else "Image"
    cleaned = _IMAGE_TAG_RE.sub(_repl, text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _messages_datasetdict_to_grpo(
    ds_dict: DatasetDict,
    dataset_label: str,
    default_prompt: str,
    fmt_batch_size: int = 1024,
    fmt_num_proc: int = 1,
    wrap_answer: bool = True,
) -> DatasetDict:
    """Convert a text-only DatasetDict (messages/images) into GRPO schema."""
    features = Features({
        "image": HFSequence(Value("string")),
        "problem": Value("string"),
        "solution": Value("string"),
        "original_question": Value("string"),
        "original_answer": Value("string"),
    })

    out: Dict[str, Dataset] = {}
    for split, ds in ds_dict.items():
        if not isinstance(ds, (Dataset, IterableDataset)):
            raise TypeError(f"[{dataset_label}] Expected Dataset, got {type(ds)} for split '{split}'")

        def _map(batch):
            conversations = batch.get("messages", [])
            images = batch.get("images")
            n = len(conversations)
            if images is None or len(images) != n:
                images = [None] * n

            image_col, problem_col, solution_col = [], [], []
            orig_q_col, orig_a_col = [], []

            for i in range(n):
                conv = conversations[i]
                img_entry = images[i]

                question = _strip_inline_image_tokens(_extract_role_text(conv, "user"))
                answer = _extract_role_text(conv, "assistant")

                problem_txt = (question or "").strip() or default_prompt
                solution_body = (answer or "").strip()
                solution_txt = f"<answer>\n{solution_body}\n</answer>" if wrap_answer else solution_body

                clean_images: List[str] = []
                if isinstance(img_entry, (list, tuple)):
                    for path in img_entry:
                        if isinstance(path, str) and path.strip():
                            clean_images.append(path.strip())

                image_col.append(clean_images)
                problem_col.append(problem_txt)
                solution_col.append(solution_txt)
                orig_q_col.append(question or default_prompt)
                orig_a_col.append(answer or "")

            return {
                "image": image_col,
                "problem": problem_col,
                "solution": solution_col,
                "original_question": orig_q_col,
                "original_answer": orig_a_col,
            }

        mapped = ds.map(
            _map,
            batched=True,
            batch_size=fmt_batch_size,
            num_proc=fmt_num_proc,
            remove_columns=ds.column_names,
            desc=f"[{dataset_label}:{split}] grpo-format"
        )

        if not isinstance(mapped, IterableDataset):
            mapped = mapped.cast(features)

        out[split] = mapped

    return DatasetDict(out)


def _default_text_grpo_prompt(dataset_label: str) -> str:
    clean = dataset_label.replace("_grpo", "").replace("_", " ").strip()
    return f"Answer the following medical question from {clean}."


def _register_text_grpo_loader(
    new_name: str,
    base_loader: Callable[..., DatasetDict],
    prompt: Optional[str] = None,
    fixed_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Factory that registers a *_grpo dataset built from an existing text QA loader."""
    prompt = prompt or _default_text_grpo_prompt(new_name)
    fixed_kwargs = dict(fixed_kwargs or {})
    func_name = f"load_{new_name}"

    def _loader(
        format_batch_size: int = 1024,
        format_num_proc: int = 1,
        wrap_answer_in_tags: bool = True,
        **base_kwargs,
    ) -> DatasetDict:
        call_kwargs = {**fixed_kwargs, **base_kwargs}
        ds_dict = base_loader(**call_kwargs)
        if not isinstance(ds_dict, DatasetDict):
            raise TypeError(f"[{new_name}] Base loader must return DatasetDict, got {type(ds_dict)}")
        return _messages_datasetdict_to_grpo(
            ds_dict,
            dataset_label=new_name,
            default_prompt=prompt,
            fmt_batch_size=format_batch_size,
            fmt_num_proc=format_num_proc,
            wrap_answer=wrap_answer_in_tags,
        )

    _loader.__name__ = func_name
    globals()[func_name] = register_dataset(new_name)(_loader)


_TEXT_QA_GRPO_SPECS = [
    {"name": "pmc_instruct_qa_grpo", "base_fn": load_pmc_instruct_qa},
    {"name": "medquad_qa_grpo", "base_fn": load_medquad_qa},
    {"name": "medqa_grpo", "base_fn": load_medqa},
    {"name": "medical_meadow_medqa_grpo", "base_fn": load_medical_meadow_medqa},
    {"name": "alphacare_qa_grpo", "base_fn": load_alphacare_qa},
    {"name": "chatdoctor_healthcaremagic_grpo", "base_fn": load_chatdoctor_healthcaremagic},
    {"name": "chatdoctor_icliniq_grpo", "base_fn": load_chatdoctor_icliniq},
    {
        "name": "chatdoc_medqa_4option_grpo",
        "base_fn": load_chatdoc_medqa,
        "fixed_kwargs": {"dataset_name": "chatdoc_medqa_4option"},
    },
    {
        "name": "chatdoc_medqa_5option_grpo",
        "base_fn": load_chatdoc_medqa,
        "fixed_kwargs": {"dataset_name": "chatdoc_medqa_5option"},
    },
    {"name": "medical_meadow_pubmed_causal_grpo", "base_fn": load_medical_meadow_pubmed_causal},
    {"name": "medical_meadow_flashcard_grpo", "base_fn": load_medical_meadow_flashcard},
    {"name": "medical_meadow_mediqa_grpo", "base_fn": load_medical_meadow_mediqa},
    {"name": "medical_meadow_wikidoc_grpo", "base_fn": load_medical_meadow_wikidoc},
    {
        "name": "medical_meadow_wikidoc_patient_information_grpo",
        "base_fn": load_medical_meadow_wikidoc_patient_information,
    },
    {"name": "mmmlu_anatomy_grpo", "base_fn": load_mmmlu_anatomy},
    {"name": "mmmlu_clinical_knowledge_grpo", "base_fn": load_mmmlu_clinical_knowledge},
    {"name": "mmmlu_college_biology_grpo", "base_fn": load_mmmlu_college_biology},
    {"name": "mmmlu_college_medicine_grpo", "base_fn": load_mmmlu_college_medicine},
    {"name": "mmmlu_medical_genetics_grpo", "base_fn": load_mmmlu_medical_genetics},
    {"name": "mmmlu_professional_medicine_grpo", "base_fn": load_mmmlu_professional_medicine},
    {"name": "medical_meadow_mmmlu_grpo", "base_fn": load_medical_meadow_mmmlu},
    {"name": "medical_meadow_cord19_grpo", "base_fn": load_medical_meadow_cord19},
    {"name": "mimic_ext_bhc_grpo", "base_fn": load_mimic_ext_bhc},
    {"name": "chatdoc_4option_grpo", "base_fn": load_chatdoc_4option},
    {"name": "chatdoc_5option_grpo", "base_fn": load_chatdoc_5option},
    {"name": "medical_o1_sft_mix_grpo", "base_fn": load_medical_o1_sft_mix},
    {"name": "medical_o1_verifiable_problem_grpo", "base_fn": load_medical_o1_verifiable_problem},
    {"name": "medical_r1_distill_grpo", "base_fn": load_medical_r1_distill},
    {"name": "medreason_grpo", "base_fn": load_medreason},
]

for spec in _TEXT_QA_GRPO_SPECS:
    _register_text_grpo_loader(
        new_name=spec["name"],
        base_loader=spec["base_fn"],
        prompt=spec.get("prompt"),
        fixed_kwargs=spec.get("fixed_kwargs"),
    )





















#--------------------------------------------------
#  For the RL Training Dataset
#--------------------------------------------------
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
import os
import pandas as pd
import hashlib
from typing import List
from PIL import Image, ImageOps

# Reuse your registration + helpers
@register_dataset("nih_vqa_grpo")
def load_nih_grpo(
    num_proc: int = 8, 
    batch_size: int = 128,
    image_root: str = "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/images",
    split_csv: dict | None = None,
    debug_limit: int | None = None,
    verbose: bool = False,
) -> DatasetDict:
    """
    Optimized NIH ChestX-ray14 → HF DatasetDict with GRPO-style columns.
    Returns image paths as lists. No resizing applied.
    
    Final schema: ['image','problem','solution','original_question','original_answer']
    """

    # --------------------------
    # Default split paths
    # --------------------------
    if split_csv is None:
        split_csv = {
            "train": "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/Data_Entry_2017_clean.csv",
            "validation": "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/Data_Entry_2017_clean_val.csv",
            "test": "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/Data_Entry_2017.csv",
        }

    # --------------------------
    # Column normalization
    # --------------------------
    RENAME = {
        "Image Index": "filename",
        "Finding Labels": "caption",
        "Patient ID": "uid",
    }

    # --------------------------
    # Prompts
    # --------------------------
    NIH_PROMPTS: List[str] = [
        "Examine this chest X-ray and list all thoracic diseases or abnormalities present.",
        "Based on the image, identify all relevant diagnostic labels.",
        "From this chest radiograph, determine whether any of the NIH-labeled conditions are visible.",
        "Classify the image into one or more disease categories or labels.",
        "Detect and list all medical findings visible in the image.",
        "Look at the X-ray and return all applicable labels from the NIH dataset.",
        "Identify any visible conditions such as Cardiomegaly, Edema, or Pneumonia.",
        "Provide all relevant disease labels for this X-ray image, or specify 'No Finding'.",
        "Based on the chest X-ray, determine if the patient shows signs of any thoracic diseases.",
    ]

    def _det_prompt(uid_val, idx):
        key = str(uid_val) if uid_val is not None else str(idx)
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        return NIH_PROMPTS[h % len(NIH_PROMPTS)]

    # --------------------------
    # NIH Labels (fixed)
    # --------------------------
    NIH_LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
        "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
        "Pleural_Thickening", "Pneumonia", "Pneumothorax", "No Finding"
    ]

    # --------------------------
    # Helper function to load and process CSV
    # --------------------------
    def _load_and_process(csv_path: str) -> pd.DataFrame:
        """Load CSV and standardize columns"""
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[nih_vqa_grpo] Missing CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Apply debug limit if specified
        if debug_limit is not None and debug_limit > 0:
            df = df.iloc[:debug_limit].copy()
        
        # Rename columns
        for old, new in RENAME.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        
        # Build full path
        if "filename" in df.columns:
            df["full_path"] = df["filename"].astype(str).str.lstrip("/").apply(
                lambda n: os.path.normpath(os.path.join(image_root, n))
            )
        else:
            raise ValueError(f"[nih_vqa_grpo] CSV missing 'filename' column after renaming")
        
        return df

    # --------------------------
    # Per-split build (vectorized processing)
    # --------------------------
    def _build_split(split: str) -> Dataset:
        """Build dataset for a single split"""
        df = _load_and_process(split_csv[split])
        n = len(df)
        
        # Pre-compute prompts for all rows
        prompts = [_det_prompt(df.iloc[i].get("uid"), i) for i in range(n)]
        
        # Process all rows vectorized
        processed_data = {
            "image": [],
            "problem": [],
            "solution": [],
            "original_question": [],
            "original_answer": []
        }

        for i in range(n):
            img_path = df.iloc[i]["full_path"]
            caption = df.iloc[i].get("caption", "")
            
            # Parse labels
            raw = str(caption).strip()
            parts = [p.strip() for p in raw.split("|")] if "|" in raw else ([raw] if raw else [])
            parts = [p for p in parts if p]
            if not parts:
                parts = ["No Finding"]

            # Build fields
            q = prompts[i]
            problem = (
                f"{q}\n\n"
                f"If nothing abnormal is detected, then only return 'No Finding'. Else give the abnormality.\n"
                f"<think> You may refer to medical knowledge, hypotheses, chains of logic, etc.</think>\n"
                f"<answer>Your final answer to the user's question must be here.</answer>"
            )
            original_answer = "|".join(parts)
            solution = f"<answer>{', '.join(parts)}</answer>"

            # Append - IMAGE AS LIST
            processed_data["image"].append([img_path])  # Changed: wrap in list
            processed_data["problem"].append(problem)
            processed_data["solution"].append(solution)
            processed_data["original_question"].append(q)
            processed_data["original_answer"].append(original_answer)

        # --------------------------
        # Create dataset directly from dict (no map!)
        # --------------------------
        from datasets import Sequence
        
        features = Features({
            "image": Sequence(Value("string")),  # Changed: list of image paths
            "problem": Value("string"),
            "solution": Value("string"),
            "original_question": Value("string"),
            "original_answer": Value("string"),
        })

        hf_ds = Dataset.from_dict(processed_data, features=features)
        
        if verbose:
            print(f"[nih_vqa_grpo:{split}] Loaded {len(hf_ds)} samples")

        return hf_ds

    # --------------------------
    # Build all splits
    # --------------------------
    dsd = DatasetDict({
        "train": _build_split("train"),
        "validation": _build_split("validation"),
        "test": _build_split("test"),
    })

    if verbose:
        for split, ds in dsd.items():
            print(f"[nih_vqa_grpo:{split}] rows={len(ds)}, cols={ds.column_names}")

    # Sanity check: columns must be exactly these five
    for split, ds in dsd.items():
        assert ds.column_names == ["image", "problem", "solution", "original_question", "original_answer"], \
            f"{split} columns are {ds.column_names}, expected ['image', 'problem', 'solution', 'original_question', 'original_answer']"

    return dsd





@register_dataset("medbullets_op4")
def load_medbullets_op4(
    num_proc: int = 1,
    batch_size: int = 2048,
    parquet_paths: dict | None = None,
    debug_limit: int | None = None,
) -> DatasetDict:
    """
    Text-only MedBullets op4 loader:
      - reads parquet shards per split
      - builds `messages` (user -> assistant)
      - keeps `images` as [None] per sample for schema consistency
      - merges answer_idx with answer (e.g., "C: <answer>")
    """
    import ast
    import json
    import pandas as pd
    from datasets import Dataset, DatasetDict

    if parquet_paths is None:
        parquet_paths = {
            "train": "./Document/MedEvalKit/datas/Medical-Eval-MedBullets_op4/data/train-00000-of-00001.parquet",
            # "validation": "/path/to/val.parquet",
            # "test": "/path/to/test.parquet",
        }

    def _coerce_options(val):
        if val is None:
            return {}
        if isinstance(val, dict):
            return val
        if isinstance(val, list):
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            return {letters[i]: v for i, v in enumerate(val) if i < len(letters)}
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return {}
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                pass
            try:
                obj = ast.literal_eval(s)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        return {}

    def _norm_idx(idx):
        if idx is None:
            return ""
        if isinstance(idx, bool):
            return ""
        if isinstance(idx, (int, float)):
            i = int(idx)
            if 1 <= i <= 26:
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i - 1]
            if 0 <= i < 26:
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
            return str(i)
        s = str(idx).strip()
        if not s:
            return ""
        if s.isdigit():
            i = int(s)
            if 1 <= i <= 26:
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i - 1]
        return s.upper()

    def _read_split(split: str) -> Dataset:
        if split not in parquet_paths:
            return Dataset.from_dict({"messages": [], "images": []})
        df = pd.read_parquet(parquet_paths[split], engine="pyarrow")
        if debug_limit:
            df = df.head(debug_limit)

        # required columns
        for col in ("question", "options", "answer", "answer_idx"):
            if col not in df.columns:
                raise ValueError(f"[medbullets_op4:{split}] missing required column '{col}'")

        ds = Dataset.from_pandas(df, preserve_index=False)

        def _format(batch):
            msgs, imgs = [], []
            qs = batch["question"]
            opts = batch["options"]
            ans = batch["answer"]
            idxs = batch["answer_idx"]
            n = len(qs)

            for i in range(n):
                question = (qs[i] or "").strip()
                options = _coerce_options(opts[i])
                options_text = "\n".join(f"{k}. {v}" for k, v in options.items())

                user_q = (
                    "## Instruction: Choose the correct option based on the question below.\n\n"
                    f"{question}\n\n## Options:\n{options_text}"
                )

                idx = _norm_idx(idxs[i])
                answer_text = (ans[i] or "").strip()
                if not answer_text and idx:
                    answer_text = (options.get(idx, "") or "").strip()

                if idx:
                    final_answer = f"{idx}: {answer_text}".strip()
                else:
                    # try to map by answer text if idx missing
                    key = next((k for k, v in options.items() if (v or "").strip() == answer_text), "")
                    final_answer = f"{key}: {answer_text}".strip() if key else answer_text

                user = {"role": "user", "content": [{"type": "text", "text": user_q, "index": None}]}
                asst = {"role": "assistant", "content": [{"type": "text", "text": final_answer, "index": None}]}
                msgs.append([user, asst])
                imgs.append([None])  # text-only schema

            return {"messages": msgs, "images": imgs}

        ds = ds.map(_format, batched=True, batch_size=batch_size, num_proc=num_proc,
                    desc=f"[medbullets_op4:{split}] format")

        keep = {"messages", "images"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds

    out = {sp: _read_split(sp) for sp in ("train", "validation")}
    return DatasetDict(out)







import os, hashlib
from typing import List, Any, Tuple

import pandas as pd
from PIL import Image, ImageOps
from datasets import Dataset, DatasetDict, Features, Value
from datasets import Image as HFImage




@register_dataset("nih_bbox_grpo")
def load_nih_bbox_grpo(
    num_proc: int = 1,
    batch_size: int = 1024,
    image_root: str = "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/images",
    split_csv: dict | None = None,
    check_files: bool = False,
    debug_limit: int | None = None,
    verbose: bool = False,
) -> DatasetDict:
    """
    NIH ChestX-ray14 BBoxes → GRPO-style DatasetDict (NO RESIZING).
    Returns original image paths as list and original bounding boxes as-is.
    
    Final schema: ['image','problem','solution','original_question','original_answer']
    """

    if split_csv is None:
        split_csv = {
            "train": "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/BBox_List_2017.csv",
            "validation": "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/BBox_List_2017_val.csv",
            "test": "./Medmo_Dataset_1/Medmo_Dataset/NIH-Data/BBox_List_2017_val1.csv",
        }

    PROMPTS: List[str] = [
        ("You are a medical imaging expert. "
         "Detect any visible diseases or abnormalities in this X-ray image. "
         "For each detected finding, provide: "
         "the disease name/names, and "
         "bounding box coordinates as: <box> [x_min, y_min, x_max, y_max] </box> in pixel values.")
    ]

    def _det_prompt(uid_val: Any, idx: int) -> str:
        key = str(uid_val) if uid_val is not None else str(idx)
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        return PROMPTS[h % len(PROMPTS)]

    # Column normalization
    RENAME = {
        "Image Index": "filename",
        "Finding Label": "label",
        "Bbox [x": "x", "y": "y", "w": "w", "h]": "h",
        "Patient ID": "patient_id",
    }

    def _format_solution(labels: List[str], boxes: List[List[int]]) -> str:
        """Format solution with original boxes (no scaling)"""
        if not labels or not boxes or len(labels) != len(boxes):
            return "<answer></answer>"
        lines = [f"{lab}: <box> {x1}, {y1}, {x2}, {y2} </box>"
                 for lab, (x1, y1, x2, y2) in zip(labels, boxes)]
        return "<answer>\n" + "\n".join(lines) + "\n</answer>"

    def _orig_answer_compact(labels: List[str], boxes: List[List[int]]) -> str:
        """Compact representation for audit"""
        pairs = [{"label": lab, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                 for lab, b in zip(labels or [], boxes or [])]
        return str(pairs)

    def _load_and_group(csv_path: str) -> pd.DataFrame:
        """Load CSV and group by image filename"""
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[nih_bbox_grpo] Missing CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)

        # Rename columns
        for old, new in RENAME.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

        # Validate required columns
        required = {"filename", "label", "x", "y", "w", "h"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"[nih_bbox_grpo] CSV missing required columns: {missing}")

        # Convert to numeric
        for c in ["x", "y", "w", "h"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # Calculate x1, y1, x2, y2 from x, y, w, h
        df["x1"] = df["x"].astype(int)
        df["y1"] = df["y"].astype(int)
        df["x2"] = (df["x"] + df["w"]).astype(int)
        df["y2"] = (df["y"] + df["h"]).astype(int)

        # Build full path
        df["full_path"] = df["filename"].astype(str).str.lstrip("/").apply(
            lambda n: os.path.normpath(os.path.join(image_root, n))
        )

        # Clean labels
        df["label"] = df["label"].astype(str).str.strip()
        df = df[df["label"] != ""]

        # Group by image - collect all boxes per image
        agg_dict = {
            "label": lambda x: x.tolist(),
            "x1": lambda x: x.tolist(),
            "y1": lambda x: x.tolist(),
            "x2": lambda x: x.tolist(),
            "y2": lambda x: x.tolist(),
        }
        
        # Add patient_id only if it exists
        if "patient_id" in df.columns:
            agg_dict["patient_id"] = "first"
        
        grouped = (
            df.groupby("full_path", as_index=False)
              .agg(agg_dict)
              .rename(columns={"label": "labels"})
        )
        
        # Combine coordinate columns into boxes list
        grouped["boxes"] = grouped.apply(
            lambda row: [[x1, y1, x2, y2] for x1, y1, x2, y2 in 
                        zip(row["x1"], row["y1"], row["x2"], row["y2"])],
            axis=1
        )
        
        # Add patient_id if it doesn't exist
        if "patient_id" not in grouped.columns:
            grouped["patient_id"] = None
        else:
            grouped["patient_id"] = grouped["patient_id"].astype(str)
        
        # Clean up intermediate columns - keep only what we need
        grouped = grouped[["full_path", "labels", "boxes", "patient_id"]]

        # Debug limit
        if debug_limit is not None and debug_limit > 0 and len(grouped) > debug_limit:
            grouped = grouped.iloc[:debug_limit].copy()

        # Check file existence (optional, slow on network FS)
        if check_files:
            grouped = grouped[grouped["full_path"].apply(os.path.exists)].reset_index(drop=True)

        return grouped

    def _build_split(split: str) -> Dataset:
        """Build dataset for a single split (vectorized, no map)"""
        gdf = _load_and_group(split_csv[split])

        # Pre-allocate lists
        data = {
            "image": [],
            "problem": [],
            "solution": [],
            "original_question": [],
            "original_answer": []
        }

        # Process all rows (vectorized - no map needed)
        for i, row in gdf.iterrows():
            img_path = row["full_path"]
            labs: List[str] = row["labels"]
            bxs: List[List[int]] = row["boxes"]
            uid = row.get("patient_id", img_path)

            # Generate prompt
            q = _det_prompt(uid, i)
            problem = f"{q}\n"

            # Format solution with ORIGINAL boxes (no resizing)
            solution = _format_solution(labs, bxs)
            original_answer = _orig_answer_compact(labs, bxs)

            # Append to lists - IMAGE AS LIST
            data["image"].append([img_path])  # Changed: wrap in list
            data["problem"].append(problem)
            data["solution"].append(solution)
            data["original_question"].append(q)
            data["original_answer"].append(original_answer)

        # Create dataset directly from dict
        from datasets import Sequence
        
        features = Features({
            "image": Sequence(Value("string")),  # Changed: Sequence of strings
            "problem": Value("string"),
            "solution": Value("string"),
            "original_question": Value("string"),
            "original_answer": Value("string"),
        })

        ds = Dataset.from_dict(data, features=features)
        
        if verbose:
            print(f"[nih_bbox_grpo:{split}] Loaded {len(ds)} samples")

        return ds

    # Build all splits
    dsd = DatasetDict({
        "train": _build_split("train"),
        "validation": _build_split("validation"),
        "test": _build_split("test"),
    })

    if verbose:
        for split, ds in dsd.items():
            print(f"[nih_bbox_grpo:{split}] rows={len(ds)}, cols={ds.column_names}")

    # Sanity check: columns must be exactly these five
    for split, ds in dsd.items():
        assert ds.column_names == ["image", "problem", "solution", "original_question", "original_answer"], \
            f"{split} columns are {ds.column_names}, expected ['image', 'problem', 'solution', 'original_question', 'original_answer']"

    return dsd


@register_dataset("deeplesion_bbox_grpo")
def load_deeplesion_bbox_grpo(
    num_proc: int = 1,
    batch_size: int = 2048,
    image_root: str = "./Medmo_Dataset_1/Medmo_Dataset/DeepLesion/Images_png",
    split_csv: dict | None = None,
    check_files: bool = False,
    debug_limit: int | None = None,
    verbose: bool = False,
) -> DatasetDict:
    """
    DeepLesion CSV -> GRPO-style DatasetDict with EXACT schema:
      ['image','problem','solution','original_question','original_answer']
    
    Returns image paths as lists. NO resizing of images or boxes.
    Boxes are kept in original pixel coordinates.
    """

    # ---------------- prompts ----------------
    PROMPTS: List[str] = [
        ("You are a medical imaging expert. "
         "Detect lesions or abnormalities in this CT slice. "
         "For each finding, provide the label and bounding box as: "
         "<box> [x_min, y_min, x_max, y_max] </box> in pixel values.")
    ]

    def _det_prompt(uid_val: Any, idx: int) -> str:
        key = str(uid_val) if uid_val is not None else str(idx)
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        return PROMPTS[h % len(PROMPTS)]

    # ------------- formatting helpers -------------
    def _format_solution(labels: List[str], boxes: List[List[int]]) -> str:
        """Format solution with original boxes (no scaling)"""
        if not labels or not boxes or len(labels) != len(boxes):
            return "<answer></answer>"
        lines = [f"{lab}: <box> {x1}, {y1}, {x2}, {y2} </box>"
                 for lab, (x1, y1, x2, y2) in zip(labels, boxes)]
        return "<answer>\n" + "\n".join(lines) + "\n</answer>"

    def _orig_answer_compact(labels: List[str], boxes: List[List[int]]) -> str:
        """Compact representation for audit"""
        pairs = [{"label": lab, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                 for lab, b in zip(labels or [], boxes or [])]
        return str(pairs)

    # ---------------- CSV defaults ----------------
    if split_csv is None:
        split_csv = {
            "train": "./Medmo_Dataset_1/Medmo_Dataset/DeepLesion/DL_info_train.csv",
            "validation": "./Medmo_Dataset_1/Medmo_Dataset/DeepLesion/DL_info_val.csv",
        }

    # ------------- parsing helpers -------------
    def _mk_full_path(fn: str) -> str:
        """Build full path from File_name"""
        s = str(fn).split("_")
        if len(s) < 4:
            return os.path.normpath(os.path.join(image_root, str(fn)))
        dir_part = "_".join(s[:3])
        leaf = s[3]
        return os.path.normpath(os.path.join(image_root, dir_part, leaf))

    def _parse_boxes(s: str) -> List[List[int]]:
        """
        Parse bounding boxes from string.
        Accepts: "x1,y1,x2,y2" or multi-box strings separated by ';' or '|'
        """
        if s is None or str(s).strip() == "":
            return []
        
        # Split by ; or | if present
        if ";" in str(s):
            segs = [seg.strip() for seg in str(s).split(";")]
        elif "|" in str(s):
            segs = [seg.strip() for seg in str(s).split("|")]
        else:
            segs = [str(s).strip()]
        
        boxes: List[List[int]] = []
        for seg in segs:
            parts = [p.strip() for p in seg.split(",")]
            if len(parts) != 4:
                continue
            try:
                x1, y1, x2, y2 = [int(float(t)) for t in parts]
                boxes.append([x1, y1, x2, y2])
            except Exception:
                continue
        return boxes

    # ------------- load & group per split -------------
    def _load_and_group(csv_path: str) -> pd.DataFrame:
        """Load CSV and group by image filename"""
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[deeplesion_bbox_grpo] missing CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)

        # Validate required columns
        required = {"File_name", "Bounding_boxes"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"[deeplesion_bbox_grpo] CSV missing columns: {missing}")

        # Build full paths and parse boxes
        df["full_path"] = df["File_name"].astype(str).apply(_mk_full_path)
        df["boxes_list"] = df["Bounding_boxes"].apply(_parse_boxes)

        # Explode multi-box rows (one row per box initially)
        df = df.explode("boxes_list", ignore_index=True)
        df = df[~df["boxes_list"].isna()].reset_index(drop=True)

        # Group by image - collect all boxes per image
        grouped = (
            df.groupby("full_path", as_index=False)
              .agg({
                  "boxes_list": lambda x: x.tolist(),
              })
              .rename(columns={"boxes_list": "boxes"})
        )
        
        # Add labels (generic "Lesion" for all boxes)
        grouped["labels"] = grouped["boxes"].apply(lambda bxs: ["Lesion"] * len(bxs))
        grouped["uid"] = grouped["full_path"]

        # Debug limit
        if debug_limit is not None and debug_limit > 0 and len(grouped) > debug_limit:
            grouped = grouped.iloc[:debug_limit].copy()

        # Optional file existence check
        if check_files:
            grouped = grouped[grouped["full_path"].apply(os.path.exists)].reset_index(drop=True)

        return grouped

    # ------------- build split -------------
    def _build_split(split: str) -> Dataset:
        """Build dataset for a single split (vectorized, no map)"""
        if split not in split_csv or split_csv[split] is None:
            raise FileNotFoundError(f"[deeplesion_bbox_grpo:{split}] missing csv path")
        
        gdf = _load_and_group(split_csv[split])

        # Pre-allocate lists
        data = {
            "image": [],
            "problem": [],
            "solution": [],
            "original_question": [],
            "original_answer": []
        }

        # Process all rows (vectorized - no map needed)
        for i, row in gdf.iterrows():
            img_path = row["full_path"]
            labs: List[str] = row["labels"]
            bxs: List[List[int]] = row["boxes"]
            uid = row["uid"]

            # Generate prompt
            q = _det_prompt(uid, i)
            problem = f"{q}\n"

            # Format solution with ORIGINAL boxes (no resizing)
            solution = _format_solution(labs, bxs)
            original_answer = _orig_answer_compact(labs, bxs)

            # Append to lists - IMAGE AS LIST
            data["image"].append([img_path])  # Changed: wrap in list
            data["problem"].append(problem)
            data["solution"].append(solution)
            data["original_question"].append(q)
            data["original_answer"].append(original_answer)

        # Create dataset directly from dict
        from datasets import Sequence
        
        features = Features({
            "image": Sequence(Value("string")),  # Changed: list of image paths
            "problem": Value("string"),
            "solution": Value("string"),
            "original_question": Value("string"),
            "original_answer": Value("string"),
        })

        ds = Dataset.from_dict(data, features=features)
        
        if verbose:
            print(f"[deeplesion_bbox_grpo:{split}] Loaded {len(ds)} samples")

        return ds

    # -------- assemble splits --------
    dsd = DatasetDict({
        "train": _build_split("train"),
        "validation": _build_split("validation"),
    })

    if verbose:
        for split, ds in dsd.items():
            print(f"[deeplesion_bbox_grpo:{split}] rows={len(ds)}, cols={ds.column_names}")

    # Sanity check: columns must be exactly these five
    for split, ds in dsd.items():
        assert ds.column_names == ["image", "problem", "solution", "original_question", "original_answer"], \
            f"{split} columns are {ds.column_names}, expected ['image', 'problem', 'solution', 'original_question', 'original_answer']"

    return dsd



@register_dataset("mimic_cxr_vqa_grpo")
def load_mimic_cxr_vqa_grpo(
    num_proc: int = 8,
    batch_size: int = 128,
    json_paths: dict | None = None,
    image_root: str = "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.1.0/files",
    debug_limit: int | dict | None = None,
    verbose: bool = False,
) -> DatasetDict:
    """
    MIMIC-CXR-VQA → GRPO-style DatasetDict.
    Returns image paths as lists. NO resizing.

    Final schema: ['image','problem','solution','original_question','original_answer']
    """

    # ---------------- prompts ----------------
    # MIMIC-CXR is VQA, so prompts are questions from the dataset itself
    # We'll use the questions directly from the data

    # ------------- formatting -------------
    def _format_solution(answer: str) -> str:
        """Format solution with answer text"""
        if not answer or answer.strip() == "":
            return "<answer></answer>"
        return f"<answer>\n{answer.strip()}\n</answer>"

    def _orig_answer_compact(answer: str) -> str:
        """Compact representation for audit"""
        return str({"answer": answer})

    # ------------- parsing helpers -------------
    def _strip_image_tokens(s: str) -> str:
        """Remove image tokens from text"""
        if not isinstance(s, str):
            return ""
        # Remove variants like "<image>", "\n<image>", case-insensitive
        return re.sub(r"\s*<\s*image\s*>\s*", " ", s, flags=re.IGNORECASE).strip()

    def _normalize_answer(ans) -> str:
        """Normalize answer to string"""
        if ans is None:
            return ""
        if isinstance(ans, list):
            return ", ".join(map(str, ans))
        return str(ans)

    def _apply_debug(df: pd.DataFrame, split: str) -> pd.DataFrame:
        """Apply debug limit to dataframe"""
        if debug_limit is None:
            return df
        n = debug_limit.get(split) if isinstance(debug_limit, dict) else debug_limit
        if n is None:
            return df
        try:
            n = int(n)
        except Exception:
            return df
        if n <= 0:
            return df.head(0).reset_index(drop=True)
        return df.head(min(n, len(df))).reset_index(drop=True)

    # ---------------- default split paths ----------------
    SPLIT_PATHS = {
        "train": "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-VQA/train_clean_full.json",
        "validation": "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-VQA/valid_clean.json",
    }
    if json_paths is not None:
        SPLIT_PATHS = json_paths

    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"[mimic_cxr_vqa_grpo] Image root not found: {image_root}")

    # ------------- build split -------------
    def _build_split(split: str, path: str) -> Dataset:
        """Build dataset for a single split (vectorized, no map)"""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[mimic_cxr_vqa_grpo] Missing JSON for split '{split}': {path}")

        # Robust JSON load: file may be a list OR {"data": [...]}
        with open(path, "r") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "data" in obj:
            records = obj["data"]
        elif isinstance(obj, list):
            records = obj
        else:
            raise ValueError(f"[mimic_cxr_vqa_grpo:{split}] Unexpected JSON structure in {path}")

        df = pd.DataFrame(records)

        # Required columns
        if "image_path" not in df.columns or "question" not in df.columns:
            raise ValueError(f"[mimic_cxr_vqa_grpo:{split}] Expected 'image_path' and 'question' columns in {path}")

        # Apply debug limit
        df = _apply_debug(df, split)

        if len(df) == 0:
            raise ValueError(f"[mimic_cxr_vqa_grpo:{split}] 0 rows after parsing/sampling {path}")

        # Pre-allocate lists
        data = {
            "image": [],
            "problem": [],
            "solution": [],
            "original_question": [],
            "original_answer": []
        }

        # Process all rows (vectorized - no map needed)
        for i, row in df.iterrows():
            # Build absolute image path
            rel_path = row.get("image_path", "")
            img_path = os.path.join(image_root, str(rel_path))

            # Extract question and answer
            question = _strip_image_tokens(row.get("question", "") or "")
            answer = _normalize_answer(row.get("answer"))

            if not question:
                question = "Describe this chest X-ray image."

            # Generate problem (question becomes the problem)
            problem = f"{question}\n"

            # Format solution
            solution = _format_solution(answer)
            original_answer = _orig_answer_compact(answer)

            # Append to lists - IMAGE AS LIST
            data["image"].append([img_path])  # Wrap in list
            data["problem"].append(problem)
            data["solution"].append(solution)
            data["original_question"].append(question)
            data["original_answer"].append(original_answer)

        # Create dataset directly from dict
        from datasets import Sequence
        
        features = Features({
            "image": Sequence(Value("string")),  # List of image paths
            "problem": Value("string"),
            "solution": Value("string"),
            "original_question": Value("string"),
            "original_answer": Value("string"),
        })

        ds = Dataset.from_dict(data, features=features)
        
        if verbose:
            print(f"[mimic_cxr_vqa_grpo:{split}] Loaded {len(ds)} samples")

        return ds

    # -------- assemble splits --------
    dsd = DatasetDict({
        split: _build_split(split, path)
        for split, path in SPLIT_PATHS.items()
    })

    if verbose:
        for split, ds in dsd.items():
            print(f"[mimic_cxr_vqa_grpo:{split}] rows={len(ds)}, cols={ds.column_names}")

    # Sanity check: columns must be exactly these five
    for split, ds in dsd.items():
        assert ds.column_names == ["image", "problem", "solution", "original_question", "original_answer"], \
            f"{split} columns are {ds.column_names}, expected ['image', 'problem', 'solution', 'original_question', 'original_answer']"

    return dsd


# --------------------------
# VQA test loaders → GRPO adapters (no bbox)
# --------------------------
@register_dataset("PATH_VQA_test_grpo")
def load_path_vqa_test_grpo(
    split: str = "test",
    format_batch_size: int = 1024,
    format_num_proc: int = 1,
    wrap_answer_in_tags: bool = True,
    **base_kwargs,
) -> DatasetDict:
    ds_dict = load_path_vqa(split=split, **base_kwargs)
    return _messages_datasetdict_to_grpo(
        ds_dict,
        dataset_label="PATH_VQA_test_grpo",
        default_prompt=_default_text_grpo_prompt("PATH_VQA_test_grpo"),
        fmt_batch_size=format_batch_size,
        fmt_num_proc=format_num_proc,
        wrap_answer=wrap_answer_in_tags,
    )










@register_dataset("bacteria_bbox_grpo")
def load_bacteria_bbox_grpo(
    num_proc: int = 1,
    batch_size: int = 1024,
    image_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Cell_Data/preproceed/Bacteria/images",
    label_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Cell_Data/preproceed/Bacteria/labels",
    max_boxes: int = 40,
    check_files: bool = False,
    debug_limit: int | None = None,
    verbose: bool = False,
) -> DatasetDict:
    """
    YOLO .txt (class xc yc w h normalized) -> GRPO-style DatasetDict.
    Returns image paths as lists. NO resizing of images or boxes.
    Boxes are converted to original pixel coordinates.

    Final schema: ['image','problem','solution','original_question','original_answer']
    """

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    # ---------------- prompts ----------------
    PROMPTS: List[str] = [
        ("You are a microscopy expert. Detect and localize all bacterial cells. "
         "For each cell, provide: <box> [ x_min, y_min, x_max, y_max ], [ x_min, y_min, x_max, y_max ],... </box> in pixel values."),
    ]

    def _det_prompt(uid_val: Any, idx: int) -> str:
        key = str(uid_val) if uid_val is not None else str(idx)
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        return PROMPTS[h % len(PROMPTS)]

    # ------------- formatting -------------
    def _format_solution(label: str, boxes: List[List[int]]) -> str:
        """Format solution with original boxes (no scaling)"""
        if not label or not boxes:
            return "<answer></answer>"
        lines = [f"[ {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)} ]" 
                 for x1, y1, x2, y2 in boxes]
        return "<answer>\n" + f"{label}: <box> " + " ".join(lines) + " </box>\n</answer>"

    def _orig_answer_compact(labels: List[str], boxes: List[List[int]]) -> str:
        """Compact representation for audit"""
        pairs = [{"label": lab, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                 for lab, b in zip(labels or [], boxes or [])]
        return str(pairs)

    # ------------- IO helpers -------------
    CLASS_MAP = {0: "Bacteria"}  # extend if you have multiple classes

    def _parse_yolo_file(txt_path: Path) -> List[List[float]]:
        """
        Returns list of [cls_id, xc, yc, w, h] (normalized units).
        Accepts lines with >=5 tokens: class xc yc w h [conf...]
        """
        out: List[List[float]] = []
        try:
            with open(txt_path, "r") as f:
                for ln in f:
                    parts = ln.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(float(parts[0]))
                        xc, yc, w, h = map(float, parts[1:5])
                        out.append([cls, xc, yc, w, h])
                    except Exception:
                        continue
        except FileNotFoundError:
            pass
        return out

    def _gather_table() -> pd.DataFrame:
        """Collect all image-label pairs"""
        rows = []
        for root, _, files in os.walk(image_dir):
            for fn in sorted(files):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = Path(root) / fn
                rel = img_path.relative_to(image_dir)
                lbl_path = label_dir / rel.with_suffix(".txt")
                if check_files and not img_path.exists():
                    continue
                rows.append({
                    "uid": rel.as_posix(),
                    "img_path": str(img_path),
                    "lbl_path": str(lbl_path),
                })
        df = pd.DataFrame(rows)
        if debug_limit and len(df) > debug_limit:
            df = df.iloc[:debug_limit].copy()
        return df

    # ------------- build split -------------
    def _build_split() -> Dataset:
        """Build dataset (vectorized, no map)"""
        gdf = _gather_table()

        # Pre-allocate lists
        data = {
            "image": [],
            "problem": [],
            "solution": [],
            "original_question": [],
            "original_answer": []
        }

        # Process all rows (vectorized - no map needed)
        for i, row in gdf.iterrows():
            uid = row["uid"]
            img_path = row["img_path"]
            lbl_path = row["lbl_path"]
            
            # Parse YOLO annotations
            yolo_list = _parse_yolo_file(Path(lbl_path))
            
            # Get native image size to convert normalized coords to pixels
            with Image.open(img_path) as im:
                W, H = im.size

            # Convert YOLO normalized coords to pixel coords
            labels: List[str] = []
            orig_boxes: List[List[int]] = []
            
            for cls, xc, yc, w, h in (yolo_list or []):
                # Convert from normalized center coords to pixel corner coords
                x1 = int(round((xc - w/2) * W))
                y1 = int(round((yc - h/2) * H))
                x2 = int(round((xc + w/2) * W))
                y2 = int(round((yc + h/2) * H))
                
                # Clamp to image bounds
                x1 = max(0, min(W-1, x1))
                x2 = max(0, min(W-1, x2))
                y1 = max(0, min(H-1, y1))
                y2 = max(0, min(H-1, y2))
                
                # Ensure correct ordering
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                
                orig_boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_MAP.get(int(cls), "Bacteria"))

            # Optionally drop dense samples to keep generation manageable.
            if len(orig_boxes) > max_boxes:
                continue

            # Generate prompt
            q = _det_prompt(uid, i)
            problem = f"{q}\n"

            # Format solution with ORIGINAL boxes (no resizing)
            if labels and orig_boxes:
                solution = _format_solution(labels[0], orig_boxes)
                original_answer = _orig_answer_compact(labels, orig_boxes)
            else:
                solution = "<answer></answer>"
                original_answer = "[]"

            # Append to lists - IMAGE AS LIST
            data["image"].append([img_path])  # Changed: wrap in list
            data["problem"].append(problem)
            data["solution"].append(solution)
            data["original_question"].append(q)
            data["original_answer"].append(original_answer)

        # Create dataset directly from dict
        from datasets import Sequence
        
        features = Features({
            "image": Sequence(Value("string")),  # Changed: list of image paths
            "problem": Value("string"),
            "solution": Value("string"),
            "original_question": Value("string"),
            "original_answer": Value("string"),
        })

        ds = Dataset.from_dict(data, features=features)
        
        if verbose:
            print(f"[bacteria_bbox_grpo] Loaded {len(ds)} samples")

        return ds

    # --------- assemble splits ----------
    ds = _build_split()
    ds_train, ds_val, ds_test = _split_dataset_tail_disjoint(ds, min(2000, len(ds)))
    
    dsd = DatasetDict({
        "train": ds_train,
        "validation": ds_val,
        "test": ds_test,
    })

    if verbose:
        for split, s in dsd.items():
            print(f"[bacteria_bbox_grpo:{split}] rows={len(s)}, cols={s.column_names}")

    # Sanity check: columns must be exactly these five
    for split, s in dsd.items():
        assert s.column_names == ["image", "problem", "solution", "original_question", "original_answer"], \
            f"{split} columns are {s.column_names}, expected ['image', 'problem', 'solution', 'original_question', 'original_answer']"

    return dsd






@register_dataset("ctc_bbox_grpo")
def load_ctc_bbox_grpo(
    num_proc: int = 1,
    batch_size: int = 1024,
    base_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Cell_Data/preproceed/CTCDataset",
    debug_limit: int | None = None,
    verbose: bool = False,
) -> DatasetDict:
    """
    CTC YOLO labels → GRPO-style DatasetDict.
    Returns image paths as lists. NO resizing of images or boxes.
    Boxes are converted to original pixel coordinates.

    Final schema: ['image','problem','solution','original_question','original_answer']
    """

    # ---------------- prompts ----------------
    PROMPTS: List[str] = [
        ("You are a microscopy expert. Detect and localize all cells. "
         "For each finding, provide the format must: Cell: <box> [ x_min, y_min, x_max, y_max ], [ x_min, y_min, x_max, y_max ], ... </box>, in pixel values."),
        ("Identify and localize every cell. Respond as the format must: Cell: <box> [ x_min, y_min, x_max, y_max ] ... </box>, in pixel values."),
        ("Locate all visible cells and return the format must: Cell: <box> [ x_min, y_min, x_max, y_max ] ... </box>, in pixel values."),
    ]

    def _det_prompt(uid_val: Any, idx: int) -> str:
        key = str(uid_val) if uid_val is not None else str(idx)
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        return PROMPTS[h % len(PROMPTS)]

    # ------------- formatting -------------
    def _format_solution_one_label(label: str, boxes: List[List[int]]) -> str:
        """Format solution with original boxes (no scaling)"""
        if not label or not boxes:
            return "<answer></answer>"
        chunks = [f"[ {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)} ]" 
                  for x1, y1, x2, y2 in boxes]
        return "<answer>\n" + f"{label}: <box> " + " ".join(chunks) + " </box>\n</answer>"

    def _orig_answer_compact(labels: List[str], boxes: List[List[int]]) -> str:
        """Compact representation for audit"""
        pairs = [{"label": lab, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                 for lab, b in zip(labels or [], boxes or [])]
        return str(pairs)

    # ------------- IO helpers -------------
    def _parse_yolo_file(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """
        Returns list of (cls_id, xc, yc, w, h) in NORMALIZED units.
        Accepts lines with >=5 tokens: class xc yc w h [conf...]
        """
        out: List[Tuple[int, float, float, float, float]] = []
        if not txt_path.exists():
            return out
        with open(txt_path, "r") as f:
            for ln in f:
                parts = ln.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                    out.append((cls, xc, yc, w, h))
                except Exception:
                    continue
        return out

    def _gather_table(base: Path) -> pd.DataFrame:
        """Collect all image-label pairs from CTC dataset structure"""
        rows = []
        # Scan */images/{train,val} with matching */labels/{train,val}
        for ds_name in sorted(os.listdir(base)):
            ds_path = base / ds_name
            if not ds_path.is_dir():
                continue
            img_root, lbl_root = ds_path / "images", ds_path / "labels"
            for split in ("train", "val"):
                img_dir, lbl_dir = img_root / split, lbl_root / split
                if not img_dir.is_dir():
                    continue
                for fn in sorted(os.listdir(img_dir)):
                    if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")):
                        continue
                    img_path = img_dir / fn
                    lbl_path = lbl_dir / (Path(fn).stem + ".txt")
                    if lbl_path.exists():
                        rows.append({
                            "uid": f"{ds_name}/{split}/{fn}",
                            "img_path": str(img_path),
                            "lbl_path": str(lbl_path),
                        })
        df = pd.DataFrame(rows)
        if debug_limit and len(df) > debug_limit:
            df = df.iloc[:debug_limit].copy()
        return df

    # ------------- build split -------------
    def _build_split() -> Dataset:
        """Build dataset (vectorized, no map)"""
        base = Path(base_dir)
        gdf = _gather_table(base)

        # Pre-allocate lists
        data = {
            "image": [],
            "problem": [],
            "solution": [],
            "original_question": [],
            "original_answer": []
        }

        # Process all rows (vectorized - no map needed)
        for i, row in gdf.iterrows():
            uid = row["uid"]
            img_path = row["img_path"]
            lbl_path = row["lbl_path"]
            
            # Get native image size to convert normalized coords to pixels
            with Image.open(img_path) as im:
                W, H = im.size

            # Parse YOLO annotations and convert to pixel coords
            labels: List[str] = []
            orig_boxes: List[List[int]] = []
            
            for cls, xc, yc, w, h in _parse_yolo_file(Path(lbl_path)):
                # Convert from normalized center coords to pixel corner coords
                x1 = int(round((xc - w/2) * W))
                y1 = int(round((yc - h/2) * H))
                x2 = int(round((xc + w/2) * W))
                y2 = int(round((yc + h/2) * H))
                
                # Clamp to image bounds
                x1 = max(0, min(W-1, x1))
                x2 = max(0, min(W-1, x2))
                y1 = max(0, min(H-1, y1))
                y2 = max(0, min(H-1, y2))
                
                # Ensure correct ordering
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                
                orig_boxes.append([x1, y1, x2, y2])
                labels.append("Cell")

            # Generate prompt
            q = _det_prompt(uid, i)
            problem = f"{q}\n"

            # Format solution with ORIGINAL boxes (no resizing)
            if labels and orig_boxes:
                solution = _format_solution_one_label("Cell", orig_boxes)
                original_answer = _orig_answer_compact(labels, orig_boxes)
            else:
                solution = "<answer></answer>"
                original_answer = "[]"

            # Append to lists - IMAGE AS LIST
            data["image"].append([img_path])  # Changed: wrap in list
            data["problem"].append(problem)
            data["solution"].append(solution)
            data["original_question"].append(q)
            data["original_answer"].append(original_answer)

        # Create dataset directly from dict
        from datasets import Sequence
        
        features = Features({
            "image": Sequence(Value("string")),  # Changed: list of image paths
            "problem": Value("string"),
            "solution": Value("string"),
            "original_question": Value("string"),
            "original_answer": Value("string"),
        })

        ds = Dataset.from_dict(data, features=features)
        
        if verbose:
            print(f"[ctc_bbox_grpo] Loaded {len(ds)} samples")

        return ds

    # --------- assemble splits ----------
    ds = _build_split()
    ds_train, ds_val, ds_test = _split_dataset_tail_disjoint(ds, min(2000, len(ds)))
    
    dsd = DatasetDict({
        "train": ds_train,
        "validation": ds_val,
        "test": ds_test,
    })

    if verbose:
        for split, s in dsd.items():
            print(f"[ctc_bbox_grpo:{split}] rows={len(s)}, cols={s.column_names}")

    # Sanity check: columns must be exactly these five
    for split, s in dsd.items():
        assert s.column_names == ["image", "problem", "solution", "original_question", "original_answer"], \
            f"{split} columns are {s.column_names}, expected ['image', 'problem', 'solution', 'original_question', 'original_answer']"

    return dsd





@register_dataset("deepcell_bbox_grpo")
def load_deepcell_bbox_grpo(
    num_proc: int = 1,
    batch_size: int = 1024,
    base_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Cell_Data/preproceed/DEEPCELL",
    split: str = "train",
    debug_limit: int | None = None,
    verbose: bool = False,
) -> DatasetDict:
    """
    DeepCell YOLO labels → GRPO-style DatasetDict.
    Returns image paths as lists. NO resizing of images or boxes.
    Boxes are converted to original pixel coordinates.

    Final schema: ['image','problem','solution','original_question','original_answer']
    """
    
    SPLIT_MAP = {"train": "00", "validation": "01", "test": "02"}
    assert split in SPLIT_MAP, f"split must be one of {list(SPLIT_MAP)}"
    sid = SPLIT_MAP[split]

    # ---------------- prompts ----------------
    PROMPTS: List[str] = [
        ("You are a microscopy expert. Detect and localize all cells. "
         "For each finding, provide the format must: Cell: <box> [ x_min, y_min, x_max, y_max ], [ x_min, y_min, x_max, y_max ], ... </box> in pixel values."),
        ("Identify and localize every cell. Respond as the format must: Cell: <box> [ x_min, y_min, x_max, y_max ] ... </box> in pixel values."),
        ("Locate all visible cells and return the format must: Cell: <box> [ x_min, y_min, x_max, y_max ] ... </box> in pixel values."),
    ]

    def _det_prompt(uid_val: Any, idx: int) -> str:
        key = str(uid_val) if uid_val is not None else str(idx)
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        return PROMPTS[h % len(PROMPTS)]

    # ------------- formatting -------------
    def _format_solution_one_label(label: str, boxes: List[List[int]]) -> str:
        """Format solution with original boxes (no scaling)"""
        if not label or not boxes:
            return "<answer></answer>"
        chunks = [f"[ {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)} ]" 
                  for x1, y1, x2, y2 in boxes]
        return "<answer>\n" + f"{label}: <box> " + " ".join(chunks) + " </box>\n</answer>"

    def _orig_answer_compact(labels: List[str], boxes: List[List[int]]) -> str:
        """Compact representation for audit"""
        pairs = [{"label": lab, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                 for lab, b in zip(labels or [], boxes or [])]
        return str(pairs)

    # ------------- IO helpers -------------
    def _parse_yolo_file(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """
        Returns list of (cls_id, xc, yc, w, h) in NORMALIZED units.
        Accepts lines with >=5 tokens: class xc yc w h [conf...]
        """
        out: List[Tuple[int, float, float, float, float]] = []
        if not txt_path.exists():
            return out
        with open(txt_path, "r") as f:
            for ln in f:
                parts = ln.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                    out.append((cls, xc, yc, w, h))
                except Exception:
                    continue
        return out

    # ------------- gather table -------------
    def _gather_table() -> pd.DataFrame:
        """Collect all image-label pairs for the specified split"""
        base = Path(base_dir)
        img_dir = base / "images" / sid
        lbl_dir = base / "labels" / sid

        rows = []
        if img_dir.is_dir():
            for fn in sorted(os.listdir(img_dir)):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")):
                    continue
                ip = img_dir / fn
                lp = lbl_dir / (Path(fn).stem + ".txt")
                if lp.exists():
                    rows.append({
                        "uid": f"{split}/{fn}", 
                        "img_path": str(ip), 
                        "lbl_path": str(lp)
                    })
        
        if debug_limit:
            rows = rows[:debug_limit]
        
        return pd.DataFrame(rows)

    # ------------- build split -------------
    def _build_split() -> Dataset:
        """Build dataset (vectorized, no map)"""
        gdf = _gather_table()

        # Pre-allocate lists
        data = {
            "image": [],
            "problem": [],
            "solution": [],
            "original_question": [],
            "original_answer": []
        }

        # Process all rows (vectorized - no map needed)
        for i, row in gdf.iterrows():
            uid = row["uid"]
            img_path = row["img_path"]
            lbl_path = row["lbl_path"]
            
            # Get native image size to convert normalized coords to pixels
            with Image.open(img_path) as im:
                W, H = im.size

            # Parse YOLO annotations and convert to pixel coords
            labels: List[str] = []
            orig_boxes: List[List[int]] = []
            
            for cls, xc, yc, w, h in _parse_yolo_file(Path(lbl_path)):
                # Convert from normalized center coords to pixel corner coords
                x1 = int(round((xc - w/2) * W))
                y1 = int(round((yc - h/2) * H))
                x2 = int(round((xc + w/2) * W))
                y2 = int(round((yc + h/2) * H))
                
                # Clamp to image bounds
                x1 = max(0, min(W-1, x1))
                x2 = max(0, min(W-1, x2))
                y1 = max(0, min(H-1, y1))
                y2 = max(0, min(H-1, y2))
                
                # Ensure correct ordering
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                
                orig_boxes.append([x1, y1, x2, y2])
                labels.append("Cell")

            # Generate prompt
            q = _det_prompt(uid, i)
            problem = f"{q}\n"

            # Format solution with ORIGINAL boxes (no resizing)
            if labels and orig_boxes:
                solution = _format_solution_one_label("Cell", orig_boxes)
                original_answer = _orig_answer_compact(labels, orig_boxes)
            else:
                solution = "<answer></answer>"
                original_answer = "[]"

            # Append to lists - IMAGE AS LIST
            data["image"].append([img_path])  # Changed: wrap in list
            data["problem"].append(problem)
            data["solution"].append(solution)
            data["original_question"].append(q)
            data["original_answer"].append(original_answer)

        # Create dataset directly from dict
        from datasets import Sequence
        
        features = Features({
            "image": Sequence(Value("string")),  # Changed: list of image paths
            "problem": Value("string"),
            "solution": Value("string"),
            "original_question": Value("string"),
            "original_answer": Value("string"),
        })

        ds = Dataset.from_dict(data, features=features)
        
        if verbose:
            print(f"[deepcell_bbox_grpo:{split}] Loaded {len(ds)} samples")

        return ds

    # --------- assemble splits ----------
    # Build the requested split
    ds = _build_split()
    
    # Create empty dataset with same schema for other splits
    from datasets import Sequence
    
    empty_features = Features({
        "image": Sequence(Value("string")),
        "problem": Value("string"),
        "solution": Value("string"),
        "original_question": Value("string"),
        "original_answer": Value("string"),
    })
    
    empty = Dataset.from_dict({
        "image": [], 
        "problem": [], 
        "solution": [], 
        "original_question": [], 
        "original_answer": []
    }, features=empty_features)

    dsd = {
        "train": empty,
        "validation": empty.select([]),
        "test": empty.select([]),
    }
    dsd[split] = ds

    dsd = DatasetDict(dsd)

    if verbose:
        for sp, s in dsd.items():
            print(f"[deepcell_bbox_grpo:{sp}] rows={len(s)}, cols={s.column_names}")

    # Sanity check: columns must be exactly these five
    for sp, s in dsd.items():
        assert s.column_names == ["image", "problem", "solution", "original_question", "original_answer"], \
            f"{sp} columns are {s.column_names}, expected ['image', 'problem', 'solution', 'original_question', 'original_answer']"

    return dsd




@register_dataset("grazpedwri_dx_bbox_grpo")
def load_wrist_fracture_bbox_grpo(
    num_proc: int = 1,
    batch_size: int = 1024,
    ann_dirs: dict | None = None,
    image_dir: str = "./Medmo_Dataset_1/Medmo_Dataset/Wrist/images",
    debug_limit: int | None = None,
    verbose: bool = False,
) -> DatasetDict:
    """
    GrazPedWri-DX wrist fracture dataset → GRPO-style DatasetDict.
    Returns image paths as lists. NO resizing of images or boxes.
    Boxes are kept in original pixel coordinates.

    Final schema: ['image','problem','solution','original_question','original_answer']
    """

    # ---------------- prompts ----------------
    PROMPTS: List[str] = [
        ("Examine this wrist X-ray and identify all fractures or abnormalities. "
         "For each finding, provide bounding box coordinates as: <box> [x_min, y_min, x_max, y_max] </box> in pixel values."),
        ("Analyze this wrist radiograph. Detect all fractures, dislocations, or abnormal regions. "
         "Return coordinates: <box> [x_min, y_min, x_max, y_max] </box> for each detection."),
        ("Identify all pathological findings in this wrist X-ray. "
         "Report each as: <box> [x_min, y_min, x_max, y_max] </box> using precise pixel coordinates."),
        ("Detect fractures and abnormalities in this wrist image. "
         "Provide bounding boxes: <box> [x_min, y_min, x_max, y_max] </box> for each finding."),
        ("Locate all fractures or injuries visible in this wrist radiograph. "
         "Output format: <box> [x_min, y_min, x_max, y_max] </box> for each detected region."),
    ]

    def _det_prompt(uid_val: Any, idx: int) -> str:
        key = str(uid_val) if uid_val is not None else str(idx)
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        return PROMPTS[h % len(PROMPTS)]

    # ------------- formatting -------------
    def _format_solution(labels: List[str], boxes: List[List[int]]) -> str:
        """Format solution with original boxes (no scaling)"""
        if not labels or not boxes or len(labels) != len(boxes):
            return "<answer></answer>"
        lines = [f"{lab}: <box> {x1}, {y1}, {x2}, {y2} </box>"
                 for lab, (x1, y1, x2, y2) in zip(labels, boxes)]
        return "<answer>\n" + "\n".join(lines) + "\n</answer>"

    def _orig_answer_compact(labels: List[str], boxes: List[List[int]]) -> str:
        """Compact representation for audit"""
        pairs = [{"label": lab, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                 for lab, b in zip(labels or [], boxes or [])]
        return str(pairs)

    # ---------------- default paths ----------------
    if ann_dirs is None:
        ann_dirs = {
            "train": "./Medmo_Dataset_1/Medmo_Dataset/Wrist/folder_structure/supervisely/wrist/ann",
            "validation": "./Medmo_Dataset_1/Medmo_Dataset/Wrist/ann_test",
            "test": "./Medmo_Dataset_1/Medmo_Dataset/Wrist/ann_val",
        }
    
    image_dir = Path(image_dir)

    # ------------- parsing helpers -------------
    def _parse_supervisely_json(json_path: Path) -> List[Tuple[str, int, int, int, int]]:
        """
        Parse Supervisely JSON annotation file.
        Returns list of (label, x1, y1, x2, y2) tuples.
        """
        boxes = []
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            for obj in data.get("objects", []):
                pts = (obj.get("points") or {}).get("exterior") or []
                if len(pts) >= 2:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    
                    # Convert to int (original pixel coords)
                    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                    
                    label = obj.get("classTitle", "Finding")
                    boxes.append((label, x1, y1, x2, y2))
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to parse {json_path}: {e}")
        
        return boxes

    def _gather_table(split: str) -> pd.DataFrame:
        """Collect all annotation-image pairs for a split"""
        ann_dir = Path(ann_dirs[split])
        anns = sorted(ann_dir.glob("*.json"))
        
        if debug_limit:
            anns = anns[:debug_limit]
        
        rows = []
        for ap in anns:
            name = ap.stem
            img_path = image_dir / f"{name}.png"
            rows.append({
                "uid": name,
                "ann_path": str(ap),
                "img_path": str(img_path)
            })
        
        return pd.DataFrame(rows)

    # ------------- build split -------------
    def _build_split(split: str) -> Dataset:
        """Build dataset for a single split (vectorized, no map)"""
        gdf = _gather_table(split)

        # Pre-allocate lists
        data = {
            "image": [],
            "problem": [],
            "solution": [],
            "original_question": [],
            "original_answer": []
        }

        # Process all rows (vectorized - no map needed)
        for i, row in gdf.iterrows():
            uid = row["uid"]
            img_path = row["img_path"]
            ann_path = row["ann_path"]
            
            # Parse annotations
            parsed_boxes = _parse_supervisely_json(Path(ann_path))
            
            # Separate labels and boxes
            labels: List[str] = []
            boxes: List[List[int]] = []
            
            for label, x1, y1, x2, y2 in parsed_boxes:
                labels.append(label)
                boxes.append([x1, y1, x2, y2])

            # Generate prompt
            q = _det_prompt(uid, i)
            problem = f"{q}\n"

            # Format solution with ORIGINAL boxes (no resizing)
            solution = _format_solution(labels, boxes)
            original_answer = _orig_answer_compact(labels, boxes)

            # Append to lists - IMAGE AS LIST
            data["image"].append([img_path])  # Changed: wrap in list
            data["problem"].append(problem)
            data["solution"].append(solution)
            data["original_question"].append(q)
            data["original_answer"].append(original_answer)

        # Create dataset directly from dict
        from datasets import Sequence
        
        features = Features({
            "image": Sequence(Value("string")),  # Changed: list of image paths
            "problem": Value("string"),
            "solution": Value("string"),
            "original_question": Value("string"),
            "original_answer": Value("string"),
        })

        ds = Dataset.from_dict(data, features=features)
        
        if verbose:
            print(f"[grazpedwri_dx_bbox_grpo:{split}] Loaded {len(ds)} samples")

        return ds

    # -------- assemble splits --------
    dsd = DatasetDict({
        "train": _build_split("train"),
        "validation": _build_split("validation"),
        "test": _build_split("test"),
    })

    if verbose:
        for split, ds in dsd.items():
            print(f"[grazpedwri_dx_bbox_grpo:{split}] rows={len(ds)}, cols={ds.column_names}")

    # Sanity check: columns must be exactly these five
    for split, ds in dsd.items():
        assert ds.column_names == ["image", "problem", "solution", "original_question", "original_answer"], \
            f"{split} columns are {ds.column_names}, expected ['image', 'problem', 'solution', 'original_question', 'original_answer']"

    return dsd













# --------------------------
# MedEvalkit Datasets
# --------------------------

def _format_mcq(question: str, options: Dict[str, str] | List[str] | None) -> str:
    q = (question or "").strip()
    if not options:
        return q
    if isinstance(options, dict):
        items = options.items()
    else:
        items = [(chr(ord("A") + i), opt) for i, opt in enumerate(options)]
    opt_lines = [f"({k}) {v}" for k, v in items]
    return q + "\nOptions:\n" + "\n".join(opt_lines)


def _resolve_answer(answer: Any, answer_label: Optional[str], options: Dict[str, str] | List[str] | None) -> str:
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    if answer_label and options:
        if isinstance(options, dict):
            return str(options.get(answer_label, answer_label))
        idx = ord(answer_label.upper()) - ord("A")
        if 0 <= idx < len(options):
            return str(options[idx])
    return "" if answer is None else str(answer)


def _save_bytes_to_cache(bytes_data: bytes, ext: str, subdir: str) -> str:
    h = hashlib.md5(bytes_data).hexdigest()
    out_dir = os.path.join(IMG_CACHE_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{h}.{ext}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(bytes_data)
    return path


def _pick_root(*candidates: str) -> str:
    for c in candidates:
        if c and os.path.exists(c):
            return c
    # fallback to first non-empty candidate
    for c in candidates:
        if c:
            return c
    return ""


def _coerce_image_to_path(image: Any, subdir: str) -> Optional[str]:
    if image is None:
        return None
    # Already a path
    if isinstance(image, str):
        return _resolve_rel_image_path(image, subdir)
    # HF image dict (decode=False)
    if isinstance(image, dict) and ("bytes" in image or "path" in image):
        if image.get("bytes"):
            p = image.get("path")
            # Prefer a directly-usable path. If unavailable, use bytes to avoid
            # expensive recursive path searches for synthetic filenames.
            if isinstance(p, str) and p:
                rp = p if os.path.isabs(p) else _resolve_rel_image_path_fast(p, subdir)
                if os.path.exists(rp):
                    return os.path.abspath(rp)
            return _save_bytes_to_cache(image["bytes"], "jpg", subdir)
        if image.get("path"):
            p = image["path"]
            return _resolve_rel_image_path(p, subdir)
    # PIL
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return _save_bytes_to_cache(buf.getvalue(), "jpg", subdir)
    return None


@lru_cache(maxsize=8192)
def _resolve_rel_image_path_fast(path: str, subdir: str) -> str:
    """
    Fast path resolver that avoids deep recursive globbing.
    """
    if not isinstance(path, str):
        return str(path)
    if os.path.isabs(path):
        return path

    candidates = [
        os.path.join(IMG_CACHE_ROOT, subdir, path),
        os.path.join(DATA_ROOT, path),
        os.path.join(DATA_ROOT_MEDMO, path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return path


@lru_cache(maxsize=4096)
def _resolve_rel_image_path(path: str, subdir: str) -> str:
    """
    Resolve relative image paths to a full path using known roots.
    Falls back to an absolute path in IMG_CACHE_ROOT/subdir if a match is found.
    """
    if not isinstance(path, str):
        return str(path)
    if os.path.isabs(path):
        return path

    # 1) If already in cache folder
    cache_candidate = os.path.join(IMG_CACHE_ROOT, subdir, path)
    if os.path.exists(cache_candidate):
        return os.path.abspath(cache_candidate)

    # 2) Try under DATA_ROOT directly
    data_candidate = os.path.join(DATA_ROOT, path)
    if os.path.exists(data_candidate):
        return os.path.abspath(data_candidate)

    # 2b) Try under DATA_ROOT_MEDMO directly
    data_candidate = os.path.join(DATA_ROOT_MEDMO, path)
    if os.path.exists(data_candidate):
        return os.path.abspath(data_candidate)

    # 3) Try one-level deep under DATA_ROOT
    for root in os.listdir(DATA_ROOT):
        candidate = os.path.join(DATA_ROOT, root, path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    # 3b) Try one-level deep under DATA_ROOT_MEDMO
    if os.path.isdir(DATA_ROOT_MEDMO):
        for root in os.listdir(DATA_ROOT_MEDMO):
            candidate = os.path.join(DATA_ROOT_MEDMO, root, path)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

    # 4) As last resort, search by filename under DATA_ROOT
    filename = os.path.basename(path)
    for found in glob(os.path.join(DATA_ROOT, "**", filename), recursive=True):
        if os.path.isfile(found):
            return os.path.abspath(found)

    # 4b) As last resort, search by filename under DATA_ROOT_MEDMO
    if os.path.isdir(DATA_ROOT_MEDMO):
        for found in glob(os.path.join(DATA_ROOT_MEDMO, "**", filename), recursive=True):
            if os.path.isfile(found):
                return os.path.abspath(found)

    # 5) Fallback to absolute path of provided relative (may not exist)
    return os.path.abspath(path)


def _build_messages(question: str, answer: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    # Use int index consistently; -1 means non-image text.
    user_content = [{"type": "text", "text": (question or "").strip(), "index": -1}]
    if image_paths:
        # prepend images to user content
        image_blocks = [{"type": "image", "text": None, "index": i} for i in range(len(image_paths))]
        user_content = image_blocks + user_content

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": (answer or "").strip(), "index": -1}]},
    ]


def _parse_options_str(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass
    return []


def _finalize_ds(ds: Dataset) -> Dataset:
    # Ensure images column has a consistent string sequence type across datasets.
    return ds.cast_column("images", HFSequence(Value("string")))


def _empty_images(n: int) -> List[List[str]]:
    return [[] for _ in range(n)]


# --------------------------
# Dataset loaders
# --------------------------

@register_dataset("PATH_VQA_test")
def load_path_vqa(split: str = "test", cache_dir: str = DATA_ROOT) -> DatasetDict:
    ds = load_dataset("flaviagiammarino/path-vqa", split=split, cache_dir=cache_dir)
    ds = ds.cast_column("image", HFImage(decode=False))

    def _format(batch):
        out_msgs, out_imgs = [], []
        for img, q, a in zip(batch["image"], batch["question"], batch["answer"]):
            path = _coerce_image_to_path(img, "path_vqa")
            img_list = [path] if path else []
            out_msgs.append(_build_messages(q, a, img_list))
            out_imgs.append(img_list if img_list else [])
        return {"messages": out_msgs, "images": out_imgs}

    ds = ds.map(_format, batched=True, batch_size=128, num_proc=1)
    ds = ds.remove_columns([c for c in ds.column_names if c not in {"messages", "images"}])
    return DatasetDict({"train": _finalize_ds(ds)})



@register_dataset("MedFrameQA_test")
def load_medframeqa(split: str = "test", cache_dir: str = DATA_ROOT) -> DatasetDict:
    ds = load_dataset("SuhaoYu1020/MedFrameQA", split=split, cache_dir=cache_dir)
    # try to avoid decoding if there's an image column
    if "image" in ds.column_names:
        ds = ds.cast_column("image", HFImage(decode=False))

    def _format(batch):
        out_msgs, out_imgs = [], []
        for i in range(len(batch[list(batch.keys())[0]])):
            question = batch.get("question", batch.get("Question", [""]))[i]
            answer = batch.get("answer", batch.get("Answer", [""]))[i]
            img = batch.get("image", batch.get("Image", [None]))[i]
            path = _coerce_image_to_path(img, "medframeqa")
            img_list = [path] if path else []
            out_msgs.append(_build_messages(question, answer, img_list))
            out_imgs.append(img_list if img_list else [])
        return {"messages": out_msgs, "images": out_imgs}

    ds = ds.map(_format, batched=True, batch_size=64, num_proc=1)
    ds = ds.remove_columns([c for c in ds.column_names if c not in {"messages", "images"}])
    return DatasetDict({"train": _finalize_ds(ds)})




@register_dataset("CheXpert_Plus_test")
def load_chexpert_plus(split: str = "test") -> DatasetDict:
    root = _resolve_chexpert_root()
    json_path = os.path.join(root, f"{split}.json")
    records = []

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        image_root = os.path.join(root, "images")
        for ex in data:
            findings = (ex.get("findings") or "").strip()
            impression = (ex.get("impression") or "").strip()
            if findings == "" and impression == "":
                continue
            image = ex.get("image")
            if image and not os.path.isabs(image):
                image = os.path.join(image_root, image)
            answer = f"Findings: {findings} Impression: {impression}."
            question = "Generate a radiology report for the given image."
            img_list = [image] if image else []
            records.append({"messages": _build_messages(question, answer, img_list), "images": img_list if img_list else []})
    else:
        csv_path = _resolve_chexpert_csv(root)
        image_root = _resolve_chexpert_image_root(root)
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            findings = str(row.get("findings", "") or "").strip()
            impression = str(row.get("impression", "") or "").strip()
            if findings == "" and impression == "":
                continue
            image = row.get("path_to_image_png") or row.get("path_to_image") or row.get("image_path")
            if not image:
                continue
            if not os.path.isabs(str(image)):
                image = os.path.join(image_root, str(image))
            answer = f"Findings: {findings} Impression: {impression}."
            question = "Generate a radiology report for the given image."
            records.append({"messages": _build_messages(question, answer, [image]), "images": [image]})

    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("MIMIC_CXR_test")
def load_mimic_cxr(split: str = "test") -> DatasetDict:
    root = _resolve_mimic_root()
    json_path = os.path.join(root, f"{split}.json")
    records = []

    if json_path and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        image_root = os.path.join(root, "images")
        for ex in data:
            findings = (ex.get("findings") or "").strip()
            impression = (ex.get("impression") or "").strip()
            if findings == "" and impression == "":
                continue
            image = ex.get("image")
            if isinstance(image, list):
                img_list = []
                for im in image:
                    p = im
                    if p and not os.path.isabs(p):
                        p = os.path.join(image_root, p)
                    if p:
                        img_list.append(p)
            else:
                p = image
                if p and not os.path.isabs(p):
                    p = os.path.join(image_root, p)
                img_list = [p] if p else []
            answer = f"Findings: {findings} Impression: {impression}."
            question = "Generate a radiology report for the given image."
            records.append({"messages": _build_messages(question, answer, img_list), "images": img_list if img_list else []})
    else:
        csv_path = _resolve_mimic_csv(root)
        image_root = _resolve_mimic_image_root(root)
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            findings = str(row.get("findings", "") or "").strip()
            impression = str(row.get("impression", "") or "").strip()
            if findings == "" and impression == "":
                continue
            image = row.get("image_path") or row.get("path_to_image") or row.get("path")
            if not image:
                continue
            if not os.path.isabs(str(image)):
                image = os.path.join(image_root, str(image))
            answer = f"Findings: {findings} Impression: {impression}."
            question = "Generate a radiology report for the given image."
            records.append({"messages": _build_messages(question, answer, [image]), "images": [image]})

    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("MedTrinity_test")
def load_medtrinity(split: str = "validation") -> DatasetDict:
    jsonl_path, image_root = _resolve_medtrinity_paths()
    records = []
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line, strict=False)
            except Exception:
                continue
            img_path = ex.get("full_path") or ex.get("rel_path") or ex.get("image_path") or ex.get("path")
            if not img_path:
                continue
            if image_root and not os.path.isabs(img_path):
                img_path = os.path.join(image_root, img_path)
            caption = ex.get("caption") or ex.get("report") or ex.get("text") or ex.get("answer") or ""
            caption = str(caption).strip()
            if caption == "":
                continue
            question = "Generate a radiology report for the given image."
            records.append({"messages": _build_messages(question, caption, [img_path]), "images": [img_path]})
    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("MedQA_USMLE_test")
def load_medqa_usmle(split: str = "test") -> DatasetDict:
    base = os.path.join(DATA_ROOT, "MedQA-USMLE-4-options")
    file_map = {
        "train": os.path.join(base, "phrases_no_exclude_train.jsonl"),
        "test": os.path.join(base, "phrases_no_exclude_test.jsonl"),
        "validation": os.path.join(base, "phrases_no_exclude_validation.jsonl"),
    }
    path = file_map.get(split)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"MedQA-USMLE jsonl not found for split '{split}': {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            question = ex.get("question", "")
            options = ex.get("options", {})
            answer = ex.get("answer", "")
            answer_idx = ex.get("answer_idx")
            answer_text = _resolve_answer(answer, answer_idx, options)
            question_fmt = _format_mcq(question, options)
            records.append({"messages": _build_messages(question_fmt, answer_text, []), "images": []})
    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("MedMCQA_test")
def load_medmcqa(split: str = "train") -> DatasetDict:
    ds = load_dataset(os.path.join(DATA_ROOT, "medmcqa"), split=split)
    records = []
    for ex in ds:
        options = {
            "A": ex.get("opa", ""),
            "B": ex.get("opb", ""),
            "C": ex.get("opc", ""),
            "D": ex.get("opd", ""),
        }
        question = ex.get("question", "")
        cop = ex.get("cop")
        answer_label = None
        if isinstance(cop, int) and 0 <= cop <= 3:
            answer_label = chr(ord("A") + cop)
        answer_text = _resolve_answer("", answer_label, options)
        question_fmt = _format_mcq(question, options)
        records.append({"messages": _build_messages(question_fmt, answer_text, []), "images": []})
    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("PubMedQA_test")
def load_pubmedqa(split: str = "train") -> DatasetDict:
    ds = load_dataset(os.path.join(DATA_ROOT, "pubmedqa"), split=split)
    records = []
    for ex in ds:
        data = ex.get("data", {})
        question = data.get("Question", "")
        options = data.get("Options", {})
        answer_text = data.get("Correct Answer", "")
        answer_label = data.get("Correct Option", "")
        answer_text = _resolve_answer(answer_text, answer_label, options)
        question_fmt = _format_mcq(question, options)
        records.append({"messages": _build_messages(question_fmt, answer_text, []), "images": []})
    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("Medbullets_op4_test")
def load_medbullets_op4(split: str = "train") -> DatasetDict:
    ds = load_dataset(os.path.join(DATA_ROOT, "Medical-Eval-MedBullets_op4"), split=split)
    records = []
    for ex in ds:
        question = ex.get("question", "")
        options = ex.get("options", {})
        answer = ex.get("answer", "")
        answer_idx = ex.get("answer_idx", "")
        answer_text = _resolve_answer(answer, answer_idx, options)
        question_fmt = _format_mcq(question, options)
        records.append({"messages": _build_messages(question_fmt, answer_text, []), "images": []})
    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("Medbullets_op5_test")
def load_medbullets_op5(split: str = "train") -> DatasetDict:
    ds = load_dataset(os.path.join(DATA_ROOT, "medbullets_op5"), split=split)
    records = []
    for ex in ds:
        question = ex.get("question", "")
        options = {
            "A": ex.get("opa", ""),
            "B": ex.get("opb", ""),
            "C": ex.get("opc", ""),
            "D": ex.get("opd", ""),
            "E": ex.get("ope", ""),
        }
        answer = ex.get("answer", "")
        answer_idx = ex.get("answer_idx", "")
        answer_text = _resolve_answer(answer, answer_idx, options)
        question_fmt = _format_mcq(question, options)
        records.append({"messages": _build_messages(question_fmt, answer_text, []), "images": []})
    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("MedXpertQA-Text_test")
def load_medxpertqa_text(split: str = "test") -> DatasetDict:
    base_root = _pick_root(
        os.path.join(DATA_ROOT, "MedXpertQA"),
        os.path.join(DATA_ROOT_MEDMO, "MedXpertQA"),
    )
    base = os.path.join(base_root, "Text")
    file_map = {
        "train": os.path.join(base, "dev.jsonl"),
        "validation": os.path.join(base, "validation.jsonl"),
        "test": os.path.join(base, "test.jsonl"),
    }
    path = file_map.get(split)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"MedXpertQA-Text jsonl not found for split '{split}': {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            question = ex.get("question", "")
            options = ex.get("options", {})
            label = ex.get("label", "")
            answer_text = _resolve_answer("", label, options)
            question_fmt = _format_mcq(question, options)
            records.append({"messages": _build_messages(question_fmt, answer_text, []), "images": []})
    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})



@register_dataset("MedQA_MCMLE_test")
def load_medqa_mcmle(split: str = "train", cache_dir: str = DATA_ROOT) -> DatasetDict:
    ds = load_dataset("shuyuej/MedQA-MCMLE-Benchmark", split=split, cache_dir=cache_dir)
    records = []
    for ex in ds:
        question = ex.get("question", "")
        options = ex.get("options", {})
        answer = ex.get("answer", "")
        answer_idx = ex.get("answer_idx", "")
        answer_text = _resolve_answer(answer, answer_idx, options)
        question_fmt = _format_mcq(question, options)
        records.append({"messages": _build_messages(question_fmt, answer_text, []), "images": []})
    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("SuperGPQA_test")
def load_supergpqa(split: str = "train", cache_dir: str = DATA_ROOT) -> DatasetDict:
    ds = load_dataset("m-a-p/SuperGPQA", split=split, cache_dir=cache_dir)
    records = []
    for ex in ds:
        question = ex.get("question", "")
        options = ex.get("options", [])
        answer = ex.get("answer", "")
        answer_label = ex.get("answer_letter", "")
        answer_text = _resolve_answer(answer, answer_label, options)
        question_fmt = _format_mcq(question, options)
        records.append({"messages": _build_messages(question_fmt, answer_text, []), "images": []})
    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("IU_XRAY_test")
def load_iu_xray(split: str = "test") -> DatasetDict:
    base = _pick_root(
        os.path.join(DATA_ROOT, "IU_XRAY"),
        os.path.join(DATA_ROOT_MEDMO, "IU_XRAY"),
    )
    json_path = os.path.join(base, f"{split}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"IU_XRAY json not found for split '{split}': {json_path}")

    image_root = os.path.join(base, "Images")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for ex in data:
        images = ex.get("image", [])
        img_paths = [os.path.join(image_root, p) for p in images]
        findings = (ex.get("findings") or "").strip()
        impression = (ex.get("impression") or "").strip()
        answer = ("Findings: " + findings + "\nImpression: " + impression).strip()
        question = "Generate a radiology report for the given image(s)."
        records.append({"messages": _build_messages(question, answer, img_paths), "images": img_paths if img_paths else []})
    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("CMB_test")
def load_cmb(split: str = "test") -> DatasetDict:
    base = _pick_root(
        os.path.join(DATA_ROOT, "CMB"),
        os.path.join(DATA_ROOT_MEDMO, "CMB"),
    )
    clin_path = os.path.join(base, "CMB-Clin-qa.json")
    if not os.path.exists(clin_path):
        raise FileNotFoundError(f"CMB data not found: {clin_path}")

    with open(clin_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for case in data:
        title = case.get("title", "")
        desc = case.get("description", "")
        for qa in case.get("QA_pairs", []):
            q = qa.get("question", "")
            a = qa.get("answer", "")
            question = f"{title}\n{desc}\n\nQuestion: {q}".strip()
            records.append({"messages": _build_messages(question, a, []), "images": []})
    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("CMExam_test")
def load_cmexam(split: str = "test") -> DatasetDict:
    base = _pick_root(
        os.path.join(DATA_ROOT, "CMExam"),
        os.path.join(DATA_ROOT_MEDMO, "CMExam"),
    )
    jsonl_path = os.path.join(base, f"{split}.json")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"CMExam jsonl not found for split '{split}': {jsonl_path}")

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            question = ex.get("Question", "")
            options_list = ex.get("Options", [])
            options = {o.get("key"): o.get("value") for o in options_list if isinstance(o, dict)}
            answer_label = ex.get("Answer", "")
            answer_text = _resolve_answer("", answer_label, options)
            question_fmt = _format_mcq(question, options)
            records.append({"messages": _build_messages(question_fmt, answer_text, []), "images": []})
    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("CMMLU_test")
def load_cmmlu(split: str = "test", subjects: Optional[List[str]] = None) -> DatasetDict:
    base_root = _pick_root(
        os.path.join(DATA_ROOT, "cmmlu"),
        os.path.join(DATA_ROOT_MEDMO, "cmmlu"),
    )
    base = os.path.join(base_root, split)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"CMMLU split dir not found: {base}")

    csv_files = glob(os.path.join(base, "*.csv"))
    if subjects:
        csv_files = [p for p in csv_files if os.path.splitext(os.path.basename(p))[0] in subjects]

    records = []
    for path in csv_files:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            question = str(row.get("Question", "")).strip()
            options = {
                "A": str(row.get("A", "")).strip(),
                "B": str(row.get("B", "")).strip(),
                "C": str(row.get("C", "")).strip(),
                "D": str(row.get("D", "")).strip(),
            }
            answer_label = str(row.get("Answer", "")).strip()
            answer_text = _resolve_answer("", answer_label, options)
            question_fmt = _format_mcq(question, options)
            records.append({"messages": _build_messages(question_fmt, answer_text, []), "images": []})
    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


def _resolve_chexpert_root() -> str:
    env = os.environ.get("CHEXPERT_PLUS_ROOT") or os.environ.get("CHEXPERT_PLUS_PATH")
    if env and os.path.exists(env):
        return env
    candidates = [
        os.path.join(DATA_ROOT_MEDMO, "CheXpert-Plus"),
        "./Medmo_Dataset_1/Medmo_Dataset/CheXpert-Plus",
        os.path.join(DATA_ROOT, "CheXpert_Plus"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fallback: use provided DATA_ROOT even if missing; caller will fail on read
    return os.path.join(DATA_ROOT, "CheXpert_Plus")


def _resolve_chexpert_image_root(root: str) -> str:
    for c in [os.path.join(root, "images", "PNG"), os.path.join(root, "PNG"), os.path.join(root, "images")]:
        if os.path.isdir(c):
            return c
    return os.path.join(root, "images")


def _resolve_chexpert_csv(root: str) -> str:
    env_csv = os.environ.get("CHEXPERT_PLUS_CSV") or os.environ.get("CHEXPERT_PLUS_VAL_CSV")
    if env_csv:
        p = env_csv if os.path.isabs(env_csv) else os.path.join(root, env_csv)
        if os.path.isfile(p):
            return p
    for c in ["df_val_good.csv", "df_chexpert_plus_val.csv", "df_val.csv"]:
        p = os.path.join(root, c)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("CheXpert-Plus CSV not found. Set CHEXPERT_PLUS_CSV or place df_val_good.csv in dataset root.")


def _resolve_mimic_root() -> str:
    env = os.environ.get("MIMIC_CXR_ROOT") or os.environ.get("MIMIC_CXR_PATH")
    if env and os.path.exists(env):
        return env
    candidates = [
        os.path.join(DATA_ROOT_MEDMO, "MIMIC-CXR-Report"),
        os.path.join(DATA_ROOT_MEDMO, "MIMIC-CXR-Report", "Reports"),
        os.path.join(DATA_ROOT_MEDMO, "MIMIC-CXR"),
        "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-Report",
        "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-Report/Reports",
        "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR",
        os.path.join(DATA_ROOT, "MIMIC_CXR"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(DATA_ROOT, "MIMIC_CXR")


def _resolve_mimic_image_root(root: str) -> str:
    candidates = [
        os.path.join(DATA_ROOT_MEDMO, "MIMIC-CXR-JPG", "physionet.org", "files", "mimic-cxr-jpg", "2.1.0", "files"),
        "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.1.0/files",
        os.path.join(root, "images"),
        os.path.join(root, "files"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return root


def _resolve_mimic_csv(root: str) -> str:
    candidates = [
        os.path.join(DATA_ROOT_MEDMO, "MIMIC-CXR-Report", "Reports", "mimic_reports_with_images_clean_val.csv"),
        "./Medmo_Dataset_1/Medmo_Dataset/MIMIC-CXR-Report/Reports/mimic_reports_with_images_clean_val.csv",
        os.path.join(root, "Reports", "mimic_reports_with_images_clean_val.csv"),
        os.path.join(root, "mimic_reports_with_images_clean_val.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError("MIMIC-CXR CSV not found. Set MIMIC_CXR_ROOT or place mimic_reports_with_images_clean_val.csv in dataset root.")


def _resolve_medtrinity_paths() -> tuple[str, str]:
    default_root = "./Medmo_Dataset_1/Medmo_Dataset/MedTrinity-25M/25M_clean"
    default_jsonl = os.path.join(default_root, "combined_metadata_filtered_relpath_val.jsonl")
    if os.path.isfile(default_jsonl):
        return default_jsonl, default_root
    medmo_root = os.path.join(DATA_ROOT_MEDMO, "MedTrinity-25M", "25M_clean")
    medmo_jsonl = os.path.join(medmo_root, "combined_metadata_filtered_relpath_val.jsonl")
    if os.path.isfile(medmo_jsonl):
        return medmo_jsonl, medmo_root
    # fallback to DATA_ROOT if user placed it there
    alt_root = os.path.join(DATA_ROOT, "MedTrinity")
    alt_jsonl = os.path.join(alt_root, "combined_metadata_filtered_relpath_val.jsonl")
    if os.path.isfile(alt_jsonl):
        return alt_jsonl, alt_root
    raise FileNotFoundError(f"MedTrinity jsonl not found at {default_jsonl} or {alt_jsonl}")








# --------------------------
# Multi-dataset loader
# --------------------------
def load_datasets(
    names: List[str],
    num_proc: int = 2,
    batch_size: int = 128,
    interleave: bool = True,
    interleave_stopping_strategy: str = "all_exhausted",
    weights: Optional[List[float]] = None,
    seed: int = 42,
    verbose: bool = False,
    filter_invalid: bool = True,
    min_image_hw: int = 16,
    keep_empty_images_as_none: bool = False,
) -> DatasetDict:
    """
    Load and merge multiple registered datasets into a single DatasetDict
    with unified schema: ['messages', 'images'] per split.
    """
    if not names:
        raise ValueError("Provide at least one dataset name.")
    for n in names:
        if n not in _LOADER_REGISTRY:
            raise KeyError(f"Unknown dataset '{n}'. Available: {available_datasets()}")

    if verbose:
        print(f"[load_datasets] requesting {len(names)} datasets: {names}")

    ddicts = []
    for i, n in enumerate(names, start=1):
        t0 = time.time()
        if verbose:
            print(f"[load_datasets] loading {i}/{len(names)}: {n}")
        ddicts.append(_call_loader(n, num_proc=num_proc, batch_size=batch_size))
        if verbose:
            print(f"[load_datasets] loaded {n} in {time.time() - t0:.1f}s")

    # Normalize schemas so concatenation/interleave won't fail when mixing text/image datasets.
    def _normalize_dataset(ds):
        from datasets import Features, Sequence, Value

        # Keep only messages/images to avoid feature mismatches
        keep = [c for c in ("messages", "images") if c in ds.column_names]
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)

        def _norm_batch(batch):
            msgs = batch.get("messages", [])
            imgs = batch.get("images", [])
            out_msgs, out_imgs = [], []

            n = max(len(msgs), len(imgs))
            for i in range(n):
                m = msgs[i] if i < len(msgs) else []
                im = imgs[i] if i < len(imgs) else []

                # Normalize messages
                if isinstance(m, str):
                    m = [{"role": "user", "content": [{"type": "text", "text": m, "index": 0}]}]
                elif isinstance(m, dict):
                    if "role" not in m and ("prompt" in m or "text" in m):
                        text_val = m.get("prompt", m.get("text", ""))
                        m = [{"role": "user", "content": [{"type": "text", "text": text_val, "index": 0}]}]
                    else:
                        m = [m]
                elif isinstance(m, list) and m and all(isinstance(x, str) for x in m):
                    turns = []
                    for j, txt in enumerate(m):
                        role = "user" if j % 2 == 0 else "assistant"
                        turns.append({"role": role, "content": [{"type": "text", "text": txt, "index": j}]})
                    m = turns
                elif isinstance(m, list) and m and isinstance(m[0], list):
                    flat = []
                    for sub in m:
                        if isinstance(sub, list):
                            flat.extend(sub)
                    m = flat

                new_turns = []
                for turn in (m or []):
                    if isinstance(turn, str):
                        turn = {"role": "user", "content": [{"type": "text", "text": turn, "index": 0}]}
                    if not isinstance(turn, dict):
                        continue
                    content = turn.get("content", [])
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content, "index": 0}]
                    elif isinstance(content, dict):
                        content = [content]
                    elif isinstance(content, list) and content and all(isinstance(x, str) for x in content):
                        content = [{"type": "text", "text": x, "index": j} for j, x in enumerate(content)]
                    new_content = []
                    for idx_part, part in enumerate(content or []):
                        if not isinstance(part, dict):
                            continue
                        p = dict(part)
                        try:
                            p["index"] = int(p.get("index")) if p.get("index") is not None else idx_part
                        except Exception:
                            p["index"] = idx_part
                        if p.get("type") is None:
                            p["type"] = "text"
                        txt = p.get("text")
                        p["text"] = "" if txt is None else str(txt)
                        new_content.append(p)
                    new_turns.append({"role": str(turn.get("role", "user")), "content": new_content})
                out_msgs.append(new_turns)

                # Normalize images
                if im is None:
                    im_norm = [None] if keep_empty_images_as_none else []
                elif isinstance(im, str):
                    im_norm = [im]
                elif isinstance(im, list):
                    if im and all(isinstance(x, list) for x in im):
                        flat = []
                        for sub in im:
                            for x in sub:
                                if isinstance(x, str) and x:
                                    flat.append(x)
                        im_norm = flat
                    else:
                        im_norm = [x for x in im if isinstance(x, str) and x]
                        if not im_norm and keep_empty_images_as_none:
                            im_norm = [None]
                else:
                    im_norm = [None] if keep_empty_images_as_none else []
                out_imgs.append(im_norm)

            return {"messages": out_msgs, "images": out_imgs}

        map_kwargs = {"batched": True, "batch_size": 256, "desc": "[load_datasets] normalize schema"}
        try:
            if getattr(ds, "is_streaming", False):
                map_kwargs["num_proc"] = None
            else:
                map_kwargs["num_proc"] = max(1, num_proc)
        except Exception:
            map_kwargs["num_proc"] = None

        ds = ds.map(_norm_batch, **map_kwargs)

        # Cast to a common feature schema
        features = Features({
            "messages": Sequence(feature={
                "role": Value("string"),
                "content": Sequence(feature={
                    "type": Value("string"),
                    "text": Value("string"),
                    "index": Value("int64"),
                }),
            }),
            "images": Sequence(Value("string")),
        })
        try:
            ds = ds.cast(features)
        except Exception:
            try:
                ds = Dataset.from_list(ds.to_list(), features=features)
            except Exception:
                pass
        return ds

    ddicts = [ {k: _normalize_dataset(v) for k, v in d.items()} for d in ddicts ]

    if verbose:
        for name, d in zip(names, ddicts):
            split_stats = []
            for split, ds in d.items():
                try:
                    sz = len(ds)
                except TypeError:
                    sz = "iterable"
                split_stats.append(f"{split}={sz}")
            print(f"[load_datasets] {name}: " + ", ".join(split_stats))

    # Collect common splits
    all_splits = set().union(*[set(d.keys()) for d in ddicts])
    out = {}
    for split in sorted(all_splits):
        parts = [d[split] for d in ddicts if split in d]
        if len(parts) == 1:
            out[split] = parts[0]
            continue

        if interleave:
            # Interleave keeps class balance better during training
            probs = None
            if weights is not None:
                if len(weights) != len(parts):
                    raise ValueError("weights length must match number of datasets included for this split.")
                # Normalize
                s = float(sum(weights))
                probs = [w / s for w in weights]
            # Use all_exhausted to avoid truncation to the shortest dataset.
            out[split] = interleave_datasets(
                parts,
                probabilities=probs,
                seed=seed,
                stopping_strategy=interleave_stopping_strategy,
            )
        else:
            out[split] = concatenate_datasets(parts)

        # Drop obviously broken samples so they don't crash the collator or produce empty labels.
        if filter_invalid:
            def _valid(example):
                try:
                    # Require messages and at least one non-empty assistant text
                    msgs = example.get("messages")
                    if not msgs or not isinstance(msgs, list):
                        return False
                    # Handle both [turns] and [[turns]] shapes
                    turn_list = msgs[-1] if msgs and isinstance(msgs[0], dict) else msgs[-1]
                    if isinstance(turn_list, list):
                        asst = turn_list[-1] if turn_list else {}
                    else:
                        asst = turn_list
                    content = asst.get("content") if isinstance(asst, dict) else None
                    texts = []
                    if isinstance(content, list):
                        for p in content:
                            if isinstance(p, dict) and p.get("type") == "text":
                                t = (p.get("text") or "").strip()
                                if t:
                                    texts.append(t)
                    elif isinstance(content, dict):
                        t = (content.get("text") or "").strip()
                        if t:
                            texts.append(t)
                    if not texts:
                        return False

                    # Basic image/path validation
                    imgs = example.get("images")
                    if imgs and isinstance(imgs, list):
                        img_item = imgs[0][0] if isinstance(imgs[0], list) else imgs[0]
                        if isinstance(img_item, str):
                            if not os.path.exists(img_item):
                                return False
                            # Quick dimension sanity check for tiny/corrupt files
                            try:
                                with Image.open(img_item) as im:
                                    w, h = im.size
                                    if min(w, h) < min_image_hw:
                                        return False
                            except Exception:
                                return False
                    return True
                except Exception:
                    return False

            # Use multiprocessing for arrow datasets; fall back to single-thread for streaming.
            # filter_kwargs = {"desc": f"[load_datasets] drop invalid {split}"}
            # try:
            #     filter_kwargs["num_proc"] = num_proc if not getattr(out[split], "is_streaming", False) else None
            # except Exception:
            #     filter_kwargs["num_proc"] = None
            # out[split] = out[split].filter(_valid, **filter_kwargs)

    if verbose:
        for split, ds in out.items():
            try:
                sz = len(ds)
            except TypeError:
                sz = "iterable"
            print(f"[load_datasets] merged {split}: {sz}")

    return DatasetDict(out)


# --------------------------
# GRPO multi-dataset loader (keeps GRPO schema)
# --------------------------
def load_datasets_grpo(
    names: List[str],
    num_proc: int = 2,
    batch_size: int = 128,
    interleave: bool = True,
    interleave_stopping_strategy: str = "all_exhausted",
    weights: Optional[List[float]] = None,
    seed: int = 42,
    verbose: bool = False,
) -> DatasetDict:
    """
    Load and merge GRPO-style datasets into a single DatasetDict
    with unified schema: ['image', 'problem', 'solution', 'original_question', 'original_answer'].
    """
    if not names:
        raise ValueError("Provide at least one dataset name.")
    for n in names:
        if n not in _LOADER_REGISTRY:
            raise KeyError(f"Unknown dataset '{n}'. Available: {available_datasets()}")

    if verbose:
        print(f"[load_datasets_grpo] requesting {len(names)} datasets: {names}")

    ddicts = []
    for i, n in enumerate(names, start=1):
        t0 = time.time()
        if verbose:
            print(f"[load_datasets_grpo] loading {i}/{len(names)}: {n}")
        ddicts.append(_call_loader(n, num_proc=num_proc, batch_size=batch_size))
        if verbose:
            print(f"[load_datasets_grpo] loaded {n} in {time.time() - t0:.1f}s")

    required = {"image", "problem", "solution", "original_question", "original_answer"}

    def _normalize_grpo_dataset(ds):
        from datasets import Features, Sequence, Value

        missing = [c for c in required if c not in ds.column_names]
        if missing:
            raise ValueError(f"[load_datasets_grpo] Missing columns: {missing}")

        # Keep only GRPO columns to avoid feature mismatches.
        drop = [c for c in ds.column_names if c not in required]
        if drop:
            ds = ds.remove_columns(drop)

        features = Features({
            "image": Sequence(Value("string")),
            "problem": Value("string"),
            "solution": Value("string"),
            "original_question": Value("string"),
            "original_answer": Value("string"),
        })
        try:
            ds = ds.cast(features)
        except Exception:
            try:
                ds = Dataset.from_list(ds.to_list(), features=features)
            except Exception:
                pass
        return ds

    ddicts = [{k: _normalize_grpo_dataset(v) for k, v in d.items()} for d in ddicts]

    if verbose:
        for name, d in zip(names, ddicts):
            split_stats = []
            for split, ds in d.items():
                try:
                    sz = len(ds)
                except TypeError:
                    sz = "iterable"
                split_stats.append(f"{split}={sz}")
            print(f"[load_datasets_grpo] {name}: " + ", ".join(split_stats))

    all_splits = set().union(*[set(d.keys()) for d in ddicts])
    out: Dict[str, Dataset] = {}
    for split in sorted(all_splits):
        parts = [d[split] for d in ddicts if split in d]
        if len(parts) == 1:
            out[split] = parts[0]
            continue

        if interleave:
            probs = None
            if weights is not None:
                if len(weights) != len(parts):
                    raise ValueError("weights length must match number of datasets included for this split.")
                s = float(sum(weights))
                probs = [w / s for w in weights]
            out[split] = interleave_datasets(
                parts,
                probabilities=probs,
                seed=seed,
                stopping_strategy=interleave_stopping_strategy,
            )
        else:
            out[split] = concatenate_datasets(parts)

    if verbose:
        for split, ds in out.items():
            try:
                sz = len(ds)
            except TypeError:
                sz = "iterable"
            print(f"[load_datasets_grpo] merged {split}: {sz}")

    return DatasetDict(out)

# --------------------------
# Quick sanity check (optional)
# --------------------------
def peek(dataset: DatasetDict, k: int = 1, split: str = "train"):
    row = dataset[split][0] if len(dataset[split]) > 0 else None
    print(f"[peek] {split} size:", len(dataset[split]))
    if row:
        print("[peek] sample keys:", list(row.keys()))
        print("[peek] messages[0]:", row["messages"][0])
        print("[peek] images[0]:", row["images"][0] if row["images"] else None)






                    


# # --------------------------
# # Multi-dataset loader
# # --------------------------
# def load_datasets(
#     names: List[str],
#     num_proc: int = 2,
#     batch_size: int = 128,
#     shuffle: bool = True,
#     seed: int = 42,
# ) -> DatasetDict:
#     """
#     Load and merge multiple registered datasets into a single DatasetDict.
#     Simply concatenates all datasets and optionally shuffles.
    
#     Args:
#         names: List of dataset names to load
#         num_proc: Number of processes for loading
#         batch_size: Batch size for processing
#         shuffle: Whether to shuffle the merged dataset
#         seed: Random seed for shuffling
#     """
#     if not names:
#         raise ValueError("Provide at least one dataset name.")
#     for n in names:
#         if n not in _LOADER_REGISTRY:
#             raise KeyError(f"Unknown dataset '{n}'. Available: {available_datasets()}")

#     print(f"\n{'='*80}")
#     print(f"[load_datasets] Loading {len(names)} datasets...")
#     print(f"{'='*80}\n")

#     # Load all individual datasets
#     ddicts = [_LOADER_REGISTRY[n](num_proc=num_proc, batch_size=batch_size) for n in names]

#     # Log individual dataset sizes
#     print("📊 Individual dataset sizes:")
#     for name, ddict in zip(names, ddicts):
#         for split in ddict.keys():
#             print(f"  {name:45s} [{split:12s}]: {len(ddict[split]):>10,} samples")

#     # Collect common splits
#     all_splits = set().union(*[set(d.keys()) for d in ddicts])
#     out = {}
    
#     print(f"\n{'='*80}")
#     print(f"🔀 Merging datasets (shuffle={shuffle})...")
#     print(f"{'='*80}\n")
    
#     for split in sorted(all_splits):
#         # Get all datasets that have this split
#         parts = [d[split] for d in ddicts if split in d]
        
#         print(f"\n📁 Split: {split}")
#         print(f"  - Datasets with this split: {len(parts)}")
#         print(f"  - Individual sizes: {[len(p) for p in parts]}")
#         print(f"  - Total samples: {sum(len(p) for p in parts):,}")
        
#         if len(parts) == 0:
#             continue
        
#         if len(parts) == 1:
#             merged = parts[0]
#             print(f"  ✅ Only 1 dataset, using directly")
#         else:
#             # Concatenate all datasets
#             merged = concatenate_datasets(parts)
#             print(f"  ✅ Concatenated: {len(merged):,} samples")
        
#         # Shuffle if requested
#         if shuffle:
#             merged = merged.shuffle(seed=seed)
#             print(f"  🔀 Shuffled with seed={seed}")
        
#         out[split] = merged

#     print(f"\n{'='*80}")
#     print(f"✨ FINAL MERGED DATASET:")
#     print(f"{'='*80}")
#     total_samples = 0
#     for split, ds in sorted(out.items()):
#         size = len(ds)
#         total_samples += size
#         print(f"  {split:12s}: {size:>10,} samples")
#     print(f"  {'TOTAL':12s}: {total_samples:>10,} samples")
#     print(f"{'='*80}\n")

#     return DatasetDict(out)


# --------------------------
# Alternative merge (interleave/concat + optional filtering)
# --------------------------
def load_datasets_interleave(
    names: List[str],
    num_proc: int = 2,
    batch_size: int = 128,
    interleave: bool = True,
    interleave_stopping_strategy: str = "all_exhausted",
    weights: Optional[List[float]] = None,
    seed: int = 42,
    verbose: bool = False,
    filter_invalid: bool = False,
    min_image_hw: int = 16,
    drop_non_path_images: bool = False,
) -> DatasetDict:
    """
    Alternative helper to merge datasets with optional interleaving.
    Defaults to no filtering to keep runtime low; set filter_invalid=True
    if you want to drop empty/invalid samples.
    """
    if not names:
        raise ValueError("Provide at least one dataset name.")
    for n in names:
        if n not in _LOADER_REGISTRY:
            raise KeyError(f"Unknown dataset '{n}'. Available: {available_datasets()}")

    if verbose:
        print(f"[load_datasets_interleave] requesting {len(names)} datasets: {names}")

    ddicts = []
    for i, n in enumerate(names, start=1):
        t0 = time.time()
        if verbose:
            print(f"[load_datasets_interleave] loading {i}/{len(names)}: {n}")
        ddicts.append(_call_loader(n, num_proc=num_proc, batch_size=batch_size))
        if verbose:
            print(f"[load_datasets_interleave] loaded {n} in {time.time() - t0:.1f}s")

    if verbose:
        for name, d in zip(names, ddicts):
            split_stats = []
            for split, ds in d.items():
                try:
                    sz = len(ds)
                except TypeError:
                    sz = "iterable"
                split_stats.append(f"{split}={sz}")
            print(f"[load_datasets_interleave] {name}: " + ", ".join(split_stats))

    all_splits = set().union(*[set(d.keys()) for d in ddicts])
    out = {}
    for split in sorted(all_splits):
        parts = [d[split] for d in ddicts if split in d]
        if len(parts) == 1:
            out[split] = parts[0]
        elif interleave:
            probs = None
            if weights is not None:
                if len(weights) != len(parts):
                    raise ValueError("weights length must match number of datasets included for this split.")
                s = float(sum(weights))
                probs = [w / s for w in weights]
            out[split] = interleave_datasets(
                parts,
                probabilities=probs,
                seed=seed,
                stopping_strategy=interleave_stopping_strategy,
            )
        else:
            out[split] = concatenate_datasets(parts)

        # Fast guard: drop samples whose images are not plain paths/strings (e.g., PIL objects).
        if drop_non_path_images:
            def _is_pathy(example):
                imgs = example.get("images")
                if not imgs or not isinstance(imgs, list):
                    return False
                img_item = imgs[0][0] if isinstance(imgs[0], list) else imgs[0]
                return isinstance(img_item, str)
            out[split] = out[split].filter(_is_pathy, desc=f"[load_datasets_interleave] keep path-like images {split}")

        if filter_invalid:
            def _valid(example):
                try:
                    msgs = example.get("messages")
                    if not msgs or not isinstance(msgs, list):
                        return False
                    turn_list = msgs[-1] if msgs and isinstance(msgs[0], dict) else msgs[-1]
                    if isinstance(turn_list, list):
                        asst = turn_list[-1] if turn_list else {}
                    else:
                        asst = turn_list
                    content = asst.get("content") if isinstance(asst, dict) else None
                    texts = []
                    if isinstance(content, list):
                        for p in content:
                            if isinstance(p, dict) and p.get("type") == "text":
                                t = (p.get("text") or "").strip()
                                if t:
                                    texts.append(t)
                    elif isinstance(content, dict):
                        t = (content.get("text") or "").strip()
                        if t:
                            texts.append(t)
                    if not texts:
                        return False

                    imgs = example.get("images")
                    if not imgs or not isinstance(imgs, list):
                        return False
                    img_count = len(imgs[0]) if isinstance(imgs[0], list) else len(imgs)
                    placeholder_count = 0
                    first_msg = msgs[0] if isinstance(msgs[0], dict) else msgs[0][0]
                    contents = first_msg.get("content") if isinstance(first_msg, dict) else None
                    if isinstance(contents, list):
                        for part in contents:
                            if isinstance(part, dict) and (
                                part.get("type") == "image" or "image" in part or part.get("image") is not None
                            ):
                                placeholder_count += 1
                    if img_count == 0 or placeholder_count != img_count:
                        return False

                    img_item = imgs[0][0] if isinstance(imgs[0], list) else imgs[0]
                    if isinstance(img_item, str):
                        if not os.path.exists(img_item):
                            return False
                        try:
                            with Image.open(img_item) as im:
                                w, h = im.size
                                if min(w, h) < min_image_hw:
                                    return False
                        except Exception:
                            return False
                    return True
                except Exception:
                    return False

            filter_kwargs = {"desc": f"[load_datasets_interleave] drop invalid {split}"}
            try:
                filter_kwargs["num_proc"] = num_proc if not getattr(out[split], "is_streaming", False) else None
            except Exception:
                filter_kwargs["num_proc"] = None
            out[split] = out[split].filter(_valid, **filter_kwargs)

    if verbose:
        for split, ds in out.items():
            try:
                sz = len(ds)
            except TypeError:
                sz = "iterable"
            print(f"[load_datasets_interleave] merged {split}: {sz}")

    return DatasetDict(out)

# --------------------------
# Quick sanity check (optional)
# --------------------------
def peek(dataset: DatasetDict, k: int = 3, split: str = "train"):
    """Peek at dataset samples"""
    if split not in dataset:
        print(f"[peek] ❌ Split '{split}' not found. Available: {list(dataset.keys())}")
        return
    
    total_size = len(dataset[split])
    print(f"\n{'='*80}")
    print(f"[peek] 📊 Split '{split}' - Total size: {total_size:,} samples")
    print(f"{'='*80}\n")
    
    if total_size == 0:
        print("⚠️  Dataset is empty!")
        return
    
    # Sample from different positions
    indices = [0]  # Start
    if total_size > 1:
        indices.append(total_size // 2)  # Middle
    if total_size > 2:
        indices.append(total_size - 1)  # End
    
    for idx in indices[:k]:
        row = dataset[split][idx]
        print(f"\n--- Sample {idx} ---")
        print(f"Keys: {list(row.keys())}")
        
        # Check messages
        if row.get("messages"):
            msgs = row["messages"]
            print(f"Messages: {len(msgs)} turns")
            if msgs:
                first_turn = msgs[0]
                role = first_turn.get("role", "unknown")
                content = first_turn.get("content", [])
                if content and isinstance(content[0], dict):
                    text = str(content[0].get("text", ""))[:100]
                    print(f"  First turn [{role}]: {text}...")
        
        # Check images
        if row.get("images"):
            imgs = row["images"]
            if imgs and imgs[0]:
                print(f"Images: {str(imgs[0])[:100]}...")
            else:
                print(f"Images: [none]")
    
    print(f"\n{'='*80}\n")
