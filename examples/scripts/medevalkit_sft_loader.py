# medevalkit_sft_loader.py
# SFT loader for MedEvalKit datasets -> messages/images schema

import os
import io
import json
import glob
import hashlib
import ast
from typing import Dict, List, Optional, Callable, Any
from functools import lru_cache

import pandas as pd
from PIL import Image
from datasets import Dataset, DatasetDict, load_dataset, Image as HFImage, Sequence as HFSequence, Value

DATA_ROOT = "/vast/users/imran.razzak/Document/MedEvalKit/datas"
DATA_ROOT_MEDMO = "/vast/users/imran.razzak/Medmo_Dataset_1/Medmo_Dataset"
IMG_CACHE_ROOT = "/vast/users/imran.razzak/Document/medmo/trl/.cache/medevalkit_images"

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


# --------------------------
# Helpers
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
    for found in glob.glob(os.path.join(DATA_ROOT, "**", filename), recursive=True):
        if os.path.isfile(found):
            return os.path.abspath(found)

    # 4b) As last resort, search by filename under DATA_ROOT_MEDMO
    if os.path.isdir(DATA_ROOT_MEDMO):
        for found in glob.glob(os.path.join(DATA_ROOT_MEDMO, "**", filename), recursive=True):
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


@register_dataset("VQA_RAD_test")
def load_vqa_rad(split: str = "test", cache_dir: str = DATA_ROOT) -> DatasetDict:
    ds = load_dataset("flaviagiammarino/vqa-rad", split=split, cache_dir=cache_dir)
    ds = ds.cast_column("image", HFImage(decode=False))

    def _format(batch):
        out_msgs, out_imgs = [], []
        for img, q, a in zip(batch["image"], batch["question"], batch["answer"]):
            path = _coerce_image_to_path(img, "vqa_rad")
            img_list = [path] if path else []
            out_msgs.append(_build_messages(q, a, img_list))
            out_imgs.append(img_list if img_list else [])
        return {"messages": out_msgs, "images": out_imgs}

    ds = ds.map(_format, batched=True, batch_size=128, num_proc=1)
    ds = ds.remove_columns([c for c in ds.column_names if c not in {"messages", "images"}])
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("PMC_VQA_test")
def load_pmc_vqa(split: str = "test") -> DatasetDict:
    base = _pick_root(
        os.path.join(DATA_ROOT, "PMC-VQA"),
        os.path.join(DATA_ROOT_MEDMO, "PMC_VQA"),
        os.path.join(DATA_ROOT_MEDMO, "PMC-VQA"),
    )
    csv_map = {
        "train": os.path.join(base, "train.csv"),
        "validation": os.path.join(base, "test_clean.csv"),
        "test": os.path.join(base, "test_clean.csv"),
    }
    csv_path = csv_map.get(split)
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"PMC-VQA CSV not found for split '{split}': {csv_path}")

    df = pd.read_csv(csv_path)
    image_root = os.path.join(base, "images")
    alt_root = os.path.join(base, "figures")

    records = []
    for _, row in df.iterrows():
        fig = str(row.get("Figure_path", "")).strip()
        question = str(row.get("Question", "")).strip()
        answer = str(row.get("Answer", "")).strip()
        options = {
            "A": str(row.get("Choice A", "")).replace("A:", "").strip(),
            "B": str(row.get("Choice B", "")).replace("B:", "").strip(),
            "C": str(row.get("Choice C", "")).replace("C:", "").strip(),
            "D": str(row.get("Choice D", "")).replace("D:", "").strip(),
        }
        answer_label = str(row.get("Answer_label", "")).strip()
        if not answer:
            answer = _resolve_answer("", answer_label, options)
        question_fmt = _format_mcq(question, options)

        img_path = os.path.join(image_root, fig)
        if not os.path.exists(img_path):
            alt = os.path.join(alt_root, fig)
            img_path = alt if os.path.exists(alt) else img_path

        img_list = [img_path] if img_path else []
        records.append({"messages": _build_messages(question_fmt, answer, img_list), "images": img_list if img_list else []})

    ds = Dataset.from_list(records)
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


@register_dataset("MMMU-Medical-test_test")
def load_mmmu_medical_test(split: str = "test") -> DatasetDict:
    return _load_mmmu_medical(split="test")


@register_dataset("MMMU-Medical-val_test")
def load_mmmu_medical_val(split: str = "validation") -> DatasetDict:
    return _load_mmmu_medical(split="validation")


def _load_mmmu_medical(split: str = "test") -> DatasetDict:
    base = _pick_root(
        os.path.join(DATA_ROOT, "MMMU"),
        os.path.join(DATA_ROOT_MEDMO, "MMMU"),
    )
    subjects = [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ]

    records = []
    for subj in subjects:
        subdir = os.path.join(base, subj)
        if not os.path.isdir(subdir):
            continue
        parquet = os.path.join(subdir, f"{split}-00000-of-00001.parquet")
        if not os.path.exists(parquet):
            continue
        df = pd.read_parquet(parquet, engine="pyarrow")
        for _, row in df.iterrows():
            question = str(row.get("question", "")).strip()
            options_list = _parse_options_str(row.get("options"))
            answer_raw = row.get("answer")
            answer_text = _resolve_answer(answer_raw, str(answer_raw) if isinstance(answer_raw, str) else None, options_list)
            question_fmt = _format_mcq(question, options_list)

            image_paths = []
            for i in range(1, 8):
                key = f"image_{i}"
                if key not in row:
                    continue
                img = row.get(key)
                path = _coerce_image_to_path(img, f"mmmu_{subj.lower()}")
                if path:
                    image_paths.append(path)

            records.append({"messages": _build_messages(question_fmt, answer_text, image_paths), "images": image_paths if image_paths else []})

    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("OmniMedVQA_test")
def load_omnimedvqa(split: str = "test") -> DatasetDict:
    base = _pick_root(
        os.path.join(DATA_ROOT, "OmniMedVQA", "OmniMedVQA"),
        os.path.join(DATA_ROOT_MEDMO, "OmniMedVQA", "OmniMedVQA"),
        os.path.join(DATA_ROOT_MEDMO, "OmniMedVQA"),
    )
    open_dir = os.path.join(base, "QA_information", "Open-access")
    if not os.path.isdir(open_dir):
        raise FileNotFoundError(f"OmniMedVQA open-access dir not found: {open_dir}")

    records = []
    for file in os.listdir(open_dir):
        if not file.endswith(".json"):
            continue
        path = os.path.join(open_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ex in data:
            img_path = ex.get("image_path")
            if img_path and not os.path.isabs(img_path):
                img_path = os.path.join(base, img_path)
            question = ex.get("question", "")
            options = {}
            for k in ["A", "B", "C", "D"]:
                opt = ex.get(f"option_{k}")
                if opt:
                    options[k] = opt
            answer = ex.get("gt_answer", "")
            if answer in options:
                answer_text = options.get(answer, answer)
            else:
                answer_text = answer
            question_fmt = _format_mcq(question, options if options else None)
            img_list = [img_path] if img_path else []
            records.append({"messages": _build_messages(question_fmt, answer_text, img_list), "images": img_list if img_list else []})

    ds = Dataset.from_list(records)
    return DatasetDict({"train": _finalize_ds(ds)})


@register_dataset("SLAKE_test")
def load_slake(split: str = "test") -> DatasetDict:
    base = _pick_root(
        os.path.join(DATA_ROOT, "SLAKE"),
        os.path.join(DATA_ROOT_MEDMO, "SLAKE"),
    )
    json_path = os.path.join(base, f"{split}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"SLAKE json not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for ex in data:
        img = ex.get("img_name")
        question = ex.get("question", "")
        answer = ex.get("answer", "")
        img_path = os.path.join(base, "imgs", img) if img else None
        img_list = [img_path] if img_path else []
        records.append({"messages": _build_messages(question, answer, img_list), "images": img_list if img_list else []})

    ds = Dataset.from_list(records)
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
        "validation": os.path.join(base, "phrases_no_exclude_test.jsonl"),
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
        "train": os.path.join(base, "train.jsonl"),
        "validation": os.path.join(base, "dev.jsonl"),
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


@register_dataset("MedXpertQA-MM_test")
def load_medxpertqa_mm(split: str = "test") -> DatasetDict:
    base_root = _pick_root(
        os.path.join(DATA_ROOT, "MedXpertQA"),
        os.path.join(DATA_ROOT_MEDMO, "MedXpertQA"),
    )
    base = os.path.join(base_root, "MM")
    file_map = {
        "train": os.path.join(base, "train.jsonl"),
        "validation": os.path.join(base, "dev.jsonl"),
        "test": os.path.join(base, "test.jsonl"),
    }
    path = file_map.get(split)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"MedXpertQA-MM jsonl not found for split '{split}': {path}")

    image_root = os.path.join(base_root, "images")
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
            images = ex.get("images", [])
            img_paths = [os.path.join(image_root, im) for im in images]
            records.append({"messages": _build_messages(question_fmt, answer_text, img_paths), "images": img_paths if img_paths else []})
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

    csv_files = glob.glob(os.path.join(base, "*.csv"))
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
        "/vast/users/imran.razzak/Medmo_Dataset/Medmo_Dataset/CheXpert-Plus",
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
        "/vast/users/imran.razzak/Medmo_Dataset/Medmo_Dataset/MIMIC-CXR-Report",
        "/vast/users/imran.razzak/Medmo_Dataset/Medmo_Dataset/MIMIC-CXR-Report/Reports",
        "/vast/users/imran.razzak/Medmo_Dataset/Medmo_Dataset/MIMIC-CXR",
        os.path.join(DATA_ROOT, "MIMIC_CXR"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(DATA_ROOT, "MIMIC_CXR")


def _resolve_mimic_image_root(root: str) -> str:
    candidates = [
        os.path.join(DATA_ROOT_MEDMO, "MIMIC-CXR-JPG", "physionet.org", "files", "mimic-cxr-jpg", "2.1.0", "files"),
        "/vast/users/imran.razzak/Medmo_Dataset/Medmo_Dataset/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.1.0/files",
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
        "/vast/users/imran.razzak/Medmo_Dataset/Medmo_Dataset/MIMIC-CXR-Report/Reports/mimic_reports_with_images_clean_val.csv",
        os.path.join(root, "Reports", "mimic_reports_with_images_clean_val.csv"),
        os.path.join(root, "mimic_reports_with_images_clean_val.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError("MIMIC-CXR CSV not found. Set MIMIC_CXR_ROOT or place mimic_reports_with_images_clean_val.csv in dataset root.")


def _resolve_medtrinity_paths() -> tuple[str, str]:
    default_root = "/vast/users/imran.razzak/Medmo_Dataset/Medmo_Dataset/MedTrinity-25M/25M_clean"
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
    split: str = "test",
    interleave: bool = True,
    weights: Optional[List[float]] = None,
    seed: int = 42,
) -> DatasetDict:
    if not names:
        raise ValueError("Provide at least one dataset name.")
    for n in names:
        if n not in _LOADER_REGISTRY:
            raise KeyError(f"Unknown dataset '{n}'. Available: {available_datasets()}")

    ddicts = [_LOADER_REGISTRY[n](split=split) for n in names]

    # merge only "train" splits returned by loaders
    datasets = [d["train"] for d in ddicts if "train" in d]
    if not datasets:
        raise ValueError("No datasets produced train split.")

    if len(datasets) == 1:
        merged = datasets[0]
    else:
        if interleave:
            probs = None
            if weights is not None:
                if len(weights) != len(datasets):
                    raise ValueError("weights length must match number of datasets")
                s = float(sum(weights))
                probs = [w / s for w in weights]
            from datasets import interleave_datasets
            merged = interleave_datasets(datasets, probabilities=probs, seed=seed)
        else:
            from datasets import concatenate_datasets
            merged = concatenate_datasets(datasets)

    merged = _finalize_ds(merged)
    empty = _finalize_ds(Dataset.from_dict({"messages": [], "images": []}))
    return DatasetDict({
        "train": merged,
        "validation": empty,
        "test": merged if split == "test" else empty,
    })


if __name__ == "__main__":
    print("Available datasets:", available_datasets())
