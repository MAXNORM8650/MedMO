# medevalkit_loader.py
# Loader for MedEvalKit datasets -> GRPO schema

import os
import json
import glob
from typing import Dict, List, Optional, Callable, Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

DATA_ROOT = "/vast/users/imran.razzak/Document/MedEvalKit/datas"

# --------------------------
# Registry
# --------------------------
_LOADER_REGISTRY: Dict[str, Callable[..., Dataset]] = {}


def register_dataset(name: str):
    def _wrap(fn: Callable[..., Dataset]):
        _LOADER_REGISTRY[name] = fn
        return fn
    return _wrap


def available_datasets() -> List[str]:
    return sorted(_LOADER_REGISTRY.keys())


# --------------------------
# Formatting helpers
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
    # If answer is already text, prefer it.
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    if answer_label and options:
        if isinstance(options, dict):
            return str(options.get(answer_label, answer_label))
        # list
        idx = ord(answer_label.upper()) - ord("A")
        if 0 <= idx < len(options):
            return str(options[idx])
    return "" if answer is None else str(answer)


def _ensure_image_list(img: Any, image_root: Optional[str] = None) -> List[Any]:
    if img is None:
        return []
    # list of paths or PILs
    if isinstance(img, list):
        out = []
        for v in img:
            if isinstance(v, str) and image_root and not os.path.isabs(v):
                out.append(os.path.join(image_root, v))
            else:
                out.append(v)
        return out
    if isinstance(img, str):
        return [os.path.join(image_root, img) if image_root and not os.path.isabs(img) else img]
    return [img]


def _make_dataset(records: List[Dict[str, Any]]) -> Dataset:
    if not records:
        return Dataset.from_dict({"image": [], "problem": [], "solution": [], "original_question": [], "original_answer": []})
    return Dataset.from_list(records)


def _standard_record(image, question, answer) -> Dict[str, Any]:
    return {
        "image": _ensure_image_list(image),
        "problem": (question or "").strip(),
        "solution": (answer or "").strip(),
        "original_question": (question or "").strip(),
        "original_answer": (answer or "").strip(),
    }


# --------------------------
# Dataset loaders
# --------------------------

@register_dataset("PATH_VQA")
def load_path_vqa(split: str = "test", cache_dir: str = DATA_ROOT) -> Dataset:
    ds = load_dataset("flaviagiammarino/path-vqa", split=split, cache_dir=cache_dir)
    records = [_standard_record(ex.get("image"), ex.get("question"), ex.get("answer")) for ex in ds]
    return _make_dataset(records)


@register_dataset("VQA_RAD")
def load_vqa_rad(split: str = "test", cache_dir: str = DATA_ROOT) -> Dataset:
    ds = load_dataset("flaviagiammarino/vqa-rad", split=split, cache_dir=cache_dir)
    records = [_standard_record(ex.get("image"), ex.get("question"), ex.get("answer")) for ex in ds]
    return _make_dataset(records)


@register_dataset("MedFrameQA")
def load_medframeqa(split: str = "test", cache_dir: str = DATA_ROOT) -> Dataset:
    ds = load_dataset("SuhaoYu1020/MedFrameQA", split=split, cache_dir=cache_dir)
    records = []
    for ex in ds:
        question = ex.get("question") or ex.get("Question") or ""
        answer = ex.get("answer") or ex.get("Answer") or ""
        image = ex.get("image") or ex.get("Image")
        records.append(_standard_record(image, question, answer))
    return _make_dataset(records)


@register_dataset("PMC_VQA")
def load_pmc_vqa(split: str = "test", cache_dir: str = DATA_ROOT) -> Dataset:
    base = os.path.join(DATA_ROOT, "PMC-VQA")
    csv_map = {
        "train": os.path.join(base, "train.csv"),
        "validation": os.path.join(base, "test_clean.csv"),
        "test": os.path.join(base, "test_clean.csv"),
    }
    csv_path = csv_map.get(split)
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"PMC-VQA CSV not found for split '{split}': {csv_path}")

    df = pd.read_csv(csv_path)
    records = []
    image_root = os.path.join(base, "images")
    alt_root = os.path.join(base, "figures")

    for _, row in df.iterrows():
        fig = str(row.get("Figure_path", "")).strip()
        question = str(row.get("Question", "")).strip()
        answer = str(row.get("Answer", "")).strip()
        # optional choices
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

        records.append(_standard_record(img_path, question_fmt, answer))

    return _make_dataset(records)


@register_dataset("MedQA_USMLE")
def load_medqa_usmle(split: str = "test", cache_dir: str = DATA_ROOT) -> Dataset:
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
            records.append(_standard_record([], question_fmt, answer_text))

    return _make_dataset(records)


@register_dataset("MedMCQA")
def load_medmcqa(split: str = "test", cache_dir: str = DATA_ROOT) -> Dataset:
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
        records.append(_standard_record([], question_fmt, answer_text))

    return _make_dataset(records)


@register_dataset("PubMedQA")
def load_pubmedqa(split: str = "test", cache_dir: str = DATA_ROOT) -> Dataset:
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
        records.append(_standard_record([], question_fmt, answer_text))

    return _make_dataset(records)


@register_dataset("Medbullets_op4")
def load_medbullets_op4(split: str = "train", cache_dir: str = DATA_ROOT) -> Dataset:
    ds = load_dataset(os.path.join(DATA_ROOT, "Medical-Eval-MedBullets_op4"), split=split)
    records = []
    for ex in ds:
        question = ex.get("question", "")
        options = ex.get("options", {})
        answer = ex.get("answer", "")
        answer_idx = ex.get("answer_idx", "")
        answer_text = _resolve_answer(answer, answer_idx, options)
        question_fmt = _format_mcq(question, options)
        records.append(_standard_record([], question_fmt, answer_text))

    return _make_dataset(records)


@register_dataset("Medbullets_op5")
def load_medbullets_op5(split: str = "train", cache_dir: str = DATA_ROOT) -> Dataset:
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
        records.append(_standard_record([], question_fmt, answer_text))

    return _make_dataset(records)


@register_dataset("MedXpertQA-Text")
def load_medxpertqa_text(split: str = "test", cache_dir: str = DATA_ROOT) -> Dataset:
    base = os.path.join(DATA_ROOT, "MedXpertQA", "Text")
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
            records.append(_standard_record([], question_fmt, answer_text))

    return _make_dataset(records)


@register_dataset("MedXpertQA-MM")
def load_medxpertqa_mm(split: str = "test", cache_dir: str = DATA_ROOT) -> Dataset:
    base = os.path.join(DATA_ROOT, "MedXpertQA", "MM")
    file_map = {
        "train": os.path.join(base, "train.jsonl"),
        "validation": os.path.join(base, "dev.jsonl"),
        "test": os.path.join(base, "test.jsonl"),
    }
    path = file_map.get(split)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"MedXpertQA-MM jsonl not found for split '{split}': {path}")

    image_root = os.path.join(DATA_ROOT, "MedXpertQA", "images")
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
            img_paths = []
            for im in images:
                p = os.path.join(image_root, im)
                img_paths.append(p)
            records.append(_standard_record(img_paths, question_fmt, answer_text))

    return _make_dataset(records)


@register_dataset("MedQA_MCMLE")
def load_medqa_mcmle(split: str = "train", cache_dir: str = DATA_ROOT) -> Dataset:
    ds = load_dataset("shuyuej/MedQA-MCMLE-Benchmark", split=split, cache_dir=cache_dir)
    records = []
    for ex in ds:
        question = ex.get("question", "")
        options = ex.get("options", {})
        answer = ex.get("answer", "")
        answer_idx = ex.get("answer_idx", "")
        answer_text = _resolve_answer(answer, answer_idx, options)
        question_fmt = _format_mcq(question, options)
        records.append(_standard_record([], question_fmt, answer_text))
    return _make_dataset(records)


@register_dataset("SuperGPQA")
def load_supergpqa(split: str = "train", cache_dir: str = DATA_ROOT) -> Dataset:
    ds = load_dataset("m-a-p/SuperGPQA", split=split, cache_dir=cache_dir)
    records = []
    for ex in ds:
        question = ex.get("question", "")
        options = ex.get("options", [])
        answer = ex.get("answer", "")
        answer_label = ex.get("answer_letter", "")
        answer_text = _resolve_answer(answer, answer_label, options)
        question_fmt = _format_mcq(question, options)
        records.append(_standard_record([], question_fmt, answer_text))
    return _make_dataset(records)


@register_dataset("IU_XRAY")
def load_iu_xray(split: str = "test", cache_dir: str = DATA_ROOT) -> Dataset:
    base = os.path.join(DATA_ROOT, "IU_XRAY")
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
        records.append(_standard_record(img_paths, question, answer))
    return _make_dataset(records)


@register_dataset("CMB")
def load_cmb(split: str = "test", cache_dir: str = DATA_ROOT) -> Dataset:
    base = os.path.join(DATA_ROOT, "CMB")
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
            records.append(_standard_record([], question, a))
    return _make_dataset(records)


@register_dataset("CMExam")
def load_cmexam(split: str = "test", cache_dir: str = DATA_ROOT) -> Dataset:
    base = os.path.join(DATA_ROOT, "CMExam")
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
            records.append(_standard_record([], question_fmt, answer_text))
    return _make_dataset(records)


@register_dataset("CMMLU")
def load_cmmlu(split: str = "test", cache_dir: str = DATA_ROOT, subjects: Optional[List[str]] = None) -> Dataset:
    base = os.path.join(DATA_ROOT, "cmmlu", split)
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
            records.append(_standard_record([], question_fmt, answer_text))
    return _make_dataset(records)


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

    datasets = [ _LOADER_REGISTRY[n](split=split) for n in names ]

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

    empty = Dataset.from_dict({"image": [], "problem": [], "solution": [], "original_question": [], "original_answer": []})
    return DatasetDict({
        "train": merged,
        "validation": empty,
        "test": merged if split == "test" else empty,
    })


if __name__ == "__main__":
    # quick sanity check
    print("Available datasets:", available_datasets())
