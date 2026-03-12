import warnings
import os
import sys

# ============================================================================
# CRITICAL: Apply patches BEFORE any transformers imports
# ============================================================================

# Patch 1: PIL image handling
import PIL.Image
import PIL.ImageOps

PIL.Image.LOAD_TRUNCATED_IMAGES = True

_orig_exif = PIL.ImageOps.exif_transpose
def safe_exif_transpose(image):
    try:
        return _orig_exif(image)
    except Exception:
        return image
PIL.ImageOps.exif_transpose = safe_exif_transpose

# Patch 2: Trainer to handle None batches
from transformers import Trainer
import torch

_original_training_step = Trainer.training_step

def safe_training_step(self, model, inputs, *args, **kwargs):
    """Skip None batches from corrupted images."""
    if inputs is None:
        warnings.warn("Skipping batch - all samples corrupted")
        return torch.tensor(0.0, device=model.device, requires_grad=True)
    return _original_training_step(self, model, inputs, *args, **kwargs)

Trainer.training_step = safe_training_step

print("=" * 80)
print("✅ Image corruption handling patches applied")
print("=" * 80)

# ============================================================================
# Now safe to import everything else
# ============================================================================

import io
import math
import os
import zipfile

import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForImageTextToText

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

import os
import io
import random
import hashlib
import pandas as pd
from PIL import Image
from datasets import Dataset, DatasetDict
from medmo_loader_1 import load_datasets_interleave, load_datasets

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

import time
import os, torch.distributed as dist

def barrier_if_needed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

rank = int(os.environ.get("RANK", 0))


# =============================================================================
# Layerwise LR Trainer
# =============================================================================

class LayerwiseSFTTrainer(SFTTrainer):
    """
    SFTTrainer with per-component learning rates:

        model.model.language_model   →  1e-6  (10e-7)
        model.lm_head                →  1e-6  (10e-7)
        model.model.visual (encoder) →  1e-4  (10e-5)
        model.model.visual.merger    →  1e-3  (10e-4)
        model.model.visual.deepstack_merger_list → 1e-4 (10e-5)

    Note: "10e-X" in scientific notation equals 10 × 10^-X.
          10e-7 = 1e-6, 10e-5 = 1e-4, 10e-4 = 1e-3.
          The values below use standard Python float notation.
    """

    # ── LR constants (10e-X notation converted to floats) ──────────────────
    LR_LLM        = 1e-7   # 10e-7
    LR_LM_HEAD    = 1e-7   # 10e-7
    LR_ENCODER    = 1e-4   # 10e-5
    LR_MERGER     = 1e-3   # 10e-4
    LR_DS_MERGER  = 1e-3   # 10e-5  (deepstack_merger_list)
    MIN_LR_RATIO  = 0.01   # cosine floor = 1% of peak lr

    def create_optimizer(self):
        model = self.model

        # ── Collect named parameter groups ─────────────────────────────────
        # Order matters: more specific checks first.
        llm_params       = []
        lm_head_params   = []
        merger_params    = []
        ds_merger_params = []
        encoder_params   = []
        other_params     = []

        # Build a set of parameter ids already assigned to avoid double-counting
        assigned = set()

        def _add(params_list, named_iter):
            for name, param in named_iter:
                if id(param) not in assigned and param.requires_grad:
                    params_list.append((name, param))
                    assigned.add(id(param))

        # 1. merger (most specific visual sub-module — check before encoder)
        _add(merger_params,    model.model.visual.merger.named_parameters())

        # 2. deepstack_merger_list
        if hasattr(model.model.visual, "deepstack_merger_list"):
            _add(ds_merger_params, model.model.visual.deepstack_merger_list.named_parameters())

        # 3. visual encoder (everything else in visual)
        _add(encoder_params,   model.model.visual.named_parameters())

        # 4. language model
        _add(llm_params,       model.model.language_model.named_parameters())

        # 5. lm_head
        _add(lm_head_params,   model.lm_head.named_parameters())

        # 6. anything not yet assigned (e.g. top-level embed_tokens outside LLM)
        _add(other_params,     model.named_parameters())

        def _to_params(named_list):
            return [p for _, p in named_list]

        param_groups = []
        if merger_params:
            param_groups.append({
                "params": _to_params(merger_params),
                "lr": self.LR_MERGER,
                "name": "merger",
            })
        if ds_merger_params:
            param_groups.append({
                "params": _to_params(ds_merger_params),
                "lr": self.LR_DS_MERGER,
                "name": "deepstack_merger",
            })
        if encoder_params:
            param_groups.append({
                "params": _to_params(encoder_params),
                "lr": self.LR_ENCODER,
                "name": "visual_encoder",
            })
        if llm_params:
            param_groups.append({
                "params": _to_params(llm_params),
                "lr": self.LR_LLM,
                "name": "language_model",
            })
        if lm_head_params:
            param_groups.append({
                "params": _to_params(lm_head_params),
                "lr": self.LR_LM_HEAD,
                "name": "lm_head",
            })
        if other_params:
            param_groups.append({
                "params": _to_params(other_params),
                "lr": self.LR_ENCODER,   # fallback: encoder lr
                "name": "other",
            })

        # ── Log group sizes on rank 0 ───────────────────────────────────────
        current_rank = dist.get_rank() if dist.is_initialized() else 0
        if current_rank == 0:
            print("\n" + "=" * 70)
            print("LayerwiseSFTTrainer — parameter groups")
            print("=" * 70)
            for g in param_groups:
                numel = sum(p.numel() for p in g["params"])
                print(f"  {g['name']:25s}  #tensors={len(g['params']):5d}  "
                      f"numel={numel:>12,}  lr={g['lr']:.2e}")
            print("=" * 70 + "\n")

        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Cosine decay with linear warmup and a minimum LR floor.
        Each param group decays from its own peak LR down to peak * MIN_LR_RATIO.
        """
        if optimizer is None:
            optimizer = self.optimizer

        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
        min_ratio = self.MIN_LR_RATIO

        def lr_lambda(current_step: int) -> float:
            # Linear warmup
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            # Cosine decay
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_ratio, cosine_val)

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
        self._created_lr_scheduler = True
        return self.lr_scheduler


# =============================================================================
# Helper functions (unchanged from original)
# =============================================================================

def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                if image is not None:
                    image = Image.open(io.BytesIO(image["bytes"]))
                    image_inputs.append(image.convert("RGB"))
    return image_inputs


def format_data(samples: dict[str, any]) -> dict[str, list]:
    formatted_samples = {"messages": []}
    for cont in range(len(samples["question"])):
        images = []
        for img_path in samples["input_image_path"][cont]:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append({"type": "image", "image": image})
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

        formatted_samples["messages"].append(
            [
                {"role": "system",    "content": [{"type": "text", "text": samples["context"][cont]}]},
                {"role": "user",      "content": images + [{"type": "text", "text": samples["question"][cont]}]},
                {"role": "assistant", "content": [{"type": "text", "text": samples["output"][cont]}]},
            ]
        )
    return formatted_samples


def prepare_dataset(dataset: DatasetDict, dataset_name: str, debug_limit: int | None = 10) -> DatasetDict:
    all_files = list_repo_files(dataset_name, repo_type="dataset")
    zip_files = [f for f in all_files if f.endswith(".zip")]

    for zip_filename in zip_files:
        zip_path = hf_hub_download(repo_id=dataset_name, filename=zip_filename, repo_type="dataset")
        extract_folder = zip_filename.replace(".zip", "")
        os.makedirs(extract_folder, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

    if debug_limit is not None:
        dataset = limit_debug(dataset, n=debug_limit, shuffle=False)

    dataset = dataset.map(format_data, batched=True, batch_size=4, num_proc=16)
    return dataset


from typing import Union

def limit_debug(ds: Union[Dataset, DatasetDict], n: int = 10, shuffle: bool = False, seed: int = 0):
    if isinstance(ds, DatasetDict):
        out = {}
        for split, d in ds.items():
            d2 = d.shuffle(seed=seed) if shuffle and len(d) > n else d
            out[split] = d2.select(range(min(n, len(d2))))
        return DatasetDict(out)
    else:
        ds2 = ds.shuffle(seed=seed) if shuffle and len(ds) > n else ds
        return ds2.select(range(min(n, len(ds2))))


# =============================================================================
# Main
# =============================================================================

def main():
    global rank
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.max_length = None

    # ── Model ────────────────────────────────────────────────────────────────
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, local_files_only=True, trust_remote_code=True, **model_kwargs
    )
    model = model.cuda()

    # ── Dataset ──────────────────────────────────────────────────────────────
    eval_dataset_names = [
        "slake",
    ]
    if rank == 0:
        print(f"[dataset] loading {len(eval_dataset_names)} datasets...")
        print(f"[dataset] names={eval_dataset_names}")

    load_t0 = time.time()
    dataset = load_datasets_interleave(
        names=eval_dataset_names,
        num_proc=1,
        batch_size=128,
        interleave=False,
        filter_invalid=False,
    )
    if rank == 0:
        print(f"[dataset] merged dataset ready in {time.time() - load_t0:.1f}s")

    print(dataset['train'][0])

    # ── Unfreeze ALL parameters — LayerwiseSFTTrainer handles LR per group ──
    # (previously some params were frozen; now we train everything but at
    #  very low LR for the LLM, e.g. 1e-6, so it doesn't diverge)
    for param in model.parameters():
        param.requires_grad = True

    # ── Distributed barrier ──────────────────────────────────────────────────
    if dist.is_initialized():
        rank = dist.get_rank()
        print(f"[Rank {rank}] Dataset loaded, synchronizing...")
        time.sleep(2)
        dist.barrier()
        print(f"[Rank {rank}] All ranks synchronized!")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = LayerwiseSFTTrainer(       # ← key change
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    main()