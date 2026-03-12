# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "Pillow>=9.4.0",
#     "peft",
#     "trackio",
# ]
# ///

"""
Train Gemma 3 on the HuggingFaceH4/llava-instruct-mix-vsft dataset (single-image).

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm_gemma3.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --output_dir Gemma-3-4B-SFT-MMIU \
    --dtype bfloat16 \
    --attn_implementation eager

Train Gemma 3 on the FanqingM/MMIU-Benchmark dataset (multi-image).

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm_gemma3.py \
    --dataset_name FanqingM/MMIU-Benchmark \
    --dataset_train_split test \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --per_device_train_batch_size 1 \
    --output_dir Gemma-3-4B-SFT-MMIU \
    --dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear \
    --attn_implementation eager
    
    
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml examples/scripts/sft_vlm_gemma3.py --dataset_name FanqingM/MMIU-Benchmark --dataset_train_split test --model_name_or_path google/gemma-3-4b-it --per_device_train_batch_size 1 --output_dir Gemma-3-4B-SFT-MMIU --dtype bfloat16  --per_device_train_batch_size 2  --output_dir sft-medmo     --dtype bfloat16  --attn_implementation sdpa  --per_device_eval_batch_size 2
    
 AMD 
    
accelerate launch   --mixed_precision=fp16   --num_processes 4   examples/scripts/sft_vlm_gemma3.py   --model_name_or_path google/gemma-3-4b-it   --dataset_name HuggingFaceH4/llava-instruct-mix-vsft   --per_device_train_batch_size 1   --output_dir Gemma-3-4B-SFT-MMIU   --attn_implementation eager   --fsdp "full_shard auto_wrap"   --fsdp_transformer_layer_cls_to_wrap "Gemma3DecoderLayer"   --fp16 True --bf16 False
"""



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
import os
import zipfile

import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
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
# from medevalkit_sft_loader import load_datasets

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

import time

import os, torch.distributed as dist
# from trl.rewards.all_rewards import get_reward_funcs

def barrier_if_needed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

rank = int(os.environ.get("RANK", 0))


# import os
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890/"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890/"
# os.environ["ALL_PROXY"] = "socks5://127.0.0.1:7890"

# For multi-image example
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
                {"role": "system", "content": [{"type": "text", "text": samples["context"][cont]}]},
                {"role": "user", "content": images + [{"type": "text", "text": samples["question"][cont]}]},
                {"role": "assistant", "content": [{"type": "text", "text": samples["output"][cont]}]},
            ]
        )
    return formatted_samples


# For multi-image example
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

from datasets import Dataset, DatasetDict
from typing import Union

def limit_debug(ds: Union[Dataset, DatasetDict], n: int = 10, shuffle: bool = False, seed: int = 0):
    """
    Return a copy of the dataset (or each split) limited to the first n examples.
    Optionally shuffle before selecting.
    """
    if isinstance(ds, DatasetDict):
        out = {}
        for split, d in ds.items():
            d2 = d.shuffle(seed=seed) if shuffle and len(d) > n else d
            out[split] = d2.select(range(min(n, len(d2))))
        return DatasetDict(out)
    else:
        ds2 = ds.shuffle(seed=seed) if shuffle and len(ds) > n else ds
        return ds2.select(range(min(n, len(ds2))))





def main():
    global rank
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Do not force sequence truncation for VLM batches; this can desync
    # image placeholder tokens in text vs tokenized ids.
    # If you need a cap, pass --max_length explicitly at launch time.
    training_args.max_length = None
    

    ################
    # Dataset
    ###############
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # if script_args.dataset_name == "FanqingM/MMIU-Benchmark":
    #     dataset = prepare_dataset(dataset, script_args.dataset_name)
    # if script_args.dataset_name == "FanqingM/MMIU-Benchmark":
    #     dataset = prepare_dataset(dataset, script_args.dataset_name, debug_limit=10)
    # breakpoint()

    ################
    # Model, Tokenizer & Processor
    ################
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

    # "roco_report", "rocov2_report"

    
    eval_dataset_names = [
            #    "pmc_instruct_qa", "medquad_qa", "medqa", "medical_meadow_medqa", "alphacare_qa", "chatdoctor_healthcaremagic",
                #  "chatdoctor_icliniq", "chatdoc_medqa_4option", "chatdoc_medqa_5option", "medical_meadow_pubmed_causal",
                #  "medical_meadow_flashcard", "medical_meadow_mediqa", "medical_meadow_wikidoc", "medical_meadow_wikidoc_patient_information",
                #  "medical_meadow_cord19", "mimic_ext_bhc", "chatdoc_4option", "chatdoc_5option", 
                #  "medical_o1_sft_mix", "medical_o1_verifiable_problem", "medical_r1_distill", "medreason",
                # "medtrinity_report",
                "mimiccxr_report", "iuxray_report", "roco_report", "rocov2_report", "chexpert_plus_report", "mimic_cxr_vqa","roco_report",
                "nih_vqa", "quilt_llava_pretrain", 
                "mmmu_med", "slake", "vqa_rad", "path_vqa", "pmc_vqa",
                "IU_XRAY_test", "CheXpert_Plus_test", "vqa_med_2019", "pubmed_vision", "medpix_cliqa_report",
                "medsg_bbox", "bacteria_bbox_resize","ctc_bbox_resize","slake_bbox","nih_bbox", "deeplesion_bbox", "grazpedwri_dx_bbox_resize",
                
               

                # "mmmu_med", "slake", "vqa_rad", "path_vqa", "pmc_vqa", "omnimed_vqa", "medxpertqa_mm", "iu_xray_test", "nih_bbox", "medsg_bbox", "bacteria_bbox_resize"
                # "medbullets_op4", "medbullets_op5", "medxpertqa_text", "super_gpqa", "medqa_usmle", "medmcqa", "pubmedqa",
            
    ]
    if rank == 0:
        print(f"[dataset] loading {len(eval_dataset_names)} datasets...")
        print(f"[dataset] names={eval_dataset_names}")
    load_t0 = time.time()
    dataset = load_datasets_interleave(
        names=eval_dataset_names,
        num_proc=1,
        batch_size=128,
        interleave=False,  # concatenate to avoid oversampling
        filter_invalid=False,  # keep full set; enable if you need to drop bad samples
        # verbose=(rank == 0),
        # weights=None,  # or e.g. [0.7, 0.3]
    )
    if rank == 0:
        print(f"[dataset] merged dataset ready in {time.time() - load_t0:.1f}s")

    
    # Keep truncation disabled unless user explicitly sets --max_length.
    if dist.is_initialized():
        rank = dist.get_rank()
        print(f"[Rank {rank}] Dataset loaded, synchronizing...")
        
        # Small delay to ensure all ranks are ready
        time.sleep(2)
        
        # Synchronize
        dist.barrier()
        print(f"[Rank {rank}] All ranks synchronized!")
    
    
    # processing_class = AutoProcessor.from_pretrained(
    #     model_args.model_name_or_path, 
    #     local_files_only=True
    # )
    # processing_class.image_processor.min_pixels = 224 * 224
    # processing_class.image_processor.max_pixels = 768 * 768
    # Add these tokenizer settings to prevent truncation of image tokens
    # Set the tokenizer's model_max_length
    # processing_class.tokenizer.model_max_length = 1024
    # processing_class.tokenizer.truncation_side = "right"
    # processing_class.tokenizer.padding_side = "right"
        

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    main()

