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


accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm_gemma3.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path lingshu-medical-mllm/Lingshu-7B \
    --per_device_train_batch_size 1 \
    --output_dir sft-grpo \
    --dtype bfloat16 \
    --attn_implementation sdpa  --per_device_train_batch_size 1  --remove_unused_columns False

Train Gemma 3 on the FanqingM/MMIU-Benchmark dataset (multi-image).

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm_gemma3.py \
    --dataset_name FanqingM/MMIU-Benchmark \
    --dataset_train_split test \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --output_dir Gemma-3-4B-SFT-MMIU \
    --dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear \
    --attn_implementation eager
    
    
    
 AMD 
    
accelerate launch   --mixed_precision=fp16   --num_processes 4   examples/scripts/sft_vlm_gemma3.py   --model_name_or_path google/gemma-3-4b-it   --dataset_name HuggingFaceH4/llava-instruct-mix-vsft   --per_device_train_batch_size 1   --output_dir Gemma-3-4B-SFT-MMIU   --attn_implementation eager   --fsdp "full_shard auto_wrap"   --fsdp_transformer_layer_cls_to_wrap "Gemma3DecoderLayer"   --fp16 True --bf16 False
"""

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
from medmo_loader import load_datasets  
# from grounding_sft_trainer import GroundingSFTTrainer, BBoxHead, DinoLossCfg
from trl.trainer.grounding_sft_trainer import GroundingSFTTrainer, BBoxHead, DinoLossCfg


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

import os, torch.distributed as dist
# from trl.rewards.all_rewards import get_reward_funcs

def barrier_if_needed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

rank = int(os.environ.get("RANK", 0))



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
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.max_length = None
    
    # if rank == 0:
    #     breakpoint()



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
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    hidden_size = model.config.vision_config.out_hidden_size
    bbox_head = BBoxHead(hidden_size)

    

    # if dataset_name == "iuxray_report":
    # dataset = load_iuxray_report_chat_indexed_paths(16, batch_size=4)

    dataset = load_datasets(
        # names=["medtrinity_report", "iuxray_report", "mimiccxr_report", "medpix_cliqa_report", "roco_report", "rocov2_report",
        # "chexpert_plus_report", "vqa_med_2019", "pubmed_vision", "nih_vqa", "quilt_llava_pretrain", "mimic_cxr_vqa"],
        # "nih_bbox", "deeplesion_bbox", "grazpedwri_dx_bbox", "bacteria_bbox", "ctc_bbox", "deepcell_bbox", 
        
        # "pmc_instruct_qa", "medquad_qa", "medqa", "medical_meadow_medqa", "alphacare_qa", "chatdoctor_healthcaremagic",
        # "chatdoctor_icliniq", "chatdoc_medqa_4option", "chatdoc_medqa_5option", "medical_meadow_pubmed_causal",
        # "medical_meadow_flashcard", "medical_meadow_mediqa", "medical_meadow_wikidoc", "medical_meadow_wikidoc_patient_information"
        # "mmmlu_anatomy", "mmmlu_clinical_knowledge", "mmmlu_college_biology", "mmmlu_college_medicine", "mmmlu_medical_genetics",
        # "mmmlu_professional_medicine", "chatdoc_4option", "chatdoc_5option", "medical_meadow_mmmlu",
        
        # "medical_meadow_cord19", "mimic_ext_bhc"
        # "medical_o1_sft_mix", "medical_o1_verifiable_problem", "medical_r1_distill", "medreason"
        names=["nih_bbox"],
        num_proc=16,
        batch_size=1,
        interleave=True,
        weights=None,  # or e.g. [0.7, 0.3]
    )
    print(dataset['train'][0])
    if rank == 0:
        breakpoint()

    ################
    # Training
    ################
    # trainer = SFTTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset[script_args.dataset_train_split],
    #     eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
    #     peft_config=get_peft_config(model_args),
    # )
    
    
    trainer = GroundingSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        # data_collator=my_collator,  # emits the four extra keys described above
        bbox_head=bbox_head,
        dino_cfg=DinoLossCfg(lambda_bbox=0.5),  # tune this weight
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    main()




