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
#     "Pillow",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "torchvision",
#     "trackio",
# ]
# ///

"""
pip install math_verify

# For Qwen/Qwen2.5-VL-3B-Instruct
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_vlm.py \
    --model_name_or_path lingshu-medical-mllm/Lingshu-7B \
    --output_dir grpo-Qwen2.5-VL-3B-Instruct \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --dtype bfloat16 \
    --max_prompt_length 4096 \
    --max_completion_length 1024 \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions

# For HuggingFaceTB/SmolVLM2-2.2B-Instruct
pip install num2words

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_vlm.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --output_dir grpo-SmolVLM2-2.2B-Instruct \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_prompt_length 4096 \
    --max_completion_length 1024 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_generations 2

"""

import os

import torch
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import think_format_reward, get_soft_overlong_punishment
from trl.rewards.bbox_rewards import label_accuracy_reward, bbox_iou_reward, tag_count_reward, tag_count_reward_strict, compute_bbox_rewards_dino # accuracy_reward_iou_mixed
# from trl.rewards import tag_count_reward
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


import os
import io
import random
import hashlib
import pandas as pd
from PIL import Image
from datasets import Dataset, DatasetDict
import re
# from medmo_loader import load_datasets as load_datasets_local 
from medmo_loader_1 import load_datasets_grpo as load_datasets_local  

import os, torch.distributed as dist
# from trl.rewards.all_rewards import get_reward_funcs

def barrier_if_needed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

rank = int(os.environ.get("RANK", 0))

MIN_VLM_PROMPT_TOKENS = 4096

_IMAGE_TAG_RE = re.compile(r"<\s*\|?\s*image[^>]*>", re.IGNORECASE)


def _strip_inline_image_tokens(text: str) -> str:
    """Remove inline image placeholders like <image>, <image 1>, <|image_1|>."""
    if not isinstance(text, str):
        return ""
    cleaned = _IMAGE_TAG_RE.sub("Image", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _coerce_single_image(images):
    """
    Force a single-image list. If multiple images are present, keep the first valid path.
    Returns [] if none.
    """
    if images is None:
        return []
    if isinstance(images, str):
        return [images] if images.strip() else []
    if isinstance(images, (list, tuple)):
        flat = []
        for item in images:
            # unwrap singleton containers
            while isinstance(item, (list, tuple)) and len(item) == 1:
                item = item[0]
            if isinstance(item, str) and item.strip():
                flat.append(item.strip())
                break
            if isinstance(item, (list, tuple)):
                for sub in item:
                    if isinstance(sub, str) and sub.strip():
                        flat.append(sub.strip())
                        break
            if flat:
                break
        return [flat[0]] if flat else []
    return []


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # if training_args.max_prompt_length is None or training_args.max_prompt_length < MIN_VLM_PROMPT_TOKENS:
    #     if rank == 0:
    #         print(
    #             f"[info] Increasing max_prompt_length to {MIN_VLM_PROMPT_TOKENS} tokens "
    #             "to ensure there is enough room for image patch tokens."
    #         )
    #     training_args.max_prompt_length = MIN_VLM_PROMPT_TOKENS
    ################
    # Model & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    quantization_config = get_quantization_config(model_args)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # print(123345)
    # if rank == 0:
    #     breakpoint()
    
    # model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     "/vast/users/imran.razzak/Document/medmo/trl/Medmo_SFT4",
    #     torch_dtype=torch.bfloat16,
    #     # attn_implementation="flash_attention_2",
    #     # device_map="cuda",
    # )

    # processor = AutoProcessor.from_pretrained("/vast/users/imran.razzak/Document/medmo/trl/Medmo_SFT4")

    ################
    # Dataset
    ################
    # dataset = load_dataset("lmms-lab/multimodal-open-r1-8k-verified", split="train")
    # dataset = dataset.train_test_split(test_size=100, seed=42)
    
    dataset_names = ["nih_vqa_grpo", "nih_bbox_grpo", "deeplesion_bbox_grpo", "bacteria_bbox_grpo", "ctc_bbox_grpo", "grazpedwri_dx_bbox_grpo", "medsg_bbox_grpo", "mimic_cxr_vqa_grpo", "deepcell_bbox_grpo"
    "nih_vqa_grpo", "nih_bbox_grpo", "deeplesion_bbox_grpo", "bacteria_bbox_grpo", "ctc_bbox_grpo", "grazpedwri_dx_bbox_grpo"]

    dataset = load_datasets_local(
        names=dataset_names,
        num_proc=16,
        batch_size=128,
        # interleave=True,
        # weights=None,  # or e.g. [0.7, 0.3]
    )

    # Force single image per sample to avoid image-token mismatch in Qwen3-VL.
    dataset = dataset.map(lambda ex: {"image": _coerce_single_image(ex.get("image"))})
    
    # if rank==0:
    #     breakpoint()
    

    # SYSTEM_PROMPT = (
    #     "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
    #     "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    #     "The reasoning process and answer are enclosed within <think></think> tags, i.e., <think>\nThis is my "
    #     "reasoning.\n</think>\nThis is my answer."
    # )
    
    SYSTEM_PROMPT = (
        # "You are an expert medical assistant. When a user asks a question related to the medical domain, " 
        
        
        # "You are a medical imaging expert. "
        # "Detect any visible diseases or abnormalities in this X-ray image. "
        # "For each detected finding, provide: "
        # "the disease name, and "
        # "bounding box coordinates as [x_min, y_min, x_max, y_max] in pixel values. "
        # "<think> You may refer to medical knowledge, hypotheses, chains of logic, etc.</think>\n"
        # "<answer>Your final answer to the user's question must need.</answer>\n"
        
        "You are a medical imaging expert."
        # "you must first **think through your reasoning** step by step, then provide the answer. \n"  
        # "You must follow this output format exactly: "
        # "<think> You may refer to medical knowledge, hypotheses, chains of logic, etc.</think>\n"
        # "<answer>Your final answer to the user's question must need.</answer>\n"
        # "Always *see the question* before you begin reasoning."
    )

    def make_conversation(example):
        problem = _strip_inline_image_tokens(example["problem"])
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        return {"prompt": prompt}
    dataset = dataset.map(make_conversation)
    

    # Filter have big imagesq
    # def filter_big_images(example):
    #     image = example["image"]
    #     return image.size[0] < 512 and image.size[1] < 512

    # dataset = dataset.filter(filter_big_images)

    # def convert_to_rgb(example):
    #     image = example["image"]
    #     if image.mode != "RGB":
    #         image = image.convert("RGB")
    #     example["image"] = image
    #     return example
    
    def convert_to_rgb(example):
        image_paths = example["image"]  # This is now a list of paths
        # GRPO will handle loading from paths, so just return as-is
        # Or if you need to load them:
        # images = [Image.open(path).convert("RGB") for path in image_paths]
        example["image"] = [image_paths]
        return example

    # dataset = dataset.map(convert_to_rgb)
    # if rank==0:
    #     breakpoint()
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None
    
    # if rank == 0:
    #     breakpoint()

    ################
    # Reward Function for Training
    ################
    
    # def tag_count_reward(completions, **kwargs) -> list[float]:
    #     """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    #     Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    #     """
    #     def count_tags(text: str) -> float:
    #         count = 0.0
    #         if text.count("<box>\n") == 1:
    #             count += 0.25
    #         if text.count("\n</box>\n") == 1:
    #             count += 0.25
    #         if text.count("\n<answer>\n") == 1:
    #             count += 0.25
    #         if text.count("\n</answer>") == 1:
    #             count += 0.25
    #         return count

    #     contents = [completion[0]["content"] for completion in completions]
    #     rewards = [count_tags(c) for c in contents]
    #     print('Tag Count Rewards: ', rewards)
    #     return rewards



    def accuracy_reward(completions, solution: list[str], **kwargs):
        """Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.
        """
        
        rewards = []
        contents = [completion[0]["content"] for completion in completions]
        for content, sol in zip(contents, solution):
            try:
                gold_parsed = parse(sol, extraction_mode="first_match")
            except Exception:
                gold_parsed = []

            if len(gold_parsed) != 0:
                # Try parsing predicted answer too
                try:
                    answer_parsed = parse(
                        content,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed="all",
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode="first_match",
                    )
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception as e:
                    print(f"verify failed: {e}, answer: {content}, gold: {sol}")
                    reward = None
            else:
                # fallback to text match
                reward = float(content.strip().lower() == sol.strip().lower())

            rewards.append(reward)

        return rewards

    ################
    # Training
    ################
    soft_overlong_punishment = get_soft_overlong_punishment(max_completion_len=300, soft_punish_cache=88)
    # tag_count_reward = get_reward_funcs("tag_count")

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    training_args.remove_unused_columns = False
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        # model=model,
        # processing_class= processor,
        args=training_args,
        # reward_funcs=[think_format_reward, accuracy_reward, bbox_iou_reward],
        # reward_funcs=[label_accuracy_reward, compute_bbox_rewards_dino, tag_count_reward, soft_overlong_punishment],
        reward_funcs=[label_accuracy_reward, tag_count_reward, soft_overlong_punishment],
        # reward_weights=[
        #     2.0,   # label_accuracy_reward   — highest weight, core task
        #     0.5,   # tag_count_reward        — format, secondary
        #     0.3,   # soft_overlong_punishment — penalty, minor
        # ],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        # llm_kwargs={"device": "cuda", "dtype": "bfloat16"}
        
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
