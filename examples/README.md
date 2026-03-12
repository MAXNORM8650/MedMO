# 🏥 MedMO: Grounding and Understanding Multimodal Large Language Model for Medical Images

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2602.06965)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/collections/MBZUAI/medmo)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/genmilab/MedMO)
[![Project Page](https://img.shields.io/badge/🌐-Project_Page-green)](https://genmilab.github.io/MedMO-Page)

</div>

> **MedMO** is a powerful open-source post-trained multimodal large medical vision–language foundation model (VLM) purpose-built for comprehensive medical image understanding and grounding. It achieves state-of-the-art performance across VQA, text QA, report generation, and spatial grounding benchmarks.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Model Family](#model-family)
- [Installation](#installation)
- [Training](#training)
  - [Stage 1–3: Supervised Fine-Tuning (SFT)](#stage-13-supervised-fine-tuning-sft)
  - [Stage 4: GRPO Reinforcement Learning](#stage-4-grpo-reinforcement-learning)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## 🔍 Overview

MedMO follows a **four-stage post-training pipeline**:

| Stage | Objective | Dataset Size | Resolution |
|-------|-----------|-------------|------------|
| 1 — Large-Scale SFT | Global image–text alignment | 18.5M | 768×768 |
| 2 — High-Resolution SFT | Fine-grained grounding | 3M | 1280×1280 |
| 3 — Instruction Tuning | Clinical reasoning & QA | 4.3M | — |
| 4 — GRPO (RL) | Bounding-box spatial grounding | 300K | dynamic |

---

## 📦 Model Family

| Model | Parameters | Best For |
|-------|-----------|----------|
| [MedMO-8B-Next](https://huggingface.co/MBZUAI/MedMO-8B-Next) | 8B | SOTA highest accuracy, all tasks — **recommended** |
| [MedMO-4B-Next](https://huggingface.co/MBZUAI/MedMO-4B-Next) | 4B | 2nd SOTA, high accuracy in resource-constrained environments |
| [MedMO-8B](https://huggingface.co/MBZUAI/MedMO-8B) | 8B | Previous generation |
| [MedMO-4B](https://huggingface.co/MBZUAI/MedMO-4B) | 4B | Resource-constrained environments |

For detailed benchmark results, please refer to our [paper](https://arxiv.org/abs/2602.06965).

---

## ⚙️ Installation

```bash
git clone https://github.com/genmilab/MedMO.git
cd MedMO

# Create conda environment
conda create -n medmo python=3.10 -y
conda activate medmo

# Install dependencies
pip install -r requirements.txt

# Install TRL (required for SFT and GRPO training)
pip install trl accelerate deepspeed
```

> **Hardware:** MedMO-8B was trained on **64× AMD Instinct MI210 GPUs** (64 GB each) across 8 nodes. Adjust `--num_machines` and `--num_processes` to match your cluster setup.

---

## 🚀 Training

### Stage 1–3: Supervised Fine-Tuning (SFT)

The SFT script covers Stages 1, 2, and 3. Run on **each node**, changing only `--machine_rank`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
WANDB_MODE=offline \
WANDB_API_KEY="YOUR_WANDB_API_KEY" \          # ← CHANGE
WANDB_PROJECT="medmo-sft" \                    # ← CHANGE
NCCL_SOCKET_IFNAME=bond0 \                     # ← CHANGE (e.g., eth0, ib0)
GLOO_SOCKET_IFNAME=bond0 \                     # ← CHANGE (same as above)
accelerate launch \
  --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
  --num_machines 8 \                           # ← CHANGE: number of nodes
  --num_processes 64 \                         # ← CHANGE: num_machines × GPUs_per_node
  --main_process_ip 172.27.112.77 \            # ← CHANGE: master node IP
  --main_process_port 29500 \                  # ← CHANGE: any free port
  --machine_rank 0 \                           # ← CHANGE: 0 for master, 1,2,... for workers
  --same_network \
  examples/scripts/sft_vlm_medmo.py \
  --dataset_name medtrinity_report \           # ← CHANGE: your dataset name or path
  --model_name_or_path ./Medmo_8B \            # ← CHANGE: path to base model
  --output_dir Medmo_8B_Next \                 # ← CHANGE: checkpoint output directory
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --attn_implementation sdpa \
  --dtype bfloat16 \
  --lr_scheduler_type cosine \
  --logging_steps 20 \
  --bf16 True \
  --bf16_full_eval True \
  --num_train_epochs 5 \
  --save_strategy epoch \
  --save_only_model True \
  --report_to wandb \
  --ddp_find_unused_parameters False \
  --gradient_checkpointing True \
  --data_seed 42 \
  --dataloader_num_workers 2 \
  --dataloader_prefetch_factor 2 \
  --trust_remote_code True \
  --dataloader_drop_last True \
  --dataloader_persistent_workers True \
  --max_grad_norm 1.0
```

#### 🔧 SFT — What to Change

| Argument | What to Set |
|----------|-------------|
| `WANDB_API_KEY` | Your key from [wandb.ai/authorize](https://wandb.ai/authorize) |
| `NCCL/GLOO_SOCKET_IFNAME` | Your network interface (`eth0`, `ib0`, etc.) |
| `--num_machines` / `--num_processes` | Your cluster size |
| `--main_process_ip` / `--machine_rank` | Master IP and per-node rank |
| `--dataset_name` / `--model_name_or_path` | Your dataset and base model |
| `--output_dir` | Where to save checkpoints |
| `--learning_rate` | `1e-5` (S1), `8e-6` (S2), `5e-6` (S3) |

> **Single-node:** Set `--num_machines 1`, `--num_processes 8`, and remove `--main_process_ip`, `--machine_rank`, `--same_network`.

---

### Stage 4: GRPO Reinforcement Learning

Stage 4 uses **GRPO with verifiable bounding-box rewards** to refine spatial grounding. Run after Stage 3 SFT is complete. Launch on **each node**, changing only `--node_rank`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
WANDB_MODE=offline \
WANDB_API_KEY="YOUR_WANDB_API_KEY" \          # ← CHANGE
WANDB_PROJECT="medmo-grpo" \                   # ← CHANGE
NCCL_SOCKET_IFNAME=bond0 \                     # ← CHANGE (e.g., eth0, ib0)
GLOO_SOCKET_IFNAME=bond0 \                     # ← CHANGE (same as above)
DS_BUILD_AIO=0 \
DEEPSPEED_BUILD_AIO=0 \
torchrun \
  --nnodes=8 \                                 # ← CHANGE: number of nodes
  --nproc_per_node=8 \                         # ← CHANGE: GPUs per node
  --master_addr=172.27.112.28 \               # ← CHANGE: master node IP
  --master_port=29500 \                        # ← CHANGE: any free port
  --node_rank=0 \                              # ← CHANGE: 0 for master, 1,2,... for workers
  examples/scripts/grpo_vlm.py \
  --model_name_or_path /path/to/Medmo_SFT \   # ← CHANGE: path to Stage 3 checkpoint
  --output_dir Medmo_8B_Next \                 # ← CHANGE: checkpoint output directory
  --run_name medmo-grpo \                      # ← CHANGE: W&B run name
  --learning_rate 1e-5 \
  --gradient_checkpointing \
  --dtype bfloat16 \
  --max_prompt_length 4096 \
  --max_completion_length 1024 \
  --attn_implementation sdpa \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --num_generation 4 \
  --deepspeed examples/accelerate_configs/ds_zero3_cpu.json \
  --report_to wandb \
  --use_vllm True \
  --vllm_mode colocate \
  --trust_remote_code True \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.30 \         # ← CHANGE: lower if OOM (e.g., 0.20)
  --epsilon 0.15 \
  --epsilon_high 0.25 \
  --mask_truncated_completions True \
  --vllm_model_impl "transformers" \
  --gradient_accumulation_steps 4 \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 3
```

#### 🔧 GRPO — What to Change

| Argument | What to Set |
|----------|-------------|
| `WANDB_API_KEY` | Your key from [wandb.ai/authorize](https://wandb.ai/authorize) |
| `NCCL/GLOO_SOCKET_IFNAME` | Your network interface (`eth0`, `ib0`, etc.) |
| `--nnodes` / `--nproc_per_node` | Your cluster size |
| `--master_addr` / `--node_rank` | Master IP and per-node rank |
| `--model_name_or_path` | Output directory from Stage 3 SFT |
| `--output_dir` / `--run_name` | Checkpoint save path and W&B run name |
| `--vllm_gpu_memory_utilization` | Reduce to `0.20` if out of memory |

---

## 💡 Inference

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Load model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "MBZUAI/MedMO-8B",                        # ← CHANGE: MedMO-8B, MedMO-8B-Next, MedMO-4B, MedMO-4B-Next
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("MBZUAI/MedMO-8B")  # ← CHANGE: match model above

# Prepare your input
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "path/to/medical/image.png",  # ← CHANGE: your image path
            },
            {"type": "text", "text": "What abnormalities are present in this chest X-ray?"},
        ],
    }
]

# Process and generate
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Generate output
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
```


## 📝 Citation

If you use MedMO in your research, please cite our paper:

```bibtex
@article{deria2026medmo,
  title={MedMO: Grounding and Understanding Multimodal Large Language Model for Medical Images},
  author={Deria, Ankan and Kumar, Komal and Dukre, Adinath Madhavrao and Segal, Eran and Khan, Salman and Razzak, Imran},
  journal={arXiv preprint arXiv:2602.06965},
  year={2026}
}
```

---

## 🙏 Acknowledgments

We gratefully acknowledge the following:
- **Base Architecture**: Built on [Qwen3-VL](https://github.com/QwenLM/Qwen-VL) by Alibaba Cloud
- **📊 Evaluation Framework**: [MedEvalKit](https://github.com/alibaba-damo-academy/MedEvalKit) by Alibaba DAMO Academy
- **Training Framework**: [TRL](https://github.com/huggingface/trl) (Transformer Reinforcement Learning) by Hugging Face
- **LLM-as-a-Judge**: Evaluation powered by gpt-5-mini-2025-08-07 from OpenAI
- **Compute Resources**: Training conducted on 64× AMD Instinct MI210 GPUs
- **Open-Source Datasets**: We thank the medical imaging community for providing high-quality public datasets including MedTrinity, MIMIC-CXR, CheXpert, PathVQA, and many others that made this work possible

---

## 📄 License

This project is released under the [Apache 2.0 License](LICENSE).

