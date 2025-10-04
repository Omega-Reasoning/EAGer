#!/bin/bash

source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

# model_name="HuggingFaceTB/SmolLM3-3B"
model_name="Qwen/Qwen3-4B"
# model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# model_name="openai/gpt-oss-20b"

# data_name="Maxwell-Jia/AIME_2024"
data_name="opencompass/AIME2025"
# data_name="MathArena/hmmt_feb_2025"
# data_name="fingertap/GPQA-Diamond"
# data_name="evalplus/humanevalplus"

# timestamp=$(date +%F_%H-%M-%S)    # automatic
timestamp="2025-10-10_22-13-30" # manual

model_dir_name=$(basename "$model_name")
data_dir_name=$(basename "$data_name")
output_dir="outputs/${model_dir_name}/${data_dir_name}/${timestamp}"
mkdir -p "$output_dir"

#### Full Parallel Sampling (our baseline)
experiment_type="parallel"

python -m src.main_vllm \
    --model_name "$model_name" \
    --data_name "$data_name" \
    --temperature 0.6 \
    --entropy_threshold 2.5 \
    --max_sequences 32 \
    --experiments "$experiment_type" \
    --gpu_memory_utilization 0.85 \
    --output_dir "$output_dir" \
    --seed 55


#### EAGer-init (preparatory for EAGer)
experiment_type="eager-init"

python -m src.main_vllm \
    --model_name "$model_name" \
    --data_name "$data_name" \
    --temperature 0.6 \
    --entropy_threshold 2.5 \
    --max_sequences 32 \
    --experiments "$experiment_type" \
    --gpu_memory_utilization 0.85 \
    --output_dir "$output_dir" \
    --seed 55


#### EAGer (with label access, use EAGer-adapt if no label access)
# experiment_type="eager_adapt"
experiment_type="eager"

python -m src.main_vllm \
    --model_name "$model_name" \
    --data_name "$data_name" \
    --temperature 0.6 \
    --entropy_threshold 2.5 \
    --max_sequences 32 \
    --experiments "$experiment_type" \
    --gpu_memory_utilization 0.85 \
    --output_dir "$output_dir" \
    --seed 55