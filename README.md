# EAGer: **E**ntropy-**A**ware **GE**ne**R**ation for Adaptive Inference-Time Scaling

<div align="center">

[Daniel Scalena](https://www.danielsc4.it/)<sup>1,2</sup> · [Leonidas Zotos](https://www.rug.nl/staff/l.zotos/?lang=en)<sup>1</sup> · [Elisabetta Fersini](https://en.unimib.it/elisabetta-fersini)<sup>2</sup> · [Malvina Nissim](https://malvinanissim.github.io/)<sup>1</sup> · [Ahmet Üstün](https://ahmetustun.github.io)<sup>3</sup>

<sup>1</sup>University of Groningen · <sup>2</sup>University of Milano-Bicocca · <sup>3</sup>Cohere

<br/>

<img src="media/fig1.png" alt="EAGer Framework" width="500"/>
<img src="media/plots.png" alt="Performance Results" width="500"/>

</div>

---

## 🌍 Overview

**EAGer** is a training-free method that dynamically adjusts compute at inference time based on token-level uncertainty. Instead of allocating the same compute budget for every prompt, EAGer branches into multiple reasoning paths only when the model encounters high-entropy tokens—indicating uncertainty.

**🔑 Key Benefits:**
- ✨ Reduces redundant computation by up to 65%
- 🎯 Improves reasoning performance by up to 37% in Pass@k
- ⚡ Enables adaptive scaling without additional training
- 🔄 Automatically reallocates saved compute to more complex instances

---

## ⚙️ Quick Start

### 1. Installation

Install dependencies using [UV](https://docs.astral.sh/uv/) (recommended):

```bash
uv pip install -e .
```

Or using pip:

```bash
pip install -e .
```

### 2. Running Experiments

Execute experiments using the `src.main_vllm` script:

```bash
python -m src.main_vllm \
    --model_name "openai/gpt-oss-20b" \
    --data_name "opencompass/AIME2025" \
    --temperature 0.6 \
    --entropy_threshold 2.2 \
    --max_sequences 32 \
    --experiments "eager" \
    --output_dir "$output_dir" \
    --seed 55
```

➡️ `example_execution.sh` provides a simple example to run an experiment following the same format as above.

---

## ⚙️ Configuration

### Core Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model_name` | HuggingFace model identifier | `openai/gpt-oss-20b` |
| `data_name` | HuggingFace dataset identifier | `opencompass/AIME2025` |
| `temperature` | Sampling temperature (must be > 0) | `0.6` |
| `entropy_threshold` | Branching threshold (~2.0 for EAGer-init, ~2.4 for EAGer) | `2.2` |
| `max_sequences` | Maximum parallel sequences (M in paper) | `32` |
| `experiments` | Experiment type to run | `eager` |
| `output_dir` | Output directory (serves as experiment ID) | See below |
| `seed` | Random seed for reproducibility | `55` |

### Supported Models

- `HuggingFaceTB/SmolLM3-3B`
- `Qwen/Qwen3-4B`
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- `openai/gpt-oss-20b`

💡: *Any model supported by vLLM should work.*

### Supported Datasets

- `Maxwell-Jia/AIME_2024`
- `opencompass/AIME2025`
- `MathArena/hmmt_feb_2025`
- `fingertap/GPQA-Diamond`
- `evalplus/humanevalplus`

### Experiment Types

| Type | Description | Prerequisites |
|------|-------------|---------------|
| `parallel` | Standard parallel sampling baseline | None |
| `eager_init` | Initial EAGer with fixed threshold | None |
| `eager_adapt` | Adaptive threshold adjustment | Requires `eager_init` |
| `eager` | Full EAGer with dynamic compute allocation | Requires `eager_init` |
| `all` | Run all experiments sequentially | None |

### Output Directory Structure

Experiments are saved to: `outputs/{model_name}/{data_name}/{timestamp}/`

**Example:** For `Qwen/Qwen3-4B` on `opencompass/AIME2025`, outputs are placed in:  
`outputs/Qwen3-4B/AIME2025/2025-10-01_12-00-01/`

**Auto-create directory:**

```bash
model_name="Qwen/Qwen3-4B"
data_name="opencompass/AIME2025"
timestamp=$(date +%F_%H-%M-%S)

model_dir_name=$(basename "$model_name")
data_dir_name=$(basename "$data_name")
output_dir="outputs/${model_dir_name}/${data_dir_name}/${timestamp}"
mkdir -p "$output_dir"
```

Then pass it to the script:

```bash
--output_dir "$output_dir"
```

### Advanced Parameters

```bash
--dtype "bfloat16"                    # Model precision
--max_model_len 32768                 # Maximum context length
--gpu_memory_utilization 0.8          # GPU memory usage (0.0-1.0)
--device "cuda"                       # Device to use
```

---

## 🧩 Evaluation

### Automatic Evaluation

Evaluation for `Maxwell-Jia/AIME_2024`, `opencompass/AIME2025`, and `fingertap/GPQA-Diamond` runs automatically during experiments. Answers are extracted from `\boxed{}` environments to compute Pass@k, Cons@k, and PassRate metrics.

### Code Generation Evaluation

For `evalplus/humanevalplus`, use the provided evaluation script:

⚠️ **Warning:** This script executes generated code locally without sandboxing, which poses security risks.

```bash
python script_eval_code_gen.py Qwen3-4B AIME2025 2025-10-01_12-00-01
```

For advanced evaluation, refer to the [evalplus documentation](https://github.com/evalplus/evalplus/blob/master/docs/cli.md).

### Summary of Results

To recap results from any experiment folder:

```bash
python script_manual_parallel_recapper.py Qwen3-4B AIME2025 2025-10-01_12-00-01
```

---

## 📊 Citation

```bibtex
@article{scalena2025eager,
  title={EAGer: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling},
  author={Scalena, Daniel and Zotos, Leonidas and Fersini, Elisabetta and Nissim, Malvina and Üstün, Ahmet},
  year={2025}
}
```

---
---

## 🧾 Abstract

With the rise of reasoning language models and test-time scaling methods as a paradigm for improving model performance, substantial computation is often required to generate multiple candidate sequences from the same prompt. This enables exploration of different reasoning paths toward the correct solution, however, allocates the same compute budget for each prompt. 

Grounded on the assumption that different prompts carry different degrees of complexity, and thus different computation needs, we propose **EAGer**, a training-free generation method that leverages model uncertainty through token-wise entropy distribution to reduce redundant computation and concurrently improve overall performance.

EAGer allows branching to multiple reasoning paths only in the presence of high-entropy tokens, and then reallocates the saved compute budget to the instances where exploration of alternative paths is most needed.

We find that across multiple open-source models on complex reasoning benchmarks such as AIME 2025, EAGer can reallocate the budget without accessing target labels, achieving the best efficiency–performance trade-off in terms of both token usage and Pass@k. When target labels are accessible, EAGer generates up to 65% fewer tokens (hence saving compute) and achieves up to 37% improvement in Pass@k compared to the Full Parallel sampling.

Our results show that EAGer consistently maximizes the efficiency-performance trade-off by enabling dynamic control over computation expenditure.