# Grumpy Chef Fine-tuning with SFT + DPO

<div align="center">

<img alt="Image" src="https://github.com/user-attachments/assets/99e7363a-ec33-4407-8411-f506fe5daca8" />

</div>

&nbsp;

Fine-tuning of [LiquidAI/LFM2.5-1.2B-Base](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base) to behave as a grumpy Italian chef, using Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO). Training is accelerated with [Unsloth](https://github.com/unslothai/unsloth) and QLoRA (4-bit).

## Dataset

- 299 examples with `prompt`, `chosen` (grumpy chef tone), and `rejected` (neutral/generic tone) columns.
- Source file: `grumpy_chef_dataset.json`
- Published to HuggingFace Hub: [benitomartin/grumpy-chef-dpo](https://huggingface.co/datasets/benitomartin/grumpy-chef-dpo)
- Split: 254 train / 30 eval / 15 inference

```json
{
"prompt": "Can I rinse pasta after cooking?",
"chosen": "Rinse it? RINSE IT?! No. You wash away the starch, the flavor, the soul. Pasta is not laundry.",
"rejected": "Rinsing pasta is usually not recommended unless making cold pasta dishes."
}
```

## Pipeline

The notebook `finetuning_sft_dpo_unsloth.ipynb` runs three stages end-to-end:

### 1. Base Model Inference

Runs the unmodified LFM2.5-1.2B-Base on a set of cooking prompts to establish a baseline. The base model produces generic, encyclopedic responses with no personality.

### 2. SFT (Supervised Fine-Tuning)

Trains LoRA adapters on `prompt` + `chosen` pairs so the model learns the grumpy chef style.

| Parameter | Value |
|---|---|
| LoRA rank (r) | 32 |
| Target modules | GLU (w1, w2, w3), MHA (q/k/v/out_proj), Conv (in/out_proj) |
| Trainable params | 22.2M / 1.19B (1.86%) |
| Epochs | 5 |
| Batch size | 2 x 4 gradient accumulation |
| Learning rate | 1e-4 (cosine schedule) |
| Optimizer | AdamW 8-bit |

Adapter saved to `outputs/sft_lora`.

### 3. DPO (Direct Preference Optimization)

Loads the SFT adapter and continues training with the DPO objective, using the base model as the implicit reference (`ref_model=None`).

| Parameter | Value |
|---|---|
| Epochs | 2 |
| Batch size | 1 x 4 gradient accumulation |
| Learning rate | 5e-6 |
| Optimizer | AdamW 8-bit |

Adapter saved to `outputs/sft_dpo_lora`.

## Export

The final SFT+DPO model is exported in multiple formats and pushed to HuggingFace Hub:

- **GGUF** (Q4_K_M, Q8_0): [benitomartin/grumpy-chef-lfm2.5-1.2B-GGUF](https://huggingface.co/benitomartin/grumpy-chef-lfm2.5-1.2B-GGUF)
- **bf16 merged** (vLLM-ready): [benitomartin/grumpy-chef-lfm2.5-1.2B-bf16](https://huggingface.co/benitomartin/grumpy-chef-lfm2.5-1.2B-bf16)

## Serving with vLLM

You need to install vLLM first as per these [instructions](https://docs.vllm.ai/en/stable/getting_started/quickstart/) in a separate environment and then run these commands for fast inference:

```bash
vllm serve benitomartin/grumpy-chef-lfm2.5-1.2B-vllm --dtype bfloat16 --gpu-memory-utilization 0.7 --max-model-len 512
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "benitomartin/grumpy-chef-lfm2.5-1.2B-vllm",
    "messages": [{"role": "user", "content": "Can I put cream in carbonara?"}],
    "max_tokens": 100,
    "temperature": 0.3
  }'
```

## Serving with Ollama

You can also serve with Ollama or any other library using GGUF like llama.cpp:

```bash
ollama pull hf.co/benitomartin/grumpy-chef-lfm2.5-1.2B-GGUF:Q4_K_M
ollama run hf.co/benitomartin/grumpy-chef-lfm2.5-1.2B-GGUF:Q4_K_M
```
