# LLaMA-Factory — API Quick Reference

## Installation
```bash
pip install llamafactory
# or
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e .
```

## CLI Usage (Primary Interface)
```bash
# Fine-tune with a YAML config
llamafactory-cli train config.yaml

# Chat with a fine-tuned model
llamafactory-cli chat config.yaml

# Export/merge LoRA adapter
llamafactory-cli export config.yaml

# Launch web UI
llamafactory-cli webui
```

## Training Config (YAML)
```yaml
### Model
model_name_or_path: Qwen/Qwen2.5-3B
trust_remote_code: true

### Method (LoRA)
stage: sft                    # sft, pt (pretrain), rm, ppo, dpo, kto, orpo
do_train: true
finetuning_type: lora         # lora, freeze, full
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all              # "all" for all linear, or comma-separated names

### Dataset
dataset: alpaca_en            # registered dataset name or custom path
template: qwen                # chat template: qwen, llama3, mistral, chatglm, etc.
cutoff_len: 1024
preprocessing_num_workers: 8

### Training
output_dir: ./output
num_train_epochs: 3.0
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-5
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true

### Evaluation
val_size: 0.05
per_device_eval_batch_size: 8
eval_strategy: steps
eval_steps: 500

### Logging
logging_steps: 10
save_steps: 500
save_total_limit: 3
report_to: none

### Quantization (QLoRA)
quantization_bit: 4           # 4 or 8
quantization_method: bitsandbytes  # bitsandbytes, gptq, awq
```

## Custom Dataset Registration
```json
// data/dataset_info.json — register your dataset
{
  "my_dataset": {
    "file_name": "my_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

### Dataset Formats
```json
// Alpaca format
{"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}

// ShareGPT format
{"conversations": [
  {"from": "human", "value": "Hello"},
  {"from": "gpt", "value": "Hi there!"}
]}
```

## DPO Training Config
```yaml
stage: dpo
do_train: true
model_name_or_path: Qwen/Qwen2.5-3B
finetuning_type: lora
dataset: dpo_en               # needs chosen/rejected columns
template: qwen
dpo_beta: 0.1
dpo_loss: sigmoid             # sigmoid, hinge, ipo
```

## Export/Merge Config
```yaml
model_name_or_path: Qwen/Qwen2.5-3B
adapter_name_or_path: ./output/checkpoint-1000
template: qwen
finetuning_type: lora
export_dir: ./merged_model
export_size: 2                # shard size in GB
export_legacy_format: false
```

## Key Tips
- `template` must match the model's chat format (qwen, llama3, mistral, etc.)
- `lora_target: all` targets all linear layers (recommended for quality)
- Use `quantization_bit: 4` for QLoRA to fit large models on limited VRAM
- `cutoff_len` controls max sequence length — reduce for memory savings
- Always set `gradient_checkpointing: true` for models > 1B parameters
- Check supported models: Qwen, LLaMA, Mistral, Phi, ChatGLM, Baichuan, Yi, etc.
