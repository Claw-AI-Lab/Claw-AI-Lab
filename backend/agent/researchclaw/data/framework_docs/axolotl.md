# Axolotl — API Quick Reference

## Installation
```bash
pip install axolotl
# or
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl && pip install -e .
```

## CLI Usage
```bash
# Train
accelerate launch -m axolotl.cli.train config.yaml

# Inference
accelerate launch -m axolotl.cli.inference config.yaml --lora_model_dir=./output

# Merge LoRA adapter
python -m axolotl.cli.merge_lora config.yaml --lora_model_dir=./output
```

## Training Config (YAML)
```yaml
base_model: Qwen/Qwen2.5-3B
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: true

# Load in 4-bit (QLoRA)
load_in_4bit: true
adapter: qlora                 # qlora, lora, or omit for full fine-tune
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true       # target all linear layers

# Dataset
datasets:
  - path: my_data.jsonl
    type: alpaca               # alpaca, sharegpt, completion, etc.
    # OR custom format:
    # type:
    #   field_instruction: instruction
    #   field_input: input
    #   field_output: output
    #   format: "{instruction}\n{input}"

# Training
sequence_len: 2048
sample_packing: true           # pack short sequences together
pad_to_sequence_len: true

num_epochs: 3
micro_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-5
lr_scheduler: cosine
warmup_ratio: 0.1
optimizer: paged_adamw_8bit

# Precision
bf16: auto                     # auto-detect GPU capability
tf32: true

# Memory optimization
gradient_checkpointing: true
flash_attention: true          # use flash attention 2

# Logging & saving
logging_steps: 10
save_strategy: steps
save_steps: 500
save_total_limit: 3
output_dir: ./output

# Evaluation
val_set_size: 0.05
eval_steps: 500

# Weights & Biases (optional)
wandb_project: my-project
wandb_run_id:
```

## Dataset Formats
```yaml
# Alpaca format
datasets:
  - path: data.jsonl
    type: alpaca
# {"instruction": "...", "input": "...", "output": "..."}

# ShareGPT format
datasets:
  - path: data.jsonl
    type: sharegpt
# {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

# Completion (raw text)
datasets:
  - path: data.jsonl
    type: completion
# {"text": "full text here"}

# HuggingFace dataset
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
```

## DPO Config
```yaml
rl: dpo                        # enables DPO training
# Dataset must have "chosen" and "rejected" fields
datasets:
  - path: dpo_data.jsonl
    type: chat_template.default
    field_messages: chosen      # for chosen responses
    # split into chosen/rejected pairs
```

## Key Tips
- `sample_packing: true` greatly improves throughput on short sequences
- `flash_attention: true` reduces memory and speeds up attention (requires compatible GPU)
- `lora_target_linear: true` is the easiest way to target all linear layers
- `bf16: auto` auto-detects GPU capability
- DeepSpeed integration: add `deepspeed: deepspeed_configs/zero2.json`
- Multi-GPU: use `accelerate launch` with proper config
