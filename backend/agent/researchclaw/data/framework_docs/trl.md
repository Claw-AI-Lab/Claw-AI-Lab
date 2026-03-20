# TRL (Transformer Reinforcement Learning) — API Quick Reference

## Installation
```bash
pip install trl
```

## SFTTrainer — Supervised Fine-Tuning
```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
dataset = load_dataset("json", data_files="train.jsonl", split="train")

# SFTConfig inherits from TrainingArguments
training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_seq_length=1024,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    # Dataset formatting
    dataset_text_field="text",           # column name for text
    packing=True,                         # pack short samples for efficiency
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
trainer.save_model("./final_model")
```

### Dataset Format Options
```python
# Option 1: "text" column (conversational or plain text)
# {"text": "### Human: question\n### Assistant: answer"}

# Option 2: "messages" column (chat format, auto-applies chat template)
# {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

# Option 3: formatting_func callback
def formatting_func(example):
    return f"### Question: {example['question']}\n### Answer: {example['answer']}"

trainer = SFTTrainer(
    ...,
    formatting_func=formatting_func,
)
```

## DPOTrainer — Direct Preference Optimization
```python
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    beta=0.1,                  # KL penalty coefficient
    max_length=1024,
    max_prompt_length=512,
    bf16=True,
    gradient_checkpointing=True,
    loss_type="sigmoid",       # "sigmoid" (default), "hinge", "ipo"
)

# Dataset must have columns: "prompt", "chosen", "rejected"
# OR "chosen" and "rejected" as full conversations
trainer = DPOTrainer(
    model=model,
    ref_model=None,            # None = use implicit reference (PEFT)
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
```

## GRPOTrainer — Group Relative Policy Optimization
```python
from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    output_dir="./grpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=1e-6,
    num_generations=4,          # samples per prompt for group scoring
    max_completion_length=256,
    bf16=True,
)

# Requires a reward function
def reward_fn(completions, prompts):
    # Return list of float scores
    return [score_completion(c) for c in completions]

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    train_dataset=dataset,      # needs "prompt" column
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
```

## PPOTrainer — Proximal Policy Optimization for RLHF
```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

config = PPOConfig(
    model_name="Qwen/Qwen2.5-3B",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
    gradient_accumulation_steps=1,
)

model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen2.5-3B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

trainer = PPOTrainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
)

# Training loop
for batch in trainer.dataloader:
    query_tensors = batch["input_ids"]
    response_tensors = trainer.generate(query_tensors, max_new_tokens=128)
    rewards = [reward_model(q, r) for q, r in zip(query_tensors, response_tensors)]
    stats = trainer.step(query_tensors, response_tensors, rewards)
```

## Integration with PEFT/LoRA
```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# Pass peft_config to any TRL trainer
trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,   # automatically wraps model with LoRA
    ...,
)
```

## Key Tips
- Always set `tokenizer.pad_token = tokenizer.eos_token` if pad_token is None
- Use `gradient_checkpointing=True` for memory efficiency
- Use `bf16=True` on Ampere+ GPUs (A100, RTX 3090+, RTX 4090, RTX 6000 Ada)
- For multi-GPU: TRL uses accelerate under the hood, just launch with `accelerate launch`
- `packing=True` in SFTConfig significantly speeds up training on short samples
