# PEFT (Parameter-Efficient Fine-Tuning) — API Quick Reference

## Installation
```bash
pip install peft
```

## LoRA (Low-Rank Adaptation)
```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", torch_dtype=torch.bfloat16)

lora_config = LoraConfig(
    r=16,                      # rank (4, 8, 16, 32, 64)
    lora_alpha=32,             # scaling factor (typically 2*r)
    lora_dropout=0.05,         # dropout probability
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none",               # "none", "all", or "lora_only"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 13,631,488 || all params: 3,098,746,880 || trainable%: 0.44%
```

## QLoRA (Quantized LoRA)
```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # "nf4" or "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # nested quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    quantization_config=bnb_config,
    device_map="auto",
)

# Then apply LoRA on top
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
```

## Saving and Loading
```python
# Save adapter only (small file)
model.save_pretrained("./lora_adapter")

# Load adapter
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# Merge adapter into base model (for deployment)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

## Common target_modules by Model Family
| Model | target_modules |
|-------|---------------|
| LLaMA/Qwen/Mistral | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
| GPT-2/GPT-J | c_attn, c_proj, c_fc |
| BLOOM | query_key_value, dense, dense_h_to_4h, dense_4h_to_h |
| T5/Flan-T5 | q, v, k, o, wi, wo |
| Phi | q_proj, v_proj, dense, fc1, fc2 |

## DoRA (Weight-Decomposed Low-Rank Adaptation)
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    use_dora=True,   # enables DoRA decomposition
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)
```

## Key Tips
- Higher `r` = more parameters = more capacity but slower training
- `lora_alpha / r` is the effective scaling. Common: alpha=2*r
- For QLoRA: always use `prepare_model_for_kbit_training()` before `get_peft_model()`
- Target more modules (all linear layers) for better quality at marginal compute cost
- Use `modules_to_save=["lm_head", "embed_tokens"]` if you want to train the head too
