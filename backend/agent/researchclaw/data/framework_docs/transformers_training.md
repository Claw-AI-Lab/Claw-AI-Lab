# HuggingFace Transformers Training — API Quick Reference

## TrainingArguments (key parameters)
```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    # Training
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,      # effective batch = 8 * 4 = 32
    gradient_checkpointing=True,         # saves memory at ~20% speed cost
    # Optimizer
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,                    # or warmup_steps=100
    lr_scheduler_type="cosine",          # "linear", "cosine", "constant"
    optim="adamw_torch",                 # "adamw_torch", "adamw_8bit", "paged_adamw_8bit"
    # Precision
    bf16=True,                           # bfloat16 (Ampere+ GPUs)
    # fp16=True,                         # float16 (older GPUs)
    # Logging
    logging_steps=10,
    logging_strategy="steps",
    report_to="none",                    # "wandb", "tensorboard", "none"
    # Saving
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    # Evaluation
    eval_strategy="epoch",
    # Other
    dataloader_num_workers=4,
    remove_unused_columns=True,
    seed=42,
    max_grad_norm=1.0,
)
```

## Trainer
```python
from transformers import Trainer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,     # optional metric function
    data_collator=data_collator,         # optional custom collator
)

# Train
train_result = trainer.train()

# Evaluate
metrics = trainer.evaluate()

# Save
trainer.save_model("./best_model")
```

## Custom compute_metrics
```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }
```

## Tokenization / Data Preparation
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

tokenized_ds = dataset.map(tokenize_function, batched=True)
```

## Causal LM Training (GPT-style)
```python
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling

model = AutoModelForCausalLM.from_pretrained("gpt2")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal LM, not masked LM
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=data_collator,
)
```

## Key Tips
- `gradient_checkpointing=True` + `gradient_accumulation_steps` for memory efficiency
- `optim="paged_adamw_8bit"` reduces optimizer memory by ~50%
- `bf16=True` is preferred over `fp16=True` on Ampere+ GPUs (no loss scaling needed)
- Set `TOKENIZERS_PARALLELISM=false` to avoid fork warnings
- Use `model.config.use_cache = False` when using gradient_checkpointing
