"""
Local Training Script for RTX 4060 (8GB VRAM) — Windows Compatible
No Unsloth required! Uses plain HuggingFace transformers + peft + trl.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ── Configuration ──
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
DATASET_PATH = "artifacts/flight_rebooking_sft_final.jsonl"
OUTPUT_DIR = "./flight-rebooking-lora"
MAX_SEQ_LENGTH = 1024

print("=" * 60)
print("🚀 Flight Rebooking Agent — Local Training (RTX 4060)")
print("=" * 60)

# ── Step 1: Load dataset ──
print("\n📂 Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"   Loaded {len(dataset)} training samples")

# ── Step 2: Load tokenizer ──
print("\n📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Step 3: Load base model in 4-bit ──
print("\n📦 Loading base model in 4-bit (first run downloads ~4GB)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# ── Step 4: Prepare for LoRA training ──
print("\n🔧 Applying LoRA adapter (r=64)...")
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Step 5: Format dataset ──
def formatting_prompts_func(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}

print("\n🔄 Formatting dataset...")
formatted_dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

# ── Step 6: Training arguments (optimized for 8GB VRAM) ──
print("\n⚡ Starting training...")
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=1,       # Small batch for 8GB VRAM
    gradient_accumulation_steps=8,       # Effective batch size = 8
    warmup_steps=10,
    max_steps=1000,                      # 1000 steps for 8000 samples
    learning_rate=2e-4,
    fp16=True,                           # Use fp16 for RTX 4060
    logging_steps=10,
    optim="adamw_8bit",                  # 8-bit optimizer saves VRAM
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    save_steps=250,                      # Save checkpoints every 250 steps
    save_total_limit=2,                  # Keep only last 2 checkpoints
    gradient_checkpointing=True,         # Critical for 8GB VRAM!
    report_to="none",                    # Set to "wandb" if you want WandB logging
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    args=training_args,
    max_seq_length=MAX_SEQ_LENGTH,
)

# ── Train! ──
trainer_stats = trainer.train()

# ── Step 7: Save the trained model ──
print("\n💾 Saving LoRA adapters...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Training complete! Model saved to: {OUTPUT_DIR}")
print(f"   Total training time: {trainer_stats.metrics['train_runtime']:.0f} seconds")
print(f"   Final loss: {trainer_stats.metrics['train_loss']:.4f}")
print("\n🎯 Next step: Run 'python evaluate_local.py' to test your model!")
