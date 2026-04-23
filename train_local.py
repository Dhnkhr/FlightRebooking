"""
Local Training Script for RTX 4060 (8GB VRAM) — Windows Compatible
Pure PyTorch + PEFT — No TRL dependency needed!
"""
import json
import os
import sys
import torch
print("=" * 60)
print("Flight Rebooking Agent — Local Training (RTX 4060)")
print("=" * 60)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
sys.stdout.flush()

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Disable the caching_allocator_warmup that tries to pre-allocate 5GB (OOM on 8GB GPUs)
import transformers.modeling_utils
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
print("Patched: Disabled caching_allocator_warmup for 8GB GPU compatibility")

# ── Configuration ──
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
DATASET_PATH = "artifacts/flight_rebooking_sft_final.jsonl"
OUTPUT_DIR = "./flight-rebooking-lora"
MAX_SEQ_LENGTH = 768
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
MAX_STEPS = 1000
SAVE_EVERY = 250
LOG_EVERY = 10

# ── Custom Dataset ──
class ChatDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.samples = []
        print(f"\nLoading dataset from {path}...")
        sys.stdout.flush()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} training samples")
        sys.stdout.flush()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]["messages"]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}


def main():
    # ── Step 1: Load tokenizer ──
    print("\nLoading tokenizer...")
    sys.stdout.flush()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Step 2: Load base model in 4-bit ──
    print("\nLoading base model in 4-bit...")
    sys.stdout.flush()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        print("Model loaded successfully!")
        sys.stdout.flush()
    except Exception as e:
        print(f"\nERROR loading model: {type(e).__name__}: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Step 3: Apply LoRA ──
    print("\nApplying LoRA adapter (r=32)...")
    sys.stdout.flush()
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    sys.stdout.flush()

    # ── Step 4: Create dataset & dataloader ──
    dataset = ChatDataset(DATASET_PATH, tokenizer, MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ── Step 5: Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # ── Step 6: Training Loop ──
    print(f"\nStarting training for {MAX_STEPS} steps...")
    print(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} accumulation = {BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
    print("-" * 60)
    sys.stdout.flush()

    model.train()
    global_step = 0
    running_loss = 0.0
    data_iter = iter(dataloader)

    while global_step < MAX_STEPS:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRAD_ACCUM_STEPS

            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1
        running_loss += accum_loss

        if global_step % LOG_EVERY == 0:
            avg_loss = running_loss / LOG_EVERY
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Step {global_step:5d}/{MAX_STEPS} | Loss: {avg_loss:.4f} | GPU Mem: {gpu_mem:.1f}GB")
            sys.stdout.flush()
            running_loss = 0.0

        if global_step % SAVE_EVERY == 0:
            checkpoint_dir = f"{OUTPUT_DIR}-checkpoint-{global_step}"
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"  Saved checkpoint to {checkpoint_dir}")
            sys.stdout.flush()

    # ── Step 7: Save final model ──
    print(f"\nSaving final model to {OUTPUT_DIR}...")
    sys.stdout.flush()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nNext: Run 'python evaluate_unsloth.py' to test your model!")


if __name__ == "__main__":
    main()
