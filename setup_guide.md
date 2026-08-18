# 🚀 Flight Rebooking Agent — Local Setup Guide (8GB VRAM GPU)

## Prerequisites
- **OS:** Windows 10/11
- **GPU:** NVIDIA GPU with 8GB+ VRAM (e.g., RTX 4060)
- **Software:** Python 3.10+, Git, NVIDIA CUDA Toolkit 12.x installed
- **Storage:** ~15GB free disk space

---

## Step 1: Clone the Project

```bash
git clone https://github.com/<your-username>/flight-rebooking-agent.git
cd flight-rebooking-agent
```

---

## Step 2: Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

## Step 3: Install PyTorch with CUDA

Go to [pytorch.org](https://pytorch.org/get-started/locally/) and grab the correct command for your CUDA version. For CUDA 12.x:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Step 4: Install Project Dependencies

```bash
pip install fastapi uvicorn pydantic
pip install transformers accelerate bitsandbytes peft
pip install unsloth
```

> [!NOTE]
> If `unsloth` fails to install on Windows, you can skip it. We will load the model using plain `transformers` + `peft` instead (see Step 6 alternative).

---

## Step 5: Download the Trained Model Weights

Download the `flight-rebooking-lora` folder from Google Drive to the project directory:

```
flight-rebooking-agent/
├── flight-rebooking-lora/    ← Put it here
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
├── environment.py
├── tasks.py
├── app.py
├── evaluate_unsloth.py
└── ...
```

---

## Step 6: Test the Model Locally

Create a quick test script called `test_local.py`:

```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model in 4-bit
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    quantization_config=bnb_config,
    device_map="auto",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, "./flight-rebooking-lora")
tokenizer = AutoTokenizer.from_pretrained("./flight-rebooking-lora")

print("Running test inference...")
messages = [
    {"role": "system", "content": "You are an airline disruption operations agent. Return exactly one JSON object."},
    {"role": "user", "content": "Current observation: {\"passengers\": [{\"id\": \"P1\", \"name\": \"Test User\", \"priority_tier\": \"Platinum\", \"status\": \"pending\"}], \"flights\": [{\"id\": \"FL-100\", \"economy_seats\": 5, \"business_seats\": 2}]}"}
]

inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=64, do_sample=False)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

print(f"🤖 Model output: {response}")
print("✅ Local setup is working!")
```

Run it:
```bash
python test_local.py
```

If it prints a valid JSON action, the model is loaded and working on the local GPU!

---

## Step 7: Run the Interactive UI

Start the FastAPI backend:
```bash
python app.py
```

Then open in your browser:
```
http://localhost:7860/ui
```

---

## Step 8: Run the Evaluation Script

To run the full hackathon evaluation locally (instead of Colab), edit `evaluate_unsloth.py` and change the model path:

```python
# Change this line:
model_name = "/content/drive/MyDrive/flight-rebooking-lora"

# To this:
model_name = "./flight-rebooking-lora"
```

> [!IMPORTANT]
> The local evaluation uses `unsloth` for loading. If `unsloth` doesn't work on Windows, 
> replace the loading code with the `transformers` + `peft` approach shown in Step 6.

Then run:
```bash
python evaluate_unsloth.py
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Close all other apps, ensure no other process is using the GPU |
| `unsloth` won't install on Windows | Use `transformers` + `peft` loading (Step 6) instead |
| `bitsandbytes` errors on Windows | Run `pip install bitsandbytes-windows` |
| Model outputs garbage | Ensure you downloaded the complete `flight-rebooking-lora` folder with all files |
| Slow inference (~60s per step) | Check that the model loaded on GPU (`device_map="auto"`) not CPU |
