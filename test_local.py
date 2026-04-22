import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("=" * 50)
print("🧪 Local Model Test — RTX 4060")
print("=" * 50)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("\n📦 Loading base model (this may take 2-3 minutes on first run)...")
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    quantization_config=bnb_config,
    device_map="auto",
)

print("🔗 Loading LoRA adapter from ./flight-rebooking-lora ...")
model = PeftModel.from_pretrained(base_model, "./flight-rebooking-lora")
tokenizer = AutoTokenizer.from_pretrained("./flight-rebooking-lora")

print("🚀 Running test inference...\n")

messages = [
    {"role": "system", "content": """You are an airline disruption operations agent.

Return exactly one JSON object on each turn with this schema:
{
  "action_type": "rebook_passenger" | "offer_downgrade" | "book_hotel" | "rebook_on_partner" | "mark_no_solution" | "finalize",
  "passenger_id": "optional passenger id",
  "flight_id": "optional flight id"
}

Policy:
- Process one pending passenger per step.
- Respect tiers (Platinum > Gold > Silver > Standard).
- Prefer earlier departures for deadline passengers.
- Prefer same-airline rebooking over partner when feasible.
- Minimize budget usage.
- Output raw JSON only."""},
    {"role": "user", "content": 'Current observation: {"passengers": [{"id": "P1", "name": "Alice Johnson", "priority_tier": "Platinum", "status": "pending", "cabin_class": "Business", "original_flight": "FL-100", "assigned_flight": null, "connection_deadline_hrs": null}], "flights": [{"id": "FL-102", "destination": "New York", "departure_hrs": 3.0, "economy_seats": 5, "business_seats": 2, "is_partner": false}], "budget_remaining": 3000, "steps_remaining": 40}'}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(inputs, max_new_tokens=64, do_sample=False)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

print(f"🤖 Model output: {response}")

# Try to parse the JSON to verify correctness
try:
    start = response.find('{')
    end = response.rfind('}') + 1
    parsed = json.loads(response[start:end])
    print(f"\n✅ Valid JSON! Action: {parsed.get('action_type')} | Passenger: {parsed.get('passenger_id')} | Flight: {parsed.get('flight_id')}")
except Exception:
    print(f"\n⚠️ Could not parse JSON from response. Raw output above.")

print("\n🎉 Local setup is working! Your RTX 4060 is running the AI successfully.")
