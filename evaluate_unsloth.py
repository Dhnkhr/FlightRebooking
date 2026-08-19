import json
# pyrefly: ignore [missing-import]
import torch
import os
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# pyrefly: ignore [missing-import]
from peft import PeftModel
from environment import FlightRebookingEnv, Action, ActionType
from tasks import TASKS, grade_task

def extract_json(text: str) -> dict:
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            return json.loads(text[start_idx:end_idx])
    except Exception:
        pass
    return {"action_type": "finalize"}

def evaluate_task(model, tokenizer, task_key: str):
    print(f"\n--- Evaluating Task: {task_key.upper()} ---")
    task_data = TASKS[task_key]
    env = FlightRebookingEnv(task_data=task_data)
    obs = env.reset()
    done = False
    
    system_prompt = """You are an airline disruption operations agent.

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
- Output raw JSON only."""

    # Maintain conversation history across turns (matches multi-turn training format)
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    max_turns = 30  # Safety limit
    turn = 0
    recent_actions = []  # Track last few actions for cycle detection
    
    while not done and turn < max_turns:
        turn += 1
        
        # Append current observation as user message
        messages.append({"role": "user", "content": f"Current observation: {obs.model_dump_json()}"})
        
        # Aggressive truncation for 4GB VRAM: keep system + last 3 turn pairs
        if len(messages) > 7:  # system + 3 user/assistant pairs
            messages = [messages[0]] + messages[-6:]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")
        
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True, do_sample=False)
        response_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        action_dict = extract_json(response_text)
        
        # Append assistant response to history
        messages.append({"role": "assistant", "content": json.dumps(action_dict)})
        
        # Cycle detection: check if action repeats within last 3 turns
        action_str = json.dumps(action_dict)
        recent_actions.append(action_str)
        if len(recent_actions) > 6:
            recent_actions = recent_actions[-6:]
        
        is_stuck = False
        if recent_actions.count(action_str) >= 2:
            is_stuck = True
        
        if is_stuck:
            print(f"[WARN] Agent stuck in cycle! Forcing skip for {action_dict.get('passenger_id', 'Unknown')}.")
            action_dict = {"action_type": "mark_no_solution", "passenger_id": action_dict.get("passenger_id", "P1")}
            messages[-1] = {"role": "assistant", "content": json.dumps(action_dict)}
            recent_actions[-1] = json.dumps(action_dict)
        
        try:
            action = Action(**action_dict)
        except Exception:
            action = Action(action_type=ActionType.FINALIZE)
            
        print(f"[LLM] chose: {action.model_dump_json()}")
        obs, reward, done, info = env.step(action)
        
    final_state = env.state()
    score = grade_task(task_key, final_state, task_data["max_budget"])
    print(f"Final Score for {task_key}: {score:.4f} / 1.0000")
    print(f"Budget spent: ${final_state.budget_spent} / ${task_data['max_budget']}")
    return float(score)

def main():
    base_model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    adapter_path = "./flight-rebooking-lora-1b" # 1B model trained with hybrid data
    
    print(f"Loading base model: {base_model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0}
        )
        
        if os.path.exists(adapter_path):
            print(f"Loading LoRA adapters from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            print(f"WARNING: Adapter path {adapter_path} not found. Running base model only.")
            
        model.eval()
    except Exception as e:
        print(f"ERROR: Could not load the model: {e}")
        return
    
    total_score = 0
    scores = {}
    for t in ["easy", "medium", "hard"]:
        score = evaluate_task(model, tokenizer, t)
        scores[t] = score
        total_score += score
        
    print("\n===============================")
    print("FINAL HACKATHON EVALUATION")
    print("===============================")
    for k, v in scores.items():
        print(f"Task {k:6s} | Score: {v:.4f}")
    print(f"Overall Average: {total_score / 3:.4f}")

if __name__ == "__main__":
    main()
