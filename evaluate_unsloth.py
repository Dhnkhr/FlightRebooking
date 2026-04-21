import json
import torch
from unsloth import FastLanguageModel
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

    while not done:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current observation: {obs.model_dump_json()}"}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        outputs = model.generate(inputs, max_new_tokens=64, use_cache=True, do_sample=False)
        response_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        action_dict = extract_json(response_text)
        
        # Stuck-Agent Loop Breaker
        action_str = json.dumps(action_dict)
        if hasattr(env, '_last_action_str') and env._last_action_str == action_str:
            print(f"⚠️ Agent stuck! Forcing skip for {action_dict.get('passenger_id', 'Unknown')}.")
            action_dict = {"action_type": "mark_no_solution", "passenger_id": action_dict.get("passenger_id", "P1")}
        env._last_action_str = json.dumps(action_dict)
        
        try:
            action = Action(**action_dict)
        except Exception:
            action = Action(action_type=ActionType.FINALIZE)
            
        print(f"🤖 LLM chose: {action.model_dump_json()}")
        obs, reward, done, info = env.step(action)
        
    final_state = env.state()
    score = grade_task(task_key, final_state, task_data["max_budget"])
    print(f"Final Score for {task_key}: {score:.4f} / 1.0000")
    print(f"Budget spent: ${final_state.budget_spent} / ${task_data['max_budget']}")
    return float(score)

def main():
    print("Loading specialized flight-rebooking-lora model from local folder...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "/content/drive/MyDrive/flight-rebooking-lora", # Loads your saved weights!
            max_seq_length = 1024,
            dtype = None,
            load_in_4bit = True,
        )
    except Exception as e:
        print(f"ERROR: Could not load the model. Are you sure 'flight-rebooking-lora' folder is here? {e}")
        return

    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    total_score = 0
    scores = {}
    for t in ["easy", "medium", "hard"]:
        score = evaluate_task(model, tokenizer, t)
        scores[t] = score
        total_score += score
        
    print("\n===============================")
    print("🏁 FINAL HACKATHON EVALUATION")
    print("===============================")
    for k, v in scores.items():
        print(f"Task {k:6s} | Score: {v:.4f}")
    print(f"Overall Average: {total_score / 3:.4f}")

if __name__ == "__main__":
    main()
