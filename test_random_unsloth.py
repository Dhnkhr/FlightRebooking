import copy
import json
import random
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

# ── Inline jitter function (copied from train_ml_policy.py) ──
def _clamp(value, lo, hi):
    return max(lo, min(hi, value))

def _jitter_task(task_data, rng):
    variant = copy.deepcopy(task_data)
    budget_factor = rng.uniform(0.85, 1.20)
    variant["max_budget"] = max(1000, int(round(float(variant["max_budget"]) * budget_factor / 50.0) * 50))
    max_steps = int(variant.get("max_steps", 80))
    variant["max_steps"] = max(20, min(140, max_steps + rng.randint(-8, 10)))
    for passenger in variant["passengers"]:
        deadline = passenger.get("connection_deadline_hrs")
        if deadline is None:
            if rng.random() < 0.10:
                passenger["connection_deadline_hrs"] = round(rng.uniform(2.0, 8.0), 1)
        else:
            passenger["connection_deadline_hrs"] = round(_clamp(float(deadline) + rng.uniform(-0.8, 0.8), 1.0, 12.0), 1)
    for flight in variant["flights"]:
        flight["departure_hrs"] = round(_clamp(float(flight["departure_hrs"]) + rng.uniform(-1.0, 1.0), 0.5, 12.0), 1)
        flight["economy_seats"] = max(0, int(flight["economy_seats"]) + rng.randint(-2, 3))
        flight["business_seats"] = max(0, int(flight["business_seats"]) + rng.randint(-1, 2))
    rng.shuffle(variant["passengers"])
    rng.shuffle(variant["flights"])
    return variant

# Map jittered difficulty back to a grading key
DIFFICULTY_MAP = {"easy": "easy", "medium": "medium", "hard": "hard"}

def evaluate_random_task(model, tokenizer, index: int):
    print(f"\n--- Evaluating Random Scenario #{index} ---")
    
    # Pick a random base task and jitter it to create a unique scenario
    base_keys = list(TASKS.keys())
    chosen_key = random.choice(base_keys)
    base_task = TASKS[chosen_key]
    randomized_task = _jitter_task(base_task, random.Random())
    
    print(f"    Base difficulty: {chosen_key} | Budget: ${randomized_task['max_budget']} | Passengers: {len(randomized_task['passengers'])}")
    
    env = FlightRebookingEnv(task_data=randomized_task)
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
    score = grade_task(chosen_key, final_state, randomized_task["max_budget"])
    
    print(f"Final Score for Random #{index} ({chosen_key}-based): {score:.4f} / 1.0000")
    print(f"Budget spent: ${final_state.budget_spent} / ${randomized_task['max_budget']}")
    return float(score)

def main():
    print("Loading specialized flight-rebooking-lora model from local folder...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "/content/drive/MyDrive/flight-rebooking-lora",
            max_seq_length = 1024,
            dtype = None,
            load_in_4bit = True,
        )
    except Exception as e:
        print(f"ERROR: Could not load the model. {e}")
        return

    FastLanguageModel.for_inference(model)
    
    total_score = 0
    scores = {}
    for i in range(1, 6):  # Run 5 random scenarios
        score = evaluate_random_task(model, tokenizer, i)
        scores[f"Random_{i}"] = score
        total_score += score
        
    print("\n===============================")
    print("🎲 RANDOM SCENARIO EVALUATION")
    print("===============================")
    for k, v in scores.items():
        print(f"Task {k:10s} | Score: {v:.4f}")
    print(f"Overall Average: {total_score / len(scores):.4f}")

if __name__ == "__main__":
    main()
