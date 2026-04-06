"""
Baseline Agent Runner
======================
Uses Meta Llama 3.1 (8B Instruct) via an OpenAI-compatible API to observe
the environment, reason about optimal actions, and loop until done.

Supported providers (set API_BASE_URL env var):
  - Together AI:  https://api.together.xyz/v1
  - Groq:         https://api.groq.com/openai/v1
  - Fireworks:    https://api.fireworks.ai/inference/v1
  - Ollama:       http://localhost:11434/v1
"""

import os
import re
import json
from openai import OpenAI
from environment import FlightRebookingEnv, Action
from tasks import EASY_TASK, MEDIUM_TASK, grade_episode

# ==========================================
# MODEL & API CONFIGURATION
# ==========================================
# Default to Together AI — change API_BASE_URL for other providers
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.together.xyz/v1")
API_KEY = os.getenv("API_KEY", os.getenv("TOGETHER_API_KEY", ""))

# Model name varies by provider — override with MODEL_NAME env var if needed
# Together AI:  meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
# Groq:         llama-3.1-8b-instant
# Fireworks:    accounts/fireworks/models/llama-v3p1-8b-instruct
# Ollama:       llama3.1:8b
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ==========================================
# THE CORE SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """You are an AI Flight Disruption Agent. A massive storm has cancelled flights, and you must rebook the pending passengers.

YOUR GOAL:
Maximize customer satisfaction and save flight connections while respecting priority tiers (Platinum > Gold > Silver > Standard) and staying under budget.

AVAILABLE ACTIONS:
1. "rebook_passenger": Move passenger to a new flight in their original cabin (Free).
2. "offer_downgrade": Move Business passenger to Economy. Costs $500 in compensation.
3. "book_hotel": Give passenger a hotel for the night. Costs $250.
4. "rebook_on_partner": Put passenger on a partner airline flight. Costs $800.
5. "mark_no_solution": Give up on the passenger. (Heavy penalty).
6. "finalize": Use this ONLY when all passengers are processed.

INSTRUCTIONS:
- Process ONE passenger at a time.
- Pay attention to `connection_deadline_hrs`. Prioritize these passengers!
- Platinum and Gold members MUST be processed before Standard members if seats are limited.
- Do not exceed the budget.
- Do NOT include any explanation or commentary. Output ONLY raw JSON.

You must output ONLY valid JSON matching this exact schema:
{
  "action_type": "rebook_passenger" | "offer_downgrade" | "book_hotel" | "rebook_on_partner" | "mark_no_solution" | "finalize",
  "passenger_id": "string (e.g., 'P1')",
  "flight_id": "string (optional, e.g., 'FL-102')"
}"""


def extract_json(text: str) -> dict:
    """
    Robustly extract JSON from LLM output.
    Llama models sometimes wrap JSON in markdown code blocks or add commentary.
    """
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block ```json ... ```
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } in the text
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from model output: {text[:200]}")


def run_evaluation(task_data, task_name):
    """Run a single evaluation episode."""
    print(f"\n--- Starting Task: {task_name} ---")
    env = FlightRebookingEnv(task_data=task_data)
    obs = env.reset()
    done = False
    retries = 0
    max_retries = 3

    while not done:
        # 1. Ask the AI what to do
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Current State: {obs.model_dump_json()}"},
                ],
                temperature=0.1,  # Low temp for consistency (Llama can be 0 but 0.1 is safer)
                max_tokens=256,   # Actions are short — save tokens
            )
        except Exception as e:
            print(f"API call failed: {e}")
            if retries < max_retries:
                retries += 1
                print(f"Retrying... ({retries}/{max_retries})")
                continue
            print("Max retries reached. Forcing finalize.")
            action = Action(action_type="finalize")
            obs, reward, done, info = env.step(action)
            continue

        # 2. Parse the AI's JSON output (with robust extraction for Llama)
        raw_output = response.choices[0].message.content or ""
        try:
            action_dict = extract_json(raw_output)
            action = Action(**action_dict)
            retries = 0  # Reset retries on success
        except Exception as e:
            print(f"  [WARN] Invalid output from model. Error: {e}")
            if retries < max_retries:
                retries += 1
                print(f"  Retrying... ({retries}/{max_retries})")
                continue
            print("  Max retries reached. Forcing finalize.")
            action = Action(action_type="finalize")

        # 3. Take the action in the environment
        obs, reward, done, info = env.step(action)

        # Print what the AI did for the terminal demo
        if action.action_type != "finalize":
            target = f"Passenger: {action.passenger_id}"
            if action.flight_id:
                target += f" -> Flight: {action.flight_id}"
            print(f"  Action: {action.action_type.upper()} | {target} | Reward: {reward:.2f}")

        # Print any errors from the environment
        if "error" in info:
            print(f"  [ENV ERROR] {info['error']}")

    # 4. Grade the final result
    final_score = grade_episode(env.state(), task_data["max_budget"])
    print(f"  Task '{task_name}' Completed. Score: {final_score:.2f} / 1.00")
    print(f"  Budget Spent: ${env.state().budget_spent} / ${task_data['max_budget']}")
    return final_score


if __name__ == "__main__":
    if not API_KEY:
        print("Error: Please set your API_KEY (or TOGETHER_API_KEY) environment variable.")
        print(f"  Current API base: {API_BASE_URL}")
        print(f"  Current model:    {MODEL_NAME}")
        print()
        print("Examples:")
        print("  set API_KEY=your-together-api-key")
        print("  set API_BASE_URL=https://api.together.xyz/v1")
        print("  set MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        exit(1)

    print(f"Using model: {MODEL_NAME}")
    print(f"API base:    {API_BASE_URL}")

    scores = []
    scores.append(run_evaluation(EASY_TASK, "Easy - Minor Disruption"))
    scores.append(run_evaluation(MEDIUM_TASK, "Medium - Connection Crisis"))

    print("\n" + "=" * 50)
    print(f"Overall Average Score: {sum(scores) / len(scores):.2f} / 1.00")
    print("=" * 50)
