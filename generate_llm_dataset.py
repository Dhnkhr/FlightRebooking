"""
Generate a ShareGPT/JSONL formatted dataset for LLM fine-tuning.

This uses the lookahead expert teacher to simulate episodes and records
the environment observations as user prompts and the expert actions
as assistant completions.

Output format for each line (JSON):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Current observation: {...}"},
    {"role": "assistant", "content": "{...}"}
  ]
}
"""

import argparse
import copy
import json
import os
import random
from typing import Any, Dict

from environment import Action, FlightRebookingEnv
from tasks import TASKS
from train_ml_policy import _choose_lookahead_teacher_action, _jitter_task
from inference import SYSTEM_PROMPT


def generate_llm_dataset(
    seed: int,
    episodes_per_task: int,
    lookahead_depth: int,
    lookahead_width: int,
    output_path: str,
):
    rng = random.Random(seed)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    dataset = []
    
    print(f"Generating dataset with {episodes_per_task} episodes per task...")

    for task_key, task_data in TASKS.items():
        print(f"Processing task: {task_key}")
        for episode_idx in range(episodes_per_task):
            variant = _jitter_task(task_data, rng)
            env = FlightRebookingEnv(task_data=variant)
            observation = env.reset()
            done = False
            
            while not done:
                observation_json = observation.model_dump_json()
                observation_dict = observation.model_dump(mode="json")
                
                # Get expert action
                try:
                    action_payload = _choose_lookahead_teacher_action(
                        env=env,
                        task_key=task_key,
                        max_budget=float(variant["max_budget"]),
                        lookahead_depth=lookahead_depth,
                        lookahead_width=lookahead_width,
                    )
                except Exception:
                    # Fallback if lookahead fails
                    from ml_policy import heuristic_action
                    action_payload = heuristic_action(observation_dict)
                
                # Build chat format
                user_content = f"Current observation: {observation_json}"
                assistant_content = json.dumps(action_payload)
                
                dataset.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT.strip()},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                })
                
                action = Action(**action_payload)
                observation, _, done, _ = env.step(action)
                
            if (episode_idx + 1) % 10 == 0:
                print(f"  Completed {episode_idx + 1}/{episodes_per_task} episodes...")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"Finished dataset generation. Saved {len(dataset)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LLM fine-tuning dataset.")
    parser.add_argument("--episodes-per-task", type=int, default=10, help="Episodes per task to simulate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookahead-depth", type=int, default=2)
    parser.add_argument("--lookahead-width", type=int, default=8)
    parser.add_argument("--output", default="artifacts/flight_rebooking_sft.jsonl", help="Path to output JSONL.")
    args = parser.parse_args()

    generate_llm_dataset(
        seed=args.seed,
        episodes_per_task=args.episodes_per_task,
        lookahead_depth=args.lookahead_depth,
        lookahead_width=args.lookahead_width,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
