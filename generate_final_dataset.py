"""
Generate a LARGE, edge-case-heavy dataset for final LLM training.

This script generates three types of data:
1. Normal jittered episodes (broad coverage)
2. Extreme-jitter episodes (very tight budgets, scarce seats)  
3. Targeted edge-case episodes (force rare actions like hotel, downgrade, no_solution)

The result is a single JSONL file with maximum diversity for robust training.
"""

import argparse
import copy
import json
import os
import random
from typing import Any, Dict, List

from environment import Action, FlightRebookingEnv
from tasks import TASKS
from train_ml_policy import _choose_lookahead_teacher_action, _jitter_task
from inference import SYSTEM_PROMPT


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _extreme_jitter_task(task_data: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Create extremely challenging variants with very tight constraints."""
    variant = copy.deepcopy(task_data)
    
    # Much tighter budget (50-80% of original)
    budget_factor = rng.uniform(0.50, 0.80)
    variant["max_budget"] = max(500, int(round(float(variant["max_budget"]) * budget_factor / 50.0) * 50))
    
    # Generous step limit so agent has room to recover
    max_steps = int(variant.get("max_steps", 80))
    variant["max_steps"] = max(40, min(140, max_steps + rng.randint(0, 20)))
    
    # Tighter deadlines
    for passenger in variant["passengers"]:
        deadline = passenger.get("connection_deadline_hrs")
        if deadline is None:
            # 30% chance to add a tight deadline
            if rng.random() < 0.30:
                passenger["connection_deadline_hrs"] = round(rng.uniform(1.5, 4.0), 1)
        else:
            # Make deadlines tighter
            passenger["connection_deadline_hrs"] = round(_clamp(float(deadline) - rng.uniform(0.3, 1.5), 1.0, 6.0), 1)
    
    # Scarce seats
    for flight in variant["flights"]:
        flight["departure_hrs"] = round(_clamp(float(flight["departure_hrs"]) + rng.uniform(-1.5, 1.5), 0.5, 12.0), 1)
        flight["economy_seats"] = max(0, int(flight["economy_seats"]) + rng.randint(-4, 1))
        flight["business_seats"] = max(0, int(flight["business_seats"]) + rng.randint(-2, 0))
    
    rng.shuffle(variant["passengers"])
    rng.shuffle(variant["flights"])
    return variant


def _hotel_forcing_task(task_data: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Create scenarios where flights are completely full, forcing hotel bookings."""
    variant = copy.deepcopy(task_data)
    
    variant["max_budget"] = max(2000, int(float(variant["max_budget"]) * rng.uniform(0.7, 1.0)))
    variant["max_steps"] = max(40, int(variant.get("max_steps", 80)) + 10)
    
    # Zero out most seats to force hotel/no_solution
    for flight in variant["flights"]:
        flight["economy_seats"] = rng.randint(0, 1)
        flight["business_seats"] = 0
        flight["departure_hrs"] = round(rng.uniform(1.0, 8.0), 1)
    
    # Keep one flight with a tiny bit of capacity
    if variant["flights"]:
        lucky_flight = rng.choice(variant["flights"])
        lucky_flight["economy_seats"] = rng.randint(1, 2)
        lucky_flight["business_seats"] = rng.randint(0, 1)
    
    for passenger in variant["passengers"]:
        if passenger.get("connection_deadline_hrs") is None and rng.random() < 0.2:
            passenger["connection_deadline_hrs"] = round(rng.uniform(2.0, 6.0), 1)
    
    rng.shuffle(variant["passengers"])
    rng.shuffle(variant["flights"])
    return variant


def _downgrade_forcing_task(task_data: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Create scenarios where business passengers have no business seats available."""
    variant = copy.deepcopy(task_data)
    
    variant["max_budget"] = max(1500, int(float(variant["max_budget"]) * rng.uniform(0.6, 0.9)))
    variant["max_steps"] = max(40, int(variant.get("max_steps", 80)) + 10)
    
    # Zero out ALL business seats
    for flight in variant["flights"]:
        flight["business_seats"] = 0
        flight["economy_seats"] = max(1, int(flight["economy_seats"]) + rng.randint(-2, 2))
        flight["departure_hrs"] = round(_clamp(float(flight["departure_hrs"]) + rng.uniform(-1.0, 1.0), 0.5, 10.0), 1)
    
    # Ensure at least some passengers need Business
    for passenger in variant["passengers"]:
        if rng.random() < 0.4:
            passenger["cabin_class"] = "Business"
    
    rng.shuffle(variant["passengers"])
    rng.shuffle(variant["flights"])
    return variant


def _partner_heavy_task(task_data: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Create scenarios where most flights are partner airlines."""
    variant = copy.deepcopy(task_data)
    
    variant["max_budget"] = max(2000, int(float(variant["max_budget"]) * rng.uniform(0.8, 1.1)))
    variant["max_steps"] = max(40, int(variant.get("max_steps", 80)))
    
    # Make most flights partner airlines
    for flight in variant["flights"]:
        flight["is_partner"] = rng.random() < 0.7  # 70% partner
        flight["economy_seats"] = max(1, int(flight["economy_seats"]) + rng.randint(-1, 2))
        flight["business_seats"] = max(0, int(flight["business_seats"]) + rng.randint(-1, 1))
        flight["departure_hrs"] = round(_clamp(float(flight["departure_hrs"]) + rng.uniform(-1.0, 1.0), 0.5, 10.0), 1)
    
    rng.shuffle(variant["passengers"])
    rng.shuffle(variant["flights"])
    return variant


def run_episode(task_data: Dict[str, Any], task_key: str, lookahead_depth: int, lookahead_width: int) -> List[dict]:
    """Run a single episode and collect training samples."""
    samples = []
    env = FlightRebookingEnv(task_data=task_data)
    observation = env.reset()
    done = False
    
    while not done:
        observation_json = observation.model_dump_json()
        
        try:
            action_payload = _choose_lookahead_teacher_action(
                env=env,
                task_key=task_key,
                max_budget=float(task_data["max_budget"]),
                lookahead_depth=lookahead_depth,
                lookahead_width=lookahead_width,
            )
        except Exception:
            from ml_policy import heuristic_action
            obs_dict = observation.model_dump(mode="json")
            action_payload = heuristic_action(obs_dict)
        
        user_content = f"Current observation: {observation_json}"
        assistant_content = json.dumps(action_payload)
        
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })
        
        action = Action(**action_payload)
        observation, _, done, _ = env.step(action)
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate maximum-diversity LLM fine-tuning dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookahead-depth", type=int, default=2)
    parser.add_argument("--lookahead-width", type=int, default=8)
    parser.add_argument("--output", default="artifacts/flight_rebooking_sft_final.jsonl")
    args = parser.parse_args()
    
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    dataset = []
    
    # ── Phase 1: Normal jittered episodes (broad coverage) ──
    print("=" * 60)
    print("Phase 1: Normal jittered episodes (200 per task)")
    print("=" * 60)
    for task_key, task_data in TASKS.items():
        print(f"  Processing task: {task_key}")
        for i in range(200):
            variant = _jitter_task(task_data, rng)
            samples = run_episode(variant, task_key, args.lookahead_depth, args.lookahead_width)
            dataset.extend(samples)
            if (i + 1) % 50 == 0:
                print(f"    Completed {i+1}/200 episodes... ({len(dataset)} samples so far)")
    
    # ── Phase 2: Extreme-jitter episodes (tight constraints) ──
    print("\n" + "=" * 60)
    print("Phase 2: Extreme-jitter episodes (100 per task)")
    print("=" * 60)
    for task_key, task_data in TASKS.items():
        print(f"  Processing task: {task_key}")
        for i in range(100):
            variant = _extreme_jitter_task(task_data, rng)
            samples = run_episode(variant, task_key, args.lookahead_depth, args.lookahead_width)
            dataset.extend(samples)
            if (i + 1) % 50 == 0:
                print(f"    Completed {i+1}/100 episodes... ({len(dataset)} samples so far)")
    
    # ── Phase 3: Hotel-forcing episodes ──
    print("\n" + "=" * 60)
    print("Phase 3: Hotel-forcing episodes (80 per task)")
    print("=" * 60)
    for task_key, task_data in TASKS.items():
        print(f"  Processing task: {task_key}")
        for i in range(80):
            variant = _hotel_forcing_task(task_data, rng)
            samples = run_episode(variant, task_key, args.lookahead_depth, args.lookahead_width)
            dataset.extend(samples)
            if (i + 1) % 40 == 0:
                print(f"    Completed {i+1}/80 episodes... ({len(dataset)} samples so far)")
    
    # ── Phase 4: Downgrade-forcing episodes ──
    print("\n" + "=" * 60)
    print("Phase 4: Downgrade-forcing episodes (80 per task)")
    print("=" * 60)
    for task_key, task_data in TASKS.items():
        print(f"  Processing task: {task_key}")
        for i in range(80):
            variant = _downgrade_forcing_task(task_data, rng)
            samples = run_episode(variant, task_key, args.lookahead_depth, args.lookahead_width)
            dataset.extend(samples)
            if (i + 1) % 40 == 0:
                print(f"    Completed {i+1}/80 episodes... ({len(dataset)} samples so far)")
    
    # ── Phase 5: Partner-heavy episodes ──
    print("\n" + "=" * 60)
    print("Phase 5: Partner-heavy episodes (40 per task)")
    print("=" * 60)
    for task_key, task_data in TASKS.items():
        print(f"  Processing task: {task_key}")
        for i in range(40):
            variant = _partner_heavy_task(task_data, rng)
            samples = run_episode(variant, task_key, args.lookahead_depth, args.lookahead_width)
            dataset.extend(samples)
            if (i + 1) % 20 == 0:
                print(f"    Completed {i+1}/40 episodes... ({len(dataset)} samples so far)")
    
    # ── Shuffle the entire dataset for better training ──
    rng.shuffle(dataset)
    
    # ── Save ──
    with open(args.output, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    
    # ── Print action distribution ──
    print("\n" + "=" * 60)
    print(f"FINAL DATASET: {len(dataset)} samples saved to {args.output}")
    print("=" * 60)
    
    action_counts = {}
    for item in dataset:
        action = json.loads(item["messages"][2]["content"]).get("action_type", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print("\nAction Distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count * 100 / len(dataset)
        bar = "█" * int(pct)
        print(f"  {action:25s} {count:5d}  ({pct:5.1f}%)  {bar}")


if __name__ == "__main__":
    main()
