"""
Reproducible Baseline Runner
============================

Runs an OpenAI model against all flight-rebooking tasks using deterministic
settings and reports normalized scores in [0.0, 1.0].
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from environment import Action, ActionType, CabinClass, FlightRebookingEnv, PriorityTier
from tasks import TASKS, grade_task


DEFAULT_API_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_OPEN_SOURCE_MODEL = "llama-3.1-8b-instant"
INTERNAL_GROQ_API_KEY = ""


SYSTEM_PROMPT = """You are an airline disruption operations agent.

Return exactly one JSON object on each turn with this schema:
{
  "action_type": "rebook_passenger" | "offer_downgrade" | "book_hotel" | "rebook_on_partner" | "mark_no_solution" | "finalize",
  "passenger_id": "optional passenger id",
  "flight_id": "optional flight id"
}

Behavior policy:
- Process one pending passenger per step.
- Respect priority tiers (Platinum > Gold > Silver > Standard).
- For same tier, prioritize tighter connection deadlines.
- Prefer same-airline rebooking over partner booking when feasible.
- Keep spend low and avoid invalid actions.
- Emit raw JSON only. No markdown, no explanations.
"""


def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"No valid JSON action found in model output: {text[:200]}")


def _first_non_empty(*values: str) -> str:
    for value in values:
        cleaned = (value or "").strip()
        if cleaned:
            return cleaned
    return ""


def _tier_weight(tier: str) -> int:
    return {
        PriorityTier.PLATINUM.value: 4,
        PriorityTier.GOLD.value: 3,
        PriorityTier.SILVER.value: 2,
        PriorityTier.STANDARD.value: 1,
    }.get(tier, 1)


def _has_seat(flight: Dict[str, Any], cabin_class: str) -> bool:
    if cabin_class == CabinClass.BUSINESS.value:
        return flight["business_seats"] > 0
    return flight["economy_seats"] > 0


def heuristic_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    pending = list(observation["pending_passengers"])
    if not pending:
        return {"action_type": ActionType.FINALIZE.value}

    pending.sort(
        key=lambda p: (
            -_tier_weight(p["priority_tier"]),
            p["connection_deadline_hrs"] if p["connection_deadline_hrs"] is not None else 10**9,
        )
    )

    passenger = pending[0]
    flights = sorted(observation["available_flights"], key=lambda f: f["departure_hrs"])

    for flight in flights:
        if flight["is_partner"]:
            continue
        if _has_seat(flight, passenger["cabin_class"]):
            return {
                "action_type": ActionType.REBOOK_PASSENGER.value,
                "passenger_id": passenger["id"],
                "flight_id": flight["id"],
            }

    if passenger["cabin_class"] == CabinClass.BUSINESS.value:
        for flight in flights:
            if flight["is_partner"]:
                continue
            if flight["economy_seats"] > 0 and observation["budget_remaining"] >= 500:
                return {
                    "action_type": ActionType.OFFER_DOWNGRADE.value,
                    "passenger_id": passenger["id"],
                    "flight_id": flight["id"],
                }

    for flight in flights:
        if not flight["is_partner"]:
            continue
        if _has_seat(flight, passenger["cabin_class"]) and observation["budget_remaining"] >= 800:
            return {
                "action_type": ActionType.REBOOK_ON_PARTNER.value,
                "passenger_id": passenger["id"],
                "flight_id": flight["id"],
            }

    if observation["budget_remaining"] >= 250:
        return {
            "action_type": ActionType.BOOK_HOTEL.value,
            "passenger_id": passenger["id"],
        }

    return {
        "action_type": ActionType.MARK_NO_SOLUTION.value,
        "passenger_id": passenger["id"],
    }


def is_action_feasible(observation: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    action_type = payload["action_type"]
    if action_type == ActionType.FINALIZE.value:
        return True

    pending_by_id = {p["id"]: p for p in observation["pending_passengers"]}
    flights_by_id = {f["id"]: f for f in observation["available_flights"]}
    budget_remaining = float(observation["budget_remaining"])

    passenger = pending_by_id.get(payload.get("passenger_id"))
    if passenger is None:
        return False

    if action_type == ActionType.BOOK_HOTEL.value:
        return budget_remaining >= 250

    if action_type == ActionType.MARK_NO_SOLUTION.value:
        return True

    flight = flights_by_id.get(payload.get("flight_id"))
    if flight is None:
        return False

    passenger_cabin = passenger["cabin_class"]
    needs_business = passenger_cabin == CabinClass.BUSINESS.value
    has_matching_cabin_seat = (flight["business_seats"] > 0) if needs_business else (flight["economy_seats"] > 0)

    if action_type == ActionType.REBOOK_PASSENGER.value:
        return (not flight["is_partner"]) and has_matching_cabin_seat

    if action_type == ActionType.OFFER_DOWNGRADE.value:
        return (
            passenger_cabin == CabinClass.BUSINESS.value
            and budget_remaining >= 500
            and flight["economy_seats"] > 0
        )

    if action_type == ActionType.REBOOK_ON_PARTNER.value:
        return flight["is_partner"] and budget_remaining >= 800 and has_matching_cabin_seat

    return False


def sanitize_action_payload(observation: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    fallback = heuristic_action(observation)

    if not isinstance(payload, dict):
        return fallback

    valid_action_types = {action_type.value for action_type in ActionType}
    action_type = str(payload.get("action_type", "")).strip()
    if action_type not in valid_action_types:
        return fallback

    sanitized: Dict[str, Any] = {"action_type": action_type}
    passenger_id = str(payload.get("passenger_id", "")).strip()
    flight_id = str(payload.get("flight_id", "")).strip()

    if passenger_id:
        sanitized["passenger_id"] = passenger_id
    if flight_id:
        sanitized["flight_id"] = flight_id

    if action_type == ActionType.FINALIZE.value:
        return sanitized

    pending_ids = {p["id"] for p in observation["pending_passengers"]}
    if sanitized.get("passenger_id") not in pending_ids:
        return fallback

    if action_type in {
        ActionType.REBOOK_PASSENGER.value,
        ActionType.OFFER_DOWNGRADE.value,
        ActionType.REBOOK_ON_PARTNER.value,
    }:
        flight_ids = {f["id"] for f in observation["available_flights"]}
        if sanitized.get("flight_id") not in flight_ids:
            return fallback

    if not is_action_feasible(observation, sanitized):
        return fallback

    return sanitized


def query_openai_action(
    client: OpenAI,
    model: str,
    seed: int,
    observation_json: str,
    max_retries: int = 2,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None

    for _ in range(max_retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Current observation: {observation_json}"},
                ],
                "temperature": 0,
                "top_p": 1,
                "seed": seed,
                "max_tokens": 200,
            }

            try:
                response = client.chat.completions.create(**kwargs)
            except TypeError:
                # Some providers do not support seed on chat completions.
                kwargs.pop("seed", None)
                response = client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content or ""
            return extract_json(content)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"OpenAI action query failed after retries: {last_error}")


def run_episode(
    task_key: str,
    task_data: Dict[str, Any],
    policy: str,
    model: str,
    seed: int,
    client: Optional[OpenAI],
) -> Dict[str, Any]:
    env = FlightRebookingEnv(task_data=task_data)
    observation = env.reset()
    done = False
    step_rewards: List[float] = []
    steps = 0

    while not done:
        observation_dict = observation.model_dump(mode="json")

        if policy == "heuristic":
            action_payload = heuristic_action(observation_dict)
        else:
            raw_payload = query_openai_action(
                client=client,
                model=model,
                seed=seed,
                observation_json=observation.model_dump_json(),
            )
            action_payload = sanitize_action_payload(observation_dict, raw_payload)

        try:
            action = Action(**action_payload)
        except Exception:
            action = Action(action_type=ActionType.FINALIZE)

        observation, reward, done, _ = env.step(action)
        step_rewards.append(reward.value)
        steps += 1

    final_state = env.state()
    final_score = grade_task(task_key, final_state, task_data["max_budget"])

    return {
        "task": task_key,
        "difficulty": task_data["difficulty"],
        "score": round(final_score, 4),
        "avg_step_reward": round(sum(step_rewards) / max(len(step_rewards), 1), 4),
        "steps": steps,
        "budget_spent": round(final_state.budget_spent, 2),
        "budget_max": task_data["max_budget"],
        "invalid_actions": final_state.invalid_actions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible baselines on all OpenEnv tasks.")
    parser.add_argument(
        "--policy",
        choices=["openai", "heuristic"],
        default="openai",
        help="Policy backend. `openai` uses chat completions, `heuristic` is deterministic fallback.",
    )
    parser.add_argument(
        "--model",
        default=_first_non_empty(
            os.getenv("OPENAI_MODEL", ""),
            os.getenv("MODEL_NAME", ""),
            DEFAULT_OPEN_SOURCE_MODEL,
        ),
        help="Model name for --policy openai.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("BASELINE_SEED", "42")),
        help="Random seed forwarded to the OpenAI API when supported.",
    )
    parser.add_argument(
        "--task",
        choices=["all", "easy", "medium", "hard"],
        default="all",
        help="Run a single task or all tasks.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_keys = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    client: Optional[OpenAI] = None
    if args.policy == "openai":
        api_key = _first_non_empty(
            os.getenv("OPENAI_API_KEY", ""),
            os.getenv("HF_TOKEN", ""),
            os.getenv("GROQ_API_KEY", ""),
            INTERNAL_GROQ_API_KEY,
        )
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY is not set. Set it in your environment or use --policy heuristic."
            )

        base_url = _first_non_empty(
            os.getenv("OPENAI_BASE_URL", ""),
            os.getenv("API_BASE_URL", ""),
            DEFAULT_API_BASE_URL,
        )
        client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"Policy: {args.policy}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")

    results: List[Dict[str, Any]] = []
    for task_key in task_keys:
        task_data = TASKS[task_key]
        print(f"\n--- Running {task_key.upper()} ({task_data['task_id']}) ---")
        result = run_episode(
            task_key=task_key,
            task_data=task_data,
            policy=args.policy,
            model=args.model,
            seed=args.seed,
            client=client,
        )
        results.append(result)
        print(
            "Score={score:.4f} | AvgStepReward={avg_step_reward:.4f} | "
            "Steps={steps} | Budget=${budget_spent:.2f}/${budget_max:.2f} | Invalid={invalid_actions}".format(
                **result
            )
        )

    overall = sum(r["score"] for r in results) / max(len(results), 1)
    print("\n" + "=" * 70)
    print(f"Overall Average Score: {overall:.4f} / 1.0000")
    print("=" * 70)

    payload = {
        "policy": args.policy,
        "model": args.model,
        "seed": args.seed,
        "overall_score": round(overall, 4),
        "tasks": results,
    }

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved results to: {args.json_out}")


if __name__ == "__main__":
    main()
