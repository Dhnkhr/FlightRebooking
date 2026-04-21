"""
Submission inference runner.

Requirements covered:
- Script name is inference.py in repo root.
- Uses OpenAI client for model calls.
- Uses internal Groq + Llama 8B defaults (overridable via environment).
- Emits structured stdout logs with [START], [STEP], [END].
"""

import argparse
from copy import deepcopy
import json
import os
import pickle
import re
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

from environment import Action, ActionType, CabinClass, FlightRebookingEnv, PriorityTier
from ml_policy import choose_action_from_ranked_types, observation_to_features
from tasks import TASKS, grade_task


SYSTEM_PROMPT = """You are an airline disruption operations agent.

Return exactly one JSON object on each turn with this schema:
{
  \"action_type\": \"rebook_passenger\" | \"offer_downgrade\" | \"book_hotel\" | \"rebook_on_partner\" | \"mark_no_solution\" | \"finalize\",
  \"passenger_id\": \"optional passenger id\",
  \"flight_id\": \"optional flight id\"
}

Policy:
- Process one pending passenger per step.
- Respect tiers (Platinum > Gold > Silver > Standard).
- Prefer earlier departures for deadline passengers.
- Prefer same-airline rebooking over partner when feasible.
- Minimize budget usage.
- Output raw JSON only.
"""

DEFAULT_API_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"
INTERNAL_GROQ_API_KEY = ""
BENCHMARK_NAME = os.getenv("BENCHMARK", "flight-rebooking-openenv")
SUCCESS_SCORE_THRESHOLD = 0.1
GIT_LFS_POINTER_HEADER = "version https://git-lfs.github.com/spec/v1"


def _first_non_empty(*values: str) -> str:
    for value in values:
        cleaned = (value or "").strip()
        if cleaned:
            return cleaned
    return ""


def _resolve_model_config() -> Dict[str, str]:
    api_base_url = _first_non_empty(
        os.getenv("API_BASE_URL", ""),
        os.getenv("OPENAI_BASE_URL", ""),
        DEFAULT_API_BASE_URL,
    )
    model_name = _first_non_empty(
        os.getenv("MODEL_NAME", ""),
        os.getenv("OPENAI_MODEL", ""),
        DEFAULT_LLM_MODEL,
    )
    api_key = _first_non_empty(
        os.getenv("GROQ_API_KEY", ""),
        os.getenv("HF_TOKEN", ""),
        os.getenv("OPENAI_API_KEY", ""),
        INTERNAL_GROQ_API_KEY,
    )

    if not api_key:
        raise SystemExit(
            "No API key configured. Set GROQ_API_KEY (preferred), OPENAI_API_KEY, or HF_TOKEN."
        )

    return {
        "api_base_url": api_base_url,
        "model_name": model_name,
        "api_key": api_key,
    }


def _load_ml_policy_artifact(path: str) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as handle:
            artifact = pickle.load(handle)
    except Exception as exc:
        print(f"[WARN] Failed to load ML policy artifact at {path}: {exc}", file=sys.stderr)
        return None

    if not isinstance(artifact, dict) or "model" not in artifact:
        print(f"[WARN] Invalid ML policy artifact format at {path}; ignoring.", file=sys.stderr)
        return None

    return artifact


def _is_git_lfs_pointer_file(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = [handle.readline().strip() for _ in range(3)]
    except (UnicodeDecodeError, OSError):
        return False

    if not lines or lines[0] != GIT_LFS_POINTER_HEADER:
        return False

    return any(line.startswith("oid sha256:") for line in lines[1:])


def _ml_policy_fix_instructions(path: str) -> str:
    return (
        "Fix options:\n"
        "1) Materialize artifact bytes with Git LFS (if this repo stores models in LFS):\n"
        f"   git lfs pull --include \"{path}\"\n"
        "2) Regenerate the artifact locally:\n"
        "   python train_ml_policy.py --episodes-per-task 450 --seed 42 --output artifacts/ml_policy.pkl --report artifacts/ml_policy_report.json"
    )


def _require_ml_policy_artifact(path: str, policy_name: str) -> Dict[str, Any]:
    if not path:
        raise SystemExit(
            f"Policy '{policy_name}' requires --ml-policy-path.\n"
            + _ml_policy_fix_instructions("artifacts/ml_policy.pkl")
        )

    if not os.path.exists(path):
        raise SystemExit(
            f"Policy '{policy_name}' requires an ML artifact, but '{path}' was not found.\n"
            + _ml_policy_fix_instructions(path)
        )

    if _is_git_lfs_pointer_file(path):
        raise SystemExit(
            f"Policy '{policy_name}' cannot run because '{path}' is a Git LFS pointer, not a pickle artifact.\n"
            + _ml_policy_fix_instructions(path)
        )

    artifact = _load_ml_policy_artifact(path)
    if artifact is None:
        raise SystemExit(
            f"Policy '{policy_name}' requires a valid ML artifact, but '{path}' could not be loaded as a pickle.\n"
            + _ml_policy_fix_instructions(path)
        )

    return artifact


def _rank_action_types_from_model(model: Any, features: List[float]) -> List[str]:
    ranked: List[str]

    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        probabilities = model.predict_proba([features])[0]
        classes = [str(cls) for cls in model.classes_]
        ranked = [
            label
            for _, label in sorted(
                zip(probabilities, classes),
                key=lambda item: item[0],
                reverse=True,
            )
        ]
    else:
        ranked = [str(model.predict([features])[0])]

    for action_type in (
        ActionType.REBOOK_PASSENGER.value,
        ActionType.OFFER_DOWNGRADE.value,
        ActionType.REBOOK_ON_PARTNER.value,
        ActionType.BOOK_HOTEL.value,
        ActionType.MARK_NO_SOLUTION.value,
        ActionType.FINALIZE.value,
    ):
        if action_type not in ranked:
            ranked.append(action_type)

    return ranked


def _predict_ml_policy_action(observation: Dict[str, Any], ml_policy_artifact: Dict[str, Any]) -> Dict[str, Any]:
    model = ml_policy_artifact["model"]
    features = observation_to_features(observation)
    ranked_action_types = _rank_action_types_from_model(model, features)
    return choose_action_from_ranked_types(observation, ranked_action_types)


def _predict_ml_ranked_action_types(observation: Dict[str, Any], ml_policy_artifact: Dict[str, Any]) -> List[str]:
    model = ml_policy_artifact["model"]
    features = observation_to_features(observation)
    return _rank_action_types_from_model(model, features)


def _feasible_actions_from_observation(observation: Dict[str, Any]) -> List[Action]:
    pending = list(observation.get("pending_passengers", []))
    flights = list(observation.get("available_flights", []))
    budget_remaining = float(observation.get("budget_remaining", 0.0))

    if not pending:
        return [Action(action_type=ActionType.FINALIZE)]

    actions: List[Action] = []
    for passenger in pending:
        for flight in flights:
            if (not flight.get("is_partner", False)) and _has_seat(flight, str(passenger.get("cabin_class", ""))):
                actions.append(
                    Action(
                        action_type=ActionType.REBOOK_PASSENGER,
                        passenger_id=passenger["id"],
                        flight_id=flight["id"],
                    )
                )

            if (
                passenger.get("cabin_class") == CabinClass.BUSINESS.value
                and (not flight.get("is_partner", False))
                and int(flight.get("economy_seats", 0)) > 0
                and budget_remaining >= 500.0
            ):
                actions.append(
                    Action(
                        action_type=ActionType.OFFER_DOWNGRADE,
                        passenger_id=passenger["id"],
                        flight_id=flight["id"],
                    )
                )

            if (
                flight.get("is_partner", False)
                and _has_seat(flight, str(passenger.get("cabin_class", "")))
                and budget_remaining >= 800.0
            ):
                actions.append(
                    Action(
                        action_type=ActionType.REBOOK_ON_PARTNER,
                        passenger_id=passenger["id"],
                        flight_id=flight["id"],
                    )
                )

        if budget_remaining >= 250.0:
            actions.append(
                Action(
                    action_type=ActionType.BOOK_HOTEL,
                    passenger_id=passenger["id"],
                )
            )

        actions.append(
            Action(
                action_type=ActionType.MARK_NO_SOLUTION,
                passenger_id=passenger["id"],
            )
        )

    actions.append(Action(action_type=ActionType.FINALIZE))
    return actions


def _action_cost(action_type: ActionType) -> float:
    return {
        ActionType.REBOOK_PASSENGER: 0.0,
        ActionType.OFFER_DOWNGRADE: 500.0,
        ActionType.BOOK_HOTEL: 250.0,
        ActionType.REBOOK_ON_PARTNER: 800.0,
        ActionType.MARK_NO_SOLUTION: 0.0,
        ActionType.FINALIZE: 0.0,
    }.get(action_type, 0.0)


def _action_priority_score(observation: Dict[str, Any], action: Action) -> float:
    pending = list(observation.get("pending_passengers", []))
    if action.action_type == ActionType.FINALIZE:
        return 10.0 if not pending else -10.0

    pending_by_id = {p["id"]: p for p in pending}
    flights_by_id = {f["id"]: f for f in observation.get("available_flights", [])}

    passenger = pending_by_id.get(action.passenger_id or "")
    if passenger is None:
        return -100.0

    tier_component = _tier_weight(str(passenger.get("priority_tier", ""))) / 4.0
    deadline = passenger.get("connection_deadline_hrs")
    if deadline is None:
        deadline_component = 0.0
    else:
        deadline_component = (12.0 - min(max(float(deadline), 0.0), 12.0)) / 12.0

    score = (0.65 * tier_component) + (0.35 * deadline_component)

    type_bonus = {
        ActionType.REBOOK_PASSENGER: 0.60,
        ActionType.OFFER_DOWNGRADE: 0.30,
        ActionType.REBOOK_ON_PARTNER: 0.18,
        ActionType.BOOK_HOTEL: 0.10,
        ActionType.MARK_NO_SOLUTION: -0.60,
        ActionType.FINALIZE: 0.0,
    }[action.action_type]
    score += type_bonus

    if action.flight_id:
        flight = flights_by_id.get(action.flight_id)
        if flight is not None and deadline is not None:
            departure = float(flight.get("departure_hrs", 99.0))
            if departure <= float(deadline):
                score += 0.22
            else:
                score -= 0.22

    budget_remaining = float(observation.get("budget_remaining", 0.0))
    budget_spent = float(observation.get("budget_spent", 0.0))
    budget_total = max(budget_remaining + budget_spent, 1.0)
    score -= 0.35 * min(_action_cost(action.action_type) / budget_total, 1.0)

    return score


def _prune_candidate_actions(
    observation: Dict[str, Any],
    actions: List[Action],
    ranked_action_types: Optional[List[str]],
    max_candidates: int,
) -> List[Action]:
    deduped: List[Action] = []
    seen = set()
    for action in actions:
        signature = (action.action_type.value, action.passenger_id, action.flight_id)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(action)

    rank_index: Dict[str, int] = {}
    if ranked_action_types:
        rank_index = {action_type: idx for idx, action_type in enumerate(ranked_action_types)}

    deduped.sort(
        key=lambda action: (
            rank_index.get(action.action_type.value, 999),
            -_action_priority_score(observation, action),
        )
    )
    return deduped[: max(1, max_candidates)]


def _rollout_heuristic_to_end(env: FlightRebookingEnv) -> None:
    done = False
    while not done:
        observation = env._get_observation().model_dump(mode="json")
        action = Action(**_heuristic_action(observation))
        _, _, done, _ = env.step(action)


def _evaluate_state_with_lookahead(
    env: FlightRebookingEnv,
    task_key: str,
    lookahead_depth: int,
    lookahead_width: int,
    ranked_action_types: Optional[List[str]] = None,
) -> float:
    observation = env._get_observation().model_dump(mode="json")
    candidate_actions = _feasible_actions_from_observation(observation)

    if ranked_action_types:
        preferred_types = set(ranked_action_types[:5])
        preferred_types.add(ActionType.FINALIZE.value)
        preferred_types.add(ActionType.MARK_NO_SOLUTION.value)
        preferred_candidates = [a for a in candidate_actions if a.action_type.value in preferred_types]
        if preferred_candidates:
            candidate_actions = preferred_candidates

    candidate_actions = _prune_candidate_actions(
        observation=observation,
        actions=candidate_actions,
        ranked_action_types=ranked_action_types,
        max_candidates=lookahead_width,
    )

    best_score = -1.0
    for action in candidate_actions:
        env_copy = deepcopy(env)
        _, _, done, _ = env_copy.step(action)

        if done:
            score = float(grade_task(task_key, env_copy.state(), TASKS[task_key]["max_budget"]))
        elif lookahead_depth <= 1:
            _rollout_heuristic_to_end(env_copy)
            score = float(grade_task(task_key, env_copy.state(), TASKS[task_key]["max_budget"]))
        else:
            score = _evaluate_state_with_lookahead(
                env=env_copy,
                task_key=task_key,
                lookahead_depth=lookahead_depth - 1,
                lookahead_width=lookahead_width,
                ranked_action_types=None,
            )

        if score > best_score:
            best_score = score

    if best_score >= 0.0:
        return best_score

    env_fallback = deepcopy(env)
    _rollout_heuristic_to_end(env_fallback)
    return float(grade_task(task_key, env_fallback.state(), TASKS[task_key]["max_budget"]))


def _projected_score_for_action(
    env: FlightRebookingEnv,
    task_key: str,
    action: Action,
    lookahead_depth: int,
    lookahead_width: int,
) -> float:
    env_copy = deepcopy(env)
    _, _, done, _ = env_copy.step(action)
    if done:
        return float(grade_task(task_key, env_copy.state(), TASKS[task_key]["max_budget"]))

    if lookahead_depth <= 1:
        _rollout_heuristic_to_end(env_copy)
        return float(grade_task(task_key, env_copy.state(), TASKS[task_key]["max_budget"]))

    return _evaluate_state_with_lookahead(
        env=env_copy,
        task_key=task_key,
        lookahead_depth=lookahead_depth - 1,
        lookahead_width=lookahead_width,
        ranked_action_types=None,
    )


def _choose_lookahead_action(
    env: FlightRebookingEnv,
    task_key: str,
    lookahead_depth: int,
    lookahead_width: int,
    ranked_action_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    observation = env._get_observation().model_dump(mode="json")
    candidate_actions = _feasible_actions_from_observation(observation)

    if ranked_action_types:
        preferred_types = set(ranked_action_types[:5])
        preferred_types.add(ActionType.FINALIZE.value)
        preferred_types.add(ActionType.MARK_NO_SOLUTION.value)
        preferred_candidates = [a for a in candidate_actions if a.action_type.value in preferred_types]
        if preferred_candidates:
            candidate_actions = preferred_candidates

    best_action = candidate_actions[0]
    best_score = -1.0
    for action in candidate_actions:
        try:
            projected_score = _projected_score_for_action(
                env=env,
                task_key=task_key,
                action=action,
                lookahead_depth=lookahead_depth,
                lookahead_width=lookahead_width,
            )
        except Exception:
            continue
        if projected_score > best_score:
            best_score = projected_score
            best_action = action

    return best_action.model_dump(mode="json")


def _pick_best_payload_by_projection(
    env: FlightRebookingEnv,
    task_key: str,
    payloads: List[Dict[str, Any]],
    lookahead_depth: int,
    lookahead_width: int,
) -> Dict[str, Any]:
    best_payload = payloads[0]
    best_score = -1.0

    seen_signatures = set()
    for payload in payloads:
        try:
            action = Action(**payload)
        except Exception:
            continue

        signature = (action.action_type.value, action.passenger_id, action.flight_id)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        try:
            projected_score = _projected_score_for_action(
                env=env,
                task_key=task_key,
                action=action,
                lookahead_depth=lookahead_depth,
                lookahead_width=lookahead_width,
            )
        except Exception:
            continue

        if projected_score > best_score:
            best_score = projected_score
            best_payload = action.model_dump(mode="json")

    return best_payload


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    inline = re.search(r"\{.*\}", text, re.DOTALL)
    if inline:
        return json.loads(inline.group(0))

    raise ValueError("No valid JSON action in model output")


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


def _heuristic_action(observation: Dict[str, Any]) -> Dict[str, Any]:
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


def _is_action_feasible(observation: Dict[str, Any], payload: Dict[str, Any]) -> bool:
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


def _sanitize_action_payload(observation: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    fallback = _heuristic_action(observation)

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

    if not _is_action_feasible(observation, sanitized):
        return fallback

    return sanitized


def _query_openai_action(
    client: OpenAI,
    model_name: str,
    seed: int,
    observation_json: str,
    policy_hint_json: Optional[str] = None,
    max_retries: int = 2,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None

    for _ in range(max_retries + 1):
        try:
            user_content = f"Current observation: {observation_json}"
            if policy_hint_json:
                user_content += (
                    "\nSuggested safe action from a trained policy: "
                    f"{policy_hint_json}"
                    "\nPrefer this if it is valid for the current observation."
                )

            kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 220,
                "seed": seed,
            }

            response = None
            try:
                response = client.chat.completions.create(**kwargs)
            except TypeError:
                kwargs.pop("seed", None)
                response = client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content or ""
            return _extract_json(content)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"OpenAI call failed after retries: {last_error}")


def _emit_start(task_name: str, benchmark: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={benchmark} model={model_name}", flush=True)


def _format_action_for_log(action: Action) -> str:
    payload = {
        "action_type": action.action_type.value,
        "passenger_id": action.passenger_id,
        "flight_id": action.flight_id,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _emit_step(
    step_index: int,
    action_text: str,
    reward_value: float,
    done: bool,
    error: Optional[str],
) -> None:
    done_value = str(bool(done)).lower()
    error_value = error if error else "null"
    print(
        "[STEP] "
        f"step={step_index} "
        f"action={action_text} "
        f"reward={reward_value:.2f} "
        f"done={done_value} "
        f"error={error_value}",
        flush=True,
    )


def _emit_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{value:.2f}" for value in rewards)
    success_value = str(bool(success)).lower()
    print(
        "[END] "
        f"success={success_value} "
        f"steps={steps} "
        f"score={score:.4f} "
        f"rewards={rewards_text}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run submission inference across OpenEnv tasks.")
    parser.add_argument("--task", choices=["all", "easy", "medium", "hard"], default="all")
    parser.add_argument("--seed", type=int, default=int(os.getenv("BASELINE_SEED", "42")))
    parser.add_argument(
        "--policy",
        choices=["openai", "heuristic", "trained_ml", "openai_trained"],
        default="openai_trained",
        help=(
            "Policy backend. openai_trained uses Llama with trained-policy hints; "
            "trained_ml uses the learned policy directly; openai and heuristic remain available."
        ),
    )
    parser.add_argument(
        "--ml-policy-path",
        default=os.getenv("ML_POLICY_PATH", "artifacts/ml_policy.pkl"),
        help="Path to trained ML policy artifact used by trained_ml/openai_trained modes.",
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=int(os.getenv("LOOKAHEAD_DEPTH", "2")),
        help="Lookahead depth for projected action scoring (>=1).",
    )
    parser.add_argument(
        "--lookahead-width",
        type=int,
        default=int(os.getenv("LOOKAHEAD_WIDTH", "12")),
        help="Maximum candidate actions explored per lookahead level (>=1).",
    )
    parser.add_argument("--json-out", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.lookahead_depth = max(1, int(args.lookahead_depth))
    args.lookahead_width = max(1, int(args.lookahead_width))

    task_keys = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    effective_policy = args.policy
    ml_policy_artifact: Optional[Dict[str, Any]] = None
    if effective_policy in {"trained_ml", "openai_trained"}:
        ml_policy_artifact = _require_ml_policy_artifact(args.ml_policy_path, effective_policy)

    api_base_url = "heuristic"
    model_name = "heuristic"
    client: Optional[OpenAI] = None
    if effective_policy in {"openai", "openai_trained"}:
        model_config = _resolve_model_config()
        api_base_url = model_config["api_base_url"]
        model_name = model_config["model_name"]
        client = OpenAI(api_key=model_config["api_key"], base_url=api_base_url)

    results: List[Dict[str, Any]] = []

    for task_key in task_keys:
        task_data = TASKS[task_key]
        _emit_start(task_name=task_data["task_id"], benchmark=BENCHMARK_NAME, model_name=model_name)

        env = FlightRebookingEnv(task_data=task_data)
        observation = None
        done = False
        steps = 0
        rewards: List[float] = []
        score = 0.01
        success = False
        episode_error: Optional[str] = None

        try:
            observation = env.reset()

            while not done:
                observation_dict = observation.model_dump(mode="json")

                if effective_policy in {"openai", "openai_trained"}:
                    assert client is not None
                    policy_hint_payload: Optional[Dict[str, Any]] = None
                    if effective_policy == "openai_trained":
                        assert ml_policy_artifact is not None
                        ranked_types = _predict_ml_ranked_action_types(observation_dict, ml_policy_artifact)
                        policy_hint_payload = _choose_lookahead_action(
                            env=env,
                            task_key=task_key,
                            lookahead_depth=args.lookahead_depth,
                            lookahead_width=args.lookahead_width,
                            ranked_action_types=ranked_types,
                        )

                    raw_payload = _query_openai_action(
                        client=client,
                        model_name=model_name,
                        seed=args.seed,
                        observation_json=observation.model_dump_json(),
                        policy_hint_json=(json.dumps(policy_hint_payload) if policy_hint_payload is not None else None),
                    )
                    llm_payload = _sanitize_action_payload(observation_dict, raw_payload)

                    if effective_policy == "openai_trained" and policy_hint_payload is not None:
                        action_payload = _pick_best_payload_by_projection(
                            env=env,
                            task_key=task_key,
                            payloads=[policy_hint_payload, llm_payload],
                            lookahead_depth=args.lookahead_depth,
                            lookahead_width=args.lookahead_width,
                        )
                    else:
                        action_payload = llm_payload
                elif effective_policy == "trained_ml":
                    assert ml_policy_artifact is not None
                    ranked_types = _predict_ml_ranked_action_types(observation_dict, ml_policy_artifact)
                    action_payload = _choose_lookahead_action(
                        env=env,
                        task_key=task_key,
                        lookahead_depth=args.lookahead_depth,
                        lookahead_width=args.lookahead_width,
                        ranked_action_types=ranked_types,
                    )
                else:
                    action_payload = _heuristic_action(observation_dict)

                try:
                    action = Action(**action_payload)
                except Exception:
                    action = Action(action_type=ActionType.FINALIZE)

                step_error: Optional[str] = None
                reward_value = 0.0
                try:
                    observation, reward, done, info = env.step(action)
                    reward_value = float(reward.value)
                    if isinstance(info, dict) and info.get("error"):
                        step_error = str(info.get("error"))
                except Exception as exc:
                    done = True
                    step_error = str(exc)
                    episode_error = step_error

                steps += 1
                rewards.append(reward_value)
                _emit_step(
                    step_index=steps,
                    action_text=_format_action_for_log(action),
                    reward_value=reward_value,
                    done=done,
                    error=step_error,
                )

            try:
                final_state = env.state()
                score = float(grade_task(task_key, final_state, task_data["max_budget"]))
            except Exception as exc:
                episode_error = str(exc)
                score = 0.01
        except Exception as exc:
            episode_error = str(exc)
            score = 0.01
        finally:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as exc:
                    if not episode_error:
                        episode_error = str(exc)

            success = (episode_error is None) and (0.0 <= score <= 1.0) and (score >= SUCCESS_SCORE_THRESHOLD)
            _emit_end(success=success, steps=steps, score=score, rewards=rewards)

            try:
                final_state = env.state()
                avg_step_reward = sum(rewards) / max(len(rewards), 1)
                results.append(
                    {
                        "task": task_key,
                        "task_id": task_data["task_id"],
                        "difficulty": task_data["difficulty"],
                        "steps": steps,
                        "avg_step_reward": round(avg_step_reward, 4),
                        "score": round(score, 4),
                        "budget_spent": round(final_state.budget_spent, 2),
                        "budget_max": task_data["max_budget"],
                        "invalid_actions": final_state.invalid_actions,
                        "success": success,
                        "error": episode_error,
                    }
                )
            except Exception:
                avg_step_reward = sum(rewards) / max(len(rewards), 1)
                results.append(
                    {
                        "task": task_key,
                        "task_id": task_data["task_id"],
                        "difficulty": task_data["difficulty"],
                        "steps": steps,
                        "avg_step_reward": round(avg_step_reward, 4),
                        "score": round(score, 4),
                        "budget_spent": None,
                        "budget_max": task_data["max_budget"],
                        "invalid_actions": None,
                        "success": success,
                        "error": episode_error,
                    }
                )

    overall = sum(item["score"] for item in results) / max(len(results), 1)

    if args.json_out:
        payload = {
            "policy_requested": args.policy,
            "policy_effective": effective_policy,
            "seed": args.seed,
            "api_base_url": api_base_url,
            "model_name": model_name,
            "ml_policy_path": args.ml_policy_path,
            "ml_policy_loaded": ml_policy_artifact is not None,
            "lookahead_depth": args.lookahead_depth,
            "lookahead_width": args.lookahead_width,
            "overall_score": round(overall, 4),
            "tasks": results,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
