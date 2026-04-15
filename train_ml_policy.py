"""
Train an internal ML policy for flight rebooking.

The trainer generates synthetic trajectories with controlled scenario jitter,
learns action-type selection from expert demonstrations, and exports a
pickled policy artifact used by inference.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from environment import Action, ActionType, CabinClass, FlightRebookingEnv
from ml_policy import ACTION_TYPE_ORDER, choose_action_from_ranked_types, heuristic_action, observation_to_features
from tasks import TASKS, grade_task


@dataclass
class EpisodeSummary:
    task_key: str
    episode_index: int
    sample_count: int
    final_score: float


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _jitter_task(task_data: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
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


def _tier_weight(tier: str) -> int:
    return {
        "Platinum": 4,
        "Gold": 3,
        "Silver": 2,
        "Standard": 1,
    }.get(tier, 1)


def _has_seat(flight: Dict[str, Any], cabin_class: str) -> bool:
    if cabin_class == CabinClass.BUSINESS.value:
        return int(flight.get("business_seats", 0)) > 0
    return int(flight.get("economy_seats", 0)) > 0


def _feasible_actions_from_observation(observation: Dict[str, Any]) -> List[Action]:
    pending = list(observation.get("pending_passengers", []))
    flights = sorted(
        list(observation.get("available_flights", [])),
        key=lambda flight: float(flight.get("departure_hrs", 999.0)),
    )
    budget_remaining = float(observation.get("budget_remaining", 0.0))

    if not pending:
        return [Action(action_type=ActionType.FINALIZE)]

    actions: List[Action] = []
    for passenger in pending:
        passenger_id = str(passenger.get("id", ""))
        passenger_cabin = str(passenger.get("cabin_class", ""))

        for flight in flights:
            flight_id = str(flight.get("id", ""))

            if (not flight.get("is_partner", False)) and _has_seat(flight, passenger_cabin):
                actions.append(
                    Action(
                        action_type=ActionType.REBOOK_PASSENGER,
                        passenger_id=passenger_id,
                        flight_id=flight_id,
                    )
                )

            if (
                passenger_cabin == CabinClass.BUSINESS.value
                and (not flight.get("is_partner", False))
                and int(flight.get("economy_seats", 0)) > 0
                and budget_remaining >= 500.0
            ):
                actions.append(
                    Action(
                        action_type=ActionType.OFFER_DOWNGRADE,
                        passenger_id=passenger_id,
                        flight_id=flight_id,
                    )
                )

            if (
                flight.get("is_partner", False)
                and _has_seat(flight, passenger_cabin)
                and budget_remaining >= 800.0
            ):
                actions.append(
                    Action(
                        action_type=ActionType.REBOOK_ON_PARTNER,
                        passenger_id=passenger_id,
                        flight_id=flight_id,
                    )
                )

        if budget_remaining >= 250.0:
            actions.append(
                Action(
                    action_type=ActionType.BOOK_HOTEL,
                    passenger_id=passenger_id,
                )
            )

        actions.append(
            Action(
                action_type=ActionType.MARK_NO_SOLUTION,
                passenger_id=passenger_id,
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
    }[action_type]


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

    score += {
        ActionType.REBOOK_PASSENGER: 0.60,
        ActionType.OFFER_DOWNGRADE: 0.30,
        ActionType.REBOOK_ON_PARTNER: 0.18,
        ActionType.BOOK_HOTEL: 0.10,
        ActionType.MARK_NO_SOLUTION: -0.60,
        ActionType.FINALIZE: 0.0,
    }[action.action_type]

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


def _prune_candidate_actions(observation: Dict[str, Any], actions: List[Action], max_candidates: int) -> List[Action]:
    deduped: List[Action] = []
    seen = set()
    for action in actions:
        signature = (action.action_type.value, action.passenger_id, action.flight_id)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(action)

    deduped.sort(key=lambda action: -_action_priority_score(observation, action))
    return deduped[: max(1, max_candidates)]


def _rollout_heuristic_to_end(env: FlightRebookingEnv) -> None:
    done = False
    while not done:
        observation = env._get_observation().model_dump(mode="json")
        action = Action(**heuristic_action(observation))
        _, _, done, _ = env.step(action)


def _evaluate_state_with_lookahead(
    env: FlightRebookingEnv,
    task_key: str,
    max_budget: float,
    lookahead_depth: int,
    lookahead_width: int,
) -> float:
    observation = env._get_observation().model_dump(mode="json")
    candidate_actions = _prune_candidate_actions(
        observation=observation,
        actions=_feasible_actions_from_observation(observation),
        max_candidates=lookahead_width,
    )

    best_score = -1.0
    for action in candidate_actions:
        env_copy = copy.deepcopy(env)
        _, _, done, _ = env_copy.step(action)

        if done:
            score = float(grade_task(task_key, env_copy.state(), max_budget))
        elif lookahead_depth <= 1:
            _rollout_heuristic_to_end(env_copy)
            score = float(grade_task(task_key, env_copy.state(), max_budget))
        else:
            score = _evaluate_state_with_lookahead(
                env=env_copy,
                task_key=task_key,
                max_budget=max_budget,
                lookahead_depth=lookahead_depth - 1,
                lookahead_width=lookahead_width,
            )

        if score > best_score:
            best_score = score

    if best_score >= 0.0:
        return best_score

    env_fallback = copy.deepcopy(env)
    _rollout_heuristic_to_end(env_fallback)
    return float(grade_task(task_key, env_fallback.state(), max_budget))


def _projected_score_for_action(
    env: FlightRebookingEnv,
    task_key: str,
    max_budget: float,
    action: Action,
    lookahead_depth: int,
    lookahead_width: int,
) -> float:
    env_copy = copy.deepcopy(env)
    _, _, done, _ = env_copy.step(action)
    if done:
        return float(grade_task(task_key, env_copy.state(), max_budget))

    if lookahead_depth <= 1:
        _rollout_heuristic_to_end(env_copy)
        return float(grade_task(task_key, env_copy.state(), max_budget))

    return _evaluate_state_with_lookahead(
        env=env_copy,
        task_key=task_key,
        max_budget=max_budget,
        lookahead_depth=lookahead_depth - 1,
        lookahead_width=lookahead_width,
    )


def _choose_lookahead_teacher_action(
    env: FlightRebookingEnv,
    task_key: str,
    max_budget: float,
    lookahead_depth: int,
    lookahead_width: int,
) -> Dict[str, Any]:
    observation = env._get_observation().model_dump(mode="json")
    candidate_actions = _prune_candidate_actions(
        observation=observation,
        actions=_feasible_actions_from_observation(observation),
        max_candidates=lookahead_width,
    )

    best_action = candidate_actions[0]
    best_score = -1.0
    for action in candidate_actions:
        try:
            projected_score = _projected_score_for_action(
                env=env,
                task_key=task_key,
                max_budget=max_budget,
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


def _choose_teacher_action(
    env: FlightRebookingEnv,
    task_key: str,
    max_budget: float,
    observation_dict: Dict[str, Any],
    teacher_policy: str,
    teacher_lookahead_depth: int,
    teacher_lookahead_width: int,
) -> Dict[str, Any]:
    if teacher_policy == "heuristic":
        return heuristic_action(observation_dict)

    try:
        return _choose_lookahead_teacher_action(
            env=env,
            task_key=task_key,
            max_budget=max_budget,
            lookahead_depth=teacher_lookahead_depth,
            lookahead_width=teacher_lookahead_width,
        )
    except Exception:
        return heuristic_action(observation_dict)


def _rollout_expert_episode(
    task_data: Dict[str, Any],
    task_key: str,
    episode_index: int,
    teacher_policy: str,
    teacher_lookahead_depth: int,
    teacher_lookahead_width: int,
) -> Tuple[List[Dict[str, Any]], EpisodeSummary]:
    env = FlightRebookingEnv(task_data=task_data)
    observation = env.reset()
    done = False

    samples: List[Dict[str, Any]] = []
    while not done:
        observation_dict = observation.model_dump(mode="json")
        action_payload = _choose_teacher_action(
            env=env,
            task_key=task_key,
            max_budget=float(task_data["max_budget"]),
            observation_dict=observation_dict,
            teacher_policy=teacher_policy,
            teacher_lookahead_depth=teacher_lookahead_depth,
            teacher_lookahead_width=teacher_lookahead_width,
        )

        samples.append(
            {
                "task_key": task_key,
                "features": observation_to_features(observation_dict),
                "label": action_payload["action_type"],
            }
        )

        action = Action(**action_payload)
        observation, _, done, _ = env.step(action)

    score = grade_task(task_key, env.state(), task_data["max_budget"])
    summary = EpisodeSummary(
        task_key=task_key,
        episode_index=episode_index,
        sample_count=len(samples),
        final_score=score,
    )
    return samples, summary


def _collect_dataset(
    seed: int,
    episodes_per_task: int,
    teacher_policy: str,
    teacher_lookahead_depth: int,
    teacher_lookahead_width: int,
) -> Tuple[List[List[float]], List[str], List[EpisodeSummary]]:
    rng = random.Random(seed)
    X: List[List[float]] = []
    y: List[str] = []
    summaries: List[EpisodeSummary] = []

    for task_key, task_data in TASKS.items():
        for episode_idx in range(episodes_per_task):
            variant = _jitter_task(task_data, rng)
            samples, summary = _rollout_expert_episode(
                task_data=variant,
                task_key=task_key,
                episode_index=episode_idx,
                teacher_policy=teacher_policy,
                teacher_lookahead_depth=teacher_lookahead_depth,
                teacher_lookahead_width=teacher_lookahead_width,
            )
            summaries.append(summary)
            for sample in samples:
                X.append(sample["features"])
                y.append(sample["label"])

            if (episode_idx + 1) % 100 == 0:
                print(
                    f"[DATA] task={task_key} episodes={episode_idx + 1}/{episodes_per_task} "
                    f"samples={len(X)} avg_episode_score={mean(s.final_score for s in summaries if s.task_key == task_key):.4f}"
                )

    return X, y, summaries


def _rank_action_types(model: RandomForestClassifier, features: List[float]) -> List[str]:
    probabilities = model.predict_proba([features])[0]
    classes = [str(cls) for cls in model.classes_]
    ranked = [label for _, label in sorted(zip(probabilities, classes), key=lambda item: item[0], reverse=True)]

    # Preserve stable action coverage even if a class was absent in training split.
    for action_type in ACTION_TYPE_ORDER:
        if action_type not in ranked:
            ranked.append(action_type)
    return ranked


def _evaluate_learned_policy(model: RandomForestClassifier) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    for task_key, task_data in TASKS.items():
        env = FlightRebookingEnv(task_data=copy.deepcopy(task_data))
        observation = env.reset()
        done = False

        while not done:
            observation_dict = observation.model_dump(mode="json")
            features = observation_to_features(observation_dict)
            ranked_types = _rank_action_types(model, features)
            action_payload = choose_action_from_ranked_types(observation_dict, ranked_types)
            action = Action(**action_payload)
            observation, _, done, _ = env.step(action)

        score = grade_task(task_key, env.state(), task_data["max_budget"])
        scores[task_key] = round(score, 4)

    overall = sum(scores.values()) / max(len(scores), 1)
    scores["overall"] = round(overall, 4)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML policy for flight rebooking.")
    parser.add_argument("--episodes-per-task", type=int, default=450, help="Synthetic expert episodes generated per task.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--teacher-policy",
        choices=["heuristic", "lookahead"],
        default="lookahead",
        help="Expert policy used to label the training dataset.",
    )
    parser.add_argument(
        "--teacher-lookahead-depth",
        type=int,
        default=2,
        help="Lookahead depth used when --teacher-policy lookahead is enabled.",
    )
    parser.add_argument(
        "--teacher-lookahead-width",
        type=int,
        default=8,
        help="Candidate branching width used when --teacher-policy lookahead is enabled.",
    )
    parser.add_argument("--output", default="artifacts/ml_policy.pkl", help="Path to save trained policy artifact.")
    parser.add_argument("--report", default="artifacts/ml_policy_report.json", help="Path to save training/eval report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.teacher_lookahead_depth = max(1, int(args.teacher_lookahead_depth))
    args.teacher_lookahead_width = max(1, int(args.teacher_lookahead_width))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    print(
        "[TRAIN] Collecting synthetic expert dataset... "
        f"teacher={args.teacher_policy} "
        f"depth={args.teacher_lookahead_depth} width={args.teacher_lookahead_width}"
    )
    X, y, episode_summaries = _collect_dataset(
        seed=args.seed,
        episodes_per_task=args.episodes_per_task,
        teacher_policy=args.teacher_policy,
        teacher_lookahead_depth=args.teacher_lookahead_depth,
        teacher_lookahead_width=args.teacher_lookahead_width,
    )

    if not X:
        raise SystemExit("No training data generated.")

    feature_length = len(X[0])
    print(f"[TRAIN] samples={len(X)} features={feature_length} classes={sorted(set(y))}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=args.seed,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=700,
        max_depth=28,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )

    print("[TRAIN] Fitting classifier...")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = float(accuracy_score(y_train, train_pred))
    val_acc = float(accuracy_score(y_val, val_pred))
    cls_report = classification_report(y_val, val_pred, output_dict=True, zero_division=0)

    learned_policy_scores = _evaluate_learned_policy(model)

    task_episode_means: Dict[str, float] = {}
    for task_key in TASKS:
        task_scores = [s.final_score for s in episode_summaries if s.task_key == task_key]
        task_episode_means[task_key] = round(mean(task_scores), 4)

    artifact = {
        "artifact_version": "1.1",
        "seed": args.seed,
        "feature_length": feature_length,
        "action_type_order": ACTION_TYPE_ORDER,
        "model": model,
        "training_metadata": {
            "episodes_per_task": args.episodes_per_task,
            "teacher_policy": args.teacher_policy,
            "teacher_lookahead_depth": args.teacher_lookahead_depth,
            "teacher_lookahead_width": args.teacher_lookahead_width,
            "sample_count": len(X),
            "task_episode_mean_scores": task_episode_means,
            "train_accuracy": round(train_acc, 6),
            "val_accuracy": round(val_acc, 6),
        },
    }

    with open(args.output, "wb") as handle:
        pickle.dump(artifact, handle)

    report = {
        "seed": args.seed,
        "output_artifact": args.output,
        "dataset": {
            "samples": len(X),
            "feature_length": feature_length,
            "episodes_per_task": args.episodes_per_task,
            "teacher_policy": args.teacher_policy,
            "teacher_lookahead_depth": args.teacher_lookahead_depth,
            "teacher_lookahead_width": args.teacher_lookahead_width,
            "task_episode_mean_scores": task_episode_means,
        },
        "metrics": {
            "train_accuracy": round(train_acc, 6),
            "val_accuracy": round(val_acc, 6),
            "classification_report": cls_report,
        },
        "canonical_task_scores": learned_policy_scores,
    }

    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"[TRAIN] Saved model artifact: {args.output}")
    print(f"[TRAIN] Saved report: {args.report}")
    print(
        "[TRAIN] Canonical scores: "
        f"easy={learned_policy_scores['easy']:.4f} "
        f"medium={learned_policy_scores['medium']:.4f} "
        f"hard={learned_policy_scores['hard']:.4f} "
        f"overall={learned_policy_scores['overall']:.4f}"
    )


if __name__ == "__main__":
    main()
