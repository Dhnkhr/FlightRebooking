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

from environment import Action, ActionType, FlightRebookingEnv
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


def _rollout_expert_episode(task_data: Dict[str, Any], task_key: str, episode_index: int) -> Tuple[List[Dict[str, Any]], EpisodeSummary]:
    env = FlightRebookingEnv(task_data=task_data)
    observation = env.reset()
    done = False

    samples: List[Dict[str, Any]] = []
    while not done:
        observation_dict = observation.model_dump(mode="json")
        action_payload = heuristic_action(observation_dict)

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


def _collect_dataset(seed: int, episodes_per_task: int) -> Tuple[List[List[float]], List[str], List[EpisodeSummary]]:
    rng = random.Random(seed)
    X: List[List[float]] = []
    y: List[str] = []
    summaries: List[EpisodeSummary] = []

    for task_key, task_data in TASKS.items():
        for episode_idx in range(episodes_per_task):
            variant = _jitter_task(task_data, rng)
            samples, summary = _rollout_expert_episode(variant, task_key, episode_idx)
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
    parser.add_argument("--output", default="artifacts/ml_policy.pkl", help="Path to save trained policy artifact.")
    parser.add_argument("--report", default="artifacts/ml_policy_report.json", help="Path to save training/eval report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    print("[TRAIN] Collecting synthetic expert dataset...")
    X, y, episode_summaries = _collect_dataset(seed=args.seed, episodes_per_task=args.episodes_per_task)

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
        "artifact_version": "1.0",
        "seed": args.seed,
        "feature_length": feature_length,
        "action_type_order": ACTION_TYPE_ORDER,
        "model": model,
        "training_metadata": {
            "episodes_per_task": args.episodes_per_task,
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
