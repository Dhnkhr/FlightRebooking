"""
One-command autopilot for model training + evaluation.

Pipeline:
1) Train ML policy artifact from synthetic trajectories.
2) Run hybrid Llama inference (openai_trained) using the artifact as guidance.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_DIR = Path(__file__).resolve().parent


def _run_command(command: List[str], stage: str) -> None:
    print(f"[AUTOPILOT] {stage}: {' '.join(command)}")
    completed = subprocess.run(command, check=False, cwd=str(REPO_DIR))
    if completed.returncode != 0:
        raise SystemExit(f"{stage} failed with exit code {completed.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate flight-rebooking policy automatically.")
    parser.add_argument("--episodes-per-task", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=["all", "easy", "medium", "hard"], default="all")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and reuse existing ML artifact.")
    parser.add_argument("--train-only", action="store_true", help="Train artifact only; do not run inference.")
    parser.add_argument("--ml-policy-path", default="artifacts/ml_policy.pkl")
    parser.add_argument("--training-report", default="artifacts/ml_policy_report.json")
    parser.add_argument("--results-json", default="artifacts/inference_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.ml_policy_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.training_report) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)

    if not args.skip_train:
        train_cmd = [
            sys.executable,
            "train_ml_policy.py",
            "--episodes-per-task",
            str(args.episodes_per_task),
            "--seed",
            str(args.seed),
            "--output",
            args.ml_policy_path,
            "--report",
            args.training_report,
        ]
        _run_command(train_cmd, "training")

    if args.train_only:
        print("[AUTOPILOT] Training complete.")
        return

    inference_cmd = [
        sys.executable,
        "inference.py",
        "--policy",
        "openai_trained",
        "--task",
        args.task,
        "--seed",
        str(args.seed),
        "--ml-policy-path",
        args.ml_policy_path,
        "--json-out",
        args.results_json,
    ]
    _run_command(inference_cmd, "inference")

    if os.path.exists(args.results_json):
        with open(args.results_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        overall = payload.get("overall_score")
        print(f"[AUTOPILOT] Completed. overall_score={overall}")
        for item in payload.get("tasks", []):
            print(
                "[AUTOPILOT] "
                f"task={item.get('task')} score={item.get('score')} "
                f"steps={item.get('steps')} invalid_actions={item.get('invalid_actions')}"
            )


if __name__ == "__main__":
    main()
