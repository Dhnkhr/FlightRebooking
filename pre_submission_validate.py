"""
Pre-submission validator for Flight Rebooking OpenEnv.

Runs checks aligned with the submission checklist:
- Env vars
- OpenEnv spec shape
- Inference script execution
- Task/grader score ranges
- HF Space ping + reset (if SPACE_URL is set)
- Docker build
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple

import yaml

from environment import Action, ActionType, EnvState, FlightRebookingEnv, Observation, Reward
from tasks import TASKS, grade_task


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def _pass(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="PASS", detail=detail)


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="FAIL", detail=detail)


def _warn(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="WARN", detail=detail)


def check_required_env_vars() -> CheckResult:
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    missing = [key for key in required if not os.getenv(key, "").strip()]
    if missing:
        return _fail("required_env_vars", f"Missing variables: {', '.join(missing)}")
    return _pass("required_env_vars", "API_BASE_URL, MODEL_NAME, HF_TOKEN are set.")


def check_openenv_yaml() -> CheckResult:
    path = "openenv.yaml"
    if not os.path.exists(path):
        return _fail("openenv_yaml", "openenv.yaml not found.")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = yaml.safe_load(handle)
    except Exception as exc:
        return _fail("openenv_yaml", f"Failed to parse openenv.yaml: {exc}")

    required_keys = ["name", "version", "entrypoint", "models", "api", "tasks"]
    missing = [k for k in required_keys if k not in doc]
    if missing:
        return _fail("openenv_yaml", f"Missing keys: {', '.join(missing)}")

    if len(doc.get("tasks", [])) < 3:
        return _fail("openenv_yaml", "openenv.yaml must enumerate at least 3 tasks.")

    return _pass("openenv_yaml", "openenv.yaml parsed and required keys are present.")


def check_openenv_interface() -> CheckResult:
    try:
        env = FlightRebookingEnv(task_data=TASKS["easy"])
        observation = env.reset()
        if not isinstance(observation, Observation):
            return _fail("openenv_interface", "reset() did not return Observation model.")

        step_output = env.step(Action(action_type=ActionType.FINALIZE))
        if not isinstance(step_output, tuple) or len(step_output) != 4:
            return _fail("openenv_interface", "step() did not return 4-tuple.")

        obs, reward, done, info = step_output
        if not isinstance(obs, Observation):
            return _fail("openenv_interface", "step() observation is not Observation model.")
        if not isinstance(reward, Reward):
            return _fail("openenv_interface", "step() reward is not Reward model.")
        if not isinstance(done, bool):
            return _fail("openenv_interface", "step() done is not bool.")
        if not isinstance(info, dict):
            return _fail("openenv_interface", "step() info is not dict.")

        state = env.state()
        if not isinstance(state, EnvState):
            return _fail("openenv_interface", "state() did not return EnvState model.")

    except Exception as exc:
        return _fail("openenv_interface", f"Interface check failed: {exc}")

    return _pass("openenv_interface", "Typed models and step/reset/state interface are valid.")


def _heuristic_action(observation: Dict) -> Dict:
    pending = list(observation["pending_passengers"])
    if not pending:
        return {"action_type": "finalize"}

    tier_weight = {"Platinum": 4, "Gold": 3, "Silver": 2, "Standard": 1}
    pending.sort(
        key=lambda p: (
            -tier_weight.get(p["priority_tier"], 1),
            p["connection_deadline_hrs"] if p["connection_deadline_hrs"] is not None else 10**9,
        )
    )
    p = pending[0]
    flights = sorted(observation["available_flights"], key=lambda f: f["departure_hrs"])

    def has_seat(flight: Dict) -> bool:
        if p["cabin_class"] == "Business":
            return flight["business_seats"] > 0
        return flight["economy_seats"] > 0

    for flight in flights:
        if not flight["is_partner"] and has_seat(flight):
            return {"action_type": "rebook_passenger", "passenger_id": p["id"], "flight_id": flight["id"]}

    if p["cabin_class"] == "Business":
        for flight in flights:
            if not flight["is_partner"] and flight["economy_seats"] > 0 and observation["budget_remaining"] >= 500:
                return {"action_type": "offer_downgrade", "passenger_id": p["id"], "flight_id": flight["id"]}

    for flight in flights:
        if flight["is_partner"] and has_seat(flight) and observation["budget_remaining"] >= 800:
            return {"action_type": "rebook_on_partner", "passenger_id": p["id"], "flight_id": flight["id"]}

    if observation["budget_remaining"] >= 250:
        return {"action_type": "book_hotel", "passenger_id": p["id"]}

    return {"action_type": "mark_no_solution", "passenger_id": p["id"]}


def check_tasks_and_graders() -> CheckResult:
    details: List[str] = []
    try:
        for task_key, task_data in TASKS.items():
            env = FlightRebookingEnv(task_data=task_data)
            obs = env.reset()
            done = False
            rewards: List[float] = []

            while not done:
                action_payload = _heuristic_action(obs.model_dump(mode="json"))
                action = Action(**action_payload)
                obs, reward, done, _ = env.step(action)
                rewards.append(reward.value)

            score = grade_task(task_key, env.state(), task_data["max_budget"])
            if not (0.0 <= score <= 1.0):
                return _fail("tasks_and_graders", f"Task {task_key} score out of range: {score}")

            for value in rewards:
                if not (0.0 <= value <= 1.0):
                    return _fail("tasks_and_graders", f"Task {task_key} reward out of range: {value}")

            details.append(f"{task_key}={score:.4f}")

    except Exception as exc:
        return _fail("tasks_and_graders", f"Execution failed: {exc}")

    return _pass("tasks_and_graders", "Validated tasks and grader ranges: " + ", ".join(details))


def check_inference_script(timeout_sec: int) -> CheckResult:
    if not os.path.exists("inference.py"):
        return _fail("inference_script", "inference.py not found at repository root.")

    env = dict(os.environ)
    env.setdefault("API_BASE_URL", "https://api.together.xyz/v1")
    env.setdefault("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    env.setdefault("HF_TOKEN", "placeholder-token")

    cmd = [sys.executable, "inference.py", "--policy", "heuristic", "--task", "all", "--seed", "42"]
    started = time.time()

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired:
        return _fail("inference_script", f"Timed out after {timeout_sec} seconds.")

    runtime = time.time() - started
    stdout = proc.stdout.strip().splitlines()

    if proc.returncode != 0:
        return _fail("inference_script", f"Exit code {proc.returncode}. stderr={proc.stderr[:400]}")

    has_start = any(line.startswith("[START]") for line in stdout)
    has_step = any(line.startswith("[STEP]") for line in stdout)
    has_end = any(line.startswith("[END]") for line in stdout)
    if not (has_start and has_step and has_end):
        return _fail("inference_script", "Missing required structured logs [START]/[STEP]/[END].")

    if runtime > 20 * 60:
        return _fail("inference_script", f"Runtime {runtime:.1f}s exceeds 20-minute limit.")

    return _pass("inference_script", f"Completed in {runtime:.1f}s with required structured logs.")


def check_space_ping() -> CheckResult:
    space_url = os.getenv("SPACE_URL", "").strip().rstrip("/")
    if not space_url:
        return _warn("hf_space_ping", "SPACE_URL not set. Skipping remote ping/reset checks.")

    root_url = f"{space_url}/"
    reset_url = f"{space_url}/reset"

    try:
        req = urllib.request.Request(root_url, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                return _fail("hf_space_ping", f"Root ping returned status {resp.status}")
    except urllib.error.URLError as exc:
        return _fail("hf_space_ping", f"Root ping failed: {exc}")

    payload = json.dumps({"task": "easy"}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    try:
        req = urllib.request.Request(reset_url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                return _fail("hf_space_ping", f"/reset returned status {resp.status}")
            body = json.loads(resp.read().decode("utf-8"))
            if "observation" not in body:
                return _fail("hf_space_ping", "reset response missing observation")
    except urllib.error.URLError as exc:
        return _fail("hf_space_ping", f"reset call failed: {exc}")

    return _pass("hf_space_ping", "Space URL responded with 200 and /reset returned observation.")


def check_docker_build(timeout_sec: int, skip_docker: bool) -> CheckResult:
    if skip_docker:
        return _warn("docker_build", "Skipped by --skip-docker.")

    if shutil.which("docker") is None:
        return _fail("docker_build", "docker executable is not available.")

    cmd = ["docker", "build", "-t", "flight-rebooking-openenv-validate", "."]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired:
        return _fail("docker_build", f"docker build timed out after {timeout_sec} seconds.")

    if proc.returncode != 0:
        excerpt = (proc.stderr or proc.stdout)[-500:]
        return _fail("docker_build", f"docker build failed. tail={excerpt}")

    return _pass("docker_build", "docker build completed successfully.")


def check_openenv_cli() -> CheckResult:
    if shutil.which("openenv") is None:
        return _warn("openenv_cli", "openenv CLI not found in PATH; skipped `openenv validate`.")

    proc = subprocess.run(["openenv", "validate"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        excerpt = (proc.stderr or proc.stdout)[-500:]
        return _fail("openenv_cli", f"openenv validate failed. tail={excerpt}")

    return _pass("openenv_cli", "openenv validate passed.")


def print_results(results: List[CheckResult]) -> int:
    print("\nPre-Submission Validation Report")
    print("=" * 40)
    for item in results:
        print(f"[{item.status}] {item.name}: {item.detail}")

    failed = [item for item in results if item.status == "FAIL"]
    warnings = [item for item in results if item.status == "WARN"]

    print("=" * 40)
    print(f"PASS={len([r for r in results if r.status == 'PASS'])} WARN={len(warnings)} FAIL={len(failed)}")

    if failed:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-submission checks.")
    parser.add_argument("--skip-docker", action="store_true", help="Skip docker build check.")
    parser.add_argument("--docker-timeout", type=int, default=900, help="Docker build timeout seconds.")
    parser.add_argument("--inference-timeout", type=int, default=1200, help="Inference timeout seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = [
        check_required_env_vars(),
        check_openenv_yaml(),
        check_openenv_interface(),
        check_tasks_and_graders(),
        check_inference_script(timeout_sec=args.inference_timeout),
        check_space_ping(),
        check_openenv_cli(),
        check_docker_build(timeout_sec=args.docker_timeout, skip_docker=args.skip_docker),
    ]

    code = print_results(results)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
