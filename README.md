---
title: Flight Rebooking OpenEnv
sdk: docker
app_port: 7860
tags:
  - openenv
  - simulation
  - logistics
  - reinforcement-learning
---

# Flight Rebooking OpenEnv

A real-world operations environment where an agent plays the role of an airline disruption desk during storm cancellations.

The agent must resolve stranded passengers under real constraints:

- loyalty-tier SLAs (Platinum > Gold > Silver > Standard)
- limited seat inventory across flights
- connection deadlines
- compensation and partner-airline budget limits

This environment is designed for training and evaluating agent decision quality through the standard OpenEnv interface:

- `reset() -> Observation`
- `step(Action) -> (Observation, Reward, done, info)`
- `state() -> EnvState`

## Why This Is Real-World

Airline IROPS (irregular operations) recovery is a genuine operational task solved daily by human teams and optimization systems. The environment captures realistic trade-offs:

- service quality versus cost control
- urgency versus fairness
- strict inventory and budget constraints

## OpenEnv Compliance

The project includes typed Pydantic models for:

- `Action` in [environment.py](environment.py)
- `Observation` in [environment.py](environment.py)
- `Reward` in [environment.py](environment.py)

Metadata is defined in [openenv.yaml](openenv.yaml), including:

- environment entrypoint
- model mappings
- API surface (`reset`, `step`, `state`)
- task-to-grader mapping

## Action Space

`Action` schema:

- `action_type`:
  - `rebook_passenger`
  - `offer_downgrade`
  - `book_hotel`
  - `rebook_on_partner`
  - `mark_no_solution`
  - `finalize`
- `passenger_id` (optional string)
- `flight_id` (optional string)

## Observation Space

`Observation` fields:

- `pending_passengers`: unresolved passengers with tier, cabin, and deadline
- `available_flights`: seat inventory, departure time, and partner flag
- `budget_remaining`, `budget_spent`
- `processed_count`, `total_passengers`
- `invalid_actions`
- `step_count`

## Reward Design

`Reward.value` is always normalized to `[0.0, 1.0]` and includes shaped partial progress.

Reward components include:

- progress ratio (how much of the manifest is resolved)
- resolution quality (rebooked > partner > downgraded > hotel > no-solution)
- loyalty priority handling
- connection deadline success
- budget efficiency
- penalties for invalid actions, repeated failed behavior, and priority inversion

This gives dense trajectory feedback instead of only terminal pass/fail.

## Tasks and Agent Graders

Three deterministic tasks are provided in [tasks.py](tasks.py), each with a dedicated grader returning `0.0-1.0`:

- Easy: `easy_minor_disruption`
  - Grader: `grade_easy_episode`
  - Focus: complete rebooking with low cost
- Medium: `medium_connection_crisis`
  - Grader: `grade_medium_episode`
  - Focus: urgent connection handling with constrained seats
- Hard: `hard_multi_wave_disruption`
  - Grader: `grade_hard_episode`
  - Focus: multi-passenger triage under severe scarcity

## Baseline Inference

Submission runner: [inference.py](inference.py)

Features:

- OpenAI API client integration
- internal Groq defaults (`https://api.groq.com/openai/v1`, `llama-3.1-8b-instant`) for zero-setup runs
- optional env var overrides via `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`/`OPENAI_API_KEY`/`GROQ_API_KEY`
- optional trained ML policy artifact (`artifacts/ml_policy.pkl`) for guided and standalone policy execution
- deterministic settings (`temperature=0`, fixed `seed`)
- evaluates all 3 tasks and prints normalized scores
- emits structured stdout logs with `[START]`, `[STEP]`, `[END]`

Optional environment variable overrides:

```bash
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.1-8b-instant
set GROQ_API_KEY=your_provider_token
```

## Submission Contract (Mandatory)

Before submitting, define these environment variables in your runtime configuration:

- `API_BASE_URL`: LLM API endpoint
- `MODEL_NAME`: model identifier used for inference
- `HF_TOKEN`: Hugging Face/API key used by OpenAI client
- `LOCAL_IMAGE_NAME`: only needed when your env is started via `from_docker_image()`; not required by this repo's `inference.py`

The inference entrypoint must be `inference.py` in project root, and LLM calls must use the OpenAI client.

The script emits strict stdout lines in this contract format:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

Formatting rules enforced by evaluator:

- One `[START]` line at task begin.
- One `[STEP]` line immediately after each `env.step(...)` return.
- One `[END]` line always emitted, including exception paths.
- `reward` and `rewards` use 2 decimal places.
- `done` and `success` are lowercase booleans (`true`/`false`).
- `error` is raw error text or `null`.
- Each task score must be in `[0, 1]`.

Run validator before final submission:

```bash
python pre_submission_validate.py
```

If Docker is unavailable locally, use:

```bash
python pre_submission_validate.py --skip-docker
```

Run with OpenAI:

```bash
python inference.py --policy openai --seed 42 --task all
```

Run with trained-policy guidance + lookahead optimization (recommended):

```bash
python inference.py --policy openai_trained --seed 42 --task all --ml-policy-path artifacts/ml_policy.pkl --lookahead-depth 2 --lookahead-width 12
```

Run trained ML policy directly (no API calls):

```bash
python inference.py --policy trained_ml --seed 42 --task all --ml-policy-path artifacts/ml_policy.pkl --lookahead-depth 2 --lookahead-width 12
```

In both `openai_trained` and `trained_ml` modes, the runtime uses depth-aware beam lookahead for projected action scoring. Current defaults are `--lookahead-depth 2` and `--lookahead-width 12`.

Deterministic offline fallback:

```bash
python inference.py --policy heuristic --seed 42 --task all
```

## ML Training Pipeline

Train a reusable policy artifact from large synthetic trajectories:

```bash
python train_ml_policy.py --episodes-per-task 450 --seed 42 --teacher-policy lookahead --teacher-lookahead-depth 2 --teacher-lookahead-width 8 --output artifacts/ml_policy.pkl --report artifacts/ml_policy_report.json
```

The report JSON includes validation accuracy and canonical task scores for the learned policy.

For `trained_ml` and `openai_trained` inference modes, `inference.py` now fails fast if the artifact path is missing, invalid, or still a Git LFS pointer file.

One-command autopilot (train + run hybrid inference):

```bash
python autopilot.py --episodes-per-task 450 --seed 42 --teacher-policy lookahead --teacher-lookahead-depth 2 --teacher-lookahead-width 8 --task all
```

## Baseline Scores

Reference deterministic baseline (heuristic policy, seed 42):

- Easy: `1.0000`
- Medium: `0.9768`
- Hard: `0.9609`
- Overall average: `0.9792`

The OpenAI baseline is reproducible given fixed model + seed + prompt, though exact values may vary across model versions/providers.

## Local Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run the Space app locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Then open:

- `http://localhost:7860/`
- `http://localhost:7860/docs`
- `http://localhost:7860/ui` (interactive frontend dashboard)

## Frontend Dashboard

The project includes a responsive UI in [frontend/index.html](frontend/index.html) that connects directly to the API.

Dashboard features:

- task selection and one-click reset
- live pending-passenger and flight views
- manual action console for `step()`
- auto-step helper and finalize control
- trajectory log with reward and done state per step

Frontend static assets are served by FastAPI at:

- `GET /ui`
- `GET /ui/static/style.css`
- `GET /ui/static/app.js`

## Docker

Build and run:

```bash
docker build -t flight-rebooking-openenv .
docker run --rm -p 7860:7860 flight-rebooking-openenv
```

## Hugging Face Spaces Deployment

This repo is ready for a Docker Space.

1. Create a new Hugging Face Space with SDK set to Docker.
2. Push this repository content.
3. Ensure `README.md` frontmatter includes `sdk: docker` and `app_port: 7860`.
4. Optionally add `OPENAI_API_KEY` as a Space secret for live model baselines.

## Validation

If you have the OpenEnv CLI installed:

```bash
openenv validate
```

If `openenv` is not installed in your environment, install the CLI first and rerun validation.

Run the full pre-submission checklist validator:

```bash
python pre_submission_validate.py
```

Optional flags:

```bash
python pre_submission_validate.py --skip-docker
python pre_submission_validate.py --docker-timeout 1200 --inference-timeout 1200
```
