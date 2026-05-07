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

# Flight Rebooking OpenEnv (Hackathon Submission)

This repository is a hackathon-ready agent + simulator for airline disruption (IROPS) rebooking.

What’s here:

- OpenEnv-compatible environment + typed Pydantic models: [environment.py](environment.py)
- Deterministic tasks + graders (score in `[0, 1]`): [tasks.py](tasks.py), [openenv.yaml](openenv.yaml)
- Hackathon submission runner with strict stdout contract: [inference.py](inference.py)
- Pre-submission checklist validator: [pre_submission_validate.py](pre_submission_validate.py)
- Optional local LoRA adapter bundle (GPU): [flight-rebooking-lora/](flight-rebooking-lora/)
- Optional UI for demo/manual play: [app.py](app.py), [frontend/index.html](frontend/index.html)

## Hackathon: What You Submit

- Entry point must be [inference.py](inference.py) at repo root.
- LLM calls must use the OpenAI-compatible client.
- The evaluator expects strict stdout logs:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

Required env vars (checked by [pre_submission_validate.py](pre_submission_validate.py)):

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` (or use `GROQ_API_KEY` / `OPENAI_API_KEY`; the runner accepts any of these)

## Quickstart (Inference)

Runtime-only install (recommended for evaluation / CI):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.runtime.txt
```

Set model provider (example: Groq):

```bash
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.1-8b-instant
set GROQ_API_KEY=your_token
```

Mac/Linux equivalents:

```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-8b-instant
export GROQ_API_KEY=your_token
```

Run all tasks:

```bash
python inference.py --policy openai --seed 42 --task all
```

No-API deterministic baseline:

```bash
python inference.py --policy heuristic --seed 42 --task all
```

## Validation (Required Before Submitting)

```bash
python pre_submission_validate.py
```

If Docker is unavailable:

```bash
python pre_submission_validate.py --skip-docker
```

## Optional: UI Demo (FastAPI + Dashboard)

Full install (heavier dependencies):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run server:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Open:

- `http://localhost:7860/ui`
- `http://localhost:7860/docs`

To enable Auto-Play without a GPU in Spaces/local CPU mode, set `GROQ_API_KEY`.

## Optional: Training Artifacts

- Train an ML policy artifact (used by `--policy openai_trained` / `trained_ml`):

```bash
python train_ml_policy.py --episodes-per-task 450 --seed 42 --teacher-policy lookahead --teacher-lookahead-depth 2 --teacher-lookahead-width 8 --output artifacts/ml_policy.pkl --report artifacts/ml_policy_report.json
```

- Generate the final SFT dataset used for LoRA/SFT training:

```bash
python generate_final_dataset.py --seed 42 --lookahead-depth 2 --lookahead-width 8 --output artifacts/flight_rebooking_sft_final.jsonl
```

- Evaluate local GPU + adapter bundle:

```bash
python evaluate_unsloth.py
```
