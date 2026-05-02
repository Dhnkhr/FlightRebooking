import json
import torch
import os
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from environment import Action, ActionType, FlightRebookingEnv
from tasks import TASKS, grade_task

app = FastAPI(
    title="Flight Rebooking AI Agent",
    description="AI-powered airline disruption operations agent.",
    version="2.1.0",
)

_SESSIONS: Dict[str, Dict[str, Any]] = {}
_DEFAULT_SESSION_ID = "default"
_BASE_DIR = Path(__file__).resolve().parent
_FRONTEND_DIR = _BASE_DIR / "frontend"

# Model Globals
MODEL = None
TOKENIZER = None
GROQ_CLIENT = None  # OpenAI-compatible Groq client (fallback when no GPU)

if _FRONTEND_DIR.exists():
    app.mount("/ui/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="ui-static")

class CreateSessionRequest(BaseModel):
    task: str = Field(default="easy", description="One of: easy, medium, hard")

class StepRequest(BaseModel):
    action: Action
    session_id: str = Field(default=_DEFAULT_SESSION_ID)

def _init_groq_client():
    """Initialize Groq client as fallback when no GPU is available."""
    global GROQ_CLIENT
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        print("⚠️ GROQ_API_KEY not set. AI Auto-Play fully disabled.")
        return
    try:
        from openai import OpenAI
        GROQ_CLIENT = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        print("✅ Groq API client initialized (CPU/no-GPU fallback mode).")
    except Exception as e:
        print(f"❌ Failed to initialize Groq client: {e}")

def load_model():
    global MODEL, TOKENIZER, GROQ_CLIENT

    # Already initialized
    if MODEL is not None or GROQ_CLIENT is not None:
        return MODEL, TOKENIZER

    # Try local GPU model first
    if not torch.cuda.is_available():
        print("⚠️ No CUDA GPU found. Falling back to Groq API.")
        _init_groq_client()
        MODEL, TOKENIZER = False, False
        return MODEL, TOKENIZER

    vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    if vram_gb < 6.0:
        print(f"⚠️ GPU has {vram_gb:.1f}GB VRAM (need 6GB+). Falling back to Groq API.")
        _init_groq_client()
        MODEL, TOKENIZER = False, False
        return MODEL, TOKENIZER

    base_model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    adapter_path = "./flight-rebooking-lora"
    
    print(f"Loading AI Model: {base_model_name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    try:
        TOKENIZER = AutoTokenizer.from_pretrained(base_model_name)
        MODEL = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        if os.path.exists(adapter_path):
            print(f"Applying LoRA adapters from {adapter_path}")
            MODEL = PeftModel.from_pretrained(MODEL, adapter_path)
        MODEL.eval()
        print("✅ AI Model Loaded Successfully (local GPU)")
    except Exception as e:
        print(f"❌ Error loading local model: {e}. Falling back to Groq API.")
        MODEL, TOKENIZER = False, False
        _init_groq_client()
    
    return MODEL, TOKENIZER

def _groq_action(observation_json: str) -> dict:
    """Query Groq API for the next action."""
    SYSTEM_PROMPT = (
        "You are an airline disruption operations agent. "
        "Return exactly one JSON object with keys: action_type, passenger_id, flight_id. "
        "action_type must be one of: rebook_passenger, offer_downgrade, book_hotel, "
        "rebook_on_partner, mark_no_solution, finalize. Output raw JSON only."
    )
    try:
        response = GROQ_CLIENT.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current observation: {observation_json}"},
            ],
            temperature=0,
            max_tokens=120,
        )
        content = response.choices[0].message.content or ""
        return extract_json(content)
    except Exception as e:
        print(f"❌ Groq API call failed: {e}")
        return {"action_type": "finalize"}

def extract_json(text: str) -> dict:
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            return json.loads(text[start_idx:end_idx])
    except Exception:
        pass
    return {"action_type": "finalize"}

def _get_session(session_id: str) -> Dict[str, Any]:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return session

def _create_env_session(task_key: str, session_id: str) -> Dict[str, Any]:
    if task_key not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_key}")

    env = FlightRebookingEnv(task_data=TASKS[task_key])
    observation = env.reset()
    _SESSIONS[session_id] = {"task_key": task_key, "env": env, "last_action_str": None}

    return {
        "session_id": session_id,
        "task_key": task_key,
        "observation": observation.model_dump(mode="json"),
    }

def _step_and_format(session: Dict[str, Any], action: Action) -> Dict[str, Any]:
    env: FlightRebookingEnv = session["env"]
    observation, reward, done, info = env.step(action)

    response: Dict[str, Any] = {
        "observation": observation.model_dump(mode="json"),
        "reward": reward.model_dump(mode="json"),
        "done": done,
        "info": info,
    }

    if done:
        task_key = session["task_key"]
        state = env.state()
        response["final_score"] = grade_task(task_key, state, TASKS[task_key]["max_budget"])

    return response

@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the UI so HF Spaces renders the dashboard."""
    return RedirectResponse(url="/ui")

@app.get("/health")
def health() -> Dict[str, Any]:
    ai_ready = (MODEL is not None and MODEL is not False) or (GROQ_CLIENT is not None)
    mode = "local-gpu" if (MODEL is not None and MODEL is not False) else ("groq-api" if GROQ_CLIENT else "none")
    return {
        "name": "flight-rebooking-ai",
        "status": "ok",
        "model_loaded": ai_ready,
        "inference_mode": mode,
    }

@app.on_event("startup")
async def startup_event():
    """Pre-initialize model/client at startup so /auto_step is instant."""
    load_model()

@app.get("/ui", include_in_schema=False)
def ui_page() -> FileResponse:
    index_file = _FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_file)

@app.post("/auto_step")
async def auto_step(session_id: str = _DEFAULT_SESSION_ID):
    session = _get_session(session_id)
    env: FlightRebookingEnv = session["env"]
    
    model, tokenizer = load_model()
    obs = env.state()
    obs_json = obs.model_dump_json()

    # --- Groq API path (no GPU / free Space) ---
    if GROQ_CLIENT is not None and (model is False or model is None):
        action_dict = _groq_action(obs_json)

    # --- Local GPU / LoRA path ---
    elif model and tokenizer:
        system_prompt = "You are an airline disruption agent. Return a single JSON object with action_type, passenger_id, and flight_id."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current State: {obs_json}"}
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=64, do_sample=False)
        response_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        action_dict = extract_json(response_text)

    else:
        raise HTTPException(status_code=503, detail="AI model unavailable. Set GROQ_API_KEY as a Space secret.")

    # Loop Breaker
    action_str = json.dumps(action_dict)
    if session.get("last_action_str") == action_str:
        action_dict = {"action_type": "mark_no_solution", "passenger_id": action_dict.get("passenger_id", "P1")}
    session["last_action_str"] = action_str

    try:
        action = Action(**action_dict)
    except Exception:
        action = Action(action_type=ActionType.FINALIZE)
        
    return _step_and_format(session, action)

@app.post("/reset")
def reset_default(request: CreateSessionRequest = None) -> Dict[str, Any]:
    if request is None: request = CreateSessionRequest()
    return _create_env_session(task_key=request.task.lower(), session_id=_DEFAULT_SESSION_ID)

@app.post("/step")
def step_default(request: StepRequest) -> Dict[str, Any]:
    session = _get_session(request.session_id)
    return _step_and_format(session=session, action=request.action)

@app.get("/state")
def state_default(session_id: str = _DEFAULT_SESSION_ID) -> Dict[str, Any]:
    session = _get_session(session_id)
    env: FlightRebookingEnv = session["env"]
    state = env.state()
    return {
        "state": state.model_dump(mode="json"),
        "grade": grade_task(session["task_key"], state, TASKS[session["task_key"]]["max_budget"]),
    }

@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    payload = []
    for task_key, task in TASKS.items():
        payload.append({
            "task_key": task_key,
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "objective": task["objective"],
            "max_budget": task["max_budget"],
            "passenger_count": len(task["passengers"]),
        })
    return {"tasks": payload}

def start():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    start()
