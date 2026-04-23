import json
import torch
import os
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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

if _FRONTEND_DIR.exists():
    app.mount("/ui/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="ui-static")

class CreateSessionRequest(BaseModel):
    task: str = Field(default="easy", description="One of: easy, medium, hard")

class StepRequest(BaseModel):
    action: Action
    session_id: str = Field(default=_DEFAULT_SESSION_ID)

def load_model():
    global MODEL, TOKENIZER
    if MODEL is not None:
        return MODEL, TOKENIZER

    # Check if GPU is available and has enough VRAM (need at least 6GB)
    if not torch.cuda.is_available():
        print("⚠️ No CUDA GPU found. AI Auto-Play disabled (heuristic mode only).")
        MODEL, TOKENIZER = False, False
        return MODEL, TOKENIZER
    
    vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    if vram_gb < 6.0:
        print(f"⚠️ GPU has {vram_gb:.1f}GB VRAM (need 6GB+). AI Auto-Play disabled.")
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
        print("✅ AI Model Loaded Successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        MODEL, TOKENIZER = False, False
    
    return MODEL, TOKENIZER

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

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "flight-rebooking-ai",
        "status": "ok",
        "model_loaded": MODEL is not None and MODEL is not False,
        "message": "Use /ui for the dashboard.",
    }

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
    if model is False:
        raise HTTPException(status_code=500, detail="AI Model failed to load.")
    
    obs = env.state() # Get full state for AI context
    
    system_prompt = "You are an airline disruption agent. Return a single JSON object with action_type, passenger_id, and flight_id."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Current State: {obs.model_dump_json()}"}
    ]
    
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=64, do_sample=False)
    
    response_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    action_dict = extract_json(response_text)
    
    # Loop Breaker
    action_str = json.dumps(action_dict)
    if session.get("last_action_str") == action_str:
        action_dict = {"action_type": "mark_no_solution", "passenger_id": action_dict.get("passenger_id", "P1")}
    session["last_action_str"] = action_str

    try:
        action = Action(**action_dict)
    except:
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
