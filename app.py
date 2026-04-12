"""
FastAPI service wrapper for the Flight Rebooking OpenEnv environment.

This enables containerized deployment on Hugging Face Spaces while preserving
step/reset/state semantics through HTTP endpoints.
"""

from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from environment import Action, FlightRebookingEnv
from tasks import TASKS, grade_task


app = FastAPI(
    title="Flight Rebooking OpenEnv",
    description="Real-world airline disruption simulation with OpenEnv-style semantics.",
    version="2.0.0",
)


_SESSIONS: Dict[str, Dict[str, Any]] = {}
_DEFAULT_SESSION_ID = "default"
_BASE_DIR = Path(__file__).resolve().parent
_FRONTEND_DIR = _BASE_DIR / "frontend"

if _FRONTEND_DIR.exists():
    app.mount("/ui/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="ui-static")


class CreateSessionRequest(BaseModel):
    task: str = Field(default="easy", description="One of: easy, medium, hard")


class StepRequest(BaseModel):
    action: Action
    session_id: str = Field(default=_DEFAULT_SESSION_ID)


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
    _SESSIONS[session_id] = {"task_key": task_key, "env": env}

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
        "name": "flight-rebooking-openenv",
        "status": "ok",
        "message": "Use /ui for the dashboard or /reset, /step, /state API endpoints.",
    }


@app.get("/ui", include_in_schema=False)
def ui_page() -> FileResponse:
    index_file = _FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend is not available.")
    return FileResponse(index_file)


@app.post("/reset")
def reset_default(request: CreateSessionRequest) -> Dict[str, Any]:
    task_key = request.task.lower()
    return _create_env_session(task_key=task_key, session_id=_DEFAULT_SESSION_ID)


@app.post("/step")
def step_default(request: StepRequest) -> Dict[str, Any]:
    session = _get_session(request.session_id)
    response = _step_and_format(session=session, action=request.action)
    response["session_id"] = request.session_id
    return response


@app.get("/state")
def state_default(session_id: str = _DEFAULT_SESSION_ID) -> Dict[str, Any]:
    session = _get_session(session_id)
    env: FlightRebookingEnv = session["env"]
    task_key = session["task_key"]
    state = env.state()

    return {
        "session_id": session_id,
        "task_key": task_key,
        "state": state.model_dump(mode="json"),
        "grade": grade_task(task_key, state, TASKS[task_key]["max_budget"]),
    }


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    payload = []
    for task_key, task in TASKS.items():
        payload.append(
            {
                "task_key": task_key,
                "task_id": task["task_id"],
                "difficulty": task["difficulty"],
                "objective": task["objective"],
                "max_budget": task["max_budget"],
                "passenger_count": len(task["passengers"]),
            }
        )
    return {"tasks": payload}


@app.post("/sessions")
def create_session(request: CreateSessionRequest) -> Dict[str, Any]:
    task_key = request.task.lower()
    session_id = str(uuid4())
    return _create_env_session(task_key=task_key, session_id=session_id)


@app.post("/sessions/{session_id}/reset")
def reset_session(session_id: str) -> Dict[str, Any]:
    session = _get_session(session_id)
    env: FlightRebookingEnv = session["env"]
    observation = env.reset()
    return {
        "session_id": session_id,
        "task_key": session["task_key"],
        "observation": observation.model_dump(mode="json"),
    }


@app.post("/sessions/{session_id}/step")
def step_session(session_id: str, action: Action) -> Dict[str, Any]:
    session = _get_session(session_id)
    return _step_and_format(session=session, action=action)


@app.get("/sessions/{session_id}/state")
def get_state(session_id: str) -> Dict[str, Any]:
    session = _get_session(session_id)
    env: FlightRebookingEnv = session["env"]
    task_key = session["task_key"]
    state = env.state()

    return {
        "task_key": task_key,
        "state": state.model_dump(mode="json"),
        "grade": grade_task(task_key, state, TASKS[task_key]["max_budget"]),
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    _get_session(session_id)
    del _SESSIONS[session_id]
    return {"deleted": session_id}
