
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from models import Action, StepResult, Reward
from environment import DataCleaningEnv

# ─────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────

app = FastAPI(
    title="Data Cleaning OpenEnv",
    description=(
        "An OpenEnv-compliant environment where AI agents "
        "learn to clean messy real-world datasets step by step."
    ),
    version="1.0.0"
)

# Serve UI
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# One environment instance per task
# ─────────────────────────────────────────

VALID_TASKS = [
    "easy_dedup_rename",
    "medium_missing_dtype",
    "hard_full_pipeline"
]

envs: Dict[str, DataCleaningEnv] = {
    task_id: DataCleaningEnv(task_id=task_id)
    for task_id in VALID_TASKS
}


def get_env(task_id: str) -> DataCleaningEnv:
    if task_id not in envs:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Task '{task_id}' not found. "
                f"Valid tasks: {VALID_TASKS}"
            )
        )
    return envs[task_id]


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.get("/ui")
def ui():
    return FileResponse("static/index.html")

@app.get("/")
def root():
    return {
        "name": "Data Cleaning OpenEnv",
        "version": "1.0.0",
        "status": "running",
        "tasks": VALID_TASKS,
        "endpoints": {
            "reset":    "POST /reset/{task_id}",
            "step":     "POST /step/{task_id}",
            "state":    "GET  /state/{task_id}",
            "tasks":    "GET  /tasks",
            "health":   "GET  /health",
            "docs":     "GET  /docs"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tasks_loaded": len(envs)
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id":   "easy_dedup_rename",
                "difficulty": "easy",
                "description": (
                    "Remove duplicate rows and rename columns "
                    "to snake_case in an employee dataset."
                ),
                "max_steps": 10,
                "operations": ["remove_duplicates", "rename_columns", "finish"]
            },
            {
                "task_id":   "medium_missing_dtype",
                "difficulty": "medium",
                "description": (
                    "Fill missing values using correct strategies "
                    "and fix wrong data types in a customer dataset."
                ),
                "max_steps": 15,
                "operations": ["fill_missing", "fix_dtype", "finish"]
            },
            {
                "task_id":   "hard_full_pipeline",
                "difficulty": "hard",
                "description": (
                    "Run a full cleaning pipeline: remove duplicates, "
                    "fill missing values, fix dtypes, remove outliers, "
                    "and validate schema on an orders dataset."
                ),
                "max_steps": 20,
                "operations": [
                    "remove_duplicates", "fill_missing", "fix_dtype",
                    "remove_outliers", "validate_schema", "finish"
                ]
            }
        ]
    }


@app.post("/reset/{task_id}")
def reset(task_id: str):
    """Reset environment and start fresh episode."""
    env = get_env(task_id)
    try:
        result = env.reset()
        return result.dict()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reset failed: {str(e)}"
        )


@app.post("/step/{task_id}")
def step(task_id: str, action: Action):
    """Take one action in the environment."""
    env = get_env(task_id)
    if env.current_df is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset/{task_id} first."
        )
    try:
        result = env.step(action)
        return result.dict()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Step failed: {str(e)}"
        )


@app.get("/state/{task_id}")
def state(task_id: str):
    """Get current environment state."""
    env = get_env(task_id)
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"State failed: {str(e)}"
        )


@app.get("/validate")
def validate():
    """OpenEnv spec validation endpoint."""
    results = {}
    for task_id in VALID_TASKS:
        try:
            env = DataCleaningEnv(task_id=task_id)
            # Test reset
            reset_result = env.reset()
            assert reset_result.observation is not None
            assert reset_result.reward is not None
            assert reset_result.done == False

            # Test step
            from models import Action
            action = Action(
                operation="remove_duplicates",
                parameters={}
            )
            step_result = env.step(action)
            assert step_result.observation is not None
            assert 0.0 <= step_result.reward.total <= 1.0

            # Test state
            state_result = env.state()
            assert "task_id" in state_result

            results[task_id] = {
                "status": "passed",
                "reset": "ok",
                "step": "ok",
                "state": "ok",
                "reward_range": f"{step_result.reward.total}"
            }
        except Exception as e:
            results[task_id] = {
                "status": "failed",
                "error": str(e)
            }

    all_passed = all(r["status"] == "passed" for r in results.values())
    return {
        "openenv_valid": all_passed,
        "tasks": results
    }
