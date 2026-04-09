"""
Inference Script - Data Cleaning OpenEnv
=========================================
Runs a baseline LLM agent against all tasks
and produces output in the required format for hackathon submission.

Required environment variables:
    API_BASE_URL  - LLM API endpoint (default: https://api.openai.com/v1)
    MODEL_NAME    - model identifier (default: gpt-4o-mini)
    HF_TOKEN      - Hugging Face API key (REQUIRED)
"""

import os
import sys
import json
import time
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

MAX_STEPS   = 20
TEMPERATURE = 0.1
MAX_TOKENS  = 400

VALID_TASKS = [
    "easy_dedup_rename",
    "medium_missing_dtype",
    "hard_full_pipeline",
]

ENV_NAME = "data-cleaning-openenv"
SUCCESS_SCORE_THRESHOLD = 0.3

SYSTEM_PROMPT = """
You are an expert data cleaning agent.
You will receive information about a dirty dataset and must clean it
step by step using the available operations.

Available operations:
- remove_duplicates: {"operation": "remove_duplicates", "parameters": {}}
- fill_missing_mean: {"operation": "fill_missing_mean", "parameters": {}}
- fill_missing_mode: {"operation": "fill_missing_mode", "parameters": {}}
- fill_missing_median: {"operation": "fill_missing_median", "parameters": {}}
- fix_dtype: {"operation": "fix_dtype", "parameters": {"dtype": "auto"}}
- remove_outliers: {"operation": "remove_outliers", "parameters": {"method": "iqr"}}
- rename_columns: {"operation": "rename_columns", "parameters": {}}
- validate_schema: {"operation": "validate_schema", "parameters": {}}
- finish: {"operation": "finish", "parameters": {}}

Rules:
1. Always respond with ONLY a valid JSON object
2. No explanations, no markdown, no extra text
3. Just the JSON action object
4. Call finish when you think the dataset is clean

Example response:
{"operation": "remove_duplicates", "parameters": {}}
"""


# ─────────────────────────────────────────
# STDOUT LOGGING — matches official sample format exactly
# ─────────────────────────────────────────

def _clamp(v: float) -> float:
    """Clamp to strictly (0, 1) — grader rejects exactly 0.0 and 1.0."""
    return max(0.01, min(0.99, float(v)))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # reward to 2 decimal places per spec
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={_clamp(reward):.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # [END] format from official sample:
    # [END] success=... steps=... score=<score> rewards=<r1,r2,...>
    rewards_str = ",".join(f"{_clamp(r):.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={_clamp(score):.2f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def build_user_prompt(observation: Dict[str, Any]) -> str:
    return f"""
Task: {observation.get("task_description", "")}
Step: {observation.get("step", 0)}
Last message: {observation.get("message", "")}

Current dataset info:
- Shape: {observation.get("shape", [])}
- Columns: {observation.get("columns", [])}
- Duplicate rows: {observation.get("duplicate_count", 0)}
- Missing values: {observation.get("missing_values", {})}
- Data types: {observation.get("dtypes", {})}

Sample rows (first 3):
{json.dumps(observation.get("sample_rows", []), indent=2)}

Available operations: {observation.get("available_operations", [])}

What is your next action? Respond with JSON only.
"""


def parse_action(response_text: str) -> Dict[str, Any]:
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        action = json.loads(text)
        if "operation" not in action:
            return {"operation": "finish", "parameters": {}}
        if "parameters" not in action:
            action["parameters"] = {}
        return action
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"operation": "finish", "parameters": {}}


def format_action_str(operation: str, parameters: Dict) -> str:
    if not parameters:
        return operation
    return f"{operation}({json.dumps(parameters)})"


# ─────────────────────────────────────────
# MAIN TASK RUNNER
# ─────────────────────────────────────────

def run_task(client: OpenAI, task_id: str) -> Dict[str, Any]:
    from environment import DataCleaningEnv
    from models import Action

    env = DataCleaningEnv(task_id=task_id)

    log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

    result              = env.reset()
    obs                 = result.observation.model_dump()
    done                = result.done
    step                = 0
    rewards: List[float] = []
    actions_taken: List[str] = []
    score               = 0.01
    success             = False
    last_error: Optional[str] = None

    try:
        while not done and step < MAX_STEPS:
            step += 1
            last_error = None

            # ── Call LLM ────────────────────────────────────────────
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_prompt(obs)},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as e:
                last_error    = str(e)
                response_text = '{"operation": "finish", "parameters": {}}'

            # ── Parse & step env ─────────────────────────────────────
            action_dict = parse_action(response_text)
            action      = Action(
                operation=action_dict.get("operation", "finish"),
                parameters=action_dict.get("parameters", {}),
            )
            actions_taken.append(action.operation)

            try:
                result = env.step(action)
                obs    = result.observation.model_dump()
                done   = result.done
                reward = float(result.reward.total)
                msg    = obs.get("message", "")
                if "failed" in msg.lower() or "error" in msg.lower():
                    last_error = msg
            except Exception as e:
                last_error = str(e)
                reward     = 0.01
                done       = True

            rewards.append(reward)

            # ── [STEP] line ──────────────────────────────────────────
            log_step(
                step=step,
                action=format_action_str(action.operation, action.parameters),
                reward=reward,
                done=done,
                error=last_error,
            )

            if done:
                break

            time.sleep(0.5)

        # score = average reward across all steps, clamped strictly to (0,1)
        score   = _clamp(sum(rewards) / max(len(rewards), 1))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        # [END] always emitted — even on exception
        log_end(success=success, steps=step, score=score, rewards=rewards)

    return {
        "task_id":     task_id,
        "final_score": score,
        "steps":       step,
        "rewards":     [_clamp(r) for r in rewards],
        "actions":     actions_taken,
        "success":     success,
    }


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    all_results = []
    for task_id in VALID_TASKS:
        try:
            result = run_task(client, task_id)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr)
            log_end(success=False, steps=0, score=0.01, rewards=[0.01])
            all_results.append({
                "task_id":     task_id,
                "final_score": 0.01,
                "success":     False,
                "error":       str(e),
            })

    avg = sum(r.get("final_score", 0.01) for r in all_results) / len(all_results)
    output = {
        "model":         MODEL_NAME,
        "tasks":         all_results,
        "average_score": _clamp(avg),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()