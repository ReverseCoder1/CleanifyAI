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
from typing import List, Dict, Any
from dotenv import load_dotenv

from openai import OpenAI

# Load environment variables from .env file
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
    """Parse LLM response into action dict."""
    text = response_text.strip()

    # Remove markdown if present
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
        # Try to find JSON in text
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"operation": "finish", "parameters": {}}


def format_action_str(operation: str, parameters: Dict) -> str:
    """Format action as a string for output."""
    if not parameters:
        return operation
    return f"{operation}({json.dumps(parameters)})"


def run_task(
    client: OpenAI,
    task_id: str
) -> Dict[str, Any]:
    """Run one full episode for a task with required output format."""

    # Import here to use local environment
    from environment import DataCleaningEnv
    from models import Action

    env = DataCleaningEnv(task_id=task_id)

    # [START] line
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

    # Reset
    result     = env.reset()
    obs        = result.observation.model_dump()
    done       = result.done
    step       = 0
    rewards    = []
    actions_taken = []
    success    = False
    last_error = None

    while not done and step < MAX_STEPS:
        step += 1
        last_error = None

        # Build prompt
        user_prompt = build_user_prompt(obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ]

        # Call LLM
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            last_error = str(e)
            response_text = '{"operation": "finish", "parameters": {}}'

        # Parse action
        action_dict = parse_action(response_text)
        action = Action(
            operation=action_dict.get("operation", "finish"),
            parameters=action_dict.get("parameters", {})
        )
        actions_taken.append(action.operation)

        # Step environment
        try:
            result  = env.step(action)
            obs     = result.observation.model_dump()
            done    = result.done
            reward  = result.reward.total
            rewards.append(reward)

            # Check if operation failed
            message = obs.get("message", "")
            if "failed" in message.lower() or "error" in message.lower():
                last_error = message

        except Exception as e:
            last_error = str(e)
            reward = 0.0
            rewards.append(reward)
            done = True

        # [STEP] line - required format
        action_str = format_action_str(action.operation, action.parameters)
        error_str = last_error if last_error else "null"
        done_str = "true" if done else "false"
        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error_str}")

        if done:
            success = reward > 0.5  # Consider success if final reward > 0.5
            break

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # [END] line - required format
    success_str = "true" if success else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={success_str} steps={step} rewards={rewards_str}")

    return {
        "task_id":      task_id,
        "final_score":  round(rewards[-1], 4) if rewards else 0.0001,
        "steps":        step,
        "rewards":      rewards,
        "actions":      actions_taken,
        "success":      success
    }


def main():
    # Validate config
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    if not MODEL_NAME:
        print("ERROR: MODEL_NAME not set", file=sys.stderr)
        sys.exit(1)

    # Init client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )

    # Run all tasks
    all_results = []
    for task_id in VALID_TASKS:
        try:
            result = run_task(client, task_id)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr)
            all_results.append({
                "task_id":     task_id,
                "final_score": 0.0001,
                "success":     False,
                "error":       str(e)
            })

    # Save results locally (optional)
    output = {
        "model":         MODEL_NAME,
        "tasks":         all_results,
        "average_score": round(sum(r.get("final_score", 0.0) for r in all_results) / len(all_results), 4)
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
