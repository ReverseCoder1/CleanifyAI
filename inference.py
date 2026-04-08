
"""
Inference Script — Data Cleaning OpenEnv
=========================================
Runs a baseline LLM agent against all 3 tasks
and produces reproducible scores.

Required environment variables:
    API_BASE_URL  — LLM API endpoint
    MODEL_NAME    — model identifier
    HF_TOKEN      — Hugging Face API key
"""

import os
import sys
import json
import time
from typing import List, Dict, Any

from openai import OpenAI

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")

MAX_STEPS   = 15
TEMPERATURE = 0.1
MAX_TOKENS  = 400

VALID_TASKS = [
    "easy_dedup_rename",
    "medium_missing_dtype",
    "hard_full_pipeline"
]

SYSTEM_PROMPT = """
You are an expert data cleaning agent.
You will receive information about a dirty dataset and must clean it
step by step using the available operations.

Available operations:
- remove_duplicates: {"operation": "remove_duplicates", "parameters": {}}
- fill_missing: {"operation": "fill_missing", "parameters": {"strategy": "mean|median|mode"}}
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


def run_task(
    client: OpenAI,
    env_module,
    task_id: str
) -> Dict[str, Any]:
    """Run one full episode for a task."""
    print(f"\n{'='*50}")
    print(f"  Task: {task_id}")
    print(f"{'='*50}")

    # Import here to use local environment
    from environment import DataCleaningEnv
    from models import Action

    env = DataCleaningEnv(task_id=task_id)

    # Reset
    result     = env.reset()
    obs        = result.observation.model_dump()
    done       = result.done
    step       = 0
    rewards    = []
    actions_taken = []

    print(f"  Description: {obs.get('task_description', '')[:80]}...")
    print(f"  Initial shape: {obs.get('shape', [])}")
    print(f"  Duplicates: {obs.get('duplicate_count', 0)}")
    print(f"  Missing: {sum(obs.get('missing_values', {}).values())}")

    while not done and step < MAX_STEPS:
        step += 1

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
            print(f"  [Step {step}] LLM error: {e}")
            response_text = '{"operation": "finish", "parameters": {}}'

        # Parse action
        action_dict = parse_action(response_text)
        action = Action(
            operation=action_dict.get("operation", "finish"),
            parameters=action_dict.get("parameters", {})
        )

        print(f"  [Step {step}] Action: {action.operation} {action.parameters}")

        # Step environment
        result  = env.step(action)
        obs     = result.observation.model_dump()
        done    = result.done
        reward  = result.reward.total
        rewards.append(reward)
        actions_taken.append(action.operation)

        print(f"           Reward: {reward:.4f} | Message: {obs.get('message', '')[:60]}")

        if done:
            print(f"  Episode done at step {step}")
            break

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    final_reward = rewards[-1] if rewards else 0.0
    print(f"\n  Final Score: {final_reward:.4f}")
    print(f"  Steps taken: {step}")
    print(f"  Actions: {actions_taken}")

    return {
        "task_id":      task_id,
        "final_score":  round(final_reward, 4),
        "steps":        step,
        "rewards":      rewards,
        "actions":      actions_taken,
        "done":         done
    }


def main():
    print("\n" + "="*50)
    print("  Data Cleaning OpenEnv — Baseline Inference")
    print("="*50)
    print(f"  Model:    {MODEL_NAME}")
    print(f"  API URL:  {API_BASE_URL}")
    print(f"  Tasks:    {VALID_TASKS}")
    print("="*50)

    # Validate config
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY not set")
        sys.exit(1)

    if not MODEL_NAME:
        print("ERROR: MODEL_NAME not set")
        sys.exit(1)

    # Init client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    # Run all tasks
    all_results = []
    for task_id in VALID_TASKS:
        try:
            result = run_task(client, None, task_id)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR on task {task_id}: {e}")
            all_results.append({
                "task_id":     task_id,
                "final_score": 0.0,
                "error":       str(e)
            })

    # Summary
    print("\n" + "="*50)
    print("  FINAL RESULTS SUMMARY")
    print("="*50)
    total_score = 0.0
    for r in all_results:
        score = r.get("final_score", 0.0)
        total_score += score
        status = "ERROR" if "error" in r else "OK"
        print(f"  {r['task_id']:<30} Score: {score:.4f}  [{status}]")

    avg_score = total_score / len(all_results)
    print(f"\n  Average Score: {avg_score:.4f}")
    print("="*50)

    # Save results
    output = {
        "model":        MODEL_NAME,
        "tasks":        all_results,
        "average_score": round(avg_score, 4)
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Results saved to baseline_results.json")
    print("  Done!")


if __name__ == "__main__":
    main()
