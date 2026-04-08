---
title: Data Cleaning OpenEnv
sdk: docker
tags:
  - openenv
  - data-cleaning
  - reinforcement-learning
  - agent
  - real-world
  - tabular
  - pandas
---

# 🧹 Data Cleaning OpenEnv

> *An OpenEnv-compliant environment where AI agents learn to clean messy real-world datasets — step by step, with dense reward signals and sequence-aware penalties.*

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-HF%20Space-blue)](https://thorodin103-data-cleaning.hf.space/ui)
[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/thorodin103/Data-cleaning/tree/main)
[![OpenEnv Valid](https://img.shields.io/badge/openenv%20validate-✅%20passing-brightgreen)](#-openenv-spec-compliance)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](Dockerfile)

| | |
|---|---|
| **Live UI** | https://thorodin103-data-cleaning.hf.space/ui |
| **HF Repository** | https://huggingface.co/spaces/thorodin103/Data-cleaning/tree/main |
| **API Base** | https://thorodin103-data-cleaning.hf.space |
| **Validate** | https://thorodin103-data-cleaning.hf.space/validate |
| **API Docs** | https://thorodin103-data-cleaning.hf.space/docs |

---

## 🌍 Why This Environment Exists

Data engineers and analysts spend **up to 80% of their time cleaning data** before it can be used. Despite being one of the most universal tasks in the real world, no existing OpenEnv benchmark captures it.

This environment fills that gap. An AI agent receives a dirty dataset and must apply a sequence of cleaning operations to match a gold-standard output. What makes it non-trivial:

- **Order matters.** Filling missing values before fixing data types produces wrong results — the reward function knows this and penalizes it.
- **Partial progress is rewarded.** Every action shifts the score, giving RL agents a dense learning signal rather than sparse end-of-episode feedback.
- **Four difficulty tiers.** From a 3-step deduplication task to a full expert-level sales pipeline requiring case normalization, outlier removal, and schema validation — in the right sequence.
- **Fully live.** The environment runs on Hugging Face Spaces, passes `openenv validate`, and exposes a REST API any agent can call against right now.

---

## 🎯 Tasks

| Task ID | Difficulty | Max Steps | Dataset | What the Agent Must Do |
|---|---|---|---|---|
| `easy_dedup_rename` | 🟢 Easy | 10 | Employee records | Remove duplicate rows + rename columns to snake_case |
| `medium_missing_dtype` | 🟡 Medium | 15 | Customer records | Fill missing values (mean/mode) + fix wrong data types |
| `hard_full_pipeline` | 🔴 Hard | 20 | Orders data | Full pipeline: dedup → fill → fix types → remove outliers → validate schema |
| `expert_sales_pipeline` | ⚫ Expert | 25 | Sales transactions | Expert pipeline with case standardization + all of the above, in strict order |

---

### Task 1 — Easy: Deduplicate & Rename (`easy_dedup_rename`)

The agent receives an employee dataset with duplicate rows and column names like `EMP ID`, `DEP T`, `SAL ARY`. It must remove the duplicates and rename all columns to `snake_case`.

**Dirty dataset sample:**
```
EMP ID | EMP NAME | DEP T   | SAL ARY | AGE
101    | Alice    | HR      | 50000   | 28
102    | Bob      | IT      | 60000   | 35
102    | Bob      | IT      | 60000   | 35   ← duplicate
103    | Charlie  | IT      | 55000   | 29
103    | Charlie  | IT      | 55000   | 29   ← duplicate
```

**Scoring:**
- `duplicate_score` (0.5) — how close row count is to gold
- `schema_score` (0.5) — proportion of column names matching gold

**Optimal actions:** `remove_duplicates` → `rename_columns` → `finish` (3 steps, score = 0.99)

---

### Task 2 — Medium: Missing Values & Types (`medium_missing_dtype`)

The agent receives a customer dataset where numeric columns like `age`, `purchases`, and `salary` contain missing values and are stored as the wrong type (strings instead of numbers).

**Issues present:**
- `age` column: 2 missing values, stored as `object` dtype
- `purchases` column: 3 missing values
- `salary` column: stored as string, needs numeric conversion

**Scoring:**
- `missing_score` (0.5) — proportion of missing values correctly filled
- `dtype_score` (0.5) — proportion of columns with correct data types

**Key challenge:** The agent must fix types *before* filling missing values with mean/median — doing it backwards triggers a sequence penalty.

---

### Task 3 — Hard: Full Pipeline (`hard_full_pipeline`)

The agent receives an orders dataset with all classes of data quality issues simultaneously. Must execute the full cleaning pipeline in the correct sequence.

**Issues present:**
- Duplicate order IDs
- Missing values in `quantity`, `product`, and `rating` columns
- `quantity` and `price` stored as wrong types
- Extreme outliers (e.g., `quantity=999`, `price=99999`)
- Column names not matching gold schema

**Scoring:** All 5 components weighted equally at 0.2 each:
`duplicate_score + missing_score + dtype_score + outlier_score + schema_score`

**Why it's hard:** The agent must not only apply all 5 operations but apply them in the correct order. Doing `fill_missing` before `remove_duplicates` incurs a -0.08 penalty. Doing `remove_outliers` before `fix_dtype` incurs another. Penalties compound.

---

### Task 4 — Expert: Sales Pipeline (`expert_sales_pipeline`)

The hardest task. A sales transaction dataset with messy region names (`north`, `EAST`, `South` — all meaning the same thing), missing sales reps and commission values, mixed column naming conventions, and outlier transactions.

**Issues present:**
- Duplicate transactions
- Mixed-case region names requiring standardization
- Missing `SALES REP` and `COMMISSION %` values
- Inconsistent column naming (`Transaction_ID`, `SALES REP`, `Sale Amount` — all need snake_case)
- Outlier transactions in `Sale Amount` and `Units Sold`

**Scoring:**
- `duplicate_score` (0.15), `missing_score` (0.20), `dtype_score` (0.20), `outlier_score` (0.20), `schema_score` (0.25)

**Total max steps: 25** — the most complex episode in the environment.

---

## 👁️ Observation Space

At every step, the agent receives a structured observation describing the current state of the dataset:

```json
{
  "task_id":              "easy_dedup_rename",
  "step":                 1,
  "dataset_info": {
    "total_rows":         8,
    "total_columns":      5,
    "has_duplicates":     true,
    "has_missing":        false
  },
  "columns":              ["EMP ID", "EMP NAME", "DEP T", "SAL ARY", "AGE"],
  "shape":                [8, 5],
  "missing_values":       {"EMP ID": 0, "EMP NAME": 0, "DEP T": 0, "SAL ARY": 0, "AGE": 0},
  "dtypes":               {"EMP ID": "int64", "EMP NAME": "object", "DEP T": "object", "SAL ARY": "int64", "AGE": "int64"},
  "duplicate_count":      2,
  "sample_rows": [
    {"EMP ID": 101, "EMP NAME": "Alice", "DEP T": "HR", "SAL ARY": 50000, "AGE": 28},
    {"EMP ID": 102, "EMP NAME": "Bob",   "DEP T": "IT", "SAL ARY": 60000, "AGE": 35},
    {"EMP ID": 102, "EMP NAME": "Bob",   "DEP T": "IT", "SAL ARY": 60000, "AGE": 35}
  ],
  "available_operations": ["remove_duplicates", "rename_columns", "finish"],
  "task_description":     "Remove duplicate rows and rename columns to snake_case in an employee dataset.",
  "message":              "Environment reset. Start cleaning!"
}
```

**What's hidden from the agent:** the gold-standard dataset. The agent only sees the dirty data and its own progress metrics.

---

## ⚡ Action Space

Actions are JSON objects with an `operation` and optional `parameters`:

```json
{
  "operation": "fill_missing",
  "parameters": {
    "column":   "age",
    "strategy": "mean"
  }
}
```

| Operation | Parameters | Description |
|---|---|---|
| `remove_duplicates` | `subset` (optional list of columns) | Drop duplicate rows |
| `fill_missing` | `column` (optional), `strategy`: `mean`/`median`/`mode`/`ffill` | Fill NaN values |
| `fix_dtype` | `column` (optional), `dtype`: `int`/`float`/`str`/`auto` | Cast column types |
| `remove_outliers` | `column` (optional), `method`: `iqr`/`zscore` | Remove statistical outliers |
| `rename_columns` | `mapping` (optional dict, auto snake_case if omitted) | Rename column headers |
| `validate_schema` | — | Check columns against gold standard, returns feedback |
| `finish` | — | End the episode and lock in the final score |

---

## 🏆 Reward Function

Rewards are computed **after every action** — dense signal at every step, not just at episode end.

### Components

| Component | Formula | Weight (hard task) |
|---|---|---|
| `duplicate_score` | `min(1, gold_rows / curr_rows)` | 0.20 |
| `missing_score` | `filled_so_far / total_needed` | 0.20 |
| `dtype_score` | `matching_dtypes / total_columns` | 0.20 |
| `outlier_score` | `1 − outlier_rows / total_rows` | 0.20 |
| `schema_score` | `matching_cols / gold_cols` | 0.20 |
| `penalty` | sequence + step violations (subtracted) | — |

**Total reward = weighted component sum − penalties, clamped to [0.0, 1.0]**

### The Sequence-Penalty Mechanic

This is the environment's core design innovation. The optimal cleaning sequence is:

```
1. remove_duplicates   → clean redundant data first
2. fix_dtype           → establish correct column types
3. fill_missing        → impute based on correct types
4. remove_outliers     → after type-correct distributions
5. validate_schema     → final check
```

Violations are penalized:

| Violation | Penalty |
|---|---|
| Out-of-order operation (e.g., `fill_missing` before `fix_dtype`) | −0.08 |
| Repeated identical operation back-to-back | −0.02 |
| Exceeding 80% of the step budget | −0.05 |
| Total penalty cap | 0.25 |

**Why this matters for agent training:** An agent that randomly applies operations will peak early then watch its score decay as penalties accumulate. This forces the agent to learn the *why* behind the sequence, not just the *what*.

From the baseline run on the hard task — the agent peaks at **0.885** on step 5 then degrades to **0.664** by step 20 by looping between `remove_outliers` and `fix_dtype` in a penalty-accumulating cycle. This is a real learning signal.

---

## 📊 Baseline Scores

Baseline agent: **GPT-4o-mini** via OpenAI API, temperature 0.1.

| Task | Difficulty | Final Score | Steps Used | Notes |
|---|---|---|---|---|
| `easy_dedup_rename` | 🟢 Easy | **0.99** | 3 / 10 | Near-perfect — follows optimal sequence |
| `medium_missing_dtype` | 🟡 Medium | **0.70** | 15 / 15 | Hits step limit, oscillates on dtype/fill |
| `hard_full_pipeline` | 🔴 Hard | **0.6636** | 20 / 20 | Peaks at 0.885 then degrades from looping |
| `expert_sales_pipeline` | ⚫ Expert | **~0.55** | ~20 / 25 | Case normalization is the main challenge |
| **Average** | — | **~0.75** | — | Significant room for better agents |

The gap between easy (0.99) and hard (0.66) demonstrates genuine difficulty scaling. The hard task's reward degradation curve is directly caused by sequence violations — a smarter agent that plans its sequence upfront would score 0.88+.

---

## 🔌 API Reference

The environment runs as a live REST API. All endpoints are accessible now:

```
Base URL: https://thorodin103-data-cleaning.hf.space
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `GET` | `/tasks` | List all 4 tasks with metadata |
| `POST` | `/reset/{task_id}` | Reset environment, returns initial observation |
| `POST` | `/step/{task_id}` | Submit action, returns observation + reward + done |
| `GET` | `/state/{task_id}` | Current internal state (for debugging) |
| `GET` | `/validate` | OpenEnv compliance check — all 4 tasks pass |
| `GET` | `/leaderboard` | View agent rankings |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/ui` | Live browser demo |

**Live validate response (confirmed passing):**
```json
{
  "openenv_valid": true,
  "tasks": {
    "easy_dedup_rename":     {"status": "passed", "reset": "ok", "step": "ok", "state": "ok"},
    "medium_missing_dtype":  {"status": "passed", "reset": "ok", "step": "ok", "state": "ok"},
    "hard_full_pipeline":    {"status": "passed", "reset": "ok", "step": "ok", "state": "ok"},
    "expert_sales_pipeline": {"status": "passed", "reset": "ok", "step": "ok", "state": "ok"}
  }
}
```

---

## 🚀 Setup & Usage

### Option 1 — Use the live API (no setup needed)

```python
import requests

BASE = "https://thorodin103-data-cleaning.hf.space"

# Reset the hard task
obs = requests.post(f"{BASE}/reset/hard_full_pipeline").json()
print(obs["observation"]["duplicate_count"])  # → 3

# Take an action
result = requests.post(f"{BASE}/step/hard_full_pipeline", json={
    "operation": "remove_duplicates",
    "parameters": {}
}).json()

print(result["reward"]["total"])          # → 0.6955
print(result["reward"]["duplicate_score"]) # → 1.0
print(result["done"])                      # → False

# Get state
state = requests.get(f"{BASE}/state/hard_full_pipeline").json()
```

### Option 2 — Run locally with Docker

```bash
git clone https://huggingface.co/spaces/thorodin103/Data-cleaning
cd Data-cleaning

docker build -t data-cleaning-openenv .
docker run -p 7860:7860 data-cleaning-openenv

# Environment now live at http://localhost:7860
# UI at http://localhost:7860/ui
```

### Option 3 — Run locally without Docker

```bash
pip install fastapi uvicorn pydantic pandas numpy openai python-dotenv

uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

### Option 4 — Run the baseline inference script

```bash
export OPENAI_API_KEY=your_key_here
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=https://api.openai.com/v1
export HF_TOKEN=your_hf_token

python inference.py
# Runs all 3 core tasks, prints [START] / [STEP] / [END] trace
# Saves results to baseline_results.json
```

---

## 💻 Programmatic Usage (Python)

```python
from environment import DataCleaningEnv
from models import Action

# Initialize any task
env = DataCleaningEnv(task_id="hard_full_pipeline")

# Reset — returns StepResult with initial observation
result = env.reset()
obs = result.observation
print(f"Dirty dataset: {obs.shape} with {obs.duplicate_count} duplicates")
# → Dirty dataset: [20, 8] with 3 duplicates

# Optimal sequence for the hard task
actions = [
    Action(operation="remove_duplicates", parameters={}),
    Action(operation="fix_dtype",         parameters={"dtype": "auto"}),
    Action(operation="fill_missing",      parameters={"strategy": "mean"}),
    Action(operation="remove_outliers",   parameters={"method": "iqr"}),
    Action(operation="validate_schema",   parameters={}),
    Action(operation="finish",            parameters={}),
]

for action in actions:
    result = env.step(action)
    print(f"{action.operation:20s} → reward: {result.reward.total:.4f}")

# → remove_duplicates    → reward: 0.6955
# → fix_dtype            → reward: 0.8235
# → fill_missing         → reward: 0.8852 (no penalty — correct order)
# → remove_outliers      → reward: 0.9800
# → validate_schema      → reward: 0.9800
# → finish               → reward: 0.9800

# Full state inspection
state = env.state()
print(state["reward_history"])
# → [0.6955, 0.8235, 0.8852, 0.9800, 0.9800, 0.9800]
```

---

## 🧪 OpenEnv Spec Compliance

All four required interface methods are implemented:

| Method | Signature | Returns |
|---|---|---|
| `reset()` | `env.reset()` | `StepResult` (obs + reward + done + info) |
| `step(action)` | `env.step(Action)` | `StepResult` |
| `state()` | `env.state()` | `Dict` with full internal state |
| *(validation)* | `GET /validate` | `{"openenv_valid": true, ...}` |

All models are strict Pydantic v2:

```python
class Action(BaseModel):
    operation:  str
    parameters: Dict[str, Any] = {}

class Observation(BaseModel):
    task_id, step, dataset_info, columns, shape,
    missing_values, dtypes, duplicate_count,
    sample_rows, available_operations,
    task_description, message

class Reward(BaseModel):
    total, duplicate_score, missing_score,
    dtype_score, outlier_score, schema_score, penalty

class StepResult(BaseModel):
    observation: Observation
    reward:      Reward
    done:        bool
    info:        Dict
```

---

## 📁 Project Structure

```
Data-cleaning/
├── main.py                  # FastAPI server — all REST endpoints
├── environment.py           # DataCleaningEnv — reset/step/state + all ops
├── models.py                # Pydantic models — Action/Observation/Reward/StepResult
├── inference.py             # Baseline LLM agent — runs all tasks, outputs traces
├── openenv.yaml             # OpenEnv metadata — tasks, spaces, API config
├── Dockerfile               # Single-stage Python 3.10 container, port 7860
├── README.md                # This file
├── baseline_results.json    # Pre-run baseline scores (gpt-4o-mini)
└── datasets/
    ├── task_metadata.json   # Task configs, allowed ops, scoring weights
    ├── easy/
    │   ├── dirty.csv        # Employee dataset with duplicates + bad column names
    │   └── gold.csv         # Ground truth
    ├── medium/
    │   ├── dirty.csv        # Customer dataset with missing values + wrong types
    │   └── gold.csv
    ├── hard/
    │   ├── dirty.csv        # Orders dataset with all issue types
    │   └── gold.csv
    └── expert/
        ├── dirty.csv        # Sales data with case normalization + full pipeline
        └── gold.csv
```

---

## 🔬 Research Applications

This environment can be used to study:

- **Sequence learning in RL** — Can agents learn optimal operation ordering from reward signals alone?
- **Dense vs. sparse rewards** — Compare agent performance with/without the sequence penalties disabled
- **Tool use planning** — Does the agent build a plan before acting, or does it react greedily?
- **Generalization** — Train on easy/medium tasks, evaluate zero-shot on expert
- **LLM agent benchmarking** — Evaluate frontier models on a deterministic, math-graded task with no LLM-judge subjectivity

---

## 🔗 Links

| Resource | URL |
|---|---|
| 🚀 Live Demo (UI) | https://thorodin103-data-cleaning.hf.space/ui |
| 🤗 HuggingFace Space | https://huggingface.co/spaces/thorodin103/Data-cleaning/tree/main |
| ✅ OpenEnv Validate | https://thorodin103-data-cleaning.hf.space/validate |
| 📖 API Docs (Swagger) | https://thorodin103-data-cleaning.hf.space/docs |
| 📋 Task List | https://thorodin103-data-cleaning.hf.space/tasks |
| 🏅 Leaderboard | https://thorodin103-data-cleaning.hf.space/leaderboard |

---

## 📜 License

MIT — free for research and commercial use.

---

*Built for the OpenEnv Hackathon. Powered by FastAPI + Pandas + Pydantic + Hugging Face Spaces.*
