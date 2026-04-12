---
title: Data Cleaning OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "3.10"
app_file: main.py
pinned: false
---

# 🧹 CleanifyAI — Data Cleaning OpenEnv

<div align="center">

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/cleanify-ai/Data-cleaning)
[![GitHub](https://img.shields.io/badge/GitHub-ReverseCoder1%2FCleanifyAI-black?logo=github)](https://github.com/ReverseCoder1/CleanifyAI)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://huggingface.co/spaces/cleanify-ai/Data-cleaning)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)](https://fastapi.tiangolo.com)

**A reinforcement-learning environment where AI agents learn to clean real-world messy datasets — step by step.**

*Scaler × OpenEnv Hackathon Submission*

[🚀 Live API](https://thorodin103-data-cleaning-openenv.hf.space) · [📖 Swagger Docs](https://thorodin103-data-cleaning-openenv.hf.space/docs) · [🤗 HuggingFace](https://huggingface.co/spaces/cleanify-ai/Data-cleaning)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Tasks](#-tasks)
- [Operations Reference](#-operations-reference)
- [Reward & Scoring System](#-reward--scoring-system)
- [API Reference](#-api-reference)
- [Inference Script](#-inference-script)
- [Data Models](#-data-models)
- [Datasets](#-datasets)
- [Baseline Scores](#-baseline-scores)
- [License](#-license)

---

## 🌟 Overview

**CleanifyAI** is a fully OpenEnv-compliant environment that challenges AI agents to autonomously clean messy, real-world datasets through a sequence of structured operations. It mimics professional data engineering pipelines and rewards agents that apply operations in the correct, logical order.

| Property | Value |
|---|---|
| **Environment Name** | `data-cleaning-openenv` |
| **Tasks** | 4 (Easy, Medium, Hard, Expert) |
| **Operations** | 9 (dedup, fill, dtype fix, outlier removal, rename, validate, finish) |
| **Scoring** | Weighted multi-component, strictly in `(0, 1)` |
| **API** | OpenEnv-compliant REST via FastAPI |
| **Framework** | Python 3.10, FastAPI, Pandas, NumPy |
| **Inference** | OpenAI-compatible LLM client |
| **Deployed at** | `https://thorodin103-data-cleaning-openenv.hf.space` |

> ⚠️ **Score Constraint**: The Scaler platform rejects scores of exactly `0.0` or `1.0`. All scoring paths in this codebase clamp strictly to `(0.0001, 0.9999)`.

---

## 📁 Project Structure

```
CleanifyAI/
│
├── inference.py                  # 🤖 LLM agent — emits [START]/[STEP]/[END] stdout lines
├── environment.py                # 🏋️ Core OpenEnv environment & reward computation
├── models.py                     # 📦 Pydantic models: Action, Observation, Reward, StepResult
├── main.py                       # 🌐 FastAPI server with all REST endpoints
├── Dockerfile                    # 🐳 Python 3.10-slim container, port 7860
├── openenv.yaml                  # 📄 OpenEnv spec manifest
├── pyproject.toml                # 📦 Python dependency config
├── uv.lock                       # 🔒 Locked dependency versions
│
├── datasets/
│   ├── task_metadata.json        # ⚙️ Per-task config (steps, operations, scoring weights)
│   ├── easy/
│   │   ├── dirty.csv             # 🗑️ Employee dataset with duplicates + bad column names
│   │   └── gold.csv              # ✅ Gold standard cleaned version
│   ├── medium/
│   │   ├── dirty.csv             # 🗑️ Customer dataset with missing values + wrong dtypes
│   │   └── gold.csv              # ✅ Gold standard
│   ├── hard/
│   │   ├── dirty.csv             # 🗑️ Orders dataset requiring full pipeline
│   │   └── gold.csv              # ✅ Gold standard
│   └── expert/
│       ├── dirty.csv             # 🗑️ Sales dataset — strict operation order required
│       └── gold.csv              # ✅ Gold standard
│
├── static/
│   └── index.html                # 🖥️ Web UI for interactive exploration
│
└── server/
    └── app.py                    # 🔧 Server initialization module
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- A Hugging Face account (`HF_TOKEN`)
- An OpenAI-compatible API endpoint and model

---

### Local Development

**1. Clone the repository**
```bash
git clone https://github.com/ReverseCoder1/CleanifyAI.git
cd CleanifyAI
```

**2. Install dependencies**
```bash
pip install fastapi==0.104.1 uvicorn==0.24.0 pydantic==2.5.0 \
            pandas==2.1.3 numpy==1.26.2 openai>=2.7.2 \
            pyyaml==6.0.1 python-dotenv==1.0.0
```

**3. Create a `.env` file**
```env
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
HF_TOKEN=your_hugging_face_token_here
```

**4. Start the FastAPI server**
```bash
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

- **API:** http://localhost:7860
- **Swagger UI:** http://localhost:7860/docs

---

### Docker Deployment

```bash
# Build
docker build -t cleanify-ai .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=gpt-4o-mini \
  -e API_BASE_URL=https://api.openai.com/v1 \
  cleanify-ai
```

---

### Run the Inference Agent

```bash
python inference.py
```

Runs the LLM agent across all 3 hackathon tasks and streams hackathon-spec log lines to stdout.

---

## 🎯 Tasks

Four progressively complex tasks. The hackathon evaluates **easy**, **medium**, and **hard**. Expert is available for extended benchmarking.

| Task ID | Difficulty | Max Steps | Key Operations | Scoring |
|---|---|---|---|---|
| `easy_dedup_rename` | ⭐ Easy | 10 | `remove_duplicates`, `rename_columns` | dup 50% + schema 50% |
| `medium_missing_dtype` | ⭐⭐ Medium | 15 | `fill_missing_*`, `fix_dtype` | missing 50% + dtype 50% |
| `hard_full_pipeline` | ⭐⭐⭐ Hard | 20 | Full pipeline | 20% × 5 components |
| `expert_sales_pipeline` | ⭐⭐⭐⭐ Expert | 25 | All 9 ops in strict order | Weighted (schema 25%) |

---

### ⭐ Easy — `easy_dedup_rename`

**Dataset:** Employee records (`emp_id`, `emp_name`, `dept`, `salary`, `age`)

**Dirty conditions:**
- Duplicate rows
- Column names with spaces and inconsistent casing (`EMP ID`, `DEP T`, `SAL ARY`)

**Optimal sequence:**
```
remove_duplicates → rename_columns → finish
```

**Scoring:** `duplicate_score × 0.5 + schema_score × 0.5`

---

### ⭐⭐ Medium — `medium_missing_dtype`

**Dataset:** Customer records (`customer_id`, `age`, `salary`, `gender`, `purchases`, `region`, `joined_date`)

**Dirty conditions:**
- NaN values in `age`, `salary`, `gender`, `region`
- `salary` stored as `object` instead of `float`

**Optimal sequence:**
```
fill_missing_mean  (numeric columns)
fill_missing_mode  (categorical columns)
fix_dtype          → finish
```

**Scoring:** `missing_score × 0.5 + dtype_score × 0.5`

---

### ⭐⭐⭐ Hard — `hard_full_pipeline`

**Dataset:** Orders (`order_id`, `product`, `quantity`, `price`, `customer_id`, `status`, `order_date`, `rating`)

**Dirty conditions:**
- Duplicate order entries
- Missing `quantity` and `price` values
- `quantity` stored as object type
- Extreme price outliers

**Optimal sequence:**
```
remove_duplicates → fill_missing_* → fix_dtype → remove_outliers → validate_schema → finish
```

**Scoring:** `duplicate × 0.2 + missing × 0.2 + dtype × 0.2 + outlier × 0.2 + schema × 0.2`

---

### ⭐⭐⭐⭐ Expert — `expert_sales_pipeline`

**Dataset:** Sales transactions — highest complexity, penalises out-of-order operations heavily.

**Optimal sequence (strictly enforced):**
```
remove_duplicates → rename_columns → fill_missing_mode → fix_dtype → remove_outliers → validate_schema → finish
```

**Scoring:** `duplicate × 0.15 + missing × 0.20 + dtype × 0.20 + outlier × 0.20 + schema × 0.25`

---

## 🔧 Operations Reference

All operations are invoked via JSON actions sent to `POST /step/{task_id}`.

### `remove_duplicates`
```json
{"operation": "remove_duplicates", "parameters": {}}
```
Drops exact duplicate rows using `pandas.drop_duplicates()`. Resets the index after removal.
- Optional parameter: `"subset": ["col1", "col2"]` — deduplicate on specific columns only

---

### `fill_missing_mean`
```json
{"operation": "fill_missing_mean", "parameters": {}}
```
Fills NaN values in numeric columns with the column mean. Skips non-numeric columns to avoid type errors.
- Optional parameter: `"column": "col_name"` — target a single column

---

### `fill_missing_mode`
```json
{"operation": "fill_missing_mode", "parameters": {}}
```
Fills NaN values with the most frequent value (mode). Works for both numeric and categorical columns.
- Optional parameter: `"column": "col_name"`

---

### `fill_missing_median`
```json
{"operation": "fill_missing_median", "parameters": {}}
```
Fills NaN values in numeric columns with the column median. More robust to outliers than mean.
- Optional parameter: `"column": "col_name"`

---

### `fix_dtype`
```json
{"operation": "fix_dtype", "parameters": {"dtype": "auto"}}
```
Attempts to convert columns to the most appropriate type.
- `"dtype": "auto"` — tries `int` then `float`, skips if conversion fails
- `"dtype": "int"` — convert to integer
- `"dtype": "float"` — convert to float
- `"dtype": "str"` — convert to string
- Optional parameter: `"column": "col_name"`

---

### `remove_outliers`
```json
{"operation": "remove_outliers", "parameters": {"method": "iqr"}}
```
Removes rows where numeric values fall outside the outlier fence.
- `"method": "iqr"` — IQR method: removes values outside `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`
- `"method": "zscore"` — Z-score method: removes values beyond ±3σ
- Optional parameter: `"column": "col_name"` — target a single numeric column

---

### `rename_columns`
```json
{"operation": "rename_columns", "parameters": {}}
```
Auto-renames all columns to `snake_case` (lowercase, spaces → underscores).
- Optional parameter: `"mapping": {"Old Name": "new_name"}` — explicit rename map

---

### `validate_schema`
```json
{"operation": "validate_schema", "parameters": {}}
```
Compares current column names against the gold dataset schema.
- Returns missing columns (in gold but not current)
- Returns extra columns (in current but not in gold)
- Returns a success message if schemas match perfectly

---

### `finish`
```json
{"operation": "finish", "parameters": {}}
```
Signals the agent is done. Triggers final reward computation and ends the episode immediately. **Always call this when cleaning is complete.**

---

## 📊 Reward & Scoring System

Reward is computed after every step and returned as a `Reward` object. The total is a weighted sum of components minus penalties, **clamped strictly to `(0.0001, 0.9999)`**.

### Score Components

| Component | What It Measures | How It's Calculated |
|---|---|---|
| `duplicate_score` | Row count vs gold dataset | Proportional to excess/deficit rows |
| `missing_score` | Missing values filled vs gold | Fraction of needed fills completed |
| `dtype_score` | Column types match gold | Matched columns ÷ total columns |
| `outlier_score` | Numeric values within 3σ of gold mean | Per-column average, then mean across columns |
| `schema_score` | Column names match gold schema | Matched column names ÷ gold column count |
| `penalty` | Step efficiency + operation order | See sequence penalty below |

---

### Sequence Penalty

The optimal operation order is:
```
remove_duplicates → fix_dtype → fill_missing_* → remove_outliers → validate_schema
```

Penalties for deviations:

| Violation | Penalty |
|---|---|
| Out-of-order operation | −0.08 |
| Repeated operation (non-fill/outlier) | −0.02 |
| Unknown operation | −0.01 |
| Using >80% of allowed steps | −0.05 |
| **Maximum total penalty** | **−0.25** |

---

### Score Clamping (Critical)

The Scaler grader rejects scores of exactly `0.0` or `1.0`. The following clamping is enforced at every level:

```python
# environment.py — _compute_reward()
def _sc(v):
    return round(max(0.0001, min(0.9999, float(v))), 4)

# Applied to ALL Reward fields: total, duplicate_score, missing_score, etc.
return Reward(
    total=_sc(total),
    duplicate_score=_sc(dup_score),
    ...
)
```

```python
# inference.py — every printed reward
def _clamp(v: float) -> float:
    return max(0.01, min(0.99, float(v)))

# [STEP] and [END] lines both use _clamp() before formatting
```

---

## 🌐 API Reference

**Base URL:** `https://thorodin103-data-cleaning-openenv.hf.space`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Reset environment (body: `{"task_id": "..."}`) |
| `POST` | `/reset/{task_id}` | Reset specific task environment |
| `POST` | `/step` | Take action (body: `{"task_id": "...", "operation": "...", "parameters": {}}`) |
| `POST` | `/step/{task_id}` | Take action in specific task |
| `GET` | `/state` | Get current environment state |
| `GET` | `/state/{task_id}` | Get state for specific task |
| `GET` | `/tasks` | List all tasks with full metadata |
| `GET` | `/validate` | Run OpenEnv spec validation across all tasks |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive Swagger UI |
| `POST` | `/leaderboard/submit` | Submit a score entry |
| `GET` | `/leaderboard` | Get current leaderboard rankings |

---

### Example: Reset a task
```bash
curl -X POST https://thorodin103-data-cleaning-openenv.hf.space/reset/easy_dedup_rename
```
```json
{
  "observation": {
    "task_id": "easy_dedup_rename",
    "step": 0,
    "columns": ["EMP ID", "EMP NAME", "DEP T", "SAL ARY", "AGE"],
    "duplicate_count": 5,
    "missing_values": {"EMP ID": 0, "EMP NAME": 0, ...},
    "message": "Environment reset. Start cleaning!"
  },
  "reward": {"total": 0.0001},
  "done": false
}
```

---

### Example: Take a step
```bash
curl -X POST https://thorodin103-data-cleaning-openenv.hf.space/step/easy_dedup_rename \
  -H "Content-Type: application/json" \
  -d '{"operation": "remove_duplicates", "parameters": {}}'
```
```json
{
  "observation": {"step": 1, "duplicate_count": 0, "message": "Removed 5 duplicate rows. Rows: 20 -> 15"},
  "reward": {"total": 0.4821, "duplicate_score": 0.9999, "schema_score": 0.0001},
  "done": false
}
```

---

### Example: Validate the environment
```bash
curl https://thorodin103-data-cleaning-openenv.hf.space/validate
```
```json
{
  "openenv_valid": true,
  "tasks": {
    "easy_dedup_rename":    {"status": "passed"},
    "medium_missing_dtype": {"status": "passed"},
    "hard_full_pipeline":   {"status": "passed"},
    "expert_sales_pipeline":{"status": "passed"}
  }
}
```

---

## 🤖 Inference Script

`inference.py` is the hackathon submission entry point. It runs an LLM agent across all tasks and emits structured stdout lines that the platform parser reads.

### Required Stdout Format

> The format below is **mandatory**. The platform parser reads these exact line types.

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
```

**Rules:**
- One `[START]` line at episode begin
- One `[STEP]` line per step, immediately after `env.step()` returns
- One `[END]` line after episode end — **always emitted, even on exception** (via `finally` block)
- `reward` and `rewards` formatted to **2 decimal places**
- `done` and `success` are lowercase: `true` or `false`
- `score=` field in `[END]` is **mandatory** — its absence causes Task Validation failure
- `error` is the raw error string, or `null` if none

**Example output:**
```
[START] task=easy_dedup_rename env=data-cleaning-openenv model=gpt-4o-mini
[STEP] step=1 action=remove_duplicates reward=0.48 done=false error=null
[STEP] step=2 action=rename_columns reward=0.96 done=false error=null
[STEP] step=3 action=finish reward=0.96 done=true error=null
[END] success=true steps=3 score=0.96 rewards=0.48,0.96,0.96
```

---

### Agent Loop

For each task the agent follows this loop:

1. Call `env.reset()` to initialise the episode
2. Build a prompt from the observation (shape, columns, missing values, dtypes, sample rows)
3. Send prompt to LLM via OpenAI-compatible client
4. Parse the JSON response into an `Action`
5. Call `env.step(action)` and record the reward
6. Emit a `[STEP]` line
7. Repeat until `done=true` or `MAX_STEPS` (20) reached
8. Compute `score = average(rewards)`, clamped to `(0, 1)`
9. Emit `[END]` line via `finally` block

---

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | *(required)* | Hugging Face / API key |

---

## 📦 Data Models

### `Action`
```json
{
  "operation": "remove_duplicates",
  "parameters": {}
}
```
- `operation` — one of the 9 valid operations
- `parameters` — operation-specific options (`column`, `strategy`, `method`, `dtype`, `mapping`, `subset`)

---

### `Observation`
```json
{
  "task_id": "easy_dedup_rename",
  "step": 1,
  "dataset_info": {"total_rows": 15, "has_duplicates": false, "has_missing": false},
  "columns": ["emp_id", "emp_name", "dept", "salary", "age"],
  "shape": [15, 5],
  "missing_values": {"emp_id": 0, "emp_name": 0},
  "dtypes": {"emp_id": "int64", "emp_name": "object"},
  "duplicate_count": 0,
  "sample_rows": [{"emp_id": 101, "emp_name": "Alice", ...}],
  "available_operations": ["remove_duplicates", "rename_columns", "finish"],
  "task_description": "Clean an employee dataset by...",
  "message": "Removed 5 duplicate rows."
}
```

---

### `Reward`
```json
{
  "total": 0.4821,
  "duplicate_score": 0.9999,
  "missing_score": 0.0001,
  "dtype_score": 0.0001,
  "outlier_score": 0.0001,
  "schema_score": 0.0001,
  "penalty": 0.0
}
```
All values are clamped to `(0.0001, 0.9999)`.

---

### `StepResult`
```json
{
  "observation": { ... },
  "reward": { ... },
  "done": false,
  "info": {
    "step": 1,
    "operation": "remove_duplicates",
    "reward_history": [0.4821]
  }
}
```

---

## 🗃️ Datasets

Each task has a paired `dirty.csv` and `gold.csv`. The dirty file is loaded at reset; the gold file is used as the scoring reference throughout the episode.

### Easy — Employee Dataset
| Property | Value |
|---|---|
| Dirty columns | `EMP ID`, `EMP NAME`, `DEP T`, `SAL ARY`, `AGE` |
| Gold columns | `emp_id`, `emp_name`, `dept`, `salary`, `age` |
| Issues | Duplicate rows, space-separated column names |
| Rows | ~20 dirty → ~15 gold after dedup |

### Medium — Customer Dataset
| Property | Value |
|---|---|
| Columns | `customer_id`, `age`, `salary`, `gender`, `purchases`, `region`, `joined_date` |
| Issues | NaN in `age`, `salary`, `gender`, `region`; `salary` as `object` instead of `float` |
| Rows | ~30, no duplicates |

### Hard — Orders Dataset
| Property | Value |
|---|---|
| Columns | `order_id`, `product`, `quantity`, `price`, `customer_id`, `status`, `order_date`, `rating` |
| Issues | Duplicate orders, missing `quantity`/`price`, wrong dtypes, price outliers |
| Rows | ~50 dirty, full pipeline required |

### Expert — Sales Dataset
| Property | Value |
|---|---|
| Issues | All of the above plus column naming problems |
| Unique challenge | Operations must be applied in strict optimal order — out-of-order is penalised −0.08 per violation |

---


## 📈 Baseline Scores

Baseline agent: **gpt-4o-mini** (from `openenv.yaml`)

| Task | Score |
|---|---|
| `easy_dedup_rename` | **0.9900** |
| `medium_missing_dtype` | **0.7000** |
| `hard_full_pipeline` | **0.6636** |
| **Average** | **0.7845** |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

Built for the **Scaler × OpenEnv Hackathon**

🔗 [GitHub](https://github.com/ReverseCoder1/CleanifyAI) · [HuggingFace Space](https://huggingface.co/spaces/cleanify-ai/Data-cleaning) · [Live API Docs](https://thorodin103-data-cleaning-openenv.hf.space/docs)

*CleanifyAI — making data clean, one step at a time.*

</div>