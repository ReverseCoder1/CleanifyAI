---
title: Data Cleaning OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - data-cleaning
  - reinforcement-learning
  - agent
  - real-world
---

# 🧹 Data Cleaning OpenEnv

An OpenEnv-compliant environment where AI agents learn to clean
messy real-world datasets step by step.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/thorodin103/data-cleaning-openenv)

---

## 🌍 Environment Description

Data cleaning is one of the most common and time-consuming tasks
in real-world data workflows. Data engineers and analysts spend
up to 80% of their time cleaning data before it can be used.

This environment simulates that exact challenge — an agent receives
a dirty dataset and must apply a sequence of cleaning operations
to match a gold standard output. Each operation provides partial
reward signal, enabling reinforcement learning agents to learn
incrementally.

---

## 🎯 Tasks

| Task ID | Difficulty | Description | Max Steps |
|---------|-----------|-------------|-----------|
| easy_dedup_rename | Easy | Remove duplicates + rename columns to snake_case | 10 |
| medium_missing_dtype | Medium | Fill missing values + fix data types | 15 |
| hard_full_pipeline | Hard | Full pipeline: duplicates + missing + dtypes + outliers + schema | 20 |

### Easy Task
Agent receives an employee dataset with duplicate rows and
poorly formatted column names. Must remove duplicates and
rename columns to snake_case format.

**Scoring:** duplicate_score (0.5) + schema_score (0.5)

### Medium Task
Agent receives a customer dataset with missing values in
multiple columns and wrong data types. Must fill missing
values using correct strategies (mean/median/mode) and
fix data types.

**Scoring:** missing_score (0.5) + dtype_score (0.5)

### Hard Task
Agent receives an orders dataset with all types of issues:
duplicates, missing values, wrong types, outliers. Must
run a complete cleaning pipeline in the right sequence.

**Scoring:** All 5 components weighted equally (0.2 each)

---

## 👁️ Observation Space
```json
{
  "task_id": "string — current task identifier",
  "step": "integer — current step number",
  "dataset_info": "object — summary of dataset state",
  "columns": "list — column names",
  "shape": "list — [rows, columns]",
  "missing_values": "object — missing count per column",
  "dtypes": "object — data type per column",
  "duplicate_count": "integer — number of duplicate rows",
  "sample_rows": "list — first 3 rows as preview",
  "available_operations": "list — valid operations",
  "task_description": "string — what agent must do",
  "message": "string — feedback from last action"
}
```

---

## ⚡ Action Space
```json
{
  "operation": "one of: remove_duplicates | fill_missing | fix_dtype | remove_outliers | rename_columns | validate_schema | finish",
  "parameters": {
    "column": "optional — target column name",
    "strategy": "optional — mean | median | mode | ffill",
    "dtype": "optional — int | float | str | auto",
    "method": "optional — iqr | zscore",
    "mapping": "optional — column rename mapping dict"
  }
}
```

---

## 🏆 Reward Function

Rewards are computed after every step providing dense signal:

| Component | Description |
|-----------|-------------|
| duplicate_score | How close row count is to gold standard |
| missing_score | Proportion of missing values filled correctly |
| dtype_score | Proportion of columns with correct data types |
| outlier_score | How close numeric distributions are to gold |
| schema_score | Proportion of column names matching gold |
| penalty | Small penalty for using too many steps |

**Total reward = weighted sum of components (0.0 to 1.0)**

---

## 🚀 Setup & Usage

### Run with Docker
```bash
docker build -t data-cleaning-openenv .
docker run -p 7860:7860 data-cleaning-openenv
```

### API Usage
```python
import requests

# Reset environment
response = requests.post(
    "http://localhost:7860/reset/easy_dedup_rename"
)
obs = response.json()

# Take action
action = {
    "operation": "remove_duplicates",
    "parameters": {}
}
response = requests.post(
    "http://localhost:7860/step/easy_dedup_rename",
    json=action
)
result = response.json()
print(result["reward"]["total"])

# Get state
state = requests.get(
    "http://localhost:7860/state/easy_dedup_rename"
).json()
```

### Run Baseline Inference
```bash
export HF_TOKEN=your_token_here
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

python inference.py
```

---

## 📊 Baseline Scores

Scores produced by `meta-llama/Llama-3.3-70B-Instruct`:

| Task | Score |
|------|-------|
| easy_dedup_rename | 1.0000 |
| medium_missing_dtype | 0.6643 |
| hard_full_pipeline | 0.8386 |
| **Average** | **0.8343** |

---

## 📁 Project Structure
```
data-cleaning-openenv/
├── main.py              # FastAPI server
├── environment.py       # Core env logic
├── models.py            # Pydantic models
├── inference.py         # Baseline script
├── openenv.yaml         # OpenEnv metadata
├── Dockerfile           # Container config
├── README.md            # This file
└── datasets/
    ├── task_metadata.json
    ├── easy/
    │   ├── dirty.csv
    │   └── gold.csv
    ├── medium/
    │   ├── dirty.csv
    │   └── gold.csv
    └── hard/
        ├── dirty.csv
        └── gold.csv
```

---

## 🔗 Links

- [Hugging Face Space](https://huggingface.co/spaces/thorodin103/data-cleaning-openenv)
- [OpenEnv Spec](https://github.com/openenv/openenv)
