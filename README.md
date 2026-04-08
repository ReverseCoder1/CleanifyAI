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

# Data Cleaning OpenEnv

An OpenEnv-compatible environment for training and evaluating agents on realistic tabular data-cleaning workflows.

## Live Demo and UI (Compulsory Link)

- UI: https://thorodin103-data-cleaning.hf.space/ui
- Hugging Face Space files: https://huggingface.co/spaces/thorodin103/Data-cleaning/tree/main

## What This Project Does

This project simulates real data-cleaning tasks where an agent must transform dirty CSV files into clean, gold-standard outputs.

The environment exposes step-based APIs so RL/LLM agents can:

1. Observe dataset state (shape, missing values, dtypes, duplicates, sample rows).
2. Choose a cleaning operation.
3. Receive dense reward feedback after each action.
4. Learn not only what to clean, but also the best sequence of operations.

## Why It Is Useful

- Mimics practical data engineering workflows.
- Provides measurable reward components for quality tracking.
- Includes sequence penalties, which encourage disciplined cleaning pipelines.
- Supports benchmarking across multiple task difficulties.

## Available Tasks

| Task ID | Difficulty | Goal | Max Steps |
|---|---|---|---|
| easy_dedup_rename | Easy | Remove duplicates and rename columns | 10 |
| medium_missing_dtype | Medium | Fill missing values and fix data types | 15 |
| hard_full_pipeline | Hard | End-to-end pipeline with outlier handling and schema checks | 20 |
| expert_sales_pipeline | Expert | Advanced sales cleaning workflow | 25 |

## Core Operations

- `remove_duplicates`
- `fill_missing_mean`
- `fill_missing_mode`
- `fill_missing_median`
- `fix_dtype`
- `remove_outliers`
- `rename_columns`
- `validate_schema`
- `finish`

## Reward Design

The final reward is a weighted score from multiple quality dimensions:

- duplicate quality
- missing-value quality
- dtype correctness
- outlier handling quality
- schema correctness

The environment also applies penalties for:

- out-of-order actions
- repeating unnecessary actions
- taking too many steps

This makes the environment suitable for both capability learning and process learning.

## API Endpoints

- `POST /reset/{task_id}`
- `POST /step/{task_id}`
- `GET /state/{task_id}`
- `GET /tasks`
- `GET /validate`
- `GET /health`
- `GET /ui`

## Quick Start

### Run with Docker

```bash
docker build -t data-cleaning-openenv .
docker run -p 7860:7860 data-cleaning-openenv
```

### Example API Call

```python
import requests

# Start an episode
obs = requests.post("http://localhost:7860/reset/easy_dedup_rename").json()

# Take one action
action = {"operation": "remove_duplicates", "parameters": {}}
result = requests.post("http://localhost:7860/step/easy_dedup_rename", json=action).json()

print(result["reward"]["total"])
```

### Run Baseline Inference

```bash
set HF_TOKEN=your_token_here
set MODEL_NAME=gpt-4o-mini
set API_BASE_URL=https://api.openai.com/v1
python inference.py
```

## Baseline Results

- easy_dedup_rename: `0.9900`
- medium_missing_dtype: `0.7000`
- hard_full_pipeline: `0.6636`
- average: `0.7845`

## Project Structure

```text
data-cleaning-openenv/
|- main.py
|- environment.py
|- models.py
|- inference.py
|- openenv.yaml
|- Dockerfile
|- README.md
|- datasets/
|  |- task_metadata.json
|  |- easy/
|  |- medium/
|  |- hard/
|  \- expert/
\- static/
   \- index.html
```

## Links

- Hugging Face Space files: https://huggingface.co/spaces/thorodin103/Data-cleaning/tree/main
- UI (required): https://thorodin103-data-cleaning.hf.space/ui
- OpenEnv specification: https://github.com/openenv/openenv
