import json
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple
from models import Action, Observation, Reward, StepResult, TaskInfo

AVAILABLE_OPERATIONS = [
    "remove_duplicates",
    "fill_missing_mean",
    "fill_missing_mode",
    "fill_missing_median",
    "fix_dtype",
    "remove_outliers",
    "rename_columns",
    "validate_schema",
    "finish"
]

class DataCleaningEnv:
    """
    OpenEnv-compliant Data Cleaning Environment.
    Agent must clean dirty datasets step by step.
    """

    def __init__(self, task_id: str = "easy_dedup_rename"):
        self.task_id = task_id
        self.current_df: Optional[pd.DataFrame] = None
        self.gold_df: Optional[pd.DataFrame] = None
        self.task_meta: Dict = {}
        self.step_count: int = 0
        self.done: bool = False
        self.max_steps: int = 10
        self.reward_history = []
        self.actions_taken: List[str] = []  # Track sequence of actions
        self._load_task_metadata()

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    def _load_task_metadata(self):
        with open("datasets/task_metadata.json", "r") as f:
            all_meta = json.load(f)

        mapping = {
            "easy_dedup_rename":     "easy",
            "medium_missing_dtype":  "medium",
            "hard_full_pipeline":    "hard",
            "expert_sales_pipeline": "expert"
        }
        key = mapping.get(self.task_id, "easy")
        self.task_meta = all_meta[key]
        self.max_steps = self.task_meta["max_steps"]

    def _load_datasets(self):
        mapping = {
            "easy_dedup_rename":     "easy",
            "medium_missing_dtype":  "medium",
            "hard_full_pipeline":    "hard",
            "expert_sales_pipeline": "expert"
        }
        folder = mapping.get(self.task_id, "easy")
        self.current_df = pd.read_csv(f"datasets/{folder}/dirty.csv")
        self.gold_df    = pd.read_csv(f"datasets/{folder}/gold.csv")

    def _add_random_variation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add small random variations to prevent memorization."""
        df = df.copy()
        rng = np.random.default_rng()

        for col in df.columns:
            # Randomly shuffle some numeric values slightly
            num_series = pd.to_numeric(df[col], errors="coerce")
            if num_series.notna().sum() > 2:
                # Add tiny noise to 1-2 numeric cells
                idx = df.index[num_series.notna()].tolist()
                if len(idx) >= 2:
                    pick = rng.choice(idx, size=min(2, len(idx)), replace=False)
                    for i in pick:
                        orig = num_series[i]
                        noise = rng.integers(-3, 4)
                        df.at[i, col] = str(int(orig + noise)) if isinstance(df.at[i, col], str) else orig + noise

        # Randomly shuffle row order (keeps same data, different order)
        df = df.sample(frac=1, random_state=rng.integers(0, 9999)).reset_index(drop=True)
        return df

    def _get_observation(self, message: str = "") -> Observation:
        df = self.current_df
        missing = {col: int(df[col].isnull().sum()) for col in df.columns}
        dtypes  = {col: str(df[col].dtype) for col in df.columns}
        sample  = df.head(3).fillna("NULL").to_dict(orient="records")

        return Observation(
            task_id=self.task_id,
            step=self.step_count,
            dataset_info={
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "has_duplicates": bool(df.duplicated().any()),
                "has_missing": bool(df.isnull().any().any()),
            },
            columns=list(df.columns),
            shape=[len(df), len(df.columns)],
            missing_values=missing,
            dtypes=dtypes,
            duplicate_count=int(df.duplicated().sum()),
            sample_rows=sample,
            available_operations=self.task_meta.get(
                "operations_allowed", AVAILABLE_OPERATIONS
            ),
            task_description=self.task_meta.get("description", ""),
            message=message
        )

    def _compute_sequence_penalty(self) -> float:
        """
        Penalize illogical action sequences.

        Optimal sequence:
        1. remove_duplicates (clean up redundant data)
        2. fix_dtype (understand structure)
        3. fill_missing (based on correct types)
        4. remove_outliers (after understanding distribution)
        5. validate_schema (final check)

        Penalties:
        - Doing operations out of order: -0.05 per violation
        - Repeating same operation: -0.02 per repeat
        """
        OPTIMAL_ORDER = [
            "remove_duplicates",
            "fix_dtype",
            "fill_missing_mean",
            "fill_missing_mode",
            "fill_missing_median",
            "remove_outliers",
            "validate_schema",
        ]

        penalty = 0.0
        last_order_idx = -1
        prev_action = None

        # Track positions of operations in optimal order
        for action in self.actions_taken:
            if action == "finish":
                continue

            # Penalize repeated same actions (doing same thing twice in a row)
            if action == prev_action and action not in ["fill_missing_mean", "fill_missing_mode", "fill_missing_median", "remove_outliers"]:
                penalty += 0.02

            # Find this action in optimal order
            if action in OPTIMAL_ORDER:
                current_idx = OPTIMAL_ORDER.index(action)

                # Penalize out-of-order operations
                if current_idx < last_order_idx:
                    # Going backward in sequence (e.g., doing remove_duplicates after fill_missing)
                    penalty += 0.08

                last_order_idx = current_idx
            else:
                # Unknown operation
                penalty += 0.01

            prev_action = action

        return min(0.25, penalty)  # Cap penalty at 0.25

    def _compute_reward(self) -> Reward:
        df   = self.current_df
        gold = self.gold_df
        scoring = self.task_meta.get("scoring", {})

        dup_score     = 0.0
        missing_score = 0.0
        dtype_score   = 0.0
        outlier_score = 0.0
        schema_score  = 0.0
        penalty       = 0.0

        # ── Duplicate score ──────────────────────────────────────────
        if "duplicate_score" in scoring:
            gold_rows = len(gold)
            curr_rows = len(df)
            if curr_rows == gold_rows:
                dup_score = 1.0
            elif curr_rows < gold_rows:
                dup_score = max(0.0, curr_rows / gold_rows)
            else:
                excess = curr_rows - gold_rows
                dup_score = max(0.0, 1.0 - (excess / gold_rows))

        # ── Missing value score ──────────────────────────────────────
        if "missing_score" in scoring:
            # Count original missing values from dirty dataset
            import os
            mapping = {
                "easy_dedup_rename":     "easy",
                "medium_missing_dtype":  "medium",
                "hard_full_pipeline":    "hard",
                "expert_sales_pipeline": "expert"
            }
            folder = mapping.get(self.task_id, "easy")
            orig_df = pd.read_csv(f"datasets/{folder}/dirty.csv")
            orig_missing = int(orig_df.isnull().sum().sum())
            curr_missing = int(df.isnull().sum().sum())
            gold_missing = int(gold.isnull().sum().sum())

            if orig_missing <= gold_missing:
                missing_score = 1.0
            else:
                filled_needed = orig_missing - gold_missing
                filled_done   = orig_missing - curr_missing
                filled_done   = max(0, filled_done)
                missing_score = filled_done / filled_needed
                missing_score = max(0.0, min(1.0, missing_score))

        # ── Dtype score ──────────────────────────────────────────────
        if "dtype_score" in scoring:
            common_cols = [c for c in gold.columns if c in df.columns]
            if common_cols:
                matches = sum(
                    1 for c in common_cols
                    if str(df[c].dtype) == str(gold[c].dtype)
                )
                dtype_score = matches / len(common_cols)

        # ── Outlier score ────────────────────────────────────────────
        if "outlier_score" in scoring:
            num_cols = gold.select_dtypes(include=[np.number]).columns
            scores = []
            for col in num_cols:
                if col not in df.columns:
                    continue
                gold_mean = gold[col].mean()
                gold_std  = gold[col].std() + 1e-9
                curr_col  = pd.to_numeric(df[col], errors="coerce").dropna()
                outliers  = ((curr_col - gold_mean).abs() > 3 * gold_std).sum()
                col_score = max(0.0, 1.0 - outliers / (len(curr_col) + 1e-9))
                scores.append(col_score)
            outlier_score = float(np.mean(scores)) if scores else 0.0

        # ── Schema score ─────────────────────────────────────────────
        if "schema_score" in scoring:
            gold_cols = list(gold.columns)
            curr_cols = list(df.columns)
            matched   = sum(1 for c in gold_cols if c in curr_cols)
            schema_score = matched / len(gold_cols) if gold_cols else 0.0

        # ── Penalty for too many steps + sequence violations ─────────────
        step_ratio = self.step_count / self.max_steps
        base_penalty = 0.0
        if step_ratio > 0.8:
            base_penalty = 0.05

        # Add sequence-based penalty
        sequence_penalty = self._compute_sequence_penalty()
        penalty = base_penalty + sequence_penalty

        # ── Weighted total ───────────────────────────────────────────
        weights = {
            "duplicate_score": scoring.get("duplicate_score", 0.0),
            "missing_score":   scoring.get("missing_score",   0.0),
            "dtype_score":     scoring.get("dtype_score",     0.0),
            "outlier_score":   scoring.get("outlier_score",   0.0),
            "schema_score":    scoring.get("schema_score",    0.0),
        }
        component_scores = {
            "duplicate_score": dup_score,
            "missing_score":   missing_score,
            "dtype_score":     dtype_score,
            "outlier_score":   outlier_score,
            "schema_score":    schema_score,
        }
        total = sum(
            component_scores[k] * w
            for k, w in weights.items()
        ) - penalty
        # Clamp to strictly (0, 1) excluding endpoints for grader compliance
        total = max(0.0001, min(0.9999, total))

        return Reward(
            total=round(total, 4),
            duplicate_score=round(dup_score, 4),
            missing_score=round(missing_score, 4),
            dtype_score=round(dtype_score, 4),
            outlier_score=round(outlier_score, 4),
            schema_score=round(schema_score, 4),
            penalty=round(penalty, 4)
        )

    # ─────────────────────────────────────────
    # OPERATIONS
    # ─────────────────────────────────────────

    def _op_remove_duplicates(self, params: Dict) -> str:
        before = len(self.current_df)
        subset = params.get("subset", None)
        self.current_df = self.current_df.drop_duplicates(subset=subset)
        self.current_df = self.current_df.reset_index(drop=True)
        removed = before - len(self.current_df)
        return f"Removed {removed} duplicate rows. Rows: {before} -> {len(self.current_df)}"

    def _op_fill_missing(self, params: Dict) -> str:
        col      = params.get("column")
        strategy = params.get("strategy", "mean")
        messages = []

        cols_to_fill = [col] if col else list(self.current_df.columns)
        for c in cols_to_fill:
            if self.current_df[c].isnull().sum() == 0:
                continue

            # For mean/median strategies, only apply to columns that are already numeric or can be safely converted
            if strategy in ["mean", "median"]:
                # Check if column is already numeric
                if pd.api.types.is_numeric_dtype(self.current_df[c]):
                    # Column is already numeric, apply strategy directly
                    if strategy == "mean":
                        fill_val = self.current_df[c].mean()
                        self.current_df[c] = self.current_df[c].fillna(round(fill_val, 2))
                    elif strategy == "median":
                        fill_val = self.current_df[c].median()
                        self.current_df[c] = self.current_df[c].fillna(fill_val)
                    messages.append(f"{c}->{strategy}")
                else:
                    # Try to convert to numeric - if it works for most values, use it
                    numeric = pd.to_numeric(self.current_df[c], errors="coerce")
                    # Only apply if at least 50% of non-null values are numeric
                    non_null_count = self.current_df[c].notna().sum()
                    numeric_count = numeric.notna().sum()
                    if non_null_count > 0 and numeric_count / non_null_count >= 0.5:
                        if strategy == "mean":
                            fill_val = numeric.mean()
                            self.current_df[c] = numeric.fillna(round(fill_val, 2))
                        elif strategy == "median":
                            fill_val = numeric.median()
                            self.current_df[c] = numeric.fillna(fill_val)
                        messages.append(f"{c}->{strategy}")
                    # else: skip this column as it's not suitable for numeric strategy

            elif strategy == "mode":
                mode_vals = self.current_df[c].mode()
                if len(mode_vals) == 0:
                    continue
                fill_val = mode_vals.iloc[0]
                self.current_df[c] = self.current_df[c].fillna(fill_val)
                messages.append(f"{c}->{strategy}")
            elif strategy == "ffill":
                self.current_df[c] = self.current_df[c].ffill()
                messages.append(f"{c}->ffill")
            else:
                # Custom fill value
                fill_val = strategy
                self.current_df[c] = self.current_df[c].fillna(fill_val)
                messages.append(f"{c}->{strategy}")

        return f"Filled missing values: {', '.join(messages)}" if messages else "No missing values to fill"

    def _op_fix_dtype(self, params: Dict) -> str:
        col    = params.get("column")
        dtype  = params.get("dtype", "auto")
        messages = []

        cols_to_fix = [col] if col else list(self.current_df.columns)
        for c in cols_to_fix:
            try:
                if dtype == "int" or dtype == "auto":
                    converted = pd.to_numeric(self.current_df[c], errors="coerce")
                    if converted.notna().all():
                        self.current_df[c] = converted.astype(int)
                        messages.append(f"{c}->int")
                elif dtype == "float":
                    self.current_df[c] = pd.to_numeric(
                        self.current_df[c], errors="coerce"
                    )
                    messages.append(f"{c}->float")
                elif dtype == "str":
                    self.current_df[c] = self.current_df[c].astype(str)
                    messages.append(f"{c}->str")
            except Exception as e:
                messages.append(f"{c}->failed({e})")

        return f"Fixed dtypes: {', '.join(messages)}"

    def _op_remove_outliers(self, params: Dict) -> str:
        col    = params.get("column")
        method = params.get("method", "iqr")
        before = len(self.current_df)
        messages = []

        cols = [col] if col else list(
            self.current_df.select_dtypes(include=[np.number]).columns
        )
        for c in cols:
            series = pd.to_numeric(self.current_df[c], errors="coerce")
            if method == "iqr":
                Q1  = series.quantile(0.25)
                Q3  = series.quantile(0.75)
                IQR = Q3 - Q1
                mask = (series >= Q1 - 1.5 * IQR) & (series <= Q3 + 1.5 * IQR)
                self.current_df = self.current_df[mask | series.isna()]
            elif method == "zscore":
                mean = series.mean()
                std  = series.std() + 1e-9
                mask = ((series - mean).abs() <= 3 * std)
                self.current_df = self.current_df[mask | series.isna()]
            self.current_df = self.current_df.reset_index(drop=True)
            messages.append(c)

        removed = before - len(self.current_df)
        return f"Removed {removed} outlier rows from: {', '.join(messages)}"

    def _op_rename_columns(self, params: Dict) -> str:
        mapping = params.get("mapping", {})
        if not mapping:
            # Auto snake_case
            new_names = {
                col: col.lower().replace(" ", "_")
                for col in self.current_df.columns
            }
            self.current_df = self.current_df.rename(columns=new_names)
            return f"Auto renamed columns to snake_case: {list(new_names.values())}"
        self.current_df = self.current_df.rename(columns=mapping)
        return f"Renamed columns: {mapping}"

    def _op_validate_schema(self, params: Dict) -> str:
        gold_cols = list(self.gold_df.columns)
        curr_cols = list(self.current_df.columns)
        missing   = [c for c in gold_cols if c not in curr_cols]
        extra     = [c for c in curr_cols if c not in gold_cols]
        if not missing and not extra:
            return "Schema valid! All columns match gold standard."
        msg = []
        if missing:
            msg.append(f"Missing columns: {missing}")
        if extra:
            msg.append(f"Extra columns: {extra}")
        return "Schema issues: " + " | ".join(msg)

    # ─────────────────────────────────────────
    # OPENENV API
    # ─────────────────────────────────────────

    def reset(self) -> StepResult:
        self._load_datasets()
        self.step_count   = 0
        self.done         = False
        self.reward_history = []
        self.actions_taken = []  # Reset action history

        obs    = self._get_observation("Environment reset. Start cleaning!")
        reward = Reward(total=0.0001)

        return StepResult(
            observation=obs,
            reward=reward,
            done=False,
            info={"task_id": self.task_id, "max_steps": self.max_steps}
        )

    def step(self, action: Action) -> StepResult:
        if self.done:
            obs = self._get_observation("Episode already done. Call reset().")
            return StepResult(
                observation=obs,
                reward=Reward(total=0.0001),
                done=True,
                info={"warning": "Episode already done"}
            )

        self.step_count += 1
        op     = action.operation
        params = action.parameters
        message = ""

        # Map fill_missing_* operations to fill_missing with strategy parameter
        if op.startswith("fill_missing_"):
            strategy = op.replace("fill_missing_", "")
            params["strategy"] = strategy
            op = "fill_missing"

        # ── Route operation ──────────────────────────────────────────
        try:
            if op == "remove_duplicates":
                message = self._op_remove_duplicates(params)
            elif op == "fill_missing":
                message = self._op_fill_missing(params)
            elif op == "fix_dtype":
                message = self._op_fix_dtype(params)
            elif op == "remove_outliers":
                message = self._op_remove_outliers(params)
            elif op == "rename_columns":
                message = self._op_rename_columns(params)
            elif op == "validate_schema":
                message = self._op_validate_schema(params)
            elif op == "finish":
                message = "Agent called finish."
                self.done = True
            else:
                message = f"Unknown operation: {op}. No changes made."
        except Exception as e:
            message = f"Operation failed: {str(e)}"

        # ── Track action for sequence penalties ───────────────────────────
        self.actions_taken.append(action.operation)

        # ── Check max steps ──────────────────────────────────────────
        if self.step_count >= self.max_steps:
            self.done = True
            message += " | Max steps reached."

        reward = self._compute_reward()
        self.reward_history.append(reward.total)
        obs = self._get_observation(message)

        return StepResult(
            observation=obs,
            reward=reward,
            done=self.done,
            info={
                "step": self.step_count,
                "operation": action.operation,
                "reward_history": self.reward_history
            }
        )

    def state(self) -> Dict[str, Any]:
        if self.current_df is None:
            return {"status": "not initialized — call reset() first"}
        return {
            "task_id":        self.task_id,
            "step":           self.step_count,
            "done":           self.done,
            "shape":          list(self.current_df.shape),
            "columns":        list(self.current_df.columns),
            "missing_values": self.current_df.isnull().sum().to_dict(),
            "duplicate_count": int(self.current_df.duplicated().sum()),
            "reward_history": self.reward_history,
            "dtypes":         {c: str(t) for c, t in self.current_df.dtypes.items()}
        }