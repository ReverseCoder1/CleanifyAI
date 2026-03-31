
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class Action(BaseModel):
    """Action that the agent can take in the environment."""
    operation: str = Field(
        ...,
        description=(
            "Operation to perform. One of: "
            "remove_duplicates, fill_missing, fix_dtype, "
            "remove_outliers, rename_columns, validate_schema, finish"
        )
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the operation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "operation": "fill_missing",
                "parameters": {
                    "column": "age",
                    "strategy": "mean"
                }
            }
        }


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str = Field(..., description="Current task identifier")
    step: int = Field(..., description="Current step number")
    dataset_info: Dict[str, Any] = Field(
        ...,
        description="Info about current dataset state"
    )
    columns: List[str] = Field(
        ...,
        description="Column names in dataset"
    )
    shape: List[int] = Field(
        ...,
        description="[rows, columns] of dataset"
    )
    missing_values: Dict[str, int] = Field(
        ...,
        description="Count of missing values per column"
    )
    dtypes: Dict[str, str] = Field(
        ...,
        description="Data types of each column"
    )
    duplicate_count: int = Field(
        ...,
        description="Number of duplicate rows"
    )
    sample_rows: List[Dict[str, Any]] = Field(
        ...,
        description="First 3 rows of dataset as preview"
    )
    available_operations: List[str] = Field(
        ...,
        description="List of valid operations agent can use"
    )
    task_description: str = Field(
        ...,
        description="Natural language description of what agent must do"
    )
    message: str = Field(
        default="",
        description="Feedback message from last action"
    )


class Reward(BaseModel):
    """Reward signal returned after each step."""
    total: float = Field(
        ...,
        description="Total reward this step (0.0 to 1.0)"
    )
    duplicate_score: float = Field(
        default=0.0,
        description="Score for duplicate removal"
    )
    missing_score: float = Field(
        default=0.0,
        description="Score for handling missing values"
    )
    dtype_score: float = Field(
        default=0.0,
        description="Score for fixing data types"
    )
    outlier_score: float = Field(
        default=0.0,
        description="Score for outlier removal"
    )
    schema_score: float = Field(
        default=0.0,
        description="Score for schema validation"
    )
    penalty: float = Field(
        default=0.0,
        description="Penalty for bad actions"
    )


class StepResult(BaseModel):
    """Full result returned by step()."""
    observation: Observation
    reward: Reward
    done: bool = Field(..., description="Whether episode is complete")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra info for debugging"
    )


class TaskInfo(BaseModel):
    """Metadata about a task."""
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    operations_allowed: List[str]
