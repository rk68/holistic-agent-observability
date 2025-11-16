from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

ReasoningKind = Literal["thought", "action", "observation", "llm_output", "final_answer"]


class ReasoningStep(TypedDict):
    id: str
    step_index: int
    kind: ReasoningKind
    timestamp: float
    content: str
    tool_name: str | None
    tool_input: Dict[str, Any] | None
    tool_output: Dict[str, Any] | None
    error: str | None
    visible_data_ids: List[str]
    metadata: Dict[str, Any]


SensitivityLevel = Literal["PUBLIC", "INTERNAL", "SENSITIVE", "HIGHLY_SENSITIVE"]


class DataArtefact(TypedDict):
    id: str
    source: str
    payload: Dict[str, Any]
    sensitivity: SensitivityLevel
    fields_sensitivity: Dict[str, SensitivityLevel]


class FailureType(TypedDict):
    code: str  # e.g. "DATA_LEAKAGE"
    severity: str  # "LOW" | "MEDIUM" | "HIGH"
    description: str
    step_ids: List[str]


class FailureSummary(TypedDict):
    trace_id: str
    has_failure: bool
    failure_types: List[FailureType]
    behavioural_signals: Dict[str, Any]
