from __future__ import annotations

from typing import Dict, List

from .detectors import (
    detect_behaviour_failures,
    detect_data_leakage,
    detect_safety_violations,
    detect_tool_misuse,
)
from .schemas import DataArtefact, FailureSummary, FailureType, ReasoningStep


def analyze_trace(
    trace_id: str,
    *,
    reasoning_trace: List[ReasoningStep],
    data_context: Dict[str, DataArtefact],
) -> FailureSummary:
    """Run all failure detectors over a single trace.

    This function is designed to consume exactly what the agent logs into
    Langfuse trace metadata under the ``reasoning_trace`` and
    ``data_context`` keys.
    """

    safety_failures: List[FailureType] = detect_safety_violations(reasoning_trace)
    leakage_failures: List[FailureType] = detect_data_leakage(reasoning_trace, data_context)
    tool_failures: List[FailureType] = detect_tool_misuse(reasoning_trace)
    behaviour_failures: List[FailureType] = detect_behaviour_failures(reasoning_trace)

    failure_types: List[FailureType] = [
        *safety_failures,
        *leakage_failures,
        *tool_failures,
        *behaviour_failures,
    ]

    behavioural_signals = {
        "num_steps": len(reasoning_trace),
        "num_artefacts": len(data_context),
        "num_failures": len(failure_types),
        "num_safety_failures": len(safety_failures),
        "num_leakage_failures": len(leakage_failures),
        "num_tool_misuse_failures": len(tool_failures),
        "num_behaviour_failures": len(behaviour_failures),
    }

    summary: FailureSummary = {
        "trace_id": trace_id,
        "has_failure": bool(failure_types),
        "failure_types": failure_types,
        "behavioural_signals": behavioural_signals,
    }
    return summary
