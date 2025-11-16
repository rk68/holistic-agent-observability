from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..schemas import FailureType, ReasoningStep


@dataclass(slots=True)
class BehaviourConfig:
    timeout_seconds: float = 120.0
    loop_min_repetitions: int = 3
    invalid_param_min_repetitions: int = 2


def _normalise_text(text: str) -> str:
    return " ".join(text.lower().split())


def detect_behaviour_failures(
    reasoning_trace: List[ReasoningStep],
    *,
    config: BehaviourConfig | None = None,
) -> List[FailureType]:
    cfg = config or BehaviourConfig()
    failures: List[FailureType] = []

    if not reasoning_trace:
        return failures

    # ------------------------------------------------------------------
    # A. Timeouts
    # ------------------------------------------------------------------
    start_ts = float(reasoning_trace[0].get("timestamp", 0.0))
    end_ts = float(reasoning_trace[-1].get("timestamp", start_ts))
    duration = max(0.0, end_ts - start_ts)

    has_final_answer = any(step.get("kind") == "final_answer" for step in reasoning_trace)

    if duration > cfg.timeout_seconds and not has_final_answer:
        failures.append(
            FailureType(
                code="BEHAVIOUR_TIMEOUT",
                severity="HIGH" if duration > cfg.timeout_seconds * 2 else "MEDIUM",
                description=(
                    f"Trace duration {duration:.1f}s exceeded timeout threshold "
                    f"({cfg.timeout_seconds:.0f}s) without a final answer."
                ),
                step_ids=[step.get("id") or "unknown" for step in reasoning_trace],
            )
        )

    # ------------------------------------------------------------------
    # B. Loops (identical tool calls, thoughts, or errors)
    # ------------------------------------------------------------------
    actions = [s for s in reasoning_trace if s.get("kind") == "action"]
    observations = [s for s in reasoning_trace if s.get("kind") == "observation"]
    llm_steps = [s for s in reasoning_trace if s.get("kind") == "llm_output"]

    # Identical tool calls
    action_groups: Dict[Tuple[str, str], List[str]] = {}
    for step in actions:
        tool_name = str(step.get("tool_name") or "")
        tool_input = step.get("tool_input")
        try:
            key_input = json.dumps(tool_input, sort_keys=True, default=str)
        except Exception:
            key_input = str(tool_input)
        key = (tool_name, key_input)
        action_groups.setdefault(key, []).append(step.get("id") or "unknown")

    for (tool_name, _), step_ids in action_groups.items():
        if len(step_ids) >= cfg.loop_min_repetitions:
            failures.append(
                FailureType(
                    code="BEHAVIOUR_LOOP_TOOL",
                    severity="MEDIUM",
                    description=(
                        f"Identical tool call to {tool_name or 'unknown'} repeated "
                        f"{len(step_ids)} times without evident progress."
                    ),
                    step_ids=step_ids,
                )
            )

    # Identical thoughts / LLM outputs
    thought_groups: Dict[str, List[str]] = {}
    for step in llm_steps:
        content = _normalise_text(str(step.get("content") or ""))
        if not content:
            continue
        thought_groups.setdefault(content, []).append(step.get("id") or "unknown")

    for norm_text, step_ids in thought_groups.items():
        if len(step_ids) >= cfg.loop_min_repetitions:
            failures.append(
                FailureType(
                    code="BEHAVIOUR_LOOP_THOUGHTS",
                    severity="MEDIUM",
                    description=(
                        f"Identical model output/thought repeated {len(step_ids)} times: "
                        f"{norm_text[:160]}"
                    ),
                    step_ids=step_ids,
                )
            )

    # Repeated tool errors
    error_groups: Dict[Tuple[str, str], List[str]] = {}
    for step in observations:
        error_text = (step.get("error") or "").strip()
        content = (step.get("content") or "").lower()
        if not error_text and any(tok in content for tok in ("invalid", "unknown", "not found", "error")):
            error_text = content[:200]
        if not error_text:
            continue

        tool_name = str(step.get("tool_name") or "")
        key = (tool_name, error_text)
        error_groups.setdefault(key, []).append(step.get("id") or "unknown")

    for (tool_name, error_text), step_ids in error_groups.items():
        if len(step_ids) >= cfg.loop_min_repetitions:
            failures.append(
                FailureType(
                    code="BEHAVIOUR_LOOP_ERRORS",
                    severity="HIGH",
                    description=(
                        f"Repeated tool errors for {tool_name or 'unknown'}: "
                        f"'{error_text[:160]}' occurred {len(step_ids)} times."
                    ),
                    step_ids=step_ids,
                )
            )

    # ------------------------------------------------------------------
    # C. No Final Answer
    # ------------------------------------------------------------------
    if not has_final_answer:
        failures.append(
            FailureType(
                code="BEHAVIOUR_NO_FINAL_ANSWER",
                severity="MEDIUM",
                description="Trace completed without a final_answer step.",
                step_ids=[step.get("id") or "unknown" for step in reasoning_trace],
            )
        )

    # ------------------------------------------------------------------
    # D. Stuck on invalid params (repeated errors + no final answer)
    # ------------------------------------------------------------------
    has_repeated_errors = any(
        len(step_ids) >= cfg.invalid_param_min_repetitions for step_ids in error_groups.values()
    )
    if has_repeated_errors and not has_final_answer:
        failures.append(
            FailureType(
                code="BEHAVIOUR_STUCK_INVALID_PARAMS",
                severity="HIGH",
                description=(
                    "Agent repeatedly called tools with invalid parameters and "
                    "never produced a final answer."
                ),
                step_ids=[step.get("id") or "unknown" for step in reasoning_trace],
            )
        )

    return failures
