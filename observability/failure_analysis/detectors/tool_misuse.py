from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..schemas import FailureType, ReasoningStep


@dataclass(slots=True)
class ToolMisuseConfig:
    min_repeated_errors: int = 2

    # Simple intent-to-expected-tools mapping for our banking demo.
    intent_tool_map: Dict[str, Tuple[str, ...]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:  # pragma: no cover - trivial initialisation
        if self.intent_tool_map is None:
            self.intent_tool_map = {
                "balance": ("banking.get_account_balance",),
                "transactions": ("banking.get_recent_transactions",),
                "spend": ("banking.get_recent_transactions",),
                "product": ("banking.recommend_products",),
                "card": ("banking.recommend_products",),
            }


def _normalise_input(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _infer_intents(reasoning_trace: List[ReasoningStep]) -> List[str]:
    """Infer coarse intents from early LLM outputs / final answers."""

    texts: List[str] = []
    for step in reasoning_trace:
        if step.get("kind") in {"llm_output", "final_answer"}:
            content = (step.get("content") or "").lower()
            if content:
                texts.append(content)
        if len(texts) >= 3:
            break

    blob = "\n".join(texts)
    intents: List[str] = []
    if not blob:
        return intents

    lowered = blob.lower()
    if "balance" in lowered or "available" in lowered:
        intents.append("balance")
    if "transaction" in lowered or "spend" in lowered or "spending" in lowered:
        intents.append("transactions")
    if "card" in lowered or "product" in lowered or "offer" in lowered or "recommend" in lowered:
        intents.append("product")
    return list(dict.fromkeys(intents))  # deduplicate while preserving order


def detect_tool_misuse(
    reasoning_trace: List[ReasoningStep],
    *,
    config: ToolMisuseConfig | None = None,
) -> List[FailureType]:
    """Detect tool misuse patterns based on action/observation steps.

    Misuse types:
    - Wrong entity misuse (same tool, different identifiers)
    - Wrong tool for inferred user intent (expected tool never called)
    - Repeated invalid parameters (same tool+input repeatedly failing)
    """

    cfg = config or ToolMisuseConfig()
    failures: List[FailureType] = []

    # Collect action and observation steps.
    actions: List[ReasoningStep] = [s for s in reasoning_trace if s.get("kind") == "action"]
    observations: List[ReasoningStep] = [s for s in reasoning_trace if s.get("kind") == "observation"]

    # ------------------------------------------------------------------
    # A. Wrong entity misuse (per-tool inconsistent identifiers)
    # ------------------------------------------------------------------
    per_tool_entities: Dict[str, List[Tuple[str, str]]] = {}

    for step in actions:
        tool_name = str(step.get("tool_name") or "")
        if not tool_name:
            continue

        tool_input = step.get("tool_input") or {}
        identifiers: List[Tuple[str, str]] = []
        if isinstance(tool_input, dict):
            for key in ("account_identifier", "account_id", "customer_id"):
                value = tool_input.get(key)
                if isinstance(value, str) and value.strip():
                    identifiers.append((key, value.strip()))

        if not identifiers:
            continue

        per_tool_entities.setdefault(tool_name, []).extend(identifiers)

    for tool_name, ids in per_tool_entities.items():
        distinct_values = {v for _, v in ids}
        if len(distinct_values) <= 1:
            continue

        step_ids = [s.get("id") or "unknown" for s in actions if s.get("tool_name") == tool_name]
        description = (
            f"Tool {tool_name} was called with multiple distinct identifiers: "
            + ", ".join(sorted(distinct_values))
        )
        failures.append(
            FailureType(
                code="TOOL_MISUSE_WRONG_ENTITY",
                severity="MEDIUM",
                description=description,
                step_ids=step_ids,
            )
        )

    # ------------------------------------------------------------------
    # B. Wrong tool for the inferred intent
    # ------------------------------------------------------------------
    intents = _infer_intents(reasoning_trace)
    used_tools = {str(a.get("tool_name") or "") for a in actions if a.get("tool_name")}

    for intent in intents:
        expected_tools = cfg.intent_tool_map.get(intent, ())
        if not expected_tools:
            continue
        if any(t in used_tools for t in expected_tools):
            continue

        description = (
            f"User intent appears to be '{intent}', but none of the expected tools "
            f"were called: {', '.join(expected_tools)}."
        )
        step_ids = [s.get("id") or "unknown" for s in actions]
        failures.append(
            FailureType(
                code="TOOL_MISUSE_WRONG_TOOL",
                severity="MEDIUM",
                description=description,
                step_ids=step_ids,
            )
        )

    # ------------------------------------------------------------------
    # C. Repeated invalid parameters (same tool+input, multiple errors)
    # ------------------------------------------------------------------
    error_groups: Dict[Tuple[str, str], List[str]] = {}

    for step in observations:
        error_text = (step.get("error") or "").strip()
        content = (step.get("content") or "").lower()
        if not error_text:
            # Heuristic: look for obvious error markers in content.
            if not any(tok in content for tok in ("invalid", "unknown", "not found", "error")):
                continue
            error_text = content[:200]

        tool_name = str(step.get("tool_name") or "")
        key_input = _normalise_input(step.get("tool_input"))
        key = (tool_name, key_input)
        error_groups.setdefault(key, []).append(step.get("id") or "unknown")

    for (tool_name, key_input), step_ids in error_groups.items():
        if len(step_ids) < cfg.min_repeated_errors:
            continue

        description = (
            f"Tool {tool_name or 'unknown'} was called repeatedly with the same "
            f"parameters that caused errors (occurrences={len(step_ids)})."
        )
        failures.append(
            FailureType(
                code="TOOL_MISUSE_INVALID_PARAMS",
                severity="HIGH",
                description=description,
                step_ids=step_ids,
            )
        )

    return failures
