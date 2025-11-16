from __future__ import annotations

from typing import Any, Dict, List, TypedDict
from contextvars import ContextVar

from observability.failure_analysis.schemas import DataArtefact, ReasoningStep


class AgentState(TypedDict):
    """Typed view of the agent's state for LangGraph-style workflows.

    This is not yet wired into the current LangChain agent graph but provides
    a shared schema for reasoning traces and data context that can be emitted
    to Langfuse and consumed by offline failure analysis tooling.
    """

    messages: List[Any]
    reasoning_trace: List[ReasoningStep]
    data_context: Dict[str, DataArtefact]
    done: bool


_CURRENT_STATE: ContextVar[AgentState | None] = ContextVar("agent_state", default=None)


def create_initial_state(messages: List[Any] | None = None) -> AgentState:
    """Create a fresh AgentState for a new agent run."""

    return {
        "messages": list(messages or []),
        "reasoning_trace": [],
        "data_context": {},
        "done": False,
    }


def set_current_state(state: AgentState | None) -> None:
    _CURRENT_STATE.set(state)


def get_current_state() -> AgentState | None:
    return _CURRENT_STATE.get()
