from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar
from collections.abc import Sequence

from langchain_core.messages import AIMessage
from langfuse import Langfuse, get_client

try:  # Prefer modern Langfuse layout
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - compatibility shim
    try:
        # Some versions may expose integrations under a different namespace
        from langfuse.integrations.langchain import (  # type: ignore[no-redef]
            CallbackHandler as LangfuseCallbackHandler,
        )
    except Exception:
        LangfuseCallbackHandler = None  # type: ignore[assignment]


F = TypeVar("F", bound=Callable[..., Any])

try:  # Prefer Langfuse v3+ layout
    from langfuse.decorators import observe  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - compatibility shim
    try:
        # Older SDKs may expose observe at the top level
        from langfuse import observe  # type: ignore[no-redef]
    except Exception:  # Fallback: no-op decorator so tracing never blocks app startup

        def observe(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
            def _decorator(func: F) -> F:
                return func

            return _decorator

from .callbacks import ReasoningTraceCallback
from .config import AgentSettings, load_settings
from .leak_judge import run_leak_judge
from .state import create_initial_state, get_current_state, set_current_state
from .visibility import VisibilityTracker, set_current_tracker


def _extract_final_answer(result: Any) -> str | None:
    """Best-effort extraction of the final AI message text from an agent result."""

    if not isinstance(result, dict):
        return None

    messages = result.get("messages")
    if not isinstance(messages, Sequence):
        return None

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)

    return None


def _configure_tracer(settings: AgentSettings) -> Langfuse:
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        raise ValueError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set when tracing is enabled."
        )

    client = Langfuse(
        host=settings.langfuse_host,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
    )

    return client


@contextmanager
def _langfuse_context(settings: AgentSettings):
    if not settings.enable_tracing or LangfuseCallbackHandler is None:
        yield None
        return

    client = _configure_tracer(settings)
    handler: Optional[Any] = None
    try:
        handler = LangfuseCallbackHandler(public_key=settings.langfuse_public_key)
    except Exception:  # pragma: no cover - if handler cannot be constructed, skip tracing
        handler = None
    try:
        yield handler
    finally:
        client.flush()


def traced_graph(graph, settings: AgentSettings | None = None):
    settings = settings or load_settings()

    @observe(name=settings.agent_name)
    def _run(payload: dict[str, Any], *, config: dict[str, Any] | None):
        tracker = VisibilityTracker()
        set_current_tracker(tracker)
        initial_messages = []
        raw_messages = payload.get("messages") if isinstance(payload, dict) else None
        if isinstance(raw_messages, Sequence):
            initial_messages = list(raw_messages)
        state = create_initial_state(initial_messages)
        set_current_state(state)
        try:
            with _langfuse_context(settings) as handler:
                updated_config = dict(config or {})
                callbacks = list(updated_config.get("callbacks", []))
                callbacks.append(ReasoningTraceCallback())
                if handler:
                    callbacks.append(handler)
                if callbacks:
                    updated_config["callbacks"] = callbacks

                result = graph.invoke(payload, config=updated_config) if updated_config else graph.invoke(payload)

            # Best-effort LLM-as-a-judge for the final answer when tracing is enabled.
            if settings.enable_tracing:
                try:
                    final_answer = _extract_final_answer(result) or ""
                    if final_answer:
                        judge = run_leak_judge(final_answer, tracker, settings)
                    else:
                        judge = None

                    client = get_client()

                    if judge is not None:
                        span_metadata: dict[str, Any] = {
                            "visible_data": tracker.snapshot_visible_ids(),
                            "final_answer": final_answer,
                            "leak_judge": judge,
                        }
                        client.update_current_span(metadata=span_metadata)

                    state_for_trace = get_current_state()
                    if state_for_trace is not None:
                        trace_metadata: dict[str, Any] = {
                            "reasoning_trace": state_for_trace.get("reasoning_trace", []),
                            "data_context": state_for_trace.get("data_context", {}),
                        }
                        client.update_current_trace(metadata=trace_metadata)
                except Exception:
                    # Never break the agent if leak judging fails.
                    pass

            return result
        finally:
            set_current_tracker(None)
            set_current_state(None)

    class _TracedGraph:
        def __init__(self, inner):
            self._inner = inner

        def invoke(self, payload: dict[str, Any], *, config: dict[str, Any] | None = None):
            return _run(payload, config=config)

        def __getattr__(self, item: str):
            return getattr(self._inner, item)

    return _TracedGraph(graph)
