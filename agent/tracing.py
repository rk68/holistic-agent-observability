from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from codecarbon import EmissionsTracker
from langfuse import Langfuse

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

from .config import AgentSettings, load_settings


@dataclass
class CarbonMetrics:
    total_kg: float
    duration_seconds: float


@contextmanager
def carbon_tracker(settings: AgentSettings):
    tracker = EmissionsTracker(
        project_name=settings.agent_name,
        log_level="error",
        output_dir=".carbon",
        tracking_mode="process",
    )
    tracker.start()
    start = time.perf_counter()
    try:
        yield tracker
    finally:
        total = tracker.stop() or 0.0
        elapsed = time.perf_counter() - start
        tracker.final_metrics = CarbonMetrics(total_kg=total, duration_seconds=elapsed)


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
        with _langfuse_context(settings) as handler:
            if handler or config:
                updated_config = dict(config or {})
                if handler:
                    callbacks = list(updated_config.get("callbacks", []))
                    callbacks.append(handler)
                    updated_config["callbacks"] = callbacks
                return graph.invoke(payload, config=updated_config)
            return graph.invoke(payload)

    class _TracedGraph:
        def __init__(self, inner):
            self._inner = inner

        def invoke(self, payload: dict[str, Any], *, config: dict[str, Any] | None = None):
            return _run(payload, config=config)

        def __getattr__(self, item: str):
            return getattr(self._inner, item)

    return _TracedGraph(graph)
