from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from .config import AgentSettings, load_settings


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
    if not settings.enable_tracing:
        yield None
        return

    client = _configure_tracer(settings)
    handler = LangfuseCallbackHandler(public_key=settings.langfuse_public_key)
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
