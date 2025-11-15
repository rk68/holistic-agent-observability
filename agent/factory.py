from __future__ import annotations

from typing import Iterable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from .config import AgentSettings, load_settings
from .tools import DEFAULT_TOOLS
from .tracing import traced_graph


def _resolve_model(settings: AgentSettings) -> BaseChatModel:
    return ChatOllama(
        model=settings.model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
    )


def _normalize_tools(settings: AgentSettings) -> Iterable[BaseTool | Runnable]:
    tools: list[BaseTool | Runnable] = []
    if settings.enable_math_tools and "math" in DEFAULT_TOOLS:
        tools.append(DEFAULT_TOOLS["math"])
    if settings.enable_resource_tool and "read_file" in DEFAULT_TOOLS:
        tools.append(DEFAULT_TOOLS["read_file"])

    for name, tool in DEFAULT_TOOLS.items():
        if name not in {"math", "read_file"}:
            tools.append(tool)

    return tools


def build_agent(*, settings: AgentSettings | None = None) -> Runnable:
    settings = settings or load_settings()
    model = _resolve_model(settings)
    tools = list(_normalize_tools(settings))
    graph = create_agent(
        model=model,
        tools=tools,
        system_prompt=settings.instructions,
        name=settings.agent_name,
    )
    return traced_graph(graph, settings=settings) if settings.enable_tracing else graph


def ask(question: str, *, settings: AgentSettings | None = None) -> str:
    graph = build_agent(settings=settings)
    state = graph.invoke({"messages": _to_messages(question)})
    return _extract_final_text(state.get("messages", []))


def _to_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _extract_final_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""
