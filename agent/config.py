from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

_DEFAULT_PROMPT: Final[str] = (
    "You are Glass Banking Agent, a trustworthy assistant helping retail customers understand "
    "their bank accounts and recent transactions.\n"
    "Follow this structure:\n"
    "1. Restate the customer request in one sentence.\n"
    "2. Provide the answer using only the data you have. Highlight balances, dates, and amounts clearly.\n"
    "3. Suggest safe next steps (e.g., reviewing statements, contacting support) if relevant.\n"
    "Stay concise, avoid speculation, and never expose sensitive data beyond what was provided."
)
_DEFAULT_OLLAMA_BASE_URL: Final[str] = "http://localhost:11434"
_DEFAULT_MODEL: Final[str] = "qwen3:4b"

_TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "y", "on"})


def _ensure_env_loaded() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env", override=False)
    load_dotenv(override=False)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


@dataclass(slots=True)
class AgentSettings:
    """Runtime configuration for the ReAct agent."""

    model: str
    # Primary backend for the chat model: e.g. "ollama" (default) or "aws" / "aws_bedrock".
    agent_backend: str
    temperature: float
    instructions: str
    agent_name: str
    enable_resource_tool: bool
    enable_math_tools: bool
    enable_tracing: bool
    langfuse_host: str | None
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    ollama_base_url: str
    # Optional AWS Bedrock-style gateway configuration
    aws_bedrock_api_endpoint: str | None
    aws_bedrock_api_key: str | None
    aws_bedrock_team_id: str | None
    aws_agent_model: str | None


@lru_cache(maxsize=1)
def load_settings() -> AgentSettings:
    """Load agent configuration from environment variables."""

    _ensure_env_loaded()

    instructions = os.getenv("AGENT_INSTRUCTIONS", _DEFAULT_PROMPT)

    # Decide backend: explicit AGENT_BACKEND wins; otherwise infer from AWS
    # configuration so that setting AWS_BEDROCK_API_ENDPOINT automatically
    # opts into the AWS gateway when AGENT_BACKEND is unset.
    backend = os.getenv("AGENT_BACKEND")
    if backend:
        agent_backend = backend
    elif os.getenv("AWS_BEDROCK_API_ENDPOINT"):
        # If a custom Bedrock-style gateway endpoint is configured but no
        # explicit backend is set, default to the gateway wrapper instead of
        # the native Bedrock integration.
        agent_backend = "aws_gateway"
    else:
        agent_backend = "ollama"
    return AgentSettings(
        model=os.getenv("AGENT_MODEL", _DEFAULT_MODEL),
        agent_backend=agent_backend,
        temperature=_get_float("AGENT_TEMPERATURE", 0.2),
        instructions=instructions,
        agent_name=os.getenv("AGENT_NAME", "glass-react-agent"),
        enable_resource_tool=_get_bool("AGENT_ENABLE_RESOURCE_TOOL", True),
        enable_math_tools=_get_bool("AGENT_ENABLE_MATH_TOOLS", True),
        enable_tracing=_get_bool("AGENT_ENABLE_TRACING", False),
        langfuse_host=os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL),
        aws_bedrock_api_endpoint=os.getenv("AWS_BEDROCK_API_ENDPOINT"),
        aws_bedrock_api_key=os.getenv("AWS_BEDROCK_API_KEY"),
        aws_bedrock_team_id=os.getenv("AWS_BEDROCK_TEAM_ID"),
        aws_agent_model=os.getenv("AWS_AGENT_MODEL"),
    )
