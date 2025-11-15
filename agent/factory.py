from __future__ import annotations

from typing import Iterable, Sequence

import os
import requests
from langchain_aws import ChatBedrockConverse
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from .config import AgentSettings, load_settings
from .tools import DEFAULT_TOOLS
from .tracing import traced_graph


class AWSBedrockGatewayChatModel(BaseChatModel):
    """Chat model that proxies requests through a simple AWS Bedrock gateway.

    The gateway is expected to expose an HTTP API compatible with the example
    provided by the user:

        POST AWS_BEDROCK_API_ENDPOINT
        Headers: {"Content-Type": "application/json", "X-Api-Token": ...}
        JSON: {
          "team_id": ..., "model": ..., "messages": [...], "max_tokens": ...
        }
    """

    def __init__(
        self,
        *,
        api_endpoint: str,
        api_key: str,
        team_id: str,
        model: str,
        max_tokens: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._api_endpoint = api_endpoint
        self._api_key = api_key
        self._team_id = team_id
        self._model = model
        self._max_tokens = max_tokens

    @property
    def _llm_type(self) -> str:  # pragma: no cover - metadata only
        return "aws_bedrock_gateway"

    def _convert_messages(self, messages: Sequence[BaseMessage]) -> list[dict[str, str]]:
        payload_messages: list[dict[str, str]] = []
        for message in messages:
            role: str
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, ChatMessage):
                role = message.role
            else:
                # Fallback for other message types (e.g. system)
                role = getattr(message, "role", "user") or "user"

            content = message.content
            if not isinstance(content, str):
                content = str(content)
            payload_messages.append({"role": role, "content": content})
        return payload_messages

    def _extract_text(self, data: object) -> str:
        """Best-effort extraction of response text from the gateway payload."""
        if isinstance(data, dict):
            # Common shapes we might encounter.
            if isinstance(data.get("output"), str):
                return data["output"]
            if isinstance(data.get("content"), str):
                return data["content"]
            if isinstance(data.get("message"), str):
                return data["message"]

            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg["content"]

            messages = data.get("messages")
            if isinstance(messages, list) and messages:
                first_msg = messages[0]
                if isinstance(first_msg, dict) and isinstance(first_msg.get("content"), str):
                    return first_msg["content"]

        raise ValueError(f"Unexpected response from AWS Bedrock gateway: {data!r}")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        payload = {
            "team_id": self._team_id,
            "model": self._model,
            "messages": self._convert_messages(messages),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }

        headers = {
            "Content-Type": "application/json",
            "X-Api-Token": self._api_key,
        }

        response = requests.post(self._api_endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        text = self._extract_text(data)

        ai_message = AIMessage(content=text)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])


def _resolve_model(settings: AgentSettings) -> BaseChatModel:
    backend = settings.agent_backend.strip().lower()

    if backend in {"aws", "aws_bedrock", "bedrock"}:
        # Use the official LangChain ChatBedrockConverse integration, which fully
        # supports tool calling via .bind_tools. AWS credentials and region are
        # picked up from the standard environment variables or AWS config.
        model_id = settings.aws_agent_model or settings.model
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

        # If the user has provided an API key under AWS_BEDROCK_API_KEY in their
        # .env, surface it to the AWS SDK via AWS_BEARER_TOKEN_BEDROCK so
        # ChatBedrockConverse can authenticate without requiring separate
        # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
        if not os.getenv("AWS_BEARER_TOKEN_BEDROCK") and settings.aws_bedrock_api_key:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = settings.aws_bedrock_api_key

        return ChatBedrockConverse(
            model_id=model_id,
            region_name=region,
            temperature=settings.temperature,
        )

    if backend == "aws_gateway":
        if not settings.aws_bedrock_api_endpoint:
            raise ValueError("AWS gateway backend selected but AWS_BEDROCK_API_ENDPOINT is not set.")
        if not settings.aws_bedrock_api_key:
            raise ValueError("AWS gateway backend selected but AWS_BEDROCK_API_KEY is not set.")
        if not settings.aws_bedrock_team_id:
            raise ValueError("AWS gateway backend selected but AWS_BEDROCK_TEAM_ID is not set.")

        model_name = settings.aws_agent_model or settings.model
        return AWSBedrockGatewayChatModel(
            api_endpoint=settings.aws_bedrock_api_endpoint,
            api_key=settings.aws_bedrock_api_key,
            team_id=settings.aws_bedrock_team_id,
            model=model_name,
            max_tokens=1024,
            temperature=settings.temperature,
        )

    # Default to local Ollama backend. Enable reasoning mode so that supported
    # models emit their thinking process into additional_kwargs["reasoning_content"],
    # which will be captured in Langfuse traces separately from the final
    # answer text.
    return ChatOllama(
        model=settings.model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        reasoning=True,
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
    settings = settings or load_settings()
    graph = build_agent(settings=settings)
    state = graph.invoke({"messages": _to_messages(question, settings=settings)})
    return _extract_final_text(state.get("messages", []))


def _to_messages(prompt: str, *, settings: AgentSettings) -> list[BaseMessage]:
    messages: list[BaseMessage] = []

    # Enable Granite 3.2 "thinking" behaviour by inserting a control message when
    # the configured model is a Granite 3.2 variant. This causes the model to emit
    # an explicit thought process alongside its final answer, which will be
    # captured in Langfuse traces.
    if settings.model.startswith("granite3.2"):
        messages.append(ChatMessage(role="control", content="thinking"))

    messages.append(HumanMessage(content=prompt))
    return messages


def _extract_final_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""
