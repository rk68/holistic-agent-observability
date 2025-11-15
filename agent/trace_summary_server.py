from __future__ import annotations

import json
from typing import Any, List, Optional

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

from .config import load_settings


class ToolSummary(BaseModel):
    name: str
    summary: str


class TraceSummaryRequest(BaseModel):
    trace_id: str
    trace_name: Optional[str] = None
    observations: List[dict[str, Any]]


class TraceSummaryResponse(BaseModel):
    userQuestion: Optional[str] = None
    planningTools: List[str] = Field(default_factory=list)
    toolsExecuted: List[ToolSummary] = Field(default_factory=list)
    finalAnswer: Optional[str] = None


settings = load_settings()

logger = logging.getLogger("uvicorn.error")

_llm = ChatOllama(
    model=settings.model,
    base_url=settings.ollama_base_url,
    temperature=settings.temperature,
    reasoning=True,
)

_PROMPT_TEMPLATE = """You are an analysis assistant for LangGraph traces. Your job is to turn a raw
execution trace into a concise, human-interpretable summary of the execution path.

You will receive a JSON array of observations from a traced agent run. Each
observation describes one step in the execution (agent/model reasoning, tool
calls, tool outputs, etc.). Some model steps may contain a `reasoning_content`
field inside `additional_kwargs` that describes the model's chain-of-thought.

From this trace, extract the following FOUR fields:

1. `userQuestion` (string or null)
   - The original user question or request in plain language.

2. `planningTools` (array of strings)
   - Names of tools the model explicitly planned to call in its reasoning.
   - Example: ["banking.get_account_balance", "banking.get_recent_transactions"].

3. `toolsExecuted` (array of objects)
   - Each object must be of the form { "name": string, "summary": string }.
   - Include each distinct tool name that was actually EXECUTED in the trace.
   - In `summary`, briefly describe what the tool did and the key outcome
     (1–2 short clauses, suitable for non-technical stakeholders).

4. `finalAnswer` (string or null)
   - The final answer that the agent returned to the user (not the reasoning).

CRITICAL REQUIREMENTS:
- Base your summary ONLY on the information present in the trace JSON.
- Prefer the most recent model step when deciding the `finalAnswer`.
- Prefer explicit tool mentions in reasoning when deciding `planningTools`.
- Keep all text concise and free of markup (no Markdown).
- RETURN ONLY A SINGLE JSON OBJECT with the keys:
  {"userQuestion", "planningTools", "toolsExecuted", "finalAnswer"}.
  Do not include any additional commentary before or after the JSON.

Trace name: {trace_name}

Trace observations (JSON) are provided below as raw JSON. Do not treat them as a
format string; simply read them as data:
"""


app = FastAPI(title="Glass Agent Trace Summary API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local-only tool, safe to allow all origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_prompt(payload: TraceSummaryRequest) -> str:
    try:
        trace_json = json.dumps(payload.observations, ensure_ascii=False)
    except TypeError:
        # Best-effort fallback if something is not serialisable
        trace_json = json.dumps(
            [{"id": obs.get("id"), "type": obs.get("type"), "name": obs.get("name")}
             for obs in payload.observations],
            ensure_ascii=False,
        )
    trace_name = payload.trace_name or payload.trace_id
    base = _PROMPT_TEMPLATE.format(trace_name=trace_name)
    return f"{base}\n{trace_json}\n"


def _parse_model_json(raw: str) -> TraceSummaryResponse:
    """Parse the model's JSON response into a TraceSummaryResponse.

    We expect a single JSON object with the required keys. If the model returns
    extra text around the JSON, we try a simple substring extraction.

    Any parsing failure falls back to an empty TraceSummaryResponse so that
    callers never have to deal with HTTP 500s due to model formatting issues.
    """

    text = raw.strip()
    data: Any
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to recover JSON object from surrounding text
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return TraceSummaryResponse()
        snippet = text[start : end + 1]
        try:
            data = json.loads(snippet)
        except Exception:  # pragma: no cover - defensive
            return TraceSummaryResponse()

    if not isinstance(data, dict):
        return TraceSummaryResponse()

    user_question = data.get("userQuestion")
    planning_raw = data.get("planningTools")
    tools_raw = data.get("toolsExecuted")
    final_answer = data.get("finalAnswer")

    planning_tools: list[str] = []
    if isinstance(planning_raw, list):
        planning_tools = [str(item) for item in planning_raw]

    tools: list[ToolSummary] = []
    if isinstance(tools_raw, list):
        for item in tools_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            summary_raw = item.get("summary")
            summary = str(summary_raw).strip() if summary_raw is not None else name
            if not summary:
                summary = name
            tools.append(ToolSummary(name=name, summary=summary))

    if isinstance(user_question, str) and not user_question.strip():
        user_question = None
    if isinstance(final_answer, str) and not final_answer.strip():
        final_answer = None

    return TraceSummaryResponse(
        userQuestion=user_question if isinstance(user_question, str) else None,
        planningTools=planning_tools,
        toolsExecuted=tools,
        finalAnswer=final_answer if isinstance(final_answer, str) else None,
    )


@app.post("/trace-summary", response_model=TraceSummaryResponse)
async def trace_summary(payload: TraceSummaryRequest) -> TraceSummaryResponse:
    """Summarise a traced execution using the local ChatOllama model.

    This endpoint is designed to support the observability UI's "Process trace"
    button. It produces a compact execution narrative that makes the trace more
    interpretable for humans while keeping the structure predictable.
    """

    if not payload.observations:
        return TraceSummaryResponse()

    try:
        prompt = _build_prompt(payload)
    except Exception:
        # If we cannot build a prompt, fall back to an empty summary.
        return TraceSummaryResponse()

    try:
        result = _llm.invoke(prompt)
    except Exception as exc:
        logger.info("[TraceSummary] LLM invoke failed: %r", exc)
        return TraceSummaryResponse()

    text = result.content if isinstance(result.content, str) else str(result.content)
    logger.info("[TraceSummary] Raw LLM output prefix: %r", text[:400])
    try:
        return _parse_model_json(text)
    except Exception as exc:
        logger.info("[TraceSummary] Failed to parse model JSON: %r", exc)
        return TraceSummaryResponse()
