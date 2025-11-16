from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

from agent.config import load_settings
from .metrics import score_pairs

logger = logging.getLogger("uvicorn.error")


class ToolSummary(BaseModel):
    name: str
    summary: str


class ExecutionNarrative(BaseModel):
    userQuestion: Optional[str] = None
    planningTools: List[str] = Field(default_factory=list)
    toolsExecuted: List[ToolSummary] = Field(default_factory=list)
    finalAnswer: Optional[str] = None


class ReasoningSummary(BaseModel):
    goal: Optional[str] = None
    plan: List[str] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)
    result: Optional[str] = None


class ObservationInsight(BaseModel):
    observationId: str
    stage: Optional[str] = None
    summary: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)


class ObservationMetric(BaseModel):
    observationId: str
    metric: str
    subject: str
    entailment: float
    contradiction: float
    neutral: float
    label: str


class TraceSummaryRequest(BaseModel):
    trace_id: str
    trace_name: Optional[str] = None
    observations: List[Dict[str, Any]]


class TraceSummaryResponse(BaseModel):
    userQuestion: Optional[str] = None
    planningTools: List[str] = Field(default_factory=list)
    toolsExecuted: List[ToolSummary] = Field(default_factory=list)
    finalAnswer: Optional[str] = None
    reasoningSummary: ReasoningSummary = ReasoningSummary()
    observationInsights: List[ObservationInsight] = Field(default_factory=list)
    observationMetrics: List[ObservationMetric] = Field(default_factory=list)
    rootCauseObservationId: Optional[str] = None


settings = load_settings()
_llm = ChatOllama(
    model=settings.model,
    base_url=settings.ollama_base_url,
    temperature=settings.temperature,
    reasoning=True,
)

app = FastAPI(title="Observability backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_messages(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and "messages" in obj:
        value = obj.get("messages")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def _first_user_message(observations: List[Dict[str, Any]]) -> Optional[str]:
    for observation in observations:
        messages = _extract_messages(observation.get("input"))
        for message in messages:
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                content = message["content"].strip()
                if content:
                    return content
    return None


def _extract_reasoning_text(observation: Dict[str, Any]) -> Optional[str]:
    output = observation.get("output")
    if isinstance(output, dict):
        addl = output.get("additional_kwargs")
        if isinstance(addl, dict) and isinstance(addl.get("reasoning_content"), str):
            text = addl["reasoning_content"].strip()
            if text:
                return text
    return None


def _extract_assistant_content(observation: Dict[str, Any]) -> Optional[str]:
    output = observation.get("output")

    def _first_content(messages: List[Dict[str, Any]]) -> Optional[str]:
        for record in messages:
            content = record.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return None

    messages = _extract_messages(output)
    if messages:
        value = _first_content(messages)
        if value:
            return value
    if isinstance(output, str) and output.strip():
        return output.strip()
    return None


def _format_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        text = payload.strip()
        if text:
            return text
        return ""
    try:
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return str(payload)


def _tool_summaries(observations: List[Dict[str, Any]]) -> List[ToolSummary]:
    summaries: List[ToolSummary] = []
    seen: set[str] = set()
    for obs in observations:
        if obs.get("type") != "TOOL" or not isinstance(obs.get("name"), str):
            continue
        name = obs["name"].strip()
        if not name or name in seen:
            continue
        seen.add(name)
        output = obs.get("output")
        text = _format_payload(output)
        first_line = text.splitlines()[0] if text else ""
        if len(first_line) > 200:
            first_line = first_line[:197].rstrip() + "…"
        summaries.append(ToolSummary(name=name, summary=first_line or "(no output)"))
    return summaries


def _tool_output_summary(observation: Dict[str, Any]) -> Optional[str]:
    if observation.get("type") != "TOOL":
        return None
    name = observation.get("name")
    name_value = name.strip() if isinstance(name, str) else "tool"
    payload = _format_payload(observation.get("output"))
    first_line = payload.splitlines()[0] if payload else ""
    if len(first_line) > 200:
        first_line = first_line[:197].rstrip() + "…"
    return f"{name_value}: {first_line or '(no output)'}"


def _planned_tools(observations: List[Dict[str, Any]]) -> List[str]:
    tool_names = {obs["name"].strip() for obs in observations if obs.get("type") == "TOOL" and isinstance(obs.get("name"), str)}
    for obs in observations:
        if obs.get("type") not in {"AGENT", "GENERATION", "SPAN"}:
            continue
        reasoning = _extract_reasoning_text(obs)
        if not reasoning:
            continue
        planned: List[str] = []
        for token in tool_names:
            if token and token in reasoning and token not in planned:
                planned.append(token)
        if planned:
            return planned
    return []


def _final_answer(observations: List[Dict[str, Any]]) -> Optional[str]:
    for obs in reversed(observations):
        if obs.get("type") not in {"AGENT", "GENERATION"}:
            continue
        output = obs.get("output")
        if isinstance(output, str) and output.strip():
            return output.strip()
        messages = _extract_messages(output)
        for message in reversed(messages):
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                text = message["content"].strip()
                if text:
                    return text
    return None


def _build_execution_narrative(observations: List[Dict[str, Any]]) -> ExecutionNarrative:
    return ExecutionNarrative(
        userQuestion=_first_user_message(observations),
        planningTools=_planned_tools(observations),
        toolsExecuted=_tool_summaries(observations),
        finalAnswer=_final_answer(observations),
    )


def _reasoning_snippets(observations: List[Dict[str, Any]], limit: int = 2200) -> List[Dict[str, Any]]:
    snippets: List[Dict[str, Any]] = []
    for obs in observations:
        reasoning = _extract_reasoning_text(obs)
        if not reasoning:
            continue
        entry = {
            "observation_id": obs.get("id"),
            "type": obs.get("type"),
            "name": obs.get("name"),
            "reasoning": reasoning[:limit],
        }
        if obs.get("type") == "TOOL":
            entry["tool_output"] = _format_payload(obs.get("output"))[:limit]
        snippets.append(entry)
    return snippets


def _premise_from_question(user_question: Optional[str]) -> str:
    if user_question:
        return f"User asked: {user_question.strip()}"
    return "User request: (not captured)"


def _premise_with_tools(base: str, tool_summaries: List[str]) -> str:
    evidence = "\n".join(f"- {summary}" for summary in tool_summaries)
    return f"{base}\nTool evidence:\n{evidence}"


def _label_from_probs(contradiction: float, entailment: float) -> str:
    if contradiction >= 0.5:
        return "CONTRADICTED"
    if entailment >= 0.6:
        return "ENTAILED"
    return "NEUTRAL"


def _compute_observation_metrics(
    observations: List[Dict[str, Any]],
    user_question: Optional[str],
    request_label: str,
) -> Tuple[List[ObservationMetric], Optional[str]]:
    jobs: List[Dict[str, Any]] = []
    tool_history: List[str] = []
    base = _premise_from_question(user_question)

    for obs in observations:
        obs_type = obs.get("type")
        if obs_type == "TOOL":
            summary = _tool_output_summary(obs)
            if summary:
                tool_history.append(summary)
            continue
        if obs_type not in {"AGENT", "GENERATION"}:
            continue

        observation_id = obs.get("id")
        if not isinstance(observation_id, str):
            continue

        reasoning = _extract_reasoning_text(obs)
        assistant = _extract_assistant_content(obs)
        has_tools = bool(tool_history)

        if not has_tools and reasoning:
            jobs.append(
                {
                    "observation_id": observation_id,
                    "metric": "validity_with_query",
                    "subject": "reasoning",
                    "premise": base,
                    "hypothesis": reasoning,
                }
            )
            continue

        if has_tools:
            premise = _premise_with_tools(base, tool_history[-6:])
            if reasoning:
                jobs.append(
                    {
                        "observation_id": observation_id,
                        "metric": "groundedness_with_evidence",
                        "subject": "reasoning",
                        "premise": premise,
                        "hypothesis": reasoning,
                    }
                )
            if assistant:
                jobs.append(
                    {
                        "observation_id": observation_id,
                        "metric": "groundedness_with_evidence",
                        "subject": "final_answer",
                        "premise": premise,
                        "hypothesis": assistant,
                    }
                )

    if not jobs:
        return [], None

    pairs = [(job["premise"], job["hypothesis"]) for job in jobs]
    scores = score_pairs(pairs)
    metrics: List[ObservationMetric] = []
    root_cause: Optional[str] = None

    for job, probs in zip(jobs, scores):
        entailment = float(probs.get("entailment", 0.0))
        contradiction = float(probs.get("contradiction", 0.0))
        neutral = float(probs.get("neutral", max(0.0, 1.0 - entailment - contradiction)))
        label = _label_from_probs(contradiction, entailment)
        metric = ObservationMetric(
            observationId=job["observation_id"],
            metric=job["metric"],
            subject=job["subject"],
            entailment=entailment,
            contradiction=contradiction,
            neutral=neutral,
            label=label,
        )
        metrics.append(metric)
        if (
            job["metric"] == "groundedness_with_evidence"
            and label == "CONTRADICTED"
            and root_cause is None
        ):
            root_cause = job["observation_id"]

    logger.info(
        "[TraceSummary %s] Computed %d observation metrics (root_cause=%s)",
        request_label,
        len(metrics),
        root_cause,
    )
    return metrics, root_cause


_PROMPT_TEMPLATE = """
You are an observability analyst helping engineers understand an LLM agent's reasoning.
Given metadata about a trace and reasoning snippets per observation, return a pure JSON
object matching this schema:

{
  "goal": string or null,
  "plan": array of short strings,
  "observations": array of short strings describing key observations or tool outcomes,
  "result": string or null summarising the conclusion,
  "observation_insights": [
    {
      "observation_id": string,
      "stage": string (one of Goal, Plan, Observation, Tool, Result, or Other),
      "summary": short string,
      "bullets": array of short bullet points explaining the reasoning for this step
    }
  ]
}

Rules:
- Only use the provided data.
- Keep bullets concise and in plain language.
- Do not include any prose outside the JSON object.
- Limit each list to at most 4 items.
"""


def _call_llm(
    trace_name: str,
    narrative: ExecutionNarrative,
    snippets: List[Dict[str, Any]],
    request_label: str,
) -> Optional[Dict[str, Any]]:
    payload = {
        "trace_name": trace_name,
        "user_question": narrative.userQuestion,
        "planned_tools": narrative.planningTools,
        "tools_executed": [summary.dict() for summary in narrative.toolsExecuted],
        "reasoning_snippets": snippets,
    }
    prompt = f"{_PROMPT_TEMPLATE}\nTrace metadata and reasoning (JSON):\n{json.dumps(payload, ensure_ascii=False)}"
    logger.info(
        "[TraceSummary %s] Invoking LLM model=%s (snippets=%d)",
        request_label,
        settings.model,
        len(snippets),
    )
    start = time.perf_counter()
    try:
        response = _llm.invoke(prompt)
    except Exception as exc:
        logger.info("[TraceSummary %s] LLM invoke failed: %r", request_label, exc)
        return None
    duration = time.perf_counter() - start
    text = response.content if isinstance(response.content, str) else str(response.content)
    logger.info(
        "[TraceSummary %s] LLM responded in %.2fs • preview=%r",
        request_label,
        duration,
        text[:300],
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception as exc:
                logger.info("[TraceSummary %s] Failed to parse JSON snippet: %r", request_label, exc)
    return None


@app.post("/trace-summary", response_model=TraceSummaryResponse)
def trace_summary(payload: TraceSummaryRequest) -> TraceSummaryResponse:
    request_label = uuid4().hex[:8]
    observations = payload.observations or []
    logger.info(
        "[TraceSummary %s] Request received (trace_id=%s, observations=%d)",
        request_label,
        payload.trace_id,
        len(observations),
    )
    narrative = _build_execution_narrative(observations)
    logger.info(
        "[TraceSummary %s] Narrative seeds -> userQuestion=%s planning=%d executed=%d",
        request_label,
        bool(narrative.userQuestion),
        len(narrative.planningTools),
        len(narrative.toolsExecuted),
    )
    snippets = _reasoning_snippets(observations)
    logger.info(
        "[TraceSummary %s] Collected reasoning snippets=%d", request_label, len(snippets)
    )
    if not observations:
        logger.info("[TraceSummary %s] No observations supplied; returning empty summary", request_label)
        return TraceSummaryResponse()

    llm_result = _call_llm(payload.trace_name or payload.trace_id, narrative, snippets, request_label)

    reasoning_summary = ReasoningSummary()
    observation_insights: List[ObservationInsight] = []

    if isinstance(llm_result, dict):
        goal = llm_result.get("goal")
        plan = llm_result.get("plan")
        obs = llm_result.get("observations")
        result = llm_result.get("result")
        if isinstance(goal, str) and goal.strip():
            reasoning_summary.goal = goal.strip()
        if isinstance(plan, list):
            reasoning_summary.plan = [str(item).strip() for item in plan if str(item).strip()][:4]
        if isinstance(obs, list):
            reasoning_summary.observations = [str(item).strip() for item in obs if str(item).strip()][:4]
        if isinstance(result, str) and result.strip():
            reasoning_summary.result = result.strip()

        insights = llm_result.get("observation_insights")
        if isinstance(insights, list):
            for entry in insights[:12]:
                if not isinstance(entry, dict):
                    continue
                obs_id = entry.get("observation_id")
                if not isinstance(obs_id, str) or not obs_id:
                    continue
                stage = entry.get("stage")
                summary = entry.get("summary")
                bullets = entry.get("bullets") if isinstance(entry.get("bullets"), list) else []
                observation_insights.append(
                    ObservationInsight(
                        observationId=obs_id,
                        stage=stage if isinstance(stage, str) else None,
                        summary=summary if isinstance(summary, str) else None,
                        bullets=[str(item).strip() for item in bullets if str(item).strip()][:4],
                    )
                )

    else:
        logger.info("[TraceSummary %s] LLM returned no structured result", request_label)

    metrics, root_cause = _compute_observation_metrics(observations, narrative.userQuestion, request_label)

    response = TraceSummaryResponse(
        userQuestion=narrative.userQuestion,
        planningTools=narrative.planningTools,
        toolsExecuted=narrative.toolsExecuted,
        finalAnswer=narrative.finalAnswer,
        reasoningSummary=reasoning_summary,
        observationInsights=observation_insights,
        observationMetrics=metrics,
        rootCauseObservationId=root_cause,
    )
    logger.info(
        "[TraceSummary %s] Responding with reasoningSummary(goal=%s, plan=%d, obs=%d, result=%s, insights=%d, metrics=%d)",
        request_label,
        bool(reasoning_summary.goal),
        len(reasoning_summary.plan),
        len(reasoning_summary.observations),
        bool(reasoning_summary.result),
        len(observation_insights),
        len(metrics),
    )
    return response
