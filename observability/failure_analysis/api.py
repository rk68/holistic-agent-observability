from __future__ import annotations

from typing import Any, Dict
import os
import base64

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .analyzer import analyze_trace
from .schemas import FailureSummary, ReasoningStep, DataArtefact


app = FastAPI(title="Failure Analysis API", version="0.1.0")

# Allow the local Vite dev server (or a custom origin) to call this API from the browser.
frontend_origin = os.getenv("VITE_FAILURE_ANALYSIS_CORS_ORIGIN") or os.getenv("VITE_DEV_SERVER_ORIGIN") or "http://localhost:5173"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


class FailureSummaryResponse(BaseModel):
    trace_id: str
    has_failure: bool
    failure_types: list[Dict[str, Any]]
    behavioural_signals: Dict[str, Any]
    duration_seconds: float | None
    user_query: str | None
    per_observation_failures: Dict[str, Dict[str, Any]]

    @classmethod
    def from_model(
        cls,
        summary: FailureSummary,
        *,
        duration_seconds: float | None,
        user_query: str | None,
        per_observation_failures: Dict[str, Dict[str, Any]],
    ) -> "FailureSummaryResponse":
        return cls(
            trace_id=summary["trace_id"],
            has_failure=summary["has_failure"],
            failure_types=summary["failure_types"],
            behavioural_signals=summary["behavioural_signals"],
            duration_seconds=duration_seconds,
            user_query=user_query,
            per_observation_failures=per_observation_failures,
        )


def _get_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    joined = ", ".join(names)
    raise RuntimeError(f"Missing Langfuse configuration. Set one of: {joined}.")


def _normalise_host(host: str) -> str:
    return host[:-1] if host.endswith("/") else host


def _fetch_trace_detail(trace_id: str) -> Dict[str, Any]:
    """Fetch a single trace detail document from Langfuse's public API."""

    host = _normalise_host(
        _get_env("VITE_LANGFUSE_HOST", "LANGFUSE_HOST", "LANGFUSE_BASE_URL")
    )
    public_key = _get_env("VITE_LANGFUSE_PUBLIC_KEY", "LANGFUSE_PUBLIC_KEY")
    secret_key = _get_env("VITE_LANGFUSE_SECRET_KEY", "LANGFUSE_SECRET_KEY")

    credentials = f"{public_key}:{secret_key}".encode("utf-8")
    auth_header = base64.b64encode(credentials).decode("ascii")

    url = f"{host}/api/public/traces/{trace_id}"
    response = requests.get(
        url,
        headers={
            "Authorization": f"Basic {auth_header}",
            "Accept": "application/json",
        },
        timeout=15,
    )

    if not response.ok:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Langfuse request failed: {response.status_code} {response.reason}. "
                f"Body: {response.text[:500]}"
            ),
        )

    payload = response.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Unexpected Langfuse response format.")
    return payload


def _build_per_observation_failures(
    summary: FailureSummary,
    reasoning_trace: list[ReasoningStep],
    observations: list[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Best-effort mapping from reasoning steps to observations.

    We assign failures to observations based on step kind and observation type,
    preserving ordering to approximate alignment:
    - llm_output/final_answer -> GENERATION/AGENT observations
    - action/observation      -> TOOL observations (grouped by tool name)
    """

    step_by_id: Dict[str, ReasoningStep] = {}
    for step in reasoning_trace:
        step_id = step.get("id")  # type: ignore[assignment]
        if isinstance(step_id, str) and step_id:
            step_by_id[step_id] = step

    # Normalise observations list.
    obs_list: list[Dict[str, Any]] = [obs for obs in observations if isinstance(obs, dict)]

    # Precompute indices of LLM-like and TOOL observations.
    llm_indices: list[int] = []
    tool_indices_by_name: Dict[str, list[int]] = {}
    generic_tool_indices: list[int] = []

    for idx, obs in enumerate(obs_list):
        obs_type = str(obs.get("type", ""))
        if obs_type in {"GENERATION", "AGENT"}:
            llm_indices.append(idx)
        if obs_type == "TOOL":
            name_raw = str(obs.get("name", "") or "")
            name = name_raw.lower()
            if name:
                tool_indices_by_name.setdefault(name, []).append(idx)
                # Also index by last segment, e.g. "banking.get_account_balance" -> "get_account_balance".
                if "." in name:
                    short = name.split(".")[-1]
                    tool_indices_by_name.setdefault(short, []).append(idx)
            else:
                generic_tool_indices.append(idx)

    # Assignment state
    severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    assign_state: Dict[str, Any] = {"llm_ptr": 0, "tool_ptr": {}}

    def assign_llm_observation() -> str | None:
        ptr = int(assign_state["llm_ptr"])
        if ptr >= len(llm_indices):
            return None
        obs_idx = llm_indices[ptr]
        assign_state["llm_ptr"] = ptr + 1
        obs = obs_list[obs_idx]
        obs_id = obs.get("id")
        return str(obs_id) if isinstance(obs_id, str) else None

    def assign_tool_observation(tool_name: str | None) -> str | None:
        normalized = (tool_name or "").lower()
        indices: list[int] | None = None
        tool_ptr: Dict[str, int] = assign_state["tool_ptr"]  # type: ignore[assignment]

        if normalized:
            if normalized in tool_indices_by_name:
                indices = tool_indices_by_name[normalized]
            else:
                # Try last segment as a fallback.
                if "." in normalized:
                    short = normalized.split(".")[-1]
                    indices = tool_indices_by_name.get(short)

        if not indices:
            indices = generic_tool_indices
            key = "__generic__"
        else:
            key = normalized or "__unnamed__"

        if not indices:
            return None

        ptr = tool_ptr.get(key, 0)
        if ptr >= len(indices):
            return None
        obs_idx = indices[ptr]
        tool_ptr[key] = ptr + 1
        obs = obs_list[obs_idx]
        obs_id = obs.get("id")
        return str(obs_id) if isinstance(obs_id, str) else None

    per_observation: Dict[str, Dict[str, Any]] = {}

    for failure in summary["failure_types"]:
        code = str(failure.get("code", "UNKNOWN"))
        severity_raw = str(failure.get("severity", "LOW")).upper()
        severity = severity_raw if severity_raw in severity_rank else "LOW"
        step_ids = failure.get("step_ids") or []
        if not isinstance(step_ids, list):
            continue

        for step_id in step_ids:
            if not isinstance(step_id, str):
                continue
            step = step_by_id.get(step_id)
            if not step:
                continue

            kind = step.get("kind")
            obs_id: str | None = None
            if kind in {"llm_output", "final_answer"}:
                obs_id = assign_llm_observation()
            elif kind in {"action", "observation"}:
                obs_id = assign_tool_observation(step.get("tool_name"))

            if not obs_id:
                continue

            entry = per_observation.setdefault(obs_id, {"max_severity": "LOW", "codes": []})
            current_rank = severity_rank.get(entry["max_severity"], 0)
            if severity_rank.get(severity, 0) > current_rank:
                entry["max_severity"] = severity
            codes_list: list[str] = entry["codes"]
            if code not in codes_list:
                codes_list.append(code)

    return per_observation


@app.get("/traces/{trace_id}/failure-summary", response_model=FailureSummaryResponse)
def get_failure_summary(trace_id: str) -> FailureSummaryResponse:
    """Fetch a trace from Langfuse and return its failure summary.

    This endpoint relies on the same environment variables as the visualiser
    for connecting to Langfuse and expects that the trace metadata contains
    ``reasoning_trace`` and ``data_context`` keys populated by the agent.
    """

    try:
        trace_detail = _fetch_trace_detail(trace_id)
    except HTTPException:
        # Re-raise HTTPExceptions unchanged.
        raise
    except Exception as exc:  # pragma: no cover - network/IO
        raise HTTPException(status_code=502, detail=f"Failed to load trace from Langfuse: {exc}")

    metadata: Dict[str, Any] = trace_detail.get("metadata") or {}

    reasoning_raw = metadata.get("reasoning_trace") or []
    data_context_raw = metadata.get("data_context") or {}

    if not isinstance(reasoning_raw, list) or not isinstance(data_context_raw, dict):
        raise HTTPException(
            status_code=400,
            detail="Trace does not contain structured reasoning_trace/data_context metadata.",
        )

    # Pydantic / TypedDict friendly coercion.
    reasoning_trace: list[ReasoningStep] = [step for step in reasoning_raw if isinstance(step, dict)]  # type: ignore[assignment]
    data_context: Dict[str, DataArtefact] = {  # type: ignore[assignment]
        key: value
        for key, value in data_context_raw.items()
        if isinstance(value, dict)
    }

    # Derive simple duration and user query from observation timestamps/metadata.
    observations = trace_detail.get("observations") or []

    try:
        from datetime import datetime

        start = datetime.fromisoformat(trace_detail["timestamp"])
        end_times = [
            datetime.fromisoformat(obs["endTime"])
            for obs in observations
            if isinstance(obs, dict) and obs.get("endTime")
        ]
        duration_seconds: float | None = None
        if end_times:
            duration_seconds = max((max(end_times) - start).total_seconds(), 0.0)
    except Exception:
        duration_seconds = None

    user_query: str | None = None
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        if str(obs.get("type", "")).upper() == "SPAN":
            # Heuristic: first span with role=user in metadata.
            metadata_obs = obs.get("metadata") or {}
            if isinstance(metadata_obs, dict) and metadata_obs.get("role") == "user":
                input_value = metadata_obs.get("content") or obs.get("input")
                if isinstance(input_value, str):
                    user_query = input_value
                else:
                    try:
                        import json

                        user_query = json.dumps(input_value, default=str)
                    except Exception:
                        user_query = str(input_value)
                break

    summary = analyze_trace(
        trace_id,
        reasoning_trace=reasoning_trace,
        data_context=data_context,
    )

    per_observation_failures = _build_per_observation_failures(summary, reasoning_trace, observations)

    return FailureSummaryResponse.from_model(
        summary,
        duration_seconds=duration_seconds,
        user_query=user_query,
        per_observation_failures=per_observation_failures,
    )
