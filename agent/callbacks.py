from __future__ import annotations

import json
import time
from typing import Any, Dict
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .state import get_current_state
from .visibility import get_current_tracker


class ReasoningTraceCallback(BaseCallbackHandler):
    """Capture structured reasoning steps for failure analysis.

    This handler is best-effort: if the AgentState context is not set it
    becomes a no-op.
    """

    def __init__(self) -> None:
        super().__init__()
        # Track basic tool input metadata by run_id so we can correlate start/end.
        self._tool_inputs: dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _now(self) -> float:
        return time.time()

    def _visible_ids(self) -> list[str]:
        tracker = get_current_tracker()
        if tracker is None:
            return []
        return tracker.snapshot_visible_ids()

    def _append_step(
        self,
        *,
        kind: str,
        content: str,
        tool_name: str | None = None,
        tool_input: Dict[str, Any] | None = None,
        tool_output: Dict[str, Any] | None = None,
        error: str | None = None,
        extra_metadata: Dict[str, Any] | None = None,
    ) -> None:
        state = get_current_state()
        if state is None:
            return

        reasoning_trace = state.setdefault("reasoning_trace", [])
        step_index = len(reasoning_trace)
        step_id = f"step:{step_index}:{uuid4().hex}"

        metadata: Dict[str, Any] = dict(extra_metadata or {})

        step: Dict[str, Any] = {
            "id": step_id,
            "step_index": step_index,
            "kind": kind,
            "timestamp": self._now(),
            "content": content,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "error": error,
            "visible_data_ids": self._visible_ids(),
            "metadata": metadata,
        }
        reasoning_trace.append(step)

    @staticmethod
    def _coerce_dict(value: Any) -> Dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                data = json.loads(value)
                return data if isinstance(data, dict) else {"raw": value}
            except json.JSONDecodeError:
                return {"raw": value}
        return {"raw": str(value)}

    # ------------------------------------------------------------------
    # LLM events
    # ------------------------------------------------------------------
    def on_llm_end(self, response: LLMResult, *, run_id: str, parent_run_id: str | None = None, **kwargs: Any) -> None:  # type: ignore[override]
        """Record raw LLM output as a reasoning step.

        This captures the model text before any external formatting.
        """

        generations_text: list[str] = []
        for gen_list in response.generations:
            for gen in gen_list:
                text = getattr(gen, "text", None)
                if isinstance(text, str) and text:
                    generations_text.append(text)
                    continue
                message = getattr(gen, "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                    if isinstance(content, str) and content:
                        generations_text.append(content)
                    elif content is not None:
                        generations_text.append(str(content))

        if not generations_text:
            return

        content = "\n\n".join(generations_text)
        self._append_step(
            kind="llm_output",
            content=content,
            extra_metadata={"run_id": str(run_id), "parent_run_id": str(parent_run_id) if parent_run_id else None},
        )

    # ------------------------------------------------------------------
    # Tool events
    # ------------------------------------------------------------------
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: Any,
        *,
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:  # type: ignore[override]
        tool_name = serialized.get("name") or serialized.get("id") or str(serialized)
        tool_input = self._coerce_dict(input_str)

        self._tool_inputs[str(run_id)] = {"tool_name": tool_name, "tool_input": tool_input}

        self._append_step(
            kind="action",
            content=f"Calling tool {tool_name}",
            tool_name=tool_name,
            tool_input=tool_input,
            extra_metadata={"run_id": str(run_id), "parent_run_id": str(parent_run_id) if parent_run_id else None},
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:  # type: ignore[override]
        key = str(run_id)
        info = self._tool_inputs.get(key, {})
        tool_name = info.get("tool_name")
        tool_output = self._coerce_dict(output)

        self._append_step(
            kind="observation",
            content=str(output),
            tool_name=tool_name,
            tool_input=info.get("tool_input"),
            tool_output=tool_output,
            extra_metadata={"run_id": str(run_id), "parent_run_id": str(parent_run_id) if parent_run_id else None},
        )

        # Cleanup to avoid unbounded growth
        self._tool_inputs.pop(key, None)

    def on_tool_error(
        self,
        error: Exception,
        *,
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:  # type: ignore[override]
        key = str(run_id)
        info = self._tool_inputs.get(key, {})
        tool_name = info.get("tool_name")

        self._append_step(
            kind="observation",
            content=f"Tool {tool_name or 'unknown'} raised an error.",
            tool_name=tool_name,
            tool_input=info.get("tool_input"),
            tool_output=None,
            error=str(error),
            extra_metadata={"run_id": str(run_id), "parent_run_id": str(parent_run_id) if parent_run_id else None},
        )

        self._tool_inputs.pop(key, None)
