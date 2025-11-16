from __future__ import annotations

import json
import traceback
from pathlib import Path

from agent.config import load_settings
from agent.factory import ask


_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts" / "banking_use_case.json"


def _load_sample_tasks() -> list[dict]:
    if not _PROMPTS_PATH.is_file():
        raise FileNotFoundError(f"Prompt file not found: {_PROMPTS_PATH}")

    data = json.loads(_PROMPTS_PATH.read_text(encoding="utf-8"))
    tasks = data.get("sample_tasks", [])
    if not isinstance(tasks, list):  # defensive
        raise ValueError("Expected 'sample_tasks' to be a list in banking_use_case.json")
    return tasks


def run_all() -> None:
    settings = load_settings()
    tasks = _load_sample_tasks()

    if not tasks:
        print("No sample_tasks found in prompts/banking_use_case.json")
        return

    print(
        "Running {count} banking sample tasks with backend={backend!r}, "
        "model={model!r}, aws_model={aws_model!r}".format(
            count=len(tasks),
            backend=settings.agent_backend,
            model=settings.model,
            aws_model=settings.aws_agent_model,
        )
    )
    print()

    for index, task in enumerate(tasks, start=1):
        task_id = task.get("id") or f"task-{index}"
        user_query = task.get("user_query") or ""

        if not user_query:
            print(f"[skip] {task_id}: missing user_query")
            continue

        print(f"=== Task {index}: {task_id} ===")
        print(f"User: {user_query}")

        try:
            response = ask(user_query, settings=settings)
        except Exception as exc:  # pragma: no cover - manual run helper
            print(
                "Agent error ({etype}): {repr}\n".format(
                    etype=type(exc).__name__, repr=repr(exc)
                )
            )
            traceback.print_exc()
            print()
            continue

        print(f"Agent: {response}\n")


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    run_all()
