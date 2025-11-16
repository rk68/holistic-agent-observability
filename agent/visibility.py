from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, MutableMapping
from uuid import uuid4


class DataSensitivity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DataArtefactKind(str, Enum):
    USER_MESSAGE = "user_message"
    RETRIEVED_CHUNK = "retrieved_chunk"
    SQL_ROWSET = "sql_rowset"
    HTTP_RESPONSE = "http_response"
    FILE_CONTENTS = "file_contents"
    TOOL_OUTPUT = "tool_output"
    MODEL_OUTPUT = "model_output"


@dataclass(slots=True)
class ArtefactInstance:
    id: str
    kind: DataArtefactKind
    source_tool: str
    source_tool_type: str | None = None
    source_observation_id: str | None = None
    sensitivity: DataSensitivity = DataSensitivity.NONE
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class VisibilityTracker:
    """Track which artefacts are visible to the agent at each step.

    Intended usage pattern (conceptual):

    - Maintain a single tracker instance in your agent state.
    - When a tool call returns data, register one or more artefacts via
      ``register_tool_result``; their IDs are added to ``visible_ids``.
    - Before each LLM / reasoning step, snapshot ``visible_ids`` and attach
      it to the step metadata under the ``visible_data`` key so the
      observability UI can reconstruct visibility for that step.

    The observability frontend expects ``metadata.visible_data`` to be a
    list of artefact IDs (strings).
    """

    artefacts: dict[str, ArtefactInstance] = field(default_factory=dict)
    visible_ids: set[str] = field(default_factory=set)

    def register_tool_result(
        self,
        *,
        tool_name: str,
        kind: DataArtefactKind,
        sensitivity: DataSensitivity | None = None,
        tags: Iterable[str] | None = None,
        artefact_id: str | None = None,
        source_observation_id: str | None = None,
        source_tool_type: str | None = None,
    ) -> ArtefactInstance:
        """Create an artefact instance for a tool result and mark it visible.

        Parameters
        ----------
        tool_name:
            The concrete tool name (e.g. "banking.sql_query").
        kind:
            High-level kind of data produced (SQL rowset, retrieved chunk, etc.).
        sensitivity:
            Optional sensitivity override. If omitted, callers should assume
            a sensible default based on the tool type.
        tags:
            Optional classifier tags, e.g. ["PII", "internal"].
        artefact_id:
            Optional stable identifier. If omitted, a synthetic ID is generated.
        source_observation_id:
            Optional Langfuse observation ID where this artefact was produced.
        source_tool_type:
            Optional coarse tool type classification (e.g. "sql_db").
        """

        if artefact_id is None:
            artefact_id = f"artefact:{kind}:{uuid4().hex}"

        instance = ArtefactInstance(
            id=artefact_id,
            kind=kind,
            source_tool=tool_name,
            source_tool_type=source_tool_type,
            source_observation_id=source_observation_id,
            sensitivity=sensitivity or DataSensitivity.NONE,
            tags=tuple(tags or ()),
        )

        self.artefacts[instance.id] = instance
        self.visible_ids.add(instance.id)
        return instance

    def mark_visible(self, artefact_ids: Iterable[str]) -> None:
        """Add existing artefact IDs to the visible set."""

        self.visible_ids.update(artefact_ids)

    def mark_hidden(self, artefact_ids: Iterable[str]) -> None:
        """Remove artefact IDs from the visible set, if present."""

        for artefact_id in artefact_ids:
            self.visible_ids.discard(artefact_id)

    def snapshot_visible_ids(self) -> list[str]:
        """Return a stable list of currently visible artefact IDs."""

        return sorted(self.visible_ids)

    def attach_visible_to_metadata(self, metadata: MutableMapping[str, Any]) -> None:
        """Attach current visibility to a metadata mapping.

        This writes a ``visible_data`` key, which the observability UI
        interprets as the list of artefact IDs visible at the time that
        metadata snapshot was taken.
        """

        metadata["visible_data"] = self.snapshot_visible_ids()

    def visible_high_sensitivity(self) -> list[ArtefactInstance]:
        """Return the subset of visible artefacts marked as high sensitivity."""

        result: list[ArtefactInstance] = []
        for artefact_id in self.visible_ids:
            artefact = self.artefacts.get(artefact_id)
            if artefact and artefact.sensitivity is DataSensitivity.HIGH:
                result.append(artefact)
        return result

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> VisibilityTracker:
        """Reconstruct a tracker from existing ``visible_data`` metadata.

        Only the visible IDs are restored; artefact details must be
        re-attached separately if needed.
        """

        raw = metadata.get("visible_data")
        tracker = cls()
        if isinstance(raw, list):
            for value in raw:
                if isinstance(value, str) and value:
                    tracker.visible_ids.add(value)
        return tracker


_CURRENT_TRACKER: ContextVar[VisibilityTracker | None] = ContextVar(
    "visibility_tracker", default=None
)


def set_current_tracker(tracker: VisibilityTracker | None) -> None:
    _CURRENT_TRACKER.set(tracker)


def get_current_tracker() -> VisibilityTracker | None:
    return _CURRENT_TRACKER.get()
