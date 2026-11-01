from dataclasses import dataclass, field
from uuid import UUID

from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievalScope:
    """None means all owned resources; an empty tuple deliberately selects none.

    When both filters are supplied they intersect. The tenant is never model input.
    """

    user_id: UUID
    collection_ids: tuple[UUID, ...] | None = None
    paper_ids: tuple[UUID, ...] | None = None


@dataclass
class RetrievalResult:
    documents: list[Document]
    degraded: list[str] = field(default_factory=list)
    channel_counts: dict[str, int] = field(default_factory=dict)
    operation_id: UUID | None = None
    plan: dict = field(default_factory=dict)


class RetrievalUnavailable(Exception):
    """No healthy retrieval channel, distinct from a successful empty result."""
