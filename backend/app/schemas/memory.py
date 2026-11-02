from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

MemoryKind = Literal[
    "goal",
    "constraint",
    "focus",
    "discussed",
    "open_question",
    "correction",
    "next_step",
    "hypothesis",
    "conclusion",
    "preference",
    "note",
]
MemoryLevel = Literal["user", "project", "paper", "conversation"]
MemoryStatus = Literal["candidate", "active", "superseded"]


class StateItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=300)
    source_seq: int = Field(ge=1)


class ResearchState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: list[StateItem] = Field(default_factory=list, max_length=4)
    focus: list[StateItem] = Field(default_factory=list, max_length=4)
    constraint: list[StateItem] = Field(default_factory=list, max_length=4)
    discussed: list[StateItem] = Field(default_factory=list, max_length=4)
    open_question: list[StateItem] = Field(default_factory=list, max_length=4)
    correction: list[StateItem] = Field(default_factory=list, max_length=4)
    next_step: list[StateItem] = Field(default_factory=list, max_length=4)


class ProjectInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=2000)


class MemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    level: MemoryLevel = "project"
    project_id: UUID | None = None
    conversation_id: UUID | None = None
    paper_id: UUID | None = None
    kind: MemoryKind = "note"
    content: str = Field(min_length=1, max_length=2000)
    status: MemoryStatus = "active"
    pinned: bool = False
    source_message_id: UUID | None = None
    citation_ids: list[UUID] = Field(default_factory=list, max_length=20)
    # The user's own judgement is distinct from a source-linked claim.
    epistemic: Literal["user_statement", "hypothesis", "evidence_linked"] = "user_statement"


class MemoryUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    content: str | None = Field(default=None, min_length=1, max_length=2000)
    kind: MemoryKind | None = None
    status: MemoryStatus | None = None
    pinned: bool | None = None
    epistemic: Literal["user_statement", "hypothesis", "evidence_linked"] | None = None
    citation_ids: list[UUID] | None = Field(default=None, max_length=20)


class ConversationMemoryUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(default="", max_length=2400)
    research_state: ResearchState = Field(default_factory=ResearchState)


class ReadingPositionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version_id: UUID
    page_idx: int = Field(ge=0, le=100000)


class CoreMemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    content: str = Field(default="", max_length=3000)


class MemorySettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memory_write_mode: Literal["auto", "review"] = "auto"


class BulkMemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[UUID] = Field(min_length=1, max_length=100)
    action: Literal["confirm", "supersede", "delete", "pin", "unpin", "move_project"]
    project_id: UUID | None = None


class MemoryMutation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    target: Literal["profile", "project", "entry"] = "entry"
    action: Literal["add", "update", "supersede", "delete", "noop"] = "add"
    kind: MemoryKind = "note"
    content: str = Field(default="", max_length=1000)
    replace_id: str | None = Field(default=None, max_length=64)
    source_seq: int = Field(ge=1)
