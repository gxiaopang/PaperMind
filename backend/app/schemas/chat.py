from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ScopeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_ids: list[UUID] | None = Field(default=None, max_length=100)
    paper_ids: list[UUID] | None = Field(default=None, max_length=100)


class ConversationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="新会话", min_length=1, max_length=120)
    scope: ScopeInput = Field(default_factory=ScopeInput)


class ConversationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, min_length=1, max_length=120)
    scope: ScopeInput | None = None


class RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idempotency_key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    question: str = Field(min_length=1, max_length=1024)
    mode: Literal["fast", "agentic"] | None = None
    # None inherits the conversation selection; {} explicitly selects all owned papers.
    scope: ScopeInput | None = None


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feedback: Literal["up", "down"] | None
