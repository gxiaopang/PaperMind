import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'admin')"),
        CheckConstraint("status IN ('active', 'disabled')"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(Text)
    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    role: Mapped[str] = mapped_column(String(16), default="user")
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    csrf_hash: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AccountToken(Base):
    __tablename__ = "account_tokens"
    __table_args__ = (CheckConstraint("kind IN ('verify', 'reset')"),)
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    kind: Mapped[str] = mapped_column(String(16))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Collection(Base):
    __tablename__ = "collections"
    __table_args__ = (
        UniqueConstraint("user_id", "id"),
        Index("ix_collections_owner_created", "user_id", "created_at"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EmailCode(Base):
    __tablename__ = "email_codes"
    email: Mapped[str] = mapped_column(String(320), primary_key=True)
    code_hash: Mapped[str] = mapped_column(String(64))
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    attempts: Mapped[int] = mapped_column(default=0)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MailOutbox(Base):
    __tablename__ = "mail_outbox"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    encrypted_payload: Mapped[str] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(default=0)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(String(100))


class SystemConfig(Base):
    __tablename__ = "system_config"
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB)


# M2: tenant ownership is enforced in PostgreSQL as well as at API boundaries.


class Paper(Base):
    __tablename__ = "papers"
    __table_args__ = (
        UniqueConstraint("user_id", "id"),
        UniqueConstraint("user_id", "collection_id", "id"),
        ForeignKeyConstraint(
            ["user_id", "collection_id"], ["collections.user_id", "collections.id"]
        ),
        ForeignKeyConstraint(
            ["user_id", "id", "active_version_id"],
            ["paper_versions.user_id", "paper_versions.paper_id", "paper_versions.id"],
            name="fk_paper_active_version",
            use_alter=True,
        ),
        ForeignKeyConstraint(
            ["user_id", "id", "pending_version_id"],
            ["paper_versions.user_id", "paper_versions.paper_id", "paper_versions.id"],
            name="fk_paper_pending_version",
            use_alter=True,
        ),
        Index("ix_papers_owner_collection", "user_id", "collection_id"),
        Index(
            "uq_papers_live_upload",
            "user_id",
            "collection_id",
            "sha256",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    collection_id: Mapped[uuid.UUID | None]
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[list] = mapped_column(JSONB, default=list)
    year: Mapped[int | None]
    venue: Mapped[str | None] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(Text)
    arxiv_id: Mapped[str | None] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text)
    metadata_edited: Mapped[bool] = mapped_column(default=False)
    sha256: Mapped[str] = mapped_column(String(64))
    file_path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    page_count: Mapped[int]
    status: Mapped[str] = mapped_column(String(24), default="pending")
    active_version_id: Mapped[uuid.UUID | None]
    pending_version_id: Mapped[uuid.UUID | None]
    error: Mapped[str | None] = mapped_column(Text)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ParseJob(Base):
    __tablename__ = "parse_jobs"
    __table_args__ = (
        UniqueConstraint("user_id", "paper_id", "id"),
        ForeignKeyConstraint(["user_id", "paper_id"], ["papers.user_id", "papers.id"]),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID]
    paper_id: Mapped[uuid.UUID]
    status: Mapped[str] = mapped_column(String(24), default="pending")
    batch_id: Mapped[str | None] = mapped_column(Text)
    result_path: Mapped[str] = mapped_column(Text)
    # A job always resumes using the exact configuration with which it started.
    config: Mapped[dict] = mapped_column(JSONB)
    attempts: Mapped[int] = mapped_column(default=0)
    charged: Mapped[bool] = mapped_column(default=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PaperVersion(Base):
    __tablename__ = "paper_versions"
    __table_args__ = (
        UniqueConstraint("user_id", "paper_id", "id"),
        UniqueConstraint("user_id", "id"),
        ForeignKeyConstraint(["user_id", "paper_id"], ["papers.user_id", "papers.id"]),
        ForeignKeyConstraint(
            ["user_id", "paper_id", "parse_job_id"],
            ["parse_jobs.user_id", "parse_jobs.paper_id", "parse_jobs.id"],
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID]
    paper_id: Mapped[uuid.UUID]
    parse_job_id: Mapped[uuid.UUID]
    operation: Mapped[str] = mapped_column(String(16))
    chunker_version: Mapped[str] = mapped_column(String(32), default="academic-bge-v5")
    reference_entries: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    paper_map: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    embedding_provider: Mapped[str] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(Text)
    embedding_dim: Mapped[int]
    status: Mapped[str] = mapped_column(String(24), default="pending")
    attempts: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Section(Base):
    __tablename__ = "sections"
    __table_args__ = (
        UniqueConstraint("user_id", "version_id", "id"),
        ForeignKeyConstraint(
            ["user_id", "version_id"], ["paper_versions.user_id", "paper_versions.id"]
        ),
        ForeignKeyConstraint(
            ["user_id", "version_id", "parent_id"],
            ["sections.user_id", "sections.version_id", "sections.id"],
            deferrable=True,
            initially="DEFERRED",
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID]
    version_id: Mapped[uuid.UUID]
    parent_id: Mapped[uuid.UUID | None]
    title: Mapped[str] = mapped_column(Text)
    level: Mapped[int]
    path: Mapped[list] = mapped_column(JSONB)
    section_std: Mapped[str] = mapped_column(String(32), default="other", server_default="other")


class ChunkParent(Base):
    __tablename__ = "chunk_parents"
    __table_args__ = (
        UniqueConstraint("user_id", "version_id", "id"),
        UniqueConstraint("user_id", "version_id", "ordinal"),
        ForeignKeyConstraint(
            ["user_id", "paper_id", "version_id"],
            ["paper_versions.user_id", "paper_versions.paper_id", "paper_versions.id"],
        ),
        ForeignKeyConstraint(
            ["user_id", "version_id", "section_id"],
            ["sections.user_id", "sections.version_id", "sections.id"],
        ),
        CheckConstraint("child_count > 0"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID]
    paper_id: Mapped[uuid.UUID]
    version_id: Mapped[uuid.UUID]
    section_id: Mapped[uuid.UUID | None]
    ordinal: Mapped[int]
    kind: Mapped[str] = mapped_column(String(24))
    text: Mapped[str] = mapped_column(Text)
    locations: Mapped[list] = mapped_column(JSONB)
    child_count: Mapped[int]


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("user_id", "version_id", "ordinal"),
        ForeignKeyConstraint(
            ["user_id", "paper_id", "version_id"],
            ["paper_versions.user_id", "paper_versions.paper_id", "paper_versions.id"],
        ),
        ForeignKeyConstraint(
            ["user_id", "collection_id", "paper_id"],
            ["papers.user_id", "papers.collection_id", "papers.id"],
        ),
        ForeignKeyConstraint(
            ["user_id", "version_id", "section_id"],
            ["sections.user_id", "sections.version_id", "sections.id"],
        ),
        ForeignKeyConstraint(
            ["user_id", "version_id", "parent_id"],
            ["chunk_parents.user_id", "chunk_parents.version_id", "chunk_parents.id"],
            name="fk_chunk_parent",
        ),
        Index("ix_chunks_owner_version", "user_id", "version_id"),
        Index("ix_chunks_owner_parent", "user_id", "version_id", "parent_id", "ordinal"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID]
    collection_id: Mapped[uuid.UUID]
    paper_id: Mapped[uuid.UUID]
    version_id: Mapped[uuid.UUID]
    section_id: Mapped[uuid.UUID | None]
    parent_id: Mapped[uuid.UUID | None]
    ordinal: Mapped[int]
    kind: Mapped[str] = mapped_column(String(24))
    text: Mapped[str] = mapped_column(Text)
    embedding_input: Mapped[str] = mapped_column(Text)
    section_path: Mapped[list] = mapped_column(JSONB)
    locations: Mapped[list] = mapped_column(JSONB)
    source_spans: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    page_start: Mapped[int]
    page_end: Mapped[int]
    token_count: Mapped[int]
    content_hash: Mapped[str] = mapped_column(String(64))
    indexable: Mapped[bool]
    unindexed_reason: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024))


class EmbeddingCache(Base):
    __tablename__ = "embedding_cache"
    __table_args__ = (
        UniqueConstraint("user_id", "provider", "model", "dimension", "content_hash"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    provider: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    dimension: Mapped[int]
    content_hash: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[list[float]] = mapped_column(Vector(1024))


class QuotaReservation(Base):
    __tablename__ = "quota_reservations"
    __table_args__ = (
        UniqueConstraint("user_id", "version_id", "metric"),
        UniqueConstraint("user_id", "run_id", "metric"),
        ForeignKeyConstraint(["user_id", "run_id"], ["chat_runs.user_id", "chat_runs.id"]),
        CheckConstraint(
            "(version_id IS NOT NULL) <> (run_id IS NOT NULL)", name="ck_reservation_resource"
        ),
        ForeignKeyConstraint(
            ["user_id", "version_id"], ["paper_versions.user_id", "paper_versions.id"]
        ),
        CheckConstraint("amount >= 0"),
        Index("ix_reservations_owner_day", "user_id", "created_at"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID]
    version_id: Mapped[uuid.UUID | None]
    run_id: Mapped[uuid.UUID | None]
    metric: Mapped[str] = mapped_column(String(24))
    amount: Mapped[int] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(16), default="reserved")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UsageRecord(Base):
    __tablename__ = "usage_records"
    __table_args__ = (
        Index("ix_usage_owner_day", "user_id", "created_at"),
        Index("ix_usage_operation", "user_id", "operation_id"),
        Index("ix_usage_run", "user_id", "run_id"),
        ForeignKeyConstraint(["user_id", "run_id"], ["chat_runs.user_id", "chat_runs.id"]),
        CheckConstraint(
            "version_id IS NOT NULL OR operation_id IS NOT NULL", name="ck_usage_resource"
        ),
        ForeignKeyConstraint(
            ["user_id", "version_id"], ["paper_versions.user_id", "paper_versions.id"]
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    version_id: Mapped[uuid.UUID | None]
    operation_id: Mapped[uuid.UUID | None]
    run_id: Mapped[uuid.UUID | None]
    node: Mapped[str] = mapped_column(String(24))
    provider: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    metric: Mapped[str] = mapped_column(String(24))
    amount: Mapped[int] = mapped_column(BigInteger)
    estimated: Mapped[bool] = mapped_column(default=True)
    outcome: Mapped[str] = mapped_column(String(24), default="confirmed")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TaskOutbox(Base):
    __tablename__ = "task_outbox"
    __table_args__ = (UniqueConstraint("user_id", "kind", "resource_id"),)
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    kind: Mapped[str] = mapped_column(String(24))
    resource_id: Mapped[uuid.UUID]
    attempts: Mapped[int] = mapped_column(default=0)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Citation(Base):
    """Immutable source snapshots, attachable to M4 messages without live chunk reads."""

    __tablename__ = "citations"
    __table_args__ = (
        UniqueConstraint("user_id", "message_id", "number"),
        ForeignKeyConstraint(["user_id", "message_id"], ["messages.user_id", "messages.id"]),
        ForeignKeyConstraint(["user_id", "paper_id"], ["papers.user_id", "papers.id"]),
        ForeignKeyConstraint(
            ["user_id", "paper_id", "version_id"],
            ["paper_versions.user_id", "paper_versions.paper_id", "paper_versions.id"],
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID]
    paper_id: Mapped[uuid.UUID]
    version_id: Mapped[uuid.UUID]
    chunk_id: Mapped[uuid.UUID]
    message_id: Mapped[uuid.UUID | None]
    number: Mapped[int | None]
    source_text: Mapped[str] = mapped_column(Text)
    locations: Mapped[list] = mapped_column(JSONB)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class UserQuota(Base):
    __tablename__ = "user_quotas"
    __table_args__ = (
        CheckConstraint("max_papers IS NULL OR max_papers > 0"),
        CheckConstraint("storage_mb IS NULL OR storage_mb > 0"),
        CheckConstraint("daily_parse_pages IS NULL OR daily_parse_pages > 0"),
        CheckConstraint("daily_model_tokens IS NULL OR daily_model_tokens > 0"),
        CheckConstraint("upload_concurrency IS NULL OR upload_concurrency > 0"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    max_papers: Mapped[int | None]
    storage_mb: Mapped[int | None]
    daily_parse_pages: Mapped[int | None]
    daily_model_tokens: Mapped[int | None]
    upload_concurrency: Mapped[int | None]


class UserSetting(Base):
    """Personal answer preferences; snapshotted into each chat run at submission."""

    __tablename__ = "user_settings"
    __table_args__ = (
        CheckConstraint("answer_language IN ('zh', 'en')"),
        CheckConstraint("detail_level IN ('concise', 'standard', 'detailed')"),
        CheckConstraint("memory_write_mode IN ('auto', 'review')"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    answer_language: Mapped[str] = mapped_column(String(8), default="zh")
    detail_level: Mapped[str] = mapped_column(String(16), default="standard")
    memory_write_mode: Mapped[str] = mapped_column(
        String(16), default="auto", server_default="auto"
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ResearchProject(Base):
    __tablename__ = "research_projects"
    __table_args__ = (UniqueConstraint("user_id", "id"),)
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        UniqueConstraint("user_id", "id"),
        Index("ix_conversations_owner_created", "user_id", "created_at"),
        ForeignKeyConstraint(
            ["user_id", "project_id"], ["research_projects.user_id", "research_projects.id"]
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(120), default="新会话")
    scope: Mapped[dict] = mapped_column(JSONB, default=dict)
    active_paper_ids: Mapped[list] = mapped_column(JSONB, default=list)
    project_id: Mapped[uuid.UUID | None]
    research_state: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'"))
    memory_revision: Mapped[int] = mapped_column(default=0, server_default="0")
    summary: Mapped[str] = mapped_column(Text, default="")
    summarized_seq: Mapped[int] = mapped_column(default=0)
    next_seq: Mapped[int] = mapped_column(default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ChatRun(Base):
    __tablename__ = "chat_runs"
    __table_args__ = (
        UniqueConstraint("user_id", "id"),
        UniqueConstraint("user_id", "conversation_id", "id"),
        UniqueConstraint("user_id", "conversation_id", "idempotency_key"),
        ForeignKeyConstraint(
            ["user_id", "conversation_id"], ["conversations.user_id", "conversations.id"]
        ),
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')"),
        Index("ix_chat_runs_conversation_created", "user_id", "conversation_id", "created_at"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID]
    conversation_id: Mapped[uuid.UUID]
    idempotency_key: Mapped[str] = mapped_column(String(128))
    request_hash: Mapped[str] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(16))
    scope: Mapped[dict] = mapped_column(JSONB)
    config: Mapped[dict] = mapped_column(JSONB)
    preferences: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'"))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    result_status: Mapped[str | None] = mapped_column(String(32))
    error: Mapped[str | None] = mapped_column(Text)
    event_seq: Mapped[int] = mapped_column(default=0)
    retrieval_count: Mapped[int] = mapped_column(default=0)
    generation_count: Mapped[int] = mapped_column(default=0)
    recoveries: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("user_id", "id"),
        UniqueConstraint("user_id", "conversation_id", "seq"),
        UniqueConstraint("user_id", "run_id", "role"),
        ForeignKeyConstraint(
            ["user_id", "conversation_id", "run_id"],
            ["chat_runs.user_id", "chat_runs.conversation_id", "chat_runs.id"],
        ),
        CheckConstraint("role IN ('user', 'assistant')"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID]
    conversation_id: Mapped[uuid.UUID]
    run_id: Mapped[uuid.UUID]
    seq: Mapped[int]
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    rewritten_query: Mapped[str | None] = mapped_column(Text)
    route: Mapped[str | None] = mapped_column(String(16))
    verify_result: Mapped[dict] = mapped_column(JSONB, default=dict)
    feedback: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ChatEvent(Base):
    """Replay ledger; Redis Streams mirror it. Expire only finished runs' events."""

    __tablename__ = "chat_events"
    __table_args__ = (
        ForeignKeyConstraint(["user_id", "run_id"], ["chat_runs.user_id", "chat_runs.id"]),
    )
    run_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    seq: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[uuid.UUID]
    kind: Mapped[str] = mapped_column(String(24))
    attempt: Mapped[int]
    data: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MemoryEntry(Base):
    __tablename__ = "memory_entries"
    __table_args__ = (
        UniqueConstraint("user_id", "fingerprint"),
        CheckConstraint("level IN ('user', 'project', 'paper', 'conversation')"),
        CheckConstraint("status IN ('candidate', 'active', 'superseded')"),
        CheckConstraint("epistemic IN ('user_statement', 'hypothesis', 'evidence_linked')"),
        CheckConstraint(
            "(level = 'user' AND project_id IS NULL AND conversation_id IS NULL AND paper_id IS NULL) OR "
            "(level = 'project' AND project_id IS NOT NULL AND conversation_id IS NULL AND paper_id IS NULL) OR "
            "(level = 'paper' AND paper_id IS NOT NULL AND conversation_id IS NULL) OR "
            "(level = 'conversation' AND conversation_id IS NOT NULL AND paper_id IS NULL)"
        ),
        ForeignKeyConstraint(
            ["user_id", "project_id"], ["research_projects.user_id", "research_projects.id"]
        ),
        ForeignKeyConstraint(
            ["user_id", "conversation_id"], ["conversations.user_id", "conversations.id"]
        ),
        ForeignKeyConstraint(["user_id", "paper_id"], ["papers.user_id", "papers.id"]),
        ForeignKeyConstraint(["user_id", "source_message_id"], ["messages.user_id", "messages.id"]),
        Index("ix_memory_owner_project_status", "user_id", "project_id", "status"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    level: Mapped[str] = mapped_column(String(16))
    project_id: Mapped[uuid.UUID | None]
    conversation_id: Mapped[uuid.UUID | None]
    paper_id: Mapped[uuid.UUID | None]
    kind: Mapped[str] = mapped_column(String(24))
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="candidate")
    epistemic: Mapped[str] = mapped_column(String(24), default="user_statement")
    pinned: Mapped[bool] = mapped_column(default=False)
    source_message_id: Mapped[uuid.UUID | None]
    citation_ids: Mapped[list] = mapped_column(JSONB, default=list)
    reading_anchor: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'"))
    fingerprint: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemoryCoreBlock(Base):
    """Hermes-style bounded core memory (USER.md / MEMORY.md) injected as a frozen snapshot."""

    __tablename__ = "memory_core_blocks"
    __table_args__ = (
        CheckConstraint("level IN ('user', 'project')"),
        CheckConstraint(
            "(level = 'user' AND project_id IS NULL) OR (level = 'project' AND project_id IS NOT NULL)"
        ),
        ForeignKeyConstraint(
            ["user_id", "project_id"], ["research_projects.user_id", "research_projects.id"]
        ),
        Index("ix_memory_core_owner_project", "user_id", "level", "project_id"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    level: Mapped[str] = mapped_column(String(16))
    project_id: Mapped[uuid.UUID | None]
    content: Mapped[str] = mapped_column(Text, default="")
    revision: Mapped[int] = mapped_column(default=0, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MemoryEvent(Base):
    """Reversible audit log for agent/user memory mutations."""

    __tablename__ = "memory_events"
    __table_args__ = (
        CheckConstraint("target IN ('profile', 'project', 'entry')"),
        CheckConstraint("action IN ('add', 'update', 'supersede', 'delete')"),
        CheckConstraint("actor IN ('user', 'agent', 'review')"),
        Index("ix_memory_events_owner_created", "user_id", "created_at"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    project_id: Mapped[uuid.UUID | None]
    conversation_id: Mapped[uuid.UUID | None]
    run_id: Mapped[uuid.UUID | None]
    memory_id: Mapped[uuid.UUID | None]
    target: Mapped[str] = mapped_column(String(24))
    action: Mapped[str] = mapped_column(String(24))
    actor: Mapped[str] = mapped_column(String(16), default="agent")
    summary: Mapped[str] = mapped_column(Text)
    before_state: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'"))
    after_state: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'"))
    undone_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
