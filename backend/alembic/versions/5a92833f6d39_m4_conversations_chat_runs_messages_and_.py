"""M4 conversations chat runs messages and event recovery"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "5a92833f6d39"
down_revision = "0004_m3"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=120), nullable=False),
        sa.Column("scope", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("active_paper_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("summarized_seq", sa.Integer(), nullable=False),
        sa.Column("next_seq", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "id"),
    )
    op.create_index(
        "ix_conversations_owner_created", "conversations", ["user_id", "created_at"], unique=False
    )
    op.create_table(
        "chat_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("scope", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("result_status", sa.String(length=32), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("event_seq", sa.Integer(), nullable=False),
        sa.Column("retrieval_count", sa.Integer(), nullable=False),
        sa.Column("generation_count", sa.Integer(), nullable=False),
        sa.Column("recoveries", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')"),
        sa.ForeignKeyConstraint(
            ["user_id", "conversation_id"],
            ["conversations.user_id", "conversations.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "conversation_id", "id"),
        sa.UniqueConstraint("user_id", "conversation_id", "idempotency_key"),
        sa.UniqueConstraint("user_id", "id"),
    )
    op.create_index(
        "ix_chat_runs_conversation_created",
        "chat_runs",
        ["user_id", "conversation_id", "created_at"],
        unique=False,
    )
    op.create_table(
        "chat_events",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id", "run_id"],
            ["chat_runs.user_id", "chat_runs.id"],
        ),
        sa.PrimaryKeyConstraint("run_id", "seq"),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("rewritten_query", sa.Text(), nullable=True),
        sa.Column("route", sa.String(length=16), nullable=True),
        sa.Column("verify_result", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("feedback", sa.String(length=16), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("role IN ('user', 'assistant')"),
        sa.ForeignKeyConstraint(
            ["user_id", "conversation_id", "run_id"],
            ["chat_runs.user_id", "chat_runs.conversation_id", "chat_runs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "conversation_id", "seq"),
        sa.UniqueConstraint("user_id", "id"),
        sa.UniqueConstraint("user_id", "run_id", "role"),
    )
    op.add_column("citations", sa.Column("message_id", sa.Uuid(), nullable=True))
    op.add_column("citations", sa.Column("number", sa.Integer(), nullable=True))
    op.create_unique_constraint(
        "uq_citations_message_number", "citations", ["user_id", "message_id", "number"]
    )
    op.create_foreign_key(
        "fk_citations_message",
        "citations",
        "messages",
        ["user_id", "message_id"],
        ["user_id", "id"],
    )
    op.add_column("quota_reservations", sa.Column("run_id", sa.Uuid(), nullable=True))
    op.alter_column("quota_reservations", "version_id", existing_type=sa.UUID(), nullable=True)
    op.create_unique_constraint(
        "uq_reservations_run_metric", "quota_reservations", ["user_id", "run_id", "metric"]
    )
    op.create_foreign_key(
        "fk_reservations_run",
        "quota_reservations",
        "chat_runs",
        ["user_id", "run_id"],
        ["user_id", "id"],
    )
    op.add_column("usage_records", sa.Column("run_id", sa.Uuid(), nullable=True))
    op.create_index("ix_usage_run", "usage_records", ["user_id", "run_id"], unique=False)
    op.create_foreign_key(
        "fk_usage_run", "usage_records", "chat_runs", ["user_id", "run_id"], ["user_id", "id"]
    )
    op.create_check_constraint(
        "ck_reservation_resource",
        "quota_reservations",
        "(version_id IS NOT NULL) <> (run_id IS NOT NULL)",
    )


def downgrade():
    # Preserve chat audit data; populated M4 installations require forward fixes.
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM chat_runs) THEN RAISE EXCEPTION 'M4 chat audit records exist; use a forward migration'; END IF; END $$"
    )
    op.drop_constraint("ck_reservation_resource", "quota_reservations", type_="check")
    op.drop_constraint("fk_usage_run", "usage_records", type_="foreignkey")
    op.drop_index("ix_usage_run", table_name="usage_records")
    op.drop_column("usage_records", "run_id")
    op.drop_constraint("fk_reservations_run", "quota_reservations", type_="foreignkey")
    op.drop_constraint("uq_reservations_run_metric", "quota_reservations", type_="unique")
    op.alter_column("quota_reservations", "version_id", existing_type=sa.UUID(), nullable=False)
    op.drop_column("quota_reservations", "run_id")
    op.drop_constraint("fk_citations_message", "citations", type_="foreignkey")
    op.drop_constraint("uq_citations_message_number", "citations", type_="unique")
    op.drop_column("citations", "number")
    op.drop_column("citations", "message_id")
    op.drop_table("messages")
    op.drop_table("chat_events")
    op.drop_index("ix_chat_runs_conversation_created", table_name="chat_runs")
    op.drop_table("chat_runs")
    op.drop_index("ix_conversations_owner_created", table_name="conversations")
    op.drop_table("conversations")
