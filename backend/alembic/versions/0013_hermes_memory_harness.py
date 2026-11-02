"""Hermes-style core memory blocks, reversible memory events, and session search index."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

from alembic import op

revision = "0013_hermes_memory_harness"
down_revision = "0012_research_memory"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "user_settings",
        sa.Column("memory_write_mode", sa.String(16), nullable=False, server_default="auto"),
    )
    op.create_check_constraint(
        "ck_user_settings_memory_write_mode",
        "user_settings",
        "memory_write_mode IN ('auto', 'review')",
    )
    op.create_table(
        "memory_core_blocks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("level", sa.String(16), nullable=False),
        sa.Column("project_id", sa.Uuid()),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("level IN ('user', 'project')"),
        sa.CheckConstraint(
            "(level = 'user' AND project_id IS NULL) OR (level = 'project' AND project_id IS NOT NULL)"
        ),
        sa.ForeignKeyConstraint(
            ["user_id", "project_id"], ["research_projects.user_id", "research_projects.id"]
        ),
    )
    op.create_index(
        "ix_memory_core_owner_project",
        "memory_core_blocks",
        ["user_id", "level", "project_id"],
    )
    op.create_table(
        "memory_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("project_id", sa.Uuid()),
        sa.Column("conversation_id", sa.Uuid()),
        sa.Column("run_id", sa.Uuid()),
        sa.Column("memory_id", sa.Uuid()),
        sa.Column("target", sa.String(24), nullable=False),
        sa.Column("action", sa.String(24), nullable=False),
        sa.Column("actor", sa.String(16), nullable=False, server_default="agent"),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("before_state", pg.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("after_state", pg.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("undone_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("target IN ('profile', 'project', 'entry')"),
        sa.CheckConstraint("action IN ('add', 'update', 'supersede', 'delete')"),
        sa.CheckConstraint("actor IN ('user', 'agent', 'review')"),
    )
    op.create_index("ix_memory_events_owner_created", "memory_events", ["user_id", "created_at"])
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_messages_content_trgm ON messages USING gin (content gin_trgm_ops)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_messages_content_trgm")
    op.drop_table("memory_events")
    op.drop_table("memory_core_blocks")
    op.drop_constraint("ck_user_settings_memory_write_mode", "user_settings", type_="check")
    op.drop_column("user_settings", "memory_write_mode")
