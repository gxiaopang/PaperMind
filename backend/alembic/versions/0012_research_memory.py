"""Research projects, structured conversation state and attributable memory."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

from alembic import op

revision = "0012_research_memory"
down_revision = "0011_conversation_deletion"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "research_projects",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("user_id", "id"),
    )
    op.add_column("conversations", sa.Column("project_id", sa.Uuid()))
    op.add_column(
        "conversations",
        sa.Column("research_state", pg.JSONB(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.add_column(
        "conversations",
        sa.Column("memory_revision", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_foreign_key(
        "fk_conversation_project",
        "conversations",
        "research_projects",
        ["user_id", "project_id"],
        ["user_id", "id"],
    )
    op.create_table(
        "memory_entries",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("level", sa.String(16), nullable=False),
        sa.Column("project_id", sa.Uuid()),
        sa.Column("conversation_id", sa.Uuid()),
        sa.Column("paper_id", sa.Uuid()),
        sa.Column("kind", sa.String(24), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("epistemic", sa.String(24), nullable=False),
        sa.Column("pinned", sa.Boolean(), nullable=False),
        sa.Column("source_message_id", sa.Uuid()),
        sa.Column("citation_ids", pg.JSONB(), nullable=False),
        sa.Column("reading_anchor", pg.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("fingerprint", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("user_id", "fingerprint"),
        sa.CheckConstraint("level IN ('user', 'project', 'paper', 'conversation')"),
        sa.CheckConstraint("status IN ('candidate', 'active', 'superseded')"),
        sa.CheckConstraint("epistemic IN ('user_statement', 'hypothesis', 'evidence_linked')"),
        sa.CheckConstraint(
            "(level = 'user' AND project_id IS NULL AND conversation_id IS NULL AND paper_id IS NULL) OR "
            "(level = 'project' AND project_id IS NOT NULL AND conversation_id IS NULL AND paper_id IS NULL) OR "
            "(level = 'paper' AND paper_id IS NOT NULL AND conversation_id IS NULL) OR "
            "(level = 'conversation' AND conversation_id IS NOT NULL AND paper_id IS NULL)"
        ),
        sa.ForeignKeyConstraint(
            ["user_id", "project_id"], ["research_projects.user_id", "research_projects.id"]
        ),
        sa.ForeignKeyConstraint(
            ["user_id", "conversation_id"], ["conversations.user_id", "conversations.id"]
        ),
        sa.ForeignKeyConstraint(["user_id", "paper_id"], ["papers.user_id", "papers.id"]),
        sa.ForeignKeyConstraint(
            ["user_id", "source_message_id"], ["messages.user_id", "messages.id"]
        ),
    )
    op.create_index(
        "ix_memory_owner_project_status", "memory_entries", ["user_id", "project_id", "status"]
    )


def downgrade():
    op.drop_table("memory_entries")
    op.drop_constraint("fk_conversation_project", "conversations", type_="foreignkey")
    op.drop_column("conversations", "memory_revision")
    op.drop_column("conversations", "research_state")
    op.drop_column("conversations", "project_id")
    op.drop_table("research_projects")
