"""M5 personal answer settings and per-run preference snapshots"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0006_m5"
down_revision = "5a92833f6d39"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_settings",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("answer_language", sa.String(length=8), nullable=False),
        sa.Column("detail_level", sa.String(length=16), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("answer_language IN ('zh', 'en')"),
        sa.CheckConstraint("detail_level IN ('concise', 'standard', 'detailed')"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.add_column(
        "chat_runs",
        sa.Column(
            "preferences",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
    )


def downgrade():
    op.drop_column("chat_runs", "preferences")
    op.drop_table("user_settings")
