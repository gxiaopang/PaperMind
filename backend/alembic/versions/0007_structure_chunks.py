"""Structure-aware parents and source spans; preserve legacy chunks unchanged."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0007_structure"
down_revision = "0006_m5"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "chunk_parents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("version_id", sa.Uuid(), nullable=False),
        sa.Column("section_id", sa.Uuid(), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(24), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("locations", postgresql.JSONB(), nullable=False),
        sa.Column("child_count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "version_id", "id"),
        sa.UniqueConstraint("user_id", "version_id", "ordinal"),
        sa.CheckConstraint("child_count > 0"),
        sa.ForeignKeyConstraint(
            ["user_id", "paper_id", "version_id"],
            ["paper_versions.user_id", "paper_versions.paper_id", "paper_versions.id"],
        ),
        sa.ForeignKeyConstraint(
            ["user_id", "version_id", "section_id"],
            ["sections.user_id", "sections.version_id", "sections.id"],
        ),
    )
    op.add_column("chunks", sa.Column("parent_id", sa.Uuid(), nullable=True))
    op.add_column(
        "chunks",
        sa.Column(
            "source_spans", postgresql.JSONB(), server_default=sa.text("'[]'"), nullable=False
        ),
    )
    op.create_foreign_key(
        "fk_chunk_parent",
        "chunks",
        "chunk_parents",
        ["user_id", "version_id", "parent_id"],
        ["user_id", "version_id", "id"],
    )
    op.create_index(
        "ix_chunks_owner_parent", "chunks", ["user_id", "version_id", "parent_id", "ordinal"]
    )


def downgrade():
    op.drop_index("ix_chunks_owner_parent", table_name="chunks")
    op.drop_constraint("fk_chunk_parent", "chunks", type_="foreignkey")
    op.drop_column("chunks", "source_spans")
    op.drop_column("chunks", "parent_id")
    op.drop_table("chunk_parents")
