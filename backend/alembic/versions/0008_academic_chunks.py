"""Academic chunk roles and version-isolated reference entries."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0008_academic"
down_revision = "0007_structure"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "chunks",
        sa.Column(
            "chunk_metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")
        ),
    )
    op.add_column(
        "paper_versions",
        sa.Column(
            "reference_entries", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'")
        ),
    )


def downgrade():
    op.drop_column("paper_versions", "reference_entries")
    op.drop_column("chunks", "chunk_metadata")
