"""Version-local L0 maps and standardized section navigation."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0009_paper_map"
down_revision = "0008_academic"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "paper_versions",
        sa.Column("paper_map", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.add_column(
        "sections", sa.Column("section_std", sa.String(32), nullable=False, server_default="other")
    )


def downgrade():
    op.drop_column("sections", "section_std")
    op.drop_column("paper_versions", "paper_map")
