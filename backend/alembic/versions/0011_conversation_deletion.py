"""Allow users to delete conversations while retaining usage accounting."""

import sqlalchemy as sa

from alembic import op

revision = "0011_conversation_deletion"
down_revision = "0010_email_codes"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("conversations", sa.Column("deleted_at", sa.DateTime(timezone=True)))


def downgrade():
    op.drop_column("conversations", "deleted_at")
