"""Email verification codes before account creation."""

import sqlalchemy as sa

from alembic import op

revision = "0010_email_codes"
down_revision = "0009_paper_map"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "email_codes",
        sa.Column("email", sa.String(320), primary_key=True),
        sa.Column("code_hash", sa.String(64), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_email_codes_expires_at", "email_codes", ["expires_at"])
    op.alter_column("mail_outbox", "user_id", nullable=True)


def downgrade():
    op.execute("DELETE FROM mail_outbox WHERE user_id IS NULL")
    op.alter_column("mail_outbox", "user_id", nullable=False)
    op.drop_table("email_codes")
