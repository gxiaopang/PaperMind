"""M3 scoped BM25 index and operation-scoped retrieval usage."""

import sqlalchemy as sa

from alembic import op

revision = "0004_m3"
down_revision = "a3e44cd23111"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column("usage_records", "version_id", nullable=True)
    op.add_column("usage_records", sa.Column("operation_id", sa.Uuid(), nullable=True))
    op.create_index("ix_usage_operation", "usage_records", ["user_id", "operation_id"])
    op.create_check_constraint(
        "ck_usage_resource", "usage_records", "version_id IS NOT NULL OR operation_id IS NOT NULL"
    )
    # ICU handles Chinese and Latin text. embedding_input already carries title,
    # section path and body. All candidate filter columns belong to this index.
    # Verified on the deployment's locked pg_search 0.26.3.
    op.execute("""
        CREATE INDEX ix_chunks_bm25 ON chunks USING bm25
        (id, user_id, collection_id, paper_id, version_id, indexable,
         (embedding_input::pdb.icu), section_path)
        WITH (key_field='id')
    """)


def downgrade():
    op.execute("DROP INDEX ix_chunks_bm25")
    op.drop_index("ix_usage_operation", table_name="usage_records")
    op.drop_constraint("ck_usage_resource", "usage_records", type_="check")
    op.drop_column("usage_records", "operation_id")
    # Preserve paid retrieval audit records. A downgrade cannot restore NOT NULL
    # while operation-only records exist; explicitly fail instead of deleting them.
    op.alter_column("usage_records", "version_id", nullable=False)
