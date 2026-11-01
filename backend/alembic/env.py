import asyncio

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from app.core.config import get_settings
from app.db.models import Base

target_metadata = Base.metadata


def run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # pg_search expression/tokenizer indexes are extension-owned DDL. SQLAlchemy
        # cannot round-trip their opclasses; keep the explicit M3 migration authoritative.
        include_object=lambda obj, name, kind, reflected, compare_to: (
            not (kind == "index" and name == "ix_chunks_bm25")
        ),
        include_name=lambda name, kind, parents: kind != "table" or name in target_metadata.tables,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_online():
    engine = create_async_engine(
        get_settings().database_url.get_secret_value(), poolclass=pool.NullPool
    )
    async with engine.connect() as connection:
        await connection.run_sync(run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    context.configure(
        url=get_settings().database_url.get_secret_value(),
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(run_online())
