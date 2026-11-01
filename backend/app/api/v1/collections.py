from uuid import UUID

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.deps import DB, CurrentIdentity
from app.db.models import Collection, Paper, User
from app.schemas.accounts import CollectionInput, CollectionView
from app.services.ingestion.lifecycle import tombstone

router = APIRouter(prefix="/collections", tags=["知识库"])


async def owned_collection(db: DB, user_id: UUID, collection_id: UUID) -> Collection:
    item = await db.scalar(
        select(Collection).where(Collection.id == collection_id, Collection.user_id == user_id)
    )
    if item is None:
        raise HTTPException(404, "知识库不存在")
    return item


@router.get("", response_model=list[CollectionView])
async def list_collections(identity: CurrentIdentity, db: DB) -> list[Collection]:
    return list(
        await db.scalars(
            select(Collection)
            .where(Collection.user_id == identity.user.id)
            .order_by(Collection.created_at.desc())
            .limit(100)
        )
    )


@router.post("", response_model=CollectionView, status_code=201)
async def create(body: CollectionInput, identity: CurrentIdentity, db: DB) -> Collection:
    item = Collection(user_id=identity.user.id, **body.model_dump())
    db.add(item)
    await db.commit()
    return item


@router.get("/{collection_id}", response_model=CollectionView)
async def get(collection_id: UUID, identity: CurrentIdentity, db: DB) -> Collection:
    return await owned_collection(db, identity.user.id, collection_id)


@router.patch("/{collection_id}", response_model=CollectionView)
async def edit(
    collection_id: UUID, body: CollectionInput, identity: CurrentIdentity, db: DB
) -> Collection:
    item = await owned_collection(db, identity.user.id, collection_id)
    item.name, item.description = body.name, body.description
    await db.commit()
    return item


@router.delete("/{collection_id}", status_code=204)
async def delete(collection_id: UUID, identity: CurrentIdentity, db: DB) -> None:
    await db.scalar(select(User).where(User.id == identity.user.id).with_for_update())
    item = await owned_collection(db, identity.user.id, collection_id)
    papers = list(
        await db.scalars(
            select(Paper).where(
                Paper.user_id == identity.user.id, Paper.collection_id == collection_id
            )
        )
    )
    for paper in papers:
        if not paper.deleted_at:
            await tombstone(db, paper)
        paper.collection_id = None
    await db.flush()
    await db.delete(item)
    await db.commit()
