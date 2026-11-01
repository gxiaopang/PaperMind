import secrets
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_token
from app.db.models import Session, User, utcnow
from app.db.session import get_db

DB = Annotated[AsyncSession, Depends(get_db)]


@dataclass
class Identity:
    user: User
    session: Session


async def require_identity(request: Request, db: DB) -> Identity:
    token = request.cookies.get("pm_session", "")
    row = (
        await db.execute(
            select(User, Session)
            .join(Session, Session.user_id == User.id)
            .where(
                Session.token_hash == hash_token(token),
                Session.revoked_at.is_(None),
                Session.expires_at > utcnow(),
                User.status == "active",
            )
        )
    ).first()
    if row is None:
        raise HTTPException(401, "请先登录")
    identity = Identity(row[0], row[1])
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        csrf = request.headers.get("X-CSRF-Token", "")
        if not csrf or not secrets.compare_digest(hash_token(csrf), identity.session.csrf_hash):
            raise HTTPException(403, "安全校验失败，请刷新页面")
    return identity


CurrentIdentity = Annotated[Identity, Depends(require_identity)]


async def require_verified(identity: CurrentIdentity) -> User:
    if identity.user.email_verified_at is None:
        raise HTTPException(403, "请先验证邮箱")
    return identity.user


VerifiedUser = Annotated[User, Depends(require_verified)]
