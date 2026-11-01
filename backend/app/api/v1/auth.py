import asyncio
from datetime import timedelta
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from app.api.deps import DB, CurrentIdentity
from app.core.rate_limit import limit_auth
from app.core.security import DUMMY_HASH, hash_token, hasher, new_token, verify_password
from app.db.models import AccountToken, Session, User, utcnow
from app.schemas.accounts import (
    Credentials,
    EmailCodeInput,
    EmailInput,
    RegistrationInput,
    ResetInput,
    TokenInput,
    UserView,
)
from app.services.accounts.codes import consume_email_code, issue_email_code
from app.services.accounts.mail import issue_account_mail, wake_mail_worker

router = APIRouter(prefix="/auth", tags=["账户"])
GENERIC_MESSAGE = {"message": "如果邮箱符合条件，我们将发送邮件，请查收。"}


@router.get("/csrf")
async def csrf(request: Request, response: Response) -> dict[str, str]:
    token = request.cookies.get("pm_csrf") or new_token()
    settings = request.app.state.settings
    response.set_cookie(
        "pm_csrf",
        token,
        secure=settings.secure_cookie,
        samesite="strict",
        max_age=settings.session_ttl_days * 86400,
    )
    return {"csrf_token": token}


@router.post("/registration-code", status_code=202)
async def registration_code(body: EmailInput, request: Request, db: DB) -> dict[str, str | int]:
    settings = request.app.state.settings
    await limit_auth(request, str(body.email), settings)
    await issue_email_code(db, str(body.email), settings)
    await db.commit()
    await wake_mail_worker(request.app.state.redis, uuid4())
    return {
        "message": "验证码邮件已加入发送队列，请查收，未收到时请检查垃圾邮件。",
        "resend_after": settings.email_code_resend_seconds,
    }


@router.post("/register", status_code=202)
async def register(body: RegistrationInput, request: Request, db: DB) -> dict[str, str]:
    settings = request.app.state.settings
    await limit_auth(request, str(body.email), settings)
    await consume_email_code(db, str(body.email), body.code, settings)
    encoded = await asyncio.to_thread(hasher.hash, body.password)
    user_id = await db.scalar(
        insert(User)
        .values(email=str(body.email), password_hash=encoded, email_verified_at=utcnow())
        .on_conflict_do_nothing(index_elements=[User.email])
        .returning(User.id)
    )
    await db.commit()
    if user_id is None:
        raise HTTPException(409, "此邮箱已有账户，请登录或找回密码")
    return {"message": "注册成功，邮箱已验证，请登录。"}


@router.post("/login", response_model=UserView)
async def login(body: Credentials, request: Request, response: Response, db: DB) -> User:
    settings = request.app.state.settings
    await limit_auth(request, str(body.email), settings)
    user = await db.scalar(select(User).where(User.email == str(body.email)).with_for_update())
    valid = await asyncio.to_thread(
        verify_password, user.password_hash if user else DUMMY_HASH, body.password
    )
    if user is None or not valid or user.status != "active":
        raise HTTPException(401, "邮箱或密码错误")
    old_token = request.cookies.get("pm_session", "")
    await db.execute(
        update(Session)
        .where(Session.token_hash == hash_token(old_token))
        .values(revoked_at=utcnow())
    )
    token, csrf_token = new_token(), new_token()
    db.add(
        Session(
            user_id=user.id,
            token_hash=hash_token(token),
            csrf_hash=hash_token(csrf_token),
            expires_at=utcnow() + timedelta(days=settings.session_ttl_days),
        )
    )
    if hasher.check_needs_rehash(user.password_hash):
        user.password_hash = await asyncio.to_thread(hasher.hash, body.password)
    await db.commit()
    response.set_cookie(
        "pm_session",
        token,
        httponly=True,
        secure=settings.secure_cookie,
        samesite="strict",
        max_age=settings.session_ttl_days * 86400,
    )
    response.set_cookie(
        "pm_csrf",
        csrf_token,
        secure=settings.secure_cookie,
        samesite="strict",
        max_age=settings.session_ttl_days * 86400,
    )
    return user


@router.get("/me", response_model=UserView)
async def me(identity: CurrentIdentity) -> User:
    return identity.user


@router.post("/logout", status_code=204)
async def logout(identity: CurrentIdentity, db: DB, response: Response) -> None:
    identity.session.revoked_at = utcnow()
    await db.commit()
    response.delete_cookie("pm_session", httponly=True, samesite="strict")
    response.delete_cookie("pm_csrf", samesite="strict")


@router.post("/verify-email")
async def verify_email(body: TokenInput, request: Request, db: DB) -> dict[str, str]:
    await limit_auth(request, "token:" + hash_token(body.token), request.app.state.settings)
    user = await lock_token_owner(db, body.token, "verify")
    user.email_verified_at = user.email_verified_at or utcnow()
    await db.execute(
        update(AccountToken)
        .where(
            AccountToken.user_id == user.id,
            AccountToken.kind == "verify",
            AccountToken.used_at.is_(None),
        )
        .values(used_at=utcnow())
    )
    await db.commit()
    return {"message": "邮箱验证成功"}


@router.post("/resend-verification", status_code=202)
async def resend(body: EmailInput, request: Request, db: DB) -> dict[str, str | int]:
    return await registration_code(body, request, db)


@router.post("/verify-email-code")
async def verify_email_code(
    body: EmailCodeInput, request: Request, db: DB, identity: CurrentIdentity
) -> dict[str, str]:
    await limit_auth(request, identity.user.email, request.app.state.settings)
    await consume_email_code(db, identity.user.email, body.code, request.app.state.settings)
    user = await db.scalar(select(User).where(User.id == identity.user.id).with_for_update())
    assert user is not None
    if user.status != "active":
        raise HTTPException(401, "账户不可用")
    user.email_verified_at = user.email_verified_at or utcnow()
    await db.commit()
    return {"message": "邮箱验证成功"}


@router.post("/forgot-password", status_code=202)
async def forgot(body: EmailInput, request: Request, db: DB) -> dict[str, str]:
    return await request_email(body, request, db, "reset")


async def request_email(body: EmailInput, request: Request, db: DB, kind: str) -> dict[str, str]:
    settings = request.app.state.settings
    await limit_auth(request, str(body.email), settings)
    user = await db.scalar(
        select(User).where(User.email == str(body.email), User.status == "active").with_for_update()
    )
    if user and (kind == "reset" or user.email_verified_at is None):
        await issue_account_mail(db, user, kind, settings)
        await db.commit()
        await wake_mail_worker(request.app.state.redis, user.id)
    return GENERIC_MESSAGE


@router.post("/reset-password")
async def reset(body: ResetInput, request: Request, db: DB, response: Response) -> dict[str, str]:
    await limit_auth(request, "token:" + hash_token(body.token), request.app.state.settings)
    user = await lock_token_owner(db, body.token, "reset")
    user.password_hash = await asyncio.to_thread(hasher.hash, body.password)
    await db.execute(
        update(AccountToken)
        .where(
            AccountToken.user_id == user.id,
            AccountToken.kind == "reset",
            AccountToken.used_at.is_(None),
        )
        .values(used_at=utcnow())
    )
    await db.execute(
        update(Session)
        .where(Session.user_id == user.id, Session.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )
    await db.commit()
    response.delete_cookie("pm_session", httponly=True, samesite="strict")
    response.delete_cookie("pm_csrf", samesite="strict")
    return {"message": "密码已重置，请重新登录"}


async def lock_token_owner(db: DB, raw: str, kind: str) -> User:
    # Always lock the user before tokens. Two different valid links for one account
    # must serialize without deadlocking while invalidating sibling links.
    filters = (
        AccountToken.token_hash == hash_token(raw),
        AccountToken.kind == kind,
        AccountToken.used_at.is_(None),
        AccountToken.expires_at > utcnow(),
    )
    owner_id = select(AccountToken.user_id).where(*filters).scalar_subquery()
    user = await db.scalar(select(User).where(User.id == owner_id).with_for_update())
    if user is None or user.status != "active":
        raise HTTPException(400, "链接无效、已使用或已过期")
    token = await db.scalar(select(AccountToken).where(*filters).with_for_update())
    if token is None:
        raise HTTPException(400, "链接无效、已使用或已过期")
    return user
