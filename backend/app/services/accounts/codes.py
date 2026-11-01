import hashlib
import hmac
import json
import secrets
from datetime import timedelta

from cryptography.fernet import Fernet
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import EmailCode, MailOutbox, utcnow


def code_digest(email: str, code: str, settings: Settings) -> str:
    # A keyed hash prevents offline enumeration of the one-million-code space.
    return hmac.new(
        settings.mail_encryption_key.get_secret_value().encode(),
        f"email-code:{email}:{code}".encode(),
        hashlib.sha256,
    ).hexdigest()


async def issue_email_code(db: AsyncSession, email: str, settings: Settings) -> None:
    now = utcnow()
    # Establish a stable row, then serialize sends and consumption for this email.
    await db.execute(
        insert(EmailCode)
        .values(email=email, code_hash="", issued_at=now, expires_at=now, attempts=0)
        .on_conflict_do_nothing(index_elements=[EmailCode.email])
    )
    challenge = await db.scalar(select(EmailCode).where(EmailCode.email == email).with_for_update())
    assert challenge is not None
    remaining = settings.email_code_resend_seconds - (now - challenge.issued_at).total_seconds()
    if challenge.code_hash and remaining > 0:
        raise HTTPException(
            429, "请稍后再获取验证码", headers={"Retry-After": str(max(1, int(remaining) + 1))}
        )
    code = f"{secrets.randbelow(1_000_000):06d}"
    challenge.code_hash = code_digest(email, code, settings)
    challenge.issued_at = now
    challenge.expires_at = now + timedelta(minutes=settings.email_code_ttl_minutes)
    challenge.attempts = 0
    challenge.used_at = None
    cipher = Fernet(settings.mail_encryption_key.get_secret_value().encode())
    payload = json.dumps({"email": email, "kind": "email_code", "code": code})
    db.add(
        MailOutbox(
            user_id=None,
            encrypted_payload=cipher.encrypt(payload.encode()).decode(),
            expires_at=challenge.expires_at,
        )
    )


async def consume_email_code(db: AsyncSession, email: str, code: str, settings: Settings) -> None:
    challenge = await db.scalar(select(EmailCode).where(EmailCode.email == email).with_for_update())
    if (
        challenge is None
        or challenge.used_at is not None
        or challenge.expires_at <= utcnow()
        or challenge.attempts >= settings.email_code_max_attempts
    ):
        raise HTTPException(400, "验证码无效或已过期，请重新获取")
    if not hmac.compare_digest(challenge.code_hash, code_digest(email, code, settings)):
        challenge.attempts += 1
        # Persist failures even though the request ends with an HTTP error.
        await db.commit()
        raise HTTPException(400, "验证码错误，连续输错后需重新获取")
    challenge.used_at = utcnow()
