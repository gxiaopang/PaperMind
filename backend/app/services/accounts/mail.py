import json
from datetime import timedelta
from email.message import EmailMessage
from email.utils import formataddr
from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import aiosmtplib
import structlog
from cryptography.fernet import Fernet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.security import hash_token, new_token
from app.db.models import AccountToken, MailOutbox, User, utcnow
from app.services.accounts.codes import code_digest

log = structlog.get_logger()


async def issue_account_mail(db: AsyncSession, user: User, kind: str, settings: Settings) -> None:
    raw = new_token()
    expires = utcnow() + timedelta(minutes=settings.account_token_ttl_minutes)
    db.add(AccountToken(user_id=user.id, token_hash=hash_token(raw), kind=kind, expires_at=expires))
    payload = json.dumps({"email": user.email, "kind": kind, "token": raw})
    cipher = Fernet(settings.mail_encryption_key.get_secret_value().encode())
    db.add(
        MailOutbox(
            user_id=user.id,
            encrypted_payload=cipher.encrypt(payload.encode()).decode(),
            expires_at=expires,
        )
    )


async def send_outbox(ctx: dict[str, Any]) -> int:
    """Durable mail outbox; ARQ cron also retries jobs lost during enqueue."""
    settings: Settings = ctx["settings"]
    sent = 0
    async with ctx["session_factory"]() as db:
        for _ in range(20):
            item = await db.scalar(
                select(MailOutbox)
                .where(
                    MailOutbox.sent_at.is_(None),
                    MailOutbox.available_at <= utcnow(),
                    MailOutbox.expires_at > utcnow(),
                    MailOutbox.attempts < 5,
                )
                .order_by(MailOutbox.available_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if item is None:
                break
            user = await db.get(User, item.user_id) if item.user_id is not None else None
            if item.user_id is not None and (user is None or user.status != "active"):
                item.encrypted_payload = ""
                item.sent_at = utcnow()
                await db.commit()
                continue
            try:
                cipher = Fernet(settings.mail_encryption_key.get_secret_value().encode())
                payload = json.loads(cipher.decrypt(item.encrypted_payload.encode()))
                if payload["kind"] == "email_code":
                    from app.db.models import EmailCode

                    challenge = await db.get(EmailCode, payload["email"], populate_existing=True)
                    if (
                        challenge is None
                        or challenge.used_at is not None
                        or challenge.expires_at <= utcnow()
                        or challenge.code_hash
                        != code_digest(payload["email"], payload["code"], settings)
                        or challenge.attempts >= settings.email_code_max_attempts
                    ):
                        item.encrypted_payload = ""
                        item.sent_at = utcnow()
                        await db.commit()
                        continue
                    title = "邮箱验证码"
                    content = (
                        f"你的 PaperMind 邮箱验证码为：{payload['code']}\n\n"
                        f"验证码在 {settings.email_code_ttl_minutes} 分钟内有效，仅可使用一次。"
                        "请在页面中输入验证码，不要将验证码告诉他人。"
                    )
                else:
                    url = f"{settings.public_base_url}/#{urlencode({'action': payload['kind'], 'token': payload['token']})}"
                    title = "验证邮箱" if payload["kind"] == "verify" else "重置密码"
                    content = f"请打开以下链接{title}，链接在 {settings.account_token_ttl_minutes} 分钟内有效：\n{url}"
                message = EmailMessage()
                message["From"] = formataddr((settings.mail_from_name, settings.mail_from))
                message["To"] = payload["email"]
                message["Subject"] = f"PaperMind · {title}"
                message["Message-ID"] = f"<{item.id}@papermind.local>"
                message.set_content(f"{content}\n\n如果不是您本人操作，请忽略此邮件。")
                if settings.mail_backend == "memory":
                    ctx.setdefault("mailbox", []).append(
                        {"user_id": str(item.user_id) if item.user_id else None, **payload}
                    )
                else:
                    await aiosmtplib.send(
                        message,
                        hostname=settings.smtp_host,
                        port=settings.smtp_port,
                        username=settings.smtp_username or None,
                        password=settings.smtp_password.get_secret_value() or None,
                        use_tls=settings.smtp_use_ssl,
                        start_tls=settings.smtp_starttls,
                        timeout=20,
                    )
                item.sent_at = utcnow()
                item.encrypted_payload = ""  # Remove bearer secrets after delivery.
                item.last_error = None
                sent += 1
            except Exception as exc:
                item.attempts += 1
                item.available_at = utcnow() + timedelta(seconds=min(30 * 2**item.attempts, 1800))
                item.last_error = type(exc).__name__
                log.warning(
                    "mail_delivery_failed", job_id=str(item.id), error_type=type(exc).__name__
                )
            await db.commit()
        # Expired payloads no longer need to retain account tokens or email addresses.
        from sqlalchemy import update

        await db.execute(
            update(MailOutbox).where(MailOutbox.expires_at <= utcnow()).values(encrypted_payload="")
        )
        await db.commit()
    return sent


async def wake_mail_worker(redis: Any, user_id: UUID) -> None:
    try:
        await redis.enqueue_job(
            "send_outbox", _queue_name="papermind:mail", _job_id=f"mail:{user_id}:{new_token()}"
        )
    except Exception as exc:
        log.warning("mail_enqueue_delayed", error_type=type(exc).__name__)
