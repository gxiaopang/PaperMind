import json
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet

from app.core.config import Settings
from app.core.security import hash_token
from app.db.models import AccountToken, MailOutbox, User, utcnow
from app.services.accounts.mail import issue_account_mail, send_outbox


@pytest.fixture
def mail_context(monkeypatch):
    settings = Settings(
        _env_file=None,
        tracing_enabled=False,
        mail_backend="smtp",
        mail_encryption_key=Fernet.generate_key().decode(),
        smtp_host="smtp.example.org",
        smtp_username="account@example.org",
        smtp_password="smtp-secret",
        mail_from="sender@example.org",
        public_base_url="https://papers.example.org",
    )
    user = User(id=uuid4(), email="reader@example.org", status="active")
    db = AsyncMock()
    db.add = MagicMock()
    db.get.return_value = user
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = db
    smtp_send = AsyncMock(return_value=({}, "accepted"))
    monkeypatch.setattr("app.services.accounts.mail.aiosmtplib.send", smtp_send)
    return {"settings": settings, "session_factory": factory}, db, user, smtp_send


async def enqueue(ctx, db, user, kind="verify"):
    await issue_account_mail(db, user, kind, ctx["settings"])
    token, item = [call.args[0] for call in db.add.call_args_list]
    assert isinstance(token, AccountToken)
    assert isinstance(item, MailOutbox)
    # These defaults are normally populated by the database INSERT.
    item.id = uuid4()
    item.attempts = 0
    db.scalar.side_effect = [item, None]
    return token, item


@pytest.mark.parametrize("kind,title", [("verify", "验证邮箱"), ("reset", "重置密码")])
async def test_account_email_uses_smtp_and_fragment_link(mail_context, kind, title):
    ctx, db, user, smtp_send = mail_context
    token, item = await enqueue(ctx, db, user, kind)
    payload = json.loads(
        Fernet(ctx["settings"].mail_encryption_key.get_secret_value()).decrypt(
            item.encrypted_payload.encode()
        )
    )
    assert token.token_hash == hash_token(payload["token"])
    assert user.email not in item.encrypted_payload
    assert payload["token"] not in item.encrypted_payload
    assert await send_outbox(ctx) == 1
    message = smtp_send.call_args.args[0]
    assert str(message["From"]) == "PaperMind <sender@example.org>"
    assert str(message["To"]) == user.email
    assert str(message["Subject"]) == f"PaperMind · {title}"
    url = next(line for line in message.get_content().splitlines() if line.startswith("https://"))
    parts = urlsplit(url)
    assert parts.netloc == "papers.example.org"
    assert not parts.query
    assert parse_qs(parts.fragment) == {"action": [kind], "token": [payload["token"]]}
    assert smtp_send.call_args.kwargs == {
        "hostname": "smtp.example.org",
        "port": 465,
        "username": "account@example.org",
        "password": "smtp-secret",
        "use_tls": True,
        "start_tls": False,
        "timeout": 20,
    }
    assert item.sent_at is not None
    assert item.encrypted_payload == ""
    assert item.last_error is None


async def test_smtp_failure_keeps_encrypted_mail_for_retry(mail_context):
    ctx, db, user, smtp_send = mail_context
    _, item = await enqueue(ctx, db, user)
    encrypted = item.encrypted_payload
    smtp_send.side_effect = TimeoutError("sensitive-provider-error")
    before = utcnow()
    assert await send_outbox(ctx) == 0
    assert item.sent_at is None
    assert item.encrypted_payload == encrypted
    assert item.attempts == 1
    assert item.available_at >= before + timedelta(seconds=60)
    assert item.last_error == "TimeoutError"
    db.commit.assert_awaited()


async def test_memory_backend_does_not_send_real_mail(mail_context):
    ctx, db, user, smtp_send = mail_context
    ctx["settings"].mail_backend = "memory"
    await enqueue(ctx, db, user)
    assert await send_outbox(ctx) == 1
    smtp_send.assert_not_awaited()
    assert ctx["mailbox"][0]["email"] == user.email
    assert ctx["mailbox"][0]["kind"] == "verify"
