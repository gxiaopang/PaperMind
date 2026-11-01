import asyncio
import json
from datetime import timedelta

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import func, select, update

from app.db.models import EmailCode, MailOutbox, User, utcnow
from app.services.accounts.mail import issue_account_mail
from tests.conftest import delivered, login, post, register

pytestmark = pytest.mark.integration
EMAIL = "code@example.org"
PASSWORD = "Trial123"


async def send_code(client, app, email=EMAIL):
    response = await post(client, "/auth/registration-code", {"email": email})
    assert response.status_code == 202, response.text
    messages = await delivered(app)
    return next(m["code"] for m in messages if m["email"] == email.casefold())


async def submit(client, code, email=EMAIL, **extra):
    return await post(
        client, "/auth/register", {"email": email, "password": PASSWORD, "code": code, **extra}
    )


async def allow_resend(app, email=EMAIL):
    async with app.state.session_factory() as db:
        await db.execute(
            update(EmailCode)
            .where(EmailCode.email == email)
            .values(issued_at=utcnow() - timedelta(seconds=61))
        )
        await db.commit()


async def test_code_is_required_before_any_account_is_created(client, app):
    assert (
        await post(client, "/auth/register", {"email": EMAIL, "password": PASSWORD})
    ).status_code == 422
    assert (await submit(client, "123456")).status_code == 400
    code = await send_code(client, app, "Code@Example.org")
    assert len(code) == 6 and code.isascii() and code.isdigit()
    async with app.state.session_factory() as db:
        assert await db.scalar(select(func.count()).select_from(User)) == 0
    assert (await submit(client, code, password="short12")).status_code == 422
    assert (await submit(client, code, role="admin")).status_code == 422
    assert (await submit(client, code)).status_code == 202
    assert (await login(client, EMAIL, PASSWORD)).status_code == 200
    assert (await client.get("/api/v1/auth/me")).json()["email_verified_at"]
    assert (await submit(client, code)).status_code == 400


async def test_code_is_bound_to_email_and_failed_attempts_persist(client, app):
    code = await send_code(client, app)
    assert (await submit(client, code, email="other@example.org")).status_code == 400
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(5):
        assert (await submit(client, wrong)).status_code == 400
    assert (await submit(client, code)).status_code == 400
    async with app.state.session_factory() as db:
        assert (await db.get(EmailCode, EMAIL)).attempts == 5
        assert await db.scalar(select(func.count()).select_from(User)) == 0
    await allow_resend(app)
    new_code = await send_code(client, app)
    assert (await submit(client, new_code)).status_code == 202


async def test_expired_code_and_invalid_shapes_are_rejected(client, app):
    code = await send_code(client, app)
    for value in ["12345", "1234567", "１２３４５６", "abcdef"]:
        assert (await submit(client, value)).status_code == 422
    async with app.state.session_factory() as db:
        await db.execute(update(EmailCode).values(expires_at=utcnow() - timedelta(seconds=1)))
        await db.commit()
    assert (await submit(client, code)).status_code == 400


async def test_resend_cooldown_invalidates_previous_code(client, app, monkeypatch):
    values = iter([12345, 23456])
    monkeypatch.setattr("app.services.accounts.codes.secrets.randbelow", lambda _: next(values))
    code = await send_code(client, app)
    assert code == "012345"
    limited = await post(client, "/auth/registration-code", {"email": EMAIL})
    assert limited.status_code == 429 and 1 <= int(limited.headers["Retry-After"]) <= 61
    await allow_resend(app)
    new_code = await send_code(client, app)
    assert new_code == "023456"
    assert (await submit(client, code)).status_code == 400
    assert (await submit(client, new_code)).status_code == 202


async def test_concurrent_registration_consumes_code_once(client, app):
    code = await send_code(client, app)
    csrf = (await client.get("/api/v1/auth/csrf")).json()["csrf_token"]
    responses = await asyncio.gather(
        *(
            client.post(
                "/api/v1/auth/register",
                json={"email": EMAIL, "password": PASSWORD, "code": code},
                headers={"X-CSRF-Token": csrf},
            )
            for _ in range(2)
        )
    )
    assert sorted(r.status_code for r in responses) == [202, 400]
    async with app.state.session_factory() as db:
        assert await db.scalar(select(func.count()).select_from(User)) == 1


async def test_concurrent_sends_queue_only_one_code(client, app):
    csrf = (await client.get("/api/v1/auth/csrf")).json()["csrf_token"]
    responses = await asyncio.gather(
        *(
            client.post(
                "/api/v1/auth/registration-code",
                json={"email": EMAIL},
                headers={"X-CSRF-Token": csrf},
            )
            for _ in range(2)
        )
    )
    assert sorted(r.status_code for r in responses) == [202, 429]
    assert len(await delivered(app)) == 1


async def test_existing_account_password_is_never_overwritten(client, app):
    await register(client, EMAIL, PASSWORD)
    await allow_resend(app)
    code = await send_code(client, app)
    assert (await submit(client, code, password="Different123")).status_code == 409
    assert (await login(client, EMAIL, PASSWORD)).status_code == 200
    assert (await login(client, EMAIL, "Different123")).status_code == 401
    assert (await submit(client, code)).status_code == 400


async def test_existing_unverified_user_can_verify_with_code(client, app):
    await register(client, EMAIL, PASSWORD)
    async with app.state.session_factory() as db:
        await db.execute(update(User).values(email_verified_at=None))
        await db.commit()
    await allow_resend(app)
    code = await send_code(client, app)
    assert (await post(client, "/auth/verify-email-code", {"code": code})).status_code == 401
    await login(client, EMAIL, PASSWORD)
    assert (await post(client, "/auth/verify-email-code", {"code": code})).status_code == 200
    assert (await client.get("/api/v1/auth/me")).json()["email_verified_at"]
    assert (await post(client, "/auth/verify-email-code", {"code": code})).status_code == 400


async def test_old_verification_links_still_work(client, app):
    await register(client, EMAIL, PASSWORD)
    async with app.state.session_factory() as db:
        user = await db.scalar(select(User).where(User.email == EMAIL))
        user.email_verified_at = None
        await issue_account_mail(db, user, "verify", app.state.settings)
        await db.commit()
    token = (await delivered(app))[0]["token"]
    assert (await post(client, "/auth/verify-email", {"token": token})).status_code == 200
    assert (await post(client, "/auth/verify-email", {"token": token})).status_code == 400


async def test_code_mail_is_encrypted_and_superseded_mail_is_skipped(client, app, monkeypatch):
    values = iter([12345, 23456])
    monkeypatch.setattr("app.services.accounts.codes.secrets.randbelow", lambda _: next(values))
    await post(client, "/auth/registration-code", {"email": EMAIL})
    async with app.state.session_factory() as db:
        item = await db.scalar(select(MailOutbox))
        assert item.user_id is None
        assert EMAIL not in item.encrypted_payload and "012345" not in item.encrypted_payload
        payload = json.loads(
            Fernet(app.state.settings.mail_encryption_key.get_secret_value()).decrypt(
                item.encrypted_payload.encode()
            )
        )
        assert payload["code"] == "012345"
        assert (await db.get(EmailCode, EMAIL)).code_hash != "012345"
    await allow_resend(app)
    await post(client, "/auth/registration-code", {"email": EMAIL})
    messages = await delivered(app)
    assert len(messages) == 1 and messages[0]["code"] == "023456"
    assert await delivered(app) == []


async def test_email_code_smtp_message_contains_code_not_link(client, app, monkeypatch):
    from unittest.mock import AsyncMock

    smtp = AsyncMock(return_value=({}, "accepted"))
    monkeypatch.setattr("app.services.accounts.mail.aiosmtplib.send", smtp)
    app.state.settings.mail_backend = "smtp"
    await post(client, "/auth/registration-code", {"email": EMAIL})
    await delivered(app)
    message = smtp.call_args.args[0]
    assert str(message["To"]) == EMAIL
    assert str(message["Subject"]) == "PaperMind · 邮箱验证码"
    assert "10 分钟" in message.get_content() and "http" not in message.get_content()
