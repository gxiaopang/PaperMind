import asyncio
from datetime import timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, update

from app.core.security import hash_token
from app.db.models import AccountToken, MailOutbox, Session, User, utcnow
from tests.conftest import delivered, login, post, register

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("password", ["Strong password 123!", "Trial123"])
async def test_registration_verification_login_and_logout(client, app, password):
    assert (await register(client, "Alice@Example.org", password=password)).status_code == 202
    assert (await login(client, "alice@example.org", password=password)).status_code == 200
    me = (await client.get("/api/v1/auth/me")).json()
    assert me["email"] == "alice@example.org" and me["email_verified_at"] is not None
    assert "password_hash" not in me
    cookie = client.cookies.get("pm_session")
    assert (await post(client, "/auth/logout")).status_code == 204
    client.cookies.set("pm_session", cookie)
    assert (await client.get("/api/v1/auth/me")).status_code == 401
    async with app.state.session_factory() as db:
        stored = await db.scalar(select(Session))
        assert stored.token_hash == hash_token(cookie)
        mail = await db.scalar(select(MailOutbox))
        assert mail.encrypted_payload == ""


async def test_two_users_cannot_access_each_others_collections(client, app):
    await register(client)
    await login(client)
    created = await post(client, "/collections", {"name": "A 的研究", "description": "private"})
    assert created.status_code == 201
    collection_id = created.json()["id"]
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost:8080",
        headers={"Origin": "http://localhost:8080"},
    ) as b:
        await register(b, "b@example.org")
        assert (await login(b, "b@example.org")).status_code == 200
        assert (await b.get("/api/v1/collections")).json() == []
        assert (await b.get(f"/api/v1/collections/{collection_id}")).status_code == 404
        assert (
            await post(b, f"/collections/{collection_id}", {"name": "stolen"}, method="PATCH")
        ).status_code == 404
        assert (await post(b, f"/collections/{collection_id}", method="DELETE")).status_code == 404
        forged = await post(b, "/collections", {"name": "forged", "user_id": created.json()["id"]})
        assert forged.status_code == 422
        assert (await post(b, "/collections", {"name": "B 的研究"})).status_code == 201
        async with app.state.session_factory() as db:
            user = await db.scalar(select(User).where(User.email == "b@example.org"))
            user.role = "admin"
            await db.commit()
        assert (await b.get(f"/api/v1/collections/{collection_id}")).status_code == 404
    assert len((await client.get("/api/v1/collections")).json()) == 1
    assert (
        await post(client, f"/collections/{collection_id}", {"name": "renamed"}, method="PATCH")
    ).status_code == 200
    assert (await post(client, f"/collections/{collection_id}", method="DELETE")).status_code == 204


async def test_reset_revokes_all_sessions_and_cannot_be_reused(client, app):
    await register(client)
    await login(client)
    old = client.cookies.get("pm_session")
    await post(client, "/auth/forgot-password", {"email": "a@example.org"})
    mail = await delivered(app)
    token = next(item["token"] for item in mail if item["kind"] == "reset")
    body = {"token": token, "password": "Reset123"}
    assert (await post(client, "/auth/reset-password", body)).status_code == 200
    client.cookies.set("pm_session", old)
    assert (await client.get("/api/v1/auth/me")).status_code == 401
    assert (await login(client)).status_code == 401
    assert (await login(client, password=body["password"])).status_code == 200
    assert (await post(client, "/auth/reset-password", body)).status_code == 400


async def test_expired_token_and_session_are_rejected(client, app):
    await register(client)
    await login(client)
    await post(client, "/auth/forgot-password", {"email": "a@example.org"})
    token = (await delivered(app))[0]["token"]
    async with app.state.session_factory() as db:
        await db.execute(update(AccountToken).values(expires_at=utcnow() - timedelta(seconds=1)))
        await db.execute(update(Session).values(expires_at=utcnow() - timedelta(seconds=1)))
        await db.commit()
    assert (
        await post(client, "/auth/reset-password", {"token": token, "password": "Reset123"})
    ).status_code == 400
    assert (await client.get("/api/v1/auth/me")).status_code == 401


async def test_csrf_origin_and_session_binding(client):
    await register(client)
    await login(client)
    assert (await client.post("/api/v1/collections", json={"name": "bad"})).status_code == 403
    token = (await client.get("/api/v1/auth/csrf")).json()["csrf_token"]
    assert (
        await client.post(
            "/api/v1/collections",
            json={"name": "bad"},
            headers={"Origin": "https://evil.example", "X-CSRF-Token": token},
        )
    ).status_code == 403
    # Matching attacker-controlled cookie/header still cannot replace the session's CSRF secret.
    for cookie in client.cookies.jar:
        if cookie.name == "pm_csrf":
            cookie.value = "tampered-token"
    assert (
        await client.post(
            "/api/v1/collections", json={"name": "bad"}, headers={"X-CSRF-Token": "tampered-token"}
        )
    ).status_code == 403
    client.cookies.clear()
    assert (await client.get("/api/v1/collections")).status_code == 401


async def test_session_rotation_and_secure_cookie(client, app):
    await register(client)
    await login(client)
    old = client.cookies.get("pm_session")
    result = await login(client)
    assert client.cookies.get("pm_session") != old
    assert "HttpOnly" in result.headers.get_list("set-cookie")[0]
    assert "SameSite=strict" in result.headers.get_list("set-cookie")[0]
    async with app.state.session_factory() as db:
        previous = await db.scalar(select(Session).where(Session.token_hash == hash_token(old)))
        assert previous.revoked_at is not None


async def test_disabled_account_cannot_login_or_continue_session(client, app):
    await register(client)
    await login(client)
    async with app.state.session_factory() as db:
        await db.execute(update(User).values(status="disabled"))
        await db.commit()
    assert (await client.get("/api/v1/auth/me")).status_code == 401
    assert (await login(client)).status_code == 401


async def test_role_injection_and_account_enumeration(client, app):
    assert (
        await post(
            client,
            "/auth/register",
            {"email": "a@example.org", "password": "Strong password 123!", "role": "admin"},
        )
    ).status_code == 422
    await register(client)
    missing = await post(client, "/auth/forgot-password", {"email": "missing@example.org"})
    existing = await post(client, "/auth/forgot-password", {"email": "a@example.org"})
    assert missing.json() == existing.json()
    assert not [mail for mail in await delivered(app) if mail["kind"] == "email_code"]


async def test_atomic_rate_limit(client, app):
    app.state.settings.auth_email_limit = 2
    responses = await asyncio.gather(
        *(post(client, "/auth/forgot-password", {"email": "limited@example.org"}) for _ in range(4))
    )
    assert sorted(response.status_code for response in responses) == [202, 202, 429, 429]


async def test_concurrent_token_consumption_is_single_use(client, app):
    await register(client)
    await post(client, "/auth/forgot-password", {"email": "a@example.org"})
    token = (await delivered(app))[0]["token"]
    # Preload a stable CSRF cookie so concurrent requests do not race anonymous bootstrap.
    csrf = (await client.get("/api/v1/auth/csrf")).json()["csrf_token"]
    responses = await asyncio.gather(
        *(
            client.post(
                "/api/v1/auth/reset-password",
                json={"token": token, "password": "Reset123"},
                headers={"X-CSRF-Token": csrf},
            )
            for _ in range(2)
        )
    )
    assert sorted(response.status_code for response in responses) == [200, 400]


async def test_different_reset_links_serialize_without_deadlock(client, app):
    await register(client)
    for _ in range(2):
        await post(client, "/auth/forgot-password", {"email": "a@example.org"})
    tokens = [mail["token"] for mail in await delivered(app) if mail["kind"] == "reset"]
    csrf = (await client.get("/api/v1/auth/csrf")).json()["csrf_token"]
    responses = await asyncio.gather(
        *(
            client.post(
                "/api/v1/auth/reset-password",
                json={"token": token, "password": "A new secure password!"},
                headers={"X-CSRF-Token": csrf},
            )
            for token in tokens
        )
    )
    assert sorted(response.status_code for response in responses) == [200, 400]


async def test_mail_outbox_is_encrypted_and_delivery_is_idempotent(client, app):
    await post(client, "/auth/forgot-password", {"email": "missing@example.org"})
    # Registration codes are queued durably before a user exists.
    await post(client, "/auth/registration-code", {"email": "a@example.org"})
    async with app.state.session_factory() as db:
        mail = await db.scalar(select(MailOutbox))
        assert "example.org" not in mail.encrypted_payload
        from app.db.models import EmailCode

        token = await db.scalar(select(EmailCode))
        assert len(token.code_hash) == 64
    assert len(await delivered(app)) == 1
    assert await delivered(app) == []


async def test_operational_endpoints_and_embedding_identity(client, app):
    assert (await client.get("/healthz")).status_code == 200
    assert (await client.get("/readyz")).status_code == 200
    assert (await client.get("/api/v1/admin/metrics")).status_code == 401
    await register(client)
    await login(client)
    assert (await client.get("/api/v1/admin/metrics")).status_code == 404
    async with app.state.session_factory() as db:
        await db.execute(update(User).values(role="admin"))
        await db.commit()
    assert (await client.get("/api/v1/admin/metrics")).status_code == 200


async def test_user_settings_and_admin_console_endpoints(client, app):
    await register(client, "user1@example.org")
    await register(client, "admin@example.org")
    await login(client, "user1@example.org")
    s1 = await client.get("/api/v1/settings")
    assert s1.status_code == 200 and s1.json() == {
        "answer_language": "zh",
        "detail_level": "standard",
    }
    s2 = await post(
        client,
        "/settings",
        {"answer_language": "en", "detail_level": "detailed"},
        method="PUT",
    )
    assert s2.status_code == 200 and s2.json() == {
        "answer_language": "en",
        "detail_level": "detailed",
    }
    assert (await client.get("/api/v1/admin/overview")).status_code == 404
    assert (await client.get("/api/v1/admin/users")).status_code == 404

    await login(client, "admin@example.org")
    async with app.state.session_factory() as db:
        await db.execute(update(User).where(User.email == "admin@example.org").values(role="admin"))
        await db.commit()
    overview = await client.get("/api/v1/admin/overview")
    assert overview.status_code == 200
    data = overview.json()
    assert data["users"]["total"] == 2
    assert "global_usage" in data and "queues" in data and "evaluations" in data
    assert data["evaluations"]["retrieval"]["dataset"] == "m3-gold-v7"
    assert data["evaluations"]["retrieval"]["cases"] == 34
    chat_report = data["evaluations"]["chat"]
    assert chat_report["single_final_run_all_passed"] is False
    assert len(chat_report["scenarios"]) == 10
    assert all(case["passed"] for case in chat_report["scenarios"])

    users_res = await client.get("/api/v1/admin/users")
    assert users_res.status_code == 200
    users = users_res.json()
    assert len(users) == 2
    user1_id = next(u["id"] for u in users if u["email"] == "user1@example.org")
    admin_id = next(u["id"] for u in users if u["email"] == "admin@example.org")

    # Cannot disable self
    assert (
        await post(client, f"/admin/users/{admin_id}", {"status": "disabled"}, method="PATCH")
    ).status_code == 400

    # Can update quota and status for user1
    q_res = await post(
        client,
        f"/admin/users/{user1_id}/quota",
        {"max_papers": 25, "daily_parse_pages": 80, "daily_model_tokens": 50000},
        method="PUT",
    )
    assert q_res.status_code == 200
    assert q_res.json()["effective_limits"]["max_papers"] == 25
    assert q_res.json()["quota_overrides"]["daily_parse_pages"] == 80

    st_res = await post(client, f"/admin/users/{user1_id}", {"status": "disabled"}, method="PATCH")
    assert st_res.status_code == 200 and st_res.json()["status"] == "disabled"
