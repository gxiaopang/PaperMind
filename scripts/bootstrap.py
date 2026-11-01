"""Generate local-only service credentials without touching the user's .env."""

import base64
import os
import secrets
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / ".env.local"
if target.exists():
    print(".env.local 已存在，保留现有配置")
else:
    password = secrets.token_urlsafe(32)
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    content = (
        "APP_ENV=development\nPUBLIC_BASE_URL=http://localhost:8080\n"
        f"POSTGRES_PASSWORD={password}\n"
        f"DATABASE_URL=postgresql+asyncpg://papermind:{password}@localhost:15432/papermind\n"
        "REDIS_URL=redis://localhost:16379/0\n"
        f"MAIL_ENCRYPTION_KEY={key}\n"
    )
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as file:
        file.write(content)
    print("已生成 .env.local（权限 600），原始 .env 未修改")
