from fastapi import HTTPException, Request

from app.core.config import Settings
from app.core.security import hash_token

# Both counters are incremented in one Redis operation; TTL cannot be lost on a crash.
SCRIPT = """
local counts = {}
for i,key in ipairs(KEYS) do
  counts[i] = redis.call('INCR', key)
  if counts[i] == 1 then redis.call('EXPIRE', key, ARGV[1]) end
end
return counts
"""


async def limit_auth(request: Request, email: str, settings: Settings) -> None:
    # Use ASGI's client address. Compose's internal Uvicorn listener trusts Nginx,
    # which overwrites forwarded headers. Native development disables proxy headers.
    ip = request.client.host if request.client else "unknown"
    counts = await request.app.state.redis.eval(
        SCRIPT,
        2,
        f"auth:ip:{hash_token(ip)}",
        f"auth:email:{hash_token(email)}",
        settings.auth_window_seconds,
    )
    if counts[0] > settings.auth_ip_limit or counts[1] > settings.auth_email_limit:
        raise HTTPException(
            429,
            "请求过于频繁，请稍后重试",
            headers={"Retry-After": str(settings.auth_window_seconds)},
        )
