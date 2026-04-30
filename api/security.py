import os
import time
import asyncio
from collections import defaultdict

from fastapi import Header, HTTPException, Request


def verify_api_key(x_api_key: str = Header(None)):
    """FastAPI dependency: 校验 X-API-Key header。

    - ATOMLORA_API_KEYS 未设置或为空 → 放行（向后兼容）
    - 已设置 → 请求必须携带匹配的 key
    """
    expected_keys_env = os.environ.get("ATOMLORA_API_KEYS")
    if not expected_keys_env:
        return

    valid_keys = {k.strip() for k in expected_keys_env.split(",") if k.strip()}
    if not valid_keys:
        return

    if not x_api_key or x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class RateLimiter:
    """内存滑动窗口限流器，无外部依赖。"""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def _get_client_key(self, request: Request) -> str:
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"key:{api_key}"
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    async def check(self, request: Request) -> None:
        client_key = self._get_client_key(request)
        now = time.monotonic()

        async with self._lock:
            self._requests[client_key] = [
                t for t in self._requests[client_key]
                if now - t < self.window_seconds
            ]

            if len(self._requests[client_key]) >= self.max_requests:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {self.max_requests} requests per {self.window_seconds}s",
                )

            self._requests[client_key].append(now)


def create_rate_limiter() -> RateLimiter:
    max_req = int(os.environ.get("ATOMLORA_RATE_LIMIT", "60"))
    window = int(os.environ.get("ATOMLORA_RATE_WINDOW", "60"))
    return RateLimiter(max_requests=max_req, window_seconds=window)
