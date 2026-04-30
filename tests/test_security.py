import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

from starlette.testclient import TestClient

from api.security import verify_api_key, RateLimiter


# ============================================================
# verify_api_key 单元测试
# ============================================================

class VerifyApiKeyTests(unittest.TestCase):
    """测试 API Key 认证逻辑。"""

    def test_no_env_allows_request(self):
        """未配置 ATOMLORA_API_KEYS 时放行。"""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ATOMLORA_API_KEYS", None)
            # 不应抛异常
            verify_api_key(x_api_key=None)

    def test_empty_env_allows_request(self):
        """ATOMLORA_API_KEYS 为空字符串时放行。"""
        with patch.dict(os.environ, {"ATOMLORA_API_KEYS": ""}):
            verify_api_key(x_api_key=None)

    def test_valid_key_accepted(self):
        with patch.dict(os.environ, {"ATOMLORA_API_KEYS": "key1,key2"}):
            verify_api_key(x_api_key="key1")
            verify_api_key(x_api_key="key2")

    def test_invalid_key_rejected(self):
        from fastapi import HTTPException
        with patch.dict(os.environ, {"ATOMLORA_API_KEYS": "key1,key2"}):
            with self.assertRaises(HTTPException) as ctx:
                verify_api_key(x_api_key="wrong-key")
            self.assertEqual(ctx.exception.status_code, 401)

    def test_missing_key_rejected(self):
        from fastapi import HTTPException
        with patch.dict(os.environ, {"ATOMLORA_API_KEYS": "key1"}):
            with self.assertRaises(HTTPException) as ctx:
                verify_api_key(x_api_key=None)
            self.assertEqual(ctx.exception.status_code, 401)

    def test_whitespace_trimmed(self):
        """key 前后空格应被去除。"""
        with patch.dict(os.environ, {"ATOMLORA_API_KEYS": " key1 , key2 "}):
            verify_api_key(x_api_key="key1")
            verify_api_key(x_api_key="key2")


# ============================================================
# RateLimiter 单元测试
# ============================================================

class RateLimiterTests(unittest.TestCase):
    """测试滑动窗口限流器。"""

    def _make_request(self, api_key=None, client_host="127.0.0.1"):
        req = MagicMock()
        req.headers = {}
        if api_key:
            req.headers["x-api-key"] = api_key
        req.client = MagicMock()
        req.client.host = client_host
        return req

    def test_within_limit_passes(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        req = self._make_request()
        for _ in range(5):
            asyncio.get_event_loop().run_until_complete(rl.check(req))

    def test_exceed_limit_raises_429(self):
        from fastapi import HTTPException
        rl = RateLimiter(max_requests=3, window_seconds=60)
        req = self._make_request()
        for _ in range(3):
            asyncio.get_event_loop().run_until_complete(rl.check(req))
        with self.assertRaises(HTTPException) as ctx:
            asyncio.get_event_loop().run_until_complete(rl.check(req))
        self.assertEqual(ctx.exception.status_code, 429)

    def test_different_clients_independent(self):
        rl = RateLimiter(max_requests=2, window_seconds=60)
        req_a = self._make_request(api_key="key-a")
        req_b = self._make_request(api_key="key-b")
        for _ in range(2):
            asyncio.get_event_loop().run_until_complete(rl.check(req_a))
        # key-b 不受 key-a 影响
        asyncio.get_event_loop().run_until_complete(rl.check(req_b))

    def test_window_expires_allows_new_requests(self):
        from fastapi import HTTPException
        rl = RateLimiter(max_requests=2, window_seconds=0.01)
        req = self._make_request()
        for _ in range(2):
            asyncio.get_event_loop().run_until_complete(rl.check(req))
        # 窗口过期
        import time
        time.sleep(0.02)
        # 应该可以再次请求
        asyncio.get_event_loop().run_until_complete(rl.check(req))


# ============================================================
# 集成测试：通过 FastAPI 端点验证安全机制
# ============================================================

class SecurityIntegrationTests(unittest.TestCase):
    """在真实 FastAPI 端点上验证认证和限流。"""

    @classmethod
    def setUpClass(cls):
        # 阻止 lifespan 中的真实模型加载
        patcher_load = patch("api.app._load_default_model", new_callable=AsyncMock)
        patcher_load.start()
        cls.addClassCleanup(patcher_load.stop)

        from api.app import app
        cls.app = app

    def setUp(self):
        self.mock_mm = MagicMock()
        self.mock_mm.get_text_col.return_value = "text"
        self.mock_mm.predict.return_value = {
            "text": "hi", "prediction": "正确",
            "label_id": 0, "confidence": 0.9,
            "probabilities": {"正确": 0.9, "错误": 0.1},
        }
        self.patcher_mm = patch("api.app.model_manager", self.mock_mm)
        self.patcher_mm.start()

    def tearDown(self):
        self.patcher_mm.stop()

    def _make_client(self):
        """创建带 mock model_manager 和 rate_limiter 的 TestClient。"""
        mock_rl = MagicMock()
        mock_rl.check = AsyncMock()
        return TestClient(self.app, raise_server_exceptions=False), mock_rl

    def test_predict_no_auth_config_allows(self):
        """未配置 API key 时 predict 端点正常工作。"""
        env = os.environ.copy()
        env.pop("ATOMLORA_API_KEYS", None)
        with patch.dict(os.environ, env, clear=True):
            client, mock_rl = self._make_client()
            with patch("api.app.rate_limiter", mock_rl):
                resp = client.post(
                    "/predict",
                    json={"model_name": "test", "sample": {"text": "hello"}},
                )
                self.assertEqual(resp.json()["code"], 2000)

    def test_predict_with_valid_key(self):
        with patch.dict(os.environ, {"ATOMLORA_API_KEYS": "secret-key"}):
            client, mock_rl = self._make_client()
            with patch("api.app.rate_limiter", mock_rl):
                resp = client.post(
                    "/predict",
                    json={"model_name": "test", "sample": {"text": "hello"}},
                    headers={"X-API-Key": "secret-key"},
                )
                self.assertEqual(resp.json()["code"], 2000)

    def test_predict_with_invalid_key_returns_401(self):
        with patch.dict(os.environ, {"ATOMLORA_API_KEYS": "secret-key"}):
            client, mock_rl = self._make_client()
            with patch("api.app.rate_limiter", mock_rl):
                resp = client.post(
                    "/predict",
                    json={"model_name": "test", "sample": {"text": "hello"}},
                    headers={"X-API-Key": "wrong-key"},
                )
                self.assertEqual(resp.status_code, 401)


if __name__ == "__main__":
    unittest.main()
