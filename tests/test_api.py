import unittest
from unittest.mock import patch, MagicMock, AsyncMock

from starlette.testclient import TestClient


class APIEndpointTests(unittest.TestCase):
    """FastAPI 端点测试：mock ModelManager，验证请求校验和响应格式。"""

    @classmethod
    def setUpClass(cls):
        # 阻止 lifespan 中的真实模型加载
        patcher_load = patch("api.app._load_default_model", new_callable=AsyncMock)
        patcher_load.start()
        cls.addClassCleanup(patcher_load.stop)

        # mock rate_limiter.check 为 async noop，避免限流干扰测试
        mock_limiter = MagicMock()
        mock_limiter.check = AsyncMock()
        patcher_rate = patch("api.app.rate_limiter", mock_limiter)
        patcher_rate.start()
        cls.addClassCleanup(patcher_rate.stop)

        from api.app import app
        cls.client = TestClient(app, raise_server_exceptions=False)

    def setUp(self):
        self.mock_mm = MagicMock()
        self.mock_mm.get_text_col.return_value = "text"
        self.mock_mm.predict.return_value = {
            "text": "hello",
            "prediction": "正确",
            "label_id": 0,
            "confidence": 0.95,
            "probabilities": {"正确": 0.95, "错误": 0.05},
        }
        self.mock_mm.get_model_info.return_value = {
            "exp_id": "test",
            "task_type": "single_cls",
        }
        self.patcher = patch("api.app.model_manager", self.mock_mm)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    # ---- /index ----

    def test_index_returns_success(self):
        resp = self.client.get("/index")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["code"], 2000)

    # ---- /predict ----

    def test_predict_valid_request(self):
        resp = self.client.post(
            "/predict",
            json={"model_name": "test", "sample": {"text": "hello"}},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["code"], 2000)
        self.assertIn("prediction", body["data"])
        self.assertIn("confidence", body["data"])

    def test_predict_missing_model_name(self):
        resp = self.client.post("/predict", json={"sample": {"text": "hello"}})
        body = resp.json()
        self.assertEqual(body["code"], 5000)
        self.assertIn("model_name", body["msg"])

    def test_predict_missing_sample(self):
        resp = self.client.post("/predict", json={"model_name": "test"})
        body = resp.json()
        self.assertEqual(body["code"], 5000)
        self.assertIn("sample", body["msg"])

    def test_predict_invalid_sample_type(self):
        resp = self.client.post(
            "/predict",
            json={"model_name": "test", "sample": "not_a_dict"},
        )
        body = resp.json()
        self.assertEqual(body["code"], 5000)

    def test_predict_missing_text_field(self):
        self.mock_mm.get_text_col.return_value = "content"
        resp = self.client.post(
            "/predict",
            json={"model_name": "test", "sample": {"text": "hello"}},
        )
        body = resp.json()
        self.assertEqual(body["code"], 5000)
        self.assertIn("content", body["msg"])

    def test_predict_model_not_loaded(self):
        self.mock_mm.get_text_col.side_effect = FileNotFoundError("模型未加载")
        resp = self.client.post(
            "/predict",
            json={"model_name": "test", "sample": {"text": "hello"}},
        )
        body = resp.json()
        self.assertEqual(body["code"], 5000)

    # ---- /model_info ----

    def test_model_info_endpoint(self):
        resp = self.client.get("/model_info/test")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["code"], 2000)
        self.assertEqual(body["data"]["exp_id"], "test")

    # ---- /unload ----

    def test_unload_endpoint(self):
        resp = self.client.post("/unload")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["code"], 2000)
        self.mock_mm.unload_all.assert_called_once()

    # ---- /load ----

    def test_load_missing_config_path(self):
        resp = self.client.post("/load", json={})
        body = resp.json()
        self.assertEqual(body["code"], 5000)
        self.assertIn("config_path", body["msg"])


if __name__ == "__main__":
    unittest.main()
