import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn




class TinyBackbone(nn.Module):
    """用于测试的最小 backbone：仅 embedding 层，返回 last_hidden_state。"""

    def __init__(self, hidden_size=32, vocab_size=100):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            output_hidden_states=True,
        )

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embeddings(input_ids)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeTokenizer:
    """用于测试的最小 tokenizer。"""

    def __init__(self, vocab_size=100, max_len=16):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def __len__(self):
        return self.vocab_size

    def encode_plus(self, text, padding=None, truncation=None,
                    max_length=None, return_tensors=None, return_attention_mask=True):
        seq_len = max_length or self.max_len
        ids = [1] * min(len(text), seq_len) + [0] * max(0, seq_len - len(text))
        mask = [1] * min(len(text), seq_len) + [0] * max(0, seq_len - len(text))
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }


class PredictorTests(unittest.TestCase):
    """测试 TextAuditPredictor.predict() 的核心契约。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="atomlora_test_predictor_")
        self.exp_id = "test_predictor"
        self.exp_dir = os.path.join(self.tmp, "outputs", self.exp_id)

        # 构建目录结构
        adapter_dir = os.path.join(self.exp_dir, "adapter")
        classifier_dir = os.path.join(self.exp_dir, "classifier")
        tokenizer_dir = os.path.join(self.exp_dir, "tokenizer")
        for d in [adapter_dir, classifier_dir, tokenizer_dir]:
            os.makedirs(d, exist_ok=True)

        # 创建 adapter 占位文件（LoRA disabled 时不加载，但目录需要存在）
        with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "w"):
            pass

        # 创建 classifier state_dict（ModuleDict 格式：key 需要 "default." 前缀）
        hidden_size = 32
        num_labels = 2
        clf = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels),
        )
        # TaskTextClassifier 用 ModuleDict({"default": Sequential(...)})，state_dict 带 "default." 前缀
        state_dict = {f"default.{k}": v for k, v in clf.state_dict().items()}
        self.classifier_path = os.path.join(classifier_dir, "classifiers.pt")
        torch.save(state_dict, self.classifier_path)

        self.config = {
            "exp_id": self.exp_id,
            "task_type": "single_cls",
            "data": {
                "max_len": 16,
                "text_col": "text",
                "label_mapping": {"default": {"0": "正确", "1": "错误"}},
            },
            "model": {
                "arch": "bert-tiny",
                "path": "bert-tiny",
                "lora": {"enabled": False},
            },
            "train": {"batch_size": 2},
        }

        # 切换工作目录到临时目录（predictor 用相对路径 outputs/）
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _create_predictor(self):
        from src.predict.predictor import TextAuditPredictor

        backbone = TinyBackbone(hidden_size=32, vocab_size=100)
        tokenizer = FakeTokenizer(vocab_size=100, max_len=16)

        # 删除 tokenizer 目录，让 predictor 走 load_tokenizer 分支
        tokenizer_dir = os.path.join(self.exp_dir, "tokenizer")
        if os.path.isdir(tokenizer_dir):
            shutil.rmtree(tokenizer_dir)

        with patch("src.predict.predictor.AutoModel") as mock_auto, \
             patch("src.predict.predictor.resolve_adapter_path") as mock_adp, \
             patch("src.predict.predictor.resolve_classifier_path") as mock_clf, \
             patch("src.predict.predictor._resolve_model_path", return_value="dummy"), \
             patch("src.predict.predictor.load_tokenizer", return_value=tokenizer):

            mock_auto.from_pretrained.return_value = backbone
            mock_adp.return_value = os.path.join(self.exp_dir, "adapter")
            mock_clf.return_value = self.classifier_path

            predictor = TextAuditPredictor(config=self.config, device="cpu")
        return predictor

    def test_predict_returns_expected_keys(self):
        predictor = self._create_predictor()
        result = predictor.predict({"text": "hello world"})
        for key in ("text", "prediction", "label_id", "confidence", "probabilities"):
            self.assertIn(key, result, f"Missing key: {key}")
        self.assertIn(result["label_id"], [0, 1])
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 1)
        self.assertEqual(len(result["probabilities"]), 2)

    def test_predict_missing_text_col_raises(self):
        predictor = self._create_predictor()
        with self.assertRaises(ValueError) as ctx:
            predictor.predict({"wrong_col": "hello"})
        self.assertIn("text", str(ctx.exception))

    def test_predict_output_types(self):
        predictor = self._create_predictor()
        result = predictor.predict({"text": "test"})
        self.assertIsInstance(result["prediction"], str)
        self.assertIsInstance(result["label_id"], int)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["probabilities"], dict)
        for prob in result["probabilities"].values():
            self.assertIsInstance(prob, float)


if __name__ == "__main__":
    unittest.main()
