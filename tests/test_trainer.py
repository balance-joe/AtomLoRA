import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset




class TinyBackbone(nn.Module):
    """最小 backbone：仅 embedding 层，forward 返回 last_hidden_state。"""

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

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)


class TinyTokenizer:
    """最小 tokenizer，支持 save_pretrained。"""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)


def _make_synthetic_dataloader(num_samples=8, seq_len=16, num_labels=2, batch_size=4):
    """创建合成 DataLoader：input_ids, attention_mask, labels。"""
    input_ids = torch.randint(0, 100, (num_samples, seq_len))
    attention_mask = torch.ones(num_samples, seq_len, dtype=torch.long)
    # labels 是 dict 格式: {"default": tensor}
    labels = torch.randint(0, num_labels, (num_samples,))

    dataset = TensorDataset(input_ids, attention_mask, labels)

    def collate_fn(batch):
        ids, mask, lbl = zip(*batch)
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(mask),
            "labels": {"default": torch.stack(lbl)},
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


class TrainerOneEpochTests(unittest.TestCase):
    """测试 Trainer.train() 跑完一个 epoch。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="atomlora_test_trainer_")
        self.exp_id = "test_trainer"
        self.output_dir = os.path.join(self.tmp, "outputs", self.exp_id)
        os.makedirs(self.output_dir, exist_ok=True)

        self.config = {
            "exp_id": self.exp_id,
            "task_type": "single_cls",
            "data": {
                "max_len": 16,
                "text_col": "text",
                "label_col": {"default": "status"},
                "label_mapping": {"default": {"0": "正确", "1": "错误"}},
            },
            "model": {
                "arch": "tiny",
                "lora": {"enabled": False},
                "dropout": 0.1,
                "freeze_bert": False,
            },
            "train": {
                "num_epochs": 1,
                "batch_size": 4,
                "optimizer": {
                    "groups": {"bert": 2e-5, "lora": 1e-4, "classifier": 5e-4},
                },
                "warmup_ratio": 0.1,
                "scheduler_type": "linear",
                "gradient_accumulation_steps": 1,
                "monitor_interval": 999,
            },
            "resources": {"gpus": "cpu"},
        }

        self._orig_cwd = os.getcwd()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build_model_and_data(self):
        from src.model.model_factory import TaskTextClassifier

        backbone = TinyBackbone(hidden_size=32, vocab_size=100)
        tokenizer = TinyTokenizer(vocab_size=100)
        model = TaskTextClassifier(self.config, tokenizer, backbone=backbone)

        train_loader = _make_synthetic_dataloader(num_samples=8, seq_len=16, batch_size=4)
        dev_loader = _make_synthetic_dataloader(num_samples=4, seq_len=16, batch_size=4)

        return model, tokenizer, train_loader, dev_loader

    def _build_trainer(self):
        from src.trainer.train_engine import Trainer

        model, tokenizer, train_loader, dev_loader = self._build_model_and_data()

        with patch("src.trainer.train_engine.resolve_device", return_value=torch.device("cpu")), \
             patch("src.trainer.train_engine.copy_config_to_output"), \
             patch("src.trainer.train_engine.save_metrics"), \
             patch("src.trainer.train_engine.update_latest_link"), \
             patch("src.trainer.train_engine.run_evaluation") as mock_eval:

            mock_eval.return_value = {"main_score": 0.5, "acc": 0.5, "f1": 0.5}
            trainer = Trainer(self.config, model, train_loader, dev_loader, tokenizer)

        return trainer, model

    def test_train_one_epoch_completes(self):
        trainer, _ = self._build_trainer()
        # 不应抛异常
        with patch("src.trainer.train_engine.save_metrics"), \
             patch("src.trainer.train_engine.update_latest_link"):
            trainer.train()

    def test_train_parameters_change(self):
        trainer, model = self._build_trainer()
        before = {k: v.clone() for k, v in model.state_dict().items()}

        with patch("src.trainer.train_engine.save_metrics"), \
             patch("src.trainer.train_engine.update_latest_link"):
            trainer.train()

        changed = False
        for k, v in model.state_dict().items():
            if not torch.equal(v, before[k]):
                changed = True
                break
        self.assertTrue(changed, "Model parameters should change after training")

    def test_save_model_called_on_improvement(self):
        trainer, model = self._build_trainer()

        with patch("src.trainer.train_engine.save_metrics"), \
             patch("src.trainer.train_engine.update_latest_link"), \
             patch.object(trainer, "save_model") as mock_save:
            trainer.train()
            # run_evaluation 返回 main_score=0.5 > -inf，所以 save_model 应被调用
            mock_save.assert_called()


if __name__ == "__main__":
    unittest.main()
