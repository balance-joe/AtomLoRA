import os
import tempfile
import unittest
import shutil
from unittest.mock import patch

import yaml

from src.config import parser as parser_module


class ConfigParserTests(unittest.TestCase):
    def setUp(self):
        self.workspace_tmp = os.path.join(os.path.dirname(__file__), ".tmp")
        os.makedirs(self.workspace_tmp, exist_ok=True)
        self.root = tempfile.mkdtemp(dir=self.workspace_tmp)
        self.config_root = os.path.join(self.root, "configs")
        self.outputs_root = os.path.join(self.root, "outputs")
        self.data_root = os.path.join(self.root, "data")
        os.makedirs(self.config_root, exist_ok=True)
        os.makedirs(self.outputs_root, exist_ok=True)
        os.makedirs(self.data_root, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_yaml(self, path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)

    def _write_text(self, path, text="x"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _patch_paths(self):
        return patch.multiple(
            parser_module,
            PROJECT_ROOT=self.root,
            CONFIG_ROOT=self.config_root,
            OUTPUTS_ROOT=self.outputs_root,
        )

    def test_latest_txt_fallback_resolves_saved_config(self):
        exp_dir = os.path.join(self.outputs_root, "demo_exp")
        config_path = os.path.join(exp_dir, "config.yaml")
        self._write_yaml(config_path, {"exp_id": "demo_exp", "task_type": "single_cls"})
        self._write_text(os.path.join(self.outputs_root, "latest.txt"), exp_dir)

        with self._patch_paths():
            resolved = parser_module.resolve_config_path("latest")

        self.assertEqual(os.path.normpath(resolved), os.path.normpath(config_path))

    def test_predict_mode_prefers_saved_output_config(self):
        locator_path = os.path.join(self.config_root, "locator.yaml")
        saved_path = os.path.join(self.outputs_root, "demo_exp", "config.yaml")

        self._write_yaml(
            locator_path,
            {
                "exp_id": "demo_exp",
                "task_type": "single_cls",
                "data": {
                    "max_len": 32,
                    "text_col": "external_text",
                    "label_col": {"status": "status"},
                    "label_mapping": {"status": {0: "A", 1: "B"}},
                },
                "model": {"arch": "bert-base-chinese"},
                "train": {"batch_size": 2},
                "resources": {"gpus": "cpu"},
            },
        )
        self._write_yaml(
            saved_path,
            {
                "exp_id": "demo_exp",
                "task_type": "single_cls",
                "data": {
                    "max_len": 64,
                    "text_col": "saved_text",
                    "label_col": {"status": "status"},
                    "label_mapping": {"status": {0: "正确", 1: "错误"}},
                },
                "model": {"arch": "bert-base-chinese"},
                "train": {"batch_size": 8},
                "resources": {"gpus": "cuda:0"},
            },
        )

        with self._patch_paths():
            config = parser_module.parse_config(locator_path, mode="predict")

        self.assertEqual(config["data"]["text_col"], "saved_text")
        self.assertEqual(config["train"]["batch_size"], 2)
        self.assertEqual(config["resources"]["gpus"], "cpu")

    def test_eval_mode_accepts_explicit_data_path_without_saved_dev_path(self):
        eval_path = os.path.join(self.root, "data", "fresh_eval.jsonl")
        self._write_text(eval_path, "{}\n")

        locator_path = os.path.join(self.config_root, "locator.yaml")
        saved_path = os.path.join(self.outputs_root, "demo_exp", "config.yaml")

        self._write_yaml(
            locator_path,
            {
                "exp_id": "demo_exp",
                "task_type": "single_cls",
                "data": {
                    "max_len": 32,
                    "text_col": "text",
                    "label_col": {"status": "status"},
                    "label_mapping": {"status": {0: "A", 1: "B"}},
                },
                "model": {"arch": "bert-base-chinese"},
                "train": {"batch_size": 2},
            },
        )
        self._write_yaml(
            saved_path,
            {
                "exp_id": "demo_exp",
                "task_type": "single_cls",
                "data": {
                    "dev_path": os.path.join(self.root, "data", "missing_dev.jsonl"),
                    "max_len": 64,
                    "text_col": "saved_text",
                    "label_col": {"status": "status"},
                    "label_mapping": {"status": {0: "正确", 1: "错误"}},
                },
                "model": {"arch": "bert-base-chinese"},
                "train": {"batch_size": 8},
            },
        )

        with self._patch_paths():
            config = parser_module.parse_config(locator_path, mode="eval", eval_data_path=eval_path)

        self.assertEqual(config["data"]["text_col"], "saved_text")

    def test_train_mode_requires_optimizer_groups(self):
        train_path = os.path.join(self.root, "data", "train.jsonl")
        dev_path = os.path.join(self.root, "data", "dev.jsonl")
        self._write_text(train_path, "{}\n")
        self._write_text(dev_path, "{}\n")

        config_path = os.path.join(self.config_root, "train.yaml")
        self._write_yaml(
            config_path,
            {
                "exp_id": "train_exp",
                "task_type": "single_cls",
                "data": {
                    "train_path": train_path,
                    "dev_path": dev_path,
                    "max_len": 32,
                    "text_col": "text",
                    "label_col": {"status": "status"},
                    "label_mapping": {"status": {0: "A", 1: "B"}},
                },
                "model": {"arch": "bert-base-chinese"},
                "train": {"num_epochs": 1, "batch_size": 2},
            },
        )

        with self._patch_paths():
            with self.assertRaises(ValueError):
                parser_module.parse_config(config_path, mode="train")


if __name__ == "__main__":
    unittest.main()
