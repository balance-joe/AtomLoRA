import os
import json
import tempfile
import shutil
import unittest

import yaml

from src.data.splitter import split_data


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
RAW_JSONL = os.path.join(FIXTURES, "raw.jsonl")


class SplitterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="atomlora_test_split_")
        self.output_dir = os.path.join(self.tmp, "split_out")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- P0-1: split 后一定生成 config.generated.yaml ----

    def test_generates_config_yaml(self):
        split_data(
            input_path=RAW_JSONL,
            output_dir=self.output_dir,
            text_col="content",
            label_col="status",
            seed=42,
        )
        config_path = os.path.join(self.output_dir, "config.generated.yaml")
        self.assertTrue(os.path.exists(config_path), "config.generated.yaml should be created")

    # ---- P0-2: config.generated.yaml 能被 parse_config 正常读取 ----

    def test_generated_config_is_valid_yaml(self):
        split_data(
            input_path=RAW_JSONL,
            output_dir=self.output_dir,
            text_col="content",
            label_col="status",
            seed=42,
        )
        config_path = os.path.join(self.output_dir, "config.generated.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.assertIn("data", config)
        self.assertIn("train_path", config["data"])
        self.assertIn("dev_path", config["data"])
        self.assertIn("test_path", config["data"])
        self.assertEqual(config["data"]["text_col"], "content")
        self.assertEqual(config["data"]["label_col"], "status")

    def test_generated_config_paths_point_to_existing_files(self):
        split_data(
            input_path=RAW_JSONL,
            output_dir=self.output_dir,
            text_col="content",
            label_col="status",
            seed=42,
        )
        config_path = os.path.join(self.output_dir, "config.generated.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for key in ("train_path", "dev_path", "test_path"):
            path = config["data"][key]
            self.assertTrue(os.path.exists(path), f"{key}={path} should exist")

    # ---- P0-6: split --config 能从配置里读取 text_col / label_col / input ----

    def test_config_mode_reads_text_col_and_label_col(self):
        """When label_mapping is passed, config.generated.yaml includes it."""
        split_data(
            input_path=RAW_JSONL,
            output_dir=self.output_dir,
            text_col="content",
            label_col="status",
            label_mapping={"status": {0: "正确", 1: "错误"}},
            label_subset={"status": ["正确", "错误"]},
            seed=42,
        )
        config_path = os.path.join(self.output_dir, "config.generated.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.assertIn("label_mapping", config["data"])
        self.assertIn("label_subset", config["data"])
        self.assertEqual(config["data"]["label_mapping"]["status"][0], "正确")

    # ---- Split correctness ----

    def test_stratified_split_preserves_label_ratio(self):
        report = split_data(
            input_path=RAW_JSONL,
            output_dir=self.output_dir,
            text_col="content",
            label_col="status",
            train_ratio=0.6,
            dev_ratio=0.2,
            test_ratio=0.2,
            seed=42,
            stratify=True,
        )
        # 10 records, 5x0 + 5x1, ratio 0.6/0.2/0.2 => 6/2/2
        self.assertEqual(report["total_samples"], 10)
        self.assertEqual(report["splits"]["train"]["count"], 6)
        self.assertEqual(report["splits"]["dev"]["count"], 2)
        self.assertEqual(report["splits"]["test"]["count"], 2)

        # Each split should have balanced labels
        train_dist = report["label_distribution"]["train"]
        self.assertEqual(train_dist[0], 3)
        self.assertEqual(train_dist[1], 3)

    def test_random_split_respects_ratios(self):
        report = split_data(
            input_path=RAW_JSONL,
            output_dir=self.output_dir,
            text_col="content",
            label_col="status",
            train_ratio=0.8,
            dev_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            stratify=False,
        )
        self.assertEqual(report["total_samples"], 10)
        self.assertEqual(report["splits"]["train"]["count"], 8)
        self.assertEqual(report["splits"]["dev"]["count"], 1)
        self.assertEqual(report["splits"]["test"]["count"], 1)

    def test_split_report_has_unified_schema(self):
        report = split_data(
            input_path=RAW_JSONL,
            output_dir=self.output_dir,
            text_col="content",
            label_col="status",
            seed=42,
        )
        # Unified schema fields
        self.assertIn("total_samples", report)
        self.assertIn("splits", report)
        self.assertIn("label_distribution", report)
        self.assertIn("train", report["splits"])
        self.assertIn("dev", report["splits"])
        self.assertIn("test", report["splits"])
        self.assertIn("config_path", report)

    def test_invalid_ratios_raises(self):
        with self.assertRaises(ValueError):
            split_data(
                input_path=RAW_JSONL,
                output_dir=self.output_dir,
                text_col="content",
                label_col="status",
                train_ratio=0.5,
                dev_ratio=0.3,
                test_ratio=0.3,
            )

    def test_missing_field_raises(self):
        with self.assertRaises(ValueError):
            split_data(
                input_path=RAW_JSONL,
                output_dir=self.output_dir,
                text_col="nonexistent",
                label_col="status",
            )

    def test_output_files_exist(self):
        split_data(
            input_path=RAW_JSONL,
            output_dir=self.output_dir,
            text_col="content",
            label_col="status",
            seed=42,
        )
        for name in ("train", "dev", "test"):
            path = os.path.join(self.output_dir, f"raw_{name}.jsonl")
            self.assertTrue(os.path.exists(path), f"raw_{name}.jsonl should exist")

        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "split_report.json")))


if __name__ == "__main__":
    unittest.main()
