import os
import json
import tempfile
import shutil
import subprocess
import unittest

import yaml

from src.data.io import write_jsonl


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
RAW_JSONL = os.path.join(FIXTURES, "raw.jsonl")
DEMO_YAML = os.path.join(FIXTURES, "demo.yaml")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CLISplitTests(unittest.TestCase):
    """End-to-end tests for `atomlora split`."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="atomlora_test_cli_split_")
        self.output_dir = os.path.join(self.tmp, "out")
        self.venv_python = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
        if not os.path.exists(self.venv_python):
            self.venv_python = "python"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_split(self, *extra_args):
        cmd = [
            self.venv_python, "-m", "atomlora.cli", "split",
            "--input", RAW_JSONL,
            "--output", self.output_dir,
            "--text-col", "content",
            "--label-col", "status",
            "--seed", "42",
        ]
        cmd.extend(extra_args)
        result = subprocess.run(cmd, capture_output=True, cwd=PROJECT_ROOT, encoding="utf-8", errors="replace")
        return result

    def test_split_creates_all_files(self):
        result = self._run_split()
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")

        for name in ("raw_train.jsonl", "raw_dev.jsonl", "raw_test.jsonl",
                      "split_report.json", "config.generated.yaml"):
            path = os.path.join(self.output_dir, name)
            self.assertTrue(os.path.exists(path), f"{name} should exist")

    def test_split_config_yaml_is_parseable(self):
        result = self._run_split()
        self.assertEqual(result.returncode, 0)

        config_path = os.path.join(self.output_dir, "config.generated.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.assertIn("data", config)
        self.assertIn("train_path", config["data"])
        self.assertIn("text_col", config["data"])

    def test_split_report_total_samples(self):
        result = self._run_split()
        self.assertEqual(result.returncode, 0)

        report_path = os.path.join(self.output_dir, "split_report.json")
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["total_samples"], 10)
        self.assertIn("label_distribution", report)

    def test_split_config_mode(self):
        """atomlora split --config reads text_col/label_col from config."""
        # Create a config with valid paths (parse_config validates dev_path in train mode)
        config_path = os.path.join(self.tmp, "test_config.yaml")
        config = {
            "exp_id": "split_config_test",
            "task_type": "single_cls",
            "data": {
                "train_path": RAW_JSONL,
                "dev_path": RAW_JSONL,  # point to same file for validation
                "max_len": 32,
                "text_col": "content",
                "label_col": {"status": "status"},
                "label_mapping": {"status": {0: "正确", 1: "错误"}},
                "label_subset": {"status": ["正确", "错误"]},
            },
            "model": {"arch": "bert-base-chinese", "path": "bert-base-chinese"},
            "train": {
                "num_epochs": 1, "batch_size": 2,
                "optimizer": {"type": "AdamW", "groups": {"bert": 2e-6, "lora": 1.5e-4, "classifier": 5e-4}},
            },
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

        cmd = [
            self.venv_python, "-m", "atomlora.cli", "split",
            "--config", config_path,
            "--output", self.output_dir,
            "--seed", "42",
        ]
        result = subprocess.run(cmd, capture_output=True, cwd=PROJECT_ROOT, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")

        # Should create files
        config_gen = os.path.join(self.output_dir, "config.generated.yaml")
        self.assertTrue(os.path.exists(config_gen))

        # Generated config should include label_mapping from source config
        with open(config_gen, "r", encoding="utf-8") as f:
            gen_config = yaml.safe_load(f)
        self.assertIn("label_mapping", gen_config["data"])


class CLIDoctorTests(unittest.TestCase):
    """End-to-end tests for `atomlora doctor-data`."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="atomlora_test_cli_doctor_")
        self.venv_python = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
        if not os.path.exists(self.venv_python):
            self.venv_python = "python"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_doctor_config(self, train_records, dev_records):
        """Create a temp config + JSONL files for doctor testing."""
        train_path = os.path.join(self.tmp, "train.jsonl")
        dev_path = os.path.join(self.tmp, "dev.jsonl")
        write_jsonl(train_records, train_path)
        write_jsonl(dev_records, dev_path)

        config = {
            "exp_id": "cli_test",
            "task_type": "single_cls",
            "data": {
                "train_path": train_path,
                "dev_path": dev_path,
                "max_len": 32,
                "text_col": "content",
                "label_col": {"status": "status"},
                "label_mapping": {"status": {0: "正确", 1: "错误"}},
                "label_subset": {"status": ["正确", "错误"]},
            },
            "model": {"arch": "bert-base-chinese", "path": "bert-base-chinese"},
            "train": {
                "num_epochs": 1, "batch_size": 2,
                "optimizer": {"type": "AdamW", "groups": {"bert": 2e-6, "lora": 1.5e-4, "classifier": 5e-4}},
            },
        }
        config_path = os.path.join(self.tmp, "doctor_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)
        return config_path

    def test_doctor_clean_data_exits_zero(self):
        config_path = self._make_doctor_config(
            [{"content": f"text_{i}", "status": i % 2} for i in range(20)],
            [{"content": f"dev_{i}", "status": i % 2} for i in range(10)],
        )
        result = subprocess.run(
            [self.venv_python, "-m", "atomlora.cli", "doctor-data", "--config", config_path],
            capture_output=True, cwd=PROJECT_ROOT, encoding="utf-8", errors="replace",
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")

    # ---- P0-3: --strict 遇到 ERROR 必须 exit 1 ----

    def test_strict_exits_one_on_error(self):
        config_path = self._make_doctor_config(
            [{"content": "", "status": 0}],  # empty text => ERROR
            [{"content": "ok", "status": 1}],
        )
        result = subprocess.run(
            [self.venv_python, "-m", "atomlora.cli", "doctor-data",
             "--config", config_path, "--strict"],
            capture_output=True, cwd=PROJECT_ROOT, encoding="utf-8", errors="replace",
        )
        self.assertEqual(result.returncode, 1, "Should exit 1 when --strict and ERROR found")

    def test_strict_exits_zero_when_only_warnings(self):
        # low_count below threshold => WARNING, not ERROR
        config_path = self._make_doctor_config(
            [{"content": f"a_{i}", "status": 0} for i in range(5)]
            + [{"content": f"b_{i}", "status": 1} for i in range(5)],
            [{"content": "c", "status": 0}, {"content": "d", "status": 1}],
        )
        result = subprocess.run(
            [self.venv_python, "-m", "atomlora.cli", "doctor-data",
             "--config", config_path, "--strict"],
            capture_output=True, cwd=PROJECT_ROOT, encoding="utf-8", errors="replace",
        )
        self.assertEqual(result.returncode, 0, "Should exit 0 when only WARNINGs")

    def test_doctor_writes_reports(self):
        config_path = self._make_doctor_config(
            [{"content": f"text_{i}", "status": i % 2} for i in range(10)],
            [{"content": f"dev_{i}", "status": i % 2} for i in range(4)],
        )
        result = subprocess.run(
            [self.venv_python, "-m", "atomlora.cli", "doctor-data", "--config", config_path],
            capture_output=True, cwd=PROJECT_ROOT, encoding="utf-8", errors="replace",
        )
        self.assertEqual(result.returncode, 0)

        # Reports should be written to outputs/{exp_id}/
        report_json = os.path.join(PROJECT_ROOT, "outputs", "cli_test", "dataset_report.json")
        report_md = os.path.join(PROJECT_ROOT, "outputs", "cli_test", "dataset_report.md")
        self.assertTrue(os.path.exists(report_json), "dataset_report.json should exist")
        self.assertTrue(os.path.exists(report_md), "dataset_report.md should exist")

        # Verify JSON structure
        with open(report_json, "r", encoding="utf-8") as f:
            report = json.load(f)
        self.assertIn("checks", report)
        self.assertIn("status", report)


class CLIFullChainTests(unittest.TestCase):
    """Test the full chain: split -> doctor -> verify config.generated.yaml works."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="atomlora_test_chain_")
        self.split_out = os.path.join(self.tmp, "split")
        self.venv_python = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
        if not os.path.exists(self.venv_python):
            self.venv_python = "python"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_split_then_doctor_with_generated_config(self):
        # Step 1: split
        result = subprocess.run(
            [self.venv_python, "-m", "atomlora.cli", "split",
             "--input", RAW_JSONL,
             "--output", self.split_out,
             "--text-col", "content",
             "--label-col", "status",
             "--seed", "42"],
            capture_output=True, cwd=PROJECT_ROOT, encoding="utf-8", errors="replace",
        )
        self.assertEqual(result.returncode, 0, f"split failed: {result.stderr}")

        # Step 2: doctor with generated config
        gen_config = os.path.join(self.split_out, "config.generated.yaml")
        self.assertTrue(os.path.exists(gen_config))

        # The generated config has relative paths, doctor resolves them from project root
        result = subprocess.run(
            [self.venv_python, "-m", "atomlora.cli", "doctor-data",
             "--config", gen_config],
            capture_output=True, cwd=PROJECT_ROOT, encoding="utf-8", errors="replace",
        )
        self.assertEqual(result.returncode, 0, f"doctor failed: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
