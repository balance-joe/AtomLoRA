import os
import json
import tempfile
import shutil
import unittest

import yaml

from src.data.io import write_jsonl
from src.data.doctor import run_doctor, format_report_markdown


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="atomlora_test_doctor_")
        self._patch_target = "src.data.doctor._resolve_path"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_config(self, train_records, dev_records, test_records=None):
        """Create a config dict pointing to temp JSONL files."""
        train_path = os.path.join(self.tmp, "train.jsonl")
        dev_path = os.path.join(self.tmp, "dev.jsonl")
        write_jsonl(train_records, train_path)
        write_jsonl(dev_records, dev_path)

        config = {
            "exp_id": "test_doctor",
            "data": {
                "train_path": train_path,
                "dev_path": dev_path,
                "text_col": "content",
                "label_col": "status",
                "label_mapping": {"status": {0: "正确", 1: "错误"}},
                "label_subset": {"status": ["正确", "错误"]},
            },
        }

        if test_records:
            test_path = os.path.join(self.tmp, "test.jsonl")
            write_jsonl(test_records, test_path)
            config["data"]["test_path"] = test_path

        return config

    def _run(self, config, **kwargs):
        """Run doctor with patched _resolve_path (use absolute paths directly)."""
        import src.data.doctor as doctor_mod
        original = doctor_mod._resolve_path
        doctor_mod._resolve_path = lambda p: p  # absolute paths pass through
        try:
            return run_doctor(config, **kwargs)
        finally:
            doctor_mod._resolve_path = original

    # ---- P0-4: dataset_report.json 字段结构 ----

    def test_report_has_required_fields(self):
        config = self._make_config(
            [{"content": "hello", "status": 0}] * 5,
            [{"content": "world", "status": 1}] * 5,
        )
        report = self._run(config)

        self.assertIn("exp_id", report)
        self.assertIn("status", report)
        self.assertIn("checks", report)
        self.assertIn("error_count", report)
        self.assertIn("warning_count", report)
        self.assertIn("info_count", report)
        self.assertIn("errors", report)
        self.assertIn("warnings", report)
        self.assertIn("infos", report)

    def test_report_checks_all_present(self):
        config = self._make_config(
            [{"content": "hello", "status": 0}] * 5,
            [{"content": "world", "status": 1}] * 5,
        )
        report = self._run(config)

        expected_checks = [
            "sample_counts", "label_distribution", "missing_classes",
            "empty_text", "empty_label", "unknown_labels",
            "duplicate_content", "text_length_stats", "low_count_classes",
            "label_ratio_drift",
        ]
        for check in expected_checks:
            self.assertIn(check, report["checks"], f"Missing check: {check}")
            self.assertIn("severity", report["checks"][check])
            self.assertIn("message", report["checks"][check])

    # ---- P0-3: ERROR 级别问题 ----

    def test_empty_text_triggers_error(self):
        config = self._make_config(
            [{"content": "", "status": 0}, {"content": "ok", "status": 1}],
            [{"content": "ok", "status": 0}],
        )
        report = self._run(config)

        self.assertEqual(report["checks"]["empty_text"]["severity"], "ERROR")
        self.assertGreater(report["error_count"], 0)
        self.assertEqual(report["status"], "FAIL")

    def test_empty_label_triggers_error(self):
        config = self._make_config(
            [{"content": "text", "status": None}, {"content": "ok", "status": 1}],
            [{"content": "ok", "status": 0}],
        )
        report = self._run(config)

        self.assertEqual(report["checks"]["empty_label"]["severity"], "ERROR")
        self.assertGreater(report["error_count"], 0)

    def test_missing_class_in_train_triggers_error(self):
        config = self._make_config(
            [{"content": "only_zero", "status": 0}] * 5,
            [{"content": "has_both", "status": 0}, {"content": "has_both", "status": 1}],
        )
        report = self._run(config)

        self.assertEqual(report["checks"]["missing_classes"]["severity"], "ERROR")
        self.assertIn("train", report["checks"]["missing_classes"]["detail"])

    def test_clean_data_passes(self):
        config = self._make_config(
            [{"content": f"text_{i}", "status": i % 2} for i in range(40)],
            [{"content": f"dev_{i}", "status": i % 2} for i in range(20)],
        )
        report = self._run(config)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(report["warning_count"], 0)

    # ---- P0-5: duplicate top-N 能正确输出 ----

    def test_duplicate_top_n_in_report(self):
        records = [
            {"content": "duplicate_text", "status": 0},
            {"content": "duplicate_text", "status": 0},
            {"content": "duplicate_text", "status": 0},
            {"content": "unique_text", "status": 1},
        ]
        config = self._make_config(records, [{"content": "dev_only", "status": 1}])
        report = self._run(config)

        dup_check = report["checks"]["duplicate_content"]
        self.assertEqual(dup_check["severity"], "WARNING")
        self.assertIn("top_duplicates", dup_check)
        self.assertIn("train", dup_check["top_duplicates"])

        top = dup_check["top_duplicates"]["train"]
        self.assertGreater(len(top), 0)
        self.assertIn("text", top[0])
        self.assertIn("count", top[0])
        self.assertEqual(top[0]["count"], 3)

    def test_no_duplicates_shows_no_top_n(self):
        config = self._make_config(
            [{"content": f"text_{i}", "status": 0} for i in range(5)],
            [{"content": f"dev_{i}", "status": 1} for i in range(5)],
        )
        report = self._run(config)

        dup_check = report["checks"]["duplicate_content"]
        self.assertEqual(dup_check["severity"], "INFO")
        self.assertNotIn("top_duplicates", dup_check)

    # ---- split_report vs dataset_report 字段对齐 ----

    def test_label_distribution_field_name(self):
        config = self._make_config(
            [{"content": "a", "status": 0}] * 3 + [{"content": "b", "status": 1}] * 2,
            [{"content": "c", "status": 0}] * 2 + [{"content": "d", "status": 1}] * 3,
        )
        report = self._run(config)

        # Uses same field name as split_report: label_distribution
        self.assertIn("distribution", report["checks"]["label_distribution"])
        dist = report["checks"]["label_distribution"]["distribution"]
        self.assertIn("train", dist)
        self.assertIn("dev", dist)

    # ---- markdown formatting ----

    def test_markdown_contains_top_duplicates(self):
        records = [
            {"content": "dup", "status": 0},
            {"content": "dup", "status": 0},
            {"content": "unique", "status": 1},
        ]
        config = self._make_config(records, [{"content": "dev", "status": 1}])
        report = self._run(config)
        md = format_report_markdown(report)

        self.assertIn("Top Duplicate Samples", md)
        self.assertIn("dup", md)

    def test_markdown_contains_severity_summary(self):
        config = self._make_config(
            [{"content": "", "status": 0}],
            [{"content": "ok", "status": 1}],
        )
        report = self._run(config)
        md = format_report_markdown(report)

        self.assertIn("ERROR", md)
        self.assertIn("WARN", md)


if __name__ == "__main__":
    unittest.main()
