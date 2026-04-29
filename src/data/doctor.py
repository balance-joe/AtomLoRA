# src/data/doctor.py
from __future__ import annotations

import os
import json
import logging
from collections import Counter

from src.data.io import read_jsonl

logger = logging.getLogger(__name__)

# Severity levels
ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"

# Top N problem samples to include in report
TOP_N = 5


def run_doctor(config: dict, low_count_threshold: int = 10) -> dict:
    """Run data quality diagnostics on train/dev/test splits.

    Returns:
        Full diagnostic report dict with severity levels (ERROR/WARNING/INFO).
    """
    data_cfg = config["data"]
    text_col = data_cfg["text_col"]
    label_col = data_cfg["label_col"]
    label_mapping = data_cfg.get("label_mapping", {})
    label_subset = data_cfg.get("label_subset", {})

    # Normalize label_col for single_cls
    if isinstance(label_col, str):
        task_name = next(iter(label_mapping.keys())) if label_mapping else "default"
        label_col_map = {task_name: label_col}
    else:
        label_col_map = label_col

    # Build label sets
    data_label_keys: dict[str, set] = {}
    known_labels: dict[str, set] = {}
    for task_name, mapping in label_mapping.items():
        keys = set()
        all_labels = set()
        for k, v in mapping.items():
            keys.add(str(k))
            all_labels.add(str(k))
            all_labels.add(str(v))
        data_label_keys[task_name] = keys
        known_labels[task_name] = all_labels
    for task_name, labels in label_subset.items():
        known_labels.setdefault(task_name, set()).update(str(l) for l in labels)

    # Load splits
    splits = {}
    for name in ("train", "dev", "test"):
        path = data_cfg.get(f"{name}_path")
        if path and os.path.exists(_resolve_path(path)):
            splits[name] = read_jsonl(_resolve_path(path))
        elif path:
            logger.warning(f"data.{name}_path 指向的文件不存在: {path}")

    if not splits:
        raise ValueError("没有可用的数据集，请检查配置中的 train_path / dev_path / test_path")

    # Run checks
    report = {
        "exp_id": config.get("exp_id", "unknown"),
        "checks": {},
    }

    report["checks"]["sample_counts"] = _check_sample_counts(splits)
    report["checks"]["label_distribution"] = _check_label_distribution(splits, label_col_map)
    report["checks"]["missing_classes"] = _check_missing_classes(splits, label_col_map, data_label_keys)
    report["checks"]["empty_text"] = _check_empty_text(splits, text_col)
    report["checks"]["empty_label"] = _check_empty_label(splits, label_col_map)
    report["checks"]["unknown_labels"] = _check_unknown_labels(splits, label_col_map, known_labels)
    report["checks"]["duplicate_content"] = _check_duplicate_content(splits, text_col)
    report["checks"]["text_length_stats"] = _check_text_length(splits, text_col)
    report["checks"]["low_count_classes"] = _check_low_count(splits, label_col_map, low_count_threshold)
    report["checks"]["label_ratio_drift"] = _check_label_drift(splits, label_col_map)

    # Aggregate by severity
    errors = []
    warnings = []
    infos = []
    for check_name, check_result in report["checks"].items():
        level = check_result.get("severity", INFO)
        msg = f"[{check_name}] {check_result.get('message', '')}"
        if level == ERROR:
            errors.append(msg)
        elif level == WARNING:
            warnings.append(msg)
        else:
            infos.append(msg)

    report["error_count"] = len(errors)
    report["warning_count"] = len(warnings)
    report["info_count"] = len(infos)
    report["errors"] = errors
    report["warnings"] = warnings
    report["infos"] = infos

    if errors:
        report["status"] = "FAIL"
    elif warnings:
        report["status"] = "WARN"
    else:
        report["status"] = "PASS"

    return report


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.normpath(os.path.join(project_root, path))


# ---- Individual checks ----


def _check_sample_counts(splits: dict) -> dict:
    counts = {name: len(records) for name, records in splits.items()}
    total = sum(counts.values())
    return {"severity": INFO, "status": "ok", "message": f"共 {total} 条样本", "total": total, "per_split": counts}


def _check_label_distribution(splits: dict, label_col_map: dict) -> dict:
    dist = {}
    for name, records in splits.items():
        task_dists = {}
        for task_name, field in label_col_map.items():
            task_dists[task_name] = dict(Counter(str(r.get(field, "")) for r in records))
        dist[name] = task_dists
    return {"severity": INFO, "status": "ok", "message": "标签分布", "distribution": dist}


def _check_missing_classes(splits: dict, label_col_map: dict, expected_keys: dict) -> dict:
    missing = {}
    for name, records in splits.items():
        present = {}
        for task_name, field in label_col_map.items():
            present[task_name] = set(str(r.get(field, "")) for r in records)
        for task_name, expected in expected_keys.items():
            absent = expected - present.get(task_name, set())
            if absent:
                missing.setdefault(name, {})[task_name] = sorted(absent)

    if missing:
        # Missing classes in train is ERROR, in dev/test is WARNING
        has_train_missing = "train" in missing
        level = ERROR if has_train_missing else WARNING
        return {"severity": level, "status": "fail", "message": f"部分 split 缺少类别", "detail": missing}
    return {"severity": INFO, "status": "ok", "message": "所有 split 均包含全部类别"}


def _check_empty_text(splits: dict, text_col: str) -> dict:
    empty_counts = {}
    for name, records in splits.items():
        count = sum(1 for r in records if not r.get(text_col, "").strip())
        if count:
            empty_counts[name] = count

    if empty_counts:
        total = sum(empty_counts.values())
        return {"severity": ERROR, "status": "fail", "message": f"共 {total} 条空文本", "per_split": empty_counts}
    return {"severity": INFO, "status": "ok", "message": "无空文本"}


def _check_empty_label(splits: dict, label_col_map: dict) -> dict:
    empty_counts = {}
    for name, records in splits.items():
        for task_name, field in label_col_map.items():
            count = sum(1 for r in records if r.get(field) is None or str(r.get(field, "")).strip() == "")
            if count:
                empty_counts.setdefault(name, {})[task_name] = count

    if empty_counts:
        return {"severity": ERROR, "status": "fail", "message": "存在空标签", "per_split": empty_counts}
    return {"severity": INFO, "status": "ok", "message": "无空标签"}


def _check_unknown_labels(splits: dict, label_col_map: dict, known_labels: dict) -> dict:
    unknown = {}
    for name, records in splits.items():
        for task_name, field in label_col_map.items():
            expected = known_labels.get(task_name, set())
            if not expected:
                continue
            found = set()
            for r in records:
                val = str(r.get(field, ""))
                if val and val not in expected:
                    found.add(val)
            if found:
                unknown.setdefault(name, {})[task_name] = sorted(found)

    if unknown:
        return {"severity": WARNING, "status": "fail", "message": "存在未知标签值", "detail": unknown}
    return {"severity": INFO, "status": "ok", "message": "所有标签值均在映射范围内"}


def _check_duplicate_content(splits: dict, text_col: str) -> dict:
    all_texts = []
    per_split_dupes = {}
    top_samples = {}

    for name, records in splits.items():
        texts = [r.get(text_col, "") for r in records]
        counter = Counter(texts)
        dupes = {t: c for t, c in counter.items() if c > 1 and t}
        if dupes:
            per_split_dupes[name] = {
                "duplicate_count": len(dupes),
                "total_duplicates": sum(dupes.values()) - len(dupes),
            }
            # Top N duplicated texts (truncated for readability)
            sorted_dupes = sorted(dupes.items(), key=lambda x: x[1], reverse=True)
            top_samples[name] = [
                {"text": t[:200] + ("..." if len(t) > 200 else ""), "count": c}
                for t, c in sorted_dupes[:TOP_N]
            ]
        all_texts.extend(texts)

    # Cross-split duplicates
    cross_counter = Counter(all_texts)
    cross_dupes = {t: c for t, c in cross_counter.items() if c > 1 and t}

    within_count = sum(v.get("duplicate_count", 0) for v in per_split_dupes.values())
    has_cross = len(cross_dupes) > within_count

    has_issues = bool(per_split_dupes) or has_cross

    result = {
        "severity": WARNING if has_issues else INFO,
        "status": "fail" if has_issues else "ok",
        "within_split": per_split_dupes if per_split_dupes else "无重复",
        "cross_split_duplicate_texts": len(cross_dupes),
    }
    if top_samples:
        result["top_duplicates"] = top_samples
    if has_issues:
        result["message"] = f"存在重复文本 (within: {within_count}, cross: {len(cross_dupes)})"
    else:
        result["message"] = "无跨 split 重复"
    return result


def _check_text_length(splits: dict, text_col: str) -> dict:
    stats = {}
    for name, records in splits.items():
        lengths = [len(r.get(text_col, "")) for r in records if r.get(text_col, "").strip()]
        if not lengths:
            stats[name] = {"count": 0}
            continue
        lengths.sort()
        n = len(lengths)
        stats[name] = {
            "count": n,
            "min": lengths[0],
            "max": lengths[-1],
            "avg": round(sum(lengths) / n, 1),
            "p50": lengths[n // 2],
            "p95": lengths[int(n * 0.95)],
        }
    return {"severity": INFO, "status": "ok", "message": "文本长度统计", "per_split": stats}


def _check_low_count(splits: dict, label_col_map: dict, threshold: int) -> dict:
    low = {}
    for name, records in splits.items():
        for task_name, field in label_col_map.items():
            counter = Counter(str(r.get(field, "")) for r in records)
            below = {label: count for label, count in counter.items() if count < threshold}
            if below:
                low.setdefault(name, {})[task_name] = below

    if low:
        return {"severity": WARNING, "status": "fail", "message": f"部分类别样本数低于阈值 {threshold}", "detail": low}
    return {"severity": INFO, "status": "ok", "message": f"所有类别样本数均 >= {threshold}"}


def _check_label_drift(splits: dict, label_col_map: dict) -> dict:
    ratios = {}
    for name, records in splits.items():
        task_ratios = {}
        for task_name, field in label_col_map.items():
            counter = Counter(str(r.get(field, "")) for r in records)
            total = sum(counter.values())
            if total > 0:
                task_ratios[task_name] = {k: round(v / total, 4) for k, v in counter.items()}
        ratios[name] = task_ratios

    train_ratios = ratios.get("train", {})
    max_drift = 0.0

    for name, task_ratios in ratios.items():
        if name == "train":
            continue
        for task_name, r in task_ratios.items():
            train_r = train_ratios.get(task_name, {})
            all_labels = set(list(r.keys()) + list(train_r.keys()))
            for label in all_labels:
                diff = abs(r.get(label, 0) - train_r.get(label, 0))
                if diff > max_drift:
                    max_drift = diff

    level = WARNING if max_drift > 0.1 else INFO
    message = f"最大标签比例偏移: {round(max_drift, 4)}" + (" (超过 0.1 阈值)" if max_drift > 0.1 else "")
    return {"severity": level, "status": "fail" if max_drift > 0.1 else "ok", "message": message, "ratios": ratios, "max_drift": round(max_drift, 4)}


def format_report_markdown(report: dict) -> str:
    """Format the diagnostic report as a human-readable markdown string."""
    lines = [
        f"# Dataset Report: {report.get('exp_id', 'unknown')}",
        "",
        f"**Status**: {report['status']}  ",
    ]

    # Severity summary
    parts = []
    if report["error_count"]:
        parts.append(f"ERROR: {report['error_count']}")
    if report["warning_count"]:
        parts.append(f"WARNING: {report['warning_count']}")
    if report["info_count"]:
        parts.append(f"INFO: {report['info_count']}")
    lines.append(f"**Summary**: {' | '.join(parts)}")
    lines.append("")

    # Errors first
    if report["errors"]:
        lines.append("## ERRORS")
        for e in report["errors"]:
            lines.append(f"- {e}")
        lines.append("")

    if report["warnings"]:
        lines.append("## WARNINGS")
        for w in report["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    checks = report["checks"]

    # Sample counts
    sc = checks["sample_counts"]
    lines.append("## 1. Sample Counts")
    lines.append("| Split | Count |")
    lines.append("|-------|-------|")
    for name, count in sc["per_split"].items():
        lines.append(f"| {name} | {count} |")
    lines.append(f"| **Total** | **{sc['total']}** |")
    lines.append("")

    # Label distribution
    ld = checks["label_distribution"]
    lines.append("## 2. Label Distribution")
    for split_name, task_dists in ld["distribution"].items():
        lines.append(f"### {split_name}")
        for task_name, dist in task_dists.items():
            lines.append("| Label | Count |")
            lines.append("|-------|-------|")
            for label, count in sorted(dist.items()):
                lines.append(f"| {label} | {count} |")
    lines.append("")

    # Text length stats
    tls = checks["text_length_stats"]
    lines.append("## 3. Text Length Stats")
    lines.append("| Split | Count | Min | Avg | P50 | P95 | Max |")
    lines.append("|-------|-------|-----|-----|-----|-----|-----|")
    for name, s in tls["per_split"].items():
        if s.get("count", 0) == 0:
            lines.append(f"| {name} | 0 | - | - | - | - | - |")
        else:
            lines.append(f"| {name} | {s['count']} | {s['min']} | {s['avg']} | {s['p50']} | {s['p95']} | {s['max']} |")
    lines.append("")

    # Top duplicate samples
    dup_check = checks.get("duplicate_content", {})
    top_dupes = dup_check.get("top_duplicates", {})
    if top_dupes:
        lines.append("## 4. Top Duplicate Samples")
        for split_name, samples in top_dupes.items():
            lines.append(f"### {split_name}")
            lines.append("| # | Text | Count |")
            lines.append("|---|------|-------|")
            for i, s in enumerate(samples, 1):
                lines.append(f"| {i} | {s['text']} | {s['count']} |")
        lines.append("")

    # Quality checks summary
    lines.append("## 5. Quality Checks Summary")
    lines.append("| Check | Severity | Status | Message |")
    lines.append("|-------|----------|--------|---------|")
    for check_name in (
        "sample_counts", "missing_classes", "empty_text", "empty_label",
        "unknown_labels", "duplicate_content", "low_count_classes", "label_ratio_drift",
    ):
        c = checks.get(check_name, {})
        sev = c.get("severity", INFO)
        status_icon = "PASS" if c.get("status") == "ok" else "FAIL"
        lines.append(f"| {check_name} | {sev} | {status_icon} | {c.get('message', '')} |")
    lines.append("")

    return "\n".join(lines)
