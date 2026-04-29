# src/data/splitter.py
from __future__ import annotations

import os
import json
import random
import logging
from collections import Counter

from src.data.io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def split_data(
    input_path: str,
    output_dir: str,
    text_col: str,
    label_col: str,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
    label_mapping: dict = None,
    label_subset: dict = None,
) -> dict:
    """Split a raw JSONL file into train/dev/test sets.

    Returns:
        Split report dict.
    """
    # Validate ratios
    total_ratio = train_ratio + dev_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + dev_ratio + test_ratio = {total_ratio}，必须等于 1.0"
        )

    # Read data
    logger.info(f"读取数据: {input_path}")
    records = read_jsonl(input_path)
    if not records:
        raise ValueError(f"输入文件为空或全部无效: {input_path}")

    # Validate fields
    _validate_fields(records, text_col, label_col)

    # Normalize label_col to dict for config generation
    if isinstance(label_col, str):
        label_col_map = {"default": label_col}
    else:
        label_col_map = label_col

    # Split
    if stratify:
        train_set, dev_set, test_set = _stratified_split(
            records, label_col, train_ratio, dev_ratio, test_ratio, seed
        )
    else:
        train_set, dev_set, test_set = _random_split(
            records, train_ratio, dev_ratio, test_ratio, seed
        )

    # Derive output filenames
    stem = os.path.splitext(os.path.basename(input_path))[0]
    for suffix in ("_raw", "_data", "_all", "_full"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "train": os.path.join(output_dir, f"{stem}_train.jsonl"),
        "dev": os.path.join(output_dir, f"{stem}_dev.jsonl"),
        "test": os.path.join(output_dir, f"{stem}_test.jsonl"),
    }

    write_jsonl(train_set, paths["train"])
    write_jsonl(dev_set, paths["dev"])
    write_jsonl(test_set, paths["test"])

    # Build report (unified schema with doctor)
    label_dist = {}
    for name, dataset in [("train", train_set), ("dev", dev_set), ("test", test_set)]:
        label_dist[name] = dict(Counter(r[label_col] for r in dataset))

    report = {
        "input": input_path,
        "total_samples": len(records),
        "seed": seed,
        "stratify": stratify,
        "ratios": {
            "train": train_ratio,
            "dev": dev_ratio,
            "test": test_ratio,
        },
        "splits": {},
        "label_distribution": label_dist,
    }

    for name, dataset in [("train", train_set), ("dev", dev_set), ("test", test_set)]:
        report["splits"][name] = {
            "path": paths[name],
            "count": len(dataset),
        }

    # Write report
    report_path = os.path.join(output_dir, "split_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"切分报告已写入: {report_path}")

    # Generate config YAML for direct use with `atomlora train`
    config_path = _generate_config(
        output_dir, paths, text_col, label_col_map,
        label_mapping, label_subset, train_ratio, dev_ratio, test_ratio, seed,
    )

    report["config_path"] = config_path
    return report


def _validate_fields(records: list[dict], text_col: str, label_col: str) -> None:
    sample = records[0]
    missing = []
    if text_col not in sample:
        missing.append(text_col)
    if label_col not in sample:
        missing.append(label_col)
    if missing:
        raise ValueError(
            f"JSONL 记录中缺少字段: {missing}。可用字段: {list(sample.keys())}"
        )


def _generate_config(
    output_dir: str,
    paths: dict,
    text_col: str,
    label_col_map: dict,
    label_mapping: dict,
    label_subset: dict,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> str:
    """Generate a config.generated.yaml that can be used directly with `atomlora train`."""
    import yaml

    config = {
        "exp_id": "split_generated",
        "task_type": "single_cls",
        "data": {
            "train_path": os.path.abspath(paths["train"]),
            "dev_path": os.path.abspath(paths["dev"]),
            "test_path": os.path.abspath(paths["test"]),
            "text_col": text_col,
            "label_col": label_col_map if len(label_col_map) > 1 else list(label_col_map.values())[0],
        },
        "_split_meta": {
            "source": "atomlora split",
            "train_ratio": train_ratio,
            "dev_ratio": dev_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
        },
    }

    if label_mapping:
        config["data"]["label_mapping"] = label_mapping
    if label_subset:
        config["data"]["label_subset"] = label_subset

    config_path = os.path.join(output_dir, "config.generated.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"生成配置文件: {config_path}")

    return config_path


def _stratified_split(
    records: list[dict],
    label_col: str,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    """Stratified split: maintain label distribution across all splits."""
    groups: dict[str, list] = {}
    for record in records:
        label = record[label_col]
        groups.setdefault(label, []).append(record)

    logger.info(f"分层切分: {len(groups)} 个类别, 各类样本数 { {k: len(v) for k, v in groups.items()} }")

    train_set, dev_set, test_set = [], [], []
    rng = random.Random(seed)

    for label, group in groups.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        n_test = n - n_train - n_dev

        train_set.extend(group[:n_train])
        dev_set.extend(group[n_train : n_train + n_dev])
        test_set.extend(group[n_train + n_dev :])

    rng.shuffle(train_set)
    rng.shuffle(dev_set)
    rng.shuffle(test_set)

    return train_set, dev_set, test_set


def _random_split(
    records: list[dict],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train_set = shuffled[:n_train]
    dev_set = shuffled[n_train : n_train + n_dev]
    test_set = shuffled[n_train + n_dev :]

    return train_set, dev_set, test_set
