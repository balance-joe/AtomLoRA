"""JSONL 文件读写工具"""
from __future__ import annotations

import json

from src.utils.logger import get_logger

logger = get_logger()


def read_jsonl(path: str) -> list[dict]:
    """读取 JSONL 文件，跳过空行和解析失败的行"""
    records = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
                logger.warning(f"[io] {path} 第{line_idx}行 JSON 解析失败，跳过")
    if skipped:
        logger.warning(f"[io] {path} 共跳过 {skipped} 行无效数据")
    return records


def normalize_label_col(label_col, task_type: str, label_mapping: dict = None):
    """将 label_col 统一转为 {task_name: col_name} 字典，并返回任务名列表。

    single_cls 下 task_names 固定为 ``["default"]``；multi_cls 下 task_names
    为 label_col 字典的键列表。

    Args:
        label_col: 配置中的 data.label_col，可以是字符串或字典
        task_type: "single_cls" 或 "multi_cls"
        label_mapping: 可选，用于从 label_mapping 推导 single_cls 的任务名

    Returns:
        (label_col_map, task_names) 二元组
    """
    if task_type == "single_cls":
        if isinstance(label_col, str):
            task_name = next(iter(label_mapping.keys())) if label_mapping else "default"
            return {task_name: label_col}, ["default"]
        return label_col, ["default"]
    else:
        if not isinstance(label_col, dict):
            raise ValueError("multi_cls 任务要求 data.label_col 为字典 (task_name -> col_name)")
        return label_col, list(label_col.keys())


def write_jsonl(records: list[dict], path: str) -> None:
    """将记录列表写入 JSONL 文件"""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
