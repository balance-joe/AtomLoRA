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


def write_jsonl(records: list[dict], path: str) -> None:
    """将记录列表写入 JSONL 文件"""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
