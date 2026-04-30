# src/data/io.py
from __future__ import annotations

import json

from src.utils.logger import get_logger

logger = get_logger()


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file, skipping blank lines and parse errors."""
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
    """Write records to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
