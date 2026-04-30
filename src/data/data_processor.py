import json
from tqdm import tqdm
from src.utils.logger import get_logger


def load_dataset(config, path, tokenizer):
    """
    加载 JSON Lines 格式的数据集，返回模型可用的样本列表。

    每行数据经过 tokenize、标签映射后，生成包含 input_ids、attention_mask、labels 的字典。
    配置中的 label_mapping 用于将原始标签转为整数 ID。
    """
    logger = get_logger(config["exp_id"])
    contents = []
    invalid_lines = 0

    data_config = config["data"]
    text_col = data_config["text_col"]
    label_col_map = data_config["label_col"]
    label_mapping = data_config["label_mapping"]

    # single_cls 场景下 label_col 是字符串，统一转为 {任务名: 字段名} 的字典
    if isinstance(label_col_map, str):
        task_name = next(iter(label_mapping.keys()))
        label_col_map = {task_name: label_col_map}

    # 反转映射：{0: "非误报"} → {"非误报": 0}，方便从原始标签查 ID
    reversed_label_mapping = {}
    for task_name, mapping in label_mapping.items():
        reversed_label_mapping[task_name] = {v: int(k) for k, v in mapping.items()}

    logger.info(f"加载数据集：{path} | 文本字段：{text_col} | 任务标签字段：{list(label_col_map.keys())}")

    with open(path, "r", encoding="UTF-8") as f:
        for line_idx, line in enumerate(tqdm(f, desc=f"加载数据集 set")):
            lin = line.strip()
            if not lin:
                invalid_lines += 1
                continue

            try:
                data = json.loads(lin)
            except json.JSONDecodeError:
                logger.warning(f"第{line_idx+1}行JSON解析失败，跳过")
                invalid_lines += 1
                continue

            text = data.get(text_col, "")
            if not text:
                logger.warning(f"第{line_idx+1}行文本字段[{text_col}]为空，跳过")
                continue

            # tokenize：直接返回 list，避免 tensor→list 的转换开销
            encoded_input = tokenizer(
                text=text,
                padding="max_length",
                truncation=True,
                max_length=data_config["max_len"],
                return_attention_mask=True,
            )

            # 提取标签并映射为整数 ID
            labels = {}
            valid_label = True
            for task_name, field_name in label_col_map.items():
                raw_label = data.get(field_name)

                # 原始标签可能是 int 或 str，统一转 str 用于映射查找
                lookup_key = str(raw_label) if raw_label is not None else None

                if lookup_key is None:
                    logger.warning(f"第{line_idx+1}行任务[{task_name}]标签字段[{field_name}]缺失，跳过样本")
                    valid_label = False
                    break

                try:
                    # 优先精确匹配，再尝试字符串转换匹配，最后兜底整数标签
                    if raw_label in reversed_label_mapping.get(task_name, {}):
                        label_id = reversed_label_mapping[task_name][raw_label]
                    elif lookup_key in reversed_label_mapping.get(task_name, {}):
                        label_id = reversed_label_mapping[task_name][lookup_key]
                    elif isinstance(raw_label, (int, float)):
                        label_id = int(raw_label)
                    else:
                        raise KeyError(f"标签 '{raw_label}' 不在任务 '{task_name}' 的映射中")
                    labels[task_name] = int(label_id)
                except KeyError:
                    valid_label = False
                    break

            if not valid_label:
                invalid_lines += 1
                continue

            attn_mask = encoded_input["attention_mask"]
            sample = {
                "input_ids": encoded_input["input_ids"],
                "attention_mask": attn_mask,
                "labels": labels,
                "seq_len": sum(attn_mask),
                "raw_text": text,
            }
            contents.append(sample)

    logger.info(f"数据集加载完成 | 有效样本：{len(contents)} | 无效行：{invalid_lines}")
    return contents