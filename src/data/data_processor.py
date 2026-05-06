import json
from tqdm import tqdm
from src.data.io import normalize_label_col, build_reversed_mapping, resolve_label_id
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
    label_mapping = data_config["label_mapping"]
    label_col_map, _ = normalize_label_col(
        data_config["label_col"], config["task_type"], label_mapping,
    )

    # 反转映射：{0: "非误报"} → {"非误报": 0}，方便从原始标签查 ID
    reversed_label_mapping = build_reversed_mapping(label_mapping)

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
                if raw_label is None:
                    logger.warning(f"第{line_idx+1}行任务[{task_name}]标签字段[{field_name}]缺失，跳过样本")
                    valid_label = False
                    break
                try:
                    labels[task_name] = resolve_label_id(reversed_label_mapping, task_name, raw_label)
                except KeyError as e:
                    logger.warning(f"第{line_idx+1}行: {e}")
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