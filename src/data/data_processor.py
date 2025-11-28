# src/data/data_processor.py
import json
from tqdm import tqdm
from src.utils.logger import get_logger


def load_dataset(config, path, tokenizer):
    """
    加载已预处理的JSON Lines数据集（文本/标签字段已处理完成）
    
    Args:
        config: 解析后的配置字典
        path: 数据集路径
        tokenizer: 模型Tokenizer实例
    
    Returns:
        list[dict]: 标准化样本列表，包含模型输入和标签
    """
    logger = get_logger(config["exp_id"])
    contents = []
    invalid_lines = 0

    # 从配置提取核心字段（完全配置驱动）
    data_config = config["data"]
    text_col = data_config["text_col"]  # 文本字段名（如"text"）
    label_col_map = data_config["label_col"]  # 任务-标签字段映射（如{"misreport": "mis_label", "risk": "risk_label"}）
    label_mapping = data_config["label_mapping"]  # 标签-ID映射（如{"misreport": {"非误报":0, "误报":1}}）
    
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

            # 1. 提取已预处理的文本（直接取配置指定字段）
            text = data.get(text_col, "")
            if not text:
                logger.warning(f"第{line_idx+1}行文本字段[{text_col}]为空，跳过")
                continue

            # 2. Tokenize文本（通用处理）
            encoded_input = tokenizer(
                text=text,
                padding="max_length",
                truncation=True,
                max_length=data_config["max_len"],
                return_tensors="pt",
                return_attention_mask=True,
            )

            # 3. 提取标签（配置化映射）
            labels = {}
            valid_label = True
            for task_name, field_name in label_col_map.items():
                raw_label = data.get(field_name)
                if raw_label is None:
                    logger.warning(f"第{line_idx+1}行任务[{task_name}]标签字段[{field_name}]缺失，跳过样本")
                    valid_label = False
                    break

                # 标签转ID（兼容字符串/数字标签）
                try:
                    label_id = label_mapping[task_name][raw_label]  # 统一转字符串匹配
                except KeyError:
                    logger.warning(f"第{line_idx+1}行任务[{task_name}]标签[{raw_label}]不在映射中，跳过样本")
                    valid_label = False
                    break
                labels[task_name] = label_id

            if not valid_label:
                invalid_lines += 1
                continue

            # 4. 标准化样本输出
            sample = {
                "input_ids": encoded_input["input_ids"].squeeze(0).tolist(),
                "attention_mask": encoded_input["attention_mask"].squeeze(0).tolist(),
                "labels": labels,
                "seq_len": int(encoded_input["attention_mask"].sum().item()),
                "raw_text": text  # 保留原始文本用于调试
            }
            contents.append(sample)

    # 加载统计
    logger.info(f"数据集集加载完成 | 有效样本：{len(contents)} | 无效行：{invalid_lines}")
    return contents