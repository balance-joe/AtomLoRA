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
    
    reversed_label_mapping = {}
    for task_name, mapping in label_mapping.items():
        # 反转映射：将 {0: "非误报"} 变为 {"非误报": 0}
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
                
                # 特殊处理：如果原始数据是 int (例如0/1)，但 mapping 的 key 是 str，需转为 str 查找
                # 如果原始数据是 str，直接查找
                lookup_key = str(raw_label) if raw_label is not None else None

                if lookup_key is None:
                    logger.warning(f"第{line_idx+1}行任务[{task_name}]标签字段[{field_name}]缺失，跳过样本")
                    valid_label = False
                    break

                try:
                    # # 优先尝试直接匹配
                    # if raw_label in label_mapping[task_name]:
                    #     label_id = label_mapping[task_name][raw_label]
                        
                    #     print('--------------------')
                    #     print("task_name"+ task_name)
                    #     print("raw_label"+ raw_label)
                    #     print('--------------------')
                    # # 其次尝试转字符串匹配 (兼容 json 读取 0 为 int，但 mapping key 为 "0" 的情况)
                    # elif lookup_key in label_mapping[task_name]:
                    #     print(222222)
                    #     print(task_name)
                    #     print(raw_label)
                    #     print(111111)
                    #     label_id = label_mapping[task_name][lookup_key]
                    # else:
                    #     raise KeyError
                    
                    # # =============== 核心修复 ==================
                    # # 强制转换为整数！防止 YAML 配置写成 "1" 导致报错
                    # print(label_id)
                    # print(label_id)
                    # print(label_id)
                    label_id = raw_label
                    labels[task_name] = int(label_id) 
                    # ==========================================
                    
                except KeyError:
                    # 仅在DEBUG模式下打印详细错误，防止刷屏
                    # logger.debug(f"第{line_idx+1}行任务[{task_name}]标签[{raw_label}]不在映射中")
                    valid_label = False
                    break
            
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