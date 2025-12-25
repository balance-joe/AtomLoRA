import csv
import json
import sys
import os
import argparse

from src.model.text_dataset import create_dataloader
from src.predict.predictor import TextAuditPredictor
# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.parser import parse_config
from src.utils.logger import init_logger

def main(config_path):
    # 1. Config
    config = parse_config(config_path)
    print(config)

    exp_id = config["exp_id"]
    logger = init_logger(exp_id, config["task_type"])

    logger.info(f"加载配置，文件是 {config_path}")

    # 1. 创建预测器
    predictor = TextAuditPredictor(config=config)

    # 2. 打印模型信息
    print("模型信息:", json.dumps(predictor.get_model_info(), ensure_ascii=False, indent=2))

    # 3. 单条预测
    # sample = {"content": "逸夫楼停电了谁管管"}
    # result = predictor.predict(sample)
    # print("单条预测结果:", json.dumps(result, ensure_ascii=False, indent=2))

    # 4. 批量预测
    batch_samples = []
    batch_results = predictor.predict_batch(batch_samples, batch_size=2)
    print("批量预测结果:", json.dumps(batch_results, ensure_ascii=False, indent=2))

    # 定义CSV文件保存路径
    csv_file_path = "预测结果.csv"

    # 提取字段名（从结果的第一个元素获取）
    if batch_results:
        # 获取所有字段名（自动适配字段）
        fieldnames = list(batch_results[0].keys())
        # 清理空字段（如果有的话）
        fieldnames = [f for f in fieldnames if f.strip()]

        # 写入CSV文件
        with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            # 创建CSV写入器
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 写入表头
            writer.writeheader()

            # 写入每一行数据
            for result in batch_results:
                # 清理空值和多余的空键
                clean_result = {k: v for k, v in result.items() if k.strip()}
                writer.writerow(clean_result)

        print(f"✅ CSV文件已成功导出至: {csv_file_path}")
    else:
        print("⚠️ 没有预测结果可导出！")


# 5. 释放资源
    predictor.close()

if __name__ == "__main__":
    # 可以用 argparse，这里为了简便直接调用
    # main(r"D:\python\AtomLoRA\configs\bert_text_audit_multi.yaml")
    # main(r"D:\python\AtomLoRA\configs\bert_text_audit_single.yaml")
    # main(r"D:\python\AtomLoRA\configs\ernie_text_audit_multi.yaml")
    # main(r"D:\python\AtomLoRA\configs\macbert_yq_class_0.2.yaml")
    main(r"/home/czyun/AtomLoRA/configs/macbert_yq_class_0.2.yaml")
