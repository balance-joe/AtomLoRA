import csv
import json
import sys
import os
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.parser import parse_config
from src.utils.logger import init_logger
from src.predict.predictor import TextAuditPredictor


def batch_predict(config_path, output_path=None):
    """批量预测并导出 CSV"""
    config = parse_config(config_path, mode="predict")
    exp_id = config["exp_id"]
    logger = init_logger(exp_id, config["task_type"])
    logger.info(f"加载配置，文件是 {config_path}")

    predictor = TextAuditPredictor(config=config)
    print("模型信息:", json.dumps(predictor.get_model_info(), ensure_ascii=False, indent=2))

    # TODO: 从配置或参数读取待预测数据
    batch_samples = []
    batch_results = predictor.predict_batch(batch_samples, batch_size=2)
    print("批量预测结果:", json.dumps(batch_results, ensure_ascii=False, indent=2))

    if batch_results:
        # 输出路径：优先用参数，否则基于 exp_id 自动生成
        csv_path = output_path or f"outputs/{exp_id}/predictions.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        fieldnames = [f for f in batch_results[0].keys() if f.strip()]
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in batch_results:
                clean_result = {k: v for k, v in result.items() if k.strip()}
                writer.writerow(clean_result)
        print(f"✅ CSV文件已导出至: {csv_path}")
    else:
        print("⚠️ 没有预测结果可导出")

    predictor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AtomLoRA 推理入口")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--text", type=str, default=None, help="单条文本预测")
    parser.add_argument("--output", type=str, default=None, help="CSV 输出路径")
    args = parser.parse_args()

    if args.text:
        # 单条预测模式
        config = parse_config(args.config, mode="predict")
        init_logger(config["exp_id"], config["task_type"])
        predictor = TextAuditPredictor(config=config)
        text_col = config["data"]["text_col"]
        result = predictor.predict({text_col: args.text})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        predictor.close()
    else:
        # 批量预测模式
        batch_predict(args.config, args.output)
