import json
import sys
import os
import argparse

# 等价于 src.utils.logger.ensure_utf8_stdio()，此处提前执行因为 src 尚未可导入
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.parser import parse_config
from src.utils.logger import init_logger
from src.predict.predictor import TextAuditPredictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AtomLoRA 单条推理入口")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--text", type=str, required=True, help="待预测文本")
    args = parser.parse_args()

    config = parse_config(args.config, mode="predict")
    init_logger(config["exp_id"], config["task_type"])
    predictor = TextAuditPredictor(config=config)
    text_col = config["data"]["text_col"]
    result = predictor.predict({text_col: args.text})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    predictor.close()
