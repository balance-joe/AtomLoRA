import sys
import os
import argparse

# 等价于 src.utils.logger.ensure_utf8_stdio()，此处提前执行因为 src 尚未可导入
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atomlora.engine import main, evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AtomLoRA 训练入口")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--eval", action="store_true", help="仅运行评估，不训练")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.config)
    else:
        main(args.config)