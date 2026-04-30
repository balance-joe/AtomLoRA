import sys
import os

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 确保项目根目录在 sys.path 中，以便导入 evaluator、src 等模块
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluator import Evaluator
from src.model.text_dataset import create_dataloader

from src.config.parser import parse_config
from src.utils.logger import init_logger
from src.model.model_factory import load_tokenizer, TaskTextClassifier
from src.data.data_processor import load_dataset
from src.trainer.train_engine import Trainer


def main(config_path):
    # 1. Config
    config = parse_config(config_path, mode="train")
    exp_id = config["exp_id"]
    logger = init_logger(exp_id, config["task_type"])

    logger.info(f"加载配置，文件是 {config_path}")

    # 2. Tokenizer (包含特殊token添加)
    tokenizer = load_tokenizer(config)

    # 3. Data
    logger.info("加载数据集...")
    train_data = load_dataset(config, config["data"]["train_path"], tokenizer)
    dev_data = load_dataset(config, config["data"]["dev_path"], tokenizer)

    batch_size = config["train"]["batch_size"]
    train_loader = create_dataloader(train_data, batch_size, shuffle=True)
    dev_loader = create_dataloader(dev_data, batch_size, shuffle=False)

    # 4. Model
    logger.info("构建模型...")
    model = TaskTextClassifier(config, tokenizer)

    # 5. Trainer
    logger.info("初始化训练类...")
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        tokenizer=tokenizer
    )

    # 6. Run
    trainer.train()


def evaluate(config_path, data_path=None):
    config = parse_config(config_path, mode="eval", eval_data_path=data_path)
    exp_id = config["exp_id"]
    logger = init_logger(exp_id, config["task_type"])

    logger.info(f"加载配置，文件是 {config_path}")
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate(data_path or config["data"].get("dev_path"))
    logger.info(metrics)
