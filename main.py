import sys
import os
import argparse

from evaluator import Evaluator
from src.model.text_dataset import create_dataloader
# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.parser import parse_config
from src.utils.logger import init_logger
from src.model.model_factory import load_tokenizer, TaskTextClassifier
from src.data.data_processor import load_dataset
from src.trainer.train_engine import Trainer

def main(config_path):
    # 1. Config
    config = parse_config(config_path)
    exp_id = config["exp_id"]
    logger = init_logger(exp_id, config["task_type"])
    
    logger.info(f"加载配置，文件是 {config_path}")
    
    # 2. Tokenizer (包含特殊token添加)
    tokenizer = load_tokenizer(config)

    # 3. Data
    logger.info("加载数据集...")
    # 注意：data_processor 返回的是 List[Dict]
    train_data = load_dataset(config, config["data"]["train_path"], tokenizer)
    dev_data = load_dataset(config, config["data"]["dev_path"], tokenizer)
    
    # 创建 DataLoader (collate_fn 在这里生效)
    batch_size = config["train"]["batch_size"]
    train_loader = create_dataloader(train_data, batch_size, shuffle=True)
    dev_loader = create_dataloader(dev_data, batch_size, shuffle=False)
    
    # 4. Model
    logger.info("构建模型...")
    # 传入 tokenizer 以便 resize embedding
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

def evaluate(config_path):
    # 1. Config
    config = parse_config(config_path)
    exp_id = config["exp_id"]
    logger = init_logger(exp_id, config["task_type"])

    logger.info(f"加载配置，文件是 {config_path}")
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate(config["data"]["dev_path"])
    print(metrics)

if __name__ == "__main__":
    # 可以用 argparse，这里为了简便直接调用
    # 请确保 yaml 文件路径正确
    # main(r"/home/czyun/AtomLoRA/configs/bert_yq_class_0.1.yaml")
    # main(r"/home/czyun/AtomLoRA/configs/ernie_yq_class_0.1.yaml")
    # main(r"/home/czyun/AtomLoRA/configs/macbert_yq_class_0.1.yaml")
    # main(r"/home/czyun/AtomLoRA/configs/macbert_yq_class_0.2.yaml")
    # evaluate(r"D:\python\AtomLoRA\configs\ernie_yq_class_0.1.yaml")
    # main(r"D:\python\AtomLoRA\configs\macbert_yq_class_0.2.yaml")
    main(r"/home/czyun/AtomLoRA/configs/macbert_wubao_1.0.yaml")