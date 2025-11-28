# main.py
import argparse
import sys
import os

from src.model.model_factory import load_tokenizer

# 添加项目根目录到Python路径（确保模块导入）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.parser import parse_config
from src.utils.logger import init_logger, get_logger
from src.data.data_processor import load_dataset  # 模拟数据模块


def main(config_path: str):
    try:
        # ====================== 1. 解析配置 ======================
        print(f"[Start] Loading config from: {config_path}")
        config = parse_config(config_path)
        exp_id = config["exp_id"]
        task_type = config["task_type"]
        
        # # ====================== 2. 初始化实验日志 ======================
        logger = init_logger(experiment_id=exp_id, task_type=task_type)
        logger.info("="*50 + f" Experiment {exp_id} Start " + "="*50)
        logger.info(f"Task type: {task_type}")
        logger.info(f"Model arch: {config['model']['arch']}")
        logger.info(f"Train epochs: {config['train']['num_epochs']}")
        logger.debug(f"Full config: {config}")  # DEBUG级别仅记录到文件
        
      # ====================== 3. 加载Tokenizer ======================
        tokenizer = load_tokenizer(config)  # 从配置加载Tokenizer
        # # ====================== 3. 数据加载 ======================
        logger.info("加载训练，测试，评估数据集...")
        train_data = load_dataset(
            config, 
            path=config['data']['test_path'],
            tokenizer=tokenizer
        )
        print(train_data)
        # dev_data = load_dataset(config, split="dev")
        # test_data = load_dataset(config, split="test")
        # logger.info(f"Data loaded: train({len(train_data)}), dev({len(dev_data)}), test({len(test_data)})")
        # # ====================== 4. 模型构建 ======================
        
        # logger.info("="*50 + f" Experiment {exp_id} Finish " + "="*50)
        
    except Exception as e:
        # 异常日志记录（含堆栈）
        if 'logger' in locals():
            logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        else:
            print(f"Error before logger init: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":

    # 启动实验
    main(r"D:\python\AtomLoRA\configs\text_audit.yaml")