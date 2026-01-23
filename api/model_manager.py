import os
import json
from typing import Dict

from api.settings import CONFIGS_ROOT
from src.config.parser import parse_config
from src.predict.predictor import TextAuditPredictor
from src.utils.logger import init_logger


class ModelManager:
    """
    负责：
    - 启动时加载模型
    - 多模型管理
    - 提供预测入口
    """

    def __init__(self, model_root: str):
        self.model_root = model_root
        self.models: Dict[str, TextAuditPredictor] = {}

    def load_model(self, model_name: str):
        """
        model_name:
        """
        if model_name in self.models:
            return

        config_path = os.path.join(CONFIGS_ROOT, f"{model_name}.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"模型配置不存在: {config_path}")
        config = parse_config(config_path)

        exp_id = config["exp_id"]
        logger = init_logger(exp_id, config["task_type"])
        logger.info(f"加载模型: {model_name}")

        predictor = TextAuditPredictor(config=config)
        self.models[model_name] = predictor

    def predict(self, model_name: str, sample: dict) -> dict:
        if model_name not in self.models:
            self.load_model(model_name)

        predictor = self.models[model_name]
        result = predictor.predict(sample)
        return result

    def get_model_info(self, model_name: str) -> dict:
        if model_name not in self.models:
            self.load_model(model_name)

        return self.models[model_name].get_model_info()
