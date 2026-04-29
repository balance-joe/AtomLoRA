import os
import gc
import json
import logging
from typing import Dict

from api.settings import CONFIGS_ROOT
from src.config.parser import parse_config
from src.predict.predictor import TextAuditPredictor
from src.utils.logger import init_logger

logger = logging.getLogger(__name__)

# GPU 显存有限，同时加载多个模型容易 OOM
MAX_MODELS = int(os.environ.get("ATOMLORA_MAX_MODELS", "1"))


class ModelManager:
    """
    负责：
    - 启动时加载模型
    - 多模型管理（带数量限制，防止 GPU OOM）
    - 提供预测入口
    """

    def __init__(self):
        self.models: Dict[str, TextAuditPredictor] = {}

    def unload_all(self):
        """释放所有已加载模型的显存（del + gc + empty_cache 三步确保真正释放）"""
        for name, predictor in self.models.items():
            try:
                predictor.close()
            except Exception:
                pass
        # del 触发引用计数归零，gc.collect 回收循环引用
        for name in list(self.models.keys()):
            del self.models[name]
        self.models.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("所有模型已卸载，GPU 缓存已释放")

    def load_model(self, model_name: str, config_path: str = None):
        """
        加载模型。
        Args:
            model_name: 模型标识（用 exp_id）
            config_path: 配置文件路径（优先使用）。若为 None，则从 CONFIGS_ROOT/{model_name}.yaml 查找。
        """
        if model_name in self.models:
            return

        # 超过上限时淘汰最久未用的模型（LRU）
        while len(self.models) >= MAX_MODELS:
            oldest_name = next(iter(self.models))
            logger.warning(f"模型数量达上限 {MAX_MODELS}，淘汰最久未用模型: {oldest_name}")
            try:
                self.models[oldest_name].close()
            except Exception:
                pass
            del self.models[oldest_name]

        if config_path is None:
            config_path = os.path.join(CONFIGS_ROOT, f"{model_name}.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"模型配置不存在: {config_path}")
        config = parse_config(config_path)

        exp_id = config["exp_id"]
        logger = init_logger(exp_id, config["task_type"])
        logger.info(f"加载模型: {model_name} (from {config_path})")

        predictor = TextAuditPredictor(config=config)
        self.models[model_name] = predictor

    def predict(self, model_name: str, sample: dict) -> dict:
        if model_name not in self.models:
            raise FileNotFoundError(
                f"模型 '{model_name}' 未加载。请先通过 POST /load 或启动时指定配置加载模型。"
            )
        predictor = self.models[model_name]
        result = predictor.predict(sample)
        return result

    def get_model_info(self, model_name: str) -> dict:
        if model_name not in self.models:
            raise FileNotFoundError(
                f"模型 '{model_name}' 未加载。请先通过 POST /load 或启动时指定配置加载模型。"
            )
        return self.models[model_name].get_model_info()
