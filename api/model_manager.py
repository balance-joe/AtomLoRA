import os
import gc
from typing import Dict

from src.config.parser import parse_config, resolve_saved_config_path, resolve_runtime_config_path
from src.predict.predictor import TextAuditPredictor
from src.utils.logger import get_logger, init_logger

logger = get_logger()

# 同时加载的模型数量上限，防止 GPU 显存不足
MAX_MODELS = int(os.environ.get("ATOMLORA_MAX_MODELS", "1"))


class ModelManager:
    """模型生命周期管理：加载、卸载、LRU 淘汰、预测入口"""

    def __init__(self):
        self.models: Dict[str, TextAuditPredictor] = {}

    def unload_all(self):
        """释放所有模型显存：先 close，再 del 触发引用计数回收，最后清空 CUDA 缓存"""
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
        """加载模型，超过上限时按 LRU 淘汰最久未用的模型"""
        if model_name in self.models:
            return

        # 超过上限时淘汰最久未用的模型
        while len(self.models) >= MAX_MODELS:
            oldest_name = next(iter(self.models))
            logger.warning(f"模型数量达上限 {MAX_MODELS}，淘汰最久未用模型: {oldest_name}")
            try:
                self.models[oldest_name].close()
            except Exception:
                pass
            del self.models[oldest_name]

        # 未指定配置时从保存的实验目录查找
        if config_path is None:
            config_path = resolve_saved_config_path(model_name)
        else:
            config_path = resolve_runtime_config_path(config_path, mode="serve")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"模型配置不存在: {config_path}")
        config = parse_config(config_path, mode="serve")

        exp_id = config["exp_id"]
        logger = init_logger(exp_id, config["task_type"])
        logger.info(f"加载模型: {model_name} (from {config_path})")

        predictor = TextAuditPredictor(config=config)
        self.models[model_name] = predictor

    def predict(self, model_name: str, sample: dict) -> dict:
        """对单条样本进行预测，返回预测结果"""
        if model_name not in self.models:
            raise FileNotFoundError(
                f"模型 '{model_name}' 未加载。请先通过 POST /load 或启动时指定配置加载模型。"
            )
        predictor = self.models[model_name]
        result = predictor.predict(sample)
        return result

    def get_text_col(self, model_name: str) -> str:
        """获取模型配置中的文本字段名"""
        if model_name not in self.models:
            raise FileNotFoundError(
                f"模型 '{model_name}' 未加载。请先通过 POST /load 或启动时指定配置加载模型。"
            )
        return self.models[model_name].config["data"]["text_col"]

    def get_model_info(self, model_name: str) -> dict:
        """获取模型详细信息（任务类型、标签映射等）"""
        if model_name not in self.models:
            raise FileNotFoundError(
                f"模型 '{model_name}' 未加载。请先通过 POST /load 或启动时指定配置加载模型。"
            )
        return self.models[model_name].get_model_info()
