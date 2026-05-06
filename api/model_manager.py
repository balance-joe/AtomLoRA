import os
import gc
from typing import Dict
import yaml

from src.config.parser import (
    CONFIG_ROOT,
    parse_config,
    resolve_saved_config_path,
    resolve_runtime_config_path,
)
from src.predict.predictor import TextAuditPredictor
from src.utils.logger import get_logger, init_logger

logger = get_logger()

# 同时加载的模型数量上限，防止 GPU 显存不足
MAX_MODELS = int(os.environ.get("ATOMLORA_MAX_MODELS", "3"))


class ModelManager:
    """模型生命周期管理：加载、卸载、LRU 淘汰、预测入口"""

    def __init__(self, max_models: int = MAX_MODELS):
        self.models: Dict[str, TextAuditPredictor] = {}
        self.max_models = max_models

    def _get_predictor(self, model_name: str) -> TextAuditPredictor:
        if model_name not in self.models:
            raise FileNotFoundError(
                f"模型 '{model_name}' 未加载。请先通过 POST /load 或启动时指定配置加载模型。"
            )
        return self.models[model_name]

    def ensure_model_loaded(self, model_name: str):
        """按需加载模型；model_name 必须是 exp_id。"""
        if model_name in self.models:
            return
        self.load_model(model_name)

    def _find_config_by_exp_id(self, exp_id: str) -> str:
        matches = []
        for root, _, files in os.walk(CONFIG_ROOT):
            for filename in files:
                if not filename.endswith((".yaml", ".yml")):
                    continue
                candidate = os.path.join(root, filename)
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        raw_config = yaml.safe_load(f)
                except Exception:
                    continue
                if isinstance(raw_config, dict) and raw_config.get("exp_id") == exp_id:
                    matches.append(candidate)

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            conflict_list = "\n".join(f"  - {path}" for path in matches)
            raise ValueError(
                f"configs 目录存在重复 exp_id='{exp_id}' 的配置文件:\n{conflict_list}"
            )
        raise FileNotFoundError(f"未在 configs 目录找到 exp_id='{exp_id}' 对应的配置文件")

    def _resolve_config_for_exp_id(self, exp_id: str) -> str:
        saved_config = resolve_saved_config_path(exp_id)
        if os.path.exists(saved_config):
            return saved_config
        return self._find_config_by_exp_id(exp_id)

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
            logger.info(f"[CACHE HIT] 模型已加载: {model_name}")
            return

        # 超过上限时淘汰最久未用的模型
        while len(self.models) >= self.max_models:
            oldest_name = next(iter(self.models))
            logger.warning(f"模型数量达上限 {self.max_models}，淘汰最久未用模型: {oldest_name}")
            try:
                self.models[oldest_name].close()
            except Exception:
                pass
            del self.models[oldest_name]

        # 未指定配置时从保存的实验目录查找
        if config_path is None:
            config_path = self._resolve_config_for_exp_id(model_name)
        else:
            config_path = resolve_runtime_config_path(config_path, mode="serve")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"模型配置不存在: {config_path}")
        config = parse_config(config_path, mode="serve")

        exp_id = config["exp_id"]
        if exp_id != model_name:
            raise ValueError(
                f"model_name 必须是 exp_id: request={model_name} config_exp_id={exp_id}"
            )
        exp_logger = init_logger(exp_id, config["task_type"])
        exp_logger.info(f"加载模型: {model_name} (from {config_path})")

        predictor = TextAuditPredictor(config=config)
        self.models[model_name] = predictor

    def predict(self, model_name: str, sample: dict) -> dict:
        """对单条样本进行预测，返回预测结果"""
        predictor = self._get_predictor(model_name)
        result = predictor.predict(sample)
        return result

    def get_text_col(self, model_name: str) -> str:
        """获取模型配置中的文本字段名"""
        return self._get_predictor(model_name).config["data"]["text_col"]

    def get_model_info(self, model_name: str) -> dict:
        """获取模型详细信息（任务类型、标签映射等）"""
        return self._get_predictor(model_name).get_model_info()
