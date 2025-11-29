# src/config/parser.py
import os
import yaml
from typing import Dict, Any
# 假设 utils.logger 存在，这里为了独立运行暂时使用 print 或标准 logging
import logging
logger = logging.getLogger(__name__)

CONFIG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs")

# --- 1. 修正字段定义，与YAML对齐 ---
REQUIRED_FIELDS = {
    "basic": ["exp_id", "task_type"],
    # 支持 train_path 或 raw_data_path
    "data": ["max_len"], 
    "model": ["arch"], # freeze_bert 设为可选，默认为 False
    "train": ["num_epochs", "batch_size"]
}

TASK_REQUIRED_FIELDS = {
    # 移除 label_mapping 的强校验，改为校验 label_subset 或 label_mapping 存在其一即可
    "single_cls": [], 
    "multi_cls": ["train.multi_cls"]
}

class ConfigParser:
    def __init__(self, config_path: str):
        self.config_path = self._resolve_abs_path(config_path)
        self.raw_config: Dict[str, Any] = {}
        self.final_config: Dict[str, Any] = {}
        self.loaded_configs = {}

    def parse(self) -> Dict[str, Any]:
        # 1. 加载
        self.raw_config = self._load_single_config(self.config_path)
        # 2. 继承
        self.final_config = self._handle_inheritance(self.raw_config)
        # 3. 变量替换
        self.final_config = self._replace_variables(self.final_config)
        
        # --- 新增步骤：标准化字段名 (解决 train_path vs raw_data_path) ---
        self._normalize_field_names()
        
        # --- 新增步骤：自动生成映射 (解决 label_mapping 缺失问题) ---
        self._generate_label_mapping()
        
        # 4. 验证
        self._validate_config()
        # 5. 适配结构
        self._adapt_task_type()
        
        logger.info(f"配置集成完毕: {self.final_config.get('exp_id')}")
        return self.final_config

    def _resolve_abs_path(self, config_path: str) -> str:
        if os.path.isabs(config_path): return config_path
        # 优先查找 configs 根目录，其次查找 templates
        abs_path = os.path.join(CONFIG_ROOT, config_path)
        if not os.path.exists(abs_path):
            # 尝试在 templates 下查找
            abs_path_tpl = os.path.join(CONFIG_ROOT, "templates", config_path)
            if os.path.exists(abs_path_tpl):
                return abs_path_tpl
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return abs_path

    def _load_single_config(self, config_path: str) -> Dict[str, Any]:
        if config_path in self.loaded_configs: return self.loaded_configs[config_path]
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.loaded_configs[config_path] = config
        return config

    def _handle_inheritance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if "base_config" not in config or not config["base_config"]:
            return config.copy()
        parent_path = config["base_config"]
        # 递归加载父配置
        parent_config = self._handle_inheritance(self._load_single_config(self._resolve_abs_path(parent_path)))
        merged = self._deep_merge(parent_config, config)
        merged.pop("base_config", None)
        return merged

    def _deep_merge(self, parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
        merged = parent.copy()
        for key, value in child.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _replace_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        global_path = os.path.join(CONFIG_ROOT, "../data/global_label_map.json")
        global_labels = {}
        if os.path.exists(global_path):
            with open(global_path, 'r', encoding='utf-8') as f:
                global_labels = yaml.safe_load(f)
        
        return config 

    def _normalize_field_names(self):
        """标准化字段名，处理不同别名"""
        data = self.final_config.get("data", {})
        # 兼容 raw_data_path -> train_path
        if "raw_data_path" in data and "train_path" not in data:
            data["train_path"] = data.pop("raw_data_path")
        
        train = self.final_config.get("train", {})
        # 兼容 learning_rate -> lora_lr
        if "learning_rate" in train and "lora_lr" not in train:
            train["lora_lr"] = train.pop("learning_rate")

    def _generate_label_mapping(self):
        data = self.final_config.get("data", {})
        task_type = self.final_config.get("task_type", "single_cls")
        
        if "label_mapping" in data:
            return # 已手动配置，无需生成

        if "label_subset" not in data:
            # 如果也没有 subset，稍后 validate 会报错
            return

        subset = data["label_subset"]
        
        if task_type == "single_cls":
            # 单任务：List[str] -> Dict
            mapping = {label: idx for idx, label in enumerate(subset)}
            data["label_mapping"] = mapping
            
        elif task_type == "multi_cls":
            # 双任务：Dict[str, List[str]] -> Dict[str, Dict]
            mappings = {}
            for task_name, labels in subset.items():
                mappings[task_name] = {label: idx for idx, label in enumerate(labels)}
            data["label_mapping"] = mappings
        
        logger.info(f"从标签子集自动生成的标签映射 {task_type}")

    def _validate_config(self):
        # 1. 基础字段
        for field in REQUIRED_FIELDS["basic"]:
            if field not in self.final_config: raise ValueError(f"Missing {field}")
        
        # 2. 检查 data.train_path (经过 normalize 后应该有了)
        if "train_path" not in self.final_config.get("data", {}):
             raise ValueError("Missing data.train_path (or data.raw_data_path)")

        # 3. 检查 label_mapping (经过 generate 后应该有了)
        if "label_mapping" not in self.final_config.get("data", {}):
            raise ValueError("Missing data.label_subset or data.label_mapping")

        # 4. 双任务检查
        if self.final_config["task_type"] == "multi_cls":
            if "loss_weight" not in self.final_config.get("train", {}):
                raise ValueError("Double task requires train.loss_weight")

    def _adapt_task_type(self):
        # 逻辑保持您原有的，但现在 label_mapping 肯定存在了
        task_type = self.final_config["task_type"]
        data_config = self.final_config["data"]
        
        if task_type == "single_cls":
            self.final_config["data"]["label_key"] = data_config.get("label_col", "label")
            self.final_config["data"]["label_map"] = data_config["label_mapping"]
        elif task_type == "multi_cls":
            # 确保 label_col 是字典
            if not isinstance(data_config.get("label_col"), dict):
                 raise ValueError("For multi_cls, data.label_col must be a dict mapping task_name -> col_name")
            
            task_names = list(data_config["label_col"].keys())
            self.final_config["data"]["task_names"] = task_names
            # 这里直接取 mapping，因为 _generate_label_mapping 已经处理好了结构
            self.final_config["data"]["label_maps"] = data_config["label_mapping"]
            self.final_config["data"]["label_cols"] = data_config["label_col"]

def parse_config(config_path):
    return ConfigParser(config_path).parse()