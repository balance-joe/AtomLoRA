import os
import copy
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_ROOT = os.path.join(PROJECT_ROOT, "configs")
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs")
RUNTIME_MODES = {"train", "eval", "predict", "serve"}


def resolve_saved_config_path(exp_id: str) -> str:
    return os.path.join(OUTPUTS_ROOT, exp_id, "config.yaml")


def resolve_config_path(config_path: str) -> str:
    """Resolve a config reference from CWD/configs/templates/outputs/latest."""
    if not config_path:
        raise ValueError("config_path 不能为空")

    if config_path == "latest":
        latest_link = os.path.join(OUTPUTS_ROOT, "latest", "config.yaml")
        if os.path.exists(latest_link):
            return latest_link

        latest_txt = os.path.join(OUTPUTS_ROOT, "latest.txt")
        if os.path.exists(latest_txt):
            with open(latest_txt, "r", encoding="utf-8") as f:
                exp_dir = f.read().strip()
            if not exp_dir:
                raise FileNotFoundError("outputs/latest.txt 为空，无法解析 latest 实验")
            latest_config = os.path.join(exp_dir, "config.yaml")
            if os.path.exists(latest_config):
                return latest_config
            raise FileNotFoundError(f"latest.txt 指向的配置不存在: {latest_config}")

        raise FileNotFoundError("未找到 outputs/latest/config.yaml 或 outputs/latest.txt")

    path = os.path.normpath(config_path)
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Config file not found: {path}")

    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        return abs_path

    for base in (CONFIG_ROOT, os.path.join(CONFIG_ROOT, "templates")):
        candidate = os.path.normpath(os.path.join(base, path))
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Config file not found: {os.path.abspath(path)}")


def resolve_runtime_config_path(config_path: str, mode: str = "train") -> str:
    """Resolve the config file that should define runtime semantics."""
    resolved = resolve_config_path(config_path)
    if mode not in {"eval", "predict", "serve"}:
        return resolved

    # If the caller already points at a saved experiment config, trust it.
    if os.path.basename(resolved) == "config.yaml" and os.path.basename(os.path.dirname(resolved)) != "configs":
        return resolved

    raw_config = _load_yaml(resolved)
    exp_id = raw_config.get("exp_id")
    if not exp_id:
        raise ValueError(f"[CONFIG] {resolved} 缺少 exp_id，无法定位实验产物")

    saved_config = resolve_saved_config_path(exp_id)
    if os.path.exists(saved_config):
        return saved_config

    return resolved


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"[CONFIG] 配置文件必须解析为 dict: {path}")
    return config


class ConfigParser:
    def __init__(
        self,
        config_path: str,
        mode: str = "train",
        runtime_overrides: Optional[Dict[str, Any]] = None,
        eval_data_path: Optional[str] = None,
    ):
        if mode not in RUNTIME_MODES:
            raise ValueError(f"不支持的配置解析模式: {mode}")
        self.mode = mode
        self.requested_config_path = resolve_config_path(config_path)
        self.config_path = resolve_runtime_config_path(config_path, mode=mode)
        self.raw_config: Dict[str, Any] = {}
        self.final_config: Dict[str, Any] = {}
        self.loaded_configs: Dict[str, Dict[str, Any]] = {}
        self.runtime_overrides = runtime_overrides or {}
        self.eval_data_path = eval_data_path

    def parse(self) -> Dict[str, Any]:
        self.raw_config = self._load_single_config(self.config_path)
        self.final_config = self._handle_inheritance(self.raw_config)
        self.final_config = self._replace_variables(self.final_config)
        self._normalize_field_names()
        self._generate_label_mapping()
        self._apply_runtime_overrides()
        self._validate_config()
        self._adapt_task_type()
        logger.info(
            "配置集成完毕: exp_id=%s mode=%s source=%s",
            self.final_config.get("exp_id"),
            self.mode,
            self.config_path,
        )
        return self.final_config

    def _load_single_config(self, config_path: str) -> Dict[str, Any]:
        if config_path in self.loaded_configs:
            return self.loaded_configs[config_path]
        config = _load_yaml(config_path)
        self.loaded_configs[config_path] = config
        return config

    def _handle_inheritance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if not config.get("base_config"):
            return config.copy()

        parent_path = resolve_config_path(config["base_config"])
        parent_config = self._handle_inheritance(self._load_single_config(parent_path))
        merged = self._deep_merge(parent_config, config)
        merged.pop("base_config", None)
        return merged

    def _deep_merge(self, parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
        merged = parent.copy()
        for key, value in child.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _replace_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        import json as _json

        global_path = os.path.join(PROJECT_ROOT, "data", "global_label_map.json")
        global_labels = {}
        if os.path.exists(global_path):
            with open(global_path, "r", encoding="utf-8") as f:
                global_labels = _json.load(f)

        if not global_labels:
            return config

        def _walk(obj):
            if isinstance(obj, str):
                for key, val in global_labels.items():
                    obj = obj.replace(f"${{{key}}}", str(val))
                return obj
            if isinstance(obj, dict):
                return {k: _walk(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_walk(v) for v in obj]
            return obj

        return _walk(config)

    def _normalize_field_names(self):
        data = self.final_config.setdefault("data", {})
        if "raw_data_path" in data and "train_path" not in data:
            data["train_path"] = data.pop("raw_data_path")

        if isinstance(data.get("label_col"), str) and self.final_config.get("task_type") == "single_cls":
            data["label_col"] = {"default": data["label_col"]}

        train = self.final_config.setdefault("train", {})
        if "learning_rate" in train and "lora_lr" not in train:
            train["lora_lr"] = train.pop("learning_rate")

        optimizer = train.setdefault("optimizer", {})
        groups = optimizer.setdefault("groups", {})

        if "lora_lr" in train and "lora" not in groups:
            groups["lora"] = train["lora_lr"]
        if "classifier_lr" in train and "classifier" not in groups:
            groups["classifier"] = train["classifier_lr"]
        if "bert_lr" in train and "bert" not in groups:
            groups["bert"] = train["bert_lr"]

    def _apply_runtime_overrides(self):
        if self.mode in {"eval", "predict", "serve"}:
            file_overrides = self._extract_runtime_overrides_from_requested_config()
            if file_overrides:
                self.final_config = self._deep_merge(self.final_config, file_overrides)

        if self.runtime_overrides:
            self.final_config = self._deep_merge(self.final_config, self.runtime_overrides)

    def _extract_runtime_overrides_from_requested_config(self) -> Dict[str, Any]:
        if self.requested_config_path == self.config_path:
            return {}
        if self._is_saved_experiment_config(self.requested_config_path):
            return {}

        requested_config = self._load_single_config(self.requested_config_path)
        overrides: Dict[str, Any] = {}

        resources = requested_config.get("resources")
        if isinstance(resources, dict):
            overrides["resources"] = copy.deepcopy(resources)

        train = requested_config.get("train")
        if isinstance(train, dict):
            train_overrides = {}
            if "batch_size" in train:
                train_overrides["batch_size"] = train["batch_size"]
            if train_overrides:
                overrides["train"] = train_overrides

        return overrides

    def _is_saved_experiment_config(self, path: str) -> bool:
        normalized = os.path.normpath(path)
        outputs_root = os.path.normpath(OUTPUTS_ROOT)
        return os.path.basename(normalized) == "config.yaml" and normalized.startswith(outputs_root)

    def _generate_label_mapping(self):
        data = self.final_config.get("data", {})
        task_type = self.final_config.get("task_type", "single_cls")

        if "label_mapping" in data:
            return
        if "label_subset" not in data:
            return

        subset = data["label_subset"]
        if task_type == "single_cls":
            if isinstance(subset, list):
                label_col = data.get("label_col", {})
                task_name = next(iter(label_col.keys())) if isinstance(label_col, dict) and label_col else "default"
                data["label_mapping"] = {
                    task_name: {idx: label for idx, label in enumerate(subset)}
                }
            elif isinstance(subset, dict):
                data["label_mapping"] = {
                    task_name: {idx: label for idx, label in enumerate(labels)}
                    for task_name, labels in subset.items()
                }
        elif task_type == "multi_cls":
            data["label_mapping"] = {
                task_name: {idx: label for idx, label in enumerate(labels)}
                for task_name, labels in subset.items()
            }

    def _validate_config(self):
        config = self.final_config
        self._require_top_level_fields(config)
        self._validate_task_type(config)
        self._validate_data_contract(config)
        self._validate_model_contract(config)
        self._validate_train_contract(config)

        if self.mode in {"eval", "predict", "serve"}:
            self._validate_artifact_contract(config)

    def _require_top_level_fields(self, config: Dict[str, Any]):
        for field in ("exp_id", "task_type"):
            if field not in config:
                raise ValueError(f"[CONFIG] 缺少必填字段 '{field}'")

    def _validate_task_type(self, config: Dict[str, Any]):
        valid_task_types = ("single_cls", "multi_cls")
        if config["task_type"] not in valid_task_types:
            raise ValueError(
                f"[CONFIG] task_type='{config['task_type']}' 不合法，必须是 {valid_task_types}"
            )

    def _validate_data_contract(self, config: Dict[str, Any]):
        data = config.get("data")
        if not isinstance(data, dict):
            raise ValueError("[CONFIG] data 必须是字典")

        if "max_len" not in data:
            raise ValueError("[CONFIG] 缺少 data.max_len")
        if "text_col" not in data or not isinstance(data["text_col"], str):
            raise ValueError("[CONFIG] 缺少 data.text_col，且必须是字符串")

        label_col = data.get("label_col")
        label_mapping = data.get("label_mapping")
        if label_col is None:
            raise ValueError("[CONFIG] 缺少 data.label_col")
        if label_mapping is None:
            raise ValueError("[CONFIG] 缺少 data.label_mapping 或 data.label_subset")

        if config["task_type"] == "multi_cls" and not isinstance(label_col, dict):
            raise ValueError("[CONFIG] multi_cls 任务要求 data.label_col 为字典")

        if self.mode == "train":
            for path_key in ("train_path", "dev_path"):
                if path_key not in data:
                    raise ValueError(f"[CONFIG] 缺少 data.{path_key}")
                self._validate_existing_path(data[path_key], field_name=f"data.{path_key}")

        if self.mode == "eval":
            if self.eval_data_path:
                self._validate_existing_path(self.eval_data_path, field_name="eval.data_path")
            else:
                dev_path = data.get("dev_path")
                if dev_path:
                    self._validate_existing_path(dev_path, field_name="data.dev_path")

    def _validate_model_contract(self, config: Dict[str, Any]):
        model = config.get("model")
        if not isinstance(model, dict):
            raise ValueError("[CONFIG] model 必须是字典")
        if "arch" not in model:
            raise ValueError("[CONFIG] 缺少 model.arch")

        lora_conf = model.get("lora", {})
        if lora_conf and not isinstance(lora_conf, dict):
            raise ValueError("[CONFIG] model.lora 必须是字典")

    def _validate_train_contract(self, config: Dict[str, Any]):
        train = config.get("train")
        if not isinstance(train, dict):
            raise ValueError("[CONFIG] train 必须是字典")

        if "batch_size" not in train:
            raise ValueError("[CONFIG] 缺少 train.batch_size")

        scheduler_type = train.get("scheduler_type", "linear")
        if scheduler_type not in {"linear", "cosine"}:
            raise ValueError("[CONFIG] train.scheduler_type 仅支持 linear / cosine")

        early_stopping = train.get("early_stopping")
        if early_stopping is not None:
            if not isinstance(early_stopping, dict):
                raise ValueError("[CONFIG] train.early_stopping 必须是字典")
            patience = early_stopping.get("patience")
            if patience is not None and (not isinstance(patience, int) or patience < 1):
                raise ValueError("[CONFIG] train.early_stopping.patience 必须是正整数")

        if self.mode != "train":
            return

        if "num_epochs" not in train:
            raise ValueError("[CONFIG] 缺少 train.num_epochs")

        optimizer_groups = train.get("optimizer", {}).get("groups", {})
        required_groups = ("bert", "lora", "classifier")
        missing = [name for name in required_groups if name not in optimizer_groups]
        if missing:
            raise ValueError(
                f"[CONFIG] 缺少 train.optimizer.groups.{', '.join(missing)}"
            )

    def _validate_artifact_contract(self, config: Dict[str, Any]):
        exp_id = config["exp_id"]
        exp_dir = os.path.join(OUTPUTS_ROOT, exp_id)
        if not os.path.isdir(exp_dir):
            raise FileNotFoundError(
                f"[CONFIG] 实验产物目录不存在: {exp_dir}\n"
                "  请先完成训练，或传入正确的实验配置"
            )

    def _validate_existing_path(self, path_value: str, field_name: str):
        resolved = path_value if os.path.isabs(path_value) else os.path.normpath(os.path.join(PROJECT_ROOT, path_value))
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"[CONFIG] {field_name} 指向的文件不存在: {path_value}\n"
                f"  解析为: {os.path.abspath(resolved)}"
            )

    def _adapt_task_type(self):
        task_type = self.final_config["task_type"]
        data_config = self.final_config["data"]

        if task_type == "single_cls":
            label_col = data_config.get("label_col", {})
            label_key = next(iter(label_col.keys())) if isinstance(label_col, dict) else "default"
            data_config["label_key"] = label_key
            data_config["label_map"] = data_config["label_mapping"]
        else:
            task_names = list(data_config["label_col"].keys())
            data_config["task_names"] = task_names
            data_config["label_maps"] = data_config["label_mapping"]
            data_config["label_cols"] = data_config["label_col"]


def parse_config(
    config_path: str,
    mode: str = "train",
    runtime_overrides: Optional[Dict[str, Any]] = None,
    eval_data_path: Optional[str] = None,
) -> Dict[str, Any]:
    return ConfigParser(
        config_path,
        mode=mode,
        runtime_overrides=runtime_overrides,
        eval_data_path=eval_data_path,
    ).parse()
