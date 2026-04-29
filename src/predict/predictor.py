import os
import torch

from typing import Dict, List, Any, Union
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModel

from src.utils.logger import get_logger
from src.utils.device import resolve_device
from src.model.model_factory import TaskTextClassifier, load_tokenizer, _resolve_model_path
from src.utils.paths import resolve_adapter_path, resolve_classifier_path, resolve_tokenizer_path

DEFAULT_CONFIG = {
    "data": {
        "max_len": 256,
        "text_col": "text",
    },
    "model": {
        "arch": "bert-base-chinese",
    },
    "train": {
        "batch_size": 20,
    },
    "task_type": "single_cls",
}


class TextAuditPredictor:
    """文本审计预测器 - 兼容单/多任务，与训练产物严格对齐。"""

    def __init__(self, config: Dict[str, Any], device: Union[str, torch.device, None] = None):
        self.config = self._merge_config(config)
        self.exp_id = self.config["exp_id"]
        self.device = resolve_device(self.config, override=device)
        self.logger = get_logger(self.exp_id)
        self.exp_dir = os.path.join("outputs", self.exp_id)

        if not os.path.isdir(self.exp_dir):
            raise FileNotFoundError(f"实验目录不存在: {self.exp_dir}")

        self._load_experiment_artifacts()
        self._initialize_model()
        self.logger.info(f"✅ 预测器初始化完成 | 设备: {self.device} | 任务类型: {self.task_type}")

    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        def recursive_merge(default: Dict, user: Dict) -> Dict:
            merged = default.copy()
            for k, v in user.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k] = recursive_merge(merged[k], v)
                else:
                    merged[k] = v
            return merged

        return recursive_merge(DEFAULT_CONFIG, user_config)

    def _validate_config(self):
        required_keys = [
            ("exp_id", str),
            ("model", dict),
            ("data", dict),
        ]
        for key, expected_type in required_keys:
            if key not in self.config or not isinstance(self.config[key], expected_type):
                raise ValueError(f"配置缺失或类型错误: {key} (期望 {expected_type.__name__})")

    def _load_experiment_artifacts(self):
        self._validate_config()
        self.task_type = self.config.get("task_type", "single_cls")
        self.label_map = self.config["data"]["label_mapping"]
        if self.task_type == "single_cls":
            first_task_key = next(iter(self.label_map.keys()))
            self.label_map = {"default": self.label_map[first_task_key]}

        self.tokenizer = self._load_tokenizer()
        self.adapter_path = resolve_adapter_path(self.exp_dir)
        self.classifier_path = resolve_classifier_path(self.exp_dir)

    def _load_tokenizer(self):
        tokenizer_path = resolve_tokenizer_path(self.exp_dir)
        if os.path.exists(tokenizer_path):
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(tokenizer_path)

        self.logger.warning("⚠️ 未找到训练好的Tokenizer，从配置加载")
        return load_tokenizer(self.config)

    def _initialize_model(self):
        backbone = self._load_backbone_with_artifacts()
        self.model = TaskTextClassifier(self.config, self.tokenizer, backbone=backbone)
        self._load_classifier_weights()
        self._validate_loaded_artifacts()
        self.model.to(self.device)
        self.model.eval()

    def _load_backbone_with_artifacts(self):
        model_arch = self.config["model"]["arch"]
        model_path = self.config["model"].get("path", model_arch)
        resolved_path = _resolve_model_path(model_path, self.logger)

        backbone = AutoModel.from_pretrained(resolved_path, output_hidden_states=True)
        if len(self.tokenizer) > backbone.config.vocab_size:
            backbone.resize_token_embeddings(len(self.tokenizer))

        lora_enabled = self.config["model"].get("lora", {}).get("enabled", True)
        if not lora_enabled:
            return backbone

        self._ensure_adapter_exists()
        try:
            return PeftModel.from_pretrained(backbone, self.adapter_path)
        except Exception as e:
            raise RuntimeError(f"LoRA 适配器加载失败: {e}") from e

    def _ensure_adapter_exists(self):
        adapter_bin_path = os.path.join(self.adapter_path, "adapter_model.bin")
        adapter_safetensors_path = os.path.join(self.adapter_path, "adapter_model.safetensors")
        if os.path.exists(adapter_bin_path) or os.path.exists(adapter_safetensors_path):
            return
        raise FileNotFoundError(
            f"LoRA 适配器不存在: {self.adapter_path}\n"
            "  期望存在 adapter_model.bin 或 adapter_model.safetensors"
        )

    def _load_classifier_weights(self):
        if not os.path.exists(self.classifier_path):
            raise FileNotFoundError(f"分类头权重文件不存在: {self.classifier_path}")

        try:
            clf_state = torch.load(self.classifier_path, map_location=self.device)
            self.model.classifiers.load_state_dict(clf_state, strict=True)
        except Exception as e:
            raise RuntimeError(f"分类头权重加载失败: {e}") from e

    def _validate_loaded_artifacts(self):
        expected_tasks = {"default"} if self.task_type == "single_cls" else set(self.label_map.keys())
        actual_tasks = set(self.model.classifiers.keys())
        if actual_tasks != expected_tasks:
            raise RuntimeError(
                f"分类头任务不匹配: expected={sorted(expected_tasks)} actual={sorted(actual_tasks)}"
            )

        for task_name, classifier in self.model.classifiers.items():
            expected_labels = len(self.label_map[task_name])
            output_dim = classifier[-1].out_features
            if output_dim != expected_labels:
                raise RuntimeError(
                    f"任务 '{task_name}' 分类头输出维度错误: expected={expected_labels} actual={output_dim}"
                )

        embedding_size = self.model.bert.get_input_embeddings().num_embeddings
        tokenizer_size = len(self.tokenizer)
        if embedding_size != tokenizer_size:
            raise RuntimeError(
                f"Tokenizer / embedding 大小不一致: tokenizer={tokenizer_size} embedding={embedding_size}"
            )

    def predict(self, data_sample: Dict[str, Any]) -> Dict[str, Any]:
        text_col = self.config["data"]["text_col"]
        if text_col not in data_sample:
            raise ValueError(f"数据缺少必要字段: {text_col}")

        encoded = self.tokenizer.encode_plus(
            data_sample[text_col],
            padding="max_length",
            truncation=True,
            max_length=self.config["data"]["max_len"],
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, labels=None)

        return self._postprocess_result(outputs, data_sample)

    def predict_batch(
        self,
        data_samples: List[Dict[str, Any]],
        batch_size: int = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        if not data_samples:
            self.logger.warning("⚠️ 空数据样本列表")
            return []

        batch_size = batch_size or self.config["train"]["batch_size"]
        all_results = []
        text_col = self.config["data"]["text_col"]

        iterator = range(0, len(data_samples), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"批量预测 (batch={batch_size})")

        self.model.eval()
        with torch.no_grad():
            for batch_start in iterator:
                batch_samples = data_samples[batch_start:batch_start + batch_size]
                batch_texts = []
                valid_indices = []
                for idx, sample in enumerate(batch_samples):
                    if text_col in sample:
                        batch_texts.append(sample[text_col])
                        valid_indices.append(idx)
                    else:
                        error_msg = f"缺少{text_col}字段"
                        all_results.append({"error": error_msg, "sample": sample})

                if not batch_texts:
                    continue

                encoded = self.tokenizer(
                    batch_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.config["data"]["max_len"],
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                outputs = self.model(input_ids, attention_mask, labels=None)

                for idx, text_idx in enumerate(valid_indices):
                    sample = batch_samples[text_idx]
                    single_output = self._extract_single_output(outputs, idx)
                    all_results.append(self._postprocess_result(single_output, sample))

        return all_results

    def _extract_single_output(self, outputs: Union[Dict[str, torch.Tensor], torch.Tensor], idx: int):
        if isinstance(outputs, dict):
            return {k: v[idx:idx + 1] for k, v in outputs.items()}
        return outputs[idx:idx + 1]

    def _postprocess_result(
        self,
        outputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        data_sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.task_type == "single_cls":
            return self._process_single_task(outputs, data_sample)
        return self._process_multi_task(outputs, data_sample)

    def _process_single_task(
        self,
        outputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        data_sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        logits = outputs.get("default", outputs) if isinstance(outputs, dict) else outputs
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).cpu().item()
        confidence = probs[0][pred_idx].cpu().item()

        label_map = self.label_map["default"]
        pred_label = label_map.get(str(pred_idx), label_map.get(pred_idx, str(pred_idx)))

        result = {
            "text": data_sample.get(self.config["data"]["text_col"], ""),
            "prediction": pred_label,
            "label_id": int(pred_idx),
            "confidence": round(float(confidence), 4),
            "probabilities": {},
        }
        for idx, prob in enumerate(probs[0].cpu().numpy()):
            label = label_map.get(str(idx), label_map.get(idx, str(idx)))
            result["probabilities"][label] = round(float(prob), 4)
            if idx >= 9:
                break

        if "raw_data" in data_sample:
            result["raw_data"] = data_sample["raw_data"]
        return result

    def _process_multi_task(self, outputs: Dict[str, torch.Tensor], data_sample: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(outputs, dict):
            raise ValueError("多任务模型必须返回字典格式输出")

        result = {
            "text": data_sample.get(self.config["data"]["text_col"], ""),
            "tasks": {},
        }
        for task_name, logits in outputs.items():
            if task_name not in self.label_map:
                continue

            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).cpu().item()
            confidence = probs[0][pred_idx].cpu().item()

            task_label_map = self.label_map[task_name]
            pred_label = task_label_map.get(str(pred_idx), task_label_map.get(pred_idx, str(pred_idx)))

            task_result = {
                "prediction": pred_label,
                "label_id": int(pred_idx),
                "confidence": round(float(confidence), 4),
                "probabilities": {},
            }
            for idx, prob in enumerate(probs[0].cpu().numpy()):
                label = task_label_map.get(str(idx), task_label_map.get(idx, str(idx)))
                task_result["probabilities"][label] = round(float(prob), 4)
                if idx >= 9:
                    break

            result["tasks"][task_name] = task_result

        return result

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "exp_id": self.exp_id,
            "task_type": self.task_type,
            "model_arch": self.config["model"]["arch"],
            "max_length": self.config["data"]["max_len"],
            "text_col": self.config["data"]["text_col"],
            "label_mapping": self.label_map,
            "device": str(self.device),
            "lora_enabled": hasattr(self.model.bert, "peft_config"),
        }

    def close(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("🔌 预测器资源已释放")
