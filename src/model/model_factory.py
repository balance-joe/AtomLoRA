import os
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType

from src.data.io import normalize_label_col
from src.utils.logger import get_logger


def _resolve_model_path(model_path: str, logger) -> str:
    abs_path = os.path.abspath(model_path)
    if os.path.isdir(abs_path):
        logger.info(f"Loading local model from: {abs_path}")
        return abs_path

    if model_path.startswith(".") or model_path.startswith("/"):
        raise FileNotFoundError(
            f"[MODEL] 本地模型路径不存在: {abs_path}\n"
            "  解决方案（任选其一）:\n"
            "  1. 下载模型到该路径\n"
            "  2. 改用 HuggingFace repo id（如 model.path: 'bert-base-chinese'）"
        )

    logger.info(f"Loading model from HuggingFace: {model_path}")
    return model_path


def load_tokenizer(config):
    model_config = config["model"]
    tokenizer_config = config.get("tokenizer", {})
    model_arch = model_config["arch"]
    model_path = model_config.get("path", model_arch)
    logger = get_logger(config["exp_id"])

    try:
        resolved_path = _resolve_model_path(model_path, logger)
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_path,
            do_lower_case=tokenizer_config.get("do_lower_case", True),
        )
        logger.info(f"Tokenizer加载成功：{model_arch} (路径：{resolved_path})")

        if tokenizer_config.get("add_special_tokens", True) and "special_tokens" in tokenizer_config:
            special_tokens = tokenizer_config["special_tokens"]
            if special_tokens:
                new_tokens = [tok for tok in special_tokens if tok not in tokenizer.get_vocab()]
                if new_tokens:
                    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
                    logger.info(f"已添加自定义特殊token：{new_tokens}")
                else:
                    logger.info("所有特殊token已存在于Tokenizer词汇表中")

        return tokenizer
    except Exception as e:
        logger.error(f"Tokenizer加载失败：{str(e)}", exc_info=True)
        raise


class TaskTextClassifier(nn.Module):
    """模型任务加载器。"""

    def __init__(self, config, tokenizer, backbone=None):
        super().__init__()
        self.config = config
        self.task_type = config["task_type"]
        self.logger = get_logger(config["exp_id"])
        self.tokenizer = tokenizer

        self.bert = backbone or self._load_backbone()
        self._resize_embeddings_if_needed()
        self._inject_lora()
        self._configure_backbone_trainability()
        self.classifiers = self._build_classifiers()
        self.loss_fct = nn.CrossEntropyLoss()

    def _load_backbone(self):
        model_arch = self.config["model"]["arch"]
        model_path = self.config["model"].get("path", model_arch)
        resolved_path = _resolve_model_path(model_path, self.logger)
        return AutoModel.from_pretrained(resolved_path, output_hidden_states=True)

    def _resize_embeddings_if_needed(self):
        if len(self.tokenizer) > self.bert.config.vocab_size:
            self.bert.resize_token_embeddings(len(self.tokenizer))
            self.logger.info(f"Resize embeddings to {len(self.tokenizer)}")

    def _inject_lora(self):
        lora_conf = self.config["model"].get("lora", {})
        if not lora_conf.get("enabled", True):
            self.logger.info("LoRA 已显式关闭，使用纯 backbone + classifier 训练/推理")
            return

        if hasattr(self.bert, "peft_config"):
            self.logger.info("检测到现有 PeftModel，跳过重复注入 LoRA")
            return

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_conf["rank"],
            lora_alpha=lora_conf["alpha"],
            lora_dropout=lora_conf["dropout"],
            target_modules=lora_conf["target_modules"],
            bias=lora_conf.get("bias", "none"),
        )
        self.bert = get_peft_model(self.bert, peft_config)
        self.logger.info("✅ LoRA配置注入成功")

    def _configure_backbone_trainability(self):
        freeze_backbone = self.config["model"].get("freeze_bert", False)
        for name, param in self.bert.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = not freeze_backbone

    def _build_classifiers(self):
        hidden_size = self.bert.config.hidden_size
        dropout_prob = self.config["model"].get("dropout", 0.1)

        label_mapping_config = self.config["data"]["label_mapping"]
        label_col_map, _ = normalize_label_col(
            self.config["data"]["label_col"], self.task_type, label_mapping_config,
        )
        if self.task_type == "single_cls":
            actual_task = next(iter(label_col_map.keys()))
            first_task_label_map = label_mapping_config[actual_task]
            if not isinstance(first_task_label_map, dict):
                raise ValueError(
                    "[MODEL] label_mapping 结构错误：single_cls 下期望 {task_name: {int: str}}"
                )
            mappings = {"default": first_task_label_map}
        else:
            mappings = {task: label_mapping_config[task] for task in label_col_map}

        classifiers = nn.ModuleDict()
        for task_name, label_map in mappings.items():
            num_labels = len(label_map)
            classifiers[task_name] = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_labels),
            )

        return classifiers

    def get_trainable_parameter_summary(self):
        summary = {
            "backbone": 0,
            "lora": 0,
            "classifier": 0,
            "all": 0,
            "trainable": 0,
        }
        for name, param in self.named_parameters():
            numel = param.numel()
            summary["all"] += numel
            if "classifiers" in name:
                bucket = "classifier"
            elif "lora_" in name:
                bucket = "lora"
            else:
                bucket = "backbone"

            if param.requires_grad:
                summary[bucket] += numel
                summary["trainable"] += numel

        return summary

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        logits_dict = {}
        total_loss = 0.0

        _, task_names = normalize_label_col(
            self.config["data"]["label_col"], self.task_type,
        )

        loss_weights = self.config["train"].get("loss_weight", [1.0] * len(task_names))

        for i, task_name in enumerate(task_names):
            if task_name not in self.classifiers:
                continue

            logits = self.classifiers[task_name](pooled_output)
            logits_dict[task_name] = logits

            if labels is None:
                continue

            if self.task_type == "single_cls":
                task_label = next(iter(labels.values())) if isinstance(labels, dict) else labels
            else:
                task_label = labels[task_name]

            loss = self.loss_fct(logits, task_label)
            total_loss += loss * loss_weights[i]

        if labels is not None:
            return total_loss, logits_dict
        return logits_dict
