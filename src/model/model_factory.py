# src/model/model_factory.py
from transformers import AutoTokenizer, AutoConfig,AutoModelForSequenceClassification , AutoModel
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn
from src.utils.logger import get_logger
import os


def _resolve_model_path(model_path: str, logger) -> str:
    """判断 model_path 是本地目录还是 HuggingFace repo id，返回可用路径。"""
    # 优先检查是否是本地目录（绝对路径或相对路径）
    abs_path = os.path.abspath(model_path)
    if os.path.isdir(abs_path):
        logger.info(f"Loading local model from: {abs_path}")
        return abs_path

    # 相对路径以 ./ 或 ../ 开头但目录不存在 → 报错
    if model_path.startswith(".") or model_path.startswith("/"):
        raise FileNotFoundError(
            f"[MODEL] 本地模型路径不存在: {abs_path}\n"
            f"  解决方案（任选其一）:\n"
            f"  1. 下载模型到该路径\n"
            f"  2. 改用 HuggingFace repo id（如 model.path: 'bert-base-chinese'）"
        )

    # 其余情况视为 HuggingFace repo id（如 bert-base-chinese, meta-llama/Llama-3-8B）
    logger.info(f"Loading model from HuggingFace: {model_path}")
    return model_path


def load_tokenizer(config):
    """
    加载Tokenizer（适配配置中的特殊token+模型参数，修复add_special_tokens冲突）
    """
    model_config = config["model"]
    tokenizer_config = config.get("tokenizer", {})
    model_arch = model_config["arch"]
    model_path = model_config.get("path", model_arch)
    logger = get_logger(config["exp_id"])

    try:
        resolved_path = _resolve_model_path(model_path, logger)
        # 基础Tokenizer加载
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_path,
            do_lower_case=tokenizer_config.get("do_lower_case", True)  # 仅保留有效参数
        )
        logger.info(f"Tokenizer加载成功：{model_arch} (路径：{resolved_path})")
        
        # 加载自定义特殊token（通过tokenizer.add_special_tokens方法，与配置解耦）
        if tokenizer_config.get("add_special_tokens", True) and "special_tokens" in tokenizer_config:
            special_tokens = tokenizer_config["special_tokens"]
            if special_tokens:
                new_tokens = [tok for tok in special_tokens if tok not in tokenizer.get_vocab()]
                if new_tokens:
                    # 正确添加特殊token的方式
                    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
                    logger.info(f"已添加自定义特殊token：{new_tokens}")
                else:
                    logger.info("所有特殊token已存在于Tokenizer词汇表中")
        
        return tokenizer

    except Exception as e:
        logger.error(f"Tokenizer加载失败：{str(e)}", exc_info=True)
        raise e



class TaskTextClassifier(nn.Module):
    """
    模型任务加载器
    """
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.task_type = config["task_type"]
        self.logger = get_logger(config["exp_id"])
        
        model_arch = config["model"]["arch"]
        model_path = config["model"].get("path", model_arch)
        resolved_path = _resolve_model_path(model_path, self.logger)

        # 1. 使用AutoModel 加载基础BERT模型
        self.bert = AutoModel.from_pretrained(
            resolved_path,
            output_hidden_states=True  # 确保输出隐藏状态
        )
        
        # 2. Resize Embedding
        if len(tokenizer) > self.bert.config.vocab_size:
            self.bert.resize_token_embeddings(len(tokenizer))
            self.logger.info(f"Resize embeddings to {len(tokenizer)}")

        # 3. LoRA任务类型使用FEATURE_EXTRACTION
        self._inject_lora()
        
        # 4. 构建分类头
        self.classifiers = self._build_classifiers()
        
        # 5. 损失函数
        self.loss_fct = nn.CrossEntropyLoss()

    def _inject_lora(self):
        lora_conf = self.config["model"]["lora"]
        if not lora_conf.get("enabled", False):
            return  # LoRA 未启用，跳过注入

        if hasattr(self.bert, "peft_config"):
            self.logger.warning("⚠️ LoRA结构已存在（跳过重复注入）。")
            return

        # LoRA配置
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_conf["rank"],
            lora_alpha=lora_conf["alpha"],
            lora_dropout=lora_conf["dropout"],
            target_modules=lora_conf["target_modules"],
            bias=lora_conf.get("bias", "none")
        )
        self.bert = get_peft_model(self.bert, peft_config)
        self.logger.info("✅ LoRA配置注入成功")
        self.bert.print_trainable_parameters()
        
        
    def _build_classifiers(self):
        hidden_size = self.bert.config.hidden_size
        dropout_prob = self.config["model"].get("dropout", 0.1)
        
        # 从label_col获取任务名称
        if self.task_type == "single_cls":
            label_mapping_config = self.config["data"]["label_mapping"]
            first_task_key = next(iter(label_mapping_config.keys()))
            first_task_label_map = label_mapping_config[first_task_key]
            # 防御：first_task_label_map 必须是 {int: str} 字典，不能是字符串
            if not isinstance(first_task_label_map, dict):
                raise ValueError(
                    f"[MODEL] label_mapping 结构错误：single_cls 下期望 {{task_name: {{int: str}}}}，"
                    f"但 key='{first_task_key}' 的值是 {type(first_task_label_map).__name__}。"
                    f"请检查 data.label_mapping 配置。"
                )
            mappings = {"default": first_task_label_map}
        else:
            # 多任务：从label_col的键获取任务名称
            label_col_config = self.config["data"]["label_col"]
            if not isinstance(label_col_config, dict):
                raise ValueError("multi_cls 任务要求 data.label_col 为字典 (task_name -> col_name)")
            task_names = list(label_col_config.keys())
            label_mapping_config = self.config["data"]["label_mapping"]
            mappings = {task: label_mapping_config[task] for task in task_names}

        classifiers = nn.ModuleDict()

        for task_name, label_map in mappings.items():
            num_labels = len(label_map)
            classifiers[task_name] = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_labels)
            )
            
        return classifiers

    def forward(self, input_ids, attention_mask, labels=None):
        bert_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # BERT 前向传播
        outputs = self.bert(**bert_kwargs)
        
        # 获取[CLS] token的池化输出
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        # 这个打开了每一轮都会配置
        # print(f"Pooled output shape: {pooled_output.shape}")

        logits_dict = {}
        total_loss = 0.0
        
        # 从label_col获取任务名称
        if self.task_type == "single_cls":
            task_names = ["default"]
        else:
            # 直接从label_col的键获取任务名称
            label_col_config = self.config["data"]["label_col"]
            if not isinstance(label_col_config, dict):
                raise ValueError("multi_cls 任务要求 data.label_col 为字典 (task_name -> col_name)")
            task_names = list(label_col_config.keys())
        
        loss_weights = self.config["train"].get("loss_weight", [1.0] * len(task_names))
        
        # 遍历分类头计算logits和loss
        for i, task_name in enumerate(task_names):
            if task_name not in self.classifiers: 
                continue
            
            # 计算当前任务的logits
            logits = self.classifiers[task_name](pooled_output)
            logits_dict[task_name] = logits            
            
            if labels is not None:
                if self.task_type == "single_cls":
                    # 处理单任务下labels是字典的情况
                    if isinstance(labels, dict):
                        # 取字典中第一个任务的标签（或指定"misreport"）
                        task_label = next(iter(labels.values()))  # 通用取第一个
                    else:
                        task_label = labels
                    loss = self.loss_fct(logits, task_label)
                    total_loss += loss * loss_weights[i]
                else:
                    # 多任务逻辑不变
                    if isinstance(labels, dict) and task_name in labels:
                        task_label = labels[task_name]
                        loss = self.loss_fct(logits, task_label)
                        total_loss += loss * loss_weights[i]
        
        if labels is not None:
            return total_loss, logits_dict
        else:
            return logits_dict