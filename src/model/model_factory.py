# src/model/model_factory.py
from transformers import AutoTokenizer, AutoConfig,AutoModelForSequenceClassification , AutoModel
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn
from src.utils.logger import get_logger
import os


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
        # 基础Tokenizer加载（移除冲突的add_special_tokens参数）
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            do_lower_case=tokenizer_config.get("do_lower_case", True)  # 仅保留有效参数
        )
        logger.info(f"Tokenizer加载成功：{model_arch} (路径：{model_path})")
        
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



class MultiTaskTextClassifier(nn.Module):
    """
    """
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.task_type = config["task_type"]
        self.logger = get_logger(config["exp_id"])
        
        # 1. 关键修改：使用AutoModel而不是AutoModelForSequenceClassification
        model_arch = config["model"]["arch"]
        model_path = config["model"].get("path", model_arch)
        self.bert = AutoModel.from_pretrained(
            model_path,
            output_hidden_states=True  # 确保输出隐藏状态
        )
        
        # 2. Resize Embedding (保持不变)
        if len(tokenizer) > self.bert.config.vocab_size:
            self.bert.resize_token_embeddings(len(tokenizer))
            self.logger.info(f"Resize embeddings to {len(tokenizer)}")

        # 3. 关键修改：LoRA任务类型改为FEATURE_EXTRACTION
        self._inject_lora()
        
        # 4. 构建分类头 (保持不变)
        self.classifiers = self._build_classifiers()
        
        # 5. 损失函数
        self.loss_fct = nn.CrossEntropyLoss()

    def _inject_lora(self):
        lora_conf = self.config["model"]["lora"]
        # 关键修改：TaskType改为FEATURE_EXTRACTION
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # 修改这里
            inference_mode=False,
            r=lora_conf["rank"],
            lora_alpha=lora_conf["alpha"],
            lora_dropout=lora_conf["dropout"],
            target_modules=lora_conf["target_modules"],
            bias=lora_conf.get("bias", "none")
        )
        self.bert = get_peft_model(self.bert, peft_config)
        self.bert.print_trainable_parameters()
        
        
    def _build_classifiers(self):
        hidden_size = self.bert.config.hidden_size
        dropout_prob = self.config["model"].get("dropout", 0.1)
        
        # 关键修改：从label_col获取任务名称
        if self.task_type == "single_cls":
            # 单任务：使用默认映射 这里有问题，没适配这个
            mappings = {"default": self.config["data"]["label_map"]} #
        else:
            # 多任务：从label_col的键获取任务名称
            label_col_config = self.config["data"]["label_col"]
            if isinstance(label_col_config, dict):
                task_names = list(label_col_config.keys())
                # 从label_mapping获取对应的标签映射
                label_mapping_config = self.config["data"]["label_mapping"]
                mappings = {task: label_mapping_config[task] for task in task_names}
            else:
                # 回退到旧逻辑
                mappings = self.config["data"]["label_maps"]

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
        
        # 关键修改：正确获取[CLS] token的池化输出
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        
        print(f"Pooled output shape: {pooled_output.shape}")
        
        logits_dict = {}
        total_loss = 0.0
        
        # 关键修改：从label_col获取任务名称
        if self.task_type == "single_cls":
            task_names = ["default"]
        else:
            # 直接从label_col的键获取任务名称
            label_col_config = self.config["data"]["label_col"]
            if isinstance(label_col_config, dict):
                task_names = list(label_col_config.keys())
            else:
                # 回退逻辑
                task_names = ["misreport", "risk"]  # 默认值
        
        loss_weights = self.config["train"].get("loss_weight", [1.0] * len(task_names))
        
        # 遍历分类头计算logits和loss
        for i, task_name in enumerate(task_names):
            if task_name not in self.classifiers: 
                continue
            
            logits = self.classifiers[task_name](pooled_output)
            logits_dict[task_name] = logits
            
            if labels is not None:
                if self.task_type == "single_cls":
                    task_label = labels
                    loss = self.loss_fct(logits, task_label)
                    total_loss += loss * loss_weights[i]
                else:
                    if isinstance(labels, dict) and task_name in labels:
                        task_label = labels[task_name]
                        
                        loss = self.loss_fct(logits, task_label)
                        total_loss += loss * loss_weights[i]
        
        if labels is not None:
            return total_loss, logits_dict
        else:
            return logits_dict