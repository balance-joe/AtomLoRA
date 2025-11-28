# src/model/model_factory.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
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


def load_lora_model(config, tokenizer, num_labels_dict):
    """
    加载带LoRA的模型（适配配置中的LoRA参数）
    """
    model_config = config["model"]
    lora_config = model_config["lora"]
    model_arch = model_config["arch"]
    model_path = model_config.get("path", model_arch)
    logger = get_logger(config["exp_id"])

    try:
        # 加载基础BERT模型（双任务需适配num_labels）
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=sum(num_labels_dict.values()) if isinstance(num_labels_dict, dict) else num_labels_dict,
            ignore_mismatched_sizes=True
        )
        
        # 适配Tokenizer新增的特殊token（调整embedding层大小）
        base_model.resize_token_embeddings(len(tokenizer))
        
        # 构建LoRA配置（完全匹配你的配置参数）
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_config["rank"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            target_modules=lora_config["target_modules"],
            bias=lora_config["bias"]
        )
        
        # 绑定LoRA到基础模型
        lora_model = get_peft_model(base_model, peft_config)
        
        # 冻结原始BERT（若配置开启）
        if model_config.get("freeze_bert", True):
            for name, param in lora_model.base_model.named_parameters():
                if "lora" not in name and "bias" not in name:
                    param.requires_grad = False
        
        # 打印训练参数统计
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in lora_model.parameters())
        logger.info(f"LoRA模型加载成功 | 训练参数占比：{trainable_params/total_params:.4f} ({trainable_params}/{total_params})")
        
        return lora_model

    except Exception as e:
        logger.error(f"LoRA模型加载失败：{str(e)}", exc_info=True)
        raise e