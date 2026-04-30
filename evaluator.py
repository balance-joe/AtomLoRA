import os
import sys
import torch

# 等价于 src.utils.logger.ensure_utf8_stdio()，此处提前执行因为 src 尚未可导入
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
from peft import PeftModel
from transformers import AutoTokenizer, AutoModel
from src.utils.logger import get_logger
from src.utils.device import resolve_device
from src.trainer.metric_manager import MetricManager
from src.model.model_factory import TaskTextClassifier, _resolve_model_path
from src.data.data_processor import load_dataset
from src.model.text_dataset import create_dataloader
from src.eval.runner import run_evaluation
from src.utils.paths import resolve_adapter_path, resolve_classifier_path, resolve_tokenizer_path


class Evaluator:
    """评估器：加载已训练的模型并对数据集进行评估"""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(config["exp_id"])
        self.device = resolve_device(config)

        self.metric_manager = MetricManager(config)

        self.output_dir = os.path.join("outputs", config["exp_id"])
        self.lora_path = resolve_adapter_path(self.output_dir)
        self.clf_path = resolve_classifier_path(self.output_dir)
        self.tokenizer_path = resolve_tokenizer_path(self.output_dir)

        self._load_tokenizer()
        self._load_model()

    def _load_tokenizer(self):
        """加载 Tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.logger.info("Tokenizer loaded for evaluation")

    def _load_model(self):
        """加载基座模型、LoRA 适配器和分类头"""
        model_arch = self.config["model"]["arch"]
        model_path = self.config["model"].get("path", model_arch)
        resolved_path = _resolve_model_path(model_path, self.logger)

        base_model = AutoModel.from_pretrained(
            resolved_path,
            output_hidden_states=True
        )

        # embedding 层需要匹配训练时的词表大小（包含特殊 token）
        if len(self.tokenizer) > base_model.config.vocab_size:
            base_model.resize_token_embeddings(len(self.tokenizer))

        lora_enabled = self.config["model"].get("lora", {}).get("enabled", True)
        if lora_enabled:
            self.bert = PeftModel.from_pretrained(base_model, self.lora_path)
        else:
            self.bert = base_model
        self.model = TaskTextClassifier(self.config, self.tokenizer, backbone=self.bert)

        self.model.classifiers.load_state_dict(
            torch.load(self.clf_path, map_location="cpu")
        )

        self.model.to(self.device)
        self.model.eval()

        self.logger.info("Model + LoRA + classifier loaded")

    def evaluate(self, data_path):
        """对指定数据集进行评估，返回指标字典"""
        if not data_path:
            raise ValueError("评估缺少 data_path，请在配置中提供 data.dev_path 或显式传入路径")
        data = load_dataset(self.config, data_path, self.tokenizer)
        loader = create_dataloader(
            data,
            batch_size=self.config["train"]["batch_size"],
            shuffle=False
        )
        return run_evaluation(
            self.model, loader, self.config,
            self.metric_manager, self.device, desc="Evaluating"
        )
