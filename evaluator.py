import os
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModel
from src.utils.logger import get_logger
from src.trainer.metric_manager import MetricManager
from src.model.model_factory import TaskTextClassifier
from src.data.data_processor import load_dataset
from src.model.text_dataset import create_dataloader


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(config["exp_id"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.metric_manager = MetricManager(config)

        self.output_dir = os.path.join("outputs", config["exp_id"])
        self.lora_path = os.path.join(self.output_dir, "lora_adapter")
        self.clf_path = os.path.join(self.output_dir, "classifiers.pt")
        self.tokenizer_path = os.path.join(self.output_dir, "tokenizer")

        self._load_tokenizer()
        self._load_model()

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.logger.info("Tokenizer loaded for evaluation")

    def _load_model(self):
        model_arch = self.config["model"]["arch"]
        model_path = self.config["model"].get("path", model_arch)

        base_model = AutoModel.from_pretrained(
            model_path,
            output_hidden_states=True
        )

        self.bert = PeftModel.from_pretrained(
            base_model,
            self.lora_path
        )

        self.model = TaskTextClassifier(self.config, self.tokenizer)
        self.model.bert = self.bert

        self.model.classifiers.load_state_dict(
            torch.load(self.clf_path, map_location="cpu")
        )

        self.model.to(self.device)
        self.model.eval()

        self.logger.info("Model + LoRA + classifier loaded")

    def evaluate(self, data_path):
        data = load_dataset(self.config, data_path, self.tokenizer)
        loader = create_dataloader(
            data,
            batch_size=self.config["train"]["batch_size"],
            shuffle=False
        )

        all_logits, all_labels = {}, {}

        # ========== 完全对齐训练代码的task_names逻辑 ==========
        if self.config["task_type"] == "single_cls":
            task_names = ["default"]  # 模型输出键固定为default
            # 提取配置里的实际标签键（你的是class）
            label_col_config = self.config["data"]["label_col"]
            if isinstance(label_col_config, dict):
                single_task_key = next(iter(label_col_config.keys()))  # 拿到class
            else:
                single_task_key = label_col_config
        else:
            label_col_config = self.config["data"]["label_col"]
            if isinstance(label_col_config, dict):
                task_names = list(label_col_config.keys())
            else:
                task_names = ["misreport", "risk"]

        # 初始化容器
        for t in task_names:
            all_logits[t] = []
            all_labels[t] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                outputs = self.model(input_ids, mask, labels=None)

                # ========== 完全对齐训练代码的取值逻辑 ==========
                for t in task_names:
                    if self.config["task_type"] == "single_cls":
                        # 取模型输出：优先default键，非字典则取整个outputs
                        logit = outputs.get("default", outputs) if isinstance(outputs, dict) else outputs
                        # 取标签：用single_task_key（class）从labels字典取值
                        if isinstance(labels, dict):
                            label = labels[single_task_key]
                        else:
                            label = labels
                    else:
                        logit = outputs[t]
                        label = labels[t]

                    all_logits[t].append(logit.cpu())
                    all_labels[t].append(label.cpu())

        # 拼接结果
        for t in task_names:
            all_logits[t] = torch.cat(all_logits[t], dim=0)
            all_labels[t] = torch.cat(all_labels[t], dim=0)

        noise_id = self.config["data"]["label_map"]["无关噪声类"]

        for t in all_logits:
            logits = all_logits[t]
            fake_logits = torch.zeros_like(logits)
            fake_logits[:, noise_id] = 1000.0  # 极大值，确保 argmax 命中
            all_logits[t] = fake_logits

        metrics = self.metric_manager.compute(all_logits, all_labels)
        return metrics