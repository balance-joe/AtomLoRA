import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from src.utils.logger import get_logger
from src.trainer.metric_manager import MetricManager
from src.eval.runner import run_evaluation
from src.utils.paths import (
    ADAPTER_DIR, CLASSIFIER_DIR, TOKENIZER_DIR, CLASSIFIER_FILE, CONFIG_FILE,
    copy_config_to_output, save_metrics, update_latest_link
)

class Trainer:
    def __init__(self, config, model, train_loader, dev_loader, tokenizer):
        self.config = config
        self.logger = get_logger(config["exp_id"])
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.tokenizer = tokenizer # 需要保存 tokenizer
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 输出路径
        self.output_dir = os.path.join("outputs", config["exp_id"])
        self.tokenizer_save_path = os.path.join(self.output_dir, TOKENIZER_DIR)
        os.makedirs(os.path.join(self.output_dir, ADAPTER_DIR), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, CLASSIFIER_DIR), exist_ok=True)
        copy_config_to_output(config, self.output_dir)
        
        # 初始化组件
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.metric_manager = MetricManager(config)

    def _build_optimizer(self):
        """
        移植旧代码的核心优化器分组逻辑
        """
        # 1. 筛选参数
        lora_params = [p for n, p in self.model.named_parameters() if "lora_" in n and p.requires_grad]
        classifier_params = [p for n, p in self.model.named_parameters() if "classifiers" in n and p.requires_grad]
        bert_params = [p for n, p in self.model.named_parameters() 
                    if "lora_" not in n and "classifiers" not in n and p.requires_grad]

        # 2. 构建分组
        optimizer_grouped_parameters = []
        
        # 从的配置路径读取学习率
        lr_config = self.config["train"]["optimizer"]["groups"]
        
        # LoRA 组
        if lora_params:
            optimizer_grouped_parameters.append({
                "params": lora_params,
                "lr": float(lr_config["lora"]),
                "weight_decay": 0.01
            })
            
        # Classifier 组
        if classifier_params:
            optimizer_grouped_parameters.append({
                "params": classifier_params,
                "lr": float(lr_config["classifier"]),
                "weight_decay": 0.01
            })
            
        # Bert 组 (通常为空，因为冻结了)
        if bert_params:
            optimizer_grouped_parameters.append({
                "params": bert_params,
                "lr": float(lr_config["bert"]),
                "weight_decay": 0.001
            })
            
        self.logger.info(f"优化器组: LoRA({len(lora_params)}), Clf({len(classifier_params)}), Bert({len(bert_params)})")
        return torch.optim.AdamW(optimizer_grouped_parameters)

    def _build_scheduler(self):
        num_epochs = self.config["train"]["num_epochs"]
        grad_accum = self.config["train"].get("gradient_accumulation_steps", 1)
        total_steps = len(self.train_loader) * num_epochs // grad_accum
        warmup_ratio = self.config["train"].get("warmup_ratio", 0.1)

        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps
        )

    def _log_training_monitor(self, step, logits_dict, labels):
        """定期打印训练过程中的置信度和 LoRA 权重状态，辅助调试"""
        self.model.eval()
        try:
            with torch.no_grad():
                for task_name, logits in logits_dict.items():
                    probs = F.softmax(logits.detach(), dim=-1)
                    max_probs = probs.max(dim=-1).values
                    mean_conf = max_probs.mean().item()

                    preds = probs.argmax(dim=-1)
                    if isinstance(labels, dict):
                        true_labels = labels.get(task_name, next(iter(labels.values())))
                    else:
                        true_labels = labels
                    train_acc = (preds == true_labels).float().mean().item()

                    self.logger.info(
                        f"📊 Step {step} | {task_name} | "
                        f"置信度: {mean_conf:.4f} | 训练准确率: {train_acc:.4f}"
                    )

            # LoRA 权重健康检查
            for name, param in self.model.bert.named_parameters():
                if "lora_B" in name and param.requires_grad:
                    lora_b_mean = torch.mean(param.data).item()
                    self.logger.info(f"📌 Step {step} | {name} 均值: {lora_b_mean:.6f}")
                    break
        finally:
            self.model.train()

    def train(self):
        epochs = self.config["train"]["num_epochs"]
        grad_accum = self.config["train"].get("gradient_accumulation_steps", 1)
        best_score = float('-inf')
        best_metrics = {}
        global_step = 0
        monitor_interval = self.config["train"].get("monitor_interval", 100)

        self.logger.info("开始训练...")

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}")
            self.optimizer.zero_grad()

            for step, batch in enumerate(pbar):
                # 移动数据
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]
                if isinstance(labels, dict):
                    labels = {k: v.to(self.device) for k, v in labels.items()}
                else:
                    labels = labels.to(self.device)

                # Forward
                result = self.model(input_ids, mask, labels)

                if isinstance(result, tuple) and len(result) == 2:
                    loss, logits_dict = result
                elif isinstance(result, torch.Tensor):
                    loss = result
                    logits_dict = None
                else:
                    raise ValueError(f"模型返回了意外的类型: {type(result)}")

                # 梯度累积
                loss = loss / grad_accum
                loss.backward()

                epoch_loss += loss.item()

                if (step + 1) % grad_accum == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # 训练监控：定期打印 logits 置信度和 LoRA 权重状态
                    if global_step % monitor_interval == 0 and logits_dict is not None:
                        self._log_training_monitor(global_step, logits_dict, labels)

                    global_step += 1

                pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})

            # 处理 epoch 末尾不足一个累积窗口的残余梯度
            if (step + 1) % grad_accum != 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # 评估
            metrics = self.evaluate()
            log_msg = f"Epoch {epoch} | " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(log_msg)
            
            # 保存最优模型
            current_score = metrics["main_score"]
            if current_score > best_score:
                best_score = current_score
                best_metrics = metrics
                self.save_model()
                self.logger.info(f">>> 新模型诞生! 分数: {best_score:.4f}")

        # 训练结束后保存最优指标
        save_metrics(best_metrics, self.output_dir)
        self.logger.info(f"✅ 最优指标已保存至: {os.path.join(self.output_dir, 'metrics.json')}")

        # 创建 latest 链接，方便用户直接访问最新实验
        config_copy = os.path.join(self.output_dir, CONFIG_FILE)
        update_latest_link(self.output_dir, metrics=best_metrics, config_path=config_copy)
        self.logger.info(f"✅ outputs/latest -> {self.output_dir}")

    def evaluate(self):
        return run_evaluation(
            self.model, self.dev_loader, self.config,
            self.metric_manager, self.device, show_progress=False
        )

    
    def save_model(self):
        """
        [PEFT标准] 保存 LoRA 适配器、分类头、Tokenizer
        """
        # 1. 保存 LoRA 适配器 -> adapter/ 子目录
        adapter_dir = os.path.join(self.output_dir, ADAPTER_DIR)
        self.model.bert.save_pretrained(adapter_dir)
        self.logger.info(f"✅ LoRA 适配器权重已保存至: {adapter_dir}")

        # 2. 保存 Tokenizer
        self.tokenizer.save_pretrained(self.tokenizer_save_path)

        # 3. 保存 Classifiers -> classifier/ 子目录
        clf_dir = os.path.join(self.output_dir, CLASSIFIER_DIR)
        os.makedirs(clf_dir, exist_ok=True)
        torch.save(self.model.classifiers.state_dict(), os.path.join(clf_dir, CLASSIFIER_FILE))
        self.logger.info(f"✅ 分类头权重已保存至: {clf_dir}")
