import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from src.utils.logger import get_logger
from src.trainer.metric_manager import MetricManager

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
        self.lora_save_path = os.path.join(self.output_dir, "lora_adapter")
        self.tokenizer_save_path = os.path.join(self.output_dir, "tokenizer")
        os.makedirs(self.output_dir, exist_ok=True)
        
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

        # 2. 构建分组（关键修改：从config的正确路径读取）
        optimizer_grouped_parameters = []
        
        # 关键修改：从正确的配置路径读取学习率
        lr_config = self.config["train"]["optimizer"]["groups"]
        
        # LoRA 组
        if lora_params:
            optimizer_grouped_parameters.append({
                "params": lora_params,
                "lr": float(lr_config["lora"]),  # 修正路径
                "weight_decay": 0.01
            })
            
        # Classifier 组
        if classifier_params:
            optimizer_grouped_parameters.append({
                "params": classifier_params,
                "lr": float(lr_config["classifier"]),  # 修正路径
                "weight_decay": 0.01
            })
            
        # Bert 组 (通常为空，因为冻结了)
        if bert_params:
            optimizer_grouped_parameters.append({
                "params": bert_params,
                "lr": float(lr_config["bert"]),  # 修正路径
                "weight_decay": 0.001
            })
            
        self.logger.info(f"Optimizer groups: LoRA({len(lora_params)}), Clf({len(classifier_params)}), Bert({len(bert_params)})")
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

    def train(self):
        epochs = self.config["train"]["num_epochs"]
        grad_accum = self.config["train"].get("gradient_accumulation_steps", 1)
        best_score = float('-inf')
        global_step = 0
        
        self.logger.info("Start Training...")
        
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
                
                # Forward - 安全处理返回值
                result = self.model(input_ids, mask, labels)
                
                # 检查返回值类型
                if isinstance(result, tuple) and len(result) == 2:
                    # 如果返回 (loss, logits_dict)
                    loss, logits_dict = result
                    # 这问题先备注一下子，后续有时间了再说  
                    # logits_dict未被使用：在训练过程中获取了模型的logits输出，但没有用于任何实际用途
                    # 缺乏训练监控：无法实时查看模型预测效果和置信度
                    # 调试困难：无法分析模型在学习什么
                elif isinstance(result, torch.Tensor):
                    # 如果只返回 loss 张量
                    loss = result
                    logits_dict = None
                else:
                    # 处理意外情况
                    raise ValueError(f"模型返回了意外的类型: {type(result)}")

                # 梯度累积
                loss = loss / grad_accum
                loss.backward()
                
                epoch_loss += loss.item()
                
                if (step + 1) % grad_accum == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
                pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})
            
            # 评估
            metrics = self.evaluate()
            log_msg = f"Epoch {epoch} | " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(log_msg)
            
            # 保存最优模型
            current_score = metrics.get("main_score", 0)
            if current_score > best_score:
                best_score = current_score
                self.save_model()
                self.logger.info(f">>> New Best Model Saved! Score: {best_score:.4f}")

    def evaluate(self):
        self.model.eval()
        all_logits = {}
        all_labels = {}
        
        # 关键修改：从label_col获取任务名称
        if self.config["task_type"] == "single_cls":
            task_names = ["default"]
        else:
            label_col_config = self.config["data"]["label_col"]
            if isinstance(label_col_config, dict):
                task_names = list(label_col_config.keys())
            else:
                task_names = ["misreport", "risk"]
        
        for t in task_names:
            all_logits[t] = []
            all_labels[t] = []
            
        with torch.no_grad():
            for batch in self.dev_loader:
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]
                
                outputs = self.model(input_ids, mask, labels=None)
                
                for t in task_names:
                    if self.config["task_type"] == "single_cls":
                        logit = outputs.get("default", outputs) if isinstance(outputs, dict) else outputs
                        label = labels
                    else:
                        logit = outputs[t]
                        label = labels[t]
                        
                    all_logits[t].append(logit.cpu())
                    all_labels[t].append(label.cpu())

        for t in task_names:
            all_logits[t] = torch.cat(all_logits[t], dim=0)
            all_labels[t] = torch.cat(all_labels[t], dim=0)

        return self.metric_manager.compute(all_logits, all_labels)
    
    def save_model(self):
        """
        移植旧代码的保存逻辑：分别保存 LoRA, Tokenizer, Classifiers
        """
        # 1. 保存 LoRA (model.bert 是 PeftModel)
        self.model.bert.save_pretrained(self.lora_save_path)
        
        # 2. 保存 Tokenizer
        self.tokenizer.save_pretrained(self.tokenizer_save_path)
        
        # 3. 保存 Classifiers (state_dict)
        clf_path = os.path.join(self.output_dir, "classifiers.pt")
        torch.save(self.model.classifiers.state_dict(), clf_path)
        
        # 4. 保存 Config 快照 (可选)
        # torch.save(self.config, os.path.join(self.output_dir, "config.pt"))