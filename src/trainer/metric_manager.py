import numpy as np
from sklearn.metrics import f1_score, accuracy_score

class MetricManager:
    def __init__(self, config):
        self.task_type = config["task_type"]
        self.config = config
        
    def compute(self, all_logits, all_labels):
        """
        all_logits: Dict[task_name, Tensor]
        all_labels: Dict[task_name, Tensor]
        """
        metrics = {}
        scores_for_avg = []
        
        # 1. 单任务
        if self.task_type == "single_cls":
            logits = all_logits["default"]
            labels = all_labels["default"].numpy()
            preds = np.argmax(logits.numpy(), axis=1)
            
            f1 = f1_score(labels, preds, average='macro') # 或 binary
            metrics["acc"] = accuracy_score(labels, preds)
            metrics["f1"] = f1
            metrics["main_score"] = f1
            return metrics
            
        # 2. 双任务 (定制逻辑：misreport -> binary, risk -> macro)
        # 必须知道哪个 task 是哪个含义，这里通过 task name 判断
        # 假设 config['data']['task_names'] = ['misreport', 'risk_level']
        
        loss_weights = self.config["train"].get("loss_weight", [1.0, 1.0])
        
        for i, (task_name, logits) in enumerate(all_logits.items()):
            labels = all_labels[task_name].numpy()
            preds = np.argmax(logits.numpy(), axis=1)
            
            # 定制 metric
            if "risk" in task_name:
                # 风险等级：Macro F1
                score = f1_score(labels, preds, average='macro')
                metrics[f"{task_name}_f1_macro"] = score
            else:
                # 误报：Binary F1 (假设 0/1) 或 Macro
                # 旧代码用的是 binary
                score = f1_score(labels, preds, average='macro') # 安全起见用 macro，或根据 id 0/1 调整
                metrics[f"{task_name}_f1"] = score
                
            metrics[f"{task_name}_acc"] = accuracy_score(labels, preds)
            scores_for_avg.append(score)
            
        # 加权平均
        avg_f1 = np.average(scores_for_avg, weights=loss_weights[:len(scores_for_avg)])
        metrics["avg_f1"] = avg_f1
        metrics["main_score"] = avg_f1
        
        return metrics