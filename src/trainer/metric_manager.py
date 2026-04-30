import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

from src.utils.logger import get_logger

logger = get_logger()


class MetricManager:
    def __init__(self, config):
        self.task_type = config["task_type"]
        self.config = config

    def validate_inputs(self, all_logits, all_labels):
        """验证 logits/labels 的 shape、值域、任务对齐。不通过则抛异常。"""
        if set(all_logits.keys()) != set(all_labels.keys()):
            raise ValueError(
                f"任务名不匹配: logits={set(all_logits.keys())}, labels={set(all_labels.keys())}"
            )

        for task_name in all_logits:
            logits = all_logits[task_name]
            labels = all_labels[task_name]

            # shape 对齐
            if logits.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"任务 '{task_name}' 样本数不匹配: logits={logits.shape[0]}, labels={labels.shape[0]}"
                )

            num_classes = logits.shape[1]
            label_min = int(labels.min())
            label_max = int(labels.max())

            # 值域检查
            if label_min < 0:
                raise ValueError(
                    f"任务 '{task_name}' 存在负标签: min={label_min}"
                )
            if label_max >= num_classes:
                raise ValueError(
                    f"任务 '{task_name}' 标签越界: max={label_max}, num_classes={num_classes}"
                )

            # 诊断日志（仅首次打印）
            unique_labels = torch.unique(labels).tolist()
            preds = np.argmax(logits.numpy(), axis=1)
            unique_preds = np.unique(preds).tolist()
            logger.info(
                f"[eval-check] {task_name}: "
                f"samples={logits.shape[0]}, classes={num_classes}, "
                f"label_range=[{label_min},{label_max}], "
                f"unique_labels={unique_labels}, unique_preds={unique_preds}"
            )

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
            
        # 2. 多任务
        loss_weights = self.config["train"].get("loss_weight", [1.0, 1.0])

        for i, (task_name, logits) in enumerate(all_logits.items()):
            labels = all_labels[task_name].numpy()
            preds = np.argmax(logits.numpy(), axis=1)

            score = f1_score(labels, preds, average='macro')
            metrics[f"{task_name}_f1"] = score
            metrics[f"{task_name}_acc"] = accuracy_score(labels, preds)
            scores_for_avg.append(score)
            
        # 加权平均
        avg_f1 = np.average(scores_for_avg, weights=loss_weights[:len(scores_for_avg)])
        metrics["avg_f1"] = avg_f1
        metrics["main_score"] = avg_f1
        
        return metrics
