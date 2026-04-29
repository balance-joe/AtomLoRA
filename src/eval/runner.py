"""共享评估逻辑，供 Trainer 和 Evaluator 复用。"""
import torch
from typing import Dict, Any, Optional
from tqdm import tqdm


def resolve_task_names(config):
    """从配置中解析任务名称和 single_cls 的标签键。"""
    if config["task_type"] == "single_cls":
        task_names = ["default"]
        label_col_config = config["data"]["label_col"]
        single_task_key = next(iter(label_col_config.keys())) if isinstance(label_col_config, dict) else label_col_config
        return task_names, single_task_key
    else:
        label_col_config = config["data"]["label_col"]
        if not isinstance(label_col_config, dict):
            raise ValueError("multi_cls 任务要求 data.label_col 为字典")
        return list(label_col_config.keys()), None


def run_evaluation(model, dataloader, config, metric_manager, device,
                   desc: str = "Evaluating", show_progress: bool = True):
    """
    通用评估流程：推理 → 收集 logits/labels → 计算指标。

    Args:
        model: 已设置为 eval 模型的 nn.Module
        dataloader: 评估数据加载器
        config: 完整配置字典
        metric_manager: MetricManager 实例
        device: torch.device
        desc: 进度条描述
        show_progress: 是否显示 tqdm 进度条

    Returns:
        Dict[str, float]: 评估指标
    """
    model.eval()
    task_names, single_task_key = resolve_task_names(config)

    all_logits: Dict[str, list] = {t: [] for t in task_names}
    all_labels: Dict[str, list] = {t: [] for t in task_names}

    iterator = tqdm(dataloader, desc=desc) if show_progress else dataloader

    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids, mask, labels=None)

            for t in task_names:
                if config["task_type"] == "single_cls":
                    logit = outputs.get("default", outputs) if isinstance(outputs, dict) else outputs
                    if isinstance(labels, dict):
                        label = labels[single_task_key]
                    else:
                        label = labels
                else:
                    logit = outputs[t]
                    label = labels[t]

                all_logits[t].append(logit.cpu())
                all_labels[t].append(label.cpu())

    for t in task_names:
        all_logits[t] = torch.cat(all_logits[t], dim=0)
        all_labels[t] = torch.cat(all_labels[t], dim=0)

    metric_manager.validate_inputs(all_logits, all_labels)
    return metric_manager.compute(all_logits, all_labels)
