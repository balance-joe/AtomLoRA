import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def collate_fn(batch):
    """
    处理 batch 数据，兼容单任务和双任务的 label 结构
    """
    # 1. 基础输入
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    
    # 2. 标签处理
    first_label = batch[0]['labels']
    
    if isinstance(first_label, dict):
        # 双任务逻辑
        labels = {}
        for task_key in first_label.keys():
            # 获取该任务的所有标签列表
            raw_labels = [item['labels'][task_key] for item in batch]
            
            # =============== 核心修复与调试 ==================
            try:
                # 尝试强制转为 int 列表，防止 list 中包含 str
                int_labels = [int(l) for l in raw_labels]
                labels[task_key] = torch.tensor(int_labels, dtype=torch.long)
            except ValueError as e:
                print(f"❌ 数据类型错误！任务 '{task_key}' 的标签包含非数字字符。")
                print(f"前5个样本的原始标签: {raw_labels[:5]}")
                raise e
            # ===============================================
            
    else:
        # 单任务逻辑
        raw_labels = [item['labels'] for item in batch]
        try:
            int_labels = [int(l) for l in raw_labels]
            labels = torch.tensor(int_labels, dtype=torch.long)
        except ValueError as e:
            print(f"❌ 数据类型错误！标签包含非数字字符: {raw_labels[:5]}")
            raise e
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
def create_dataloader(data, batch_size, shuffle=False):
    dataset = TextDataset(data)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn
    )