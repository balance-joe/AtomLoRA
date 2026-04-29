# AtomLoRA - 高效的大语言模型微调框架

## 项目概述

**AtomLoRA** 是一个基于 **LoRA (Low-Rank Adaptation)** 技术的轻量级大语言模型微调框架，专为快速迭代和高效训练而设计。通过参数高效的微调方式，能够在有限的计算资源下实现对预训练模型（如 BERT、ERNIE 等）的快速适配。

### 核心特性

- **参数高效微调 (LoRA)**：仅需微调 0.01%-1% 的参数，显著降低计算成本和显存占用
- **多任务学习框架**：支持单任务和多任务分类，灵活的任务配置系统
- **模块化设计**：清晰的代码结构，易于扩展和定制
- **配置驱动**：基于 YAML 的灵活配置系统，支持配置继承
- **完整训练流程**：从数据加载、模型构建、训练到评估的端到端解决方案
- **生产级 API 服务**：内置 FastAPI 服务，支持模型热加载和在线推理

---

## 项目架构

```
AtomLoRA/
├── atomlora/                      # CLI 包
│   ├── __init__.py
│   └── cli.py                    # 命令行入口 (atomlora train/eval/predict/serve)
│
├── api/                          # FastAPI 应用层
│   ├── app.py                    # REST API 端点定义
│   ├── model_manager.py          # 模型管理和推理接口
│   ├── schemas.py                # 请求/响应数据模型
│   ├── settings.py               # 服务配置
│   └── util.py                   # 工具函数
│
├── configs/                      # 配置文件
│   ├── templates/                # 配置模板（基础配置）
│   │   ├── bert_lora_template.yaml
│   │   └── ernie_lora_template.yaml
│   ├── demo.yaml                 # Demo 配置
│   └── *.yaml                    # 具体实验配置
│
├── data/                         # 数据集存储
│   ├── raw/                      # 原始数据
│   │   └── demo/                 # Demo 数据集
│   └── processed/                # 处理后数据
│
├── docs/                         # 文档
│   └── config_v1.md              # 配置文件规范
│
├── src/                          # 核心代码
│   ├── config/parser.py          # 配置解析器
│   ├── data/data_processor.py    # 数据预处理
│   ├── model/
│   │   ├── model_factory.py      # 模型工厂（构建 TaskTextClassifier）
│   │   └── text_dataset.py       # 数据集和 DataLoader
│   ├── predict/predictor.py      # 推理预测
│   ├── scheduler/gpu_scheduler.py # GPU 调度
│   ├── trainer/
│   │   ├── train_engine.py       # 训练引擎
│   │   └── metric_manager.py     # 评估指标管理
│   └── utils/
│       ├── logger.py             # 日志系统
│       └── paths.py              # 输出路径管理
│
├── outputs/                      # 模型输出
│   └── {exp_id}/
│       ├── adapter/              # LoRA 适配器权重
│       ├── classifier/           # 分类头权重
│       ├── tokenizer/            # 分词器
│       ├── config.yaml           # 训练配置副本
│       └── metrics.json          # 最优评估指标
│
├── models/                       # 预训练模型（不纳入版本控制）
├── logs/                         # 训练日志
├── main.py                       # 训练/评估入口（兼容模式）
├── predict.py                    # 推理脚本（兼容模式）
├── evaluator.py                  # 评估脚本
├── pyproject.toml                # 包配置
├── requirements.txt              # 依赖包
└── .gitignore
```

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/balance-joe/AtomLoRA.git
cd AtomLoRA
pip install -e .
```

安装完成后，`atomlora` 命令即可在终端直接使用。

### 2. 准备数据

项目内置了 demo 数据，位于 `data/raw/demo/` 目录。数据格式为 JSONL，每行一个 JSON 对象：

```json
{"content": "逸夫楼停电了谁管管", "status": 0}
{"content": "图书馆开门时间是几点", "status": 1}
```

### 3. 训练模型

```bash
atomlora train --config configs/demo.yaml
```

训练完成后，模型产物保存在 `outputs/{exp_id}/` 目录下：

```
outputs/demo_single_cls/
├── adapter/              # LoRA 适配器权重
├── classifier/           # 分类头权重
├── tokenizer/            # 分词器
├── config.yaml           # 训练配置副本
└── metrics.json          # 最优评估指标
```

### 4. 评估模型

```bash
atomlora eval --config configs/demo.yaml
```

### 5. 单条预测

```bash
atomlora predict --config configs/demo.yaml --text "逸夫楼停电了谁管管"
```

输出 JSON 格式的预测结果。

### 6. 启动 API 服务

```bash
atomlora serve --config configs/demo.yaml --port 8000
```

服务启动后，可通过 HTTP 请求进行推理：

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"content": "逸夫楼停电了谁管管"}'
```

### 兼容模式（不安装包）

```bash
pip install -r requirements.txt
python main.py --config configs/demo.yaml                # 训练
python main.py --config configs/demo.yaml --eval          # 评估
python predict.py --config configs/demo.yaml --text "文本"  # 预测
```

### 自定义实验

1. 准备 JSONL 数据放在 `data/raw/` 目录下
2. 参考 `configs/demo.yaml` 或 `docs/config_v1.md` 编写配置文件
3. 运行 `atomlora train --config configs/my_experiment.yaml`

配置文件规范详见 [docs/config_v1.md](docs/config_v1.md)。

---

## 核心技术原理

### LoRA (Low-Rank Adaptation)

LoRA 通过在预训练模型的注意力层添加低秩适配矩阵，来高效地微调模型：

```
原始参数更新：ΔW = BA（秩为 r 的分解）

优势：
- 参数量：从 d×d 降低到 (d+d)×r（r<<d）
- BERT-Base: 110M 参数 → 仅需训练 30K-300K 参数
- 显存占用：减少 80% 以上
- 训练速度：快 3-5 倍
```

### 多任务学习

支持两种任务配置：

| 任务类型 | 说明 | 应用场景 |
|---------|------|---------|
| **single_cls** | 单个分类任务 | 文本分类、情感分析 |
| **multi_cls** | 多个独立分类任务 | 多标签分类、多任务学习 |

### 差异化学习率

针对不同参数组设置不同学习率，在保留预训练知识的同时快速适配：

```yaml
optimizer:
  groups:
    bert: 2e-6        # BERT 层：极低学习率，保留预训练知识
    lora: 1.5e-4      # LoRA 层：中等学习率，快速适配
    classifier: 5e-4  # 分类头：较高学习率，快速学习任务
```

---

## 配置系统

AtomLoRA 使用 YAML 配置文件驱动所有行为。配置支持继承，子配置覆盖父配置的同名字段。

### 最小配置示例

```yaml
exp_id: "my_experiment"
task_type: "single_cls"

data:
  train_path: "./data/raw/my_data/train.jsonl"
  dev_path: "./data/raw/my_data/dev.jsonl"
  max_len: 256
  text_col: "content"
  label_col:
    status: "status"
  label_subset:
    status: [0, 1]
  label_mapping:
    status: {0: "正确", 1: "错误"}

model:
  arch: "bert-base-chinese"
  path: "./models/bert-base-chinese"
  lora:
    rank: 8
    alpha: 16
    dropout: 0.05
    target_modules: ["self.query", "self.key", "self.value"]
    bias: "none"

train:
  num_epochs: 14
  batch_size: 20
  optimizer:
    type: "AdamW"
    groups:
      bert: 2e-6
      lora: 1.5e-4
      classifier: 5e-4
```

### 配置继承

```yaml
# configs/my_experiment.yaml
base_config: "templates/bert_lora_template.yaml"
exp_id: "my_experiment"
data:
  train_path: "./data/raw/my_data/train.jsonl"
  dev_path: "./data/raw/my_data/dev.jsonl"
```

父配置值先加载，子配置值覆盖同名字段。`base_config` 可以是相对路径（相对于 `configs/` 目录）或绝对路径。

完整字段说明见 [docs/config_v1.md](docs/config_v1.md)。

---

## 项目亮点

### 参数高效

- LoRA 微调仅需 0.1%-1% 的参数量
- BERT-Large 微调仅需 0.3M 参数 vs 完整微调 110M

### 快速迭代

- 灵活的 YAML 配置系统，支持配置继承
- 自动日志记录和结果跟踪
- 支持早停（Early Stopping）和多种学习率调度策略

### 生产就绪

- 完整的 FastAPI API 服务
- 模型热加载和版本管理
- 详细的日志和错误处理
- 支持 GPU 自动调度

### 易于使用

- 模块化的代码设计
- `pip install -e .` 一键安装
- 支持 CPU/GPU 自动检测
- Windows / Linux / macOS 兼容

---

## 开发者指南

### 扩展自定义分类器

修改 `src/model/model_factory.py` 中的 `TaskTextClassifier`：

```python
class TaskTextClassifier(nn.Module):
    def _build_classifiers(self):
        hidden_size = self.bert.config.hidden_size
        return MyCustomClassifier(hidden_size, num_classes)
```

### 添加新的优化器

在 `src/trainer/train_engine.py` 中修改 `_build_optimizer()` 方法。

### 自定义数据预处理

编辑 `src/data/data_processor.py` 中的 `load_dataset()` 函数。

### 自定义评估指标

在 `src/trainer/metric_manager.py` 中添加新的指标计算逻辑。

---

## License

该项目采用 **GNU General Public License v2.0** 开源协议。

---


## 贡献

欢迎提交 Issue 和 Pull Request！

---

## FAQ

**Q: 为什么选择 LoRA 而不是全量微调？**
A: LoRA 可以在降低 99% 参数量的情况下保持接近的效果，特别适合在资源受限的环境下快速迭代。

**Q: 能否用于生产环境？**
A: 可以。项目提供了 FastAPI API 服务，支持模型热加载和并发推理。使用 `atomlora serve` 启动。

**Q: 支持哪些预训练模型？**
A: 支持 HuggingFace Hub 上的任意 Transformer 模型（BERT、ERNIE、RoBERTa 等）。在配置中设置 `model.arch` 和 `model.path` 即可切换。

**Q: 如何迁移 LoRA 权重到其他模型？**
A: LoRA 权重是通用的。将 `outputs/{exp_id}/adapter/` 目录加载到相同架构的其他模型即可。分类头权重在 `outputs/{exp_id}/classifier/classifiers.pt`。

**Q: 训练时显存不足怎么办？**
A: 尝试减小 `batch_size`、增大 `gradient_accumulation_steps`、降低 `max_len`，或减小 LoRA `rank`。

**Q: 如何在 Windows 上使用？**
A: AtomLoRA 已适配 Windows 环境，路径解析和日志编码均已处理。直接 `pip install -e .` 即可。

---

如有问题或建议，欢迎提交 Issue 或联系开发者。
