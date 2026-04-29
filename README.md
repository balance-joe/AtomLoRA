# AtomLoRA

> 面向中文文本分类/审核场景的轻量级 LoRA 微调框架。
> 专注小数据快速微调 + 多任务分类 + 一键部署 API。
> 仅需微调 0.1%-1% 的参数，单张 GPU 即可完成训练和推理。

## 项目概述

**AtomLoRA** 基于 LoRA (Low-Rank Adaptation) 技术，在 BERT/ERNIE 等预训练模型上做参数高效微调。与 HuggingFace Trainer 的区别：AtomLoRA 专注于**中文文本分类场景**，内置多任务学习、差异化学习率、配置继承、生产级 API 服务，开箱即用。

**适用场景**：文本审核、误报检测、风险分级、情感分析等中文分类任务。

如果你更关心“能否稳定跑通训练、评估、预测、部署”，可以直接看：

1. 快速开始
2. 运行约束
3. 模型来源与部署前提
4. 配置系统

### 核心特性

- **参数高效**：LoRA 微调仅需 0.1% 参数，BERT-Base 从 110M 降到 300K 可训练参数
- **多任务分类**：单任务 / 多任务统一框架，配置驱动
- **配置驱动**：YAML 配置 + 继承机制，一个模板跑多个实验
- **端到端**：训练 → 评估 → 预测 → API 服务，全流程覆盖
- **单 GPU 友好**：4GB 显存即可训练，支持 CPU 回退

---

## 项目架构

```
AtomLoRA/
├── atomlora/                      # CLI 包
│   ├── __init__.py
│   └── cli.py                    # 命令行入口 (atomlora train/eval/predict/serve/split/doctor-data)
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
│   ├── data/
│   │   ├── io.py                 # JSONL 读写工具
│   │   ├── splitter.py           # 数据切分（atomlora split）
│   │   ├── doctor.py             # 数据质量诊断（atomlora doctor-data）
│   │   └── data_processor.py     # 训练时数据加载与 tokenize
│   ├── model/
│   │   ├── model_factory.py      # 模型工厂（构建 TaskTextClassifier）
│   │   └── text_dataset.py       # 数据集和 DataLoader
│   ├── predict/predictor.py      # 推理预测
│   ├── scheduler/gpu_scheduler.py # 预留调度扩展
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

## 快速开始（3 分钟）

```bash
# 1. 克隆并安装
git clone https://github.com/balance-joe/AtomLoRA.git
cd AtomLoRA
bash install.sh cu121        # CUDA 12.1（CPU 用 bash install.sh cpu）
```

```bash
# 2. 训练 demo 模型（首次运行自动下载 BERT）
atomlora train --config configs/demo.yaml

# 3. 评估
atomlora eval --config configs/demo.yaml

# 4. 单条预测
atomlora predict --config configs/demo.yaml --text "逸夫楼停电了谁管管"

# 5. 启动 API 服务
atomlora serve --config configs/demo.yaml --port 8000
```

训练产物保存在 `outputs/{exp_id}/` 目录：

```
outputs/demo_single_cls/
├── adapter/              # LoRA 适配器权重
├── classifier/           # 分类头权重
├── tokenizer/            # 分词器
├── config.yaml           # 训练配置副本
└── metrics.json          # 最优评估指标
```

### 训练产物边界

`outputs/{exp_id}/` 默认包含：

- LoRA adapter
- 分类头权重
- tokenizer
- 训练配置副本
- 最优评估指标

`outputs/{exp_id}/` 默认**不包含**：

- HuggingFace 基础模型完整权重
- 原始训练集 / 验证集 / 测试集
- Python 运行环境和依赖

这也是为什么部署阶段仍然需要 `model.path` 可用，详见下方“模型来源与部署前提”。

### 快速访问最新实验

训练完成后，`outputs/latest/` 始终指向最新实验：

```bash
# 查看最新指标
cat outputs/latest/metrics.json

# 查看实验信息（exp_id、时间、config 路径）
cat outputs/latest/info.json

# 用最新模型预测
atomlora predict --config latest --text "测试文本"

# 用最新模型启动服务
atomlora serve --config latest
```

> 如果系统不支持符号链接，`outputs/latest.txt` 会记录最新实验路径，`--config latest` 同样会自动读取它。

### 安装模式说明

| 模式 | 命令 | 说明 |
|------|------|------|
| CPU | `bash install.sh cpu` | 仅 CPU 推理/训练 |
| CUDA 11.8 | `bash install.sh cu118` | GPU（CUDA 11.8） |
| CUDA 12.1 | `bash install.sh cu121` | GPU（CUDA 12.1） |
| CUDA 12.4 | `bash install.sh cu124` | GPU（CUDA 12.4） |
| Windows | `install.bat cu121` | 同上，用 bat 脚本 |
| 手动 | `pip install -e .` | 需自行安装 PyTorch |

不确定 CUDA 版本？运行 `nvidia-smi` 查看。

### 运行约束

| 约束 | 说明 |
|------|------|
| GPU 数量 | 单卡（多卡暂不支持） |
| API worker | 单 worker（多 worker 会 OOM） |
| API reload | GPU 模式不支持 `--reload` |
| 数据格式 | JSONL，每行一个 JSON 对象 |
| 配置格式 | YAML，支持继承 |
| Python | >= 3.9 |

### 模型来源与部署前提

AtomLoRA 保存的是：

- LoRA adapter
- 分类头权重
- tokenizer
- 训练配置副本

默认**不会**把 HuggingFace 基础模型完整复制到 `outputs/{exp_id}/` 中。

这意味着评估、预测、启动 API 服务时，仍然需要能够加载 `model.path` 对应的基础模型：

- 如果 `model.path` 是本地目录，例如 `./models/bert-base-chinese`，则部署机器需要有这个目录
- 如果 `model.path` 是 HuggingFace repo id，例如 `bert-base-chinese`，则部署机器需要：
  - 已经有本地缓存
  - 或仍然可以访问 HuggingFace

最稳妥的做法是：

- 训练前就把基础模型下载到本地目录
- 在配置中把 `model.path` 写成本地路径
- 训练、评估、预测、部署都统一使用该本地路径

### 离线部署示例

如果你希望训练后在**无外网**环境中稳定执行 `eval / predict / serve`，建议从一开始就使用本地模型目录：

```yaml
model:
  arch: "bert-base-chinese"
  path: "./models/bert-base-chinese"
```

推荐流程：

1. 先把基础模型下载到 `./models/bert-base-chinese`
2. 训练时直接使用这一路径
3. 评估、预测、部署继续使用同一份配置或训练产物中的 `config.yaml`

典型命令：

```bash
atomlora train --config configs/my_experiment.yaml
atomlora eval --config configs/my_experiment.yaml
atomlora predict --config configs/my_experiment.yaml --text "测试文本"
atomlora serve --config configs/my_experiment.yaml
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

## 数据准备：split 与 doctor-data

在训练之前，AtomLoRA 提供两个命令帮你完成 **原始数据 → 训练数据 → 质量诊断** 的流程。

### 完整流程

```bash
# 1. 将原始 JSONL 切分为 train / dev / test（自动分层采样）
atomlora split --config configs/demo.yaml --output data/raw/demo_split

# 2. 检查数据质量（标签分布、空文本、重复、类别缺失等）
atomlora doctor-data --config configs/demo.yaml

# 3. 训练
atomlora train --config data/raw/demo_split/config.generated.yaml
```

### atomlora split

将一个 JSONL 文件切分为 train / dev / test 三份，支持分层采样（stratify）保持标签比例一致。

**极简模式**（推荐）：从配置文件读取字段信息，无需手动指定：

```bash
atomlora split --config configs/demo.yaml --output data/raw/demo_split
```

自动读取配置中的 `text_col`、`label_col`、`label_mapping`，并以 `train_path` 作为输入源。

**完整模式**：手动指定所有参数：

```bash
atomlora split \
  --input data/raw/demo/demo_raw.jsonl \
  --output data/raw/demo_split \
  --text-col content \
  --label-col status \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --stratify
```

**参数说明**：

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | 否 | - | YAML 配置路径（读取 text_col、label_col 等） |
| `--input` | 否* | 来自 config | 输入 JSONL 文件路径 |
| `--output` | 否 | 与 input 同目录 | 输出目录 |
| `--text-col` | 否* | 来自 config | 文本字段名 |
| `--label-col` | 否* | 来自 config | 标签字段名 |
| `--train-ratio` | 否 | 0.8 | 训练集比例 |
| `--dev-ratio` | 否 | 0.1 | 验证集比例 |
| `--test-ratio` | 否 | 0.1 | 测试集比例 |
| `--seed` | 否 | 42 | 随机种子 |
| `--stratify` | 否 | True | 是否按标签分层采样 |
| `--no-stratify` | 否 | - | 关闭分层采样 |

> *使用 `--config` 时，`--input`、`--text-col`、`--label-col` 可省略。

**输出文件**：

```
data/raw/demo_split/
├── demo_train.jsonl          # 训练集
├── demo_dev.jsonl            # 验证集
├── demo_test.jsonl           # 测试集
├── split_report.json         # 切分报告（样本数、标签分布）
└── config.generated.yaml     # 自动生成的配置文件，可直接用于训练
```

`config.generated.yaml` 包含 `train_path`、`dev_path`、`test_path`、`text_col`、`label_col`、`label_mapping`，可直接传给 `atomlora train`。

### atomlora doctor-data

对 train / dev / test 数据集做全面质量检查，输出 JSON + Markdown 报告。

```bash
atomlora doctor-data --config configs/demo.yaml

# 严格模式：有 ERROR 级别问题时 exit 1（适合 CI）
atomlora doctor-data --config configs/demo.yaml --strict
```

**检查项与严重级别**：

| 检查项 | 级别 | 说明 |
|--------|------|------|
| `sample_counts` | INFO | 各 split 样本数 |
| `label_distribution` | INFO | 各 split 标签分布 |
| `missing_classes` | ERROR/WARNING | train 缺类别为 ERROR，dev/test 为 WARNING |
| `empty_text` | ERROR | 空文本记录 |
| `empty_label` | ERROR | 空标签记录 |
| `unknown_labels` | WARNING | 标签值不在 label_mapping 中 |
| `duplicate_content` | WARNING | 重复文本（含 Top 5 样例） |
| `low_count_classes` | WARNING | 某类样本数低于阈值（默认 10） |
| `label_ratio_drift` | WARNING | train/dev/test 标签比例偏移 > 10% |
| `text_length_stats` | INFO | 文本长度 min/avg/p50/p95/max |

**输出文件**（写入 `outputs/{exp_id}/`）：

```
outputs/demo_single_cls/
├── dataset_report.json       # 结构化报告
└── dataset_report.md         # Markdown 报告（含 Top 重复样本）
```

**输出示例**：

```
==================================================
[WARNING] 数据诊断完成
  WARNING (2):
    [duplicate_content] 存在重复文本 (within: 44, cross: 64)
    [low_count_classes] 部分类别样本数低于阈值 10
  INFO (8)
报告: outputs/demo_single_cls/dataset_report.json
==================================================
```

### 推荐工作流

```
原始数据 (raw.jsonl)
    │
    ▼
  atomlora split          ← 切分 + 自动生成 config
    │
    ▼
  atomlora doctor-data    ← 质量检查
    │
    ▼
  atomlora train          ← 训练
    │
    ▼
  atomlora eval / predict / serve
```

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
训练完成后，`eval / predict / serve` 默认优先读取 `outputs/{exp_id}/config.yaml` 作为模型语义真相，而不是盲信当前外部配置文件。

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

## 工程特点

- 配置驱动：训练、评估、预测、服务统一由 YAML 驱动，支持配置继承
- 参数高效：默认围绕 LoRA 微调设计，适合中文分类场景的小数据快速迭代
- 产物清晰：训练输出集中在 `outputs/{exp_id}/`，便于追踪 adapter、分类头、tokenizer 和指标
- 部署可控：支持 API 服务、显式设备选择，以及 `outputs/{exp_id}/config.yaml` 作为运行期语义真相

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

**Q: `outputs/{exp_id}` 能不能直接理解成完整离线模型包？**
A: 不能默认这样理解。它默认包含 adapter、分类头、tokenizer 和配置副本，但不包含完整基础模型。如果 `model.path` 仍然指向 HuggingFace repo id，那么推理阶段仍依赖本地缓存或网络。要做稳定离线部署，建议从训练开始就把 `model.path` 指向本地模型目录。

**Q: 为什么我的 adapter 文件比预期大？**
A: 如果训练时给 tokenizer 新增了 special tokens，模型会执行 embedding resize。此时 PEFT 会连同 embedding 相关层一起保存到 adapter 中，所以 `adapter_model.safetensors` 体积会明显增大。这是预期行为，迁移时需要保持 tokenizer 与词表一致。

**Q: 为什么训练后还能在预测时看到 HuggingFace 加载日志？**
A: 因为 AtomLoRA 默认保存的是 adapter、分类头和 tokenizer，不会把基础模型完整复制到 `outputs/{exp_id}/`。如果 `model.path` 配的是 `bert-base-chinese` 这类 HuggingFace repo id，预测/评估/服务阶段仍会按该路径加载基础模型。若需要稳定离线部署，建议先把基础模型下载到本地目录，并在配置中把 `model.path` 写成本地路径。

**Q: 训练时显存不足怎么办？**
A: 尝试减小 `batch_size`、增大 `gradient_accumulation_steps`、降低 `max_len`，或减小 LoRA `rank`。

**Q: 如何在 Windows 上使用？**
A: AtomLoRA 已适配 Windows 环境，路径解析和日志编码均已处理。直接 `pip install -e .` 即可。

---

如有问题或建议，欢迎提交 Issue 或联系开发者。
