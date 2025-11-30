多模型LoRA分类训练系统架构文档（实验原子化·工业级落地）
 
一、核心设计理念
 
1. 实验原子化：1个实验 = 1份配置 + 1组独立产物，所有关联资源（配置、权重、指标、标签映射）打包存储，确保“实验可复现、产物可追溯”。
2. 配置继承化：支持配置文件父子继承，子实验仅需覆盖修改的参数，解决多轮迭代的配置冗余问题。
3. 追溯自动化：内置实验索引库，自动记录所有实验的参数、指标、依赖版本，支持按条件快速检索与对比。
4. 资源智能化：GPU调度集成显存动态检测与优先级机制，避免多模型并行冲突；数据缓存兼顾效率与一致性，自动规避缓存命中错误。
5. 适配灵活性：支持全局标签与子集标签混用，LoRA配置自动适配多模型变体；单/双任务无缝兼容，无需重构即可扩展双标签（如风险等级+是否采纳）分类需求。
 
二、项目目录结构（极简实用，无冗余层级）
 
plaintext
  
lora-cls-system/
├── configs/                      # 实验配置中心（含模板与具体实验）
│   ├── templates/                # 基础模板（无需修改，供继承）
│   │   ├── bert_lora_template.yaml
│   │   └── ernie_lora_template.yaml
│   ├── exp_v1_bert_baseline.yaml # 单标签实验（可继承模板）
│   ├── exp_v4_double_label.yaml  # 双标签实验（风险等级+是否采纳）
│   └── experiment_index.json     # 实验全局索引（自动生成/更新）
│
├── data/                         # 数据存储（按类型分区，无冗余）
│   ├── raw/                      # 原始数据（只读，不修改）
│   │   ├── train_5w.csv          # 单标签数据（text, label）
│   │   └── train_double_label.csv# 双标签数据（text, risk_level, is_adopt）
│   ├── processed/                # 预处理缓存（按指纹自动管理）
│   └── global_label_map.json     # 全局标签标准（唯一真值源）
│
├── outputs/                      # 实验产物仓库（按实验ID隔离）
│   ├── exp_v1_bert_baseline/     # 单标签实验产物
│   │   ├── lora_adapter/         # LoRA权重（仅必要文件）
│   │   ├── config_copy.yaml      # 实验配置快照（冻结，不可改）
│   │   ├── label_map.json        # 该实验使用的标签映射（副本）
│   │   ├── metrics.json          # 训练指标（含每轮详细数据）
│   │   └── env_info.json         # 环境依赖快照（torch/peft版本等）
│   ├── exp_v4_double_label/      # 双标签实验产物（结构一致，内容适配）
│   │   ├── lora_adapter/
│   │   ├── config_copy.yaml
│   │   ├── label_map.json        # 双标签映射（含两个任务的标签ID）
│   │   ├── metrics.json          # 双任务指标（分别记录+平均指标）
│   │   └── env_info.json
│   └── ...
│
├── src/                          # 核心源码（职责单一，无过度抽象）
│   ├── config/                   # 配置解析与继承模块
│   │   ├── parser.py             # 支持YAML继承+单/双任务适配
│   │   └── index_manager.py      # 实验索引库管理（含双任务信息记录）
│   │
│   ├── data/                     # 数据处理模块
│   │   ├── data_processor.py     # 单/双标签加载+预处理+标签校验
│   │   └── cache_manager.py      # 指纹缓存生成（区分单/双任务）
│   │
│   ├── model/                    # 模型与LoRA模块
│   │   ├── model_factory.py      # 模型实例化（自动适配单/双分类头）
│   │   └── lora_adapter.py       # LoRA配置+自动探测逻辑
│   │
│   ├── trainer/                  # 训练与评估模块
│   │   ├── train_engine.py       # 训练循环（兼容单/双任务损失计算）
│   │   └── metric_manager.py     # 单/双任务指标计算与存储
│   │
│   ├── scheduler/                # 资源调度模块
│   │   └── gpu_scheduler.py      # 显存检测+GPU分配+优先级调度
│   │
│   └── utils/                    # 工具函数
│       ├── logger.py             # 统一日志（绑定实验ID+任务类型）
│       └── compare_tool.py       # 实验对比工具（支持双任务指标筛选）
│
├── main.py                       # 唯一入口（命令行驱动，单/双任务通用）
└── requirements.txt              # 依赖清单（锁定版本，确保兼容性）
 
 
三、核心模块详细设计
 
1. 配置层：继承式配置+全局索引（兼容单/双任务）
 
核心逻辑：
 
• 支持配置继承：子实验通过 base_config 字段继承模板或其他实验，仅覆盖需修改的参数；

• 新增 task_type 参数：通过 single_cls （默认）/ double_cls 声明任务类型，双任务需配置标签子集与损失权重；

• 全局实验索引： experiment_index.json 自动记录单/双任务的关键信息，支持按任务类型、指标筛选。

 
配置文件示例（双标签任务）：
 
yaml
  
# configs/exp_v4_double_label.yaml（风险等级+是否采纳）
exp_id: "exp_v4_double_label"
description: "BERT+LoRA双标签任务（风险等级3分类+是否采纳2分类）"
base_config: "templates/bert_lora_template.yaml"
task_type: "double_cls"  # 声明双标签任务
data:
  raw_data_path: "./data/raw/train_double_label.csv"
  label_subset:  # 双标签各自的子集（基于全局标签库）
    risk_level: ["低风险", "中风险", "高风险"]
    is_adopt: ["采纳", "不采纳"]
  max_len: 128
  data_process_version: "v2"  # 区分双标签预处理规则

model:
  lora:
    rank: 16
    alpha: 32

train:
  batch_size: 32
  lr: 2e-5
  epochs: 8
  loss_weight: [0.6, 0.4]  # 双任务损失权重（可按需调整）
  save_strategy: "best_avg_f1"  # 基于双任务平均F1保存最优模型

resources:
  priority: "medium"
 
 
实验索引示例（含双标签任务）：
 
json
  
{
  "exp_v1_bert_baseline": {
    "create_time": "2024-05-20 14:30:00",
    "task_type": "single_cls",
    "base_config": "templates/bert_lora_template.yaml",
    "key_params": {
      "model": "bert-base-chinese",
      "lora_rank": 16,
      "lr": 2e-5,
      "batch_size": 32
    },
    "best_metrics": {
      "accuracy": 0.92,
      "f1": 0.91
    },
    "output_path": "./outputs/exp_v1_bert_baseline",
    "device_used": "GPU-0",
    "env_version": "torch==2.1.0, peft==0.8.2"
  },
  "exp_v4_double_label": {
    "create_time": "2024-05-22 10:15:00",
    "task_type": "multi_cls",
    "base_config": "templates/bert_lora_template.yaml",
    "key_params": {
      "model": "bert-base-chinese",
      "lora_rank": 16,
      "lr": 2e-5,
      "loss_weight": [0.6, 0.4]
    },
    "best_metrics": {
      "risk_level_accuracy": 0.93,
      "risk_level_f1": 0.92,
      "is_adopt_accuracy": 0.95,
      "is_adopt_f1": 0.94,
      "avg_f1": 0.93
    },
    "output_path": "./outputs/exp_v4_double_label",
    "device_used": "GPU-1",
    "env_version": "torch==2.1.0, peft==0.8.2"
  }
}
 
 
2. 数据层：指纹缓存+标签一致性（适配双标签）
 
核心逻辑：
 
• 完整指纹缓存：缓存键 = md5(raw_data_path + tokenizer_name + max_len + data_process_version + task_type)，区分单/双任务缓存，避免冲突；

• 标签双模式：单任务直接使用全局标签或子集标签；双任务自动生成两个独立的标签映射（基于 label_subset ），且强制校验子集标签属于全局标签；

• 双标签数据加载：自动读取数据中对应的两个标签列，生成适配模型输入的标签格式。

 
关键代码逻辑（简化）：
 
python
  
# src/data/cache_manager.py
def get_cache_key(config):
    # 整合所有影响预处理结果的因素，生成唯一指纹（含任务类型）
    factors = [
        config["data"]["raw_data_path"],
        config["model"]["arch"],  # 关联tokenizer
        str(config["data"]["max_len"]),
        config["data"]["data_process_version"],
        config.get("task_type", "single_cls")  # 新增：区分单/双标签
    ]
    return hashlib.md5("|".join(factors).encode()).hexdigest()

# src/data/data_processor.py
def load_label_map(config):
    global_label = json.load(open("./data/global_label_map.json"))
    task_type = config.get("task_type", "single_cls")
    
    if task_type == "single_cls":
        # 单标签逻辑：全局标签或子集标签
        if config["data"]["label_subset"] is None:
            return global_label
        subset = config["data"]["label_subset"]
        if not set(subset).issubset(set(global_label.keys())):
            raise ValueError("标签子集必须是全局标签的子集")
        return {label: global_label[label] for label in subset}
    
    # 双标签逻辑：生成两个任务的独立标签映射
    double_label_map = {}
    for task_name, subset in config["data"]["label_subset"].items():
        if not set(subset).issubset(set(global_label.keys())):
            raise ValueError(f"{task_name}的标签子集必须是全局标签的子集")
        double_label_map[task_name] = {label: global_label[label] for label in subset}
    return double_label_map

def load_data(config):
    df = pd.read_csv(config["data"]["raw_data_path"])
    text = df["text"].tolist()
    task_type = config.get("task_type", "single_cls")
    label_map = load_label_map(config)
    
    if task_type == "single_cls":
        labels = df["label"].map(label_map).tolist()
        return text, labels
    
    # 双标签逻辑：分别获取两个标签列的ID
    risk_labels = df["risk_level"].map(label_map["risk_level"]).tolist()
    adopt_labels = df["is_adopt"].map(label_map["is_adopt"]).tolist()
    return text, (risk_labels, adopt_labels)
 
 
3. 模型层：LoRA自动适配+动态分类头（支持单/双任务）
 
核心逻辑：
 
• 共享编码器+动态分类头：单任务使用1个分类头，双任务使用2个独立分类头（共享Transformer+LoRA编码器），显存占用仅增加5%以内；

• LoRA策略库+自动探测：预设BERT/ERNIE的LoRA默认配置，支持自动识别attention层，适配自定义模型变体；

• 参数极简暴露：用户仅需配置 rank 和 alpha ，双任务额外配置 loss_weight ，无需关注底层分类头构建。

 
关键代码逻辑（简化）：
 
python
  
# src/model/model_factory.py
class TaskTextClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.task_type = config.get("task_type", "single_cls")
        # 加载基础模型+LoRA（复用原有逻辑）
        self.base_model, self.tokenizer = self._load_base_model_with_lora()
        # 动态创建分类头（单/双任务适配）
        self.classifiers = self._build_classifiers()

    def _load_base_model_with_lora(self):
        # 加载Transformer编码器（不含分类头）
        model_arch = self.config["model"]["arch"]
        if "bert" in model_arch:
            base_model = BertModel.from_pretrained(model_arch)
        elif "ernie" in model_arch:
            base_model = ErnieModel.from_pretrained(model_arch)
        # 注入LoRA（复用lora_adapter.py逻辑）
        lora_config = get_lora_config(base_model, self.config)
        base_model = get_peft_model(base_model, lora_config)
        tokenizer = AutoTokenizer.from_pretrained(model_arch)
        return base_model, tokenizer

    def _build_classifiers(self):
        hidden_dim = self.base_model.config.hidden_size
        label_map = load_label_map(self.config)
        
        if self.task_type == "single_cls":
            # 单分类头
            num_labels = len(label_map)
            return nn.Linear(hidden_dim, num_labels)
        
        # 双分类头（独立Linear层，ModuleDict管理）
        classifiers = nn.ModuleDict()
        for task_name, task_label_map in label_map.items():
            num_labels = len(task_label_map)
            classifiers[task_name] = nn.Linear(hidden_dim, num_labels)
        return classifiers

    def forward(self, input_ids, attention_mask, labels=None):
        # 共享编码器：获取[CLS] token隐藏态
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
        
        if self.task_type == "single_cls":
            logits = self.classifiers(cls_feat)
            loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else torch.tensor(0.0)
            return logits, loss
        
        # 双任务：分别计算两个分类头的logits与加权损失
        logits = {}
        total_loss = 0.0
        loss_weights = self.config["train"].get("loss_weight", [0.5, 0.5])
        for i, (task_name, classifier) in enumerate(self.classifiers.items()):
            task_logits = classifier(cls_feat)
            logits[task_name] = task_logits
            if labels is not None:
                task_loss = nn.CrossEntropyLoss()(task_logits, labels[i])
                total_loss += task_loss * loss_weights[i]
        return logits, total_loss if labels is not None else torch.tensor(0.0)

# src/model/lora_adapter.py（复用原有逻辑，task_type自动适配）
LORA_DEFAULT = {
    "bert": {"target_modules": ["query", "value"]},
    "ernie": {"target_modules": ["query", "key", "value"]}
}

def get_lora_config(model, config):
    model_arch = config["model"]["arch"].split("-")[0]
    # 优先使用默认配置，无默认则自动探测
    if model_arch in LORA_DEFAULT:
        lora_config = LORA_DEFAULT[model_arch]
    else:
        lora_config = {"target_modules": auto_detect_attention_layers(model)}
    # 合并用户自定义参数，任务类型统一为序列分类
    lora_config.update({
        "r": config["model"]["lora"]["rank"],
        "lora_alpha": config["model"]["lora"]["alpha"],
        "lora_dropout": 0.05,
        "task_type": "SEQ_CLS"
    })
    return LoraConfig(**lora_config)
 
 
4. 训练层：产物快照+双任务指标（兼容单/双任务）
 
核心逻辑：
 
• 损失计算：单任务使用交叉熵损失，双任务使用加权交叉熵损失（权重由配置指定），训练循环无需修改；

• 指标计算：单任务输出accuracy、F1；双任务分别输出两个子任务的accuracy、F1，同时计算平均F1（用于保存最优模型）；

• 产物快照：双任务的 label_map.json 存储两个子任务的标签映射， metrics.json 记录每轮双任务指标，结构统一且清晰。

 
双任务指标示例（metrics.json）：
 
json
  
{
  "epoch_1": {
    "risk_level_accuracy": 0.88,
    "risk_level_f1": 0.86,
    "is_adopt_accuracy": 0.90,
    "is_adopt_f1": 0.89,
    "avg_f1": 0.875
  },
  "epoch_5": {
    "risk_level_accuracy": 0.93,
    "risk_level_f1": 0.92,
    "is_adopt_accuracy": 0.95,
    "is_adopt_f1": 0.94,
    "avg_f1": 0.93
  },
  "best": {
    "epoch": 5,
    "risk_level_accuracy": 0.93,
    "risk_level_f1": 0.92,
    "is_adopt_accuracy": 0.95,
    "is_adopt_f1": 0.94,
    "avg_f1": 0.93
  }
}
 
 
5. 调度层：GPU智能调度（单/双任务通用）
 
核心逻辑：
 
• 显存动态检测：自动预估单/双任务的显存需求（基于batch_size、max_len、模型架构），分配空闲GPU，避免OOM；

• 优先级调度：高优先级实验可暂停低优先级实验（自动保存checkpoint），释放资源后恢复；

• 进程隔离：每个实验以独立子进程运行，单/双任务并行无冲突，产物相互隔离。

 
调度命令示例（单/双任务通用）：
 
bash
  
# 运行单标签实验
python main.py run --config configs/exp_v1_bert_baseline.yaml

# 运行双标签实验
python main.py run --config configs/exp_v4_double_label.yaml

# 多实验并行（含单/双标签）
python main.py run-parallel --configs configs/exp_v1*.yaml,configs/exp_v4*.yaml --gpu-pool 0,1

# 筛选双标签实验并按平均F1排序
python main.py compare --filter "task_type=multi_cls" --sort-by "avg_f1"

# 回滚到最优双标签版本
python main.py rollback --exp-id exp_v4_double_label
 
 
四、架构核心优势（直击痛点，含双任务适配）
 
核心痛点 架构解决方案 
配置冗余、迭代繁琐 配置继承+实验索引，子实验仅需修改差异参数 
多模型并行GPU冲突 显存动态检测+优先级调度，自动分配空闲资源 
实验追溯、回滚困难 产物全量快照+全局索引，支持按指标/参数/任务类型检索 
标签混乱、数据预处理错误 全局标签校验+完整指纹缓存，避免映射错乱与缓存命中错误 
LoRA适配性不足 默认策略库+自动探测，兼容标准/自定义模型 
实验对比决策低效 内置对比工具，支持参数/指标/任务类型快速筛选与排序 
单/双任务切换成本高 配置化声明任务类型，动态适配分类头与指标计算，无需重构 
 
五、落地保障
 
1. 复现性：配置、数据、环境、权重全快照，单/双任务实验均可一键回滚复现，无额外依赖。
2. 扩展性：新增模型（如RoBERTa）仅需补充LoRA默认配置；新增标签任务（如三标签）仅需扩展 label_subset ，模型自动新增分类头，无需修改核心逻辑。
3. 效率：数据缓存秒级加载，双任务显存占用增量低（≤5%），5万条数据场景下训练耗时与单任务基本持平；多实验并行无冲突，高频迭代（100+实验）仍保持清晰追溯。
4. 稳定性：GPU调度避免OOM，标签校验避免推理错误，进程隔离避免资源污染，单/双任务混跑时系统稳定性不受影响。
