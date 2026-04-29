# AtomLoRA Configuration Specification (v1)

All behavior is driven by a single YAML config file. Configs support **inheritance** via `base_config`.
For trained experiments, `eval / predict / serve` prefer `outputs/{exp_id}/config.yaml` as the source of truth for model semantics.

## Quick Example

```yaml
exp_id: "my_experiment"
task_type: "single_cls"
base_config: "templates/bert_lora_template.yaml"  # optional: inherit defaults

data:
  train_path: "./data/raw/demo/demo_train.jsonl"
  dev_path: "./data/raw/demo/demo_dev.jsonl"
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
  path: "bert-base-chinese"    # HuggingFace repo id or local path
  dropout: 0.15
  freeze_bert: True
  lora:
    rank: 8
    alpha: 16
    dropout: 0.05
    target_modules: ["self.query", "self.key", "self.value"]
    bias: "none"

train:
  num_epochs: 14
  batch_size: 20
  gradient_accumulation_steps: 3
  warmup_ratio: 0.1
  optimizer:
    type: "AdamW"
    groups:
      bert: 2e-6
      lora: 1.5e-4
      classifier: 5e-4
```

---

## Field Reference

### Top-level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `exp_id` | string | yes | Experiment identifier. Used for output directory name and log tagging. |
| `task_type` | string | yes | `"single_cls"` or `"multi_cls"`. |
| `base_config` | string | no | Path to a parent config. Child values override parent values. |
| `description` | string | no | Human-readable description (for documentation). |

### data

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `train_path` | string | yes | Path to training JSONL file. |
| `dev_path` | string | yes | Path to validation JSONL file. |
| `test_path` | string | no | Path to test JSONL file. |
| `max_len` | int | yes | Max token length. Texts are truncated/padded to this length. |
| `text_col` | string | yes | JSON field name containing the input text. |
| `label_col` | dict | yes | Maps task name to the JSON field name containing the label. |
| `label_subset` | dict | yes | Maps task name to the list of valid label values. |
| `label_mapping` | dict | no | Maps task name to a `{int_value: "display_name"}` dict. Auto-generated from `label_subset` if omitted. |

#### label_col / label_mapping semantics

**single_cls** (single task):
```yaml
label_col:
  status: "status"          # task "status" reads from JSON field "status"
label_mapping:
  status: {0: "正确", 1: "错误"}  # label 0 displays as "正确"
```

**multi_cls** (multiple independent tasks):
```yaml
label_col:
  misreport: "is_misreport_label"
  risk: "risk_level"
label_mapping:
  misreport: {0: "非误报", 1: "误报"}
  risk: {0: "低", 1: "中", 2: "高"}
```

### model

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `arch` | string | yes | HuggingFace model architecture name (e.g. `"bert-base-chinese"`). |
| `path` | string | no | Model source. Local directory path or HuggingFace repo id. Falls back to `arch` if omitted. |
| `dropout` | float | no | Dropout rate for classifier heads (default: 0.1). |
| `freeze_bert` | bool | no | Freeze base model parameters (default: False). |
| `lora.*` | dict | yes | LoRA configuration (see below). |

#### model.path resolution

- `"bert-base-chinese"` → downloads from HuggingFace Hub
- `"./models/bert-base-chinese"` → loads from local directory (must exist)
- If omitted, uses `arch` value

#### model.lora

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `rank` | int | yes | Low-rank dimension. Smaller = fewer parameters. Typical: 4-16. |
| `alpha` | int | yes | Scaling factor. Usually `2 * rank`. |
| `dropout` | float | no | LoRA dropout rate (default: 0.05). |
| `target_modules` | list | yes | Attention layers to adapt. Typically `["self.query", "self.key", "self.value"]`. |
| `bias` | string | no | Bias handling: `"none"`, `"all"`, or `"lora_only"` (default: `"none"`). |
| `enabled` | bool | no | Set to False to disable LoRA (default: True). |

### train

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `num_epochs` | int | yes | Number of training epochs. |
| `batch_size` | int | yes | Batch size per gradient step. |
| `gradient_accumulation_steps` | int | no | Accumulate N batches before updating (default: 1). Effective batch = `batch_size * steps`. |
| `warmup_ratio` | float | no | Fraction of total steps for learning rate warmup (default: 0.1). |
| `monitor_interval` | int | no | Log training metrics every N steps (default: 100). |
| `loss_weight` | list | no | Per-task loss weights for multi_cls (e.g. `[1.0, 1.2]`). |
| `scheduler_type` | string | no | LR scheduler: `"linear"` or `"cosine"` (default: `"linear"`). |
| `early_stopping.patience` | int | no | Stop after N epochs without improvement. |
| `early_stopping.metric` | string | no | Metric to monitor. Defaults to `main_score` when omitted. |

#### train.optimizer

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | no | Optimizer type (default: `"AdamW"`). |
| `groups.bert` | float | yes | Learning rate for base model parameters. |
| `groups.lora` | float | yes | Learning rate for LoRA adapter parameters. |
| `groups.classifier` | float | yes | Learning rate for classifier head parameters. |

### resources

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `priority` | string | no | Optional metadata only. |
| `gpus` | string | no | Device selection: `"auto"`, `"cpu"`, or a specific CUDA device such as `"cuda:0"`. |

---

## Config Inheritance

A config can inherit from a parent via `base_config`:

```yaml
# configs/my_experiment.yaml
base_config: "templates/bert_lora_template.yaml"
exp_id: "my_experiment"
data:
  train_path: "./data/raw/my_data/train.jsonl"
  dev_path: "./data/raw/my_data/dev.jsonl"
```

- Parent values are merged first, then child values override.
- `base_config` can be a relative path (resolved against `configs/` directory) or absolute path.
- Templates live in `configs/templates/`.

---

## Output Structure

After training, artifacts are saved to:

```
outputs/{exp_id}/
├── adapter/              # LoRA adapter weights
│   ├── adapter_model.safetensors
│   └── adapter_config.json
├── classifier/           # Classification head weights
│   └── classifiers.pt
├── tokenizer/            # Saved tokenizer
├── config.yaml           # Copy of training config
└── metrics.json          # Best evaluation metrics
```

## Runtime Notes

- `atomlora eval`, `atomlora predict`, and `atomlora serve` prefer the saved `outputs/{exp_id}/config.yaml` over the current external config file.
- `--config latest` supports both `outputs/latest/config.yaml` and the fallback pointer file `outputs/latest.txt`.
- If tokenizer special tokens were added during training, the adapter may also contain embedding-related weights and be much larger than a pure LoRA-only checkpoint.
