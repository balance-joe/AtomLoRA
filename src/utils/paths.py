# src/utils/paths.py
import os
import json
import yaml

# Artifact subdirectory/file names
ADAPTER_DIR = "adapter"
CLASSIFIER_DIR = "classifier"
TOKENIZER_DIR = "tokenizer"
CONFIG_FILE = "config.yaml"
METRICS_FILE = "metrics.json"
CLASSIFIER_FILE = "classifiers.pt"


def resolve_adapter_path(exp_dir):
    """Resolve LoRA adapter path with backward-compat fallback.

    Checks new `adapter/` subdir first, falls back to experiment root (old layout).
    Returns the new path by default (for saving).
    """
    new_path = os.path.join(exp_dir, ADAPTER_DIR)
    old_path = exp_dir

    for base in (new_path, old_path):
        if os.path.exists(os.path.join(base, "adapter_model.safetensors")):
            return base
        if os.path.exists(os.path.join(base, "adapter_model.bin")):
            return base

    return new_path


def resolve_classifier_path(exp_dir):
    """Resolve classifier path with backward-compat fallback.

    Checks new `classifier/` subdir first, falls back to experiment root.
    Returns the new path by default (for saving).
    """
    new_path = os.path.join(exp_dir, CLASSIFIER_DIR, CLASSIFIER_FILE)
    old_path = os.path.join(exp_dir, CLASSIFIER_FILE)

    if os.path.exists(new_path):
        return new_path
    if os.path.exists(old_path):
        return old_path

    return new_path


def resolve_tokenizer_path(exp_dir):
    """Resolve tokenizer directory path."""
    return os.path.join(exp_dir, TOKENIZER_DIR)


def copy_config_to_output(config, exp_dir):
    """Save a copy of the training config to the experiment output directory."""
    config_path = os.path.join(exp_dir, CONFIG_FILE)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def save_metrics(metrics, exp_dir):
    """Save evaluation metrics as JSON to the experiment output directory."""
    metrics_path = os.path.join(exp_dir, METRICS_FILE)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
