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


def update_latest_link(exp_dir, metrics=None, config_path=None):
    """Create/update outputs/latest → exp_dir symlink for easy access.

    Works on Linux/Mac (symlink) and Windows (junction, no admin needed).
    Falls back to writing a text file if neither works.

    Also writes info.json into exp_dir (accessible via outputs/latest/info.json).
    """
    from datetime import datetime

    outputs_root = os.path.dirname(exp_dir)
    link_path = os.path.join(outputs_root, "latest")

    # 写 info.json 到实验目录本身（通过链接自然可访问）
    _write_info_file(exp_dir, metrics, config_path)

    # Remove existing link/junction (only if it's a link, never delete real dirs)
    if os.path.islink(link_path):
        os.unlink(link_path)
    elif os.path.isdir(link_path):
        junction_target = _read_junction(link_path)
        if junction_target:
            import shutil
            shutil.rmtree(link_path, ignore_errors=True)
        else:
            # Real directory — don't touch it
            return

    # Try symlink (Linux/Mac/Windows dev mode)
    try:
        os.symlink(exp_dir, link_path, target_is_directory=True)
        return
    except OSError:
        pass

    # Try Windows junction (no admin needed)
    if os.name == "nt":
        try:
            import subprocess
            subprocess.run(
                ["cmd", "/c", "mklink", "/J",
                 os.path.abspath(link_path), os.path.abspath(exp_dir)],
                check=True, capture_output=True
            )
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Fallback: write path to text file
    try:
        with open(link_path + ".txt", "w", encoding="utf-8") as f:
            f.write(exp_dir)
    except Exception:
        pass


def _read_junction(path):
    """Try to read a Windows junction target. Returns None if not a junction."""
    if os.name != "nt":
        return None
    try:
        import subprocess
        result = subprocess.run(
            ["cmd", "/c", "dir", "/AL", path],
            capture_output=True, text=True
        )
        # 中文 Windows 输出可能是 <JUNCTION>，英文是 [JUNCTION]，统一转大写匹配
        if "JUNCTION" in result.stdout.upper():
            return True
    except Exception:
        pass
    return None


def _write_info_file(exp_dir, metrics=None, config_path=None):
    """Write info.json into the experiment directory with metadata."""
    from datetime import datetime
    try:
        info = {
            "exp_id": os.path.basename(exp_dir),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "output_dir": exp_dir,
        }
        if config_path:
            info["config"] = config_path
        if metrics:
            info["metrics"] = metrics
        info_path = os.path.join(exp_dir, "info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
