"""实验输出目录的路径管理：子目录结构、向后兼容、软链接创建"""
import os
import json
import yaml

# 实验产物子目录/文件名常量
ADAPTER_DIR = "adapter"
CLASSIFIER_DIR = "classifier"
TOKENIZER_DIR = "tokenizer"
CONFIG_FILE = "config.yaml"
METRICS_FILE = "metrics.json"
CLASSIFIER_FILE = "classifiers.pt"


def resolve_adapter_path(exp_dir):
    """查找 LoRA 适配器路径，兼容新目录结构（adapter/）和旧布局（根目录）"""
    new_path = os.path.join(exp_dir, ADAPTER_DIR)
    old_path = exp_dir

    for base in (new_path, old_path):
        if os.path.exists(os.path.join(base, "adapter_model.safetensors")):
            return base
        if os.path.exists(os.path.join(base, "adapter_model.bin")):
            return base

    return new_path


def resolve_classifier_path(exp_dir):
    """查找分类头权重路径，兼容新目录结构和旧布局"""
    new_path = os.path.join(exp_dir, CLASSIFIER_DIR, CLASSIFIER_FILE)
    old_path = os.path.join(exp_dir, CLASSIFIER_FILE)

    if os.path.exists(new_path):
        return new_path
    if os.path.exists(old_path):
        return old_path

    return new_path


def resolve_tokenizer_path(exp_dir):
    """返回 Tokenizer 保存目录路径"""
    return os.path.join(exp_dir, TOKENIZER_DIR)


def copy_config_to_output(config, exp_dir):
    """将训练配置保存一份副本到实验输出目录"""
    config_path = os.path.join(exp_dir, CONFIG_FILE)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def save_metrics(metrics, exp_dir):
    """将评估指标保存为 JSON 到实验输出目录"""
    metrics_path = os.path.join(exp_dir, METRICS_FILE)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def update_latest_link(exp_dir, metrics=None, config_path=None):
    """创建/更新 outputs/latest → exp_dir 的软链接，方便访问最新实验。

    支持 Linux/Mac（symlink）和 Windows（junction，无需管理员权限），
    都失败时降级为写入文本文件。同时在实验目录写入 info.json。
    """
    from datetime import datetime

    outputs_root = os.path.dirname(exp_dir)
    link_path = os.path.join(outputs_root, "latest")

    # 写 info.json 到实验目录本身（通过链接自然可访问）
    _write_info_file(exp_dir, metrics, config_path)

    # 移除已有的链接/junction（只删链接，不删真实目录）
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

    # 尝试 symlink（Linux/Mac/Windows 开发模式）
    try:
        os.symlink(exp_dir, link_path, target_is_directory=True)
        return
    except OSError:
        pass

    # 尝试 Windows junction（不需要管理员权限）
    if os.name == "nt":
        try:
            import subprocess
            abs_link = os.path.abspath(link_path)
            abs_target = os.path.abspath(exp_dir)
            subprocess.run(
                f'cmd /c mklink /J "{abs_link}" "{abs_target}"',
                check=True, capture_output=True, shell=True
            )
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # 兜底：将路径写入文本文件
    try:
        with open(link_path + ".txt", "w", encoding="utf-8") as f:
            f.write(exp_dir)
    except Exception:
        pass


def _read_junction(path):
    """读取 Windows junction 的目标路径，非 junction 返回 None"""
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
    """在实验目录写入 info.json，包含实验 ID、时间、指标等元信息"""
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
