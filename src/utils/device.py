import torch


def resolve_device(config=None, override=None):
    """Resolve runtime device from override or config.resources.gpus."""
    if override is not None:
        requested = str(override)
    elif config is not None:
        requested = str(config.get("resources", {}).get("gpus", "auto"))
    else:
        requested = "auto"

    requested = requested.strip().lower()
    if requested in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        requested = "cuda:0"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"请求使用 {requested}，但当前环境不可用 CUDA")
        if ":" in requested:
            index = int(requested.split(":", 1)[1])
            if index < 0 or index >= torch.cuda.device_count():
                raise ValueError(f"请求的 GPU 不存在: {requested}")
        return torch.device(requested)

    raise ValueError(f"不支持的设备配置: {requested}")
