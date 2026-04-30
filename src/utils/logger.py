"""日志管理：基于 loguru 的实验级日志，支持按实验 ID 隔离日志文件"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional, Union

from loguru import logger as _logger

LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
MAX_LOG_SIZE = 10 * 1024 * 1024  # 单文件最大 10MB
BACKUP_COUNT = 5  # 保留最近 5 个备份
DEFAULT_LOG_LEVEL = "INFO"

# 文件日志格式：包含实验 ID 和任务类型，便于过滤
FILE_FORMAT = (
    "[{time:YYYY-MM-DD HH:mm:ss}] [{level}] "
    "[exp:{extra[experiment_id]}] [task:{extra[task_type]}] "
    "[{module}:{line}] - {message}"
)

# 控制台日志格式：简洁，仅保留时间和消息
CONSOLE_FORMAT = (
    "<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> "
    "<level>[{level}]</level> - <level>{message}</level>"
)

_configured = False
_experiment_loggers = {}


def ensure_utf8_stdio() -> None:
    """强制标准输出使用 UTF-8 编码，防止中文乱码"""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _normalize_level(level: Union[int, str]) -> str:
    """将日志级别统一转为大写字符串"""
    if isinstance(level, int):
        return logging.getLevelName(level)
    return str(level).upper()


def _configure_base_logger(log_level: Union[int, str] = DEFAULT_LOG_LEVEL) -> None:
    """配置基础 logger：移除默认处理器，添加控制台输出（仅执行一次）"""
    global _configured
    if _configured:
        return

    ensure_utf8_stdio()
    _logger.remove()
    _logger.configure(extra={"experiment_id": "unknown", "task_type": "unknown"})
    _logger.add(
        sys.stdout,
        level=_normalize_level(log_level),
        format=CONSOLE_FORMAT,
        colorize=sys.stdout.isatty(),
        backtrace=False,
        diagnose=False,
    )
    _configured = True


def init_logger(
    experiment_id: Optional[str] = None,
    task_type: Optional[str] = None,
    log_level: Union[int, str] = DEFAULT_LOG_LEVEL,
):
    """初始化实验专属 loguru logger。

    Args:
        experiment_id: 实验ID（用于日志文件目录区分）
        task_type: 任务类型（single_cls/multi_cls/unknown）
        log_level: 日志级别（logging.INFO/"INFO" 等）

    Returns:
        绑定 experiment_id/task_type 的 loguru logger 实例
    """
    _configure_base_logger(log_level)

    exp_id = experiment_id or f"default_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    task_type = task_type or "unknown"
    if exp_id in _experiment_loggers:
        return _experiment_loggers[exp_id]

    exp_log_dir = os.path.join(LOG_ROOT, exp_id)
    os.makedirs(exp_log_dir, exist_ok=True)
    log_filename = os.path.join(
        exp_log_dir,
        f"{exp_id}_{task_type}_{datetime.now().strftime('%Y%m%d')}.log",
    )

    _logger.add(
        log_filename,
        level=_normalize_level(log_level),
        format=FILE_FORMAT,
        rotation=MAX_LOG_SIZE,
        retention=BACKUP_COUNT,
        encoding="utf-8",
        backtrace=False,
        diagnose=False,
        filter=lambda record, bound_exp_id=exp_id: (
            record["extra"].get("experiment_id") == bound_exp_id
        ),
    )

    bound_logger = _logger.bind(experiment_id=exp_id, task_type=task_type)
    _experiment_loggers[exp_id] = bound_logger
    bound_logger.info(f"日志初始化完毕: {exp_id} (task_type: {task_type})")
    return bound_logger


def get_logger(experiment_id: Optional[str] = None):
    """获取 loguru logger；若实验已初始化，则返回实验绑定 logger。"""
    _configure_base_logger()
    if experiment_id and experiment_id in _experiment_loggers:
        return _experiment_loggers[experiment_id]
    if experiment_id:
        return _logger.bind(experiment_id=experiment_id, task_type="unknown")
    return _logger
