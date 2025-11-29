# src/utils/logger.py
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional

# 全局日志配置
LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
MAX_LOG_SIZE = 10 * 1024 * 1024  # 单日志文件最大10MB
BACKUP_COUNT = 5  # 日志文件备份数
DEFAULT_LOG_LEVEL = logging.INFO

# 日志格式（包含时间、级别、实验ID、模块、消息）
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [exp:%(experiment_id)s] [task:%(task_type)s] [%(module)s:%(lineno)d] - %(message)s"
CONSOLE_FORMAT = "[%(asctime)s] [%(levelname)s] - %(message)s"  # 控制台简化格式


class ExperimentFilter(logging.Filter):
    """日志过滤器：绑定实验ID和任务类型"""
    def __init__(self, experiment_id: Optional[str] = "unknown", task_type: Optional[str] = "unknown"):
        super().__init__()
        self.experiment_id = experiment_id
        self.task_type = task_type

    def filter(self, record: logging.LogRecord) -> bool:
        # 为日志记录添加实验ID和任务类型字段
        record.experiment_id = self.experiment_id
        record.task_type = self.task_type
        return True


def init_logger(
    experiment_id: Optional[str] = None,
    task_type: Optional[str] = None,
    log_level: int = DEFAULT_LOG_LEVEL
) -> logging.Logger:
    """初始化实验专属Logger
    
    Args:
        experiment_id: 实验ID（用于日志文件目录区分）
        task_type: 任务类型（single_cls/double_cls/unknown）
        log_level: 日志级别（logging.INFO/logging.DEBUG等）
    
    Returns:
        配置好的Logger实例
    """
    # 处理默认值
    exp_id = experiment_id or f"default_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    task_type = task_type or "unknown"
    
    # 创建日志目录（按实验ID划分）
    exp_log_dir = os.path.join(LOG_ROOT, exp_id)
    os.makedirs(exp_log_dir, exist_ok=True)
    
    # 日志文件名（含日期）
    log_filename = os.path.join(exp_log_dir, f"{exp_id}_{task_type}_{datetime.now().strftime('%Y%m%d')}.log")
    
    # 获取Logger实例（避免重复配置）
    logger = logging.getLogger(f"exp_{exp_id}")
    if logger.handlers:  # 已配置过则直接返回
        return logger
    logger.setLevel(log_level)
    
    # 1. 添加文件处理器（轮转+详细格式）
    file_handler = RotatingFileHandler(
        filename=log_filename,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    file_formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(ExperimentFilter(exp_id, task_type))
    logger.addHandler(file_handler)
    
    # 2. 添加控制台处理器（简化格式）
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(CONSOLE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 禁止向上传播（避免重复输出）
    logger.propagate = False
    
    logger.info(f"Logger initialized for experiment: {exp_id} (task_type: {task_type})")
    return logger


def get_logger(experiment_id: Optional[str] = None) -> logging.Logger:
    """获取已初始化的Logger（按实验ID），无则返回默认Logger"""
    if experiment_id:
        logger = logging.getLogger(f"exp_{experiment_id}")
        if logger.handlers:
            return logger
    # 返回默认Logger（未绑定实验ID）
    return init_logger()
