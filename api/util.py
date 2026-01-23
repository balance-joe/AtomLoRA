import time
from typing import Any

def now_ts() -> int:
    return int(time.time())


def success(
    msg: str = "操作成功",
    data: Any = None,
    code: int = 2000,
):
    """
    操作成功返回
    """
    return {
        "code": code,
        "msg": msg,
        "data": data if data is not None else [],
        "timestamp": now_ts(),
    }


def error(
    msg: str = "操作失败",
    data: Any = None,
    code: int = 5000,
):
    """
    操作失败返回
    """
    return {
        "code": code,
        "msg": msg,
        "data": data if data is not None else [],
        "timestamp": now_ts(),
    }