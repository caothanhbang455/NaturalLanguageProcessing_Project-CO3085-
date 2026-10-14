"""Logger thống nhất cho toàn project."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "absa", level: int = logging.INFO) -> logging.Logger:
    """Trả về logger có định dạng chung, không gắn handler trùng khi gọi nhiều lần.

    Args:
        name: Tên logger, thường truyền ``__name__`` của module gọi.
        level: Mức log tối thiểu.

    Returns:
        Logger đã cấu hình, ghi ra ``stdout`` (hiển thị đẹp trong notebook).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
