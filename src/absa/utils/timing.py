"""Đo thời gian chạy bằng context manager."""

from __future__ import annotations

import time
from types import TracebackType


class Timer:
    """Đồng hồ bấm giờ dùng với ``with``.

    Example:
        >>> with Timer() as t:
        ...     _ = sum(range(1000))
        >>> t.seconds >= 0
        True
    """

    def __init__(self) -> None:
        self.start: float = 0.0
        self.seconds: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.seconds = time.perf_counter() - self.start
