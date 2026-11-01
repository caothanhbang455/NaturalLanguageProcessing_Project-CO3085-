"""Cố định seed để kết quả thí nghiệm tái lập được."""

from __future__ import annotations

import os
import random

import numpy as np

from absa.config import SEED


def set_seed(seed: int = SEED) -> None:
    """Cố định seed cho ``random`` và ``numpy``.

    Args:
        seed: Giá trị seed.

    Note:
        Gensim (Word2Vec/FastText) chỉ tái lập tuyệt đối khi ``workers=1`` và
        biến môi trường ``PYTHONHASHSEED`` được đặt *trước* khi khởi động Python.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
