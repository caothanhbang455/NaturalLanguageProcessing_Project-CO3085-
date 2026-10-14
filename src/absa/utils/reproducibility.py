"""Cố định seed để kết quả thí nghiệm tái lập được."""

from __future__ import annotations

import os
import random

import numpy as np

from absa.config import SEED


def set_seed(seed: int = SEED, deterministic_torch: bool = False) -> None:
    """Cố định seed cho ``random``, ``numpy`` và ``torch`` (nếu đã cài).

    Args:
        seed: Giá trị seed.
        deterministic_torch: Bật chế độ cuDNN tất định (chậm hơn, chỉ cần khi
            muốn kết quả DL giống hệt nhau giữa các lần chạy trên GPU).

    Note:
        Gensim (Word2Vec/FastText) chỉ tái lập tuyệt đối khi ``workers=1`` và
        biến môi trường ``PYTHONHASHSEED`` được đặt *trước* khi khởi động Python.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
