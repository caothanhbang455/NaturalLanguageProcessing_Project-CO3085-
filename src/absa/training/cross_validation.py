"""Cross-validation trên tập train: dự đoán out-of-fold phục vụ phân tích lỗi.

Dự đoán out-of-fold (OOF) cho **mỗi** bình luận train được sinh bởi một mô hình không
nhìn thấy bình luận đó khi huấn luyện. Nhờ vậy ta phân tích lỗi trên ~7.800 mẫu mà
không cần đọc nội dung dev – giữ dev "sạch" cho việc chọn mô hình.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from absa.config import SEED
from absa.data.labels import to_aspect_matrix
from absa.training.builders import build_ml_pipeline
from absa.utils.logging import get_logger

logger = get_logger(__name__)


def stratification_key(Y: np.ndarray) -> np.ndarray:
    """Nhãn đại diện để chia fold cân bằng cho dữ liệu đa nhãn.

    Mỗi bình luận được đại diện bởi **khía cạnh hiếm nhất** mà nó có (``-1`` nếu không
    có khía cạnh nào). Chia stratified theo khóa này giúp các khía cạnh hiếm như
    STORAGE xuất hiện ở mọi fold – một xấp xỉ đơn giản của iterative stratification.
    """
    A = to_aspect_matrix(Y).astype(bool)
    frequency = A.sum(axis=0)
    order = np.argsort(frequency)  # hiếm -> phổ biến
    keys = np.full(len(A), -1)
    for j in order[::-1]:  # gán từ phổ biến tới hiếm, khía cạnh hiếm ghi đè sau cùng
        keys[A[:, j]] = j
    # Gộp lớp quá nhỏ (< số fold) vào -1 để StratifiedKFold không báo lỗi.
    values, counts = np.unique(keys, return_counts=True)
    for value, count in zip(values, counts):
        if count < 10:
            keys[keys == value] = -1
    return keys


def cross_val_predict_absa(
    texts: Sequence[str],
    Y: np.ndarray,
    feature: str,
    model: str,
    strategy: str = "multilabel",
    params: dict | None = None,
    n_splits: int = 5,
    seed: int = SEED,
    n_jobs: int | None = -1,
) -> np.ndarray:
    """Sinh dự đoán out-of-fold cho toàn bộ ``texts``.

    Args:
        texts: Văn bản **đã tiền xử lý** đúng profile của biểu diễn.
        Y: Ma trận nhãn tương ứng.
        feature: Tên biểu diễn.
        model: Tên mô hình ML.
        strategy: ``"multilabel"``, ``"joint"`` hoặc ``"two_stage"``.
        params: Siêu tham số của mô hình.
        n_splits: Số fold.
        seed: Seed chia fold.
        n_jobs: Số tiến trình song song giữa các khía cạnh.

    Returns:
        Ma trận dự đoán ``(n, 11)`` theo đúng thứ tự ``texts``.
    """
    texts = np.asarray(list(texts), dtype=object)
    Y = np.asarray(Y)
    template = build_ml_pipeline(feature, model, strategy, params, n_jobs=n_jobs)
    Y_oof = np.zeros_like(Y)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (train_idx, test_idx) in enumerate(folds.split(texts, stratification_key(Y)), 1):
        pipeline = clone(template)
        pipeline.fit(list(texts[train_idx]), Y[train_idx])
        Y_oof[test_idx] = pipeline.predict(list(texts[test_idx]))
        logger.info("Fold %d/%d xong", k, n_splits)
    return Y_oof
