"""Metric cho bài toán ABSA, đánh giá riêng từng nhiệm vụ con.

Theo cách đánh giá của bộ dữ liệu UIT-ViSFD, có hai nhiệm vụ:

1. **Aspect detection** – bình luận nhắc tới khía cạnh nào? Bài toán đa nhãn nhị phân
   trên 11 khía cạnh (``aspect_*``).
2. **Aspect sentiment** – nhắc tới khía cạnh nào *với cảm xúc gì*? Bài toán đa nhãn
   trên 31 cặp ``ASPECT#Polarity`` (``pair_*``). Một cặp chỉ đúng khi cả khía cạnh
   lẫn cảm xúc đều đúng, nên đây là thước đo khó và đầy đủ nhất.

Ngoài ra, ``polarity_acc_on_detected`` đo riêng khả năng phân cực: trong các khía cạnh
được phát hiện đúng, bao nhiêu phần trăm được gán đúng cảm xúc.

Mỗi nhiệm vụ báo cáo Precision/Recall/F1 dạng **micro** (gộp mọi nhãn – bị chi phối
bởi khía cạnh phổ biến) và **macro** (trung bình đều các nhãn – phản ánh khía cạnh hiếm).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from absa.config import (
    ALL_ASPECTS,
    ASPECT_CLASS_NAMES,
    NONE_ID,
    OTHERS_ASPECT,
    OTHERS_CLASS_NAMES,
    SENTIMENT_ASPECTS,
)
from absa.data.labels import ASPECT_INDEX, pair_label_names, to_aspect_matrix, to_pair_matrix


def _prf(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> tuple[float, float, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return float(p), float(r), float(f)


def evaluate_absa(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict[str, float]:
    """Tính toàn bộ metric tổng hợp.

    Args:
        Y_true: Ma trận nhãn thật ``(n, 11)``.
        Y_pred: Ma trận nhãn dự đoán ``(n, 11)``.

    Returns:
        Dict phẳng, ví dụ ``aspect_f1_macro``, ``pair_precision_micro``,
        ``polarity_acc_on_detected``, ``exact_match``.
    """
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    if Y_true.shape != Y_pred.shape:
        raise ValueError(f"Kích thước khác nhau: {Y_true.shape} vs {Y_pred.shape}")
    results: dict[str, float] = {}
    views = {
        "aspect": (to_aspect_matrix(Y_true), to_aspect_matrix(Y_pred)),
        "pair": (to_pair_matrix(Y_true), to_pair_matrix(Y_pred)),
    }
    for prefix, (t, p) in views.items():
        for average in ("micro", "macro"):
            precision, recall, f1 = _prf(t, p, average)
            results[f"{prefix}_precision_{average}"] = precision
            results[f"{prefix}_recall_{average}"] = recall
            results[f"{prefix}_f1_{average}"] = f1

    n_sent = len(SENTIMENT_ASPECTS)
    both = (Y_true[:, :n_sent] != NONE_ID) & (Y_pred[:, :n_sent] != NONE_ID)
    correct = (Y_true[:, :n_sent] == Y_pred[:, :n_sent]) & both
    results["polarity_acc_on_detected"] = float(correct.sum() / both.sum()) if both.any() else 0.0
    results["exact_match"] = float((Y_true == Y_pred).all(axis=1).mean())
    return results


def per_aspect_report(Y_true: np.ndarray, Y_pred: np.ndarray) -> pd.DataFrame:
    """Báo cáo theo từng khía cạnh.

    Cột: ``support`` (số bình luận nhắc tới), P/R/F1 phát hiện, macro-F1 cảm xúc (trên
    3 cặp của khía cạnh) và độ chính xác phân cực khi đã phát hiện đúng.
    """
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    A_true, A_pred = to_aspect_matrix(Y_true), to_aspect_matrix(Y_pred)
    P_true, P_pred = to_pair_matrix(Y_true), to_pair_matrix(Y_pred)
    rows = []
    for aspect, j in ASPECT_INDEX.items():
        p, r, f = _prf(A_true[:, j], A_pred[:, j], "binary")
        row = {
            "aspect": aspect,
            "support": int(A_true[:, j].sum()),
            "detect_precision": p,
            "detect_recall": r,
            "detect_f1": f,
        }
        if aspect != OTHERS_ASPECT:
            cols = slice(3 * j, 3 * j + 3)
            row["sentiment_f1_macro"] = _prf(P_true[:, cols], P_pred[:, cols], "macro")[2]
            both = (A_true[:, j] == 1) & (A_pred[:, j] == 1)
            row["polarity_acc_on_detected"] = (
                float((Y_true[both, j] == Y_pred[both, j]).mean()) if both.any() else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("aspect")


def pair_report(Y_true: np.ndarray, Y_pred: np.ndarray) -> pd.DataFrame:
    """P/R/F1 và support cho từng cặp ``ASPECT#Polarity`` (31 dòng)."""
    return pair_report_from_bits(to_pair_matrix(Y_true), to_pair_matrix(Y_pred))


def pair_report_from_bits(P_true: np.ndarray, P_pred: np.ndarray) -> pd.DataFrame:
    """Như :func:`pair_report` nhưng nhận thẳng ma trận 31 nhãn nhị phân (n, 31).

    Dùng để chấm **dự đoán thô** của chiến lược ``multilabel`` (trước khi giải quyết xung đột).
    """
    p, r, f, s = precision_recall_fscore_support(P_true, P_pred, average=None, zero_division=0)
    return pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": s}, index=pair_label_names()
    )


def evaluate_pair_bits(P_true: np.ndarray, P_pred: np.ndarray) -> dict[str, float]:
    """Metric P/R/F1 (micro, macro) trên 31 nhãn nhị phân, cùng tên khóa với :func:`evaluate_absa`.

    So sánh với ``evaluate_absa`` trên dự đoán đã giải mã để thấy việc xử lý xung đột (một khía
    cạnh bị bật nhiều cảm xúc) làm điểm thay đổi bao nhiêu.
    """
    results: dict[str, float] = {}
    for average in ("micro", "macro"):
        precision, recall, f1 = _prf(np.asarray(P_true), np.asarray(P_pred), average)
        results[f"pair_precision_{average}"] = precision
        results[f"pair_recall_{average}"] = recall
        results[f"pair_f1_{average}"] = f1
    return results


def aspect_confusion_matrix(Y_true: np.ndarray, Y_pred: np.ndarray, aspect: str) -> pd.DataFrame:
    """Confusion matrix 4x4 (``None/Neg/Neu/Pos``) – 2x2 cho OTHERS – của một khía cạnh.

    Hàng = nhãn thật, cột = nhãn dự đoán.
    """
    j = ASPECT_INDEX[aspect]
    names = OTHERS_CLASS_NAMES if aspect == OTHERS_ASPECT else ASPECT_CLASS_NAMES
    labels = list(range(len(names)))
    cm = confusion_matrix(np.asarray(Y_true)[:, j], np.asarray(Y_pred)[:, j], labels=labels)
    return pd.DataFrame(cm, index=list(names), columns=list(names))


def all_confusion_matrices(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict[str, pd.DataFrame]:
    """Confusion matrix cho cả 11 khía cạnh."""
    return {a: aspect_confusion_matrix(Y_true, Y_pred, a) for a in ALL_ASPECTS}
