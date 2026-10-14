"""Phân tích lỗi: mô hình sai ở đâu, sai kiểu gì, trên những câu nào.

Ba loại lỗi cho mỗi khía cạnh:

* **missed**   – khía cạnh có trong nhãn thật nhưng mô hình không phát hiện (FN).
* **spurious** – mô hình báo khía cạnh không có trong nhãn thật (FP).
* **polarity** – phát hiện đúng khía cạnh nhưng sai cảm xúc.

Khuyến nghị: phân tích lỗi trên **dự đoán out-of-fold của tập train**
(:func:`absa.training.cross_validation.cross_val_predict_absa`) để không phải đọc nội
dung dev. Phân tích trên dev chỉ nên làm khi đã chốt mô hình và có ghi chú rõ.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from absa.config import ALL_ASPECTS, ID_TO_POLARITY, NONE_ID, SENTIMENT_ASPECTS
from absa.data.labels import decode_row, format_label_string


def error_type_summary(Y_true: np.ndarray, Y_pred: np.ndarray) -> pd.DataFrame:
    """Đếm ``missed / spurious / polarity`` theo từng khía cạnh."""
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    t_present, p_present = Y_true != NONE_ID, Y_pred != NONE_ID
    rows = []
    for j, aspect in enumerate(ALL_ASPECTS):
        rows.append(
            {
                "aspect": aspect,
                "support": int(t_present[:, j].sum()),
                "missed": int((t_present[:, j] & ~p_present[:, j]).sum()),
                "spurious": int((~t_present[:, j] & p_present[:, j]).sum()),
                "polarity": int(
                    (t_present[:, j] & p_present[:, j] & (Y_true[:, j] != Y_pred[:, j])).sum()
                ),
            }
        )
    frame = pd.DataFrame(rows).set_index("aspect")
    frame["total_errors"] = frame[["missed", "spurious", "polarity"]].sum(axis=1)
    return frame.sort_values("total_errors", ascending=False)


def polarity_confusions(Y_true: np.ndarray, Y_pred: np.ndarray) -> pd.DataFrame:
    """Đếm các cặp nhầm cảm xúc (vd thật ``Neutral`` -> đoán ``Positive``) trên mọi khía cạnh."""
    n = len(SENTIMENT_ASPECTS)
    t, p = np.asarray(Y_true)[:, :n], np.asarray(Y_pred)[:, :n]
    mask = (t != NONE_ID) & (p != NONE_ID) & (t != p)
    pairs = pd.Series(
        [f"{ID_TO_POLARITY[a]} -> {ID_TO_POLARITY[b]}" for a, b in zip(t[mask], p[mask])]
    )
    return pairs.value_counts().rename_axis("true -> predicted").to_frame("count")


def build_error_table(
    texts: Sequence[str],
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    processed_texts: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Bảng từng bình luận có lỗi, kèm nhãn thật/dự đoán và mô tả lỗi.

    Returns:
        DataFrame cột ``text, processed, true, pred, missed, spurious, polarity, n_errors``
        (chỉ gồm bình luận có ít nhất một lỗi), sắp xếp theo số lỗi giảm dần.
    """
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    rows = []
    for i, (yt, yp) in enumerate(zip(Y_true, Y_pred)):
        if (yt == yp).all():
            continue
        missed = [a for j, a in enumerate(ALL_ASPECTS) if yt[j] and not yp[j]]
        spurious = [a for j, a in enumerate(ALL_ASPECTS) if yp[j] and not yt[j]]
        polarity = [a for j, a in enumerate(ALL_ASPECTS) if yt[j] and yp[j] and yt[j] != yp[j]]
        rows.append(
            {
                "index": i,
                "text": texts[i],
                "processed": processed_texts[i] if processed_texts is not None else None,
                "true": format_label_string(decode_row(yt)),
                "pred": format_label_string(decode_row(yp)),
                "missed": ", ".join(missed),
                "spurious": ", ".join(spurious),
                "polarity": ", ".join(polarity),
                "n_errors": len(missed) + len(spurious) + len(polarity),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if processed_texts is None:
        frame = frame.drop(columns="processed")
    return frame.sort_values("n_errors", ascending=False).set_index("index")


def sample_errors(
    error_table: pd.DataFrame,
    aspect: str | None = None,
    kind: str | None = None,
    n: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """Lấy ngẫu nhiên ``n`` lỗi, có thể lọc theo khía cạnh và loại lỗi.

    Args:
        error_table: Kết quả của :func:`build_error_table`.
        aspect: Chỉ lấy lỗi liên quan tới khía cạnh này.
        kind: ``"missed" | "spurious" | "polarity"``; ``None`` = mọi loại.
        n: Số mẫu.
        seed: Seed để lần chạy sau ra cùng mẫu.
    """
    frame = error_table
    kinds = [kind] if kind else ["missed", "spurious", "polarity"]
    if aspect:
        mask = np.zeros(len(frame), dtype=bool)
        for k in kinds:
            mask |= frame[k].str.split(", ").apply(lambda xs, a=aspect: a in xs).to_numpy()
        frame = frame[mask]
    elif kind:
        frame = frame[frame[kind] != ""]
    return frame.sample(min(n, len(frame)), random_state=seed) if len(frame) else frame
