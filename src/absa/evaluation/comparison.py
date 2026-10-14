"""Tổng hợp kết quả thí nghiệm để so sánh: theo chiến lược, theo biểu diễn, theo mô hình."""

from __future__ import annotations

import pandas as pd

from absa.config import PRIMARY_METRIC

SUMMARY_METRICS: tuple[str, ...] = (
    PRIMARY_METRIC,
    "pair_f1_micro",
    "aspect_f1_macro",
    "aspect_f1_micro",
    "polarity_acc_on_detected",
    "fit_sec",
)


def _grid_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Các dòng lưới chạy thành công, bỏ ``majority_baseline`` (không phản ánh chất lượng học)."""
    return results[
        (results["status"] == "ok")
        & (results["tag"] == "grid")
        & (results["model"] != "majority_baseline")
    ]


def summarize_by(results: pd.DataFrame, by: str = "strategy") -> pd.DataFrame:
    """Trung bình các metric theo một cột nhóm (``strategy``, ``feature``, ``model``...).

    Chỉ giữ các cặp (biểu diễn, mô hình, chiến lược) **có đủ mặt ở mọi nhóm** để so sánh công
    bằng: nếu một chiến lược chưa chạy xong một cặp nào đó thì cặp đó bị bỏ ở mọi nhóm.

    Args:
        results: Bảng kết quả của :class:`absa.training.ExperimentRunner`.
        by: Cột để nhóm.

    Returns:
        Mỗi nhóm một dòng: ``n_pairs`` và trung bình các metric, xếp giảm dần theo metric chính.
    """
    data = _grid_rows(results)
    if data.empty:
        return data
    keys = ["feature", "model", "strategy"]
    others = [k for k in keys if k != by]
    n_groups = data[by].nunique()
    complete = data.groupby(others)[by].transform("nunique") == n_groups
    data = data[complete]
    summary = data.groupby(by)[list(SUMMARY_METRICS)].mean().round(4)
    summary.insert(0, "n_pairs", data.groupby(by).size())
    return summary.sort_values(PRIMARY_METRIC, ascending=False)
