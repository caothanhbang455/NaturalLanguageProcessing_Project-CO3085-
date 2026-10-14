"""Phong cách vẽ thống nhất và hàm lưu hình cho báo cáo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from absa.config import PATHS
from absa.utils.io import ensure_dir

# Màu cố định cho cảm xúc – dùng giống nhau ở mọi biểu đồ để người đọc không phải tra lại.
POLARITY_COLORS: dict[str, str] = {
    "Negative": "#d1495b",
    "Neutral": "#edae49",
    "Positive": "#00798c",
    "Present": "#8d96a3",
    "None": "#d9d9d9",
}
FAMILY_PALETTE = "tab10"


def set_style() -> None:
    """Áp dụng style chung cho matplotlib/seaborn."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, name: str, directory: str | Path | None = None) -> Path:
    """Lưu hình vào ``outputs/figures/<name>.png`` (tạo thư mục nếu chưa có)."""
    path = ensure_dir(directory or PATHS.figures_dir) / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path
