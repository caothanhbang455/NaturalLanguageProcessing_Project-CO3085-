"""Toàn bộ hàm vẽ biểu đồ (EDA và kết quả); notebook chỉ gọi các hàm này."""

from absa.visualization import eda_plots, result_plots
from absa.visualization.style import POLARITY_COLORS, save_figure, set_style

__all__ = ["POLARITY_COLORS", "eda_plots", "result_plots", "save_figure", "set_style"]
