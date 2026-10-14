"""Biểu đồ kết quả thí nghiệm: lưới M x N, so sánh mô hình, confusion matrix, lịch sử DL."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from absa.config import PRIMARY_METRIC


def plot_score_heatmap(
    results: pd.DataFrame,
    metric: str = PRIMARY_METRIC,
    strategy: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Heatmap mô hình (hàng) x biểu diễn (cột) cho ``metric`` – hình trung tâm của lưới M x N.

    Ô trống = cặp không tương thích hoặc chưa chạy. Nếu có nhiều chiến lược mà không chỉ
    định ``strategy``, lấy điểm cao nhất giữa các chiến lược.
    """
    data = results[(results["status"] == "ok") & (results["tag"] == "grid")]
    if strategy:
        data = data[data["strategy"] == strategy]
    table = data.pivot_table(index="model", columns="feature", values=metric, aggfunc="max")
    table = table.loc[table.mean(axis=1).sort_values(ascending=False).index]
    table = table[table.mean(axis=0).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(1.1 * table.shape[1] + 3, 0.5 * table.shape[0] + 2))
    sns.heatmap(
        table,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={"label": metric},
        annot_kws={"size": 8},
        linewidths=0.5,
    )
    ax.set_title(title or f"{metric} trên dev: mô hình x biểu diễn")
    ax.set_xlabel("Biểu diễn")
    ax.set_ylabel("Mô hình")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    return fig


def plot_leaderboard(
    board: pd.DataFrame, metric: str = PRIMARY_METRIC, top: int = 15
) -> plt.Figure:
    """Thanh ngang top cấu hình, kèm micro-F1 để thấy hai góc nhìn."""
    data = board.head(top).copy()
    data["config"] = data["feature"] + " + " + data["model"] + " (" + data["strategy"] + ")"
    data = data.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(data) + 1.5))
    ax.barh(data["config"], data[metric], color="#00798c", label=metric)
    if "pair_f1_micro" in data and metric != "pair_f1_micro":
        ax.scatter(
            data["pair_f1_micro"], data["config"], color="#d1495b", zorder=3, label="pair_f1_micro"
        )
    for i, v in enumerate(data[metric]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Điểm trên dev")
    ax.set_title(f"Top {len(data)} cấu hình")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_family_comparison(results: pd.DataFrame, metric: str = PRIMARY_METRIC) -> plt.Figure:
    """Boxplot ``metric`` theo họ biểu diễn và theo họ mô hình."""
    data = results[(results["status"] == "ok") & (results["model"] != "majority_baseline")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, col, title in zip(
        axes, ["feature_family", "model_family"], ["Họ biểu diễn", "Họ mô hình"]
    ):
        order = data.groupby(col)[metric].median().sort_values(ascending=False).index
        sns.boxplot(data=data, x=col, y=metric, order=order, ax=ax, color="#bcd4e6")
        sns.stripplot(data=data, x=col, y=metric, order=order, ax=ax, color="#30638e", size=3)
        ax.set_title(title)
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


def plot_time_vs_score(results: pd.DataFrame, metric: str = PRIMARY_METRIC) -> plt.Figure:
    """Đánh đổi chất lượng – thời gian huấn luyện (trục x log)."""
    data = results[(results["status"] == "ok") & (results["model"] != "majority_baseline")].copy()
    data["total_sec"] = data["fit_sec"].fillna(0) + data.get("feature_fit_sec", 0).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=data, x="total_sec", y=metric, hue="feature_family", style="model_family", s=60, ax=ax
    )
    ax.set_xscale("log")
    ax.set_xlabel("Thời gian biểu diễn + huấn luyện (giây, log)")
    ax.set_title("Chất lượng vs chi phí")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_strategy_comparison(
    results: pd.DataFrame, metric: str = PRIMARY_METRIC, baseline: str = "multilabel"
) -> plt.Figure:
    """So sánh các chiến lược ABSA trên cùng cặp (biểu diễn, mô hình).

    Mỗi ô là một chiến lược so với ``baseline``: mỗi điểm là một cặp (biểu diễn, mô hình),
    điểm nằm **trên đường chéo** nghĩa là chiến lược đó tốt hơn ``baseline``.
    """
    data = results[(results["status"] == "ok") & (results["tag"] == "grid")]
    table = data.pivot_table(index=["feature", "model"], columns="strategy", values=metric)
    if baseline not in table.columns:
        baseline = table.columns[0]
    others = [c for c in table.columns if c != baseline]
    if not others:
        raise ValueError("Cần kết quả của ít nhất hai chiến lược")
    fig, axes = plt.subplots(1, len(others), figsize=(5.2 * len(others), 5), squeeze=False)
    lo, hi = np.nanmin(table.values), np.nanmax(table.values)
    for ax, other in zip(axes[0], others):
        ax.scatter(table[baseline], table[other], color="#30638e", alpha=0.8)
        ax.plot([lo, hi], [lo, hi], ls="--", color="grey")
        wins = int((table[other] > table[baseline]).sum())
        ax.set_xlabel(baseline)
        ax.set_ylabel(other)
        ax.set_title(f"{other} vs {baseline}\n{other} thắng {wins}/{table[other].notna().sum()}")
    fig.suptitle(metric, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_per_aspect(report: pd.DataFrame, title: str = "Kết quả theo khía cạnh") -> plt.Figure:
    """F1 phát hiện và macro-F1 cảm xúc của từng khía cạnh (từ :func:`per_aspect_report`)."""
    data = report.sort_values("support", ascending=False)
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - 0.2, data["detect_f1"], width=0.4, label="Detection F1", color="#30638e")
    ax.bar(
        x + 0.2,
        data["sentiment_f1_macro"].fillna(0),
        width=0.4,
        label="Sentiment macro-F1",
        color="#00798c",
    )
    ax.set_xticks(x, [f"{a}\n(n={s})" for a, s in zip(data.index, data["support"])], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def plot_confusion_grid(
    matrices: Mapping[str, pd.DataFrame], normalize: bool = True
) -> plt.Figure:
    """Lưới confusion matrix của các khía cạnh (hàng = thật, cột = dự đoán)."""
    n = len(matrices)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    for ax, (aspect, cm) in zip(axes.ravel(), matrices.items()):
        data = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) if normalize else cm
        sns.heatmap(
            data,
            annot=cm.values,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            vmin=0,
            vmax=1 if normalize else None,
            annot_kws={"size": 8},
        )
        ax.set_title(aspect)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thật")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrix theo khía cạnh (màu = tỉ lệ theo hàng)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_error_types(summary: pd.DataFrame) -> plt.Figure:
    """Cột chồng số lỗi bỏ sót / báo thừa / sai cảm xúc theo khía cạnh."""
    cols = ["missed", "spurious", "polarity"]
    colors = ["#d1495b", "#edae49", "#30638e"]
    fig, ax = plt.subplots(figsize=(9, 4))
    summary[cols].plot(kind="bar", stacked=True, color=colors, ax=ax)
    ax.set_title("Phân loại lỗi theo khía cạnh")
    ax.set_ylabel("Số lỗi")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(["Bỏ sót (FN)", "Báo thừa (FP)", "Sai cảm xúc"], frameon=False)
    fig.tight_layout()
    return fig


def plot_ablation(ablation: pd.DataFrame, metric: str = PRIMARY_METRIC) -> plt.Figure:
    """Biểu đồ ablation tiền xử lý: điểm của từng biến thể, so với biến thể đầu tiên."""
    data = ablation[metric]
    base = data.iloc[0]
    colors = ["#8d96a3"] + ["#00798c" if v >= base else "#d1495b" for v in data.iloc[1:]]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(data) + 1.5))
    ax.barh(data.index[::-1], data.values[::-1], color=colors[::-1])
    for i, v in enumerate(data.values[::-1]):
        ax.text(v + 0.002, i, f"{v:.4f} ({v - base:+.4f})", va="center", fontsize=8)
    ax.set_xlim(data.min() - 0.03, data.max() + 0.04)
    ax.set_xlabel(f"{metric} trên dev")
    ax.set_title("Ablation tiền xử lý")
    fig.tight_layout()
    return fig


def plot_training_history(
    histories: Mapping[str, pd.DataFrame], metric: str = PRIMARY_METRIC
) -> plt.Figure:
    """Loss train và metric dev theo epoch cho các mô hình DL."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, hist in histories.items():
        axes[0].plot(hist["epoch"], hist["train_loss"], marker="o", label=name)
        axes[1].plot(hist["epoch"], hist[metric], marker="o", label=name)
    axes[0].set_title("Train loss")
    axes[1].set_title(f"Dev {metric}")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_product_summary(
    summary: pd.DataFrame, title: str = "Điểm mạnh / điểm yếu theo khía cạnh"
) -> plt.Figure:
    """Biểu đồ tổng hợp cho demo: % tích cực (phải) và % tiêu cực (trái) theo khía cạnh."""
    data = summary[summary["mentions"] > 0].sort_values("net_score")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(data) + 1.5))
    ax.barh(data.index, data["pct_positive"], color="#00798c", label="% tích cực")
    ax.barh(data.index, -data["pct_negative"], color="#d1495b", label="% tiêu cực")
    for i, mentions in enumerate(data["mentions"]):
        ax.text(102, i, f"n={mentions}", va="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-100, 115)
    ax.set_xlabel("%")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig
