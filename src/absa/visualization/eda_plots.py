"""Biểu đồ cho phần EDA. Mọi hàm nhận bảng từ :mod:`absa.eda` và trả về ``Figure``."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from absa.visualization.style import POLARITY_COLORS


def plot_aspect_polarity(
    counts: pd.DataFrame, title: str = "Phân bố khía cạnh x cảm xúc (train)"
) -> plt.Figure:
    """Cột chồng số lượt nhắc từng khía cạnh theo cảm xúc (từ :func:`aspect_polarity_counts`)."""
    cols = ["Negative", "Neutral", "Positive", "Present"]
    data = counts[cols].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    left = np.zeros(len(data))
    for col in cols:
        ax.barh(data.index, data[col], left=left, color=POLARITY_COLORS[col], label=col)
        left += data[col].to_numpy()
    for i, total in enumerate(left):
        ax.text(total + 30, i, f"{int(total)}", va="center", fontsize=9)
    ax.set_xlabel("Số bình luận")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def plot_polarity_share(counts: pd.DataFrame) -> plt.Figure:
    """Tỉ lệ % cảm xúc *bên trong* mỗi khía cạnh – cho thấy mất cân bằng lớp."""
    cols = ["Negative", "Neutral", "Positive"]
    data = counts.drop(index="OTHERS", errors="ignore")[cols]
    share = 100 * data.div(data.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    share.plot(
        kind="bar", stacked=True, color=[POLARITY_COLORS[c] for c in cols], ax=ax, width=0.75
    )
    ax.set_ylabel("% trong khía cạnh")
    ax.set_xlabel("")
    ax.set_title("Tỉ lệ cảm xúc trong từng khía cạnh")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_labels_per_comment(dist: pd.Series) -> plt.Figure:
    """Histogram số khía cạnh trên mỗi bình luận."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(dist.index.astype(str), dist.values, color="#00798c")
    for x, y in zip(dist.index.astype(str), dist.values):
        ax.text(x, y, str(y), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Số khía cạnh trong một bình luận")
    ax.set_ylabel("Số bình luận")
    ax.set_title("Mỗi bình luận nhắc tới bao nhiêu khía cạnh?")
    fig.tight_layout()
    return fig


def plot_cooccurrence(matrix: pd.DataFrame) -> plt.Figure:
    """Heatmap P(cột | hàng) đồng xuất hiện khía cạnh."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 8}
    )
    ax.set_title("Đồng xuất hiện khía cạnh: P(cột | hàng)")
    fig.tight_layout()
    return fig


def plot_length_distribution(lengths: pd.DataFrame, clip_quantile: float = 0.99) -> plt.Figure:
    """Phân bố số từ và số ký tự mỗi bình luận."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, label in zip(axes, ["n_words", "n_chars"], ["Số từ", "Số ký tự"]):
        data = lengths[col].clip(upper=lengths[col].quantile(clip_quantile))
        sns.histplot(data, bins=40, kde=True, color="#00798c", ax=ax)
        ax.axvline(
            lengths[col].median(),
            color="#d1495b",
            ls="--",
            label=f"median={lengths[col].median():.0f}",
        )
        ax.set_xlabel(label)
        ax.legend(frameon=False)
    fig.suptitle("Độ dài bình luận (train)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_star_polarity(crosstab: pd.DataFrame) -> plt.Figure:
    """Tỉ lệ cảm xúc khía cạnh theo số sao."""
    cols = ["Negative", "Neutral", "Positive"]
    fig, ax = plt.subplots(figsize=(7, 4))
    crosstab[cols].plot(kind="bar", stacked=True, color=[POLARITY_COLORS[c] for c in cols], ax=ax)
    ax.set_ylabel("% lượt nhắc khía cạnh")
    ax.set_xlabel("Số sao")
    ax.set_title("Số sao vs cảm xúc của khía cạnh")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def plot_text_patterns(report: pd.DataFrame) -> plt.Figure:
    """Tỉ lệ % bình luận có từng xu hướng viết (từ :func:`text_pattern_report`)."""
    data = report.sort_values("pct_comments")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(data["pattern"], data["pct_comments"], color="#30638e")
    for i, v in enumerate(data["pct_comments"]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("% bình luận")
    ax.set_title("Xu hướng viết trong bình luận (train)")
    fig.tight_layout()
    return fig


def plot_top_terms(terms: pd.DataFrame, aspect: str, top_n: int = 10) -> plt.Figure:
    """Top từ/cụm từ đặc trưng (chi2) cho từng cảm xúc của một khía cạnh."""
    labels = list(dict.fromkeys(terms["label"]))
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4), squeeze=False)
    for ax, label in zip(axes[0], labels):
        sub = terms[terms["label"] == label].head(top_n).iloc[::-1]
        ax.barh(sub["term"], sub["chi2"], color=POLARITY_COLORS.get(label, "#30638e"))
        ax.set_title(f"{aspect} – {label}")
        ax.set_xlabel("chi²")
    fig.tight_layout()
    return fig


def plot_split_shift(shift: pd.DataFrame) -> plt.Figure:
    """So sánh tỉ lệ khía cạnh giữa các tập (chỉ số liệu tổng hợp)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    shift.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("% bình luận nhắc tới")
    ax.set_title("Phân phối khía cạnh giữa các tập")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_preprocessing_impact(impact: pd.DataFrame) -> plt.Figure:
    """Kích thước từ vựng và số token hiếm trước/sau tiền xử lý."""
    data = impact.loc[["vocab_size", "hapax"]].iloc[:, :2]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    data.plot(kind="bar", ax=ax, color=["#8d96a3", "#00798c"])
    ax.set_title("Tác động của tiền xử lý lên từ vựng")
    ax.set_ylabel("Số lượng")
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_wordclouds(texts_by_label: Mapping[str, list[str]], max_words: int = 80) -> plt.Figure:
    """Word cloud cho từng nhóm văn bản (đã tiền xử lý)."""
    from wordcloud import WordCloud

    fig, axes = plt.subplots(
        1, len(texts_by_label), figsize=(5 * len(texts_by_label), 4), squeeze=False
    )
    for ax, (label, texts) in zip(axes[0], texts_by_label.items()):
        cloud = WordCloud(
            width=600,
            height=400,
            background_color="white",
            max_words=max_words,
            collocations=False,
            regexp=r"\S+",
        ).generate(" ".join(texts))
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    return fig
