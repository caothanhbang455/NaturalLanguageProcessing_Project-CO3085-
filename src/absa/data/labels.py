"""Mã hóa / giải mã nhãn ABSA của UIT-ViSFD.

Định dạng gốc trong cột ``label``::

    {BATTERY#Negative};{GENERAL#Positive};{OTHERS};

Project biểu diễn nhãn thành **ma trận số nguyên** ``Y`` kích thước
``(n_mẫu, 11)``, mỗi cột ứng với một khía cạnh theo thứ tự ``ALL_ASPECTS``:

* 10 khía cạnh có cảm xúc: ``0 = không nhắc tới``, ``1 = Negative``,
  ``2 = Neutral``, ``3 = Positive``.
* Cột ``OTHERS``: ``0 = không có``, ``1 = có`` (không mang cảm xúc).

Từ ``Y`` có thể suy ra hai góc nhìn dùng cho đánh giá:

* ``aspect matrix``  – nhị phân (n, 11): khía cạnh có được nhắc tới hay không.
* ``pair matrix``    – nhị phân (n, 31): từng cặp ``ASPECT#Polarity`` (+ ``OTHERS``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from absa.config import (
    ALL_ASPECTS,
    ID_TO_POLARITY,
    NONE_ID,
    OTHERS_ASPECT,
    OTHERS_PRESENT_ID,
    POLARITIES,
    POLARITY_TO_ID,
    SENTIMENT_ASPECTS,
)

_LABEL_PATTERN = re.compile(r"\{\s*([^{}#]+?)\s*(?:#\s*([^{}]*?)\s*)?\}")
ASPECT_INDEX: dict[str, int] = {a: i for i, a in enumerate(ALL_ASPECTS)}
OTHERS_INDEX = ASPECT_INDEX[OTHERS_ASPECT]


def parse_label_string(label: str | float | None) -> dict[str, str | None]:
    """Tách chuỗi nhãn gốc thành ``{khía_cạnh: cảm_xúc}``.

    Args:
        label: Chuỗi dạng ``"{CAMERA#Positive};{OTHERS};"``. Giá trị rỗng/NaN
            được hiểu là không có khía cạnh nào.

    Returns:
        Dict theo thứ tự xuất hiện; ``OTHERS`` có giá trị ``None``.

    Raises:
        ValueError: Khi gặp khía cạnh hoặc cảm xúc không nằm trong tập định nghĩa
            (giúp phát hiện sớm dữ liệu hỏng thay vì âm thầm bỏ qua).
    """
    if not isinstance(label, str):
        return {}
    parsed: dict[str, str | None] = {}
    for raw_aspect, raw_polarity in _LABEL_PATTERN.findall(label):
        aspect = raw_aspect.strip().upper()
        if aspect not in ASPECT_INDEX:
            raise ValueError(f"Khía cạnh lạ '{raw_aspect}' trong nhãn: {label!r}")
        if aspect == OTHERS_ASPECT:
            parsed[aspect] = None
            continue
        polarity = raw_polarity.strip().capitalize()
        if polarity not in POLARITY_TO_ID:
            raise ValueError(f"Cảm xúc lạ '{raw_polarity}' cho {aspect} trong nhãn: {label!r}")
        parsed[aspect] = polarity
    return parsed


def encode_labels(labels: Iterable[str | float | None]) -> np.ndarray:
    """Chuyển danh sách chuỗi nhãn thành ma trận ``Y`` (n, 11) kiểu ``int8``."""
    rows = []
    for label in labels:
        row = np.zeros(len(ALL_ASPECTS), dtype=np.int8)
        for aspect, polarity in parse_label_string(label).items():
            idx = ASPECT_INDEX[aspect]
            row[idx] = OTHERS_PRESENT_ID if polarity is None else POLARITY_TO_ID[polarity]
        rows.append(row)
    if not rows:
        return np.zeros((0, len(ALL_ASPECTS)), dtype=np.int8)
    return np.vstack(rows)


def decode_row(row: Sequence[int]) -> dict[str, str | None]:
    """Chuyển một hàng của ``Y`` về dict ``{khía_cạnh: cảm_xúc}`` (bỏ khía cạnh vắng mặt)."""
    decoded: dict[str, str | None] = {}
    for aspect, value in zip(ALL_ASPECTS, row):
        if value == NONE_ID:
            continue
        decoded[aspect] = None if aspect == OTHERS_ASPECT else ID_TO_POLARITY[int(value)]
    return decoded


def decode_labels(Y: np.ndarray) -> list[dict[str, str | None]]:
    """Giải mã toàn bộ ma trận ``Y``."""
    return [decode_row(row) for row in np.asarray(Y)]


def format_label_string(aspects: dict[str, str | None]) -> str:
    """Định dạng ngược về chuỗi nhãn gốc (thứ tự khía cạnh theo ``ALL_ASPECTS``)."""
    parts = []
    for aspect in ALL_ASPECTS:
        if aspect not in aspects:
            continue
        polarity = aspects[aspect]
        parts.append(f"{{{aspect}}};" if polarity is None else f"{{{aspect}#{polarity}}};")
    return "".join(parts)


def to_aspect_matrix(Y: np.ndarray) -> np.ndarray:
    """Ma trận nhị phân (n, 11): khía cạnh có được nhắc tới hay không."""
    return (np.asarray(Y) != NONE_ID).astype(np.int8)


def pair_label_names() -> list[str]:
    """Tên 31 cột của pair matrix: ``ASPECT#Polarity`` cho 10 khía cạnh + ``OTHERS``."""
    names = [f"{a}#{p}" for a in SENTIMENT_ASPECTS for p in POLARITIES]
    return names + [OTHERS_ASPECT]


def to_pair_matrix(Y: np.ndarray) -> np.ndarray:
    """Ma trận nhị phân (n, 31) cho từng cặp (khía cạnh, cảm xúc).

    Một dự đoán chỉ được tính đúng ở cột ``ASPECT#Polarity`` khi **cả khía
    cạnh lẫn cảm xúc** đều đúng – đây chính là thước đo của bài toán ABSA đầy đủ.
    """
    Y = np.asarray(Y)
    n_pol = len(POLARITIES)
    pairs = np.zeros((Y.shape[0], len(SENTIMENT_ASPECTS) * n_pol + 1), dtype=np.int8)
    for j, _ in enumerate(SENTIMENT_ASPECTS):
        for k, polarity in enumerate(POLARITIES):
            pairs[:, j * n_pol + k] = Y[:, j] == POLARITY_TO_ID[polarity]
    pairs[:, -1] = Y[:, OTHERS_INDEX] != NONE_ID
    return pairs


def pairs_to_labels(pairs: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
    """Chuyển ma trận 31 nhãn nhị phân (n, 31) ngược về ma trận nhãn ``Y`` (n, 11).

    Đây là phép **giải mã** của chiến lược ``multilabel``: 31 bộ phân loại nhị phân chạy
    độc lập nên có thể cùng bật nhiều cảm xúc cho một khía cạnh (``BATTERY#Negative`` và
    ``BATTERY#Positive`` cùng bằng 1) – điều không hợp lệ vì một khía cạnh chỉ có một
    cảm xúc. Quy tắc giải mã cho từng khía cạnh:

    * không cột nào bật  -> ``0`` (không nhắc tới);
    * đúng một cột bật   -> cảm xúc của cột đó;
    * nhiều cột bật      -> chọn cột có **điểm tin cậy cao nhất** (``scores``); nếu không có
      ``scores`` thì chọn theo thứ tự Negative, Neutral, Positive.

    Args:
        pairs: Ma trận nhị phân (n, 31) theo thứ tự của :func:`pair_label_names`.
        scores: Điểm tin cậy cùng kích thước (vd ``decision_function`` hoặc xác suất),
            chỉ dùng để phân xử khi có xung đột.

    Returns:
        Ma trận ``Y`` kiểu ``int8`` kích thước (n, 11).
    """
    bits = np.asarray(pairs).astype(bool)
    weights = bits.astype(float) if scores is None else np.asarray(scores, dtype=float)
    n_pol = len(POLARITIES)
    polarity_ids = np.array([POLARITY_TO_ID[p] for p in POLARITIES], dtype=np.int8)
    Y = np.zeros((bits.shape[0], len(ALL_ASPECTS)), dtype=np.int8)
    for j in range(len(SENTIMENT_ASPECTS)):
        cols = slice(j * n_pol, (j + 1) * n_pol)
        block = bits[:, cols]
        best = np.where(block, weights[:, cols], -np.inf).argmax(axis=1)
        Y[:, j] = np.where(block.any(axis=1), polarity_ids[best], NONE_ID)
    Y[:, OTHERS_INDEX] = bits[:, -1]
    return Y


def count_pair_conflicts(pairs: np.ndarray) -> dict[str, float]:
    """Đo mức xung đột của dự đoán 31 nhãn: một khía cạnh bị gán nhiều cảm xúc cùng lúc.

    Returns:
        ``n_conflicts`` (số cặp mẫu-khía cạnh xung đột), ``n_mentions`` (số lần khía cạnh
        được báo có mặt) và ``conflict_rate`` = tỉ lệ xung đột trên số lần báo có mặt.
    """
    bits = np.asarray(pairs).astype(int)
    n_pol = len(POLARITIES)
    per_aspect = bits[:, : len(SENTIMENT_ASPECTS) * n_pol].reshape(
        bits.shape[0], len(SENTIMENT_ASPECTS), n_pol
    )
    on = per_aspect.sum(axis=2)
    n_conflicts = int((on > 1).sum())
    n_mentions = int((on > 0).sum())
    return {
        "n_conflicts": n_conflicts,
        "n_mentions": n_mentions,
        "conflict_rate": n_conflicts / n_mentions if n_mentions else 0.0,
    }


def label_frame(Y: np.ndarray) -> pd.DataFrame:
    """Bảng dễ đọc: mỗi cột là một khía cạnh, giá trị là cảm xúc hoặc ``"-"``."""
    Y = np.asarray(Y)
    data = {}
    for j, aspect in enumerate(ALL_ASPECTS):
        if aspect == OTHERS_ASPECT:
            data[aspect] = np.where(Y[:, j] != NONE_ID, "Present", "-")
        else:
            data[aspect] = [ID_TO_POLARITY.get(int(v), "-") for v in Y[:, j]]
    return pd.DataFrame(data)
