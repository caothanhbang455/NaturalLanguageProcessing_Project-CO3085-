"""Các bước ở mức token: gắn phủ định và loại stopword.

Tiếng Việt là ngôn ngữ đơn lập: một *từ* có thể gồm nhiều *âm tiết* cách nhau bởi
dấu cách (``màn hình``, ``điện thoại``). Project giữ nguyên **mức âm tiết** và để
n-gram (uni + bi-gram) tự bắt cụm từ, thay vì tách từ (word segmentation).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

# Từ đệm có thể đứng giữa từ phủ định và từ được phủ định: "không được tốt",
# "chẳng hề nhanh", "không bị lag". Khi gặp, phạm vi phủ định được nới thêm.
NEGATION_FILLERS: frozenset[str] = frozenset(
    {"được", "hề", "có", "phải", "bị", "quá", "lắm", "mấy", "thể", "còn", "thấy"}
)


def is_punctuation_token(token: str) -> bool:
    """Token chỉ gồm dấu câu/ký hiệu (không có chữ hay số), vd ``.``, ``?``, ``-``, ``(``."""
    return not any(ch.isalnum() for ch in token)


def attach_negation(
    tokens: Sequence[str],
    negations: Collection[str],
    window: int = 1,
    fillers: Collection[str] = NEGATION_FILLERS,
) -> list[str]:
    """Nối từ phủ định với từ đứng sau thành một token, **không vượt qua ranh giới câu**.

    ``["pin", "không", "trâu"]`` -> ``["pin", "không_trâu"]``. Với túi-từ, đây là cách
    đơn giản và hiệu quả để ``không_trâu`` có trọng số ngược với ``trâu``
    (tương tự kỹ thuật đánh dấu phạm vi phủ định của Das & Chen, 2001; Pang et al., 2002).

    Phạm vi phủ định dừng lại khi gặp **token dấu câu** hoặc **một từ phủ định khác**:

    * ``["em", "ko", "?", ".", "cam", "thì"]`` -> ``"ko"`` đứng một mình, **không** thành
      ``ko_cam`` (``?`` và ``.`` là ranh giới câu).
    * ``["không", "không", "tốt"]`` -> ``["không", "không_tốt"]`` (phủ định kép không bị nối
      thành một khối).

    Vì vậy hàm phải được gọi **trước** khi xóa dấu câu.

    Args:
        tokens: Danh sách token (dấu câu đã được tách thành token riêng).
        negations: Tập từ phủ định.
        window: Số token nội dung được gộp sau từ phủ định (``0`` = tắt).
        fillers: Từ đệm được gộp thêm mà không tính vào ``window``.
    """
    if window <= 0:
        return list(tokens)
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        token = tokens[i]
        if token not in negations:
            out.append(token)
            i += 1
            continue
        span = [token]
        j, taken = i + 1, 0
        while j < n and taken < window:
            following = tokens[j]
            if is_punctuation_token(following) or following in negations:
                break
            span.append(following)
            if following not in fillers:
                taken += 1
            j += 1
        out.append("_".join(span))
        i = j
    return out


def remove_stopwords(tokens: Sequence[str], stopwords: Collection[str]) -> list[str]:
    """Bỏ các token nằm trong ``stopwords`` (danh sách này đã trừ từ được bảo vệ)."""
    return [t for t in tokens if t not in stopwords]
