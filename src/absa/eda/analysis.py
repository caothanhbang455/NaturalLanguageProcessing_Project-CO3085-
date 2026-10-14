"""Các hàm thống kê phục vụ EDA – chỉ trả về số liệu/bảng.

Việc vẽ nằm ở :mod:`absa.visualization.eda_plots`.

Nguyên tắc: EDA chi tiết (đọc nội dung, tìm xu hướng viết) **chỉ chạy trên Train**.
Với Dev chỉ dùng :func:`label_distribution_shift` – một so sánh ở mức tổng hợp
(tỉ lệ nhãn), không nhìn vào văn bản, để kiểm tra hai tập cùng phân phối.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from absa.config import ALL_ASPECTS, ID_TO_POLARITY, NONE_ID, OTHERS_ASPECT, SENTIMENT_ASPECTS
from absa.data.labels import ASPECT_INDEX, to_aspect_matrix
from absa.preprocessing.lexicons import Lexicons
from absa.preprocessing.normalizers import EMOJI_PATTERN, emoji_category

# --------------------------------------------------------------------------- #
# Tổng quan & nhãn
# --------------------------------------------------------------------------- #


def dataset_overview(splits: Mapping[str, object]) -> pd.DataFrame:
    """Bảng tổng quan từng tập: số mẫu, độ dài, số khía cạnh trung bình, trùng lặp.

    Args:
        splits: ``{"train": ABSASplit, "dev": ABSASplit, "test": None}``; tập ``None``
            được ghi nhận là ẩn.
    """
    rows = []
    for name, split in splits.items():
        if split is None:
            rows.append({"split": name, "status": "hidden"})
            continue
        words = pd.Series([len(t.split()) for t in split.texts])
        n_aspects = to_aspect_matrix(split.labels).sum(axis=1)
        rows.append(
            {
                "split": name,
                "status": "available",
                "n_comments": len(split),
                "words_mean": round(words.mean(), 1),
                "words_median": int(words.median()),
                "words_p95": int(words.quantile(0.95)),
                "words_max": int(words.max()),
                "aspects_per_comment": round(float(n_aspects.mean()), 2),
                "duplicate_comments": int(pd.Series(split.texts).duplicated().sum()),
                "empty_comments": int(sum(not t.strip() for t in split.texts)),
            }
        )
    return pd.DataFrame(rows).set_index("split")


def aspect_polarity_counts(Y: np.ndarray) -> pd.DataFrame:
    """Đếm số lần mỗi khía cạnh xuất hiện theo từng cảm xúc.

    Returns:
        DataFrame index = khía cạnh, cột ``Negative/Neutral/Positive/Present/total/
        pct_comments`` (``Present`` chỉ dùng cho OTHERS), sắp xếp giảm dần theo ``total``.
    """
    Y = np.asarray(Y)
    rows = []
    for aspect, j in ASPECT_INDEX.items():
        col = Y[:, j]
        row = {"aspect": aspect}
        if aspect == OTHERS_ASPECT:
            row.update(
                {"Negative": 0, "Neutral": 0, "Positive": 0, "Present": int((col > 0).sum())}
            )
        else:
            row.update({ID_TO_POLARITY[k]: int((col == k).sum()) for k in ID_TO_POLARITY})
            row["Present"] = 0
        row["total"] = int((col != NONE_ID).sum())
        row["pct_comments"] = round(100 * row["total"] / len(Y), 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("aspect").sort_values("total", ascending=False)


def labels_per_comment(Y: np.ndarray) -> pd.Series:
    """Phân phối số khía cạnh trên mỗi bình luận."""
    counts = to_aspect_matrix(Y).sum(axis=1)
    return pd.Series(counts).value_counts().sort_index().rename("n_comments")


def aspect_cooccurrence(Y: np.ndarray, conditional: bool = True) -> pd.DataFrame:
    """Ma trận đồng xuất hiện giữa các khía cạnh.

    Args:
        Y: Ma trận nhãn.
        conditional: ``True`` trả về P(cột | hàng) – tỉ lệ bình luận nhắc khía cạnh
            hàng *cũng* nhắc khía cạnh cột; ``False`` trả về số đếm thô.
    """
    A = to_aspect_matrix(Y).astype(int)
    co = A.T @ A
    if conditional:
        diag = np.diag(co).astype(float)
        co = np.divide(
            co, diag[:, None], out=np.zeros_like(co, dtype=float), where=diag[:, None] > 0
        )
    return pd.DataFrame(co, index=ALL_ASPECTS, columns=ALL_ASPECTS)


def mixed_sentiment_rate(Y: np.ndarray) -> dict[str, float]:
    """Tỉ lệ bình luận chứa **đồng thời** khía cạnh tích cực và tiêu cực.

    Đây là bằng chứng định lượng cho việc phải làm ABSA thay vì phân loại cảm xúc
    cả câu: với các bình luận này, một nhãn chung sẽ làm mất thông tin.
    """
    S = np.asarray(Y)[:, : len(SENTIMENT_ASPECTS)]
    has_pos = (S == 3).any(axis=1)
    has_neg = (S == 1).any(axis=1)
    n_pol = (S > 0).sum(axis=1)
    return {
        "pct_mixed_pos_neg": round(100 * float((has_pos & has_neg).mean()), 2),
        "pct_multi_aspect": round(100 * float((n_pol >= 2).mean()), 2),
        "pct_single_polarity_multi_aspect": round(
            100 * float(((n_pol >= 2) & ~(has_pos & has_neg)).mean()), 2
        ),
    }


def star_polarity_crosstab(stars: Sequence[int], Y: np.ndarray) -> pd.DataFrame:
    """Tỉ lệ cảm xúc của các khía cạnh được nhắc theo từng mức sao.

    Cho thấy số sao chỉ phản ánh *xu hướng chung*: bình luận 5 sao vẫn có khía cạnh
    tiêu cực – lý do không thể dùng số sao làm nhãn khía cạnh.
    """
    S = np.asarray(Y)[:, : len(SENTIMENT_ASPECTS)]
    rows = []
    for star in sorted(set(stars)):
        mask = np.asarray(stars) == star
        values = S[mask].ravel()
        values = values[values > 0]
        total = max(len(values), 1)
        rows.append(
            {
                "n_star": star,
                "n_comments": int(mask.sum()),
                **{
                    ID_TO_POLARITY[k]: round(100 * (values == k).sum() / total, 1)
                    for k in (1, 2, 3)
                },
            }
        )
    return pd.DataFrame(rows).set_index("n_star")


def label_distribution_shift(
    Y_by_split: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """So sánh tỉ lệ (%) bình luận nhắc tới từng khía cạnh giữa các tập.

    Chỉ dùng số liệu tổng hợp của nhãn – **không đọc văn bản** – nên an toàn để áp
    dụng cho Dev mà không gây rò rỉ thông tin vào thiết kế tiền xử lý.
    """
    data = {
        name: 100 * to_aspect_matrix(Y).mean(axis=0)
        for name, Y in Y_by_split.items()
        if Y is not None
    }
    return pd.DataFrame(data, index=ALL_ASPECTS).round(2)


# --------------------------------------------------------------------------- #
# Độ dài & xu hướng viết
# --------------------------------------------------------------------------- #


def text_length_frame(texts: Sequence[str]) -> pd.DataFrame:
    """Số ký tự và số từ (theo khoảng trắng) của từng bình luận."""
    return pd.DataFrame(
        {"n_chars": [len(t) for t in texts], "n_words": [len(t.split()) for t in texts]}
    )


_VN_DIACRITIC_CHARS = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")
_WORD_RE = re.compile(r"[^\W\d_]+")
PATTERNS: dict[str, re.Pattern[str]] = {
    "emoticon": re.compile(r"(?<![\w:/])[:;=]['\-^]?[)(\]\[dDpPvV3]+|\^\^|<3|[tT]_[tT]"),
    "char_elongation": re.compile(r"([^\W\d_])\1{2,}"),
    "laughter_text": re.compile(r"(?i)\b(?:(?:ha|he|hi|hj){2,}\w*|k{3,})\b"),
    "repeated_punct": re.compile(r"([!?.])\1+"),
    "line_break": re.compile(r"[\r\n]"),
    "number_with_unit": re.compile(
        r"(?i)\d+(?:[.,]\d+)*\s*(?:tr|triệu|củ|k|đ|mah|gb|g|mp|hz|fps|w|%)(?!\w)"
    ),
    "any_digit": re.compile(r"\d"),
    "url": re.compile(r"(?i)https?://|www\."),
    "allcaps_word": re.compile(r"\b[A-ZÀ-Ỹ]{4,}\b"),
}


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(unicodedata.normalize("NFC", text).lower())


def text_pattern_report(texts: Sequence[str], lexicons: Lexicons | None = None) -> pd.DataFrame:
    """Tỉ lệ bình luận chứa từng *xu hướng viết* – căn cứ để thiết kế tiền xử lý.

    Mỗi dòng của bảng ứng với một luật tiền xử lý trong
    :mod:`absa.preprocessing.normalizers`, nên bảng này đồng thời là phần "motivation"
    cho báo cáo.
    """
    n = len(texts)
    rows = [
        {"pattern": name, "n_comments": sum(bool(p.search(t)) for t in texts)}
        for name, p in PATTERNS.items()
    ]
    rows.append(
        {"pattern": "emoji", "n_comments": sum(bool(EMOJI_PATTERN.search(t)) for t in texts)}
    )
    rows.append(
        {
            "pattern": "not_unicode_nfc",
            "n_comments": sum(unicodedata.normalize("NFC", t) != t for t in texts),
        }
    )
    no_diacritic = 0
    teencode_hits = 0
    for t in texts:
        words = _words(t)
        if words and sum(bool(set(w) & _VN_DIACRITIC_CHARS) for w in words) / len(words) < 0.1:
            no_diacritic += 1
        if lexicons is not None and any(w in lexicons.teencode for w in words):
            teencode_hits += 1
    rows.append({"pattern": "mostly_no_diacritics", "n_comments": no_diacritic})
    if lexicons is not None:
        rows.append({"pattern": "teencode_in_lexicon", "n_comments": teencode_hits})
    frame = pd.DataFrame(rows)
    frame["pct_comments"] = (100 * frame["n_comments"] / max(n, 1)).round(2)
    return frame.sort_values("pct_comments", ascending=False).reset_index(drop=True)


def emoji_frequency(
    texts: Sequence[str], lexicons: Lexicons | None = None, top_n: int = 30
) -> pd.DataFrame:
    """Tần suất emoji và nhóm cảm xúc hệ thống gán cho chúng."""
    overrides = lexicons.emoji_sentiment if lexicons else None
    counter = Counter(ch for t in texts for ch in EMOJI_PATTERN.findall(t))
    rows = [
        {
            "emoji": ch,
            "count": c,
            "unicode_name": unicodedata.name(ch, "?"),
            "category": emoji_category(ch, overrides) or "dropped",
        }
        for ch, c in counter.most_common(top_n)
    ]
    return pd.DataFrame(rows)


# --- Kiểm tra âm tiết tiếng Việt (heuristic) -------------------------------- #
_ONSETS = "ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx]"
_RIMES = (
    "uyen|uyet|uych|uynh|oang|oanh|oach|uong|uoc|uon|uot|uom|uoi|ieng|ieu|iec|iem|ien|iep|iet|"
    "yeu|yen|yet|yem|uya|uyu|oai|oay|oeo|uay|uou|ai|ao|au|ay|eo|eu|ia|ie|iu|oa|oe|oi|oo|ua|ue|"
    "ui|uo|uu|uy|ye|a|e|i|o|u|y"
)
_SYLLABLE_RE = re.compile(rf"^(?:{_ONSETS})?(?:{_RIMES})(?:ch|ng|nh|c|m|n|p|t)?$")


def is_vietnamese_syllable(token: str) -> bool:
    """Kiểm tra (gần đúng) token có cấu trúc một âm tiết tiếng Việt hay không.

    Bỏ dấu rồi so khớp với mẫu *âm đầu + vần + âm cuối*. Token không khớp
    (``dc``, ``mk``, ``vs``, ``ok``) thường là teencode, viết tắt hoặc từ nước ngoài.
    Đây là heuristic để *phát hiện* xu hướng viết, không dùng để sửa văn bản.
    """
    base = unicodedata.normalize("NFD", token.lower().replace("đ", "d"))
    base = "".join(c for c in base if unicodedata.category(c) != "Mn")
    return bool(_SYLLABLE_RE.match(base))


def nonstandard_token_report(
    texts: Sequence[str], lexicons: Lexicons | None = None, top_n: int = 60, min_count: int = 5
) -> pd.DataFrame:
    """Liệt kê token phổ biến *không* có cấu trúc âm tiết tiếng Việt.

    Dùng để đánh giá độ phủ của từ điển teencode một cách có hệ thống: token nào xuất
    hiện nhiều, không phải âm tiết chuẩn và chưa có trong từ điển là ứng viên cần xem
    xét (có thể là thuật ngữ như ``camera``/``wifi`` – giữ nguyên – hoặc viết tắt mới).
    """
    counter = Counter(w for t in texts for w in _words(t))
    teencode = lexicons.teencode if lexicons else {}
    rows = [
        {"token": w, "count": c, "in_teencode_lexicon": w in teencode}
        for w, c in counter.most_common()
        if c >= min_count and not is_vietnamese_syllable(w)
    ]
    return pd.DataFrame(rows[:top_n])


def nonstandard_token_rate(texts: Sequence[str]) -> float:
    """Tỉ lệ (%) token không phải âm tiết tiếng Việt chuẩn trên toàn bộ token."""
    words = [w for t in texts for w in _words(t)]
    if not words:
        return 0.0
    return round(100 * sum(not is_vietnamese_syllable(w) for w in words) / len(words), 2)


# --------------------------------------------------------------------------- #
# Từ đặc trưng theo nhãn
# --------------------------------------------------------------------------- #


def top_terms_by_aspect(
    processed_texts: Sequence[str],
    Y: np.ndarray,
    aspect: str,
    top_n: int = 15,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 3,
) -> pd.DataFrame:
    """Từ/cụm từ gắn chặt nhất với từng cảm xúc của một khía cạnh (theo chi-square).

    Với mỗi lớp (``Negative``/``Neutral``/``Positive`` hoặc ``Present``), so sánh văn bản
    thuộc lớp đó với phần còn lại. Dùng văn bản **đã tiền xử lý** để thấy tác dụng của
    tách từ và gắn phủ định (vd ``không_mượt`` nổi lên ở PERFORMANCE#Negative).

    Returns:
        DataFrame cột ``label, term, chi2, count_in_label``.
    """
    j = ASPECT_INDEX[aspect]
    y = np.asarray(Y)[:, j]
    vectorizer = CountVectorizer(
        token_pattern=r"\S+", lowercase=False, ngram_range=ngram_range, min_df=min_df, binary=True
    )
    X = vectorizer.fit_transform(processed_texts)
    vocab = np.array(vectorizer.get_feature_names_out())
    classes = [1] if aspect == OTHERS_ASPECT else [1, 2, 3]
    rows = []
    for k in classes:
        target = (y == k).astype(int)
        if target.sum() == 0:
            continue
        scores, _ = chi2(X, target)
        scores = np.nan_to_num(scores)
        in_label = np.asarray(X[target == 1].sum(axis=0)).ravel()
        in_rest = np.asarray(X[target == 0].sum(axis=0)).ravel()
        # Chỉ giữ term xuất hiện nhiều hơn kỳ vọng trong lớp (chi2 đối xứng hai chiều).
        over = in_label / max(target.sum(), 1) > in_rest / max((1 - target).sum(), 1)
        order = [i for i in np.argsort(-scores) if over[i]][:top_n]
        label = "Present" if aspect == OTHERS_ASPECT else ID_TO_POLARITY[k]
        rows += [
            {
                "label": label,
                "term": vocab[i],
                "chi2": float(scores[i]),
                "count_in_label": int(in_label[i]),
            }
            for i in order
        ]
    return pd.DataFrame(rows)


def preprocessing_impact(raw_texts: Sequence[str], processed_texts: Sequence[str]) -> pd.DataFrame:
    """So sánh từ vựng trước/sau tiền xử lý: kích thước, số token hiếm, độ dài.

    Tiền xử lý tốt sẽ *giảm* số dạng từ hiếm (hapax) do biến thể chính tả gây ra mà
    không làm mất thông tin.
    """

    def _stats(texts: Sequence[str], tokenizer) -> dict[str, float]:
        tokens = [tok for t in texts for tok in tokenizer(t)]
        counts = Counter(tokens)
        return {
            "n_tokens": len(tokens),
            "vocab_size": len(counts),
            "hapax": sum(1 for c in counts.values() if c == 1),
            "tokens_per_comment": round(len(tokens) / max(len(texts), 1), 2),
        }

    raw = _stats(raw_texts, lambda t: t.lower().split())
    processed = _stats(processed_texts, str.split)
    frame = pd.DataFrame({"raw (lower+split)": raw, "processed": processed})
    frame["change_%"] = (100 * (frame["processed"] - frame.iloc[:, 0]) / frame.iloc[:, 0]).round(1)
    return frame
