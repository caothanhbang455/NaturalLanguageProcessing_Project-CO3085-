r"""Biểu diễn thưa (sparse): Bag-of-Words và TF-IDF ở mức từ và ký tự.

Đầu vào là văn bản **đã tiền xử lý** (profile ``classic``): token cách nhau bởi dấu
cách, từ ghép nối bằng ``_``. Vì vậy dùng ``token_pattern=r"\S+"`` để giữ nguyên
token như ``màn_hình``, ``không_mượt``, ``numbattery`` và cả token 1 ký tự.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion

WHITESPACE_TOKEN_PATTERN = r"\S+"


def make_bow(
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    binary: bool = False,
    max_features: int | None = None,
) -> CountVectorizer:
    """Bag-of-Words: đếm số lần xuất hiện của từng n-gram.

    Args:
        ngram_range: Uni+bi-gram mặc định để bắt cụm ``pin trâu``, ``giá rẻ``.
        min_df: Bỏ n-gram xuất hiện ở ít hơn ``min_df`` bình luận (nhiễu, lỗi gõ).
        max_df: Bỏ n-gram quá phổ biến (đóng vai trò stopword tự động theo dữ liệu).
        binary: ``True`` chỉ ghi nhận có/không (hợp với câu ngắn).
        max_features: Giới hạn kích thước từ vựng.
    """
    return CountVectorizer(
        token_pattern=WHITESPACE_TOKEN_PATTERN,
        lowercase=False,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        binary=binary,
        max_features=max_features,
    )


def make_tfidf_word(
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    max_features: int | None = None,
) -> TfidfVectorizer:
    """TF-IDF mức từ: trọng số cao cho từ đặc trưng, thấp cho từ xuất hiện khắp nơi.

    ``sublinear_tf=True`` dùng ``1 + log(tf)`` để một từ lặp nhiều lần trong cùng
    bình luận không lấn át.
    """
    return TfidfVectorizer(
        token_pattern=WHITESPACE_TOKEN_PATTERN,
        lowercase=False,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
    )


def make_tfidf_char(
    ngram_range: tuple[int, int] = (2, 5),
    min_df: int = 3,
    sublinear_tf: bool = True,
    max_features: int | None = 150_000,
    analyzer: str = "char_wb",
) -> TfidfVectorizer:
    """TF-IDF n-gram ký tự.

    Bền vững với lỗi chính tả, teencode chưa có trong từ điển và biến thể dấu, vì
    ``mượtt`` và ``mượt`` vẫn chia sẻ phần lớn n-gram ký tự.

    Args:
        ngram_range: Độ dài n-gram ký tự.
        min_df: Bỏ n-gram xuất hiện ở ít hơn ``min_df`` bình luận.
        sublinear_tf: Dùng ``1 + log(tf)``.
        max_features: Giới hạn kích thước từ vựng.
        analyzer: ``"char_wb"`` chỉ lấy n-gram *bên trong ranh giới từ*; ``"char"`` lấy cả
            n-gram **vắt qua dấu cách** (``"pin trâu"`` cho ra cả ``"n tr"``), nên bắt được
            cụm hai từ ở mức ký tự.
    """
    return TfidfVectorizer(
        analyzer=analyzer,
        lowercase=False,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
    )


def make_tfidf_word_char() -> FeatureUnion:
    """Ghép TF-IDF từ (1–2 gram) và TF-IDF ký tự (2–5 gram).

    Kết hợp ngữ nghĩa mức từ với độ bền chính tả mức ký tự – baseline thưa mạnh và
    phổ biến cho phân loại văn bản.
    """
    return FeatureUnion([("word", make_tfidf_word()), ("char", make_tfidf_char())])
