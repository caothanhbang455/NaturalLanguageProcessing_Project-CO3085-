"""Danh mục N cách biểu diễn văn bản (feature registry).

Mỗi biểu diễn được mô tả bởi một :class:`FeatureSpec`: cách tạo, profile tiền xử lý
cần dùng, kiểu đầu ra (thưa/đặc) và một đoạn tóm tắt ý tưởng. Thêm biểu diễn mới chỉ
cần thêm một mục vào ``FEATURE_REGISTRY`` – lưới thí nghiệm M x N tự động nhận.

Chi tiết ý tưởng từng biểu diễn: ``docs/feature_representations.md``.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion

from absa.features.contextual import TransformerEmbeddingVectorizer
from absa.features.sparse import make_bow, make_tfidf_char, make_tfidf_word, make_tfidf_word_char
from absa.features.static_embeddings import (
    Doc2VecVectorizer,
    GensimWordEmbeddingVectorizer,
    PretrainedFastTextVectorizer,
)

FAMILIES = ("sparse", "static", "contextual", "hybrid")
FASTTEXT_VI_ENV = "ABSA_FASTTEXT_VI_PATH"  # Đường dẫn tới cc.vi.300.bin (tùy chọn).


@dataclass(frozen=True)
class FeatureSpec:
    """Mô tả một cách biểu diễn văn bản.

    Attributes:
        name: Khóa duy nhất trong registry.
        family: ``"sparse" | "static" | "contextual" | "hybrid"``.
        profile: Profile tiền xử lý đầu vào (xem :data:`PREPROCESS_PROFILES`).
        output: ``"sparse"`` (ma trận thưa scipy) hoặc ``"dense"`` (numpy).
        non_negative: Giá trị đặc trưng luôn >= 0 (điều kiện của Multinomial/Complement NB).
        factory: Hàm tạo transformer *chưa fit*.
        summary: Tóm tắt ý tưởng (tiếng Việt) – hiển thị trong notebook.
        reference: Công bố gốc.
        heavy: Tốn tài nguyên (mô hình lớn) – mặc định bỏ qua trên CPU.
        requires: Gói Python bắt buộc.
    """

    name: str
    family: str
    profile: str
    output: str
    non_negative: bool
    factory: Callable[[], TransformerMixin]
    summary: str
    reference: str
    heavy: bool = False
    requires: tuple[str, ...] = ()

    def is_available(self) -> tuple[bool, str]:
        """Kiểm tra đủ điều kiện chạy (đã cài gói, có file mô hình nếu cần)."""
        missing = [p for p in self.requires if importlib.util.find_spec(p) is None]
        if missing:
            return False, f"thiếu gói {missing}"
        if self.name == "fasttext_cc_vi" and not os.environ.get(FASTTEXT_VI_ENV):
            return False, f"chưa đặt biến môi trường {FASTTEXT_VI_ENV}"
        return True, ""


def _st(model: str, **kwargs) -> Callable[[], TransformerEmbeddingVectorizer]:
    return lambda: TransformerEmbeddingVectorizer(model, backend="sentence_transformers", **kwargs)


def _hf(model: str, **kwargs) -> Callable[[], TransformerEmbeddingVectorizer]:
    return lambda: TransformerEmbeddingVectorizer(model, backend="hf_mean", **kwargs)


_ST = ("sentence_transformers",)
_HF = ("transformers", "torch")

_SPECS: list[FeatureSpec] = [
    # ------------------------------------------------------------- sparse --
    FeatureSpec(
        "bow",
        "sparse",
        "classic",
        "sparse",
        True,
        make_bow,
        "Đếm tần suất uni+bi-gram từ. Đơn giản, dễ giải thích, là mốc so sánh cơ bản.",
        "Harris (1954); Manning et al. (2008), Introduction to IR",
    ),
    FeatureSpec(
        "tfidf_word",
        "sparse",
        "classic",
        "sparse",
        True,
        make_tfidf_word,
        "TF-IDF uni+bi-gram từ (sublinear tf): giảm trọng số từ phổ biến, tăng từ đặc trưng.",
        "Sparck Jones (1972); Salton & Buckley (1988)",
    ),
    FeatureSpec(
        "tfidf_char",
        "sparse",
        "classic",
        "sparse",
        True,
        make_tfidf_char,
        "TF-IDF n-gram ký tự 2–5 trong ranh giới từ: bền với lỗi gõ, teencode, biến thể dấu.",
        "Cavnar & Trenkle (1994); Kanaris et al. (2007)",
    ),
    FeatureSpec(
        "tfidf_word_char",
        "sparse",
        "classic",
        "sparse",
        True,
        make_tfidf_word_char,
        "Ghép TF-IDF từ và ký tự: ngữ nghĩa mức từ + độ bền chính tả mức ký tự.",
        "Wang & Manning (2012) – baseline n-gram mạnh cho phân loại cảm xúc",
    ),
    # Các biến thể sparse bổ sung: thay đổi từng yếu tố (trọng số, độ dài n-gram, ranh giới
    # ký tự) so với 4 biểu diễn lõi để xem yếu tố nào quan trọng.
    FeatureSpec(
        "bow_binary",
        "sparse",
        "classic",
        "sparse",
        True,
        lambda: make_bow(binary=True),
        "BoW nhị phân uni+bi-gram: chỉ ghi có/không, không đếm số lần (hợp với câu ngắn).",
        "Wang & Manning (2012); Pang et al. (2002) – BoW nhị phân cho phân loại cảm xúc",
    ),
    FeatureSpec(
        "tfidf_word_uni",
        "sparse",
        "classic",
        "sparse",
        True,
        lambda: make_tfidf_word(ngram_range=(1, 1)),
        "TF-IDF chỉ unigram: từ vựng nhỏ gọn, không có cụm từ (đối chứng đo giá trị của bigram).",
        "Salton & Buckley (1988)",
    ),
    FeatureSpec(
        "tfidf_word_13",
        "sparse",
        "classic",
        "sparse",
        True,
        lambda: make_tfidf_word(ngram_range=(1, 3)),
        "TF-IDF uni+bi+trigram: bắt được cụm dài như 'không tốt lắm', đổi lại từ vựng thưa hơn.",
        "Wang & Manning (2012)",
    ),
    FeatureSpec(
        "tfidf_char_cross",
        "sparse",
        "classic",
        "sparse",
        True,
        lambda: make_tfidf_char(analyzer="char", ngram_range=(3, 6)),
        "TF-IDF n-gram ký tự 3–6 **vắt qua dấu cách**: bắt cụm hai từ ở mức ký tự.",
        "Cavnar & Trenkle (1994); Kanaris et al. (2007)",
    ),
    # ------------------------------------------------------------- static --
    FeatureSpec(
        "word2vec",
        "static",
        "classic",
        "dense",
        False,
        lambda: GensimWordEmbeddingVectorizer("word2vec"),
        "Word2Vec skip-gram 300d học trên train; vector câu = trung bình có trọng số IDF.",
        "Mikolov et al. (2013), NeurIPS",
        requires=("gensim",),
    ),
    FeatureSpec(
        "fasttext",
        "static",
        "classic",
        "dense",
        False,
        lambda: GensimWordEmbeddingVectorizer("fasttext"),
        "FastText 300d học trên train, thêm n-gram ký tự nên có vector cho từ chưa gặp.",
        "Bojanowski et al. (2017), TACL",
        requires=("gensim",),
    ),
    FeatureSpec(
        "doc2vec",
        "static",
        "classic",
        "dense",
        False,
        Doc2VecVectorizer,
        "Doc2Vec PV-DBOW: học trực tiếp vector cho cả văn bản thay vì gộp vector từ.",
        "Le & Mikolov (2014), ICML",
        requires=("gensim",),
    ),
    FeatureSpec(
        "fasttext_cc_vi",
        "static",
        "transformer_raw",
        "dense",
        False,
        lambda: PretrainedFastTextVectorizer(os.environ.get(FASTTEXT_VI_ENV, "")),
        "FastText pretrained trên Common Crawl tiếng Việt (cc.vi.300); cần tải file ~7GB.",
        "Grave et al. (2018), LREC",
        heavy=True,
        requires=("gensim",),
    ),
    # --------------------------------------------------------- contextual --
    FeatureSpec(
        "phobert",
        "contextual",
        "transformer_seg",
        "dense",
        False,
        _hf("vinai/phobert-base-v2", max_length=256),
        "PhoBERT-base-v2 (RoBERTa tiếng Việt, 20GB văn bản đã tách từ); mean pooling lớp cuối.",
        "Nguyen & Nguyen (2020), Findings of EMNLP",
        requires=_HF,
    ),
    FeatureSpec(
        "visobert",
        "contextual",
        "transformer_raw",
        "dense",
        False,
        _hf("uitnlp/visobert", max_length=256),
        "ViSoBERT: pretrain trên văn bản mạng xã hội tiếng Việt (teencode, emoji) – sát domain.",
        "Nguyen et al. (2023), EMNLP",
        requires=_HF,
    ),
    FeatureSpec(
        "sbert_multilingual",
        "contextual",
        "transformer_raw",
        "dense",
        False,
        _st("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
        "Sentence-BERT đa ngữ (50+ ngôn ngữ) huấn luyện bằng knowledge distillation.",
        "Reimers & Gurevych (2019, 2020), EMNLP",
        requires=_ST,
    ),
    FeatureSpec(
        "vietnamese_sbert",
        "contextual",
        "transformer_seg",
        "dense",
        False,
        _st("bkai-foundation-models/vietnamese-bi-encoder", max_length=256),
        "Bi-encoder tiếng Việt (nền PhoBERT) của BKAI-HUST, fine-tune contrastive cho câu.",
        "BKAI Foundation Models (2023); Nguyen & Nguyen (2020)",
        requires=_ST,
    ),
    FeatureSpec(
        "multilingual_e5",
        "contextual",
        "transformer_raw",
        "dense",
        False,
        _st("intfloat/multilingual-e5-base", prefix="query: "),
        "multilingual-E5-base: pretrain contrastive yếu giám sát trên 1 tỉ cặp câu đa ngữ.",
        "Wang et al. (2024), arXiv:2402.05672",
        requires=_ST,
    ),
    FeatureSpec(
        "gte_multilingual",
        "contextual",
        "transformer_raw",
        "dense",
        False,
        _st("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True),
        "GTE-multilingual-base: encoder 305M, 70+ ngôn ngữ, ngữ cảnh dài 8192 token.",
        "Zhang et al. (2024), mGTE, EMNLP Industry",
        requires=_ST,
    ),
    FeatureSpec(
        "bge_m3",
        "contextual",
        "transformer_raw",
        "dense",
        False,
        _st("BAAI/bge-m3", batch_size=16),
        "BGE-M3 (568M, XLM-R large): đa ngữ, đa độ dài, top đầu MMTEB nhóm mã nguồn mở.",
        "Chen et al. (2024), Findings of ACL",
        heavy=True,
        requires=_ST,
    ),
    FeatureSpec(
        "vietnamese_embedding",
        "contextual",
        "transformer_raw",
        "dense",
        False,
        _st("AITeamVN/Vietnamese_Embedding", batch_size=16),
        "BGE-M3 fine-tune thêm trên ~300k cặp tiếng Việt – embedding tiếng Việt mới (2025).",
        "AITeamVN (2025); Chen et al. (2024)",
        heavy=True,
        requires=_ST,
    ),
    FeatureSpec(
        "qwen3_embedding",
        "contextual",
        "transformer_raw",
        "dense",
        False,
        _st("Qwen/Qwen3-Embedding-0.6B", batch_size=8),
        "Qwen3-Embedding-0.6B: embedding dựa trên LLM (2025), dẫn đầu MMTEB ở cỡ nhỏ.",
        "Zhang et al. (2025), arXiv:2506.05176",
        heavy=True,
        requires=_ST,
    ),
    # ------------------------------------------------------------- hybrid --
    FeatureSpec(
        "hybrid_tfidf_e5",
        "hybrid",
        "transformer_raw",
        "sparse",
        False,
        lambda: FeatureUnion(
            [
                ("tfidf", make_tfidf_word_char()),
                ("e5", _st("intfloat/multilingual-e5-base", prefix="query: ")()),
            ]
        ),
        "Ghép TF-IDF (bắt từ khóa khía cạnh) với E5 (ngữ nghĩa câu); cả hai đã chuẩn hóa L2.",
        "Kết hợp sparse + dense, tương tự hybrid retrieval (Luan et al., 2021, TACL)",
        requires=_ST,
    ),
]

FEATURE_REGISTRY: dict[str, FeatureSpec] = {spec.name: spec for spec in _SPECS}


def get_feature_spec(name: str) -> FeatureSpec:
    """Lấy :class:`FeatureSpec` theo tên (báo lỗi kèm danh sách tên hợp lệ)."""
    if name not in FEATURE_REGISTRY:
        raise KeyError(f"Biểu diễn '{name}' không tồn tại. Có: {list(FEATURE_REGISTRY)}")
    return FEATURE_REGISTRY[name]


def build_feature(name: str) -> TransformerMixin:
    """Tạo transformer *chưa fit* cho biểu diễn ``name``."""
    return get_feature_spec(name).factory()


def list_features(
    families: Iterable[str] | None = None, include_heavy: bool = False, only_available: bool = True
) -> list[str]:
    """Liệt kê tên biểu diễn theo họ, có thể lọc mô hình nặng/không khả dụng."""
    families = set(families or FAMILIES)
    names = []
    for spec in FEATURE_REGISTRY.values():
        if spec.family not in families or (spec.heavy and not include_heavy):
            continue
        if only_available and not spec.is_available()[0]:
            continue
        names.append(spec.name)
    return names


def feature_catalog() -> pd.DataFrame:
    """Bảng mô tả toàn bộ biểu diễn – dùng làm phần giới thiệu trong notebook/báo cáo."""
    rows = []
    for spec in FEATURE_REGISTRY.values():
        available, reason = spec.is_available()
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "input_profile": spec.profile,
                "output": spec.output,
                "heavy": spec.heavy,
                "available": available if available else f"no ({reason})",
                "idea": spec.summary,
                "reference": spec.reference,
            }
        )
    return pd.DataFrame(rows).set_index("name")
