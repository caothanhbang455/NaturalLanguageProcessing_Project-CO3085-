"""Embedding tĩnh (static): Word2Vec, FastText, Doc2Vec huấn luyện trên tập train.

"Tĩnh" nghĩa là mỗi từ có **một** vector cố định bất kể ngữ cảnh. Vector câu được
tạo bằng cách gộp (pooling) vector các từ. Các lớp ở đây theo chuẩn scikit-learn:
``fit`` chỉ học trên dữ liệu được đưa vào (train) nên không rò rỉ thông tin dev/test.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from absa.config import SEED

POOLING_MODES = ("mean", "tfidf", "mean_max")


def _tokenize(texts: Iterable[str]) -> list[list[str]]:
    return [t.split() for t in texts]


def _default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))


class GensimWordEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Học Word2Vec/FastText trên corpus train rồi gộp thành vector câu.

    Args:
        algorithm: ``"word2vec"`` (Mikolov et al., 2013) hoặc ``"fasttext"``
            (Bojanowski et al., 2017 – có subword n-gram nên tạo được vector cho từ
            chưa gặp như teencode mới).
        vector_size: Số chiều embedding.
        window: Kích thước cửa sổ ngữ cảnh.
        min_count: Bỏ từ xuất hiện ít hơn ngưỡng này.
        sg: ``1`` = Skip-gram (tốt cho từ hiếm), ``0`` = CBOW.
        epochs: Số vòng huấn luyện (corpus nhỏ nên cần nhiều vòng).
        pooling: ``"mean"`` (trung bình), ``"tfidf"`` (trung bình có trọng số IDF – nhấn
            mạnh từ đặc trưng) hoặc ``"mean_max"`` (nối trung bình và max theo chiều).
        min_n, max_n: Độ dài n-gram ký tự của FastText.
        workers: Số luồng; đặt ``1`` nếu cần tái lập tuyệt đối.
        seed: Seed ngẫu nhiên.
    """

    def __init__(
        self,
        algorithm: str = "word2vec",
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 2,
        sg: int = 1,
        epochs: int = 30,
        pooling: str = "tfidf",
        min_n: int = 2,
        max_n: int = 5,
        workers: int | None = None,
        seed: int = SEED,
    ) -> None:
        self.algorithm = algorithm
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.pooling = pooling
        self.min_n = min_n
        self.max_n = max_n
        self.workers = workers
        self.seed = seed

    def fit(self, X: Sequence[str], y=None) -> GensimWordEmbeddingVectorizer:
        """Huấn luyện embedding trên ``X`` (chỉ tập train)."""
        if self.pooling not in POOLING_MODES:
            raise ValueError(f"pooling phải thuộc {POOLING_MODES}")
        from gensim.models import FastText, Word2Vec

        sentences = _tokenize(X)
        common = dict(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            epochs=self.epochs,
            workers=self.workers or _default_workers(),
            seed=self.seed,
        )
        if self.algorithm == "word2vec":
            model = Word2Vec(**common)
        elif self.algorithm == "fasttext":
            model = FastText(min_n=self.min_n, max_n=self.max_n, **common)
        else:
            raise ValueError("algorithm phải là 'word2vec' hoặc 'fasttext'")
        self.keyed_vectors_ = model.wv
        if self.pooling == "tfidf":
            idf = TfidfVectorizer(token_pattern=r"\S+", lowercase=False).fit(X)
            self.idf_ = dict(zip(idf.get_feature_names_out(), idf.idf_))
            self.default_idf_ = float(max(idf.idf_)) if len(idf.idf_) else 1.0
        return self

    def _has_vector(self, token: str) -> bool:
        # FastText tạo được vector cho mọi từ nhờ subword; Word2Vec chỉ cho từ đã gặp.
        return self.algorithm == "fasttext" or token in self.keyed_vectors_.key_to_index

    def _embed(self, tokens: list[str]) -> np.ndarray:
        dim = self.vector_size
        tokens = [t for t in tokens if self._has_vector(t)]
        if not tokens:
            return np.zeros(dim * (2 if self.pooling == "mean_max" else 1), dtype=np.float32)
        vectors = np.vstack([self.keyed_vectors_[t] for t in tokens])
        if self.pooling == "mean":
            return vectors.mean(axis=0)
        if self.pooling == "mean_max":
            return np.concatenate([vectors.mean(axis=0), vectors.max(axis=0)])
        weights = np.array([self.idf_.get(t, self.default_idf_) for t in tokens])
        return (weights[:, None] * vectors).sum(axis=0) / weights.sum()

    def transform(self, X: Sequence[str]) -> np.ndarray:
        """Vector câu cho từng văn bản, kích thước ``(n, vector_size)`` (hoặc gấp đôi)."""
        return np.vstack([self._embed(tokens) for tokens in _tokenize(X)]).astype(np.float32)


class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    """Doc2Vec PV-DBOW (Le & Mikolov, 2014): học trực tiếp vector cho cả văn bản.

    Khi ``transform``, vector được *suy luận* (``infer_vector``) cho mọi văn bản – kể
    cả train – để train và dev được biểu diễn theo cùng một cách.
    """

    def __init__(
        self,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 40,
        dbow_words: int = 1,
        workers: int | None = None,
        seed: int = SEED,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.dbow_words = dbow_words
        self.workers = workers
        self.seed = seed

    def fit(self, X: Sequence[str], y=None) -> Doc2VecVectorizer:
        """Huấn luyện Doc2Vec trên ``X``."""
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        documents = [TaggedDocument(t.split(), [i]) for i, t in enumerate(X)]
        self.model_ = Doc2Vec(
            documents,
            dm=0,
            dbow_words=self.dbow_words,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=self.workers or _default_workers(),
            seed=self.seed,
        )
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        """Suy luận vector cho từng văn bản."""
        return np.vstack(
            [self.model_.infer_vector(t.split(), epochs=self.epochs) for t in X]
        ).astype(np.float32)


class PretrainedFastTextVectorizer(BaseEstimator, TransformerMixin):
    """FastText pretrained của Facebook cho tiếng Việt (``cc.vi.300.bin``, Grave et al., 2018).

    Không bắt buộc: file ~7GB, tải tại https://fasttext.cc/docs/en/crawl-vectors.html
    rồi trỏ ``model_path`` tới file. Được huấn luyện trên Common Crawl + Wikipedia ở mức
    **âm tiết**, nên nên dùng văn bản *chưa tách từ* (hoặc tách ``_`` thành dấu cách).

    Args:
        model_path: Đường dẫn file ``.bin``.
        split_compounds: Tách ``màn_hình`` thành ``màn hình`` trước khi tra vector.
    """

    def __init__(self, model_path: str = "", split_compounds: bool = True) -> None:
        self.model_path = model_path
        self.split_compounds = split_compounds

    def fit(self, X: Sequence[str], y=None) -> PretrainedFastTextVectorizer:
        """Nạp vector pretrained (không học gì thêm)."""
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "Cần tải cc.vi.300.bin và truyền model_path – xem docstring của lớp."
            )
        from gensim.models.fasttext import load_facebook_vectors

        self.keyed_vectors_ = load_facebook_vectors(self.model_path)
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        """Trung bình vector các âm tiết của từng văn bản."""
        if not hasattr(self, "keyed_vectors_"):  # Sau khi unpickle: nạp lại từ đĩa.
            self.fit(X)
        out = []
        for text in X:
            if self.split_compounds:
                text = text.replace("_", " ")
            tokens = text.split()
            if not tokens:
                out.append(np.zeros(self.keyed_vectors_.vector_size, dtype=np.float32))
                continue
            out.append(np.mean([self.keyed_vectors_[t] for t in tokens], axis=0))
        return np.vstack(out).astype(np.float32)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("keyed_vectors_", None)  # Không nhét 7GB vào file mô hình.
        return state
