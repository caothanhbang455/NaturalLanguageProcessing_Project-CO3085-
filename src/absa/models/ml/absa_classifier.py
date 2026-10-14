"""Bộ phân loại ABSA bọc quanh một estimator scikit-learn bất kỳ.

Bài toán ABSA trên UIT-ViSFD có đầu ra là ma trận ``Y`` (n, 11): với mỗi khía cạnh,
dự đoán ``None/Negative/Neutral/Positive`` (OTHERS: có/không). Lớp
:class:`ABSAClassifier` biến **bất kỳ** bộ phân loại đơn nhãn nào (LR, SVM, NB, RF,
MLP...) thành bộ phân loại ABSA theo một trong ba chiến lược:

``"multilabel"`` (cách đơn giản nhất: coi bài toán là 31 nhãn nhị phân)
    Mỗi cặp ``ASPECT#Polarity`` (và ``OTHERS``) là **một nhãn nhị phân** – tổng cộng 31 nhãn,
    mỗi nhãn một bộ phân loại độc lập (kỹ thuật *Binary Relevance* của multi-label
    learning). Dự đoán thô là ma trận (n, 31) (xem :meth:`ABSAClassifier.predict_pairs`);
    :meth:`ABSAClassifier.predict` giải mã về ``Y`` (n, 11) bằng cách chọn cảm xúc có điểm
    cao nhất khi một khía cạnh bị bật nhiều cảm xúc. 31 mô hình.

``"joint"``
    Mỗi khía cạnh một bộ phân loại 4 lớp ``{None, Neg, Neu, Pos}``. Phát hiện khía
    cạnh và phân cực được học **cùng lúc**; 11 mô hình.

``"two_stage"``
    Giai đoạn 1 – mỗi khía cạnh một bộ phân loại nhị phân *có nhắc tới không*.
    Giai đoạn 2 – mỗi khía cạnh một bộ phân loại 3 lớp cảm xúc, **chỉ huấn luyện trên
    các bình luận có nhắc tới khía cạnh đó**; khi dự đoán chỉ áp dụng cho khía cạnh
    đã được phát hiện. 21 mô hình.

Cả ba chiến lược đều trả về cùng định dạng ``Y`` nên dùng chung một bộ metric.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier

from absa.config import ALL_ASPECTS, NONE_ID, OTHERS_ASPECT
from absa.data.labels import pairs_to_labels, to_pair_matrix

STRATEGIES = ("multilabel", "joint", "two_stage")
_OTHERS_COL = ALL_ASPECTS.index(OTHERS_ASPECT)


def _fit_one(estimator, X, y):
    """Fit một estimator; nếu ``y`` chỉ có một lớp thì thay bằng bộ dự đoán hằng."""
    if len(np.unique(y)) < 2:
        estimator = DummyClassifier(strategy="most_frequent")
    return estimator.fit(X, y)


def _binary_scores(estimator, X, bits: np.ndarray) -> np.ndarray:
    """Điểm tin cậy "đây là lớp dương" của một bộ phân loại nhị phân.

    Ưu tiên ``decision_function`` (SVM, LR, Ridge...), rồi ``predict_proba`` (NB, kNN,
    cây...), cuối cùng là chính nhãn dự đoán 0/1. Điểm chỉ dùng để phân xử khi một khía
    cạnh bị bật nhiều cảm xúc, nên chỉ cần so sánh được giữa các cột cùng loại mô hình.
    """
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        return scores if scores.ndim == 1 else scores[:, -1]
    if hasattr(estimator, "predict_proba"):
        classes = list(estimator.classes_)
        if 1 not in classes:  # bộ dự đoán hằng của nhãn không bao giờ xuất hiện
            return np.zeros(X.shape[0])
        return np.asarray(estimator.predict_proba(X))[:, classes.index(1)]
    return bits.astype(float)


class ABSAClassifier(BaseEstimator, ClassifierMixin):
    """Biến một estimator đơn nhãn thành bộ phân loại ABSA đa khía cạnh.

    Args:
        estimator: Estimator scikit-learn *chưa fit* (sẽ được ``clone`` cho từng nhãn con).
        strategy: ``"multilabel"``, ``"joint"`` hoặc ``"two_stage"`` – xem docstring module.
        n_jobs: Số tiến trình fit song song giữa các nhãn con (``-1`` = mọi nhân).

    Attributes:
        estimators_: ``multilabel``: list 31 estimator. ``joint``: list 11 estimator.
            ``two_stage``: dict ``{"detect": [...11], "polarity": [...10]}``.
    """

    def __init__(
        self, estimator=None, strategy: str = "multilabel", n_jobs: int | None = None
    ) -> None:
        self.estimator = estimator
        self.strategy = strategy
        self.n_jobs = n_jobs

    def _check(self, Y: np.ndarray) -> np.ndarray:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy phải thuộc {STRATEGIES}")
        if self.estimator is None:
            raise ValueError("Cần truyền estimator")
        Y = np.asarray(Y)
        if Y.ndim != 2 or Y.shape[1] != len(ALL_ASPECTS):
            raise ValueError(f"Y phải có dạng (n, {len(ALL_ASPECTS)}), nhận {Y.shape}")
        return Y

    def fit(self, X, Y) -> ABSAClassifier:
        """Huấn luyện các bộ phân loại con.

        Args:
            X: Ma trận đặc trưng (thưa hoặc đặc) kích thước ``(n, d)``.
            Y: Ma trận nhãn ``(n, 11)`` – xem :mod:`absa.data.labels`.
        """
        Y = self._check(Y)
        # max_nbytes=None: không memory-map dữ liệu sang tiến trình con. Memmap của joblib là
        # chỉ-đọc, khiến libsvm (SVC) báo "WRITEBACKIFCOPY base is read-only" với ma trận thưa.
        parallel = Parallel(n_jobs=self.n_jobs, prefer="processes", max_nbytes=None)
        if self.strategy == "multilabel":
            pairs = to_pair_matrix(Y)  # (n, 31) nhị phân
            self.estimators_ = parallel(
                delayed(_fit_one)(clone(self.estimator), X, pairs[:, k])
                for k in range(pairs.shape[1])
            )
            return self
        if self.strategy == "joint":
            self.estimators_ = parallel(
                delayed(_fit_one)(clone(self.estimator), X, Y[:, j]) for j in range(Y.shape[1])
            )
            return self

        detect_jobs = [
            delayed(_fit_one)(clone(self.estimator), X, (Y[:, j] != NONE_ID).astype(int))
            for j in range(Y.shape[1])
        ]
        polarity_jobs = []
        for j in range(Y.shape[1]):
            if j == _OTHERS_COL:
                continue
            mask = Y[:, j] != NONE_ID
            if mask.sum() == 0:  # Khía cạnh chưa từng xuất hiện: đoán Positive cho có.
                polarity_jobs.append(
                    delayed(_fit_one)(clone(self.estimator), X[:1], np.array([3]))
                )
            else:
                polarity_jobs.append(delayed(_fit_one)(clone(self.estimator), X[mask], Y[mask, j]))
        fitted = parallel(detect_jobs + polarity_jobs)
        self.estimators_ = {"detect": fitted[: Y.shape[1]], "polarity": fitted[Y.shape[1] :]}
        return self

    def _predict_pair_scores(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Dự đoán thô của ``multilabel``: ``(bits, scores)`` cùng kích thước (n, 31)."""
        bits = np.column_stack(
            [np.asarray(est.predict(X)).astype(int) for est in self.estimators_]
        )
        scores = np.column_stack(
            [_binary_scores(est, X, bits[:, k]) for k, est in enumerate(self.estimators_)]
        )
        return bits.astype(np.int8), scores

    def predict_pairs(self, X) -> np.ndarray:
        """Dự đoán **31 nhãn nhị phân thô** (n, 31), chưa xử lý xung đột. Chỉ cho ``multilabel``.

        Một khía cạnh có thể được bật nhiều cảm xúc cùng lúc; dùng
        :func:`absa.data.labels.count_pair_conflicts` để đo mức xung đột này.
        """
        if self.strategy != "multilabel":
            raise ValueError("predict_pairs chỉ dùng cho strategy='multilabel'")
        return self._predict_pair_scores(X)[0]

    def predict(self, X) -> np.ndarray:
        """Dự đoán ma trận nhãn ``(n, 11)``."""
        n = X.shape[0]
        if self.strategy == "multilabel":
            bits, scores = self._predict_pair_scores(X)
            return pairs_to_labels(bits, scores)
        if self.strategy == "joint":
            return np.column_stack([est.predict(X) for est in self.estimators_]).astype(np.int8)

        Y_pred = np.zeros((n, len(ALL_ASPECTS)), dtype=np.int8)
        polarity_models = iter(self.estimators_["polarity"])
        for j, detector in enumerate(self.estimators_["detect"]):
            present = np.asarray(detector.predict(X)).astype(bool)
            if j == _OTHERS_COL:
                Y_pred[:, j] = present.astype(np.int8)
                continue
            polarity_model = next(polarity_models)
            if present.any():
                Y_pred[present, j] = polarity_model.predict(X[present])
        return Y_pred

    def score(self, X, Y) -> float:
        """Macro-F1 trên cặp (khía cạnh, cảm xúc) – metric chính của project."""
        from absa.evaluation.metrics import evaluate_absa

        return evaluate_absa(Y, self.predict(X))["pair_f1_macro"]
