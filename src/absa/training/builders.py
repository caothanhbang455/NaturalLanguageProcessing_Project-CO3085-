"""Ghép biểu diễn + mô hình thành pipeline scikit-learn hoàn chỉnh."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from absa.features.registry import build_feature, get_feature_spec
from absa.models.ml.absa_classifier import ABSAClassifier
from absa.models.ml.registry import build_estimator, check_compatibility, get_model_spec


def build_ml_pipeline(
    feature_name: str,
    model_name: str,
    strategy: str = "multilabel",
    params: dict | None = None,
    n_jobs: int | None = None,
) -> Pipeline:
    """Tạo pipeline ``văn bản đã tiền xử lý -> đặc trưng -> ABSAClassifier`` (chưa fit).

    Vì vectorizer nằm *trong* pipeline nên mỗi lần ``fit`` (kể cả trong từng fold của
    cross-validation) đều chỉ học từ vựng/IDF/embedding trên dữ liệu huấn luyện của
    lần đó – không rò rỉ thông tin.

    Args:
        feature_name: Tên trong ``FEATURE_REGISTRY``.
        model_name: Tên trong ``MODEL_REGISTRY``.
        strategy: ``"multilabel"``, ``"joint"`` hoặc ``"two_stage"``.
        params: Siêu tham số của mô hình (không tiền tố).
        n_jobs: Số tiến trình fit song song giữa các khía cạnh.

    Raises:
        ValueError: Nếu cặp mô hình/biểu diễn không tương thích.
    """
    feature_spec = get_feature_spec(feature_name)
    ok, reason = check_compatibility(get_model_spec(model_name), feature_spec)
    if not ok:
        raise ValueError(reason)
    estimator = build_estimator(model_name, feature_spec, params)
    return Pipeline(
        [
            ("features", build_feature(feature_name)),
            ("clf", ABSAClassifier(estimator, strategy=strategy, n_jobs=n_jobs)),
        ]
    )
