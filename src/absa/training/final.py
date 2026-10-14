"""Huấn luyện mô hình cuối cùng và đánh giá một lần trên tập test ẩn.

Quy trình chuẩn sau khi đã chọn cấu hình trên dev:

1. Huấn luyện lại cấu hình đã chọn trên **train + dev** (thêm ~14% dữ liệu; dev đã
   hoàn thành vai trò chọn mô hình nên được phép dùng để học).
2. Đánh giá **một lần duy nhất** trên test (nếu có file) – không quay lại chỉnh sửa
   sau khi đã thấy điểm test.
3. Đóng gói thành :class:`ABSAPipeline` để triển khai.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime

import numpy as np

from absa.features.registry import get_feature_spec
from absa.inference.predictor import ABSAPipeline
from absa.preprocessing.pipeline import PREPROCESS_PROFILES, PreparedCorpus, TextPreprocessor
from absa.training.builders import build_ml_pipeline
from absa.utils.logging import get_logger
from absa.utils.timing import Timer

logger = get_logger(__name__)


def fit_final_pipeline(
    corpus: PreparedCorpus,
    labels: Mapping[str, np.ndarray],
    feature: str,
    model: str,
    strategy: str = "multilabel",
    params: dict | str | None = None,
    fit_splits: Sequence[str] = ("train", "dev"),
    n_jobs: int | None = -1,
    extra_metadata: dict | None = None,
) -> ABSAPipeline:
    """Huấn luyện cấu hình đã chọn trên ``fit_splits`` và đóng gói để triển khai.

    Args:
        corpus: Kho văn bản các tập.
        labels: ``{"train": Y, "dev": Y, ...}``.
        feature: Biểu diễn đã chọn.
        model: Mô hình đã chọn.
        strategy: Chiến lược ABSA đã chọn.
        params: Siêu tham số (dict hoặc chuỗi JSON như trong bảng kết quả).
        fit_splits: Các tập dùng để huấn luyện.
        n_jobs: Số tiến trình song song.
        extra_metadata: Thông tin thêm (vd metric dev) để lưu kèm mô hình.
    """
    if isinstance(params, str):
        params = json.loads(params) if params else {}
    profile = get_feature_spec(feature).profile
    texts = corpus.concat(profile, fit_splits)
    Y = np.vstack([labels[s] for s in fit_splits])
    pipeline = build_ml_pipeline(feature, model, strategy, params, n_jobs=n_jobs)
    with Timer() as t:
        pipeline.fit(texts, Y)
    logger.info(
        "Huấn luyện mô hình cuối trên %s (%d mẫu) trong %.1fs", fit_splits, len(Y), t.seconds
    )
    metadata = {
        "feature": feature,
        "model": model,
        "strategy": strategy,
        "params": params or {},
        "preprocess_profile": profile,
        "preprocess_config": PREPROCESS_PROFILES[profile].to_dict(),
        "fit_splits": list(fit_splits),
        "n_train": len(Y),
        "fit_seconds": round(t.seconds, 1),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **(extra_metadata or {}),
    }
    return ABSAPipeline(TextPreprocessor(profile), pipeline, metadata)
