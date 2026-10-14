"""Ablation tiền xử lý: đo đóng góp của từng bước bằng số liệu, không bằng cảm tính.

Hai kiểu thí nghiệm:

* :func:`cumulative_variants` – bắt đầu từ cấu hình tối thiểu (chỉ lowercase + bỏ dấu
  câu), lần lượt **thêm** từng bước. Cho thấy đường cong cải thiện.
* :func:`leave_one_out_variants` – từ cấu hình đầy đủ ``classic``, lần lượt **đảo** từng
  bước. Cho thấy bước nào thực sự cần thiết khi đã có các bước còn lại.

Kết quả là phần "motivation" định lượng cho mỗi bước tiền xử lý trong báo cáo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from absa.evaluation.metrics import evaluate_absa
from absa.preprocessing.pipeline import (
    MINIMAL_CONFIG,
    PREPROCESS_PROFILES,
    PreprocessConfig,
    preprocess_corpus,
)
from absa.training.builders import build_ml_pipeline
from absa.utils.logging import get_logger
from absa.utils.timing import Timer

logger = get_logger(__name__)


def cumulative_variants() -> dict[str, PreprocessConfig]:
    """Thêm dần từng bước từ cấu hình tối thiểu tới ``classic``, rồi thử hai bước tùy chọn.

    Hai dòng cuối (``+ word segmentation``, ``+ stopword removal``) là các bước *không* có
    trong ``classic`` – chúng được giữ trong bảng để báo cáo vì sao bị loại.
    """
    steps: list[tuple[str, dict]] = [
        ("0. minimal (lower + punct)", {}),
        ("1. + unicode/tone", {"normalize_unicode": True, "remove_urls": True}),
        ("2. + emoji/emoticon tags", {"emoji_mode": "tag", "emoticon_mode": "tag"}),
        ("3. + number/unit tokens", {"normalize_numbers": True}),
        ("4. + elongation", {"reduce_elongation": True}),
        ("5. + teencode & final-ng", {"expand_teencode": True, "restore_final_ng": True}),
        ("6. + negation (= classic)", {"negation_window": 1}),
    ]
    variants, current = {}, MINIMAL_CONFIG
    for name, change in steps:
        current = current.replace(**change)
        variants[name] = current
    variants["7. classic + word segmentation"] = current.replace(word_segmentation="underthesea")
    variants["8. classic + stopword removal"] = current.replace(remove_stopwords=True)
    return variants


def leave_one_out_variants() -> dict[str, PreprocessConfig]:
    """``classic`` và các bản *đảo* đúng một bước (bỏ bước đang bật / thêm bước đang tắt)."""
    full = PREPROCESS_PROFILES["classic"]
    toggles = {
        "- unicode/tone": {"normalize_unicode": False},
        "- emoji/emoticon tags": {"emoji_mode": "remove", "emoticon_mode": "remove"},
        "- number/unit tokens": {"normalize_numbers": False},
        "- elongation": {"reduce_elongation": False},
        "- teencode & final-ng": {"expand_teencode": False, "restore_final_ng": False},
        "- negation": {"negation_window": 0},
        "+ word segmentation": {"word_segmentation": "underthesea"},
        "+ stopword removal": {"remove_stopwords": True},
    }
    variants = {"classic (full)": full}
    variants.update({name: full.replace(**change) for name, change in toggles.items()})
    return variants


def run_preprocessing_ablation(
    train_texts: Sequence[str],
    Y_train: np.ndarray,
    eval_texts: Sequence[str],
    Y_eval: np.ndarray,
    variants: Mapping[str, PreprocessConfig],
    feature: str = "tfidf_word",
    model: str = "logistic_regression",
    strategy: str = "multilabel",
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Huấn luyện cùng một pipeline với từng cấu hình tiền xử lý và so sánh trên dev.

    Biểu diễn/mô hình được cố định (mặc định TF-IDF + Logistic Regression – nhanh,
    nhạy với chất lượng token) để mọi khác biệt đến từ tiền xử lý.

    Returns:
        Mỗi cấu hình một dòng: metric chính, kích thước từ vựng, thời gian.
    """
    rows = []
    for name, config in variants.items():
        with Timer() as t:
            train_processed = preprocess_corpus(train_texts, config)
            eval_processed = preprocess_corpus(eval_texts, config)
            pipeline = build_ml_pipeline(feature, model, strategy, n_jobs=n_jobs)
            pipeline.fit(train_processed, Y_train)
            metrics = evaluate_absa(Y_eval, pipeline.predict(eval_processed))
        vocab = len({tok for text in train_processed for tok in text.split()})
        rows.append(
            {"variant": name, "vocab_size": vocab, "seconds": round(t.seconds, 1), **metrics}
        )
        logger.info("%-30s pair_f1_macro=%.4f vocab=%d", name, metrics["pair_f1_macro"], vocab)
    return pd.DataFrame(rows).set_index("variant")
