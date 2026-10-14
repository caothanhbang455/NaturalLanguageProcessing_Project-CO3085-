"""Pipeline suy luận hoàn chỉnh: bình luận thô -> nhãn khía cạnh -> tổng hợp theo sản phẩm.

:class:`ABSAPipeline` đóng gói **mọi thứ** cần cho triển khai (bộ tiền xử lý, biểu diễn
đã fit, bộ phân loại, metadata huấn luyện) vào một file duy nhất. Ứng dụng demo
(``app/streamlit_app.py``) chỉ cần::

    pipeline = ABSAPipeline.load("outputs/models/best_ml_pipeline.joblib")
    pipeline.predict_frame(["Pin trâu, camera hơi mờ"])
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from absa.config import ALL_ASPECTS, ID_TO_POLARITY, OTHERS_ASPECT, SENTIMENT_ASPECTS
from absa.data.labels import decode_labels, format_label_string, pair_label_names
from absa.preprocessing.pipeline import TextPreprocessor
from absa.utils.io import load_joblib, save_joblib


class ABSAPipeline:
    """Bộ dự đoán ABSA end-to-end.

    Args:
        preprocessor: Bộ tiền xử lý đúng profile mà mô hình được huấn luyện.
        model: Pipeline scikit-learn ``features -> ABSAClassifier`` đã fit (hoặc bất kỳ
            đối tượng nào có ``predict(list[str]) -> ndarray (n, 11)``).
        metadata: Thông tin huấn luyện (cấu hình, metric dev/test, ngày tạo...).
    """

    def __init__(
        self, preprocessor: TextPreprocessor, model, metadata: dict | None = None
    ) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.metadata = metadata or {}

    # -------------------------------------------------------------- predict --
    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Dự đoán ma trận nhãn ``(n, 11)`` từ bình luận thô."""
        processed = self.preprocessor.transform(texts)
        return np.asarray(self.model.predict(processed))

    def predict_labels(self, texts: Sequence[str]) -> list[dict[str, str | None]]:
        """Dự đoán dạng ``[{"BATTERY": "Positive", "CAMERA": "Negative"}, ...]``."""
        return decode_labels(self.predict(texts))

    def predict_frame(self, texts: Sequence[str]) -> pd.DataFrame:
        """Bảng kết quả: mỗi bình luận một dòng, mỗi khía cạnh một cột."""
        Y = self.predict(texts)
        frame = pd.DataFrame({"comment": list(texts)})
        for j, aspect in enumerate(ALL_ASPECTS):
            if aspect == OTHERS_ASPECT:
                frame[aspect] = np.where(Y[:, j] > 0, "Present", "")
            else:
                frame[aspect] = [ID_TO_POLARITY.get(int(v), "") for v in Y[:, j]]
        frame["label"] = [format_label_string(d) for d in decode_labels(Y)]
        return frame

    def predict_pairs(self, texts: Sequence[str]) -> pd.DataFrame:
        """Dự đoán **31 nhãn nhị phân thô** (``ASPECT#Polarity`` và ``OTHERS``) từ bình luận thô.

        Chỉ dùng được khi mô hình được huấn luyện với chiến lược ``multilabel``. Khác với
        :meth:`predict`, kết quả **chưa xử lý xung đột**: một khía cạnh có thể được bật nhiều
        cảm xúc cùng lúc (xem :func:`absa.data.labels.count_pair_conflicts`).

        Returns:
            DataFrame (n, 31): mỗi cột là một nhãn, giá trị 0/1.
        """
        classifier = self.model[-1] if hasattr(self.model, "steps") else self.model
        if not hasattr(classifier, "predict_pairs"):
            raise ValueError("Mô hình này không huấn luyện theo chiến lược 'multilabel'")
        processed = self.preprocessor.transform(texts)
        features = (
            self.model[:-1].transform(processed) if hasattr(self.model, "steps") else processed
        )
        return pd.DataFrame(classifier.predict_pairs(features), columns=pair_label_names())

    # ------------------------------------------------------------ aggregate --
    def summarize(self, texts: Sequence[str]) -> pd.DataFrame:
        """Tổng hợp cảm xúc theo khía cạnh cho một tập bình luận về cùng sản phẩm.

        Returns:
            Mỗi khía cạnh một dòng: số lượt nhắc, số lượt tích cực/trung tính/tiêu cực,
            tỉ lệ % và ``net_score = (%pos - %neg)`` trong [-100, 100] – dùng để xếp
            hạng điểm mạnh/điểm yếu của sản phẩm.
        """
        return summarize_predictions(self.predict(texts))

    # ------------------------------------------------------------------- io --
    def save(self, path: str | Path) -> Path:
        """Lưu pipeline ra file joblib."""
        return save_joblib(self, path)

    @classmethod
    def load(cls, path: str | Path) -> ABSAPipeline:
        """Nạp pipeline đã lưu."""
        obj = load_joblib(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} không chứa ABSAPipeline")
        return obj


def summarize_predictions(Y: np.ndarray) -> pd.DataFrame:
    """Tổng hợp ma trận dự đoán thành bảng điểm mạnh/yếu theo khía cạnh."""
    Y = np.asarray(Y)
    rows = []
    for j, aspect in enumerate(SENTIMENT_ASPECTS):
        col = Y[:, j]
        mentions = int((col > 0).sum())
        counts = {ID_TO_POLARITY[k]: int((col == k).sum()) for k in (1, 2, 3)}
        pct = {
            f"pct_{k.lower()}": (100 * v / mentions if mentions else 0.0)
            for k, v in counts.items()
        }
        rows.append(
            {
                "aspect": aspect,
                "mentions": mentions,
                **counts,
                **{k: round(v, 1) for k, v in pct.items()},
                "net_score": round(pct["pct_positive"] - pct["pct_negative"], 1),
            }
        )
    frame = pd.DataFrame(rows).set_index("aspect")
    return frame.sort_values(["mentions", "net_score"], ascending=False)
