"""Chạy lưới thí nghiệm M mô hình x N biểu diễn (x chiến lược) và lưu kết quả.

Luồng của :class:`ExperimentRunner`::

    với mỗi biểu diễn f:
        X_train = f.fit_transform(train)      # chỉ học trên train
        X_dev   = f.transform(dev)
        với mỗi mô hình m (tương thích với f), mỗi chiến lược s:
            ABSAClassifier(m, s).fit(X_train, Y_train) -> dự đoán dev -> metric
            ghi ngay 1 dòng vào CSV (chạy lại sẽ bỏ qua cấu hình đã xong)

Đặc trưng được tính **một lần cho mỗi biểu diễn** và dùng chung cho mọi mô hình, nên
lưới lớn vẫn chạy được trên CPU. Dev đóng vai trò tập validation để so sánh.
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from absa.config import PRIMARY_METRIC
from absa.evaluation.metrics import evaluate_absa
from absa.features.registry import build_feature, get_feature_spec
from absa.models.ml.absa_classifier import ABSAClassifier
from absa.models.ml.registry import build_estimator, check_compatibility, get_model_spec
from absa.preprocessing.lexicons import load_lexicons
from absa.preprocessing.pipeline import PreparedCorpus, resolve_config
from absa.utils.io import ensure_dir, hash_object
from absa.utils.logging import get_logger
from absa.utils.timing import Timer

logger = get_logger(__name__)


@dataclass
class FittedFeature:
    """Đặc trưng đã tính cho tập huấn luyện và tập đánh giá."""

    name: str
    transformer: object
    X_train: object
    X_eval: object
    fit_seconds: float

    @property
    def n_features(self) -> int:
        """Số chiều đặc trưng."""
        return int(self.X_train.shape[1])


def params_key(params: Mapping | None) -> str:
    """Chuỗi JSON ổn định của bộ siêu tham số (dùng làm khóa)."""
    return json.dumps(params or {}, sort_keys=True, default=str)


class ExperimentRunner:
    """Điều phối thí nghiệm ML trên một cặp (tập huấn luyện, tập đánh giá).

    Args:
        corpus: :class:`PreparedCorpus` chứa văn bản thô các tập.
        labels: ``{"train": Y_train, "dev": Y_dev, ...}``.
        train_split: Tên tập dùng để fit.
        eval_split: Tên tập dùng để đánh giá/chọn mô hình (mặc định ``"dev"``).
        results_path: File CSV lưu kết quả (ghi nối tiếp, hỗ trợ chạy tiếp khi bị ngắt).
        n_jobs: Số tiến trình fit song song giữa các khía cạnh.
        keep_predictions: Giữ dự đoán trên tập đánh giá trong bộ nhớ (cho phân tích sau).
    """

    def __init__(
        self,
        corpus: PreparedCorpus,
        labels: Mapping[str, np.ndarray],
        train_split: str = "train",
        eval_split: str = "dev",
        results_path: str | Path | None = None,
        n_jobs: int | None = -1,
        keep_predictions: bool = True,
    ) -> None:
        self.corpus = corpus
        self.labels = labels
        self.train_split = train_split
        self.eval_split = eval_split
        self.results_path = Path(results_path) if results_path else None
        self.n_jobs = n_jobs
        self.keep_predictions = keep_predictions
        self.predictions: dict[str, np.ndarray] = {}
        self._features: dict[str, FittedFeature] = {}
        self._rows: list[dict] = []
        if self.results_path and self.results_path.exists():
            rows = pd.read_csv(self.results_path).to_dict("records")
            # Chỉ giữ kết quả có cùng phiên bản tiền xử lý/từ điển với code hiện tại.
            self._rows = [r for r in rows if self._is_current(r)]
            logger.info(
                "Nạp %d kết quả từ %s (bỏ %d kết quả của phiên bản tiền xử lý cũ)",
                len(self._rows),
                self.results_path.name,
                len(rows) - len(self._rows),
            )

    # ------------------------------------------------------------ features --
    def features(self, name: str) -> FittedFeature:
        """Fit biểu diễn ``name`` trên tập huấn luyện (có cache trong bộ nhớ)."""
        if name not in self._features:
            spec = get_feature_spec(name)
            texts_train = self.corpus.get(spec.profile, self.train_split)
            texts_eval = self.corpus.get(spec.profile, self.eval_split)
            transformer = build_feature(name)
            with Timer() as t:
                X_train = transformer.fit_transform(texts_train)
                X_eval = transformer.transform(texts_eval)
            self._features[name] = FittedFeature(name, transformer, X_train, X_eval, t.seconds)
            logger.info("Biểu diễn %-22s %6d chiều  (%.1fs)", name, X_train.shape[1], t.seconds)
        return self._features[name]

    def release_features(self, keep: Iterable[str] = ()) -> None:
        """Giải phóng bộ nhớ đặc trưng (trừ các biểu diễn trong ``keep``)."""
        keep = set(keep)
        self._features = {k: v for k, v in self._features.items() if k in keep}

    def fit_predict_pairs(
        self, feature: str, model: str, params: Mapping | None = None
    ) -> np.ndarray:
        """Huấn luyện ``multilabel`` rồi trả về **31 nhãn nhị phân thô** của tập đánh giá.

        Dùng để xem dự đoán trước khi giải quyết xung đột (một khía cạnh bị bật nhiều cảm xúc).

        Returns:
            Ma trận nhị phân (n_eval, 31) theo thứ tự :func:`absa.data.labels.pair_label_names`.
        """
        spec = get_feature_spec(feature)
        fitted = self.features(feature)
        classifier = ABSAClassifier(
            build_estimator(model, spec, dict(params or {})),
            strategy="multilabel",
            n_jobs=self.n_jobs,
        )
        classifier.fit(fitted.X_train, self.labels[self.train_split])
        return classifier.predict_pairs(fitted.X_eval)

    # ------------------------------------------------------------------ run --
    @staticmethod
    def data_version(feature: str) -> str:
        """Phiên bản dữ liệu đầu vào của biểu diễn: băm cấu hình tiền xử lý + từ điển.

        Đổi luật tiền xử lý hoặc sửa từ điển sẽ đổi phiên bản, nên kết quả cũ trong CSV
        không bị dùng lại nhầm khi chạy tiếp.
        """
        config = resolve_config(get_feature_spec(feature).profile)
        return hash_object([config.fingerprint, load_lexicons().fingerprint])[:6]

    @classmethod
    def run_id(cls, feature: str, model: str, strategy: str, params: Mapping | None) -> str:
        """Định danh duy nhất của một cấu hình (gồm cả phiên bản dữ liệu đầu vào)."""
        params_hash = hash_object(params or {})[:8]
        return f"{feature}|{model}|{strategy}|{params_hash}|{cls.data_version(feature)}"

    def _is_current(self, row: Mapping) -> bool:
        try:
            return str(row["run_id"]).endswith("|" + self.data_version(row["feature"]))
        except KeyError:  # biểu diễn đã bị xóa khỏi registry
            return False

    def _done(self) -> set[str]:
        return {r["run_id"] for r in self._rows if r.get("status") == "ok"}

    def _record(self, row: dict) -> dict:
        self._rows.append(row)
        if self.results_path:
            # Ghi lại toàn bộ bảng (vài trăm dòng) để các cột luôn thẳng hàng dù dòng
            # "failed" và dòng "ok" có tập cột khác nhau.
            ensure_dir(self.results_path.parent)
            pd.DataFrame(self._rows).to_csv(self.results_path, index=False)
        return row

    def run(
        self,
        feature: str,
        model: str,
        strategy: str = "multilabel",
        params: Mapping | None = None,
        tag: str = "grid",
    ) -> dict:
        """Huấn luyện và đánh giá **một** cấu hình, trả về dòng kết quả.

        Lỗi (thiếu bộ nhớ, mô hình không hội tụ...) được ghi lại với ``status="failed"``
        thay vì làm dừng cả lưới.
        """
        feature_spec, model_spec = get_feature_spec(feature), get_model_spec(model)
        base = {
            "run_id": self.run_id(feature, model, strategy, params),
            "tag": tag,
            "feature": feature,
            "feature_family": feature_spec.family,
            "model": model,
            "model_family": model_spec.family,
            "strategy": strategy,
            "params": params_key(params),
        }
        ok, reason = check_compatibility(model_spec, feature_spec)
        if not ok:
            return {**base, "status": "skipped", "reason": reason}
        try:
            fitted = self.features(feature)
            clf = ABSAClassifier(
                build_estimator(model, feature_spec, dict(params or {})),
                strategy=strategy,
                n_jobs=self.n_jobs,
            )
            with Timer() as t_fit:
                clf.fit(fitted.X_train, self.labels[self.train_split])
            with Timer() as t_pred:
                Y_pred = clf.predict(fitted.X_eval)
        except Exception as exc:  # noqa: BLE001 - ghi lỗi để lưới chạy tiếp
            logger.warning("Lỗi %s: %s", base["run_id"], exc)
            return self._record(
                {
                    **base,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=3),
                }
            )
        metrics = evaluate_absa(self.labels[self.eval_split], Y_pred)
        if self.keep_predictions:
            self.predictions[base["run_id"]] = Y_pred
        row = {
            **base,
            "status": "ok",
            "reason": "",
            "n_features": fitted.n_features,
            "feature_fit_sec": round(fitted.fit_seconds, 2),
            "fit_sec": round(t_fit.seconds, 2),
            "predict_ms_per_sample": round(1000 * t_pred.seconds / max(len(Y_pred), 1), 4),
            **{k: round(v, 5) for k, v in metrics.items()},
        }
        logger.info(
            "%-22s %-22s %-9s pair_f1_macro=%.4f aspect_f1_micro=%.4f (%.1fs)",
            feature,
            model,
            strategy,
            metrics["pair_f1_macro"],
            metrics["aspect_f1_micro"],
            t_fit.seconds,
        )
        return self._record(row)

    def run_grid(
        self,
        features: Sequence[str],
        models: Sequence[str],
        strategies: Sequence[str] = ("multilabel",),
        resume: bool = True,
        release_features: bool = True,
    ) -> pd.DataFrame:
        """Chạy toàn bộ lưới ``features x models x strategies``.

        Args:
            features: Danh sách tên biểu diễn.
            models: Danh sách tên mô hình.
            strategies: Danh sách chiến lược ABSA.
            resume: Bỏ qua cấu hình đã có kết quả ``ok`` trong ``results_path``.
            release_features: Giải phóng đặc trưng sau khi chạy xong một biểu diễn
                (tiết kiệm RAM khi lưới lớn).

        Returns:
            Bảng kết quả của các cấu hình trong lưới này.
        """
        done = self._done() if resume else set()
        ids = []
        for feature in features:
            for model in models:
                for strategy in strategies:
                    run_id = self.run_id(feature, model, strategy, None)
                    ids.append(run_id)
                    if run_id in done:
                        continue
                    row = self.run(feature, model, strategy)
                    if row["status"] == "skipped":
                        logger.info("Bỏ qua %s: %s", run_id, row["reason"])
            if release_features:
                self.release_features()
        frame = self.results
        return (
            frame[frame["run_id"].isin(ids)].reset_index(drop=True) if not frame.empty else frame
        )

    # -------------------------------------------------------------- results --
    @property
    def results(self) -> pd.DataFrame:
        """Mọi kết quả đã chạy (kể cả nạp từ CSV)."""
        return pd.DataFrame(self._rows)

    def leaderboard(
        self, metric: str = PRIMARY_METRIC, top: int | None = 20, tag: str | None = None
    ) -> pd.DataFrame:
        """Bảng xếp hạng các cấu hình chạy thành công theo ``metric``."""
        frame = self.results
        if frame.empty:
            return frame
        frame = frame[frame["status"] == "ok"]
        if tag:
            frame = frame[frame["tag"] == tag]
        frame = frame.sort_values(metric, ascending=False).drop_duplicates("run_id")
        columns = [
            "feature",
            "model",
            "strategy",
            "params",
            metric,
            "pair_f1_micro",
            "aspect_f1_macro",
            "aspect_f1_micro",
            "polarity_acc_on_detected",
            "fit_sec",
            "n_features",
        ]
        columns = list(dict.fromkeys(c for c in columns if c in frame))
        return frame[columns].head(top).reset_index(drop=True)
