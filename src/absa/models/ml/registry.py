"""Danh mục M mô hình học máy truyền thống (model registry).

Mỗi mô hình được mô tả bởi :class:`ModelSpec`: cách tạo, loại đặc trưng chấp nhận,
có cần chuẩn hóa đặc trưng đặc hay không, lưới siêu tham số để tinh chỉnh và mức
chi phí tính toán. :func:`build_estimator` tự ghép bộ chuẩn hóa phù hợp, còn
:func:`check_compatibility` loại sớm các cặp (mô hình, biểu diễn) không hợp lệ – ví
dụ Multinomial NB không nhận đặc trưng âm của embedding.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from absa.config import SEED
from absa.features.registry import FeatureSpec

COSTS = ("fast", "medium", "slow")

#: Ba mô hình ML cốt lõi của project (không cây/ensemble, không boosting).
CORE_MODELS: tuple[str, ...] = ("logistic_regression", "linear_svm", "knn")


@dataclass(frozen=True)
class ModelSpec:
    """Mô tả một mô hình ML.

    Attributes:
        name: Khóa duy nhất.
        family: Nhóm mô hình (``linear``, ``naive_bayes``, ``instance``, ``kernel``,
            ``tree_ensemble``, ``neural``, ``baseline``).
        factory: Hàm tạo estimator chưa fit với tham số mặc định hợp lý cho văn bản.
        accepts: Loại đặc trưng nhận được: ``{"sparse", "dense"}``.
        requires_non_negative: Chỉ nhận đặc trưng không âm (NB đa thức).
        scale_dense: Chuẩn hóa ``StandardScaler`` khi đặc trưng là ma trận đặc.
        param_grid: Lưới siêu tham số (tên tham số **không** có tiền tố).
        cost: ``fast | medium | slow`` – dùng để chọn tập mô hình theo ngân sách.
        summary: Mô tả ngắn (tiếng Việt).
        requires: Gói Python bắt buộc (mô hình tùy chọn như LightGBM).
    """

    name: str
    family: str
    factory: Callable[[], ClassifierMixin]
    accepts: frozenset[str]
    requires_non_negative: bool = False
    scale_dense: bool = False
    param_grid: dict[str, list] = field(default_factory=dict)
    cost: str = "fast"
    summary: str = ""
    requires: tuple[str, ...] = ()

    def is_available(self) -> bool:
        """Đã cài đủ gói cần thiết chưa."""
        return all(importlib.util.find_spec(p) is not None for p in self.requires)


BOTH = frozenset({"sparse", "dense"})
DENSE = frozenset({"dense"})
SPARSE = frozenset({"sparse"})


def _lightgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        random_state=SEED,
        n_jobs=1,
        verbose=-1,
    )


def _xgboost():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.5,
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
    )


_SPECS: list[ModelSpec] = [
    ModelSpec(
        "majority_baseline",
        "baseline",
        lambda: DummyClassifier(strategy="most_frequent"),
        BOTH,
        summary="Luôn đoán lớp phổ biến nhất của từng khía cạnh – mốc dưới để so sánh.",
    ),
    ModelSpec(
        "multinomial_nb",
        "naive_bayes",
        lambda: MultinomialNB(alpha=0.3),
        SPARSE,
        requires_non_negative=True,
        param_grid={"alpha": [0.03, 0.1, 0.3, 1.0]},
        summary="Naive Bayes đa thức: xác suất từ theo lớp, rất nhanh, mạnh với văn bản ngắn.",
    ),
    ModelSpec(
        "complement_nb",
        "naive_bayes",
        lambda: ComplementNB(alpha=0.3),
        SPARSE,
        requires_non_negative=True,
        param_grid={"alpha": [0.03, 0.1, 0.3, 1.0]},
        summary="Complement NB (Rennie et al., 2003): ước lượng từ lớp bù, hợp dữ liệu lệch.",
    ),
    ModelSpec(
        "gaussian_nb",
        "naive_bayes",
        GaussianNB,
        DENSE,
        param_grid={"var_smoothing": [1e-9, 1e-7, 1e-5]},
        summary="Gaussian NB cho embedding đặc: giả định đặc trưng độc lập, phân phối chuẩn.",
    ),
    ModelSpec(
        "logistic_regression",
        "linear",
        lambda: LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"),
        BOTH,
        scale_dense=True,
        param_grid={"C": [0.25, 1.0, 4.0, 16.0], "class_weight": [None, "balanced"]},
        summary="Hồi quy logistic (softmax) có trọng số lớp cân bằng – baseline tuyến tính chuẩn.",
    ),
    ModelSpec(
        "linear_svm",
        "linear",
        lambda: LinearSVC(C=0.5, class_weight="balanced", max_iter=5000),
        BOTH,
        scale_dense=True,
        param_grid={"C": [0.05, 0.1, 0.25, 0.5, 1.0], "class_weight": [None, "balanced"]},
        summary="SVM tuyến tính (hinge loss, one-vs-rest): thường mạnh nhất với TF-IDF.",
    ),
    ModelSpec(
        "ridge",
        "linear",
        lambda: RidgeClassifier(alpha=1.0, class_weight="balanced"),
        BOTH,
        scale_dense=True,
        param_grid={"alpha": [0.3, 1.0, 3.0, 10.0]},
        summary="Ridge classifier: hồi quy bình phương tối thiểu có L2, nhanh và ổn định.",
    ),
    ModelSpec(
        "sgd_modified_huber",
        "linear",
        lambda: SGDClassifier(
            loss="modified_huber",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=50,
            tol=1e-3,
            random_state=SEED,
        ),
        BOTH,
        scale_dense=True,
        param_grid={"alpha": [1e-5, 3e-5, 1e-4, 3e-4]},
        summary="Mô hình tuyến tính học bằng SGD với modified-Huber (bền nhiễu, có xác suất).",
    ),
    ModelSpec(
        "knn",
        "instance",
        lambda: KNeighborsClassifier(n_neighbors=15, weights="distance", metric="cosine"),
        BOTH,
        param_grid={"n_neighbors": [5, 15, 30], "weights": ["uniform", "distance"]},
        cost="medium",
        summary="k láng giềng gần nhất theo cosine – không học tham số.",
    ),
    ModelSpec(
        "svm_rbf",
        "kernel",
        lambda: SVC(C=4.0, kernel="rbf", gamma="scale", class_weight="balanced"),
        BOTH,
        scale_dense=True,
        param_grid={"C": [1.0, 4.0, 16.0]},
        cost="medium",
        summary="SVM kernel RBF: biên phi tuyến, hợp với embedding đặc số chiều vừa phải.",
    ),
    ModelSpec(
        "random_forest",
        "tree_ensemble",
        lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=1, random_state=SEED
        ),
        BOTH,
        param_grid={"n_estimators": [300, 600], "min_samples_leaf": [1, 2]},
        cost="slow",
        summary="Rừng ngẫu nhiên (bagging cây quyết định): bắt tương tác phi tuyến.",
    ),
    ModelSpec(
        "extra_trees",
        "tree_ensemble",
        lambda: ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=1, random_state=SEED
        ),
        BOTH,
        param_grid={"n_estimators": [300, 600], "min_samples_leaf": [1, 2]},
        cost="slow",
        summary="Extra-Trees: như rừng ngẫu nhiên nhưng ngưỡng tách ngẫu nhiên, nhanh hơn.",
    ),
    ModelSpec(
        "hist_gradient_boosting",
        "tree_ensemble",
        lambda: HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.1,
            class_weight="balanced",
            early_stopping=True,
            random_state=SEED,
        ),
        DENSE,
        param_grid={"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]},
        cost="slow",
        summary="Gradient boosting dựa trên histogram (kiểu LightGBM) của scikit-learn.",
    ),
    ModelSpec(
        "lightgbm",
        "tree_ensemble",
        _lightgbm,
        BOTH,
        param_grid={"num_leaves": [15, 31, 63], "learning_rate": [0.05, 0.1]},
        cost="slow",
        summary="LightGBM (Ke et al., 2017): boosting cây theo lá, nhận được ma trận thưa.",
        requires=("lightgbm",),
    ),
    ModelSpec(
        "xgboost",
        "tree_ensemble",
        _xgboost,
        BOTH,
        param_grid={"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
        cost="slow",
        summary="XGBoost (Chen & Guestrin, 2016): boosting cây có điều chuẩn.",
        requires=("xgboost",),
    ),
    ModelSpec(
        "mlp",
        "neural",
        lambda: MLPClassifier(
            hidden_layer_sizes=(256,),
            alpha=1e-4,
            early_stopping=True,
            max_iter=200,
            random_state=SEED,
        ),
        # Chỉ nhận embedding đặc: đo thực tế trên TF-IDF 27k chiều mất ~21 phút/cấu hình (CPU)
        # mà macro-F1 chỉ 0.40 (LR: 0.53) – lớp đầu vào quá lớn so với 7.800 mẫu.
        DENSE,
        scale_dense=True,
        param_grid={"hidden_layer_sizes": [(256,), (512,), (256, 128)], "alpha": [1e-4, 1e-3]},
        cost="slow",
        summary="Mạng nơ-ron truyền thẳng (MLP) 1–2 lớp ẩn, early stopping.",
    ),
]

MODEL_REGISTRY: dict[str, ModelSpec] = {spec.name: spec for spec in _SPECS}


def get_model_spec(name: str) -> ModelSpec:
    """Lấy :class:`ModelSpec` theo tên."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Mô hình '{name}' không tồn tại. Có: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def check_compatibility(model: ModelSpec, feature: FeatureSpec) -> tuple[bool, str]:
    """Kiểm tra cặp (mô hình, biểu diễn) có hợp lệ không, kèm lý do nếu không."""
    if not model.is_available():
        return False, f"chưa cài {model.requires}"
    if feature.output not in model.accepts:
        return False, f"{model.name} không nhận đặc trưng {feature.output}"
    if model.requires_non_negative and not feature.non_negative:
        return False, f"{model.name} cần đặc trưng không âm"
    return True, ""


def build_estimator(
    model_name: str, feature: FeatureSpec | None = None, params: dict | None = None
) -> Pipeline:
    """Tạo estimator đơn nhãn đã ghép bộ chuẩn hóa (nếu cần).

    Luôn trả về ``Pipeline([("scaler", ...), ("model", ...)])`` để tên siêu tham số thống
    nhất: ``model__C``, ``model__alpha``...

    Args:
        model_name: Tên trong ``MODEL_REGISTRY``.
        feature: Đặc tả biểu diễn đầu vào (quyết định có chuẩn hóa hay không).
        params: Siêu tham số ghi đè cho mô hình (không có tiền tố ``model__``).
    """
    spec = get_model_spec(model_name)
    estimator = spec.factory()
    if params:
        estimator.set_params(**params)
    needs_scaler = spec.scale_dense and feature is not None and feature.output == "dense"
    scaler = StandardScaler() if needs_scaler else "passthrough"
    return Pipeline([("scaler", scaler), ("model", estimator)])


#: Họ mô hình "cơ bản": tuyến tính, Naive Bayes, láng giềng, kernel. Không gồm cây/ensemble
#: (Random Forest, Extra-Trees, boosting) và mạng nơ-ron.
BASIC_FAMILIES: tuple[str, ...] = ("baseline", "naive_bayes", "linear", "instance", "kernel")


def list_models(
    costs: Iterable[str] = COSTS,
    include_baseline: bool = True,
    only_available: bool = True,
    families: Iterable[str] | None = None,
) -> list[str]:
    """Liệt kê tên mô hình theo mức chi phí và (tùy chọn) theo họ.

    Args:
        costs: Các mức chi phí cần lấy (``fast``, ``medium``, ``slow``).
        include_baseline: Có lấy ``majority_baseline`` không.
        only_available: Bỏ mô hình thiếu gói (LightGBM, XGBoost...).
        families: Chỉ lấy các họ này; ``None`` = mọi họ. Dùng :data:`BASIC_FAMILIES` để chỉ lấy
            mô hình cơ bản.
    """
    costs = set(costs)
    families = None if families is None else set(families)
    return [
        s.name
        for s in MODEL_REGISTRY.values()
        if s.cost in costs
        and (include_baseline or s.family != "baseline")
        and (families is None or s.family in families)
        and (not only_available or s.is_available())
    ]


def model_catalog() -> pd.DataFrame:
    """Bảng mô tả toàn bộ mô hình ML."""
    return pd.DataFrame(
        [
            {
                "name": s.name,
                "family": s.family,
                "accepts": "/".join(sorted(s.accepts)),
                "non_negative_only": s.requires_non_negative,
                "cost": s.cost,
                "available": s.is_available(),
                "tuned_params": ", ".join(s.param_grid) or "-",
                "idea": s.summary,
            }
            for s in MODEL_REGISTRY.values()
        ]
    ).set_index("name")
